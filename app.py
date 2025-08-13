# app.py
"""
Music Cover Tool (Full, Clean) - Offline / Free
Python 3.13 (64-bit) compatible

- Bỏ Spleeter, dùng Demucs để tách giọng (two-stems=vocals)
- Biến đổi instrumental: pitch (semitones), tempo (time-stretch), reverb đơn giản, volume
- Trộn nền mưa (tùy chọn)
- Xuất: full bài hoặc loop N giờ
- Ghép video: ảnh tĩnh + audio đã xử lý bằng ffmpeg
- Log rõ ràng, lỗi gọn gàng
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa


# ----------------------------- Utilities -----------------------------

def which_or_raise(cmd_name: str):
    """Check command exists in PATH, else raise a clear error."""
    path = shutil.which(cmd_name)
    if path is None:
        raise EnvironmentError(
            f"Không tìm thấy '{cmd_name}' trong PATH. "
            f"Vui lòng cài đặt và thêm vào PATH. (cmd: {cmd_name})"
        )
    return path


def run_cmd(cmd: List[str], log: bool = True) -> str:
    """Run a command; raise RuntimeError on non-zero exit. Capture and print logs safely (UTF-8)."""
    if log:
        print("RUN:", " ".join(map(str, cmd)))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            shell=False,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Không chạy được lệnh: {cmd[0]}\nLý do: {e}") from e

    out = proc.stdout or ""
    err = proc.stderr or ""
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại (exit={proc.returncode}): {' '.join(map(str, cmd))}\n{err}")
    return out


def make_temp_wav(suffix="_tmp.wav") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    return f.name


# ----------------------------- Step 1: Separate vocals with Demucs -----------------------------

def _find_instrumental_in(folder: Path) -> Optional[Path]:
    """Tìm file instrumental do Demucs xuất (thường là 'no_vocals.wav' hoặc 'accompaniment.wav')."""
    candidates = [
        "no_vocals.wav",
        "accompaniment.wav",
        "instrumental.wav",
    ]
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    # Fallback: lấy file WAV dài nhất trong folder
    wavs = list(folder.glob("*.wav"))
    if wavs:
        return sorted(wavs, key=lambda x: x.stat().st_size, reverse=True)[0]
    return None


def separate_vocals_demucs(input_path: str, out_dir: Optional[str] = None) -> str:
    """
    Dùng Demucs (--two-stems=vocals) để tách giọng & nền.
    Trả về đường dẫn file instrumental (không có vocal).
    """
    which_or_raise("ffmpeg")
    python_exe = sys.executable or "python"

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="demucs_out_")
    else:
        os.makedirs(out_dir, exist_ok=True)

    # Ví dụ lệnh:
    # python -m demucs --two-stems=vocals -n htdemucs_ft -o <out_dir> <input>
    cmd = [
        python_exe, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs_ft",
        "-o", out_dir,
        input_path
    ]
    run_cmd(cmd)

    # Demucs tạo: out_dir/<model>/<basename>/{no_vocals.wav, vocals.wav}
    base = Path(input_path).stem
    model_dir = Path(out_dir)
    # tìm thư mục con chứa base
    target_sub = None
    for sub in model_dir.glob("*"):
        if sub.is_dir():
            candidate = sub / base
            if candidate.exists():
                target_sub = candidate
                break

    # Nếu không theo cấu trúc model/base, thử tìm sâu hơn
    if target_sub is None:
        for sub in model_dir.rglob(base):
            if sub.is_dir():
                target_sub = sub
                break

    if target_sub is None:
        raise FileNotFoundError("Không tìm được thư mục output của Demucs.")

    instrumental = _find_instrumental_in(target_sub)
    if not instrumental or not instrumental.exists():
        raise FileNotFoundError(f"Không tìm thấy file instrumental sau khi tách trong: {target_sub}")

    print(f"[OK] Instrumental: {instrumental}")
    return str(instrumental)


# ----------------------------- Step 2: Transform audio -----------------------------

def transform_audio(
    input_wav: str,
    pitch_semitones: float = 0.0,
    tempo: float = 1.0,
    apply_reverb: bool = False,
    reverb_amount: float = 0.3,
    volume: float = 1.0
) -> str:
    """
    - Pitch shift (semitones) using librosa
    - Time stretch with librosa (tempo factor)
    - Simple reverb (generated impulse response)
    - Volume (linear gain)
    Returns a processed wav file path.
    """
    y, sr = librosa.load(input_wav, sr=None, mono=True)
    print(f"[Audio] Loaded: {input_wav} (samples={len(y)}, sr={sr})")

    # Pitch
    if abs(pitch_semitones) > 1e-6:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_semitones)

        print(f"[Audio] Pitch shift: {pitch_semitones} semitones")

    # Tempo
    if abs(tempo - 1.0) > 1e-6:
        y = librosa.effects.time_stretch(y, rate=tempo)
        print(f"[Audio] Time-stretch tempo={tempo}")

    # Simple Reverb
    if apply_reverb:
        ir_dur = 0.8  # seconds
        ir_len = int(sr * ir_dur)
        t = np.linspace(0, ir_dur, ir_len, endpoint=False)
        # exponential decay with tiny noise
        ir = (np.exp(-3.0 * t) * (np.random.randn(ir_len) * 0.001 + 1.0)).astype(np.float32)
        ir = ir / max(1e-9, np.max(np.abs(ir)))
        conv = np.convolve(y, ir, mode="full")[: len(y)]
        y = (1.0 - reverb_amount) * y + reverb_amount * conv
        # normalize prevent clipping
        peak = max(1e-9, np.max(np.abs(y)))
        y = y / peak
        print(f"[Audio] Reverb applied (amount={reverb_amount})")

    # Volume
    if abs(volume - 1.0) > 1e-6:
        y = y * volume
        peak = np.max(np.abs(y))
        if peak > 0.999:
            y = y / peak  # avoid clipping
        print(f"[Audio] Volume factor: {volume}")

    out_wav = make_temp_wav("_processed.wav")
    sf.write(out_wav, y, sr)
    print(f"[Audio] Wrote processed wav: {out_wav}")
    return out_wav


# ----------------------------- Step 2.5: Instrumental -> Melody MIDI -> Synth WAV -----------------------------

def audio_to_melody_midi(
    input_wav: str,
    out_midi_path: Optional[str] = None,
    method: str = "pyin",
    hop_length: int = 512,
    fmin_hz: float = 82.41,  # E2
    fmax_hz: float = 1046.5,  # C6 (~ soprano upper)
    min_note_ms: int = 60,
    program: int = 0,
    velocity: int = 100,
) -> str:
    """
    Extract simple monophonic melody from audio and write a MIDI file.
    - method: 'pyin' (default) with fallback to 'yin'
    - program: MIDI program number (0=Acoustic Grand Piano)
    Returns path to MIDI file.
    """
    try:
        import mido
    except Exception as e:
        raise RuntimeError("Cần thư viện 'mido' để xuất MIDI. Hãy cài: pip install mido") from e

    y, sr = librosa.load(input_wav, sr=None, mono=True)
    print(f"[MIDI] Phân tích melody từ: {input_wav} (sr={sr})")

    f0 = None
    voiced_mask = None
    try:
        if method.lower() == "pyin":
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=fmin_hz,
                fmax=fmax_hz,
                sr=sr,
                hop_length=hop_length,
            )
            voiced_mask = (voiced_flag == True) & (~np.isnan(f0))
        else:
            raise ValueError
    except Exception:
        # Fallback: YIN
        print("[MIDI] pyin thất bại, fallback sang librosa.yin ...")
        f0 = librosa.yin(y, fmin=fmin_hz, fmax=fmax_hz, sr=sr, hop_length=hop_length)
        # Simple voicing: keep positive frequencies
        voiced_mask = np.isfinite(f0) & (f0 > 0)

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Group frames into note segments with same quantized MIDI
    segments = []  # list of (start_time, end_time, midi_note)
    current_note = None
    seg_start_idx = None

    def hz_to_midi_int(hz: float) -> int:
        return int(np.clip(np.round(librosa.hz_to_midi(hz)), 0, 127))

    for idx, t in enumerate(times):
        if voiced_mask[idx]:
            note_num = hz_to_midi_int(float(f0[idx]))
        else:
            note_num = None

        if current_note is None and note_num is not None:
            current_note = note_num
            seg_start_idx = idx
        elif current_note is not None and note_num == current_note:
            # continue segment
            pass
        else:
            # close previous segment if exists
            if current_note is not None and seg_start_idx is not None:
                start_t = float(times[seg_start_idx])
                end_t = float(times[idx])
                segments.append((start_t, end_t, current_note))
            # start new segment if note present
            current_note = note_num
            seg_start_idx = idx if note_num is not None else None

    # close last
    if current_note is not None and seg_start_idx is not None:
        start_t = float(times[seg_start_idx])
        end_t = float(times[-1])
        segments.append((start_t, end_t, current_note))

    # Filter short notes
    min_len_sec = max(0.01, min_note_ms / 1000.0)
    segments = [(s, e, n) for (s, e, n) in segments if (e - s) >= min_len_sec]

    if not segments:
        raise RuntimeError("Không tìm thấy đoạn melody hợp lệ để xuất MIDI.")

    # Build MIDI
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo 120 BPM
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # Program change at start
    program_clamped = int(np.clip(program, 0, 127))
    track.append(mido.Message('program_change', program=program_clamped, channel=0, time=0))

    current_tick = 0
    for (s, e, note) in segments:
        start_tick = int(mido.second2tick(s, mid.ticks_per_beat, tempo))
        end_tick = int(mido.second2tick(e, mid.ticks_per_beat, tempo))
        dur_tick = max(1, end_tick - start_tick)
        delta = max(0, start_tick - current_tick)
        if delta > 0:
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta, channel=0))
        else:
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=0, channel=0))
        track.append(mido.Message('note_off', note=note, velocity=0, time=dur_tick, channel=0))
        current_tick = start_tick + dur_tick

    if out_midi_path is None:
        fd, out_midi_path = tempfile.mkstemp(suffix="_melody.mid")
        os.close(fd)
    mid.save(out_midi_path)
    print(f"[MIDI] Đã ghi: {out_midi_path} (n_notes={len(segments)})")
    return out_midi_path


def shift_midi_semitones(midi_in_path: str, semitones: float, midi_out_path: Optional[str] = None) -> str:
    """
    Shift all note_on/note_off by integer semitones. Fractional values are rounded.
    """
    if abs(semitones) < 1e-6:
        return midi_in_path

    try:
        import mido
    except Exception as e:
        raise RuntimeError("Cần thư viện 'mido' để chỉnh MIDI. Hãy cài: pip install mido") from e

    mid = mido.MidiFile(midi_in_path)
    shift = int(np.round(semitones))

    for track in mid.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                new_note = int(np.clip(msg.note + shift, 0, 127))
                msg.note = new_note

    if midi_out_path is None:
        fd, midi_out_path = tempfile.mkstemp(suffix="_shift.mid")
        os.close(fd)
    mid.save(midi_out_path)
    print(f"[MIDI] Shift pitch {shift} semitones -> {midi_out_path}")
    return midi_out_path


def synth_midi_with_sf2(
    midi_path: str,
    soundfont_path: str,
    out_wav_path: Optional[str] = None,
    sample_rate: int = 44100,
    gain: float = 0.8,
) -> str:
    """
    Synthesize MIDI to WAV using FluidSynth CLI and a SoundFont (.sf2).
    """
    which_or_raise("fluidsynth")
    if not os.path.isfile(soundfont_path):
        raise FileNotFoundError(f"Không tìm thấy SoundFont: {soundfont_path}")

    if out_wav_path is None:
        out_wav_path = make_temp_wav("_synth.wav")

    cmd = [
        "fluidsynth",
        "-ni",
        "-F", out_wav_path,
        "-r", str(sample_rate),
        "-g", str(gain),
        soundfont_path,
        midi_path,
    ]
    run_cmd(cmd)
    print(f"[MIDI] Synth -> {out_wav_path}")
    return out_wav_path


# ----------------------------- Step 3: Mix background (rain) -----------------------------

def mix_background(main_wav: str, background_audio: Optional[str], bg_db: float = -18.0) -> str:
    """
    Mix main audio with background (e.g., rain).
    background_audio can be any format supported by ffmpeg.
    bg_db: relative loudness (negative dB to make bg quieter).
    Returns new wav path.
    """
    if not background_audio:
        return main_wav

    main = AudioSegment.from_file(main_wav)
    bg = AudioSegment.from_file(background_audio)

    # Loop BG to match duration
    if len(bg) < len(main):
        times = int(np.ceil(len(main) / max(1, len(bg))))
        bg = bg * max(1, times)
    bg = bg[: len(main)]

    # Adjust BG level (e.g., -18 dB)
    bg = bg + bg_db

    mixed = main.overlay(bg)
    out_wav = make_temp_wav("_mixed_bg.wav")
    mixed.export(out_wav, format="wav")
    print(f"[Mix] Mixed audio -> {out_wav}")
    return out_wav


# ----------------------------- Step 4: Render video -----------------------------

def get_audio_duration_ffprobe(audio_path: str) -> float:
    which_or_raise("ffprobe")
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    out = run_cmd(cmd, log=False)
    try:
        return float(out.strip())
    except Exception:
        return 0.0


def concat_loop_audio_to_duration(src_audio: str, target_seconds: int) -> str:
    """Loop audio by concat to reach target length, then trim exact."""
    which_or_raise("ffmpeg")
    dur = get_audio_duration_ffprobe(src_audio)
    if dur <= 0:
        raise RuntimeError("Không lấy được duration audio để loop.")

    loops = int(np.ceil(target_seconds / dur))
    list_file = tempfile.NamedTemporaryFile(delete=False, suffix="_list.txt")
    list_file_path = list_file.name
    list_file.close()

    with open(list_file_path, "w", encoding="utf-8") as f:
        for _ in range(loops):
            f.write(f"file '{os.path.abspath(src_audio)}'\n")

    concat_out = make_temp_wav("_concat.wav")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", concat_out])
    os.remove(list_file_path)

    trimmed_out = make_temp_wav("_trimmed.wav")
    run_cmd(["ffmpeg", "-y", "-i", concat_out, "-t", str(target_seconds), "-c", "copy", trimmed_out])

    try:
        os.remove(concat_out)
    except Exception:
        pass

    return trimmed_out


def create_video_from_image(image_path: str, audio_path: str, output_path: str,
                            resolution: str = "1920x1080", loop_hours: Optional[float] = None):
    which_or_raise("ffmpeg")

    if loop_hours is None:
        # Match audio duration
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage",
            "-vf", f"scale={resolution}",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        run_cmd(cmd)
    else:
        target_seconds = int(loop_hours * 3600)
        # build looped audio exact
        looped_audio = concat_loop_audio_to_duration(audio_path, target_seconds)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", looped_audio,
            "-c:v", "libx264", "-tune", "stillimage",
            "-vf", f"scale={resolution}",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        run_cmd(cmd)
    print(f"[Video] Created: {output_path}")


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Music Cover Tool - Full (offline, Demucs)")
    p.add_argument("--input_audio", required=True, help="Đường dẫn file nhạc đầu vào (mp3/wav/...)")
    p.add_argument("--image", required=True, help="Ảnh nền (jpg/png/...)")
    p.add_argument("--rain", type=str, default="", help="(Tùy chọn) Đường dẫn file tiếng mưa để trộn")
    p.add_argument("--rain_db", type=float, default=-18.0, help="Âm lượng tương đối của tiếng mưa (dBFS, âm là nhỏ hơn)")
    p.add_argument("--pitch", type=float, default=0.0, help="Đổi cao độ (semitones, ví dụ -2 hoặc 3)")
    p.add_argument("--tempo", type=float, default=1.0, help="Tempo multiplier (1.05 = nhanh 5%)")
    p.add_argument("--reverb", action="store_true", help="Bật reverb đơn giản")
    p.add_argument("--volume", type=float, default=1.0, help="Hệ số volume sau cùng (1.0 giữ nguyên)")

    # New: Melody->MIDI->Synth options
    p.add_argument("--soundfont", type=str, default="", help="Đường dẫn SoundFont (.sf2) để synth MIDI")
    p.add_argument("--midi_pitch", type=float, default=0.0, help="Dịch cao độ MIDI (semitones, ví dụ ±1)")
    p.add_argument("--midi_program", type=int, default=0, help="MIDI program (0=Grand Piano)")
    p.add_argument("--midi_velocity", type=int, default=100, help="Độ mạnh nốt MIDI (0-127)")
    p.add_argument("--midi_min_note_ms", type=int, default=60, help="Bỏ qua nốt ngắn hơn (ms)")
    p.add_argument("--melody_method", choices=["pyin"], default="pyin", help="Thuật toán tách melody")
    p.add_argument("--fs_gain", type=float, default=0.8, help="FluidSynth gain (0-5)")

    p.add_argument("--mode", choices=["full", "loop"], default="full", help="Xuất full bài hoặc loop N giờ")
    p.add_argument("--loop_hours", type=float, default=1.0, help="Số giờ lặp nếu mode=loop")
    p.add_argument("--resolution", type=str, default="1920x1080", help="Độ phân giải video (ví dụ 1920x1080)")
    p.add_argument("--output", required=True, help="Đường dẫn file MP4 đầu ra")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate input files
    if not os.path.isfile(args.input_audio):
        raise FileNotFoundError(f"Không tìm thấy file nhạc: {args.input_audio}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {args.image}")
    if args.rain and not os.path.isfile(args.rain):
        raise FileNotFoundError(f"Không tìm thấy file tiếng mưa: {args.rain}")

    print("=== Music Cover Tool (Demucs) ===")
    print("[1/4] Tách giọng (Demucs) ...")
    instrumental = separate_vocals_demucs(args.input_audio)

    print("[2/4] Biến đổi instrumental (pitch/tempo/reverb/volume) ...")
    processed = transform_audio(
        instrumental,
        pitch_semitones=args.pitch,
        tempo=args.tempo,
        apply_reverb=args.reverb,
        reverb_amount=0.3,
        volume=args.volume
    )

    # Optional: Melody -> MIDI -> Synth using SoundFont
    processed_for_mix = processed
    if args.soundfont:
        try:
            print("[2.5/4] Instrumental → MIDI (melody), chỉnh pitch MIDI, synth bằng SoundFont ...")
            midi_tmp = audio_to_melody_midi(
                processed,
                out_midi_path=None,
                method=args.melody_method,
                hop_length=512,
                fmin_hz=82.41,
                fmax_hz=1046.5,
                min_note_ms=int(args.midi_min_note_ms),
                program=int(args.midi_program),
                velocity=int(args.midi_velocity),
            )
            midi_shifted = shift_midi_semitones(midi_tmp, args.midi_pitch)
            synth_wav = synth_midi_with_sf2(midi_shifted, args.soundfont, out_wav_path=None, sample_rate=44100, gain=float(args.fs_gain))
            processed_for_mix = synth_wav
        except Exception as e:
            print(f"[Cảnh báo] Melody→MIDI→Synth lỗi, sẽ dùng audio đã xử lý ban đầu. Lý do: {e}")

    print("[3/4] Trộn nền (mưa) nếu có ...")
    mixed = mix_background(processed_for_mix, args.rain, bg_db=args.rain_db) if args.rain else processed_for_mix

    print("[4/4] Render video (ffmpeg) ...")
    if args.mode == "full":
        create_video_from_image(args.image, mixed, args.output, resolution=args.resolution, loop_hours=None)
    else:
        create_video_from_image(args.image, mixed, args.output, resolution=args.resolution, loop_hours=args.loop_hours)

    print(f"All done! Output -> {args.output}")


if __name__ == "__main__":
    main()
