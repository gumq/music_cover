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

    print("[3/4] Trộn nền (mưa) nếu có ...")
    mixed = mix_background(processed, args.rain, bg_db=args.rain_db) if args.rain else processed

    print("[4/4] Render video (ffmpeg) ...")
    if args.mode == "full":
        create_video_from_image(args.image, mixed, args.output, resolution=args.resolution, loop_hours=None)
    else:
        create_video_from_image(args.image, mixed, args.output, resolution=args.resolution, loop_hours=args.loop_hours)

    print(f"All done! Output -> {args.output}")


if __name__ == "__main__":
    main()
