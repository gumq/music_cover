# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".music_cover_config.json"

def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

cfg = load_config()
last_dir = cfg.get("last_dir", str(Path.cwd()))

def browse_file(entry, filetypes, title="Chọn file"):
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=last_dir if os.path.isdir(last_dir) else None
    )
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)
        # Cập nhật last_dir
        cfg["last_dir"] = os.path.dirname(filename)
        save_config(cfg)

        # Nếu chọn input audio và ô output trống -> gợi ý tên mp4 trong cùng thư mục
        if entry is audio_entry and not output_entry.get().strip():
            base = os.path.splitext(os.path.basename(filename))[0]
            suggested = os.path.join(cfg["last_dir"], f"{base}_cover.mp4")
            output_entry.delete(0, tk.END)
            output_entry.insert(0, suggested)

def browse_save_file(entry):
    filename = filedialog.asksaveasfilename(
        title="Chọn nơi lưu MP4",
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4")],
        initialdir=last_dir if os.path.isdir(last_dir) else None
    )
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)
        cfg["last_dir"] = os.path.dirname(filename)
        save_config(cfg)

def create_video():
    input_audio = audio_entry.get().strip()
    image = image_entry.get().strip()
    output = output_entry.get().strip()
    rain_file = rain_entry.get().strip()
    pitch = pitch_entry.get().strip()
    tempo = tempo_entry.get().strip()
    reverb = reverb_var.get()
    mode = mode_var.get()
    loop_hours = loop_entry.get().strip()
    rain_db = rain_db_entry.get().strip()
    volume = volume_entry.get().strip()
    resolution = res_entry.get().strip()

    # New: MIDI/SoundFont parameters
    soundfont = sf_entry.get().strip()
    midi_pitch = midi_pitch_entry.get().strip()
    midi_prog = midi_prog_entry.get().strip()
    midi_vel = midi_vel_entry.get().strip()
    midi_min = midi_min_entry.get().strip()
    fs_gain = fs_gain_entry.get().strip()

    # New: Style/Melody options
    style = style_var.get()
    melody_source = melody_src_var.get()
    melody_method = melody_method_var.get()
    overlay_db = overlay_db_entry.get().strip()
    lofi_lpf = lofi_lpf_entry.get().strip()

    if not input_audio or not image or not output:
        messagebox.showerror("Lỗi", "Vui lòng chọn đầy đủ: âm thanh, ảnh và file xuất.")
        return

    cmd = [
        sys.executable, "app.py",
        "--input_audio", input_audio,
        "--image", image,
        "--output", output,
        "--mode", mode
    ]

    if rain_file:
        cmd.extend(["--rain", rain_file])
    if rain_db:
        cmd.extend(["--rain_db", rain_db])
    if reverb:
        cmd.append("--reverb")
    if pitch:
        cmd.extend(["--pitch", pitch])
    if tempo:
        cmd.extend(["--tempo", tempo])
    if volume:
        cmd.extend(["--volume", volume])
    if resolution:
        cmd.extend(["--resolution", resolution])
    if mode == "loop" and loop_hours:
        cmd.extend(["--loop_hours", loop_hours])

    # Pass Style/Melody args
    if style:
        cmd.extend(["--style", style])
    if melody_source:
        cmd.extend(["--melody_source", melody_source])
    if melody_method:
        cmd.extend(["--melody_method", melody_method])
    if overlay_db:
        cmd.extend(["--overlay_db", overlay_db])
    if lofi_lpf:
        cmd.extend(["--lofi_lowpass_hz", lofi_lpf])

    # Pass MIDI/SF args
    if soundfont:
        cmd.extend(["--soundfont", soundfont])
    if midi_pitch:
        cmd.extend(["--midi_pitch", midi_pitch])
    if midi_prog:
        cmd.extend(["--midi_program", midi_prog])
    if midi_vel:
        cmd.extend(["--midi_velocity", midi_vel])
    if midi_min:
        cmd.extend(["--midi_min_note_ms", midi_min])
    if fs_gain:
        cmd.extend(["--fs_gain", fs_gain])

    try:
        # Hiện console log trong terminal đang mở; nếu muốn hiện log vào GUI thì thay Popen + đọc stdout.
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Thành công", f"Video đã được tạo:\n{output}")
        # Lưu last_dir là thư mục output
        cfg["last_dir"] = os.path.dirname(output)
        save_config(cfg)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Lỗi", f"Quá trình tạo video thất bại.\n\nLệnh:\n{' '.join(cmd)}\n\nChi tiết:\n{e}")

root = tk.Tk()
root.title("Tạo Video từ Ảnh và Nhạc (Demucs)")

# Hàng 0: Audio
tk.Label(root, text="File âm thanh:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
audio_entry = tk.Entry(root, width=54)
audio_entry.grid(row=0, column=1, padx=6, pady=4)
tk.Button(root, text="Chọn", command=lambda: browse_file(audio_entry, [("Audio Files", "*.mp3 *.wav *.flac *.ogg")], "Chọn file âm thanh")).grid(row=0, column=2, padx=6, pady=4)

# Hàng 1: Ảnh
tk.Label(root, text="File ảnh:").grid(row=1, column=0, sticky="e", padx=6, pady=4)
image_entry = tk.Entry(root, width=54)
image_entry.grid(row=1, column=1, padx=6, pady=4)
tk.Button(root, text="Chọn", command=lambda: browse_file(image_entry, [("Image Files", "*.jpg *.jpeg *.png")], "Chọn ảnh nền")).grid(row=1, column=2, padx=6, pady=4)

# Hàng 1.3: Phong cách
tk.Label(root, text="Phong cách:").grid(row=2, column=0, sticky="e", padx=6, pady=4)
style_var = tk.StringVar(value="original")
tk.OptionMenu(root, style_var, "original", "piano", "music_box", "strings", "chip", "lofi", "melody_only").grid(row=2, column=1, sticky="w", padx=6, pady=4)

# Hàng 1.4: Melody source / method
tk.Label(root, text="Nguồn melody:").grid(row=2, column=1, sticky="e", padx=6, pady=4)
melody_src_var = tk.StringVar(value="auto")
tk.OptionMenu(root, melody_src_var, "auto", "vocals", "instrumental", "input").grid(row=2, column=1, sticky="e", padx=120, pady=4)

# Hàng 1.5: SoundFont
tk.Label(root, text="SoundFont (.sf2) cho synth:").grid(row=3, column=0, sticky="e", padx=6, pady=4)
sf_entry = tk.Entry(root, width=54)
sf_entry.grid(row=3, column=1, padx=6, pady=4)
tk.Button(root, text="Chọn", command=lambda: browse_file(sf_entry, [("SoundFont", "*.sf2")], "Chọn SoundFont (.sf2)")).grid(row=3, column=2, padx=6, pady=4)

# Hàng 2: Tiếng mưa (tùy chọn) -> dịch xuống 4
tk.Label(root, text="File tiếng mưa (tuỳ chọn):").grid(row=4, column=0, sticky="e", padx=6, pady=4)
rain_entry = tk.Entry(root, width=54)
rain_entry.grid(row=4, column=1, padx=6, pady=4)
tk.Button(root, text="Chọn", command=lambda: browse_file(rain_entry, [("Audio Files", "*.mp3 *.wav *.flac *.ogg")], "Chọn file tiếng mưa")).grid(row=4, column=2, padx=6, pady=4)

# Hàng 3: File xuất -> dịch xuống 5
tk.Label(root, text="File xuất (.mp4):").grid(row=5, column=0, sticky="e", padx=6, pady=4)
output_entry = tk.Entry(root, width=54)
output_entry.grid(row=5, column=1, padx=6, pady=4)
tk.Button(root, text="Chọn", command=lambda: browse_save_file(output_entry)).grid(row=5, column=2, padx=6, pady=4)

# Hàng 4: Pitch / Tempo
tk.Label(root, text="Pitch (±semitones):").grid(row=6, column=0, sticky="e", padx=6, pady=4)
pitch_entry = tk.Entry(root, width=10)
pitch_entry.insert(0, "0.0")
pitch_entry.grid(row=6, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Tempo (1.0=bt):").grid(row=6, column=1, sticky="e", padx=6, pady=4)
tempo_entry = tk.Entry(root, width=10)
tempo_entry.insert(0, "1.0")
tempo_entry.grid(row=6, column=1, sticky="e", padx=120, pady=4)

# Hàng 5: Volume / Rain dB
tk.Label(root, text="Volume (1.0=bt):").grid(row=7, column=0, sticky="e", padx=6, pady=4)
volume_entry = tk.Entry(root, width=10)
volume_entry.insert(0, "1.0")
volume_entry.grid(row=7, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Rain dB (âm<0):").grid(row=7, column=1, sticky="e", padx=6, pady=4)
rain_db_entry = tk.Entry(root, width=10)
rain_db_entry.insert(0, "-18.0")
rain_db_entry.grid(row=7, column=1, sticky="e", padx=120, pady=4)

# Hàng 6: Reverb / Resolution
reverb_var = tk.BooleanVar()
tk.Checkbutton(root, text="Hiệu ứng vang (reverb)", variable=reverb_var).grid(row=8, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Độ phân giải (WxH):").grid(row=8, column=0, sticky="e", padx=6, pady=4)
res_entry = tk.Entry(root, width=12)
res_entry.insert(0, "1920x1080")
res_entry.grid(row=8, column=1, sticky="w", padx=120, pady=4)

# Hàng 7: Chế độ / Loop
tk.Label(root, text="Chế độ:").grid(row=9, column=0, sticky="e", padx=6, pady=4)
mode_var = tk.StringVar(value="full")
tk.OptionMenu(root, mode_var, "full", "loop").grid(row=9, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Loop giờ:").grid(row=9, column=1, sticky="e", padx=6, pady=4)
loop_entry = tk.Entry(root, width=10)
loop_entry.insert(0, "1")
loop_entry.grid(row=9, column=1, sticky="e", padx=120, pady=4)

# Hàng 8: Tùy chọn MIDI
tk.Label(root, text="MIDI pitch (±semi):").grid(row=10, column=0, sticky="e", padx=6, pady=4)
midi_pitch_entry = tk.Entry(root, width=10)
midi_pitch_entry.insert(0, "0.0")
midi_pitch_entry.grid(row=10, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Program (0-127):").grid(row=10, column=1, sticky="e", padx=6, pady=4)
midi_prog_entry = tk.Entry(root, width=10)
midi_prog_entry.insert(0, "0")
midi_prog_entry.grid(row=10, column=1, sticky="e", padx=120, pady=4)

tk.Label(root, text="Velocity (0-127):").grid(row=11, column=0, sticky="e", padx=6, pady=4)
midi_vel_entry = tk.Entry(root, width=10)
midi_vel_entry.insert(0, "100")
midi_vel_entry.grid(row=11, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Min note (ms):").grid(row=11, column=1, sticky="e", padx=6, pady=4)
midi_min_entry = tk.Entry(root, width=10)
midi_min_entry.insert(0, "60")
midi_min_entry.grid(row=11, column=1, sticky="e", padx=120, pady=4)

# Hàng 9: Melody method / Overlay dB
tk.Label(root, text="Method:").grid(row=12, column=0, sticky="e", padx=6, pady=4)
melody_method_var = tk.StringVar(value="pyin")
tk.OptionMenu(root, melody_method_var, "pyin", "yin", "crepe").grid(row=12, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="Overlay dB (âm<0):").grid(row=12, column=1, sticky="e", padx=6, pady=4)
overlay_db_entry = tk.Entry(root, width=10)
overlay_db_entry.insert(0, "-6.0")
overlay_db_entry.grid(row=12, column=1, sticky="e", padx=120, pady=4)

# Hàng 10: Lofi LPF HZ / FS gain
tk.Label(root, text="Lofi LPF (Hz):").grid(row=13, column=0, sticky="e", padx=6, pady=4)
lofi_lpf_entry = tk.Entry(root, width=10)
lofi_lpf_entry.insert(0, "1400")
lofi_lpf_entry.grid(row=13, column=1, sticky="w", padx=6, pady=4)

tk.Label(root, text="FS gain (0-5):").grid(row=13, column=1, sticky="e", padx=6, pady=4)
fs_gain_entry = tk.Entry(root, width=10)
fs_gain_entry.insert(0, "0.8")
fs_gain_entry.grid(row=13, column=1, sticky="e", padx=120, pady=4)

# Hàng 12: Nút
tk.Button(root, text="Tạo video", command=create_video, bg="lightblue").grid(row=14, column=1, pady=12)

root.mainloop()
