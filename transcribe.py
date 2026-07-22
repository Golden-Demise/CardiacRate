"""
把 video.MOV (或已抽出的 video_audio.wav) 轉成逐字稿。

使用方式 (在你電腦上、有裝好的 venv 或新環境中):
    pip install faster-whisper
    python transcribe.py

會產生:
    transcript.txt  - 純文字逐字稿
    transcript.srt   - 帶時間軸的字幕檔

備註:
- faster-whisper 內建用 PyAV 解碼音訊，不需要另外裝 ffmpeg。
- 如果你的電腦有 NVIDIA GPU (你的 venv 裡已經有 torch 2.13 + cu126)，
  下面會自動改用 GPU 加速；沒有的話會退回 CPU。
- 模型第一次執行時會從 Hugging Face 下載 (需要網路)，之後會快取在本機。
- model_size 可改成 "tiny"/"base"/"small"/"medium"/"large-v3"，
  越大越準但越慢。中文語音建議至少用 "small" 以上。
"""

from pathlib import Path
from faster_whisper import WhisperModel

AUDIO_PATH = Path(__file__).parent / "video_audio.wav"  # 沒有的話也可以直接指到 video.MOV
MODEL_SIZE = "medium"

def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    device, compute_type = pick_device()
    print(f"使用裝置: {device} ({compute_type}), 模型: {MODEL_SIZE}")

    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        language=None,       # 自動偵測語言；若確定是中文可改成 "zh"
        vad_filter=True,     # 過濾靜音，減少幻覺輸出
    )

    print(f"偵測語言: {info.language} (信心度 {info.language_probability:.2f})")

    txt_lines = []
    srt_lines = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        print(f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {text}")
        txt_lines.append(text)
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        srt_lines.append(text)
        srt_lines.append("")
        idx += 1

    out_dir = Path(__file__).parent
    (out_dir / "transcript.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    (out_dir / "transcript.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    print("完成！已輸出 transcript.txt 與 transcript.srt")

if __name__ == "__main__":
    main()
