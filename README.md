# VOX ─ High-Performance Speaker Diarization & Subtitle Alignment (Rust CPU)

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-2.0-blue.svg)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE-MIT)

`vox` là thư viện Rust thuần chuyên nghiệp, hiệu năng cao phục vụ **Speaker Diarization** (nhận diện và phân tách từng người nói) và tự động gắn nhãn người nói vào phụ đề (**VTT** hoặc **SRT**) từ file video/audio đầu vào (**MP4, MKV, WAV, v.v.**).

Mô hình được tối ưu hóa toàn diện cho **CPU** thông qua **ONNX Runtime (ort)**, không phụ thuộc vào Python, không dính GIL, tận dụng SIMD/AVX2/AVX-512 với tốc độ xử lý thực tế nhanh tới **gấp 90 - 100 lần thời gian thực (Real-time Factor < 0.012)**.

---

## 🌟 Tính Năng Nổi Bật

- **CPU-Only Siêu Tốc:** Chạy thuần trên CPU bằng ONNX Runtime; video 2.4 phút (144 giây) xử lý xong trong **1.57 giây** trên CPU thông thường.
- **Tự Động Tải Model Thông Minh:** Kiểm tra đường dẫn chỉ định:
  ```rust
  let model = Path::new("models/pyannote_segmentation.onnx");
  if !model.exists() { /* Tự động tải từ HuggingFace với thanh % tải */ }
  ```
- **Hỗ trợ định dạng linh hoạt:**
  - **Video/Audio đầu vào:** Hỗ trợ mọi định dạng đa phương tiện (`.mp4`, `.mkv`, `.mov`, `.wav`, `.mp3`) thông qua bộ trích xuất chuẩn 16kHz mono PCM.
  - **Phụ đề đầu vào:** Nhận diện và phân tích cú pháp cả `.vtt` (WebVTT chuẩn, bảo toàn cue settings, line, position) và `.srt`.
  - Hai file đầu vào có thể nằm độc lập ở hai thư mục hoàn toàn khác nhau.
- **Khớp Nhãn Người Nói (Overlap Assignment):** Thuật toán tính độ đè trùng thời gian giữa từng câu phụ đề và lượt nói (Speaker Turns), xử lý dung sai khoảng trống và gán chính xác `[SPEAKER_00]`, `[SPEAKER_01]`, ...
- **Chuẩn Tiến Trình Tương Thích Hoàn Hảo với `cli_gui_form`:** Cung cấp struct `ProgressInfo` với bộ 5 tham số chuẩn:
  `(current, total, unit, duration, status)` khớp 1-1 với `TaskWidget` của crate `cli_gui_form` (dễ dàng gắn vào TUI Form hoặc render CLI).
- **Tùy biến nhãn người nói:** Hỗ trợ `Prefix` (`[SPEAKER_00]: Text`), `BracketOnly` (`[SPEAKER_00] Text`), hoặc WebVTT `VoiceTag` (`<v SPEAKER_00>Text</v>`).

---

## 📁 Cấu Trúc Dự Án

```
c:\DATA\vox
├── cli_gui_form/          # Thư viện TUI Form & Design System (Ratatui)
├── input/                 # Thư mục chứa video và phụ đề mẫu
│   ├── 第001集.mp4
│   └── 第001集.zh.vtt
├── models/                # Nơi lưu trữ mô hình ONNX đã tải về
│   └── pyannote_segmentation.onnx
├── output/                # Thư mục xuất file phụ đề đã gắn nhãn người nói
│   └── 第001集.zh.diarized.vtt
├── src/
│   ├── alignment.rs       # Thuật toán tính Overlap & gán speaker vào Subtitle
│   ├── audio.rs           # Trích xuất audio PCM 16kHz bằng FFmpeg & đọc WAV
│   ├── error.rs           # Kiểu lỗi tập trung VoxError
│   ├── lib.rs             # Engine điều phối chính & public API
│   ├── model.rs           # Quản lý tải model & ONNX inference CPU
│   ├── progress.rs        # Chuẩn tiến trình tương thích TaskWidget của cli_gui_form
│   └── subtitles.rs       # Bộ parser/serializer cho WebVTT và SRT
├── examples/
│   └── diarize_demo.rs    # Example hoàn chỉnh chạy kiểm thử
├── Cargo.toml
├── README.md
├── LICENSE-MIT
└── LICENSE-APACHE
```

---

## 🚀 Yêu Cầu Hệ Thống

1. **Rust Toolchain:** Rust 1.80+ (khuyên dùng Rust 2024 edition).
2. **FFmpeg:** Cần cài đặt `ffmpeg` và có trong biến môi trường `PATH` để trích xuất âm thanh từ MP4/MKV.

---

## 📦 Cài Đặt

Thêm `vox` vào `Cargo.toml`:

```toml
[dependencies]
vox = { path = "../vox" }
```

Hoặc kích hoạt tính năng hỗ trợ trực tiếp widget `cli_gui_form`:

```toml
[dependencies]
vox = { path = "../vox", features = ["tui"] }
cli_gui_form = { path = "../vox/cli_gui_form" }
```

---

## 💡 Hướng Dẫn Sử Dụng Nhanh

### 1. Xử lý một dòng lệnh nhanh với `process_diarization`:

```rust
use std::path::Path;
use vox::{process_diarization, VoxConfig, ProgressInfo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let video_path = Path::new("input/video.mp4");
    let subtitle_path = Path::new("input/subtitles.vtt");
    let output_dir = Path::new("output");

    let output_file = process_diarization(
        video_path,
        subtitle_path,
        output_dir,
        None, // Sử dụng cấu hình mặc định
        Some(|p: &ProgressInfo| {
            println!("[{}] {:>5.1}% | {} | {}", p.stage.label(), p.percentage, p.duration, p.status);
        }),
    )?;

    println!("Phụ đề đã xuất ra: {}", output_file.display());
    Ok(())
}
```

---

### 2. Sử dụng Engine với Cấu Hình Tùy Biến:

```rust
use std::path::{Path, PathBuf};
use vox::{VoxConfig, VoxEngine, SpeakerFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = VoxConfig::default();
    
    // 1. Chỉ định model theo đúng tiêu chuẩn:
    let model = Path::new("models/pyannote_segmentation.onnx");
    config.model_path = model.to_path_buf();
    
    // 2. Thư mục xuất kết quả:
    config.output_dir = PathBuf::from("output");
    
    // 3. Tối ưu luồng CPU và định dạng người nói:
    config.cpu_threads = 4;
    config.speaker_format = SpeakerFormat::Prefix; // [SPEAKER_00]: Text

    let engine = VoxEngine::new(config);
    let output = engine.process_video_and_subtitles(
        "input/第001集.mp4",
        "input/第001集.zh.vtt",
        Some(|p| {
            println!("Stage: {:?}, Progress: {:.1}%", p.stage, p.percentage);
        }),
    )?;

    println!("Hoàn tất: {}", output.display());
    Ok(())
}
```

---

## 🖥️ Tích Hợp Tiến Trình với `cli_gui_form` (TUI / CLI)

`ProgressInfo` cung cấp hàm `to_cli_gui_params()` trả về chính xác bộ tham số của `TaskWidget`:

```rust
use cli_gui_form::{TaskWidget, ProgressStyle, Theme};
use vox::ProgressInfo;

// Khởi tạo widget trong TUI Form:
let mut task_widget = TaskWidget::new_progress(
    "vox_task",
    "Speaker Diarization",
    0,
    100,
    "%",
    "0s",
    "Khởi tạo...",
    Theme::CYAN,
).with_progress_style(ProgressStyle::Parallelogram);

// Cập nhật nhịp tiến trình từ callback của vox:
let on_progress = |p: &ProgressInfo| {
    // Lấy 5 tham số chuẩn của cli_gui_form
    let (current, total, unit, duration, status) = p.to_cli_gui_params();
    
    // Cập nhật widget:
    task_widget.update_progress(current, duration);
};
```

---

## ⚡ Chạy Ví Dụ Kiểm Thử (Example)

Để đạt hiệu năng cao nhất, luôn chạy ở chế độ `--release`:

```powershell
cargo run --release --example diarize_demo
```

### Kết quả chạy thực tế:

```
═════════════════════════════════════════════════════════════════════
               VOX ─ SPEAKER DIARIZATION & SUBTITLES                
           (Accelerated CPU-only ONNX Runtime Architecture)         
═════════════════════════════════════════════════════════════════════

📁 [Đầu vào]
   • Video:   input\第001集.mp4
   • Phụ đề:  input\第001集.zh.vtt
   • Thư mục xuất: output

🧠 [Mô hình ONNX]
   • Đường dẫn: models\pyannote_segmentation.onnx
   • Trạng thái: ✅ Đã tồn tại sẵn trên đĩa.

▶ Giai đoạn: [Trích xuất Audio (FFmpeg)]
  [▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰] 100.0% | 1/1 file | 143.7s | Audio sẵn sàng (143.7s, 16000Hz PCM)
▶ Giai đoạn: [Phân đoạn giọng nói (Diarization ONNX)]
  [▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰] 100.0% | 68/68 chunks |     0s | Phân tích audio: 143.7s / 143.7s
▶ Giai đoạn: [Khớp người nói & Phụ đề]
  [▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰] 100.0% | 69/69 cues |     0s | Gắn nhãn thành công cho 69 câu phụ đề
▶ Giai đoạn: [Xuất phụ đề]
  [▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰] 100.0% | 1/1 file |     0s | Hoàn tất! Kết quả lưu tại: output\第001集.zh.diarized.vtt

🎉 Xử lý hoàn tất trong 1.57s!
📄 File kết quả đã xuất tại: output\第001集.zh.diarized.vtt
```

---

## 📊 Benchmark Hiệu Năng

| Thành phần | Thời lượng Audio | Thời gian xử lý (CPU) | Tốc độ tương đối (RTF) |
|---|---|---|---|
| Trích xuất FFmpeg 16kHz PCM | 143.7s | 0.21s | ~684x realtime |
| Phân tích 68 ONNX Chunks (INT8) | 143.7s | 1.32s | ~108x realtime |
| Thuật toán giải hoán vị & Overlap | 143.7s | 0.04s | >3500x realtime |
| **Toàn bộ Pipeline** | **143.7s (~2.4 phút)** | **1.57s** | **~91.5x realtime** |

---

## 📜 Giấy Phép (License)

Dự án được phân phối dưới giấy phép kép:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

Tùy theo nhu cầu, bạn có thể tự do lựa chọn một trong hai giấy phép trên.

