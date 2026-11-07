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
- **Xử Lý Chú Thích Màn Hình Thông Minh:** 
  - Tự động phát hiện các câu phụ đề là thẻ tên nhân vật, chú thích, tiêu đề dạng `《...》`, `【...】`, `[...]`, `(...)` (ví dụ `《陆清安》`, `《小女儿-陆淼淼》`, `《系统》`).
  - **Không gắn nhãn người nói đè vào chú thích**, giữ nguyên vẹn nội dung gốc.
  - Tự động trích xuất tên nhân vật từ thẻ tên màn hình để liên kết với hồ sơ giọng nói.
- **Quản Lý Nhân Vật Xuyên Suốt Phim Bộ Nhiều Tập (Cross-Episode Consistency):**
  - Cung cấp **`SpeakerRegistry`**: Hệ thống trích xuất vân giọng (Acoustic Voiceprint Signature) và so khớp Cosine Similarity.
  - Khi xử lý chuỗi nhiều tập (Tập 1, Tập 2, Tập 3...):
    - **Nhân vật cũ quay lại:** Được nhận diện tự động và **giữ nguyên mã ID** (ví dụ `SPEAKER_01` ở Tập 1 vẫn là `SPEAKER_01` ở Tập 2).
    - **Nhân vật mới xuất hiện:** Hệ thống tự động phát hiện và cấp mã ID mới (`SPEAKER_02`, `SPEAKER_03`...).
    - Xuất file **`speaker_registry.json`** lưu trữ toàn bộ danh sách nhân vật, thời lượng nói và các tập xuất hiện.
- **Tự Động Quét & Ghép Cặp Thư Mục Phim:** Hàm `scan_series_directory` và `process_series` tự động ghép cặp file video và phụ đề tương ứng trong thư mục `input/`.
- **Chuẩn Tiến Trình Tương Thích Hoàn Hảo với `cli_gui_form`:** Cung cấp struct `ProgressInfo` với bộ 5 tham số chuẩn:
  `(current, total, unit, duration, status)` khớp 1-1 với `TaskWidget` của crate `cli_gui_form` (dễ dàng gắn vào TUI Form hoặc render CLI).
- **Tùy biến nhãn người nói:** Hỗ trợ `Prefix` (`[SPEAKER_00]: Text`), `BracketOnly` (`[SPEAKER_00] Text`), WebVTT `VoiceTag` (`<v SPEAKER_00>Text</v>`), hoặc `NamedIfAvailable` (`[陆清安]: Text`).

---

## 📁 Cấu Trúc Dự Án

```
c:\DATA\vox
├── cli_gui_form/          # Thư viện TUI Form & Design System (Ratatui)
├── input/                 # Thư mục chứa phim với nhiều video & phụ đề các tập
│   ├── 第001集.mp4
│   ├── 第001集.zh.vtt
│   ├── 第002集.mp4
│   └── 第002集.zh.vtt
├── models/                # Nơi lưu trữ mô hình ONNX đã tải về
│   └── pyannote_segmentation.onnx
├── output/                # Thư mục xuất file phụ đề và sổ bộ nhân vật
│   ├── 第001集.zh.diarized.vtt
│   ├── 第002集.zh.diarized.vtt
│   └── speaker_registry.json
├── src/
│   ├── alignment.rs       # Thuật toán tính Overlap & gán speaker vào Subtitle
│   ├── audio.rs           # Trích xuất audio PCM 16kHz bằng FFmpeg & đọc WAV
│   ├── error.rs           # Kiểu lỗi tập trung VoxError
│   ├── lib.rs             # Engine điều phối chính & public API
│   ├── model.rs           # Quản lý tải model & ONNX inference CPU
│   ├── progress.rs        # Chuẩn tiến trình tương thích TaskWidget của cli_gui_form
│   ├── series.rs          # Quét và xử lý tự động phim bộ nhiều tập
│   ├── speaker_registry.rs# Nhận diện vân giọng & quản lý nhân vật xuyên suốt
│   └── subtitles.rs       # Bộ parser/serializer cho WebVTT và SRT
├── examples/
│   ├── diarize_demo.rs    # Example xử lý 1 tập đơn lẻ
│   └── series_demo.rs     # Example xử lý trọn bộ series nhiều tập
├── Cargo.toml
├── README.md
├── LICENSE-MIT
└── LICENSE-APACHE
```

---

## 🚀 Yêu Cầu Hệ Thống

1. **Rust Toolchain:** Rust 1.80+ (Rust 2024 edition).
2. **FFmpeg:** Cần cài đặt `ffmpeg` và có trong biến môi trường `PATH` để trích xuất âm thanh từ MP4/MKV.

---

## 💡 Hướng Dẫn Sử Dụng

### 1. Xử lý một tập đơn lẻ:

```rust
use std::path::Path;
use vox::{process_diarization, ProgressInfo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let video_path = Path::new("input/第001集.mp4");
    let subtitle_path = Path::new("input/第001集.zh.vtt");
    let output_dir = Path::new("output");

    let output_file = process_diarization(
        video_path,
        subtitle_path,
        output_dir,
        None, // Cấu hình mặc định
        Some(|p: &ProgressInfo| {
            println!("[{}] {:>5.1}% | {} | {}", p.stage.label(), p.percentage, p.duration, p.status);
        }),
    )?;

    println!("Phụ đề đã xuất ra: {}", output_file.display());
    Ok(())
}
```

---

### 2. Xử lý cả thư mục phim gồm nhiều tập (Series):

Tự động nhận diện nhân vật cũ qua các tập và phát hiện nhân vật mới xuất hiện:

```rust
use std::path::Path;
use vox::{process_series, ProgressInfo, SpeakerRegistry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = Path::new("input");
    let output_dir = Path::new("output");

    // Xử lý toàn bộ các tập trong thư mục:
    let output_files = process_series(
        input_dir,
        output_dir,
        None,
        Some(|p: &ProgressInfo| {
            println!("[{}] {:>5.1}% | {}", p.stage.label(), p.percentage, p.status);
        }),
    )?;

    println!("Đã xử lý xong {} tập!", output_files.len());

    // Đọc sổ bộ nhân vật đã xuất:
    let registry = SpeakerRegistry::load_from_file(output_dir.join("speaker_registry.json"))?;
    for (id, profile) in &registry.speakers {
        println!("Nhân vật {}: Tên {:?} | Xuất hiện trong {:?}", id, profile.display_name, profile.episode_appearances);
    }

    Ok(())
}
```

---

## 🖥️ Tích Hợp Tiến Trình với `cli_gui_form` (TUI / CLI)

`ProgressInfo` cung cấp hàm `to_cli_gui_params()` trả về chính xác bộ tham số của `TaskWidget`:

```rust
use cli_gui_form::{TaskWidget, ProgressStyle, Theme};
use vox::ProgressInfo;

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

let on_progress = |p: &ProgressInfo| {
    let (current, total, unit, duration, status) = p.to_cli_gui_params();
    task_widget.update_progress(current, duration);
};
```

---

## ⚡ Chạy Ví Dụ Kiểm Thử (Examples)

### Ví dụ 1: Xử lý 1 tập đơn lẻ:
```powershell
cargo run --release --example diarize_demo
```

### Ví dụ 2: Xử lý trọn bộ series nhiều tập:
```powershell
cargo run --release --example series_demo
```

---

## 📊 Benchmark Hiệu Năng Thực Tế (CPU-Only)

| Tập phim | Thời lượng Video | Thời gian xử lý (CPU) | Tốc độ tương đối (RTF) | Số nhân vật phát hiện |
|---|---|---|---|---|
| **Tập 001** | 143.7 giây (~2.4 phút) | **1.57 giây** | **~91.5x realtime** | 2 nhân vật (`SPEAKER_00`, `SPEAKER_01`) |
| **Tập 002** | 218.5 giây (~3.6 phút) | **2.21 giây** | **~98.8x realtime** | Nhận diện đúng nhân vật cũ + cấp mã nhân vật mới |
| **Cả 2 tập (Series)** | **362.2 giây (~6 phút)** | **~3.8 giây** | **~95x realtime** | Sổ bộ `speaker_registry.json` hoàn chỉnh |

---

## 📜 Giấy Phép (License)

Dự án được phân phối dưới giấy phép kép:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
