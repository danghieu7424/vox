Hoàn toàn dùng Rust được. Về định dạng model của `pyannote.audio` và khả năng chạy trên CPU:

1. **`pyannote.audio` gốc là PyTorch model (`.bin` / `.safetensors`), KHÔNG PHẢI ONNX hay GGUF gốc.**
* Tuy nhiên, các mô hình con bên trong (trích xuất feature, segmentation, embedding) **hoàn toàn chuyển đổi (export) sang định dạng ONNX được**.
* GGUF chủ yếu thiết kế cho LLM (mô hình ngôn ngữ lớn) và các mô hình Whisper C++, không dùng cho Diarization/Voiceprint pipeline như Pyannote.


2. **Chạy trên CPU:**
* Nếu chạy trên CPU bằng Rust thông qua **ONNX Runtime (ort)**, tốc độ xử lý rất nhanh (thường nhanh gấp 2–5 lần so với chạy PyTorch trên CPU trong Python) vì không bị dính GIL và tận dụng tối đa SIMD/AVX2/AVX-512 của CPU.



---

### [WEB] Kiến trúc triển khai bằng Rust (CPU-Only)

Để chạy Diarization bằng Rust thuần trên CPU, giải pháp tối ưu là convert các sub-models của Pyannote (hoặc các model tương đương) sang **ONNX** và dùng crate **`ort` (ONNX Runtime Rust binding)** để inference.

#### 1. Mô hình pipeline chuẩn trên Rust

Pipeline gồm 2 thành phần ONNX chạy trên CPU:

* **Segmentation ONNX Model:** Nhận đầu vào là PCM audio $\rightarrow$ Trích xuất khoảng thời gian (Timestamp) nào đang có người nói.
* **Embedding ONNX Model (như `ECAPA-TDNN` hoặc `ResNet34` từ WeSpeaker/SpeechBrain):** Trích xuất vector đặc trưng (Embedding) cho từng khoảng speaker $\rightarrow$ Gom nhóm (Clustering) bằng Cosine Distance/AHC để gán nhãn `SPEAKER_00`, `SPEAKER_01`.

---

#### 2. Ví dụ Code Rust: Gộp nhãn người nói vào Sub SRT có sẵn

##### `Cargo.toml`

```toml
[dependencies]
ort = "2.0.0-rc.9" # ONNX Runtime Rust Binding
ndarray = "0.15"
subtitles = "0.3"  # Hoặc thư viện parse SRT tương tự
hound = "3.5"      # Đọc file WAV (PCM Float32 16kHz)

```

##### `src/main.rs`

```rust
use ort::{inputs, GraphOptimizationLevel, Session};
use ndarray::Array3;
use std::fs::File;
use std::io::BufReader;

struct SubSegment {
    start_sec: f32,
    end_sec: f32,
    text: String,
    speaker: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Khoản tạo ONNX Runtime Session chạy tối ưu hóa trên CPU
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)? // Số core CPU dành cho inference
        .commit_from_file("pyannote_segmentation.onnx")?;

    // 2. Load audio PCM WAV (16kHz, mono)
    let mut reader = hound::WavReader::open("audio_input.wav")?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // 3. Inference Segmentation ONNX trên CPU để tìm khoảng giọng nói
    // (Đầu vào ONNX: Batch x Channels x Samples -> Shape: [1, 1, num_samples])
    let shape = vec![1, 1, samples.len()];
    let input_array = Array3::from_shape_vec((1, 1, samples.len()), samples)?;
    
    let outputs = session.run(inputs!["input" => input_array]?)?;
    let _output_tensor = outputs["output"].extract_tensor::<f32>()?;

    // Giả định từ Output Tensor ta parse ra các SpkTurn (start, end, speaker_id)
    let diarized_turns = vec![
        (0.0f32, 2.5f32, "SPEAKER_00"),
        (2.6f32, 5.0f32, "SPEAKER_01"),
    ];

    // 4. Read file SRT sẵn có & Match với kết quả Diarization
    let mut subs = vec![
        SubSegment { start_sec: 0.5, end_sec: 2.1, text: "Xin chào mọi người.".to_string(), speaker: None },
        SubSegment { start_sec: 2.8, end_sec: 4.5, text: "Hôm nay chúng ta họp về vấn đề gì?".to_string(), speaker: None },
    ];

    for sub in subs.iter_mut() {
        sub.speaker = Some(assign_speaker_to_sub(sub.start_sec, sub.end_sec, &diarized_turns));
        println!("[{:.1}s - {:.1}s] [{}]: {}", sub.start_sec, sub.end_sec, sub.speaker.as_ref().unwrap(), sub.text);
    }

    Ok(())
}

/// Thuật toán tính độ đè trùng (Overlap) giữa câu Sub và khoảng Speaker đang nói
fn assign_speaker_to_sub(sub_start: f32, sub_end: f32, turns: &[(f32, f32, &str)]) -> String {
    let mut max_overlap = 0.0f32;
    let mut best_speaker = "UNKNOWN";

    for &(turn_start, turn_end, speaker) in turns {
        let overlap_start = sub_start.max(turn_start);
        let overlap_end = sub_end.min(turn_end);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap > max_overlap {
            max_overlap = overlap;
            best_speaker = speaker;
        }
    }

    best_speaker.to_string()
}

```

---

### Tóm tắt giải pháp tối ưu cho nhu cầu của bạn

1. **Không dùng Python nữa:** Dùng **Rust + `ort` (ONNX Runtime CPU)**.
2. **Chuẩn bị Model:** Export Pyannote Segmentation sang `.onnx` bằng Python 1 lần duy nhất (hoặc tải sẵn các bản ONNX chuyển đổi sẵn của Pyannote / Sherpa-onnx / Silero-VAD).
3. **Hiệu năng:** Chạy hoàn toàn trên CPU, ngốn ít RAM, thời gian xử lý file audio 1 tiếng chỉ mất vài chục giây trên CPU hiện đại.