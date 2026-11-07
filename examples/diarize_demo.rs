use std::path::{Path, PathBuf};
use std::time::Instant;
use vox::{
    process_diarization, ProgressInfo, SpeakerFormat, VoxConfig,
    DEFAULT_MODEL_URL,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═════════════════════════════════════════════════════════════════════");
    println!("               VOX ─ SPEAKER DIARIZATION & SUBTITLES                ");
    println!("           (Accelerated CPU-only ONNX Runtime Architecture)         ");
    println!("═════════════════════════════════════════════════════════════════════\n");

    // 1. Chỉ định đường dẫn đầu vào (2 file có thể ở 2 thư mục khác nhau)
    let video_path = PathBuf::from("input/第001集.mp4");
    let subtitle_path = PathBuf::from("input/第001集.zh.vtt");
    let output_dir = PathBuf::from("output");

    println!("📁 [Đầu vào]");
    println!("   • Video:   {}", video_path.display());
    println!("   • Phụ đề:  {}", subtitle_path.display());
    println!("   • Thư mục xuất: {}\n", output_dir.display());

    // 2. Chỉ định đường dẫn lưu và sử dụng model theo đúng yêu cầu:
    //    let model = Path::new("models/..."); model.exists()
    let model = Path::new("models/pyannote_segmentation.onnx");
    println!("🧠 [Mô hình ONNX]");
    println!("   • Đường dẫn: {}", model.display());
    if model.exists() {
        println!("   • Trạng thái: ✅ Đã tồn tại sẵn trên đĩa.");
    } else {
        println!("   • Trạng thái: ⏳ Chưa có trên đĩa. Sẽ tự động tải từ HuggingFace...");
    }
    println!();

    // 3. Cấu hình Engine
    let mut config = VoxConfig::default();
    config.model_path = model.to_path_buf();
    config.model_url = DEFAULT_MODEL_URL.to_string();
    config.output_dir = output_dir.clone();
    config.speaker_format = SpeakerFormat::Prefix; // [SPEAKER_00]: Text
    config.cpu_threads = 4; // Tối ưu CPU 4 threads inference
    config.onset_threshold = 0.5;
    config.offset_threshold = 0.5;

    // 4. Hook tiến trình theo chuẩn TaskWidget của `cli_gui_form`
    // Tham khảo TaskWidget::new_progress(id, label, current, total, unit, duration, status, color)
    let mut last_stage = None;
    let total_start = Instant::now();

    let progress_callback = |p: &ProgressInfo| {
        // Lấy bộ 5 tham số chuẩn TaskWidget của cli_gui_form:
        // (current, total, unit, duration, status)
        let (current, total, unit, duration, status) = p.to_cli_gui_params();

        if last_stage != Some(p.stage) {
            println!("\n▶ Giai đoạn: [{}]", p.stage.label());
            last_stage = Some(p.stage);
        }

        // Vẽ thanh tiến trình phong cách Parallelogram/Line của cli_gui_form
        let bar_width: usize = 30;
        let filled_chars = ((p.percentage / 100.0) * bar_width as f32).round() as usize;
        let empty_chars = bar_width.saturating_sub(filled_chars);
        let filled_bar = "▰".repeat(filled_chars);
        let empty_bar = "▱".repeat(empty_chars);

        // Hiển thị tiến trình trên 1 dòng chuẩn CLI / TUI Bridge
        print!(
            "\r  [{}{}] {:>5.1}% | {}/{} {} | {:>6} | {}",
            filled_bar, empty_bar, p.percentage, current, total, unit, duration, status
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    };

    println!("⚡ Bắt đầu tiến trình xử lý Diarization...");
    let output_file = process_diarization(
        &video_path,
        &subtitle_path,
        &output_dir,
        Some(config),
        Some(progress_callback),
    )?;

    println!("\n\n🎉 Xử lý hoàn tất trong {:.2}s!", total_start.elapsed().as_secs_f32());
    println!("📄 File kết quả đã xuất tại: {}\n", output_file.display());

    // Hiển thị 5 dòng đầu tiên của file kết quả để nghiệm thu
    if output_file.exists() {
        println!("🔎 [Trích xuất 20 dòng đầu tiên của kết quả:]");
        println!("─────────────────────────────────────────────────────────────────────");
        let content = std::fs::read_to_string(&output_file)?;
        for line in content.lines().take(20) {
            println!("{}", line);
        }
        println!("─────────────────────────────────────────────────────────────────────");
    }

    Ok(())
}
