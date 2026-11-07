use std::path::{Path, PathBuf};
use std::time::Instant;
use vox::{
    process_diarization, ProgressInfo, SpeakerFormat, VoxConfig,
    DEFAULT_EMBEDDING_MODEL_URL,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═════════════════════════════════════════════════════════════════════");
    println!("               VOX ─ SPEAKER DIARIZATION & SUBTITLES                ");
    println!("           (Accelerated CPU-only ONNX Runtime Architecture)         ");
    println!("═════════════════════════════════════════════════════════════════════\n");

    // 1. Chỉ định đường dẫn đầu vào (2 file có thể ở 2 thư mục khác nhau)
    // Tự động ưu tiên file vocal-separated .vox.wav nếu có trong cùng thư mục
    let video_path = PathBuf::from("input/第001集.mp4");
    let subtitle_path = PathBuf::from("input/第001集.zh.vtt");
    let output_dir = PathBuf::from("output");

    println!("📁 [Đầu vào]");
    println!("   • Video/Audio: {}", video_path.display());
    println!("   • Phụ đề:      {}", subtitle_path.display());
    println!("   • Thư mục xuất: {}\n", output_dir.display());

    // 2. Chỉ định đường dẫn lưu và sử dụng model theo đúng yêu cầu:
    //    let model = Path::new("models/..."); model.exists()
    let model = Path::new("models/campplus.onnx");
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
    config.model_url = DEFAULT_EMBEDDING_MODEL_URL.to_string();
    config.output_dir = output_dir.clone();
    config.speaker_format = SpeakerFormat::NamedIfAvailable; // [陆清安]: Text hoặc [SPEAKER_00]: Text nếu chưa rõ tên
    config.cpu_threads = 4; // Tối ưu CPU 4 threads inference

    // 4. Hook tiến trình theo chuẩn TaskWidget của `cli_gui_form`
    // Tham khảo TaskWidget::new_progress(id, label, current, total, unit, duration, status, color)
    let mut last_stage = None;
    let total_start = Instant::now();

    let progress_callback = |p: &ProgressInfo| {
        let (current, total, unit, duration, status) = p.to_cli_gui_params();

        if last_stage != Some(p.stage) {
            println!("\n▶ Giai đoạn: [{}]", p.stage.label());
            last_stage = Some(p.stage);
        }

        let bar_width: usize = 30;
        let filled_chars = ((p.percentage / 100.0) * bar_width as f32).round() as usize;
        let empty_chars = bar_width.saturating_sub(filled_chars);
        let filled_bar = "▰".repeat(filled_chars);
        let empty_bar = "▱".repeat(empty_chars);

        print!(
            "\r  [{}{}] {:>5.1}% | {}/{} {} | {:>6} | {}",
            filled_bar, empty_bar, p.percentage, current, total, unit, duration, status
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    };

    // 5. Tiến hành chạy Diarization
    println!("⚡ Bắt đầu tiến trình xử lý Diarization...");
    let result_path = process_diarization(
        &video_path,
        &subtitle_path,
        &output_dir,
        Some(config),
        Some(progress_callback),
    )?;

    println!(
        "\n\n🎉 Xử lý hoàn tất trong {:.2}s!",
        total_start.elapsed().as_secs_f32()
    );
    println!("📄 File kết quả đã xuất tại: {}\n", result_path.display());

    // Đọc thử 20 dòng đầu tiên của kết quả để kiểm tra
    if result_path.exists() {
        println!("🔎 [Trích xuất 25 dòng đầu tiên của kết quả:]");
        println!("─────────────────────────────────────────────────────────────────────");
        let content = std::fs::read_to_string(&result_path)?;
        for line in content.lines().take(25) {
            println!("{}", line);
        }
        println!("─────────────────────────────────────────────────────────────────────");
    }

    Ok(())
}
