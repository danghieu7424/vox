use std::path::{Path, PathBuf};
use std::time::Instant;
use vox::{process_series, ProgressInfo, SpeakerFormat, SpeakerRegistry, VoxConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═════════════════════════════════════════════════════════════════════");
    println!("        VOX ─ MULTI-EPISODE SERIES SPEAKER DIARIZATION             ");
    println!("      (Cross-Episode Consistency & Dynamic Character Detection)     ");
    println!("═════════════════════════════════════════════════════════════════════\n");

    let input_dir = PathBuf::from("input");
    let output_dir = PathBuf::from("output");

    println!("📁 [Thư mục làm việc]");
    println!("   • Input:  {}", input_dir.display());
    println!("   • Output: {}\n", output_dir.display());

    let model = Path::new("models/campplus.onnx");
    println!("🧠 [Mô hình ONNX]");
    println!("   • Đường dẫn: {}", model.display());
    if model.exists() {
        println!("   • Trạng thái: ✅ Đã tồn tại sẵn trên đĩa.");
    } else {
        println!("   • Trạng thái: ⏳ Chưa có trên đĩa. Sẽ tự động tải từ HuggingFace...");
    }
    println!();

    let mut config = VoxConfig::default();
    config.model_path = model.to_path_buf();
    config.output_dir = output_dir.clone();
    config.speaker_format = SpeakerFormat::NamedIfAvailable;
    config.cpu_threads = 4;

    let total_start = Instant::now();
    let mut last_stage = None;

    let progress_callback = |p: &ProgressInfo| {
        let (current, total, unit, duration, status) = p.to_cli_gui_params();

        if last_stage != Some(p.stage) {
            println!("\n▶ [{}]", p.stage.label());
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

    println!("⚡ Bắt đầu quét thư mục và xử lý trọn bộ series đa tập...");
    let output_files = process_series(
        &input_dir,
        &output_dir,
        Some(config),
        Some(progress_callback),
    )?;

    println!(
        "\n\n🎉 Toàn bộ {} tập phim đã được xử lý xong trong {:.2}s!",
        output_files.len(),
        total_start.elapsed().as_secs_f32()
    );
    for f in &output_files {
        println!("   ✅ {}", f.display());
    }

    // Đọc và hiển thị Sổ bộ nhân vật đã tích luỹ xuyên suốt các tập
    let registry_file = output_dir.join("speaker_registry.json");
    if registry_file.exists() {
        println!("\n👥 [SỔ BỘ NHÂN VẬT TOÀN DIỆN CỦA BỘ PHIM (speaker_registry.json)]");
        println!("─────────────────────────────────────────────────────────────────────");
        let registry = SpeakerRegistry::load_from_file(&registry_file)?;
        for (id, profile) in &registry.speakers {
            let name_display = profile.display_name.as_deref().unwrap_or("Chưa rõ tên");
            println!(
                " • {:<12}: [{}] | Thời lượng nói: {:>5.1}s | F0: {:>5.1}Hz | Các tập: {:?}",
                id, name_display, profile.total_duration_sec, profile.f0_pitch, profile.episode_appearances
            );
        }
        println!("─────────────────────────────────────────────────────────────────────");
    }

    // Trích xuất thử vài dòng của tập 1 và tập 2 để kiểm tra tên nhân vật và chú thích
    for ep_name in ["第001集.zh.diarized.vtt", "第002集.zh.diarized.vtt"] {
        let ep_out = output_dir.join(ep_name);
        if ep_out.exists() {
            println!("\n🔎 [Trích xuất mẫu từ {}]:", ep_name);
            println!("─────────────────────────────────────────────────────────────────────");
            let content = std::fs::read_to_string(&ep_out)?;
            for line in content.lines().take(25) {
                println!("{}", line);
            }
            println!("─────────────────────────────────────────────────────────────────────");
        }
    }

    Ok(())
}
