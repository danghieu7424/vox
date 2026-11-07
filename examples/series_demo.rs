use std::path::PathBuf;
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

    let mut config = VoxConfig::default();
    config.output_dir = output_dir.clone();
    config.speaker_format = SpeakerFormat::Prefix;
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

    println!("\n\n🎉 Toàn bộ {} tập phim đã được xử lý xong trong {:.2}s!", output_files.len(), total_start.elapsed().as_secs_f32());
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
                " • {}: [{}] | Thời lượng nói: {:.1}s | Xuất hiện trong các tập: {:?}",
                id, name_display, profile.total_duration_sec, profile.episode_appearances
            );
        }
        println!("─────────────────────────────────────────────────────────────────────");
    }

    // Trích xuất thử vài dòng của tập 2 để xem chú thích và nhân vật mới
    let ep2_output = output_dir.join("第002集.zh.diarized.vtt");
    if ep2_output.exists() {
        println!("\n🔎 [Trích xuất đoạn có nhân vật mới trong Tập 2 (Lu Miaomiao)]:");
        println!("─────────────────────────────────────────────────────────────────────");
        let content = std::fs::read_to_string(&ep2_output)?;
        let lines: Vec<&str> = content.lines().collect();
        for line in lines.iter().skip(35).take(20) {
            println!("{}", line);
        }
        println!("─────────────────────────────────────────────────────────────────────");
    }

    Ok(())
}
