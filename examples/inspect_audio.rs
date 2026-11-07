use std::collections::HashMap;
use std::path::Path;
use vox::subtitles::load_subtitles;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let final_vtt_path = Path::new("final/第002集.zh.diarized.vtt");
    let (cues, _) = load_subtitles(final_vtt_path)?;

    let mut speaker_counts: HashMap<String, usize> = HashMap::new();

    for (i, c) in cues.iter().enumerate() {
        let text = c.text.trim();
        let speaker = if text.starts_with('[') && text.contains("]: ") {
            let spk_end = text.find("]: ").unwrap();
            &text[1..spk_end]
        } else if text.starts_with('《') {
            "ANNOTATION"
        } else {
            "UNKNOWN"
        };

        *speaker_counts.entry(speaker.to_string()).or_insert(0) += 1;
        if i < 25 {
            println!("[{:>2}] {:>6.2}s - {:>6.2}s | {:<12} | {}", i + 1, c.start_sec, c.end_sec, speaker, text);
        }
    }

    println!("\nSpeaker distribution in Episode 2 (total {} cues):", cues.len());
    for (spk, count) in &speaker_counts {
        println!("  • {:<12}: {} cues", spk, count);
    }

    Ok(())
}
