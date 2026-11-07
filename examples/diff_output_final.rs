use vox::load_subtitles;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DIFF BETWEEN OUTPUT AND FINAL ===");

    for ep in ["第001集", "第002集"] {
        let out_vtt = format!("output/{}.zh.diarized.vtt", ep);
        let fin_vtt = format!("final/{}_final.zh.diarized.vtt", ep);

        let (out_subs, _) = load_subtitles(&out_vtt)?;
        let (fin_subs, _) = load_subtitles(&fin_vtt)?;

        println!("\n==================== {} ====================", ep);
        let mut total = 0;
        let mut matched = 0;
        let mut diffs = Vec::new();

        for i in 0..out_subs.len().min(fin_subs.len()) {
            let o = &out_subs[i];
            let f = &fin_subs[i];

            if o.is_annotation && f.is_annotation {
                continue;
            }

            total += 1;
            let o_speaker = extract_speaker(&o.text);
            let f_speaker = extract_speaker(&f.text);

            if o_speaker == f_speaker {
                matched += 1;
            } else {
                diffs.push((o.start_sec, o.end_sec, o_speaker, f_speaker, clean_text(&f.text)));
            }
        }

        println!("Accuracy: {}/{} ({:.1}%)", matched, total, (matched as f32 / total as f32) * 100.0);
        println!("Mismatches ({}):", diffs.len());
        for (start, end, pred, gt, text) in diffs {
            println!("  [{:.1}s - {:.1}s] Pred: {:<12} | Final: {:<12} | Text: {}",
                start, end, pred, gt, &text[..text.chars().take(20).map(|c| c.len_utf8()).sum()]);
        }
    }

    Ok(())
}

fn extract_speaker(text: &str) -> String {
    if text.starts_with('[') {
        if let Some(c) = text.find(']') {
            return text[1..c].to_string();
        }
    }
    "NONE".to_string()
}

fn clean_text(text: &str) -> String {
    if text.starts_with('[') {
        if let Some(c) = text.find(']') {
            return text[c + 1..].trim_start_matches(':').trim().to_string();
        }
    }
    text.to_string()
}
