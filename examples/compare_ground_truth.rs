use std::collections::HashMap;
use std::path::Path;
use vox::{load_subtitles, load_wav, extract_voiceprint, cosine_similarity};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ground Truth Audio & Voiceprint Analysis ===");

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        if !Path::new(&final_vtt).exists() || !Path::new(&wav_path).exists() {
            println!("Skipping {}, files not found", ep);
            continue;
        }

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        println!("\n--- {} (Total segments: {}, Audio: {:.1}s) ---", ep, segments.len(), audio.duration_sec);

        // Group audio slices by speaker
        let mut speaker_slices: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut speaker_cue_count: HashMap<String, usize> = HashMap::new();

        for seg in &segments {
            // Extract speaker name from format: [Name]: text or [SPEAKER_xx]: text
            if seg.text.starts_with('[') {
                if let Some(close_idx) = seg.text.find(']') {
                    let spk_name = seg.text[1..close_idx].to_string();
                    *speaker_cue_count.entry(spk_name.clone()).or_insert(0) += 1;

                    let start_sample = (seg.start_sec * audio.sample_rate as f32) as usize;
                    let end_sample = (seg.end_sec * audio.sample_rate as f32) as usize;
                    if start_sample < audio.samples.len() {
                        let actual_end = end_sample.min(audio.samples.len());
                        if actual_end > start_sample {
                            let slice = audio.samples[start_sample..actual_end].to_vec();
                            speaker_slices.entry(spk_name).or_default().push(slice);
                        }
                    }
                }
            }
        }

        println!("Speakers found in ground truth:");
        let mut speaker_voiceprints: HashMap<String, Vec<f32>> = HashMap::new();

        for (spk, count) in &speaker_cue_count {
            let slices = &speaker_slices[spk];
            let all_samples: Vec<f32> = slices.iter().flat_map(|s| s.iter().copied()).collect();
            let vp = extract_voiceprint(&all_samples, audio.sample_rate);
            let dur = all_samples.len() as f32 / audio.sample_rate as f32;
            println!("  • {:<12}: {:>2} cues, {:>5.1}s total audio", spk, count, dur);
            speaker_voiceprints.insert(spk.clone(), vp);
        }

        // Print similarity matrix between speakers
        let spk_list: Vec<String> = speaker_voiceprints.keys().cloned().collect();
        println!("\nCosine similarity matrix between speakers in {}:", ep);
        print!("{:<12}", "");
        for spk2 in &spk_list {
            print!("{:>10}", &spk2[..spk2.chars().take(4).map(|c| c.len_utf8()).sum()]);
        }
        println!();

        for spk1 in &spk_list {
            let name_short: String = spk1.chars().take(4).collect();
            print!("{:<12}", name_short);
            for spk2 in &spk_list {
                let sim = cosine_similarity(&speaker_voiceprints[spk1], &speaker_voiceprints[spk2]);
                print!("{:>10.2}", sim);
            }
            println!();
        }
    }

    Ok(())
}
