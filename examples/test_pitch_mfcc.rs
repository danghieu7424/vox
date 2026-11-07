use std::collections::HashMap;
use vox::{load_subtitles, load_wav};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing F0 Pitch & MFCC Analysis on Ep 1 & 2 ===");

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        println!("\n==================== {} ====================", ep);

        // Group segments by speaker
        let mut speaker_cues: HashMap<String, Vec<(f32, f32, String)>> = HashMap::new();

        for seg in &segments {
            if seg.text.starts_with('[') {
                if let Some(close_idx) = seg.text.find(']') {
                    let spk_name = seg.text[1..close_idx].to_string();
                    speaker_cues.entry(spk_name).or_default().push((seg.start_sec, seg.end_sec, seg.text.clone()));
                }
            }
        }

        for (spk, cues) in &speaker_cues {
            let mut f0_list = Vec::new();
            let mut zcr_list = Vec::new();

            for (start, end, _) in cues {
                let start_sample = (*start * audio.sample_rate as f32) as usize;
                let end_sample = (*end * audio.sample_rate as f32) as usize;
                if start_sample >= audio.samples.len() { continue; }
                let actual_end = end_sample.min(audio.samples.len());
                let slice = &audio.samples[start_sample..actual_end];

                // Compute F0 and Zero Crossing Rate for this cue
                let (f0, zcr) = analyze_audio_slice(slice, audio.sample_rate);
                if f0 > 60.0 && f0 < 600.0 {
                    f0_list.push(f0);
                }
                zcr_list.push(zcr);
            }

            if !f0_list.is_empty() {
                f0_list.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_f0 = f0_list[f0_list.len() / 2];
                let mean_zcr = zcr_list.iter().sum::<f32>() / zcr_list.len() as f32;
                println!(
                    "Speaker: {:<12} | Cues: {:>2} | Median F0: {:>5.1} Hz | Min-Max: {:>5.1} - {:>5.1} Hz | ZCR: {:.3}",
                    spk, cues.len(), median_f0, f0_list[0], f0_list[f0_list.len() - 1], mean_zcr
                );
            } else {
                println!("Speaker: {:<12} | Cues: {:>2} | No voiced frames detected", spk, cues.len());
            }
        }
    }

    Ok(())
}

fn analyze_audio_slice(samples: &[f32], sample_rate: u32) -> (f32, f32) {
    if samples.len() < 512 {
        return (0.0, 0.0);
    }

    // Zero Crossing Rate
    let mut zcr_count = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0 && samples[i - 1] < 0.0) || (samples[i] < 0.0 && samples[i - 1] >= 0.0) {
            zcr_count += 1;
        }
    }
    let zcr = zcr_count as f32 / samples.len() as f32;

    // Autocorrelation Pitch Estimation on windowed center
    let frame_size = 1024.min(samples.len());
    let center = samples.len() / 2;
    let start = center.saturating_sub(frame_size / 2);
    let frame = &samples[start..(start + frame_size).min(samples.len())];

    let min_lag = (sample_rate / 500) as usize; // ~32 samples (500Hz)
    let max_lag = (sample_rate / 70) as usize;  // ~228 samples (70Hz)

    let mut best_corr = 0.0f32;
    let mut best_lag = 0;
    let energy: f32 = frame.iter().map(|x| x * x).sum();

    if energy < 1e-4 {
        return (0.0, zcr);
    }

    for lag in min_lag..max_lag.min(frame.len() / 2) {
        let mut corr = 0.0f32;
        let mut e1 = 0.0f32;
        let mut e2 = 0.0f32;
        for n in 0..(frame.len() - lag) {
            corr += frame[n] * frame[n + lag];
            e1 += frame[n] * frame[n];
            e2 += frame[n + lag] * frame[n + lag];
        }
        let norm = (e1 * e2).sqrt().max(1e-12);
        let ncorr = corr / norm;
        if ncorr > best_corr && ncorr > 0.4 {
            best_corr = ncorr;
            best_lag = lag;
        }
    }

    let f0 = if best_lag > 0 {
        sample_rate as f32 / best_lag as f32
    } else {
        0.0
    };

    (f0, zcr)
}
