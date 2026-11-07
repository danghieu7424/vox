use std::collections::HashMap;
use vox::{load_subtitles, load_wav, extract_character_name_from_annotation, is_annotation_text};

#[derive(Debug, Clone)]
struct CueFeature {
    index: usize,
    start_sec: f32,
    end_sec: f32,
    text: String,
    ground_truth_speaker: String,
    f0: f32,
    zcr: f32,
    mfcc: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Cue-Level Diarization Against Ground Truth ===");

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        let mut cue_features = Vec::new();
        let mut anchor_names: HashMap<usize, String> = HashMap::new();

        // 1. Extract ground truth speaker and features for each cue
        for (i, seg) in segments.iter().enumerate() {
            let mut spk = "UNKNOWN".to_string();
            let mut clean_text = seg.text.clone();

            if is_annotation_text(&seg.text) {
                if let Some(name) = extract_character_name_from_annotation(&seg.text) {
                    // Normalize name (e.g., "大女儿-陆紫萱" -> "陆紫萱")
                    let pure_name = if let Some(dash_idx) = name.rfind('-') {
                        name[dash_idx + 1..].to_string()
                    } else {
                        name
                    };
                    anchor_names.insert(i, pure_name);
                }
                continue;
            }

            if seg.text.starts_with('[') {
                if let Some(close_idx) = seg.text.find(']') {
                    spk = seg.text[1..close_idx].to_string();
                    clean_text = seg.text[close_idx + 1..].trim_start_matches(':').trim().to_string();
                }
            }

            let start_sample = (seg.start_sec * audio.sample_rate as f32) as usize;
            let end_sample = (seg.end_sec * audio.sample_rate as f32) as usize;
            let actual_end = end_sample.min(audio.samples.len());
            let slice = if start_sample < audio.samples.len() && actual_end > start_sample {
                &audio.samples[start_sample..actual_end]
            } else {
                &[]
            };

            let (f0, zcr, mfcc) = extract_cue_features(slice, audio.sample_rate);

            cue_features.push(CueFeature {
                index: seg.index,
                start_sec: seg.start_sec,
                end_sec: seg.end_sec,
                text: clean_text,
                ground_truth_speaker: spk,
                f0,
                zcr,
                mfcc,
            });
        }

        println!("\n--- {} (Total dialogue cues: {}) ---", ep, cue_features.len());
        println!("Anchor screen tags found in {}:", ep);
        for (idx, name) in &anchor_names {
            println!("  • Cue #{} (around {:.1}s): 《{}》", segments[*idx].index, segments[*idx].start_sec, name);
        }
    }

    Ok(())
}

fn extract_cue_features(samples: &[f32], sample_rate: u32) -> (f32, f32, Vec<f32>) {
    if samples.len() < 512 {
        return (0.0, 0.0, vec![0.0; 13]);
    }

    // 1. ZCR
    let mut zcr_count = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0 && samples[i - 1] < 0.0) || (samples[i] < 0.0 && samples[i - 1] >= 0.0) {
            zcr_count += 1;
        }
    }
    let zcr = zcr_count as f32 / samples.len() as f32;

    // 2. F0 via Autocorrelation
    let frame_len = 1024.min(samples.len());
    let center = samples.len() / 2;
    let start = center.saturating_sub(frame_len / 2);
    let frame = &samples[start..(start + frame_len).min(samples.len())];

    let min_lag = (sample_rate / 500) as usize;
    let max_lag = (sample_rate / 70) as usize;
    let mut best_corr = 0.0f32;
    let mut best_lag = 0;

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
        if ncorr > best_corr && ncorr > 0.35 {
            best_corr = ncorr;
            best_lag = lag;
        }
    }
    let f0 = if best_lag > 0 { sample_rate as f32 / best_lag as f32 } else { 0.0 };

    // 3. Fast MFCC (13 coefficients)
    let mfcc = compute_mfcc_fast(samples, sample_rate);

    (f0, zcr, mfcc)
}

fn compute_mfcc_fast(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame_size = 512;
    let hop_size = 256;
    let num_mel_bins = 24;
    let num_mfcc = 13;

    let num_frames = (samples.len().saturating_sub(frame_size)) / hop_size + 1;
    if num_frames == 0 {
        return vec![0.0; num_mfcc];
    }

    let mut mel_energies = vec![0.0f32; num_mel_bins];

    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];

        // Approximate energy in frequency bands
        for bin in 0..num_mel_bins {
            let k_low = (bin * 128 / num_mel_bins).max(1);
            let k_high = ((bin + 1) * 128 / num_mel_bins).min(128);

            let mut bin_e = 0.0f32;
            for k in k_low..k_high {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for (n, &s) in frame.iter().enumerate().step_by(4) {
                    let angle = -2.0 * std::f32::consts::PI * (k * n) as f32 / frame_size as f32;
                    re += s * angle.cos();
                    im += s * angle.sin();
                }
                bin_e += (re * re + im * im).sqrt();
            }
            mel_energies[bin] += (bin_e + 1e-6).ln();
        }
    }

    // Average across frames
    for bin in 0..num_mel_bins {
        mel_energies[bin] /= num_frames as f32;
    }

    // DCT-II to get orthogonal MFCCs
    let mut mfcc = vec![0.0f32; num_mfcc];
    for i in 0..num_mfcc {
        let mut sum = 0.0f32;
        for m in 0..num_mel_bins {
            let angle = std::f32::consts::PI * (i as f32) * (m as f32 + 0.5) / (num_mel_bins as f32);
            sum += mel_energies[m] * angle.cos();
        }
        mfcc[i] = sum;
    }

    // Normalize MFCC (c1..c12, ignoring c0 energy offset)
    let mut norm_sq = 0.0f32;
    for &c in &mfcc[1..] {
        norm_sq += c * c;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    for c in mfcc[1..].iter_mut() {
        *c /= norm;
    }

    mfcc
}
