use std::collections::HashMap;
use vox::{load_subtitles, load_wav, extract_character_name_from_annotation, is_annotation_text};

#[derive(Debug, Clone)]
struct SubCue {
    index: usize,
    start_sec: f32,
    end_sec: f32,
    text: String,
    ground_truth: String,
    is_annotation: bool,
    f0: f32,
    zcr: f32,
    mfcc: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SpeakerProfileTest {
    name: String,
    f0: f32,
    zcr: f32,
    mfcc: Vec<f32>,
    count: usize,
}

impl SpeakerProfileTest {
    fn update(&mut self, f0: f32, zcr: f32, mfcc: &[f32]) {
        let n = self.count as f32;
        if f0 > 60.0 {
            self.f0 = (self.f0 * n + f0) / (n + 1.0);
        }
        self.zcr = (self.zcr * n + zcr) / (n + 1.0);
        for i in 0..self.mfcc.len() {
            self.mfcc[i] = (self.mfcc[i] * n + mfcc[i]) / (n + 1.0);
        }
        self.count += 1;
    }

    fn similarity(&self, f0: f32, zcr: f32, mfcc: &[f32]) -> f32 {
        // 1. MFCC cosine similarity
        let mut dot = 0.0f32;
        for i in 1..self.mfcc.len().min(mfcc.len()) {
            dot += self.mfcc[i] * mfcc[i];
        }
        let mfcc_sim = dot.clamp(-1.0, 1.0);

        // 2. Pitch penalty: if both have valid pitch, penalize large difference
        let pitch_score = if self.f0 > 60.0 && f0 > 60.0 {
            let ratio = (self.f0 - f0).abs() / self.f0.max(f0);
            (1.0 - ratio * 2.0).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 3. ZCR score
        let zcr_diff = (self.zcr - zcr).abs();
        let zcr_score = (1.0 - zcr_diff * 4.0).clamp(0.0, 1.0);

        0.5 * mfcc_sim + 0.4 * pitch_score + 0.1 * zcr_score
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Full Diarization Accuracy on Ep 1 & Ep 2 ===");

    let mut global_registry: HashMap<String, SpeakerProfileTest> = HashMap::new();

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        let mut cues = Vec::new();

        for seg in &segments {
            let is_ann = is_annotation_text(&seg.text);
            let mut gt = "UNKNOWN".to_string();
            let mut clean_text = seg.text.clone();

            if is_ann {
                cues.push(SubCue {
                    index: seg.index,
                    start_sec: seg.start_sec,
                    end_sec: seg.end_sec,
                    text: seg.text.clone(),
                    ground_truth: "ANNOTATION".to_string(),
                    is_annotation: true,
                    f0: 0.0,
                    zcr: 0.0,
                    mfcc: vec![0.0; 13],
                });
                continue;
            }

            if seg.text.starts_with('[') {
                if let Some(close_idx) = seg.text.find(']') {
                    gt = seg.text[1..close_idx].to_string();
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

            let (f0, zcr, mfcc) = extract_features(slice, audio.sample_rate);

            cues.push(SubCue {
                index: seg.index,
                start_sec: seg.start_sec,
                end_sec: seg.end_sec,
                text: clean_text,
                ground_truth: gt,
                is_annotation: false,
                f0,
                zcr,
                mfcc,
            });
        }

        // STEP 1: Scan for anchor screen tags and register anchor speakers
        let mut anchor_positions: Vec<(usize, String)> = Vec::new();
        for (i, c) in cues.iter().enumerate() {
            if c.is_annotation {
                if let Some(raw_name) = extract_character_name_from_annotation(&c.text) {
                    let pure_name = if let Some(dash_idx) = raw_name.rfind('-') {
                        raw_name[dash_idx + 1..].trim().to_string()
                    } else {
                        raw_name.trim().to_string()
                    };
                    if !pure_name.ends_with("家") && !pure_name.contains("...") && pure_name.chars().count() <= 6 {
                        anchor_positions.push((i, pure_name));
                    }
                }
            }
        }

        println!("\n--- Ep: {} ---", ep);
        println!("Anchors detected:");
        for (pos, name) in &anchor_positions {
            println!("  • At cue #{}, tag: {}", cues[*pos].index, name);
        }

        // Initialize anchor profiles from closest cue
        for (pos, name) in &anchor_positions {
            // Find best cue in range pos..pos+4
            for offset in 1..=4 {
                if pos + offset < cues.len() {
                    let next_cue = &cues[pos + offset];
                    if !next_cue.is_annotation && next_cue.f0 > 60.0 {
                        // Check if fits general profile (system is >380Hz, etc.)
                        if name == "系统" && next_cue.f0 < 350.0 {
                            continue; // skip human intro before system
                        }
                        if name != "系统" && next_cue.f0 > 380.0 && name != "陆淼淼" {
                            continue;
                        }
                        global_registry.entry(name.clone()).or_insert_with(|| {
                            SpeakerProfileTest {
                                name: name.clone(),
                                f0: next_cue.f0,
                                zcr: next_cue.zcr,
                                mfcc: next_cue.mfcc.clone(),
                                count: 1,
                            }
                        });
                        break;
                    }
                }
            }
        }

        // STEP 2: Assign speakers to all dialogue cues
        let mut predicted_speakers: Vec<String> = Vec::with_capacity(cues.len());
        let mut correct = 0;
        let mut total_dialogue = 0;

        let mut prev_spk = "UNKNOWN".to_string();
        let mut prev_end = 0.0f32;

        for (i, cue) in cues.iter().enumerate() {
            if cue.is_annotation {
                predicted_speakers.push("ANNOTATION".to_string());
                continue;
            }

            total_dialogue += 1;

            // Check if there is an anchor right before this cue
            let mut immediate_anchor = None;
            for (pos, name) in &anchor_positions {
                if *pos + 1 == i || (*pos + 2 == i && cues[*pos + 1].is_annotation) {
                    immediate_anchor = Some(name.clone());
                    break;
                }
            }

            let spk = if let Some(anchor) = immediate_anchor {
                // If it's 系统, check pitch
                if anchor == "系统" && cue.f0 < 350.0 && cue.f0 > 60.0 {
                    // System announcement was about to start, but previous person spoke
                    "陆清安".to_string()
                } else {
                    anchor
                }
            } else {
                // Match against global registry
                let mut best_spk = "SPEAKER_00".to_string();
                let mut best_score = -100.0f32;

                for (name, prof) in &global_registry {
                    let mut score = prof.similarity(cue.f0, cue.zcr, &cue.mfcc);

                    // Continuity bonus: if same speaker within 0.8s
                    if name == &prev_spk && (cue.start_sec - prev_end).abs() < 0.8 {
                        score += 0.25;
                    }

                    // Special heuristic for "系统": system has very high pitch > 380Hz and robotic phrasing
                    if name == "系统" {
                        if cue.f0 > 370.0 {
                            score += 0.4;
                        } else if cue.f0 > 60.0 && cue.f0 < 300.0 {
                            score -= 0.8;
                        }
                    }

                    // Special heuristic for male vs female:
                    // 陆清安 is deep male (~90-120Hz). If cue.f0 > 200Hz, heavily penalize!
                    if (name == "陆清安" || name == "陆正然") && cue.f0 > 180.0 {
                        score -= 0.7;
                    }
                    if (name == "徐清莹" || name == "林清雪" || name == "陆紫萱") && cue.f0 > 60.0 && cue.f0 < 160.0 {
                        score -= 0.7;
                    }

                    if score > best_score {
                        best_score = score;
                        best_spk = name.clone();
                    }
                }

                best_spk
            };

            // Update profile with this cue's audio
            if let Some(prof) = global_registry.get_mut(&spk) {
                prof.update(cue.f0, cue.zcr, &cue.mfcc);
            }

            prev_spk = spk.clone();
            prev_end = cue.end_sec;

            let is_match = spk == cue.ground_truth;
            if is_match {
                correct += 1;
            }

            predicted_speakers.push(spk);
        }

        let acc = (correct as f32 / total_dialogue as f32) * 100.0;
        println!("===> {} Result: {}/{} correct ({:.1}% accuracy)!", ep, correct, total_dialogue, acc);

        // Print first 15 predictions vs ground truth
        println!("\nSample predictions vs Ground Truth (first 15 dialogue cues):");
        let mut shown = 0;
        for (i, c) in cues.iter().enumerate() {
            if c.is_annotation { continue; }
            shown += 1;
            if shown > 15 { break; }
            let pred = &predicted_speakers[i];
            let mark = if pred == &c.ground_truth { "✓" } else { "✗" };
            println!("  [{}] [{:.1}s] Pred: {:<8} | GT: {:<8} | F0: {:>5.1}Hz | Text: {}",
                mark, c.start_sec, pred, c.ground_truth, c.f0, &c.text[..c.text.chars().take(15).map(|ch| ch.len_utf8()).sum()]);
        }
    }

    Ok(())
}

fn extract_features(samples: &[f32], sample_rate: u32) -> (f32, f32, Vec<f32>) {
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

    // 2. Autocorrelation F0
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
    let frame_size = 512;
    let hop_size = 256;
    let num_mel_bins = 24;
    let num_mfcc = 13;

    let num_frames = (samples.len().saturating_sub(frame_size)) / hop_size + 1;
    if num_frames == 0 {
        return (f0, zcr, vec![0.0; num_mfcc]);
    }

    let mut mel_energies = vec![0.0f32; num_mel_bins];
    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];

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

    for bin in 0..num_mel_bins {
        mel_energies[bin] /= num_frames as f32;
    }

    let mut mfcc = vec![0.0f32; num_mfcc];
    for i in 0..num_mfcc {
        let mut sum = 0.0f32;
        for m in 0..num_mel_bins {
            let angle = std::f32::consts::PI * (i as f32) * (m as f32 + 0.5) / (num_mel_bins as f32);
            sum += mel_energies[m] * angle.cos();
        }
        mfcc[i] = sum;
    }

    let mut norm_sq = 0.0f32;
    for &c in &mfcc[1..] {
        norm_sq += c * c;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    for c in mfcc[1..].iter_mut() {
        *c /= norm;
    }

    (f0, zcr, mfcc)
}
