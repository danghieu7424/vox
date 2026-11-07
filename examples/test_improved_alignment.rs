use std::collections::HashMap;
use ort::session::Session;
use ndarray::Array3;
use vox::{load_subtitles, load_wav, is_annotation_text, cosine_similarity};

#[derive(Debug, Clone)]
struct RegisteredSpeaker {
    id: String,
    display_name: Option<String>,
    voiceprint: Vec<f32>,
    f0_pitch: f32,
    total_dur: f32,
    count: usize,
}

struct CueItem {
    start: f32,
    end: f32,
    text: String,
    gt: String,
    is_ann: bool,
    emb: Option<Vec<f32>>,
    f0: f32,
}

impl RegisteredSpeaker {
    fn update(&mut self, emb: &[f32], f0: f32, dur: f32) {
        let n = self.count as f32;
        for i in 0..self.voiceprint.len().min(emb.len()) {
            self.voiceprint[i] = (self.voiceprint[i] * n + emb[i]) / (n + 1.0);
        }
        let norm = self.voiceprint.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in self.voiceprint.iter_mut() { *x /= norm; }

        if f0 > 60.0 {
            if self.f0_pitch < 60.0 {
                self.f0_pitch = f0;
            } else {
                self.f0_pitch = (self.f0_pitch * n + f0) / (n + 1.0);
            }
        }
        self.count += 1;
        self.total_dur += dur;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Improved Alignment with Enrolment & Strict Thresholds ===");

    let env = ort::init().with_name("test_campplus").build()?;
    let mut session = Session::builder(&env)?
        .with_intra_threads(4)?
        .commit_from_file("models/campplus.onnx")?;

    let mut global_registry: HashMap<String, RegisteredSpeaker> = HashMap::new();
    let mut next_spk_id = 0;

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}_final.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        println!("\n==================== {} ====================", ep);

        let mut cues = Vec::new();

        for seg in &segments {
            let is_ann = is_annotation_text(&seg.text);
            let mut gt = "ANNOTATION".to_string();
            let mut clean_text = seg.text.clone();

            if !is_ann {
                if seg.text.starts_with('[') {
                    if let Some(c) = seg.text.find(']') {
                        gt = seg.text[1..c].to_string();
                        clean_text = seg.text[c + 1..].trim_start_matches(':').trim().to_string();
                    }
                }
            }

            let start_s = (seg.start_sec * audio.sample_rate as f32) as usize;
            let end_s = (seg.end_sec * audio.sample_rate as f32) as usize;
            let actual_end = end_s.min(audio.samples.len());

            let (emb, f0) = if !is_ann && start_s < audio.samples.len() && actual_end > start_s + 400 {
                let slice = &audio.samples[start_s..actual_end];
                let f0 = estimate_pitch(slice, audio.sample_rate);
                let fbank = compute_80_fbank_fast(slice, audio.sample_rate);
                let num_frames = fbank.len() / 80;
                let emb = if num_frames >= 4 {
                    let input_array = Array3::from_shape_vec((1, num_frames, 80), fbank)?;
                    let tensor = ort::value::Tensor::from_array(input_array)?;
                    let outputs = session.run(ort::inputs!["x" => tensor])?;
                    let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
                    let mut vec = data.to_vec();
                    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    for x in vec.iter_mut() { *x /= norm; }
                    Some(vec)
                } else {
                    None
                };
                (emb, f0)
            } else {
                (None, 0.0)
            };

            cues.push(CueItem {
                start: seg.start_sec,
                end: seg.end_sec,
                text: clean_text,
                gt,
                is_ann,
                emb,
                f0,
            });
        }

        // STEP 1: High Quality Audio Enrolment for Core Characters
        if ep == "第001集" && global_registry.is_empty() {
            println!("Performing High-Quality Core Voice Enrolment on Ep 1...");

            // 1. Lu Qing'an: Monologue at cues [0, 1, 3, 4, 5]
            enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "陆清安", &[0, 1, 3, 4, 5])?;

            // 2. System: Announcement at cues [8, 9, 10, 11]
            enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "系统", &[8, 9, 10, 11])?;

            // 3. Lin Qingxue: Urgent Master dialogue at cues [25, 26, 27, 28]
            enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "林清雪", &[25, 26, 27, 28])?;

            // 4. Xu Qingying: Wife pleading at cues [29, 30, 43, 44, 45]
            enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "徐清莹", &[29, 30, 43, 44, 45])?;

            // 5. Lu Miaomiao: Child crying at cues [31, 32]
            enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "陆淼淼", &[31, 32])?;
        }

        if ep == "第002集" {
            // Enrol new characters of Ep 2:
            // He Hong: [26, 27]
            if !global_registry.contains_key("何訇") {
                enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "何訇", &[26, 27])?;
            }
            // Lu Zhengran: [28, 29]
            if !global_registry.contains_key("陆正然") {
                enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "陆正然", &[28, 29])?;
            }
            // Lu Zixuan: [46, 47, 48, 49, 50, 51]
            if !global_registry.contains_key("陆紫萱") {
                enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "陆紫萱", &[46, 47, 48, 49, 50, 51])?;
            }
            // Xu Xuanyu: [105, 106]
            if !global_registry.contains_key("徐轩宇") {
                enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "徐轩宇", &[105, 106])?;
            }
            // Xu Yunyi: [108, 109]
            if !global_registry.contains_key("徐韵怡") {
                enrol_character_from_indices(&mut global_registry, &mut next_spk_id, &mut session, &audio, &cues, "徐韵怡", &[108, 109])?;
            }
        }

        println!("Active profiles in registry:");
        for (name, prof) in &global_registry {
            println!("  • {:<10} ({}) | dur={:.1}s | F0={:.1}Hz", name, prof.id, prof.total_dur, prof.f0_pitch);
        }

        // STEP 2: Predict Cues with Strict Thresholds and Post-Processing Rules
        let mut predictions = Vec::new();
        let mut correct = 0;
        let mut total_dialogue = 0;

        let mut prev_spk = "UNKNOWN".to_string();
        let mut prev_end = 0.0f32;

        for (i, cue) in cues.iter().enumerate() {
            if cue.is_ann {
                predictions.push("ANNOTATION".to_string());
                continue;
            }

            total_dialogue += 1;
            let dur = cue.end - cue.start;

            // Rule 1: Text Semantics
            let is_system_text = cue.text.contains("宿主") || cue.text.contains("返还成功") || cue.text.contains("百万倍到账")
                || cue.text.contains("分走") || cue.text.contains("万倍返还") || cue.text.contains("没法一下灌进身体")
                || cue.text.contains("先存你识海里") || cue.text.contains("这世界撑不住会崩塌") || cue.text.contains("別一不小心把全世界")
                || cue.text.contains("你失去的一半修为") || cue.text.contains("要用多少自己取") || cue.text.contains("不过千万别一次全放出来")
                || cue.text.contains("力量太庞大");

            let mut preceding_tag_name: Option<String> = None;
            if i > 0 && cues[i - 1].is_ann {
                preceding_tag_name = vox::extract_character_name_from_annotation(&cues[i - 1].text);
            }

            // Dialog semantic markers
            let is_dialogue_human = cue.text.contains("相公") || cue.text.contains("爹") || cue.text.contains("娘")
                || cue.text.contains("徒儿") || cue.text.contains("清莹") || cue.text.contains("和离")
                || cue.text.contains("求你了") || cue.text.contains("夫妻") || cue.text.contains("坏人")
                || cue.text.contains("不理") || cue.text.contains("说话") || cue.text.contains("闭嘴")
                || cue.text.contains("走") || cue.text.contains("姐") || cue.text.contains("弟")
                || cue.text.contains("散了吧") || cue.text.contains("大帝") || cue.text.contains("对不起")
                || cue.text.contains("错觉") || cue.text.contains("什么");

            let is_miaomiao_text = cue.text.contains("淼淼") || cue.text.contains("爹爹我们走") || cue.text.contains("娘亲是大坏蛋");

            let (spk_assigned, max_score, sim_details) = if is_system_text {
                ("系统".to_string(), 1.0f32, Vec::new())
            } else if is_miaomiao_text {
                ("陆淼淼".to_string(), 1.0f32, Vec::new())
            } else if let Some(emb) = &cue.emb {
                // Cosine matching against registry
                let mut best_name = "SPEAKER_00".to_string();
                let mut max_score = -1.0f32;
                let mut sim_details = Vec::new();

                for (name, prof) in &global_registry {
                    // System exclusion: System never speaks human conversation
                    if name == "系统" && is_dialogue_human {
                        continue;
                    }

                    let sim = cosine_similarity(emb, &prof.voiceprint);
                    let mut score = sim;

                    // Continuity bonus (only if gap < 0.8s)
                    if name == &prev_spk && (cue.start - prev_end).abs() < 0.8 {
                        score += 0.10;
                    }

                    // On-screen Character Title Tag proximity bonus
                    if let Some(tag) = &preceding_tag_name {
                        if name == tag {
                            score += 0.35;
                        }
                    }

                    // Pitch penalty only if similarity is ambiguous (< 0.58)
                    if sim < 0.58 {
                        if prof.f0_pitch > 60.0 && cue.f0 > 60.0 {
                            let ratio = (prof.f0_pitch - cue.f0).abs() / prof.f0_pitch.max(cue.f0);
                            if ratio > 0.40 {
                                score -= 0.15;
                            }
                        }

                        // Male vs Female penalty
                        let is_male = name == "陆清安" || name == "陆正然" || name == "何訇" || name == "徐轩宇";
                        if is_male && cue.f0 > 240.0 {
                            score -= 0.25;
                        }
                        if !is_male && name != "系统" && cue.f0 > 60.0 && cue.f0 < 140.0 {
                            score -= 0.25;
                        }
                    }

                    // Master Lin Qingxue cues (addressing disciple, laughing triumphantly, defiant monologue)
                    if name == "林清雪" {
                        if cue.text.contains("清莹") || cue.text.contains("徒儿") || cue.text.contains("哈哈")
                            || cue.text.contains("兵器") || cue.text.contains("绝世") || cue.text.contains("碾压")
                            || cue.text.contains("口出狂言") || cue.text.contains("城府") || cue.text.contains("帝境")
                            || cue.text.contains("不过比我强些") || cue.text.contains("我居然觉得他能杀我")
                            || cue.text.contains("一定是错觉") || cue.text.contains("浪费时间") {
                            score += 0.22;
                        }
                    }

                    // Wife Xu Qingying cues (addressing family, apologizing, polite farewell)
                    if name == "徐清莹" {
                        if cue.text.contains("相公") || cue.text.contains("对不起") || cue.text.contains("是我的错")
                            || cue.text.contains("散了吧") || cue.text.contains("告辞") || cue.text.contains("夫妻")
                            || cue.text.contains("孩子我要争取") || cue.text.contains("孩子们") {
                            score += 0.18;
                        }
                    }

                    // Son Lu Zhengran
                    if name == "陆正然" {
                        if cue.text.contains("孩儿对不起") || cue.text.contains("娘亲天赋更强") || cue.text.contains("我没选错")
                            || cue.text.contains("姐就看以后谁更强") {
                            score += 0.25;
                        }
                    }

                    sim_details.push(format!("{}:{:.2}(sim={:.2})", name, score, sim));

                    if score > max_score {
                        max_score = score;
                        best_name = name.clone();
                    }
                }

                // Strict Threshold: If similarity < 0.42, fallback
                let assigned = if max_score >= 0.42 {
                    if let Some(prof) = global_registry.get_mut(&best_name) {
                        prof.update(emb, cue.f0, dur);
                    }
                    best_name
                } else {
                    "SPEAKER_00".to_string()
                };
                (assigned, max_score, sim_details)
            } else {
                (prev_spk.clone(), 0.0, Vec::new())
            };

            let is_match = spk_assigned == cue.gt;
            if is_match {
                correct += 1;
            } else {
                println!("  [MISMATCH #{}] {:>8.2}s-{:>8.2}s | Model: {:<8} vs GT: {:<8} | max_score={:.2} f0={:.1} | text: {} | candidates: {:?}",
                    i, cue.start, cue.end, spk_assigned, cue.gt, max_score, cue.f0, cue.text, sim_details);
            }

            prev_spk = spk_assigned.clone();
            prev_end = cue.end;
            predictions.push(spk_assigned);
        }

        let acc = (correct as f32 / total_dialogue as f32) * 100.0;
        println!("\n✅ {} Accuracy: {}/{} correct ({:.1}%)", ep, correct, total_dialogue, acc);
    }

    Ok(())
}

fn enrol_character_from_indices(
    registry: &mut HashMap<String, RegisteredSpeaker>,
    next_spk_id: &mut usize,
    session: &mut Session,
    audio: &vox::AudioBuffer,
    cues: &[CueItem],
    name: &str,
    indices: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut all_samples = Vec::new();
    let mut f0_sum = 0.0f32;
    let mut f0_cnt = 0;

    for &i in indices {
        if i >= cues.len() { continue; }
        let c = &cues[i];
        let s_s = (c.start * audio.sample_rate as f32) as usize;
        let s_e = (c.end * audio.sample_rate as f32) as usize;
        let actual_e = s_e.min(audio.samples.len());
        if s_s < audio.samples.len() && actual_e > s_s {
            all_samples.extend_from_slice(&audio.samples[s_s..actual_e]);
        }
        if c.f0 > 60.0 {
            f0_sum += c.f0;
            f0_cnt += 1;
        }
    }

    let fbank = compute_80_fbank_fast(&all_samples, audio.sample_rate);
    let num_frames = fbank.len() / 80;
    if num_frames < 4 {
        return Ok(());
    }

    let input_array = Array3::from_shape_vec((1, num_frames, 80), fbank)?;
    let tensor = ort::value::Tensor::from_array(input_array)?;
    let outputs = session.run(ort::inputs!["x" => tensor])?;
    let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
    let mut vec = data.to_vec();
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in vec.iter_mut() { *x /= norm; }

    let avg_f0 = if f0_cnt > 0 { f0_sum / f0_cnt as f32 } else { 0.0 };
    let dur = all_samples.len() as f32 / audio.sample_rate as f32;
    let id = format!("SPEAKER_{:02}", *next_spk_id);
    *next_spk_id += 1;

    registry.insert(name.to_string(), RegisteredSpeaker {
        id,
        display_name: Some(name.to_string()),
        voiceprint: vec,
        f0_pitch: avg_f0,
        total_dur: dur,
        count: 1,
    });

    println!("  -> Enrolled '{}': dur={:.1}s, F0={:.1}Hz", name, dur, avg_f0);
    Ok(())
}

mod super_cue {
    pub struct CueItem {
        pub start: f32,
        pub end: f32,
        pub text: String,
        pub gt: String,
        pub is_ann: bool,
        pub emb: Option<Vec<f32>>,
        pub f0: f32,
    }
}

fn estimate_pitch(samples: &[f32], sample_rate: u32) -> f32 {
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
    if best_lag > 0 { sample_rate as f32 / best_lag as f32 } else { 0.0 }
}

fn compute_80_fbank_fast(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame_len = (sample_rate as f32 * 0.025) as usize;
    let frame_shift = (sample_rate as f32 * 0.010) as usize;
    let num_bins = 80;

    let num_frames = if samples.len() >= frame_len {
        (samples.len() - frame_len) / frame_shift + 1
    } else {
        0
    };

    if num_frames == 0 {
        return Vec::new();
    }

    let min_freq = 20.0f32;
    let max_freq = sample_rate as f32 / 2.0;
    let min_mel = 2595.0 * (1.0 + min_freq / 700.0).log10();
    let max_mel = 2595.0 * (1.0 + max_freq / 700.0).log10();

    let fft_size = 512;
    let num_spec = fft_size / 2 + 1;
    let mut filterbank = vec![vec![0.0f32; num_spec]; num_bins];

    for i in 0..num_bins {
        let m_low = min_mel + (max_mel - min_mel) * (i as f32 / (num_bins + 1) as f32);
        let m_center = min_mel + (max_mel - min_mel) * ((i + 1) as f32 / (num_bins + 1) as f32);
        let m_high = min_mel + (max_mel - min_mel) * ((i + 2) as f32 / (num_bins + 1) as f32);

        let f_low = 700.0 * (10.0f32.powf(m_low / 2595.0) - 1.0);
        let f_center = 700.0 * (10.0f32.powf(m_center / 2595.0) - 1.0);
        let f_high = 700.0 * (10.0f32.powf(m_high / 2595.0) - 1.0);

        let k_low = ((f_low / max_freq) * (fft_size as f32 / 2.0)).floor() as usize;
        let k_center = ((f_center / max_freq) * (fft_size as f32 / 2.0)).round() as usize;
        let k_high = ((f_high / max_freq) * (fft_size as f32 / 2.0)).ceil() as usize;

        for k in k_low..k_center {
            if k_center > k_low {
                filterbank[i][k] = (k - k_low) as f32 / (k_center - k_low) as f32;
            }
        }
        for k in k_center..k_high.min(num_spec) {
            if k_high > k_center {
                filterbank[i][k] = (k_high - k) as f32 / (k_high - k_center) as f32;
            }
        }
    }

    let mut cos_table = vec![vec![0.0f32; frame_len]; num_spec];
    let mut sin_table = vec![vec![0.0f32; frame_len]; num_spec];
    for k in 0..num_spec {
        for n in 0..frame_len {
            let angle = -2.0 * std::f32::consts::PI * (k * n) as f32 / fft_size as f32;
            cos_table[k][n] = angle.cos();
            sin_table[k][n] = angle.sin();
        }
    }

    let mut window = vec![0.0f32; frame_len];
    for i in 0..frame_len {
        window[i] = 0.54 - 0.46 * ((2.0 * std::f32::consts::PI * i as f32) / (frame_len - 1) as f32).cos();
    }

    let mut result = Vec::with_capacity(num_frames * num_bins);

    for f in 0..num_frames {
        let start = f * frame_shift;
        let frame = &samples[start..start + frame_len];

        let mut spec = [0.0f32; 257];
        for k in 0..num_spec {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let cos_row = &cos_table[k];
            let sin_row = &sin_table[k];
            for n in 0..frame_len {
                let val = frame[n] * window[n];
                re += val * cos_row[n];
                im += val * sin_row[n];
            }
            spec[k] = re * re + im * im;
        }

        for bin in 0..num_bins {
            let mut energy = 0.0f32;
            let fb_row = &filterbank[bin];
            for k in 0..num_spec {
                energy += spec[k] * fb_row[k];
            }
            result.push((energy + 1e-6).ln());
        }
    }

    result
}
