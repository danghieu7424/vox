use std::collections::HashMap;
use ort::session::Session;
use ndarray::Array3;
use vox::{load_subtitles, load_wav, cosine_similarity};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing CAM++ 192-d Speaker Embeddings on Ep 1 & 2 ===");

    let env = ort::init().with_name("test_campplus").build()?;
    let mut session = Session::builder(&env)?
        .with_intra_threads(4)?
        .commit_from_file("models/campplus.onnx")?;

    for ep in ["第001集", "第002集"] {
        let final_vtt = format!("final/{}.zh.diarized.vtt", ep);
        let wav_path = format!("input/{}.vox.wav", ep);

        let (segments, _) = load_subtitles(&final_vtt)?;
        let audio = load_wav(&wav_path)?;

        println!("\n==================== {} ====================", ep);

        let mut speaker_samples: HashMap<String, Vec<f32>> = HashMap::new();

        for seg in &segments {
            if seg.text.starts_with('[') {
                if let Some(close_idx) = seg.text.find(']') {
                    let spk_name = seg.text[1..close_idx].to_string();
                    let start_sample = (seg.start_sec * audio.sample_rate as f32) as usize;
                    let end_sample = (seg.end_sec * audio.sample_rate as f32) as usize;
                    if start_sample < audio.samples.len() {
                        let actual_end = end_sample.min(audio.samples.len());
                        let slice = &audio.samples[start_sample..actual_end];
                        speaker_samples.entry(spk_name).or_default().extend_from_slice(slice);
                    }
                }
            }
        }

        let mut speaker_embeddings: HashMap<String, Vec<f32>> = HashMap::new();

        for (spk, samples) in &speaker_samples {
            if samples.len() < 1600 { continue; } // At least 0.1s
            let fbank = compute_80_fbank(samples, audio.sample_rate);
            let num_frames = fbank.len() / 80;
            if num_frames < 10 { continue; }

            let input_array = Array3::from_shape_vec((1, num_frames, 80), fbank)?;
            let tensor = ort::value::Tensor::from_array(input_array)?;

            let outputs = session.run(ort::inputs!["x" => tensor])?;
            let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;

            let mut emb = data.to_vec();
            // L2 normalize
            let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in emb.iter_mut() { *x /= norm; }

            speaker_embeddings.insert(spk.clone(), emb);
            println!("  • {:<12}: {:>5.1}s speech -> 192-d embedding extracted", spk, samples.len() as f32 / audio.sample_rate as f32);
        }

        let spk_list: Vec<String> = speaker_embeddings.keys().cloned().collect();
        println!("\nCosine Similarity Matrix for {}:", ep);
        print!("{:<12}", "");
        for s2 in &spk_list {
            let short: String = s2.chars().take(4).collect();
            print!("{:>8}", short);
        }
        println!();

        for s1 in &spk_list {
            let short1: String = s1.chars().take(4).collect();
            print!("{:<12}", short1);
            for s2 in &spk_list {
                let sim = cosine_similarity(&speaker_embeddings[s1], &speaker_embeddings[s2]);
                print!("{:>8.2}", sim);
            }
            println!();
        }
    }

    Ok(())
}

fn compute_80_fbank(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame_len = (sample_rate as f32 * 0.025) as usize; // 400 samples
    let frame_shift = (sample_rate as f32 * 0.010) as usize; // 160 samples
    let num_bins = 80;

    let num_frames = if samples.len() >= frame_len {
        (samples.len() - frame_len) / frame_shift + 1
    } else {
        0
    };

    if num_frames == 0 {
        return Vec::new();
    }

    // Mel filterbank cutoffs
    let min_freq = 20.0f32;
    let max_freq = sample_rate as f32 / 2.0;
    let min_mel = 2595.0 * (1.0 + min_freq / 700.0).log10();
    let max_mel = 2595.0 * (1.0 + max_freq / 700.0).log10();

    let fft_size = 512;
    let mut filterbank = vec![vec![0.0f32; fft_size / 2 + 1]; num_bins];

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
        for k in k_center..k_high.min(fft_size / 2 + 1) {
            if k_high > k_center {
                filterbank[i][k] = (k_high - k) as f32 / (k_high - k_center) as f32;
            }
        }
    }

    // Hamming window
    let mut window = vec![0.0f32; frame_len];
    for i in 0..frame_len {
        window[i] = 0.54 - 0.46 * ((2.0 * std::f32::consts::PI * i as f32) / (frame_len - 1) as f32).cos();
    }

    let mut result = Vec::with_capacity(num_frames * num_bins);

    for f in 0..num_frames {
        let start = f * frame_shift;
        let frame = &samples[start..start + frame_len];

        // Apply window and compute power spectrum
        let mut spec = [0.0f32; 257];
        for k in 0..257 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for n in 0..frame_len {
                let val = frame[n] * window[n];
                let angle = -2.0 * std::f32::consts::PI * (k * n) as f32 / fft_size as f32;
                re += val * angle.cos();
                im += val * angle.sin();
            }
            spec[k] = re * re + im * im;
        }

        // Apply 80 mel filters
        for bin in 0..num_bins {
            let mut energy = 0.0f32;
            for k in 0..257 {
                energy += spec[k] * filterbank[bin][k];
            }
            result.push((energy + 1e-6).ln());
        }
    }

    result
}
