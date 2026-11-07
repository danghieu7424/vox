use crate::alignment::SpeakerTurn;
use crate::error::VoxError;
use crate::progress::{ProgressInfo, ProgressStage};
use crate::subtitles::SpeakerFormat;
use ndarray::Array3;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

pub const DEFAULT_EMBEDDING_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

pub const DEFAULT_SEGMENTATION_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.int8.onnx";

// Backward compatibility
pub const DEFAULT_MODEL_URL: &str = DEFAULT_EMBEDDING_MODEL_URL;

static GLOBAL_ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

pub fn get_or_init_environment() -> Result<&'static ort::environment::Environment, VoxError> {
    if let Some(env) = GLOBAL_ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("vox_inference")
        .build()?;
    let _ = GLOBAL_ORT_ENV.set(env);
    Ok(GLOBAL_ORT_ENV.get().unwrap())
}

#[derive(Debug, Clone)]
pub struct VoxConfig {
    /// Đường dẫn file model Speaker Embedding (CAM++)
    pub model_path: PathBuf,
    /// URL tải model Speaker Embedding nếu chưa có trên đĩa
    pub model_url: String,
    /// Đường dẫn file model Segmentation (Pyannote)
    pub segmentation_model_path: PathBuf,
    pub segmentation_model_url: String,
    /// Thư mục xuất phụ đề đầu ra
    pub output_dir: PathBuf,
    /// Định dạng tiền tố người nói (mặc định NamedIfAvailable: [Tên]: text hoặc [SPEAKER_xx]: text)
    pub speaker_format: SpeakerFormat,
    /// Số luồng CPU sử dụng
    pub cpu_threads: usize,
    pub onset_threshold: f32,
    pub offset_threshold: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/campplus.onnx"),
            model_url: DEFAULT_EMBEDDING_MODEL_URL.to_string(),
            segmentation_model_path: PathBuf::from("models/pyannote_segmentation.onnx"),
            segmentation_model_url: DEFAULT_SEGMENTATION_MODEL_URL.to_string(),
            output_dir: PathBuf::from("output"),
            speaker_format: SpeakerFormat::NamedIfAvailable,
            cpu_threads: 4,
            onset_threshold: 0.5,
            offset_threshold: 0.5,
            min_duration_on: 0.3,
            min_duration_off: 0.5,
        }
    }
}

/// Đảm bảo model tồn tại ở đường dẫn chỉ định:
/// `let model = Path::new("models/..."); if !model.exists() { ensure_model(...) }`
pub fn ensure_model(
    model_path: &Path,
    model_url: &str,
    mut progress_cb: Option<&mut dyn FnMut(&ProgressInfo)>,
) -> Result<PathBuf, VoxError> {
    if model_path.exists() {
        if let Some(cb) = progress_cb.as_mut() {
            cb(&ProgressInfo::new(
                ProgressStage::CheckingModel,
                1,
                1,
                "file",
                "0s",
                format!("Đã tìm thấy mô hình ONNX tại: {}", model_path.display()),
            ));
        }
        return Ok(model_path.to_path_buf());
    }

    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if let Some(cb) = progress_cb.as_mut() {
        cb(&ProgressInfo::new(
            ProgressStage::DownloadingModel,
            0,
            100,
            "%",
            "...",
            format!("Đang tải mô hình ONNX từ: {}", model_url),
        ));
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let mut response = client
        .get(model_url)
        .send()?
        .error_for_status()
        .map_err(|e| VoxError::ModelDownloadFailed(format!("HTTP request failed: {}", e)))?;

    let total_bytes = response.content_length().unwrap_or(28_281_138);

    let temp_dest = model_path.with_extension("tmp");
    let mut file = File::create(&temp_dest)?;
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 16384];
    let start_time = Instant::now();

    loop {
        let bytes_read = std::io::Read::read(&mut response, &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        if let Some(cb) = progress_cb.as_mut() {
            let percentage = (downloaded as f32 / total_bytes as f32 * 100.0).min(100.0);
            let elapsed = start_time.elapsed().as_secs_f32();
            let eta = if downloaded > 0 && percentage < 100.0 {
                let speed = downloaded as f32 / elapsed;
                let remaining_bytes = total_bytes.saturating_sub(downloaded) as f32;
                format!("{:.0}s còn lại", remaining_bytes / speed)
            } else {
                format!("{:.0}s", elapsed)
            };

            cb(&ProgressInfo::new(
                ProgressStage::DownloadingModel,
                downloaded as usize / 1024,
                total_bytes as usize / 1024,
                "KB",
                eta,
                format!(
                    "Tải mô hình: {:.1} MB / {:.1} MB",
                    downloaded as f32 / 1_048_576.0,
                    total_bytes as f32 / 1_048_576.0
                ),
            ));
        }
    }

    file.flush()?;
    std::fs::rename(&temp_dest, model_path)?;

    if let Some(cb) = progress_cb.as_mut() {
        cb(&ProgressInfo::new(
            ProgressStage::DownloadingModel,
            100,
            100,
            "%",
            format!("{:.1}s", start_time.elapsed().as_secs_f32()),
            format!("Tải mô hình hoàn tất: {}", model_path.display()),
        ));
    }

    Ok(model_path.to_path_buf())
}

/// Trích xuất Deep Speaker Embedding 192-chiều bằng mô hình CAM++ ONNX
pub struct SpeakerEmbeddingExtractor {
    session: Session,
    cos_table: Vec<Vec<f32>>,
    sin_table: Vec<Vec<f32>>,
    filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
    frame_len: usize,
    frame_shift: usize,
    num_bins: usize,
}

impl SpeakerEmbeddingExtractor {
    pub fn load<P: AsRef<Path>>(model_path: P, cpu_threads: usize) -> Result<Self, VoxError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }

        let env = get_or_init_environment()?;
        let session = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(cpu_threads)?
            .commit_from_file(path)?;

        let sample_rate = 16000;
        let frame_len = (sample_rate as f32 * 0.025) as usize; // 400
        let frame_shift = (sample_rate as f32 * 0.010) as usize; // 160
        let num_bins = 80;

        let min_freq = 20.0f32;
        let max_freq = sample_rate as f32 / 2.0;
        let min_mel = 2595.0 * (1.0 + min_freq / 700.0).log10();
        let max_mel = 2595.0 * (1.0 + max_freq / 700.0).log10();

        let fft_size = 512;
        let num_spec = fft_size / 2 + 1; // 257
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

        Ok(Self {
            session,
            cos_table,
            sin_table,
            filterbank,
            window,
            frame_len,
            frame_shift,
            num_bins,
        })
    }

    /// Trích xuất vector đặc trưng embedding 192 chiều (đã chuẩn hoá L2) từ đoạn audio
    pub fn extract_embedding(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>, VoxError> {
        if samples.len() < self.frame_len {
            return Ok(None);
        }

        let num_frames = (samples.len() - self.frame_len) / self.frame_shift + 1;
        if num_frames < 4 {
            return Ok(None);
        }

        let mut fbank = Vec::with_capacity(num_frames * self.num_bins);
        let num_spec = self.cos_table.len();

        for f in 0..num_frames {
            let start = f * self.frame_shift;
            let frame = &samples[start..start + self.frame_len];

            let mut spec = [0.0f32; 257];
            for k in 0..num_spec {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let cos_row = &self.cos_table[k];
                let sin_row = &self.sin_table[k];
                for n in 0..self.frame_len {
                    let val = frame[n] * self.window[n];
                    re += val * cos_row[n];
                    im += val * sin_row[n];
                }
                spec[k] = re * re + im * im;
            }

            for bin in 0..self.num_bins {
                let mut energy = 0.0f32;
                let fb_row = &self.filterbank[bin];
                for k in 0..num_spec {
                    energy += spec[k] * fb_row[k];
                }
                fbank.push((energy + 1e-6).ln());
            }
        }

        let input_array = Array3::from_shape_vec((1, num_frames, self.num_bins), fbank)
            .map_err(|e| VoxError::ShapeError(format!("Array3 error: {}", e)))?;
        let tensor = ort::value::Tensor::from_array(input_array)?;

        let outputs = self.session.run(ort::inputs!["x" => tensor])?;
        let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;

        let mut emb = data.to_vec();
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in emb.iter_mut() {
            *x /= norm;
        }

        Ok(Some(emb))
    }
}

/// Ước tính cao độ cơ bản (F0 pitch in Hz) của đoạn audio bằng phương pháp Autocorrelation
pub fn estimate_f0(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.len() < 512 {
        return 0.0;
    }

    let frame_len = 1024.min(samples.len());
    let center = samples.len() / 2;
    let start = center.saturating_sub(frame_len / 2);
    let frame = &samples[start..(start + frame_len).min(samples.len())];

    let min_lag = (sample_rate / 500) as usize; // ~500Hz
    let max_lag = (sample_rate / 70) as usize;  // ~70Hz
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

    if best_lag > 0 {
        sample_rate as f32 / best_lag as f32
    } else {
        0.0
    }
}

/// Mô hình Pyannote Segmentation chạy inference trên CPU
pub struct VoxModel {
    pub session: Session,
    pub window_size: usize,
    pub window_shift: usize,
    pub sample_rate: u32,
    pub num_frames_per_window: usize,
    pub receptive_field_shift_samples: f64,
    pub seconds_per_frame: f64,
}

impl VoxModel {
    pub fn load<P: AsRef<Path>>(model_path: P, cpu_threads: usize) -> Result<Self, VoxError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }

        let env = get_or_init_environment()?;

        let session = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(cpu_threads)?
            .commit_from_file(path)?;

        let window_size = 160_000;
        let window_shift = 32_000;
        let sample_rate = 16_000;
        let num_frames_per_window = 589;
        let receptive_field_shift_samples = window_size as f64 / num_frames_per_window as f64;
        let seconds_per_frame = 10.0 / num_frames_per_window as f64;

        Ok(Self {
            session,
            window_size,
            window_shift,
            sample_rate,
            num_frames_per_window,
            receptive_field_shift_samples,
            seconds_per_frame,
        })
    }

    pub fn run_diarization(
        &mut self,
        samples: &[f32],
        config: &VoxConfig,
        mut progress_cb: Option<&mut dyn FnMut(&ProgressInfo)>,
    ) -> Result<Vec<SpeakerTurn>, VoxError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let total_samples = samples.len();
        let total_audio_sec = total_samples as f32 / self.sample_rate as f32;

        let num_chunks = if total_samples <= self.window_size {
            1
        } else {
            let rem = total_samples - self.window_size;
            rem / self.window_shift + 1 + if rem % self.window_shift > 0 { 1 } else { 0 }
        };

        let start_time = Instant::now();

        let mut hamming = Vec::with_capacity(self.num_frames_per_window);
        for i in 0..self.num_frames_per_window {
            let w = 0.54
                - 0.46
                    * ((2.0 * std::f64::consts::PI * i as f64)
                        / (self.num_frames_per_window - 1) as f64)
                        .cos();
            hamming.push(w as f32);
        }

        let total_frames = ((total_samples as f64 / self.receptive_field_shift_samples).ceil()
            as usize)
            + self.num_frames_per_window
            + 10;

        let mut timeline_probs = vec![[0.0f32; 3]; total_frames];
        let mut timeline_weights = vec![0.0f32; total_frames];
        let mut prev_chunk_probs: Option<Vec<[f32; 3]>> = None;

        for chunk_idx in 0..num_chunks {
            let start_sample = chunk_idx * self.window_shift;
            let end_sample = (start_sample + self.window_size).min(total_samples);

            let mut chunk_samples = vec![0.0f32; self.window_size];
            let copy_len = end_sample - start_sample;
            chunk_samples[..copy_len].copy_from_slice(&samples[start_sample..end_sample]);

            let input_array = Array3::from_shape_vec((1, 1, self.window_size), chunk_samples)
                .map_err(|e| VoxError::ShapeError(format!("Array3 error: {}", e)))?;
            let input_tensor = ort::value::Tensor::from_array(input_array)?;

            let outputs = self.session.run(ort::inputs!["x" => input_tensor])?;
            let (_shape, data) = outputs["y"].try_extract_tensor::<f32>()?;

            let mut current_chunk_probs = Vec::with_capacity(self.num_frames_per_window);
            for f in 0..self.num_frames_per_window {
                let offset = f * 7;
                let c_vals = &data[offset..offset + 7];

                let mut max_val = c_vals[0];
                for &v in c_vals.iter() {
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut exp_vals = [0.0f32; 7];
                let mut sum_exp = 0.0f32;
                for (j, &v) in c_vals.iter().enumerate() {
                    let ev = (v - max_val).exp();
                    exp_vals[j] = ev;
                    sum_exp += ev;
                }
                let sum_safe = sum_exp.max(1e-12);
                for v in exp_vals.iter_mut() {
                    *v /= sum_safe;
                }

                let spk0 = exp_vals[1] + exp_vals[4] + exp_vals[5];
                let spk1 = exp_vals[2] + exp_vals[4] + exp_vals[6];
                let spk2 = exp_vals[3] + exp_vals[5] + exp_vals[6];
                current_chunk_probs.push([spk0, spk1, spk2]);
            }

            if let Some(prev) = prev_chunk_probs {
                let overlap_samples = self.window_size.saturating_sub(self.window_shift);
                let overlap_frames = ((overlap_samples as f64
                    / self.receptive_field_shift_samples)
                    .round() as usize)
                    .min(self.num_frames_per_window);

                if overlap_frames > 0 {
                    let prev_start_idx = self.num_frames_per_window.saturating_sub(overlap_frames);
                    let perm = find_best_permutation(
                        &prev[prev_start_idx..],
                        &current_chunk_probs[..overlap_frames],
                    );

                    for frame_prob in current_chunk_probs.iter_mut() {
                        let original = *frame_prob;
                        frame_prob[0] = original[perm[0]];
                        frame_prob[1] = original[perm[1]];
                        frame_prob[2] = original[perm[2]];
                    }
                }
            }

            let frame_start_timeline =
                (start_sample as f64 / self.receptive_field_shift_samples).round() as usize;

            for f in 0..self.num_frames_per_window {
                let t_idx = frame_start_timeline + f;
                if t_idx < total_frames {
                    let w = hamming[f];
                    timeline_probs[t_idx][0] += current_chunk_probs[f][0] * w;
                    timeline_probs[t_idx][1] += current_chunk_probs[f][1] * w;
                    timeline_probs[t_idx][2] += current_chunk_probs[f][2] * w;
                    timeline_weights[t_idx] += w;
                }
            }

            prev_chunk_probs = Some(current_chunk_probs);

            if let Some(cb) = progress_cb.as_mut() {
                let current_chunk = chunk_idx + 1;
                let current_sec = (end_sample as f32 / self.sample_rate as f32).min(total_audio_sec);
                let elapsed = start_time.elapsed().as_secs_f32();
                let eta = if current_chunk > 0 {
                    let pace = elapsed / current_chunk as f32;
                    let remaining = (num_chunks - current_chunk) as f32 * pace;
                    format!("{:.0}s còn lại", remaining)
                } else {
                    format!("{:.0}s", elapsed)
                };

                cb(&ProgressInfo::new(
                    ProgressStage::Inference,
                    current_chunk,
                    num_chunks,
                    "chunks",
                    eta,
                    format!("Phân tích audio: {:.1}s / {:.1}s", current_sec, total_audio_sec),
                ));
            }
        }

        for i in 0..total_frames {
            let w = timeline_weights[i];
            if w > 1e-6 {
                timeline_probs[i][0] /= w;
                timeline_probs[i][1] /= w;
                timeline_probs[i][2] /= w;
            }
        }

        let mut raw_turns = Vec::new();
        let actual_total_frames = (total_samples as f64 / self.receptive_field_shift_samples)
            .ceil() as usize;
        let actual_frames = actual_total_frames.min(total_frames);

        for spk_idx in 0..3 {
            let spk_name = format!("SPEAKER_{:02}", spk_idx);
            let mut is_active = false;
            let mut start_frame = 0;

            for f in 0..actual_frames {
                let prob = timeline_probs[f][spk_idx];
                if !is_active {
                    if prob >= config.onset_threshold {
                        is_active = true;
                        start_frame = f;
                    }
                } else if prob < config.offset_threshold {
                    is_active = false;
                    let start_sec = start_frame as f32 * self.seconds_per_frame as f32;
                    let end_sec = f as f32 * self.seconds_per_frame as f32;
                    if end_sec - start_sec >= config.min_duration_on {
                        raw_turns.push(SpeakerTurn::new(start_sec, end_sec, &spk_name));
                    }
                }
            }

            if is_active {
                let start_sec = start_frame as f32 * self.seconds_per_frame as f32;
                let end_sec = actual_frames as f32 * self.seconds_per_frame as f32;
                if end_sec - start_sec >= config.min_duration_on {
                    raw_turns.push(SpeakerTurn::new(start_sec, end_sec, &spk_name));
                }
            }
        }

        raw_turns.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
        let merged_turns = merge_consecutive_turns(raw_turns, config.min_duration_off);

        Ok(merged_turns)
    }
}

fn find_best_permutation(prev_overlap: &[[f32; 3]], curr_overlap: &[[f32; 3]]) -> [usize; 3] {
    let n = prev_overlap.len().min(curr_overlap.len());
    if n == 0 {
        return [0, 1, 2];
    }

    let mut corr = [[0.0f32; 3]; 3];
    for t in 0..n {
        for i in 0..3 {
            for j in 0..3 {
                corr[i][j] += prev_overlap[t][i] * curr_overlap[t][j];
            }
        }
    }

    let permutations: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut best_score = -f32::MAX;
    let mut best_perm = [0, 1, 2];

    for p in permutations {
        let score = corr[0][p[0]] + corr[1][p[1]] + corr[2][p[2]];
        if score > best_score {
            best_score = score;
            best_perm = p;
        }
    }

    best_perm
}

fn merge_consecutive_turns(turns: Vec<SpeakerTurn>, min_gap: f32) -> Vec<SpeakerTurn> {
    if turns.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<SpeakerTurn> = Vec::new();

    for turn in turns {
        if let Some(last) = merged.last_mut() {
            if last.speaker_id == turn.speaker_id && turn.start_sec - last.end_sec <= min_gap {
                last.end_sec = turn.end_sec.max(last.end_sec);
                continue;
            }
        }
        merged.push(turn);
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_turns() {
        let turns = vec![
            SpeakerTurn::new(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn::new(1.2, 2.5, "SPEAKER_00"),
            SpeakerTurn::new(2.8, 4.0, "SPEAKER_01"),
        ];

        let merged = merge_consecutive_turns(turns, 0.5);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].speaker_id, "SPEAKER_00");
        assert_eq!(merged[0].start_sec, 0.0);
        assert_eq!(merged[0].end_sec, 2.5);
        assert_eq!(merged[1].speaker_id, "SPEAKER_01");
    }

    #[test]
    fn test_permutation_finder() {
        let prev = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let curr_swapped = vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        let perm = find_best_permutation(&prev, &curr_swapped);
        assert_eq!(perm, [1, 0, 2]);
    }
}
