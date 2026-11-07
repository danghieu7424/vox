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
use std::time::Instant;

pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.int8.onnx";

#[derive(Debug, Clone)]
pub struct VoxConfig {
    pub model_path: PathBuf,
    pub model_url: String,
    pub output_dir: PathBuf,
    pub speaker_format: SpeakerFormat,
    pub cpu_threads: usize,
    pub onset_threshold: f32,
    pub offset_threshold: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/pyannote_segmentation.onnx"),
            model_url: DEFAULT_MODEL_URL.to_string(),
            output_dir: PathBuf::from("output"),
            speaker_format: SpeakerFormat::Prefix,
            cpu_threads: 4,
            onset_threshold: 0.5,
            offset_threshold: 0.5,
            min_duration_on: 0.3,
            min_duration_off: 0.5,
        }
    }
}

/// Đảm bảo model tồn tại ở đường dẫn chỉ định:
/// `let model = Path::new("models/..."); if !model.exists() { ... }`
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
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let mut response = client
        .get(model_url)
        .send()?
        .error_for_status()
        .map_err(|e| VoxError::ModelDownloadFailed(format!("HTTP request failed: {}", e)))?;

    let total_bytes = response.content_length().unwrap_or(1540506); // ~1.54MB default INT8

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
            let elapsed = start_time.elapsed().as_secs();
            let duration_str = format!("{}s", elapsed);
            let mb_down = downloaded as f32 / 1_048_576.0;
            let mb_total = total_bytes as f32 / 1_048_576.0;
            let percent = ((downloaded as f64 / total_bytes as f64) * 100.0).min(100.0) as usize;
            cb(&ProgressInfo::new(
                ProgressStage::DownloadingModel,
                percent,
                100,
                "%",
                duration_str,
                format!("Đã tải {:.2} / {:.2} MB", mb_down, mb_total),
            ));
        }
    }

    file.flush()?;
    drop(file);
    std::fs::rename(&temp_dest, model_path)?;

    if let Some(cb) = progress_cb.as_mut() {
        cb(&ProgressInfo::new(
            ProgressStage::DownloadingModel,
            100,
            100,
            "%",
            format!("{:.1}s", start_time.elapsed().as_secs_f32()),
            "Tải mô hình thành công!",
        ));
    }

    Ok(model_path.to_path_buf())
}

pub struct VoxModel {
    session: Session,
    pub window_size: usize,
    pub window_shift: usize,
    pub sample_rate: usize,
    pub num_frames_per_window: usize,
    pub receptive_field_shift_samples: f64,
    pub seconds_per_frame: f64,
}

use std::sync::OnceLock;

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

impl VoxModel {
    /// Nạp mô hình ONNX Runtime được tối ưu hóa cho CPU
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

        let window_size = 160_000; // 10.0 giây ở 16kHz
        let window_shift = 32_000; // 2.0 giây bước nhảy (80% overlap)
        let sample_rate = 16_000;
        let num_frames_per_window = 589;
        let receptive_field_shift_samples = window_size as f64 / num_frames_per_window as f64; // ~271.64
        let seconds_per_frame = 10.0 / num_frames_per_window as f64; // ~0.0169779

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

    /// Chạy Diarization inference trên toàn bộ audio buffer với thuật toán trượt cửa sổ (Sliding Window)
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

        // Tính số lượng chunk
        let num_chunks = if total_samples <= self.window_size {
            1
        } else {
            let rem = total_samples - self.window_size;
            rem / self.window_shift + 1 + if rem % self.window_shift > 0 { 1 } else { 0 }
        };

        let start_time = Instant::now();

        // Chuẩn bị mảng Hamming window
        let mut hamming = Vec::with_capacity(self.num_frames_per_window);
        for i in 0..self.num_frames_per_window {
            let w = 0.54
                - 0.46
                    * ((2.0 * std::f64::consts::PI * i as f64)
                        / (self.num_frames_per_window - 1) as f64)
                        .cos();
            hamming.push(w as f32);
        }

        // Ước lượng tổng số frame timeline
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

            // Cắt hoặc pad cho đủ self.window_size
            let mut chunk_samples = vec![0.0f32; self.window_size];
            let copy_len = end_sample - start_sample;
            chunk_samples[..copy_len].copy_from_slice(&samples[start_sample..end_sample]);

            // Chuẩn bị tensor (1, 1, 160000)
            let input_array = Array3::from_shape_vec((1, 1, self.window_size), chunk_samples)
                .map_err(|e| VoxError::ShapeError(format!("Array3 error: {}", e)))?;
            let input_tensor = ort::value::Tensor::from_array(input_array)?;

            let outputs = self
                .session
                .run(ort::inputs!["x" => input_tensor])?;

            let (_shape, data) = outputs["y"]
                .try_extract_tensor::<f32>()?;

            // Parse 589 frames x 7 classes sang 589 frames x 3 speakers
            let mut current_chunk_probs = Vec::with_capacity(self.num_frames_per_window);
            for f in 0..self.num_frames_per_window {
                let offset = f * 7;
                let c_vals = &data[offset..offset + 7];

                // Chuyển log-probabilities thành probabilities (softmax hoặc exp)
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

                // Powerset mapping to 3 speakers:
                // Class 0: silence
                // Class 1: speaker 0
                // Class 2: speaker 1
                // Class 3: speaker 2
                // Class 4: speaker 0 + 1
                // Class 5: speaker 0 + 2
                // Class 6: speaker 1 + 2
                let spk0 = exp_vals[1] + exp_vals[4] + exp_vals[5];
                let spk1 = exp_vals[2] + exp_vals[4] + exp_vals[6];
                let spk2 = exp_vals[3] + exp_vals[5] + exp_vals[6];
                current_chunk_probs.push([spk0, spk1, spk2]);
            }

            // Giải quyết hoán vị người nói (Speaker permutation alignment) với chunk trước đó trong vùng overlap
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

                    // Áp dụng hoán vị lên current_chunk_probs
                    for frame_prob in current_chunk_probs.iter_mut() {
                        let original = *frame_prob;
                        frame_prob[0] = original[perm[0]];
                        frame_prob[1] = original[perm[1]];
                        frame_prob[2] = original[perm[2]];
                    }
                }
            }

            // Tích luỹ lên timeline chung với trọng số Hamming
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

            // Báo tiến trình cho từng chunk
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

        // Chuẩn hóa timeline bằng tổng trọng số
        for i in 0..total_frames {
            let w = timeline_weights[i];
            if w > 1e-6 {
                timeline_probs[i][0] /= w;
                timeline_probs[i][1] /= w;
                timeline_probs[i][2] /= w;
            }
        }

        // Trích xuất các lượt nói (Speaker Turns) từ timeline_probs
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

        // Sắp xếp các turns theo thời gian bắt đầu
        raw_turns.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());

        // Hợp nhất các khoảng trống nhỏ giữa các turns của cùng 1 speaker
        let merged_turns = merge_consecutive_turns(raw_turns, config.min_duration_off);

        Ok(merged_turns)
    }
}

/// Tìm hoán vị tối ưu của 3 speaker channels để tương đồng nhất với chunk trước đó
fn find_best_permutation(prev_overlap: &[[f32; 3]], curr_overlap: &[[f32; 3]]) -> [usize; 3] {
    let n = prev_overlap.len().min(curr_overlap.len());
    if n == 0 {
        return [0, 1, 2];
    }

    // Tính ma trận tương quan giữa prev_speaker i và curr_speaker j
    let mut corr = [[0.0f32; 3]; 3];
    for t in 0..n {
        for i in 0..3 {
            for j in 0..3 {
                corr[i][j] += prev_overlap[t][i] * curr_overlap[t][j];
            }
        }
    }

    // 6 hoán vị khả dĩ của 3 phần tử
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
        // Điểm số: prev 0 khớp với curr p[0], prev 1 khớp với curr p[1], prev 2 khớp với curr p[2]
        let score = corr[0][p[0]] + corr[1][p[1]] + corr[2][p[2]];
        if score > best_score {
            best_score = score;
            best_perm = p;
        }
    }

    best_perm
}

/// Hợp nhất các turns liên tiếp của cùng 1 speaker nếu khoảng trống nhỏ hơn min_gap
fn merge_consecutive_turns(mut turns: Vec<SpeakerTurn>, min_gap: f32) -> Vec<SpeakerTurn> {
    if turns.len() <= 1 {
        return turns;
    }

    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < turns.len() {
            if turns[i].speaker_id == turns[i + 1].speaker_id
                && turns[i + 1].start_sec >= turns[i].end_sec
                && (turns[i + 1].start_sec - turns[i].end_sec) <= min_gap
            {
                turns[i].end_sec = turns[i + 1].end_sec;
                turns.remove(i + 1);
                changed = true;
            } else {
                i += 1;
            }
        }
    }

    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_finder() {
        let prev = vec![[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]];
        let curr_swapped = vec![[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]; // 0 và 1 bị đảo
        let perm = find_best_permutation(&prev, &curr_swapped);
        assert_eq!(perm, [1, 0, 2]);
    }

    #[test]
    fn test_merge_turns() {
        let turns = vec![
            SpeakerTurn::new(1.0, 2.0, "SPEAKER_00"),
            SpeakerTurn::new(2.2, 3.5, "SPEAKER_00"), // gap = 0.2s <= 0.5s -> merge!
            SpeakerTurn::new(4.0, 5.0, "SPEAKER_01"),
        ];
        let merged = merge_consecutive_turns(turns, 0.5);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].end_sec, 3.5);
    }
}
