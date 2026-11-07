use crate::alignment::SpeakerTurn;
use crate::error::VoxError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Hồ sơ giọng nói (Voiceprint Profile) của một nhân vật xuyên suốt các tập phim
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeakerProfile {
    pub speaker_id: String,                  // Ví dụ "SPEAKER_00"
    pub display_name: Option<String>,        // Tên nhân vật (ví dụ "陆清安", "系统", "小女儿-陆淼淼")
    pub voiceprint: Vec<f32>,                // Vector đặc trưng âm sắc (L2-normalized)
    pub total_duration_sec: f32,             // Tổng thời lượng đã nói qua tất cả các tập
    pub sample_count: usize,                 // Số lượt nói đã học
    pub episode_appearances: Vec<String>,    // Danh sách các tập đã xuất hiện: ["第001集", "第002集"]
}

impl SpeakerProfile {
    pub fn new(
        speaker_id: impl Into<String>,
        voiceprint: Vec<f32>,
        duration_sec: f32,
        episode_name: impl Into<String>,
        display_name: Option<String>,
    ) -> Self {
        Self {
            speaker_id: speaker_id.into(),
            display_name,
            voiceprint,
            total_duration_sec: duration_sec,
            sample_count: 1,
            episode_appearances: vec![episode_name.into()],
        }
    }

    /// Cập nhật voiceprint centroid bằng trung bình động có trọng số thời lượng
    pub fn update(
        &mut self,
        new_voiceprint: &[f32],
        duration_sec: f32,
        episode_name: &str,
        name_hint: Option<&str>,
    ) {
        if self.voiceprint.len() == new_voiceprint.len() {
            let total_dur = (self.total_duration_sec + duration_sec).max(1e-6);
            let w_old = self.total_duration_sec / total_dur;
            let w_new = duration_sec / total_dur;

            let mut updated = Vec::with_capacity(self.voiceprint.len());
            for i in 0..self.voiceprint.len() {
                updated.push(self.voiceprint[i] * w_old + new_voiceprint[i] * w_new);
            }
            normalize_l2(&mut updated);
            self.voiceprint = updated;
            self.total_duration_sec += duration_sec;
            self.sample_count += 1;
        }

        if !self.episode_appearances.iter().any(|e| e == episode_name) {
            self.episode_appearances.push(episode_name.to_string());
        }

        if self.display_name.is_none() {
            if let Some(name) = name_hint {
                self.display_name = Some(name.to_string());
            }
        }
    }
}

/// Sổ bộ lưu trữ và nhận diện nhân vật (Global Speaker Registry)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerRegistry {
    pub speakers: HashMap<String, SpeakerProfile>,
    pub next_speaker_index: usize,
    pub similarity_threshold: f32, // Ngưỡng tương đồng cosine để xác định cùng 1 người (mặc định 0.68)
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self {
            speakers: HashMap::new(),
            next_speaker_index: 0,
            similarity_threshold: 0.68,
        }
    }
}

impl SpeakerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Đọc sổ bộ từ file JSON (để tiếp tục chuỗi phim đã xử lý trước đó)
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, VoxError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let registry: Self = serde_json::from_reader(reader)
            .map_err(|e| VoxError::InvalidData(format!("Failed to parse speaker registry: {}", e)))?;
        Ok(registry)
    }

    /// Lưu sổ bộ ra file JSON
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), VoxError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| VoxError::InvalidData(format!("Failed to save speaker registry: {}", e)))?;
        Ok(())
    }

    /// Khớp hoặc đăng ký một nhân vật mới dựa trên voiceprint:
    /// - Nếu độ tương đồng cosine >= threshold: Trả về ID của nhân vật cũ (ví dụ "SPEAKER_01"), cập nhật profile
    /// - Nếu không trùng ai: Đăng ký nhân vật mới với ID tiếp theo ("SPEAKER_03", "SPEAKER_04"...)
    pub fn match_or_register(
        &mut self,
        voiceprint: &[f32],
        duration_sec: f32,
        episode_name: &str,
        name_hint: Option<&str>,
    ) -> (String, bool, f32) {
        let mut best_id = None;
        let mut max_sim = -1.0f32;

        for (id, profile) in &self.speakers {
            let sim = cosine_similarity(voiceprint, &profile.voiceprint);
            if sim > max_sim {
                max_sim = sim;
                best_id = Some(id.clone());
            }
        }

        if max_sim >= self.similarity_threshold {
            if let Some(matched_id) = best_id {
                let profile = self.speakers.get_mut(&matched_id).unwrap();
                profile.update(voiceprint, duration_sec, episode_name, name_hint);
                return (matched_id, false, max_sim);
            }
        }

        // Tạo nhân vật mới
        let new_id = format!("SPEAKER_{:02}", self.next_speaker_index);
        self.next_speaker_index += 1;

        let profile = SpeakerProfile::new(
            &new_id,
            voiceprint.to_vec(),
            duration_sec,
            episode_name,
            name_hint.map(|s| s.to_string()),
        );
        self.speakers.insert(new_id.clone(), profile);

        (new_id, true, max_sim)
    }
}

/// Tính độ tương đồng Cosine giữa 2 vector đặc trưng
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot.clamp(-1.0, 1.0)
}

/// Chuẩn hóa L2 cho vector
fn normalize_l2(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Trích xuất vector đặc trưng âm sắc (Acoustic Voiceprint) từ các mẫu âm thanh của một nhân vật.
/// Bao gồm 32 băng tần lọc Mel-Frequency + ước lượng cao độ Pitch (F0) + độ biến thiên phổ.
pub fn extract_voiceprint(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.len() < 512 {
        let mut zero = vec![0.0f32; 36];
        zero[0] = 1.0;
        return zero;
    }

    let frame_size = 512; // 32ms ở 16kHz
    let hop_size = 256;   // 16ms
    let num_mel_bins = 32;

    let num_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut mel_energies = vec![0.0f32; num_mel_bins];
    let mut pitches = Vec::with_capacity(num_frames);
    let mut centroids = Vec::with_capacity(num_frames);

    let mel_cutoffs = get_mel_cutoffs(num_mel_bins, sample_rate, frame_size);

    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];

        // 1. Tính biến đổi năng lượng tần số cơ bản (Power Spectrum thô)
        let mut spec = [0.0f32; 256];
        for k in 0..256 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (n, &s) in frame.iter().enumerate().step_by(2) {
                let angle = -2.0 * std::f32::consts::PI * (k * n) as f32 / frame_size as f32;
                re += s * angle.cos();
                im += s * angle.sin();
            }
            spec[k] = (re * re + im * im).sqrt();
        }

        // 2. Gom nhóm theo băng tần Mel
        for bin in 0..num_mel_bins {
            let (b_start, b_end) = mel_cutoffs[bin];
            let mut bin_energy = 0.0f32;
            for k in b_start..b_end.min(256) {
                bin_energy += spec[k];
            }
            mel_energies[bin] += (bin_energy + 1e-6).ln();
        }

        // 3. Ước lượng Pitch cơ bản qua Autocorrelation (dải 80Hz - 400Hz cho giọng người)
        let min_lag = (sample_rate / 400) as usize; // ~40 samples ở 16kHz
        let max_lag = (sample_rate / 80) as usize;  // ~200 samples
        let mut best_corr = 0.0f32;
        let mut best_lag = min_lag;

        for lag in min_lag..max_lag.min(frame_size / 2) {
            let mut corr = 0.0f32;
            for n in 0..(frame_size - lag) {
                corr += frame[n] * frame[n + lag];
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
        let est_f0 = sample_rate as f32 / best_lag as f32;
        pitches.push(est_f0);

        // 4. Trọng tâm phổ (Spectral Centroid)
        let mut num = 0.0f32;
        let mut den = 1e-12f32;
        for (k, &p) in spec.iter().enumerate() {
            num += (k as f32) * p;
            den += p;
        }
        centroids.push(num / den);
    }

    // Trung bình hoá các đặc trưng
    let n_f = num_frames as f32;
    for bin in 0..num_mel_bins {
        mel_energies[bin] /= n_f;
    }

    // Trừ trung bình (Cepstral Mean Subtraction) để triệt tiêu âm lượng tổng thể,
    // chỉ giữ lại hình dạng dải cộng hưởng thanh quản (Formant profile) đặc trưng cho từng người
    let mean_e = mel_energies.iter().sum::<f32>() / num_mel_bins as f32;
    for bin in 0..num_mel_bins {
        mel_energies[bin] -= mean_e;
    }

    let median_pitch = if !pitches.is_empty() {
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (pitches[pitches.len() / 2] - 180.0) / 100.0 // Mean-center quanh 180Hz
    } else {
        0.0
    };

    let mean_centroid = (centroids.iter().sum::<f32>() / n_f - 64.0) / 32.0;

    let mut feature_vec = mel_energies;
    feature_vec.push(median_pitch * 2.0); // Tăng trọng số cao độ pitch để phân biệt nam/nữ/trẻ em rõ rệt
    feature_vec.push(mean_centroid);

    normalize_l2(&mut feature_vec);
    feature_vec
}

/// Tính các ngưỡng chỉ số bin tần số theo thang Mel
fn get_mel_cutoffs(num_bins: usize, sample_rate: u32, frame_size: usize) -> Vec<(usize, usize)> {
    let mut cutoffs = Vec::with_capacity(num_bins);
    let nyquist = sample_rate as f32 / 2.0;
    let min_mel = 2595.0f32 * (1.0f32 + 80.0f32 / 700.0f32).log10();
    let max_mel = 2595.0f32 * (1.0f32 + nyquist / 700.0f32).log10();

    for i in 0..num_bins {
        let mel_low = min_mel + (max_mel - min_mel) * (i as f32 / num_bins as f32);
        let mel_high = min_mel + (max_mel - min_mel) * ((i + 1) as f32 / num_bins as f32);

        let hz_low = 700.0 * (10.0f32.powf(mel_low / 2595.0) - 1.0);
        let hz_high = 700.0 * (10.0f32.powf(mel_high / 2595.0) - 1.0);

        let bin_low = ((hz_low / nyquist) * (frame_size as f32 / 2.0)).floor() as usize;
        let bin_high = ((hz_high / nyquist) * (frame_size as f32 / 2.0)).ceil() as usize;

        cutoffs.push((bin_low.max(1), bin_high.max(bin_low + 1)));
    }
    cutoffs
}

/// Trích xuất voiceprint tổng hợp cho từng speaker từ danh sách lượt nói
pub fn compute_speaker_voiceprints(
    audio_samples: &[f32],
    sample_rate: u32,
    turns: &[SpeakerTurn],
) -> HashMap<String, (Vec<f32>, f32)> {
    let mut speaker_samples: HashMap<String, (Vec<f32>, f32)> = HashMap::new();

    for turn in turns {
        let start_sample = (turn.start_sec * sample_rate as f32) as usize;
        let end_sample = (turn.end_sec * sample_rate as f32) as usize;
        if start_sample < audio_samples.len() {
            let actual_end = end_sample.min(audio_samples.len());
            if actual_end > start_sample {
                let chunk = &audio_samples[start_sample..actual_end];
                let dur = turn.duration();
                let entry = speaker_samples
                    .entry(turn.speaker_id.clone())
                    .or_insert_with(|| (Vec::new(), 0.0));
                entry.0.extend_from_slice(chunk);
                entry.1 += dur;
            }
        }
    }

    let mut result = HashMap::new();
    for (spk, (samples, total_dur)) in speaker_samples {
        let vp = extract_voiceprint(&samples, sample_rate);
        result.insert(spk, (vp, total_dur));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_registry_matching() {
        let mut registry = SpeakerRegistry::new().with_threshold(0.70);

        let voiceprint_a = vec![0.5, 0.5, 0.5, 0.5]; // Speaker A
        let voiceprint_b = vec![0.0, 0.0, 1.0, 0.0]; // Speaker B

        // Tập 1: Thêm Speaker A
        let (id1, is_new1, _) = registry.match_or_register(&voiceprint_a, 10.0, "Ep1", Some("Nhân vật chính"));
        assert_eq!(id1, "SPEAKER_00");
        assert!(is_new1);

        // Tập 2: Speaker A quay lại với biến thiên nhẹ (sim ~0.99 > 0.70)
        let voiceprint_a_slight_variation = vec![0.51, 0.49, 0.50, 0.50];
        let (id2, is_new2, _) = registry.match_or_register(&voiceprint_a_slight_variation, 8.0, "Ep2", None);
        assert_eq!(id2, "SPEAKER_00"); // Khớp đúng SPEAKER_00 từ tập 1!
        assert!(!is_new2); // Không phải nhân vật mới!

        // Tập 2: Xuất hiện Speaker B hoàn toàn mới (sim thấp < 0.70)
        let (id3, is_new3, _) = registry.match_or_register(&voiceprint_b, 5.0, "Ep2", Some("Kẻ thù mới"));
        assert_eq!(id3, "SPEAKER_01"); // Được cấp mã mới SPEAKER_01!
        assert!(is_new3); // Đúng là nhân vật mới!
    }
}
