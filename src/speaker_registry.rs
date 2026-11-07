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
    pub voiceprint: Vec<f32>,                // Vector đặc trưng âm sắc 192-d (L2-normalized)
    pub f0_pitch: f32,                       // Cao độ giọng nói F0 (Hz)
    pub total_duration_sec: f32,             // Tổng thời lượng đã nói qua tất cả các tập
    pub sample_count: usize,                 // Số lượt nói đã học
    pub episode_appearances: Vec<String>,    // Danh sách các tập đã xuất hiện: ["第001集", "第002集"]
}

impl SpeakerProfile {
    pub fn new(
        speaker_id: impl Into<String>,
        voiceprint: Vec<f32>,
        f0_pitch: f32,
        duration_sec: f32,
        episode_name: impl Into<String>,
        display_name: Option<String>,
    ) -> Self {
        Self {
            speaker_id: speaker_id.into(),
            display_name,
            voiceprint,
            f0_pitch,
            total_duration_sec: duration_sec,
            sample_count: 1,
            episode_appearances: vec![episode_name.into()],
        }
    }

    /// Cập nhật voiceprint centroid bằng trung bình động có trọng số thời lượng
    pub fn update(
        &mut self,
        new_voiceprint: &[f32],
        new_f0: f32,
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

            if new_f0 > 60.0 {
                if self.f0_pitch < 60.0 {
                    self.f0_pitch = new_f0;
                } else {
                    self.f0_pitch = self.f0_pitch * w_old + new_f0 * w_new;
                }
            }

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
    pub similarity_threshold: f32, // Ngưỡng tương đồng cosine để xác định cùng 1 người (mặc định 0.42 cho CAM++ 192-d)
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self {
            speakers: HashMap::new(),
            next_speaker_index: 0,
            similarity_threshold: 0.42,
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

    /// Khớp hoặc đăng ký một nhân vật mới dựa trên voiceprint và F0 pitch:
    /// - Nếu độ tương đồng cosine >= threshold: Trả về ID của nhân vật cũ (ví dụ "SPEAKER_01"), cập nhật profile
    /// - Nếu không trùng ai: Đăng ký nhân vật mới với ID tiếp theo ("SPEAKER_03", "SPEAKER_04"...)
    pub fn match_or_register(
        &mut self,
        voiceprint: &[f32],
        f0: f32,
        duration_sec: f32,
        episode_name: &str,
        name_hint: Option<&str>,
    ) -> (String, bool, f32) {
        let mut best_id = None;
        let mut max_score = -1.0f32;

        for (id, profile) in &self.speakers {
            let sim = cosine_similarity(voiceprint, &profile.voiceprint);
            let mut score = sim;

            // Kiểm tra tính tương thích pitch F0
            if profile.f0_pitch > 60.0 && f0 > 60.0 {
                let diff_ratio = (profile.f0_pitch - f0).abs() / profile.f0_pitch.max(f0);
                if diff_ratio > 0.35 {
                    score -= 0.25;
                }
            }

            if score > max_score {
                max_score = score;
                best_id = Some(id.clone());
            }
        }

        if max_score >= self.similarity_threshold {
            if let Some(matched_id) = best_id {
                let profile = self.speakers.get_mut(&matched_id).unwrap();
                profile.update(voiceprint, f0, duration_sec, episode_name, name_hint);
                return (matched_id, false, max_score);
            }
        }

        // Tạo nhân vật mới
        let new_id = format!("SPEAKER_{:02}", self.next_speaker_index);
        self.next_speaker_index += 1;

        let profile = SpeakerProfile::new(
            &new_id,
            voiceprint.to_vec(),
            f0,
            duration_sec,
            episode_name,
            name_hint.map(|s| s.to_string()),
        );
        self.speakers.insert(new_id.clone(), profile);

        (new_id, true, max_score)
    }

    /// Đăng ký trước một nhân vật với tên hiển thị cụ thể
    pub fn register_named(
        &mut self,
        display_name: &str,
        voiceprint: &[f32],
        f0: f32,
        duration_sec: f32,
        episode_name: &str,
    ) -> String {
        // Kiểm tra xem đã có nhân vật mang tên này chưa
        for (id, prof) in self.speakers.iter_mut() {
            if prof.display_name.as_deref() == Some(display_name) {
                prof.update(voiceprint, f0, duration_sec, episode_name, Some(display_name));
                return id.clone();
            }
        }

        let new_id = format!("SPEAKER_{:02}", self.next_speaker_index);
        self.next_speaker_index += 1;

        let profile = SpeakerProfile::new(
            &new_id,
            voiceprint.to_vec(),
            f0,
            duration_sec,
            episode_name,
            Some(display_name.to_string()),
        );
        self.speakers.insert(new_id.clone(), profile);
        new_id
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_registry_matching() {
        let mut registry = SpeakerRegistry::new().with_threshold(0.70);

        let voiceprint_a = vec![0.5, 0.5, 0.5, 0.5];
        let voiceprint_b = vec![0.0, 0.0, 1.0, 0.0];

        let (id1, is_new1, _) = registry.match_or_register(&voiceprint_a, 120.0, 10.0, "Ep1", Some("Nhân vật chính"));
        assert_eq!(id1, "SPEAKER_00");
        assert!(is_new1);

        let voiceprint_a_slight_variation = vec![0.51, 0.49, 0.50, 0.50];
        let (id2, is_new2, _) = registry.match_or_register(&voiceprint_a_slight_variation, 125.0, 8.0, "Ep2", None);
        assert_eq!(id2, "SPEAKER_00");
        assert!(!is_new2);

        let (id3, is_new3, _) = registry.match_or_register(&voiceprint_b, 240.0, 5.0, "Ep2", Some("Kẻ thù mới"));
        assert_eq!(id3, "SPEAKER_01");
        assert!(is_new3);
    }
}
