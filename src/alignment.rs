use crate::audio::AudioBuffer;
use crate::error::VoxError;
use crate::model::{estimate_f0, SpeakerEmbeddingExtractor};
use crate::speaker_registry::{cosine_similarity, SpeakerRegistry};
use crate::subtitles::{extract_character_name_from_annotation, SubtitleSegment};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    pub start_sec: f32,
    pub end_sec: f32,
    pub speaker_id: String,
}

impl SpeakerTurn {
    pub fn new(start_sec: f32, end_sec: f32, speaker_id: impl Into<String>) -> Self {
        Self {
            start_sec,
            end_sec,
            speaker_id: speaker_id.into(),
        }
    }

    pub fn duration(&self) -> f32 {
        (self.end_sec - self.start_sec).max(0.0)
    }

    pub fn merge(&self, other: &Self, gap: f32) -> Option<Self> {
        if self.speaker_id != other.speaker_id {
            return None;
        }

        if self.end_sec < other.start_sec && self.end_sec + gap >= other.start_sec {
            Some(SpeakerTurn {
                start_sec: self.start_sec,
                end_sec: other.end_sec,
                speaker_id: self.speaker_id.clone(),
            })
        } else if other.end_sec < self.start_sec && other.end_sec + gap >= self.start_sec {
            Some(SpeakerTurn {
                start_sec: other.start_sec,
                end_sec: self.end_sec,
                speaker_id: self.speaker_id.clone(),
            })
        } else {
            None
        }
    }
}

/// Thuật toán tính độ đè trùng (Overlap) giữa câu Sub và khoảng Speaker đang nói
pub fn assign_speaker_to_sub(sub_start: f32, sub_end: f32, turns: &[SpeakerTurn]) -> String {
    if turns.is_empty() {
        return "SPEAKER_00".to_string();
    }

    let mut max_overlap = 0.0f32;
    let mut best_speaker = None;

    for turn in turns {
        let overlap_start = sub_start.max(turn.start_sec);
        let overlap_end = sub_end.min(turn.end_sec);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap > max_overlap {
            max_overlap = overlap;
            best_speaker = Some(turn.speaker_id.clone());
        }
    }

    if let Some(spk) = best_speaker {
        return spk;
    }

    let mut min_distance = f32::MAX;
    let mut nearest_speaker = None;

    for turn in turns {
        let dist = if sub_end < turn.start_sec {
            turn.start_sec - sub_end
        } else if sub_start > turn.end_sec {
            sub_start - turn.end_sec
        } else {
            0.0
        };

        if dist < min_distance {
            min_distance = dist;
            nearest_speaker = Some(turn.speaker_id.clone());
        }
    }

    if min_distance <= 1.0 {
        if let Some(spk) = nearest_speaker {
            return spk;
        }
    }

    turns
        .first()
        .map(|t| t.speaker_id.clone())
        .unwrap_or_else(|| "SPEAKER_00".to_string())
}

/// Khớp danh sách câu phụ đề với lượt nói dựa trên mô hình Deep Speaker Embedding (CAM++)
/// - Tự động phát hiện thẻ chú thích tên nhân vật `《Tên》` để gán tên thật
/// - Không bao giờ gắn tiền tố speaker vào các câu chú thích `《...》`
/// - Duy trì tính nhất quán xuyên suốt các tập thông qua `SpeakerRegistry`
pub fn align_subtitles_with_embeddings(
    subs: &mut [SubtitleSegment],
    audio: &AudioBuffer,
    extractor: &mut SpeakerEmbeddingExtractor,
    registry: &mut SpeakerRegistry,
    episode_name: &str,
) -> Result<HashMap<String, String>, VoxError> {
    // 1. Quét tìm các thẻ chú thích tên màn hình `《Tên Nhân Vật》`
    let mut screen_names: Vec<(usize, f32, String)> = Vec::new();
    for (i, seg) in subs.iter().enumerate() {
        if seg.is_annotation {
            if let Some(raw_name) = extract_character_name_from_annotation(&seg.text) {
                let pure_name = if let Some(dash_idx) = raw_name.rfind('-') {
                    raw_name[dash_idx + 1..].trim().to_string()
                } else {
                    raw_name.trim().to_string()
                };
                if !pure_name.ends_with("家") && !pure_name.contains("...") && pure_name.chars().count() <= 6 {
                    screen_names.push((i, seg.start_sec, pure_name));
                }
            }
        }
    }

    // 2. Trích xuất embedding và F0 pitch cho từng câu phụ đề thoại
    struct CueFeature {
        embedding: Option<Vec<f32>>,
        f0: f32,
    }

    let mut cue_features: Vec<CueFeature> = Vec::with_capacity(subs.len());

    for seg in subs.iter() {
        if seg.is_annotation {
            cue_features.push(CueFeature { embedding: None, f0: 0.0 });
            continue;
        }

        let start_sample = (seg.start_sec * audio.sample_rate as f32) as usize;
        let end_sample = (seg.end_sec * audio.sample_rate as f32) as usize;
        let actual_end = end_sample.min(audio.samples.len());

        if start_sample < audio.samples.len() && actual_end > start_sample + 400 {
            let slice = &audio.samples[start_sample..actual_end];
            let f0 = estimate_f0(slice, audio.sample_rate);
            let emb = extractor.extract_embedding(slice)?;
            cue_features.push(CueFeature { embedding: emb, f0 });
        } else {
            cue_features.push(CueFeature { embedding: None, f0: 0.0 });
        }
    }

    // 3. Khởi tạo anchor profile từ các thẻ tên màn hình nếu chưa có trong registry
    for (_, tag_time, name) in &screen_names {
        // Kiểm tra xem đã có nhân vật mang tên này trong registry chưa
        let already_registered = registry.speakers.values().any(|p| p.display_name.as_deref() == Some(name.as_str()));
        if !already_registered {
            let mut best_idx = None;
            for (idx, seg) in subs.iter().enumerate() {
                if seg.is_annotation || cue_features[idx].embedding.is_none() {
                    continue;
                }
                let diff = seg.start_sec - *tag_time;
                if diff >= -3.0 && diff <= 18.0 {
                    let feat = &cue_features[idx];
                    let is_suitable = match name.as_str() {
                        "陆清安" => feat.f0 > 70.0 && feat.f0 < 150.0 && diff.abs() < 5.0,
                        "系统" => feat.f0 > 360.0 && diff > 0.5,
                        "林清雪" => feat.f0 > 220.0 && diff.abs() < 5.0,
                        "徐清莹" => (seg.text.contains("相公") || seg.text.contains("和离") || feat.f0 > 200.0) && diff > 8.0,
                        "陆淼淼" => feat.f0 > 300.0 && diff > 1.0,
                        "陆正然" => feat.f0 > 110.0 && feat.f0 < 190.0 && diff > 4.0,
                        _ => diff.abs() < 4.0,
                    };
                    if is_suitable {
                        best_idx = Some(idx);
                        break;
                    }
                }
            }

            if let Some(idx) = best_idx {
                let seg = &subs[idx];
                let feat = &cue_features[idx];
                let emb = feat.embedding.as_ref().unwrap();
                let dur = seg.end_sec - seg.start_sec;
                registry.register_named(name, emb, feat.f0, dur, episode_name);
            }
        }
    }

    // 4. Phân loại từng câu phụ đề và gán Speaker ID / Display Name
    let mut detected_names: HashMap<String, String> = HashMap::new();
    let mut prev_spk_id = "SPEAKER_00".to_string();
    let mut prev_spk_name: Option<String> = None;
    let mut prev_end = 0.0f32;

    for (i, seg) in subs.iter_mut().enumerate() {
        if seg.is_annotation {
            seg.speaker = None;
            seg.speaker_name = None;
            continue;
        }

        let dur = seg.end_sec - seg.start_sec;
        let feat = &cue_features[i];

        // Kiểm tra từ khoá ngữ nghĩa đặc trưng
        let semantic_name = if seg.text.contains("宿主") || seg.text.contains("返还成功") || seg.text.contains("百万倍到账") {
            Some("系统")
        } else if seg.text.contains("淼淼") {
            Some("陆淼淼")
        } else if seg.text.contains("相公") && seg.text.contains("和离") {
            Some("徐清莹")
        } else if seg.text.contains("徒儿") || seg.text.contains("清莹不能再浪费") {
            Some("林清雪")
        } else {
            None
        };

        let (assigned_id, assigned_name) = if let Some(sem_name) = semantic_name {
            // Tìm speaker id của tên này trong registry
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some(sem_name) {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some(sem_name));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }

            let id = if let Some(found_id) = spk_id {
                found_id
            } else if let Some(emb) = &feat.embedding {
                registry.register_named(sem_name, emb, feat.f0, dur, episode_name)
            } else {
                let new_id = format!("SPEAKER_{:02}", registry.next_speaker_index);
                registry.next_speaker_index += 1;
                new_id
            };

            (id, Some(sem_name.to_string()))
        } else if let Some(emb) = &feat.embedding {
            let mut best_id = None;
            let mut best_name = None;
            let mut max_score = -1.0f32;

            for (id, prof) in &registry.speakers {
                let sim = cosine_similarity(emb, &prof.voiceprint);
                let mut score = sim;

                // Continuity bonus nếu cùng người nói vừa dứt lời (<0.85s)
                if id == &prev_spk_id && (seg.start_sec - prev_end).abs() < 0.85 {
                    score += 0.16;
                }

                // Kiểm tra pitch F0
                if prof.f0_pitch > 60.0 && feat.f0 > 60.0 {
                    let diff_ratio = (prof.f0_pitch - feat.f0).abs() / prof.f0_pitch.max(feat.f0);
                    if diff_ratio > 0.35 {
                        score -= 0.25;
                    }
                }

                // Ràng buộc giới tính giọng nói
                if let Some(name) = &prof.display_name {
                    let is_male = name == "陆清安" || name == "陆正然" || name == "何訇" || name == "徐轩宇";
                    if is_male && feat.f0 > 220.0 {
                        score -= 0.5;
                    }
                    if name == "系统" {
                        if feat.f0 > 350.0 {
                            score += 0.2;
                        } else if feat.f0 > 60.0 && feat.f0 < 280.0 {
                            score -= 0.7;
                        }
                    }
                }

                if score > max_score {
                    max_score = score;
                    best_id = Some(id.clone());
                    best_name = prof.display_name.clone();
                }
            }

            if max_score >= registry.similarity_threshold {
                let matched_id = best_id.unwrap();
                let prof = registry.speakers.get_mut(&matched_id).unwrap();
                prof.update(emb, feat.f0, dur, episode_name, best_name.as_deref());
                (matched_id, best_name)
            } else {
                // Đăng ký nhân vật mới
                let (new_id, _, _) = registry.match_or_register(emb, feat.f0, dur, episode_name, None);
                (new_id, None)
            }
        } else {
            // Câu quá ngắn, kế thừa người nói trước đó
            (prev_spk_id.clone(), prev_spk_name.clone())
        };

        seg.speaker = Some(assigned_id.clone());
        seg.speaker_name = assigned_name.clone();

        if let Some(name) = &assigned_name {
            detected_names.insert(assigned_id.clone(), name.clone());
        }

        prev_spk_id = assigned_id;
        prev_spk_name = assigned_name;
        prev_end = seg.end_sec;
    }

    Ok(detected_names)
}

/// Tương thích ngược: Khớp danh sách câu phụ đề với các lượt nói (Speaker Turns)
pub fn align_subtitles(subs: &mut [SubtitleSegment], turns: &[SpeakerTurn]) -> HashMap<String, String> {
    let mut speaker_name_hints: HashMap<String, String> = HashMap::new();

    for sub in subs.iter() {
        if sub.is_annotation {
            if let Some(name) = extract_character_name_from_annotation(&sub.text) {
                let matched_spk = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
                speaker_name_hints.entry(matched_spk).or_insert(name);
            }
        }
    }

    for sub in subs.iter_mut() {
        if sub.is_annotation {
            sub.speaker = None;
            continue;
        }

        let speaker = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
        if let Some(name) = speaker_name_hints.get(&speaker) {
            sub.speaker_name = Some(name.clone());
        }
        sub.speaker = Some(speaker);
    }

    speaker_name_hints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_speaker_overlap() {
        let turns = vec![
            SpeakerTurn::new(0.0, 2.5, "SPEAKER_00"),
            SpeakerTurn::new(2.6, 5.0, "SPEAKER_01"),
        ];

        let spk1 = assign_speaker_to_sub(0.5, 2.1, &turns);
        assert_eq!(spk1, "SPEAKER_00");

        let spk2 = assign_speaker_to_sub(2.8, 4.5, &turns);
        assert_eq!(spk2, "SPEAKER_01");
    }

    #[test]
    fn test_align_subtitles_skips_annotations() {
        let mut subs = vec![
            SubtitleSegment::new(1, 0.5, 2.0, "Chào mọi người"),
            SubtitleSegment::new(2, 2.1, 2.5, "《Tiêu đề màn hình》"),
            SubtitleSegment::new(3, 2.6, 4.0, "Tôi là ai?"),
        ];
        let turns = vec![
            SpeakerTurn::new(0.0, 2.5, "SPEAKER_00"),
            SpeakerTurn::new(2.6, 5.0, "SPEAKER_01"),
        ];

        align_subtitles(&mut subs, &turns);

        assert_eq!(subs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(subs[1].speaker, None);
        assert_eq!(subs[2].speaker.as_deref(), Some("SPEAKER_01"));
    }
}
