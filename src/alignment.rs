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

/// Thuật toán tính độ đè trùng (Overlap) giữa câu Sub và khoảng Speaker đang nói theo đúng plan.md
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

    // Nếu không overlap trực tiếp, tìm turn gần nhất trong khoảng dung sai 1.0 giây
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

    // Mặc định gán speaker đầu tiên nếu có
    turns
        .first()
        .map(|t| t.speaker_id.clone())
        .unwrap_or_else(|| "SPEAKER_00".to_string())
}

/// Khớp danh sách câu phụ đề với các lượt nói (Speaker Turns)
/// - Bỏ qua các câu là chú thích (như `《陆清安》`, `《系统》`) để giữ nguyên văn bản gốc
/// - Tự động phát hiện tên nhân vật từ các thẻ chú thích `《Tên Nhân Vật》` xuất hiện gần lượt nói
pub fn align_subtitles(subs: &mut [SubtitleSegment], turns: &[SpeakerTurn]) -> HashMap<String, String> {
    let mut speaker_name_hints: HashMap<String, String> = HashMap::new();

    // 1. Quét tìm các thẻ tên màn hình `《Tên》` để gán vào speaker đang xuất hiện tại thời điểm đó
    for sub in subs.iter() {
        if sub.is_annotation {
            if let Some(name) = extract_character_name_from_annotation(&sub.text) {
                // Tìm speaker đang nói trong khoảng thời gian của thẻ tên này (+/- 2 giây)
                let matched_spk = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
                if !speaker_name_hints.contains_key(&matched_spk) {
                    speaker_name_hints.insert(matched_spk, name);
                }
            }
        }
    }

    // 2. Gán người nói cho các câu lời thoại thực tế
    for sub in subs.iter_mut() {
        if sub.is_annotation {
            // Chú thích thì không gán nhãn người nói
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
        assert_eq!(subs[1].speaker, None); // Chú thích không có speaker
        assert_eq!(subs[2].speaker.as_deref(), Some("SPEAKER_01"));
    }
}
