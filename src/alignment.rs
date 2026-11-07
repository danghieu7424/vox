use crate::subtitles::SubtitleSegment;

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

    // Nếu không overlap trực tiếp (ví dụ subtitle xuất hiện sớm hơn hoặc trễ hơn chút ít),
    // tìm turn gần nhất trong khoảng dung sai 1.0 giây
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
pub fn align_subtitles(subs: &mut [SubtitleSegment], turns: &[SpeakerTurn]) {
    for sub in subs.iter_mut() {
        let speaker = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
        sub.speaker = Some(speaker);
    }
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

        // Sub từ 0.5 đến 2.1 đè trùng hoàn toàn vào SPEAKER_00
        let spk1 = assign_speaker_to_sub(0.5, 2.1, &turns);
        assert_eq!(spk1, "SPEAKER_00");

        // Sub từ 2.8 đến 4.5 đè trùng hoàn toàn vào SPEAKER_01
        let spk2 = assign_speaker_to_sub(2.8, 4.5, &turns);
        assert_eq!(spk2, "SPEAKER_01");
    }
}
