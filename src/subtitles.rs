use crate::error::VoxError;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Vtt,
    Srt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeakerFormat {
    #[default]
    Prefix,      // [SPEAKER_00]: Text
    BracketOnly, // [SPEAKER_00] Text
    VoiceTag,    // <v SPEAKER_00>Text</v> (Web VTT standard)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleSegment {
    pub index: usize,
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub speaker: Option<String>,
    pub settings: Option<String>,
}

impl SubtitleSegment {
    pub fn new(index: usize, start_sec: f32, end_sec: f32, text: impl Into<String>) -> Self {
        Self {
            index,
            start_sec,
            end_sec,
            text: text.into(),
            speaker: None,
            settings: None,
        }
    }

    pub fn with_settings(mut self, settings: impl Into<String>) -> Self {
        self.settings = Some(settings.into());
        self
    }

    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    pub fn formatted_text(&self, format: SpeakerFormat) -> String {
        match &self.speaker {
            Some(spk) => match format {
                SpeakerFormat::Prefix => format!("[{}]: {}", spk, self.text),
                SpeakerFormat::BracketOnly => format!("[{}] {}", spk, self.text),
                SpeakerFormat::VoiceTag => format!("<v {}>{}</v>", spk, self.text),
            },
            None => self.text.clone(),
        }
    }
}

pub fn parse_timestamp_to_seconds(ts: &str) -> Result<f32, VoxError> {
    let ts = ts.trim();
    // Chấp nhận cả dấu phẩy (SRT: 00:00:01,050) và dấu chấm (VTT: 00:00:01.050)
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() == 3 {
        let hours: f32 = parts[0]
            .parse()
            .map_err(|_| VoxError::SubtitleParse(format!("Invalid hour: {}", parts[0])))?;
        let minutes: f32 = parts[1]
            .parse()
            .map_err(|_| VoxError::SubtitleParse(format!("Invalid minute: {}", parts[1])))?;
        let sec_part = parts[2].replace(',', ".");
        let seconds: f32 = sec_part
            .parse()
            .map_err(|_| VoxError::SubtitleParse(format!("Invalid second: {}", parts[2])))?;
        Ok(hours * 3600.0 + minutes * 60.0 + seconds)
    } else if parts.len() == 2 {
        let minutes: f32 = parts[0]
            .parse()
            .map_err(|_| VoxError::SubtitleParse(format!("Invalid minute: {}", parts[0])))?;
        let sec_part = parts[1].replace(',', ".");
        let seconds: f32 = sec_part
            .parse()
            .map_err(|_| VoxError::SubtitleParse(format!("Invalid second: {}", parts[1])))?;
        Ok(minutes * 60.0 + seconds)
    } else {
        Err(VoxError::SubtitleParse(format!("Unsupported timestamp: {}", ts)))
    }
}

pub fn format_seconds_to_vtt_timestamp(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

pub fn format_seconds_to_srt_timestamp(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

pub fn parse_vtt(content: &str) -> Result<Vec<SubtitleSegment>, VoxError> {
    let mut segments = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    let mut cue_idx = 1;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("WEBVTT")
            || line.starts_with("NOTE")
            || line.starts_with("STYLE")
            || line.starts_with("REGION")
        {
            i += 1;
            continue;
        }

        // Kiểm tra dòng timestamp "-->"
        let timestamp_line = if line.contains("-->") {
            line
        } else if i + 1 < lines.len() && lines[i + 1].contains("-->") {
            i += 1;
            lines[i].trim()
        } else {
            i += 1;
            continue;
        };

        let parts: Vec<&str> = timestamp_line.split("-->").collect();
        if parts.len() != 2 {
            i += 1;
            continue;
        }

        let start_str = parts[0].trim();
        let end_and_settings = parts[1].trim();

        let mut end_parts = end_and_settings.split_whitespace();
        let end_str = match end_parts.next() {
            Some(s) => s,
            None => {
                i += 1;
                continue;
            }
        };

        let settings = end_parts.collect::<Vec<&str>>().join(" ");
        let settings = if settings.is_empty() {
            None
        } else {
            Some(settings)
        };

        let start_sec = parse_timestamp_to_seconds(start_str)?;
        let end_sec = parse_timestamp_to_seconds(end_str)?;

        // Đọc nội dung text phía sau
        i += 1;
        let mut text_lines = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() && !lines[i].contains("-->") {
            text_lines.push(lines[i].trim());
            i += 1;
        }

        let text = text_lines.join("\n");
        if !text.is_empty() {
            let mut seg = SubtitleSegment::new(cue_idx, start_sec, end_sec, text);
            seg.settings = settings;
            segments.push(seg);
            cue_idx += 1;
        }
    }

    Ok(segments)
}

pub fn parse_srt(content: &str) -> Result<Vec<SubtitleSegment>, VoxError> {
    let mut segments = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    let mut cue_idx = 1;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            continue;
        }

        let timestamp_line = if line.contains("-->") {
            line
        } else if i + 1 < lines.len() && lines[i + 1].contains("-->") {
            i += 1;
            lines[i].trim()
        } else {
            i += 1;
            continue;
        };

        let parts: Vec<&str> = timestamp_line.split("-->").collect();
        if parts.len() != 2 {
            i += 1;
            continue;
        }

        let start_str = parts[0].trim();
        let end_str = parts[1].trim();

        let start_sec = parse_timestamp_to_seconds(start_str)?;
        let end_sec = parse_timestamp_to_seconds(end_str)?;

        i += 1;
        let mut text_lines = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() && !lines[i].contains("-->") {
            text_lines.push(lines[i].trim());
            i += 1;
        }

        let text = text_lines.join("\n");
        if !text.is_empty() {
            segments.push(SubtitleSegment::new(cue_idx, start_sec, end_sec, text));
            cue_idx += 1;
        }
    }

    Ok(segments)
}

pub fn load_subtitles<P: AsRef<Path>>(path: P) -> Result<(Vec<SubtitleSegment>, SubtitleFormat), VoxError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .map_err(|e| VoxError::Io(std::io::Error::new(e.kind(), format!("Failed to read subtitle file {}: {}", path.display(), e))))?;

    let is_vtt = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        ext.eq_ignore_ascii_case("vtt")
    } else {
        content.starts_with("WEBVTT")
    };

    if is_vtt {
        let segments = parse_vtt(&content)?;
        Ok((segments, SubtitleFormat::Vtt))
    } else {
        let segments = parse_srt(&content)?;
        Ok((segments, SubtitleFormat::Srt))
    }
}

pub fn serialize_vtt(segments: &[SubtitleSegment], format: SpeakerFormat) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        let start_str = format_seconds_to_vtt_timestamp(seg.start_sec);
        let end_str = format_seconds_to_vtt_timestamp(seg.end_sec);
        let settings_str = seg.settings.as_deref().map(|s| format!(" {}", s)).unwrap_or_default();
        out.push_str(&format!("{} --> {}{}\n", start_str, end_str, settings_str));
        out.push_str(&seg.formatted_text(format));
        out.push_str("\n\n");
    }
    out
}

pub fn serialize_srt(segments: &[SubtitleSegment], format: SpeakerFormat) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        let start_str = format_seconds_to_srt_timestamp(seg.start_sec);
        let end_str = format_seconds_to_srt_timestamp(seg.end_sec);
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", start_str, end_str));
        out.push_str(&seg.formatted_text(format));
        out.push_str("\n\n");
    }
    out
}

pub fn save_subtitles<P: AsRef<Path>>(
    segments: &[SubtitleSegment],
    path: P,
    format: SubtitleFormat,
    speaker_format: SpeakerFormat,
) -> Result<(), VoxError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let serialized = match format {
        SubtitleFormat::Vtt => serialize_vtt(segments, speaker_format),
        SubtitleFormat::Srt => serialize_srt(segments, speaker_format),
    };

    let mut file = File::create(path)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vtt() {
        let vtt_data = r#"WEBVTT

00:00:01.050 --> 00:00:03.050
我在魔域拼杀万年

00:00:03.150 --> 00:00:04.650
成就最強魔帝

00:00:03.207 --> 00:00:05.126 position:50% line:20% align:center
《陆清安》
"#;
        let segs = parse_vtt(vtt_data).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].text, "我在魔域拼杀万年");
        assert!((segs[0].start_sec - 1.050).abs() < 1e-4);
        assert!((segs[0].end_sec - 3.050).abs() < 1e-4);
        assert_eq!(segs[2].settings.as_deref(), Some("position:50% line:20% align:center"));
    }

    #[test]
    fn test_parse_srt() {
        let srt_data = r#"1
00:00:01,050 --> 00:00:03,050
Xin chào các bạn.

2
00:00:03,500 --> 00:00:05,200
Hôm nay chúng ta họp.
"#;
        let segs = parse_srt(srt_data).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Xin chào các bạn.");
        assert!((segs[0].start_sec - 1.050).abs() < 1e-4);
        assert!((segs[1].end_sec - 5.200).abs() < 1e-4);
    }

    #[test]
    fn test_speaker_formatting() {
        let seg = SubtitleSegment::new(1, 1.0, 2.5, "Hello")
            .with_speaker("SPEAKER_01");

        assert_eq!(seg.formatted_text(SpeakerFormat::Prefix), "[SPEAKER_01]: Hello");
        assert_eq!(seg.formatted_text(SpeakerFormat::BracketOnly), "[SPEAKER_01] Hello");
        assert_eq!(seg.formatted_text(SpeakerFormat::VoiceTag), "<v SPEAKER_01>Hello</v>");
    }
}
