use crate::error::VoxError;
use std::path::{Path, PathBuf};

/// Cặp file đầu vào của 1 tập phim (Video + Phụ đề)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodePair {
    pub episode_name: String,
    pub video_path: PathBuf,
    pub subtitle_path: PathBuf,
}

/// Tự động quét một thư mục chứa nhiều tập phim, ghép cặp file video (MP4/MKV) với file phụ đề (VTT/SRT) tương ứng
pub fn scan_series_directory<P: AsRef<Path>>(dir: P) -> Result<Vec<EpisodePair>, VoxError> {
    let dir_path = dir.as_ref();
    if !dir_path.is_dir() {
        return Err(VoxError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Directory not found: {}", dir_path.display()),
        )));
    }

    let mut videos: Vec<PathBuf> = Vec::new();
    let mut subtitles: Vec<PathBuf> = Vec::new();

    let video_extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm", "wav", "mp3"];
    let subtitle_extensions = ["vtt", "srt"];

    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()) {
                if video_extensions.contains(&ext.as_str()) {
                    videos.push(path);
                } else if subtitle_extensions.contains(&ext.as_str()) {
                    subtitles.push(path);
                }
            }
        }
    }

    // Ghép cặp video/audio và subtitle dựa trên tên tập (file stem)
    // Tự động ưu tiên file vocal-separated .vox.wav nếu có
    let mut episode_map: std::collections::HashMap<String, (PathBuf, PathBuf)> = std::collections::HashMap::new();

    for sub in &subtitles {
        let sub_name = sub.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // Trích xuất mã tập (ví dụ "第001集.zh.vtt" -> "第001集")
        let ep_name = if let Some(dot_idx) = sub_name.find('.') {
            &sub_name[..dot_idx]
        } else {
            sub_name
        };

        // Tìm audio/video tương ứng cho tập này
        // Ưu tiên 1: <ep_name>.vox.wav
        // Ưu tiên 2: <ep_name>.wav
        // Ưu tiên 3: <ep_name>.mp4 / .mkv / video khác
        let mut best_media = None;
        let mut best_prio = 0; // 3: vox.wav, 2: wav, 1: other

        for vid in &videos {
            let vid_name = vid.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if vid_name.starts_with(ep_name) {
                let prio = if vid_name.ends_with(".vox.wav") {
                    3
                } else if vid_name.ends_with(".wav") {
                    2
                } else {
                    1
                };

                if prio > best_prio {
                    best_prio = prio;
                    best_media = Some(vid.clone());
                }
            }
        }

        if let Some(media) = best_media {
            episode_map.insert(ep_name.to_string(), (media, sub.clone()));
        }
    }

    let mut pairs = Vec::new();
    for (ep_name, (media, sub)) in episode_map {
        pairs.push(EpisodePair {
            episode_name: ep_name,
            video_path: media,
            subtitle_path: sub,
        });
    }

    // Sắp xếp các tập theo thứ tự tên tự nhiên (Tập 1, Tập 2...)
    pairs.sort_by(|a, b| a.episode_name.cmp(&b.episode_name));

    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_series() {
        let dir = Path::new("input");
        if dir.exists() {
            let pairs = scan_series_directory(dir).unwrap();
            println!("Found {} episode pairs in input:", pairs.len());
            for p in &pairs {
                println!("  • {} -> {:?} & {:?}", p.episode_name, p.video_path.file_name(), p.subtitle_path.file_name());
            }
            assert!(pairs.len() >= 2);
        }
    }
}

