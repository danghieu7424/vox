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

    // Ghép cặp video và subtitle dựa trên tên tập (file stem)
    let mut pairs = Vec::new();

    for vid in &videos {
        let vid_stem = vid.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if vid_stem.is_empty() {
            continue;
        }

        // Tìm sub có stem tương đồng nhất (ví dụ: "第001集" khớp với "第001集.zh" hoặc "第001集")
        let matched_sub = subtitles.iter().find(|sub| {
            let sub_name = sub.file_name().and_then(|s| s.to_str()).unwrap_or("");
            sub_name.starts_with(vid_stem)
        });

        if let Some(sub) = matched_sub {
            pairs.push(EpisodePair {
                episode_name: vid_stem.to_string(),
                video_path: vid.clone(),
                subtitle_path: sub.clone(),
            });
        }
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

