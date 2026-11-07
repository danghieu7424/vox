use crate::error::VoxError;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration_sec: f32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        let duration_sec = samples.len() as f32 / sample_rate as f32;
        Self {
            samples,
            sample_rate,
            duration_sec,
        }
    }
}

/// Đọc trực tiếp file WAV bằng `hound`. Hỗ trợ 16-bit PCM, 32-bit float, v.v.
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer, VoxError> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    // Chuyển stereo -> mono nếu có nhiều hơn 1 channel
    let mono_samples = if channels > 1 {
        let mut mono = Vec::with_capacity(raw_samples.len() / channels);
        for chunk in raw_samples.chunks_exact(channels) {
            let sum: f32 = chunk.iter().sum();
            mono.push(sum / channels as f32);
        }
        mono
    } else {
        raw_samples
    };

    // Nếu sample_rate khác 16000Hz, resample tuyến tính về 16000Hz
    let final_samples = if sample_rate != 16000 {
        resample_linear(&mono_samples, sample_rate, 16000)
    } else {
        mono_samples
    };

    Ok(AudioBuffer::new(final_samples, 16000))
}

/// Resample tuyến tính đơn giản cho audio PCM
fn resample_linear(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / target_rate as f64;
    let out_len = (samples.len() as f64 / ratio).round() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;

        if idx0 >= samples.len() {
            break;
        }

        let s0 = samples[idx0];
        let s1 = if idx0 + 1 < samples.len() {
            samples[idx0 + 1]
        } else {
            s0
        };

        output.push(s0 + frac * (s1 - s0));
    }

    output
}

/// Trích xuất audio từ file video/audio bất kỳ (MP4, MKV, MP3, ...) sang WAV 16kHz mono PCM bằng FFmpeg
pub fn extract_audio_with_ffmpeg<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_wav_path: P2,
) -> Result<(), VoxError> {
    let input = input_path.as_ref();
    let output = output_wav_path.as_ref();

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(input)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg(output)
        .status()
        .map_err(|e| {
            VoxError::Ffmpeg(format!(
                "Failed to execute FFmpeg. Is FFmpeg installed and in PATH? Error: {}",
                e
            ))
        })?;

    if !status.success() {
        return Err(VoxError::Ffmpeg(format!(
            "FFmpeg exited with non-zero exit status: {:?}",
            status.code()
        )));
    }

    Ok(())
}

/// Nạp audio từ file media (hỗ trợ .wav trực tiếp hoặc tự động trích xuất từ .mp4 / .mkv / v.v.)
/// Tự động ưu tiên file vocal-separated `<stem>.vox.wav` nếu tồn tại trong cùng thư mục
pub fn load_media_audio<P: AsRef<Path>>(media_path: P) -> Result<AudioBuffer, VoxError> {
    let path = media_path.as_ref();

    // Kiểm tra xem có file vocal track (.vox.wav) cùng thư mục không
    let vox_wav_path = if let Some(parent) = path.parent() {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            let pure_stem = stem.trim_end_matches(".vox");
            let candidate = parent.join(format!("{}.vox.wav", pure_stem));
            if candidate.exists() {
                Some(candidate)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let actual_path = if let Some(v_path) = vox_wav_path {
        v_path
    } else {
        path.to_path_buf()
    };

    if !actual_path.exists() {
        return Err(VoxError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Media file not found: {}", actual_path.display()),
        )));
    }

    let is_wav = actual_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        load_wav(&actual_path)
    } else {
        // Tạo temp file WAV để trích xuất bằng FFmpeg
        let temp_dir = std::env::temp_dir();
        let stem = actual_path.file_stem().and_then(|s| s.to_str()).unwrap_or("temp");
        let temp_wav: PathBuf = temp_dir.join(format!("vox_temp_{}_{}.wav", stem, std::process::id()));

        extract_audio_with_ffmpeg(&actual_path, &temp_wav)?;
        let audio_buffer = load_wav(&temp_wav);
        let _ = std::fs::remove_file(&temp_wav); // Dọn dẹp temp
        audio_buffer
    }
}

