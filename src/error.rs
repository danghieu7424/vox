use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VoxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("Network download error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("WAV format error: {0}")]
    Hound(#[from] hound::Error),

    #[error("FFmpeg error: {0}")]
    Ffmpeg(String),

    #[error("Subtitle parse error: {0}")]
    SubtitleParse(String),

    #[error("Model not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("Model download failed: {0}")]
    ModelDownloadFailed(String),

    #[error("Audio processing error: {0}")]
    AudioProcessing(String),

    #[error("Invalid shape or dimensions in inference: {0}")]
    ShapeError(String),
}

impl<T> From<ort::Error<T>> for VoxError {
    fn from(err: ort::Error<T>) -> Self {
        VoxError::Ort(err.to_string())
    }
}
