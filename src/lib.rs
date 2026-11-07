pub mod alignment;
pub mod audio;
pub mod error;
pub mod model;
pub mod progress;
pub mod subtitles;

pub use alignment::{align_subtitles, assign_speaker_to_sub, SpeakerTurn};
pub use audio::{extract_audio_with_ffmpeg, load_media_audio, load_wav, AudioBuffer};
pub use error::VoxError;
pub use model::{ensure_model, VoxConfig, VoxModel, DEFAULT_MODEL_URL};
pub use progress::{ProgressCallback, ProgressInfo, ProgressStage};
pub use subtitles::{
    format_seconds_to_srt_timestamp, format_seconds_to_vtt_timestamp, load_subtitles, parse_srt,
    parse_timestamp_to_seconds, parse_vtt, save_subtitles, serialize_srt, serialize_vtt,
    SpeakerFormat, SubtitleFormat, SubtitleSegment,
};

use std::path::{Path, PathBuf};

/// Bộ máy chính điều phối toàn bộ quá trình Diarization & Phụ đề
pub struct VoxEngine {
    pub config: VoxConfig,
}

impl VoxEngine {
    pub fn new(config: VoxConfig) -> Self {
        Self { config }
    }

    /// Khởi tạo với cấu hình mặc định (model tại models/pyannote_segmentation.onnx, xuất ra thư mục output/)
    pub fn default_engine() -> Self {
        Self::new(VoxConfig::default())
    }

    /// Thực hiện trọn gói quy trình:
    /// 1. Kiểm tra / Tải model ONNX nếu chưa có
    /// 2. Trích xuất audio từ MP4 / Media sang 16kHz WAV mono
    /// 3. Phân đoạn giọng nói (Diarization ONNX inference trên CPU)
    /// 4. Đọc phụ đề VTT/SRT và khớp Speaker (gán nhãn người nói)
    /// 5. Xuất phụ đề mới ra thư mục output chỉ định
    pub fn process_video_and_subtitles<P1, P2, F>(
        &self,
        video_or_audio_path: P1,
        subtitle_path: P2,
        mut progress_cb: Option<F>,
    ) -> Result<PathBuf, VoxError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(&ProgressInfo),
    {
        let video_path = video_or_audio_path.as_ref();
        let sub_path = subtitle_path.as_ref();

        let mut report = |info: ProgressInfo| {
            if let Some(cb) = progress_cb.as_mut() {
                cb(&info);
            }
        };

        // 1. Kiểm tra / Tải mô hình ONNX
        let model_file = Path::new(&self.config.model_path);
        if !model_file.exists() {
            let mut model_cb = |p: &ProgressInfo| report(p.clone());
            ensure_model(
                model_file,
                &self.config.model_url,
                Some(&mut model_cb),
            )?;
        } else {
            report(ProgressInfo::new(
                ProgressStage::CheckingModel,
                1,
                1,
                "file",
                "0s",
                format!("Đã sẵn sàng mô hình: {}", model_file.display()),
            ));
        }

        // 2. Trích xuất Audio
        report(ProgressInfo::new(
            ProgressStage::ExtractingAudio,
            0,
            1,
            "file",
            "...",
            format!("Đang trích xuất audio từ: {}", video_path.display()),
        ));

        let audio_buffer = load_media_audio(video_path)?;

        report(ProgressInfo::new(
            ProgressStage::ExtractingAudio,
            1,
            1,
            "file",
            format!("{:.1}s", audio_buffer.duration_sec),
            format!(
                "Audio sẵn sàng ({:.1}s, 16000Hz PCM)",
                audio_buffer.duration_sec
            ),
        ));

        // 3. Đọc phụ đề VTT / SRT
        let (mut segments, format) = load_subtitles(sub_path)?;

        // 4. Khởi tạo session & Inference ONNX trên CPU
        let mut model = VoxModel::load(model_file, self.config.cpu_threads)?;
        let mut infer_cb = |p: &ProgressInfo| report(p.clone());
        let turns = model.run_diarization(
            &audio_buffer.samples,
            &self.config,
            Some(&mut infer_cb),
        )?;

        // 5. Khớp phụ đề với các Speaker Turns
        report(ProgressInfo::new(
            ProgressStage::AligningSubtitles,
            0,
            segments.len(),
            "cues",
            "0s",
            format!("Đang gắn nhãn người nói cho {} câu phụ đề...", segments.len()),
        ));

        align_subtitles(&mut segments, &turns);

        report(ProgressInfo::new(
            ProgressStage::AligningSubtitles,
            segments.len(),
            segments.len(),
            "cues",
            "0s",
            format!("Gắn nhãn thành công cho {} câu phụ đề", segments.len()),
        ));

        // 6. Xuất ra thư mục output
        let output_dir = &self.config.output_dir;
        std::fs::create_dir_all(output_dir)?;

        let original_file_name = sub_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("output.diarized");

        let ext = match format {
            SubtitleFormat::Vtt => "vtt",
            SubtitleFormat::Srt => "srt",
        };

        let output_filename = if original_file_name.ends_with(&format!(".{}", ext)) {
            let base = &original_file_name[..original_file_name.len() - ext.len() - 1];
            format!("{}.diarized.{}", base, ext)
        } else {
            format!("{}.diarized.{}", original_file_name, ext)
        };

        let output_path = output_dir.join(output_filename);

        report(ProgressInfo::new(
            ProgressStage::Exporting,
            0,
            1,
            "file",
            "...",
            format!("Đang ghi phụ đề ra: {}", output_path.display()),
        ));

        save_subtitles(&segments, &output_path, format, self.config.speaker_format)?;

        report(ProgressInfo::new(
            ProgressStage::Completed,
            1,
            1,
            "file",
            "0s",
            format!("Hoàn tất! Kết quả lưu tại: {}", output_path.display()),
        ));

        Ok(output_path)
    }
}

/// Hàm tiện ích chạy nhanh quá trình Diarization từ đường dẫn 2 file độc lập
pub fn process_diarization<P1, P2, P3, F>(
    video_path: P1,
    sub_path: P2,
    output_dir: P3,
    config: Option<VoxConfig>,
    progress_cb: Option<F>,
) -> Result<PathBuf, VoxError>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    F: FnMut(&ProgressInfo),
{
    let mut cfg = config.unwrap_or_default();
    cfg.output_dir = output_dir.as_ref().to_path_buf();
    let engine = VoxEngine::new(cfg);
    engine.process_video_and_subtitles(video_path, sub_path, progress_cb)
}
