pub mod alignment;
pub mod audio;
pub mod error;
pub mod model;
pub mod progress;
pub mod series;
pub mod speaker_registry;
pub mod subtitles;

pub use alignment::{align_subtitles, assign_speaker_to_sub, SpeakerTurn};
pub use audio::{extract_audio_with_ffmpeg, load_media_audio, load_wav, AudioBuffer};
pub use error::VoxError;
pub use model::{ensure_model, VoxConfig, VoxModel, DEFAULT_MODEL_URL};
pub use progress::{ProgressCallback, ProgressInfo, ProgressStage};
pub use series::{scan_series_directory, EpisodePair};
pub use speaker_registry::{
    compute_speaker_voiceprints, extract_voiceprint, SpeakerProfile, SpeakerRegistry,
};
pub use subtitles::{
    extract_character_name_from_annotation, format_seconds_to_srt_timestamp,
    format_seconds_to_vtt_timestamp, is_annotation_text, load_subtitles, parse_srt,
    parse_timestamp_to_seconds, parse_vtt, save_subtitles, serialize_srt, serialize_vtt,
    SpeakerFormat, SubtitleFormat, SubtitleSegment,
};

use std::collections::HashMap;
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

    /// Thực hiện trọn gói quy trình cho 1 tập đơn lẻ:
    /// 1. Kiểm tra / Tải model ONNX nếu chưa có
    /// 2. Trích xuất audio từ MP4 / Media sang 16kHz WAV mono
    /// 3. Phân đoạn giọng nói (Diarization ONNX inference trên CPU)
    /// 4. Đọc phụ đề VTT/SRT và khớp Speaker (gán nhãn người nói, bỏ qua chú thích 《...》)
    /// 5. Xuất phụ đề mới ra thư mục output chỉ định
    pub fn process_video_and_subtitles<P1, P2, F>(
        &self,
        video_or_audio_path: P1,
        subtitle_path: P2,
        progress_cb: Option<F>,
    ) -> Result<PathBuf, VoxError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(&ProgressInfo),
    {
        let mut registry = SpeakerRegistry::new();
        self.process_episode_with_registry(
            video_or_audio_path,
            subtitle_path,
            &mut registry,
            progress_cb,
        )
    }

    /// Xử lý 1 tập phim với Sổ bộ nhân vật (SpeakerRegistry) dùng chung qua các tập.
    /// Giúp nhân vật cũ giữ nguyên mã ID (ví dụ SPEAKER_01) và nhân vật mới được cấp mã mới.
    pub fn process_episode_with_registry<P1, P2, F>(
        &self,
        video_or_audio_path: P1,
        subtitle_path: P2,
        registry: &mut SpeakerRegistry,
        mut progress_cb: Option<F>,
    ) -> Result<PathBuf, VoxError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(&ProgressInfo),
    {
        let video_path = video_or_audio_path.as_ref();
        let sub_path = subtitle_path.as_ref();
        let episode_name = video_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("episode");

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
        let mut local_turns = model.run_diarization(
            &audio_buffer.samples,
            &self.config,
            Some(&mut infer_cb),
        )?;

        // 5. Trích xuất Voiceprints & Khớp với SpeakerRegistry toàn cục (Cross-Episode Matching)
        let local_voiceprints = compute_speaker_voiceprints(
            &audio_buffer.samples,
            audio_buffer.sample_rate,
            &local_turns,
        );

        // Quét trước các thẻ chú thích `《Tên Nhân Vật》` gần lượt nói để thu thập gợi ý tên
        let mut local_name_hints: HashMap<String, String> = HashMap::new();
        for sub in &segments {
            if sub.is_annotation {
                if let Some(name) = extract_character_name_from_annotation(&sub.text) {
                    let matched_local_spk = assign_speaker_to_sub(sub.start_sec, sub.end_sec, &local_turns);
                    local_name_hints.entry(matched_local_spk).or_insert(name);
                }
            }
        }

        // Ánh xạ từ Local Speaker (trong tập) sang Global Speaker (xuyên suốt bộ phim)
        let mut local_to_global: HashMap<String, String> = HashMap::new();
        for (local_spk, (vp, dur)) in &local_voiceprints {
            let name_hint = local_name_hints.get(local_spk).map(|s| s.as_str());
            let (global_id, is_new, sim) = registry.match_or_register(
                vp,
                *dur,
                episode_name,
                name_hint,
            );

            let status_msg = if is_new {
                format!("Nhân vật MỚI: {} (gợi ý: {:?})", global_id, name_hint)
            } else {
                format!("Nhân vật CŨ: {} (độ khớp: {:.1}%)", global_id, sim * 100.0)
            };
            report(ProgressInfo::new(
                ProgressStage::AligningSubtitles,
                local_to_global.len() + 1,
                local_voiceprints.len(),
                "speakers",
                "0s",
                status_msg,
            ));

            local_to_global.insert(local_spk.clone(), global_id);
        }

        // Cập nhật lại speaker_id trong turns theo mã Global
        for turn in local_turns.iter_mut() {
            if let Some(global_id) = local_to_global.get(&turn.speaker_id) {
                turn.speaker_id = global_id.clone();
            }
        }

        // 6. Khớp phụ đề với các Speaker Turns (Bỏ qua các câu chú thích 《...》)
        report(ProgressInfo::new(
            ProgressStage::AligningSubtitles,
            0,
            segments.len(),
            "cues",
            "0s",
            format!("Đang gắn nhãn người nói cho {} câu phụ đề...", segments.len()),
        ));

        let detected_names = align_subtitles(&mut segments, &local_turns);

        // Gán display_name từ registry nếu phụ đề chưa có
        for sub in segments.iter_mut() {
            if !sub.is_annotation {
                if sub.speaker_name.is_none() {
                    if let Some(spk_id) = &sub.speaker {
                        if let Some(profile) = registry.speakers.get(spk_id) {
                            sub.speaker_name = profile.display_name.clone();
                        }
                    }
                }
            }
        }

        report(ProgressInfo::new(
            ProgressStage::AligningSubtitles,
            segments.len(),
            segments.len(),
            "cues",
            "0s",
            format!(
                "Gắn nhãn thành công cho {} câu phụ đề (nhận diện {} tên nhân vật)",
                segments.len(),
                detected_names.len()
            ),
        ));

        // 7. Xuất ra thư mục output
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

    /// Xử lý trọn bộ cả thư mục phim gồm nhiều tập:
    /// - Tự động quét và ghép cặp video + phụ đề từng tập
    /// - Quản lý nhân vật xuyên suốt các tập (nhân vật cũ giữ nguyên ID, nhân vật mới cấp ID mới)
    /// - Xuất phụ đề đã gắn nhãn cho từng tập vào output_dir
    /// - Lưu `speaker_registry.json` chứa danh sách hồ sơ toàn bộ nhân vật của phim
    pub fn process_series_directory<P1, P2, F>(
        &self,
        input_dir: P1,
        output_dir: P2,
        mut progress_cb: Option<F>,
    ) -> Result<Vec<PathBuf>, VoxError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(&ProgressInfo),
    {
        let input_path = input_dir.as_ref();
        let output_path = output_dir.as_ref();
        std::fs::create_dir_all(output_path)?;

        let episodes = scan_series_directory(input_path)?;
        if episodes.is_empty() {
            return Err(VoxError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Không tìm thấy cặp video/phụ đề nào trong: {}", input_path.display()),
            )));
        }

        // Tải hoặc khởi tạo Sổ bộ nhân vật
        let registry_file = output_path.join("speaker_registry.json");
        let mut registry = SpeakerRegistry::load_from_file(&registry_file).unwrap_or_default();

        let mut output_files = Vec::new();

        for (ep_idx, ep) in episodes.iter().enumerate() {
            if let Some(cb) = progress_cb.as_mut() {
                cb(&ProgressInfo::new(
                    ProgressStage::Preparing,
                    ep_idx + 1,
                    episodes.len(),
                    "episodes",
                    "...",
                    format!("Bắt đầu xử lý tập: {}", ep.episode_name),
                ));
            }

            let mut ep_engine_config = self.config.clone();
            ep_engine_config.output_dir = output_path.to_path_buf();
            let engine = VoxEngine::new(ep_engine_config);

            let out_file = engine.process_episode_with_registry(
                &ep.video_path,
                &ep.subtitle_path,
                &mut registry,
                progress_cb.as_mut(),
            )?;

            // Lưu sổ bộ nhân vật sau mỗi tập để đảm bảo an toàn dữ liệu
            let _ = registry.save_to_file(&registry_file);
            output_files.push(out_file);
        }

        // Lưu sổ bộ nhân vật hoàn chỉnh
        registry.save_to_file(&registry_file)?;

        Ok(output_files)
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

/// Hàm tiện ích xử lý nhanh trọn bộ một thư mục phim gồm nhiều tập
pub fn process_series<P1, P2, F>(
    input_dir: P1,
    output_dir: P2,
    config: Option<VoxConfig>,
    progress_cb: Option<F>,
) -> Result<Vec<PathBuf>, VoxError>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    F: FnMut(&ProgressInfo),
{
    let mut cfg = config.unwrap_or_default();
    cfg.output_dir = output_dir.as_ref().to_path_buf();
    let engine = VoxEngine::new(cfg);
    engine.process_series_directory(input_dir, output_dir, progress_cb)
}
