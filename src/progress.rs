#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    Preparing,
    CheckingModel,
    DownloadingModel,
    ExtractingAudio,
    Inference,
    AligningSubtitles,
    Exporting,
    Completed,
}

impl ProgressStage {
    pub fn label(&self) -> &'static str {
        match self {
            ProgressStage::Preparing => "Chuẩn bị",
            ProgressStage::CheckingModel => "Kiểm tra Model ONNX",
            ProgressStage::DownloadingModel => "Tải mô hình ONNX",
            ProgressStage::ExtractingAudio => "Trích xuất Audio (FFmpeg)",
            ProgressStage::Inference => "Phân đoạn giọng nói (Diarization ONNX)",
            ProgressStage::AligningSubtitles => "Khớp người nói & Phụ đề",
            ProgressStage::Exporting => "Xuất phụ đề",
            ProgressStage::Completed => "Hoàn tất",
        }
    }
}

/// Thông tin tiến trình được chuẩn hóa theo chuẩn `TaskWidget` của `cli_gui_form`.
/// Có thể chuyển đổi hoặc gắn trực tiếp vào `TaskWidget::new_progress` hoặc `TaskWidget::update_progress`.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub stage: ProgressStage,
    pub current: usize,
    pub total: usize,
    pub unit: String,
    pub duration: String,
    pub status: String,
    pub percentage: f32,
}

impl ProgressInfo {
    pub fn new(
        stage: ProgressStage,
        current: usize,
        total: usize,
        unit: impl Into<String>,
        duration: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        let total_safe = total.max(1);
        let current_safe = current.min(total_safe);
        let percentage = (current_safe as f32 / total_safe as f32) * 100.0;

        Self {
            stage,
            current: current_safe,
            total: total_safe,
            unit: unit.into(),
            duration: duration.into(),
            status: status.into(),
            percentage,
        }
    }

    /// Trả về bộ 5 giá trị tương thích chuẩn `TaskWidget::new_progress` của `cli_gui_form`:
    /// `(current, total, unit, duration, status)`
    pub fn to_cli_gui_params(&self) -> (usize, usize, &str, &str, &str) {
        (
            self.current,
            self.total,
            &self.unit,
            &self.duration,
            &self.status,
        )
    }

    #[cfg(feature = "tui")]
    /// Cập nhật trực tiếp vào `cli_gui_form::TaskWidget`
    pub fn update_task_widget(&self, widget: &mut cli_gui_form::TaskWidget) {
        widget.update_progress(self.current, &self.duration);
    }

    #[cfg(feature = "tui")]
    /// Chuyển đổi trạng thái `cli_gui_form::TaskWidget` sang stage hiện tại
    pub fn switch_task_widget(&self, widget: &mut cli_gui_form::TaskWidget) {
        widget.switch_to_progress(
            self.stage.label(),
            self.total,
            &self.unit,
            &self.duration,
            &self.status,
        );
    }
}

pub type ProgressCallback<'a> = Box<dyn FnMut(&ProgressInfo) + Send + 'a>;

