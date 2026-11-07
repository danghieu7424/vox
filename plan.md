Nhận xét của bạn đâm trúng **gót chân Achilles** của toàn bộ ngành Speaker Recognition/Diarization: **Intra-speaker variance (sự biến thiên giọng nói của cùng 1 người khi đổi cảm xúc) thường lớn hơn Inter-speaker variance (sự khác biệt giữa 2 người khác nhau ở trạng thái bình thường).**

Khi nhân vật gào thét, khóc lóc, thì thầm hoặc biến điệu giọng giả tưởng:

1. **Embedding bị trôi (Drift):** Vector 192-dim của câu hét sẽ văng ra cực xa so với vector của câu nói thầm, dù cùng 1 voice actor.
2. **Pitch F0 tĩnh bị phản tác dụng:** F0 nhảy từ $130\text{ Hz}$ (nói thường) lên $350\text{ Hz}+$ (gào thét) hoặc mất F0 (thì thầm - unvoiced). Việc lấy F0 trung bình làm tiêu chí so sánh sẽ thành nhiễu độc hại (toxic feature).

Để chuyển thư viện **`vox`** từ cơ chế đối phó (naive static matching) sang một **kiến trúc toàn diện (robust/production-grade)**, hệ thống cần được tái thiết kế theo 4 trụ cột kỹ thuật sau:

---

### 1. Multi-Centroid Codebook (Tập hợp cụm trạng thái cảm xúc)

Thay vì nén giọng của 1 nhân vật thành **1 vector trung bình duy nhất**, mỗi nhân vật trong `SpeakerRegistry` cần được lưu trữ dưới dạng một **Codebook / Cluster Set** gồm nhiều centroid đại diện cho các miền cảm xúc/trạng thái âm thanh khác nhau (`Neutral`, `Shouting/High-energy`, `Whisper`, `Emotional/Crying`).

$$\text{Distance}(e_{\text{query}}, \text{Speaker}_K) = \min_{i \in \text{Clusters}(K)} \mathcal{D}(e_{\text{query}}, c_{K, i})$$

Hoặc tính weighted $k$-Nearest Neighbors ($k$-NN) trên tập tất cả các sample đã label của speaker đó thay vì lấy trung bình hình học.

---

### 2. Chiếu qua không gian PLDA (Probabilistic Linear Discriminant Analysis)

Trong xử lý âm thanh chuyên sâu, người ta không dùng Cosine Distance trực tiếp trên raw embedding khi có sự biến thiên lớn về môi trường/cảm xúc.

**PLDA** phân tách biến thiên của vector giọng nói $x$ thành 2 thành phần:


$$x = \mu + F \cdot y + G \cdot z + \epsilon$$

* $F \cdot y$: Biến thiên **giữa các speaker khác nhau** (Inter-speaker identity - cái ta muốn giữ).
* $G \cdot z$: Biến thiên **nội tại cùng 1 speaker** (Intra-speaker / Emotion / Pitch shift / Channel noise - cái ta muốn loại bỏ).

Bằng cách train một ma trận PLDA (hoặc LDA/WCCN) nhẹ để transform embedding trước khi đo khoảng cách, ta sẽ "triệt tiêu" được chiều không gian chứa thông tin cảm xúc và chỉ giữ lại nét đặc trưng sinh học của thanh quản.

---

### 3. Tối ưu hóa chuỗi theo thời gian (Viterbi / HMM Decoding)

Gán nhãn độc lập từng câu (Greedy Matching) rất dễ bị gán sai khi gặp câu hét/nói thầm. Trong một cảnh phim, tần suất đổi người nói tuân theo quy luật ngữ cảnh (Temporal Continuity).

Xây dựng ma trận chuyển trạng thái **Hidden Markov Model (HMM)** hoặc **Viterbi Algorithm** trên toàn bộ episode:

* **Emission Probability:** Độ tương đồng giữa Audio Embedding và Speaker Registry (kết hợp PLDA).
* **Transition Probability:** Phạt nặng việc đổi người nói liên tục ở các khoảng thời gian quá ngắn ($< 0.5\text{s}$), ưu tiên duy trì luồng hội thoại giữa các nhân vật đang trong cảnh.

---

### 4. Codebook Refactoring cho `vox` (Rust Structural Design)

Dưới đây là blueprint nâng cấp cấu trúc dữ liệu trong Rust từ dạng Naive sang Multi-Modal/Cluster-based:

```rust
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Cấu trúc lưu trữ đa điểm (Codebook) đại diện cho các miền cảm xúc
#[derive(Serialize, Deserialize, Clone)]
pub struct SpeakerCluster {
    pub cluster_id: String,       // e.g., "neutral", "shouting", "whisper"
    pub centroid: Vec<f32>,       // Vector trung bình của riêng cụm cảm xúc này
    pub f0_range: (f32, f32),     // Khoảng F0 (Min, Max) thay vì 1 số tĩnh
    pub sample_count: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RobustSpeakerProfile {
    pub speaker_id: String,
    pub display_name: String,
    
    /// Thay vì 1 voiceprint đơn lẻ, lưu danh sách các cụm trạng thái
    pub clusters: Vec<SpeakerCluster>,
    
    /// Tất cả raw samples để phục vụ online adaptive clustering (K-Means/DBSCAN)
    #[serde(skip)]
    pub raw_embeddings: Vec<Vec<f32>>,
}

impl RobustSpeakerProfile {
    /// Tính khoảng cách nhỏ nhất từ Query Embedding tới bất kỳ cụm cảm xúc nào của Speaker
    pub fn min_distance_to_clusters(&self, query_emb: &[f32], plda_mat: Option<&Array2<f32>>) -> f32 {
        self.clusters
            .iter()
            .map(|cluster| {
                if let Some(mat) = plda_mat {
                    compute_plda_distance(query_emb, &cluster.centroid, mat)
                } else {
                    compute_cosine_distance(query_emb, &cluster.centroid)
                }
            })
            .fold(f32::MAX, f32::min)
    }

    /// Thêm sample mới và tự động phân cụm lại (Adaptive Incremental Clustering)
    pub fn register_sample(&mut self, new_emb: Vec<f32>, f0: f32) {
        // Nếu sample mới quá xa tất cả cụm hiện tại -> Tạo cụm cảm xúc mới (e.g. Shouting)
        // Nếu gần cụm có sẵn -> Recalculate Centroid của cụm đó
    }
}

fn compute_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Math Cosine Distance...
    0.0
}

fn compute_plda_distance(a: &[f32], b: &[f32], plda_mat: &Array2<f32>) -> f32 {
    // Project vector qua không gian loại bỏ emotion trước khi đo
    0.0
}

```

---

### Tóm tắt định hướng phát triển `vox`

| Thành phần | Cơ chế cũ (Naive / Đối phó) | Kiến trúc nâng cấp (Comprehensive) |
| --- | --- | --- |
| **Speaker Voiceprint** | 1 Vector duy nhất (Mean)

 | **Multi-Centroid Codebook** (Tập hợp các cụm cảm xúc) |
| **F0 Pitch** | 1 Giá trị trung bình tĩnh (`f0_pitch: f32`)

 | **F0 Dynamic Bandwidth** (Khoảng biến thiên F0 theo cụm) |
| **Metric Scoring** | Raw Cosine Similarity | **PLDA / Scoring Projection** (Triệt tiêu Emotion Variance) |
| **Decision Logic** | Match từng câu độc lập (Greedy) | **Viterbi / HMM Sequence Optimization** (Tối ưu chuỗi toàn cục) |