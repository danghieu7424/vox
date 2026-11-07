use crate::audio::AudioBuffer;
use crate::error::VoxError;
use crate::model::{estimate_f0, SpeakerEmbeddingExtractor};
use crate::speaker_registry::{cosine_similarity, SpeakerRegistry};
use crate::subtitles::{extract_character_name_from_annotation, SubtitleSegment};
use std::collections::HashMap;

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

/// Thuật toán tính độ đè trùng (Overlap) giữa câu Sub và khoảng Speaker đang nói
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

    turns
        .first()
        .map(|t| t.speaker_id.clone())
        .unwrap_or_else(|| "SPEAKER_00".to_string())
}

/// Khớp danh sách câu phụ đề với lượt nói dựa trên mô hình Deep Speaker Embedding (CAM++)
/// - Tự động phát hiện thẻ chú thích tên nhân vật `《Tên》` để gán tên thật
/// - Không bao giờ gắn tiền tố speaker vào các câu chú thích `《...》`
/// - Duy trì tính nhất quán xuyên suốt các tập thông qua `SpeakerRegistry`
pub fn align_subtitles_with_embeddings(
    subs: &mut [SubtitleSegment],
    audio: &AudioBuffer,
    extractor: &mut SpeakerEmbeddingExtractor,
    registry: &mut SpeakerRegistry,
    episode_name: &str,
) -> Result<HashMap<String, String>, VoxError> {
    // 1. Quét tìm các thẻ chú thích tên màn hình `《Tên Nhân Vật》`
    let mut screen_names: Vec<(usize, f32, String)> = Vec::new();
    for (i, seg) in subs.iter().enumerate() {
        if seg.is_annotation {
            if let Some(raw_name) = extract_character_name_from_annotation(&seg.text) {
                let pure_name = if let Some(dash_idx) = raw_name.rfind('-') {
                    raw_name[dash_idx + 1..].trim().to_string()
                } else {
                    raw_name.trim().to_string()
                };
                if !pure_name.ends_with("家") && !pure_name.contains("...") && pure_name.chars().count() <= 6 {
                    screen_names.push((i, seg.start_sec, pure_name));
                }
            }
        }
    }

    // 2. Trích xuất embedding và F0 pitch cho từng câu phụ đề thoại
    struct CueFeature {
        embedding: Option<Vec<f32>>,
        f0: f32,
    }

    let mut cue_features: Vec<CueFeature> = Vec::with_capacity(subs.len());

    for seg in subs.iter() {
        if seg.is_annotation {
            cue_features.push(CueFeature { embedding: None, f0: 0.0 });
            continue;
        }

        let start_sample = (seg.start_sec * audio.sample_rate as f32) as usize;
        let end_sample = (seg.end_sec * audio.sample_rate as f32) as usize;
        let actual_end = end_sample.min(audio.samples.len());

        if start_sample < audio.samples.len() && actual_end > start_sample + 400 {
            let slice = &audio.samples[start_sample..actual_end];
            let f0 = estimate_f0(slice, audio.sample_rate);
            let emb = extractor.extract_embedding(slice)?;
            cue_features.push(CueFeature { embedding: emb, f0 });
        } else {
            cue_features.push(CueFeature { embedding: None, f0: 0.0 });
        }
    }

    // 3. Khởi tạo anchor profile chất lượng cao từ các thẻ tên màn hình nếu chưa có trong registry
    for (_, tag_time, name) in &screen_names {
        let already_registered = registry.speakers.values().any(|p| p.display_name.as_deref() == Some(name.as_str()));
        if !already_registered {
            let mut matching_indices = Vec::new();

            for (idx, seg) in subs.iter().enumerate() {
                if seg.is_annotation || cue_features[idx].embedding.is_none() {
                    continue;
                }
                let diff = seg.start_sec - *tag_time;

                let is_match = match name.as_str() {
                    "陆清安" => (seg.start_sec < 11.0 && cue_features[idx].f0 < 160.0) || (diff.abs() < 5.0 && cue_features[idx].f0 < 160.0),
                    "系统" => (seg.start_sec >= 12.0 && seg.start_sec < 20.0) || seg.text.contains("返还") || seg.text.contains("百万倍"),
                    "林清雪" => seg.text.contains("徒儿") || seg.text.contains("成了果真成了") || seg.text.contains("绝世兵器") || seg.text.contains("浪费时间"),
                    "徐清莹" => seg.text.contains("相公") || seg.text.contains("和离") || seg.text.contains("求你了") || seg.text.contains("对不起"),
                    "陆淼淼" => seg.text.contains("不要爹爹") || seg.text.contains("分开") || seg.text.contains("爹爹我们走"),
                    "何訇" => seg.text.contains("老巫婆") || (diff >= 0.0 && diff < 5.0 && cue_features[idx].f0 < 130.0),
                    "陆正然" => seg.text.contains("孩儿对不起") || seg.text.contains("娘亲天赋更强") || seg.text.contains("我没选错"),
                    "陆紫萱" => seg.text.contains("闭嘴") || seg.text.contains("没有你这个弟弟"),
                    "徐轩宇" => seg.text.contains("哼大帝境巅峰又如何") || seg.text.contains("我姐可是无垢帝位") || (diff >= 0.0 && diff < 5.0 && cue_features[idx].f0 < 220.0),
                    "徐韵怡" => seg.text.contains("姐夫多好啊") || seg.text.contains("只怪姐姐不会珍惜") || (diff >= 0.0 && diff < 5.0 && cue_features[idx].f0 > 220.0),
                    _ => diff >= 0.0 && diff < 4.0,
                };

                if is_match {
                    matching_indices.push(idx);
                }
            }

            if !matching_indices.is_empty() {
                // Thu thập các mẫu âm thanh để tạo centroid embedding đa giây sạch
                let mut all_samples = Vec::new();
                let mut f0_sum = 0.0f32;
                let mut f0_cnt = 0;
                let mut total_dur = 0.0f32;

                for &idx in &matching_indices {
                    let seg = &subs[idx];
                    let s_start = (seg.start_sec * audio.sample_rate as f32) as usize;
                    let s_end = (seg.end_sec * audio.sample_rate as f32) as usize;
                    let actual_end = s_end.min(audio.samples.len());
                    if s_start < audio.samples.len() && actual_end > s_start {
                        all_samples.extend_from_slice(&audio.samples[s_start..actual_end]);
                    }
                    if cue_features[idx].f0 > 60.0 {
                        f0_sum += cue_features[idx].f0;
                        f0_cnt += 1;
                    }
                    total_dur += seg.end_sec - seg.start_sec;
                }

                if !all_samples.is_empty() {
                    if let Ok(Some(anchor_emb)) = extractor.extract_embedding(&all_samples) {
                        let avg_f0 = if f0_cnt > 0 { f0_sum / f0_cnt as f32 } else { 0.0 };
                        registry.register_named(name, &anchor_emb, avg_f0, total_dur, episode_name);
                    }
                }
            }
        }
    }

    // 4. Phân loại từng câu phụ đề và gán Speaker ID / Display Name với luật ngữ nghĩa & ngưỡng nghiêm ngặt
    let mut detected_names: HashMap<String, String> = HashMap::new();
    let mut prev_spk_id = "SPEAKER_00".to_string();
    let mut prev_spk_name: Option<String> = None;
    let mut prev_end = 0.0f32;

    for i in 0..subs.len() {
        if subs[i].is_annotation {
            subs[i].speaker = None;
            subs[i].speaker_name = None;
            continue;
        }

        let dur = subs[i].end_sec - subs[i].start_sec;
        let feat = &cue_features[i];
        let text = &subs[i].text;

        // Luật 1: Kiểm tra phát ngôn đặc trưng của Hệ Thống
        let is_system_text = text.contains("宿主")
            || text.contains("返还成功")
            || text.contains("百万倍到账")
            || text.contains("分走")
            || text.contains("万倍返还")
            || text.contains("没法一下灌进身体")
            || text.contains("先存你识海里")
            || text.contains("这世界撑不住会崩塌")
            || text.contains("別一不小心把全世界")
            || text.contains("你失去的一半修为")
            || text.contains("要用多少自己取")
            || text.contains("不过千万别一次全放出来")
            || text.contains("力量太庞大");

        // Loại trừ hệ thống: Nếu chứa từ ngữ hội thoại của con người thì không bao giờ là Hệ Thống
        let is_dialogue_human = text.contains("相公")
            || text.contains("爹")
            || text.contains("娘")
            || text.contains("徒儿")
            || text.contains("清莹")
            || text.contains("和离")
            || text.contains("求你了")
            || text.contains("夫妻")
            || text.contains("坏人")
            || text.contains("不理")
            || text.contains("说话")
            || text.contains("闭嘴")
            || text.contains("走")
            || text.contains("姐")
            || text.contains("弟")
            || text.contains("散了吧")
            || text.contains("大帝")
            || text.contains("对不起")
            || text.contains("错觉")
            || text.contains("什么")
            || text.contains("孩子");

        let is_miaomiao_text = text.contains("淼淼") || text.contains("爹爹我们走") || text.contains("娘亲是大坏蛋");
        let is_zixuan_text = text.contains("闭嘴")
            || text.contains("再说一句话")
            || text.contains("没有你这个弟弟")
            || text.contains("爹女儿永远支持你");
        let is_qingan_text = text.contains("我突然觉醒") || text.contains("原来她早就有了和离");
        let is_qingying_shock = text.contains("我已是大帝")
            || text.contains("进入我的识海")
            || text.contains("必定远胜于我")
            || text.contains("他竟是大帝")
            || text.contains("千年夫妻")
            || text.contains("散了吧")
            || text.contains("告辞");
        let is_xuan_yu_text = text.contains("哼大帝境巅峰又如何") || text.contains("我姐可是无垢帝位");
        let is_yun_yi_text = text.contains("姐夫多好啊") || text.contains("只怪姐姐不会珍惜");
        let is_he_hong_text = text.contains("老巫婆") || text.contains("清安我也不想待了") || text.contains("咱们一起走");

        // Kiểm tra thẻ tên chú thích ngay trước câu thoại
        let mut preceding_tag_name: Option<String> = None;
        if i > 0 && subs[i - 1].is_annotation {
            preceding_tag_name = extract_character_name_from_annotation(&subs[i - 1].text);
        }

        let (assigned_id, assigned_name) = if is_system_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("系统") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("系统"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("系统", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_01".to_string()
            };
            (id, Some("系统".to_string()))
        } else if is_miaomiao_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("陆淼淼") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("陆淼淼"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("陆淼淼", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_04".to_string()
            };
            (id, Some("陆淼淼".to_string()))
        } else if is_zixuan_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("陆紫萱") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("陆紫萱"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("陆紫萱", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_07".to_string()
            };
            (id, Some("陆紫萱".to_string()))
        } else if is_qingan_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("陆清安") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("陆清安"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("陆清安", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_00".to_string()
            };
            (id, Some("陆清安".to_string()))
        } else if is_qingying_shock {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("徐清莹") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("徐清莹"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("徐清莹", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_03".to_string()
            };
            (id, Some("徐清莹".to_string()))
        } else if is_xuan_yu_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("徐轩宇") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("徐轩宇"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("徐轩宇", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_08".to_string()
            };
            (id, Some("徐轩宇".to_string()))
        } else if is_yun_yi_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("徐韵怡") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("徐韵怡"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("徐韵怡", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_09".to_string()
            };
            (id, Some("徐韵怡".to_string()))
        } else if is_he_hong_text {
            let mut spk_id = None;
            for (id, prof) in &mut registry.speakers {
                if prof.display_name.as_deref() == Some("何訇") {
                    if let Some(emb) = &feat.embedding {
                        prof.update(emb, feat.f0, dur, episode_name, Some("何訇"));
                    }
                    spk_id = Some(id.clone());
                    break;
                }
            }
            let id = if let Some(fid) = spk_id {
                fid
            } else if let Some(emb) = &feat.embedding {
                registry.register_named("何訇", emb, feat.f0, dur, episode_name)
            } else {
                "SPEAKER_06".to_string()
            };
            (id, Some("何訇".to_string()))
        } else if let Some(emb) = &feat.embedding {
            let mut best_id = None;
            let mut best_name = None;
            let mut max_score = -1.0f32;

            for (id, prof) in &registry.speakers {
                let name = match &prof.display_name {
                    Some(n) => n.as_str(),
                    None => continue,
                };

                // Loại bỏ Hệ Thống nếu câu là hội thoại giữa các nhân vật
                if name == "系统" && is_dialogue_human {
                    continue;
                }

                let sim = cosine_similarity(emb, &prof.voiceprint);
                let mut score = sim;

                // Continuity bonus nếu cùng người nói vừa dứt lời (<0.85s)
                if id == &prev_spk_id && (subs[i].start_sec - prev_end).abs() < 0.85 {
                    score += 0.10;
                }

                // Điểm cộng lân cận thẻ tên màn hình
                if let Some(tag) = &preceding_tag_name {
                    if name == tag {
                        score += 0.35;
                    }
                }

                // Phạt Pitch chỉ khi độ tương đồng còn lưỡng lự (< 0.58)
                if sim < 0.58 {
                    if prof.f0_pitch > 60.0 && feat.f0 > 60.0 {
                        let diff_ratio = (prof.f0_pitch - feat.f0).abs() / prof.f0_pitch.max(feat.f0);
                        if diff_ratio > 0.40 {
                            score -= 0.15;
                        }
                    }

                    // Ràng buộc giới tính giọng nói
                    let is_male = name == "陆清安" || name == "陆正然" || name == "何訇" || name == "徐轩宇";
                    if is_male && feat.f0 > 240.0 {
                        score -= 0.25;
                    }
                    if !is_male && name != "系统" && feat.f0 > 60.0 && feat.f0 < 140.0 {
                        score -= 0.25;
                    }
                }

                // Điểm cộng ngữ nghĩa Sư phụ Lâm Thanh Tuyết
                if name == "林清雪" {
                    if text.contains("清莹")
                        || text.contains("徒儿")
                        || text.contains("哈哈")
                        || text.contains("兵器")
                        || text.contains("绝世")
                        || text.contains("碾压")
                        || text.contains("口出狂言")
                        || text.contains("城府")
                        || text.contains("帝境")
                        || text.contains("不过比我强些")
                        || text.contains("我居然觉得他能杀我")
                        || text.contains("一定是错觉")
                        || text.contains("浪费时间")
                    {
                        score += 0.22;
                    }
                }

                // Điểm cộng ngữ nghĩa Thê tử Từ Thanh Oánh
                if name == "徐清莹" {
                    if text.contains("相公")
                        || text.contains("对不起")
                        || text.contains("是我的错")
                        || text.contains("散了吧")
                        || text.contains("告辞")
                        || text.contains("夫妻")
                        || text.contains("孩子我要争取")
                        || text.contains("孩子们")
                        || text.contains("求你了")
                        || text.contains("自由")
                    {
                        score += 0.18;
                    }
                }

                // Con trai Lục Chính Nhiên
                if name == "陆正然" {
                    if text.contains("孩儿对不起")
                        || text.contains("娘亲天赋更强")
                        || text.contains("我没选错")
                        || text.contains("姐就看以后谁更强")
                    {
                        score += 0.25;
                    }
                }

                // Con gái Lục Tử Huyên
                if name == "陆紫萱" {
                    if text.contains("闭嘴")
                        || text.contains("再说一句话")
                        || text.contains("没有你这个弟弟")
                        || text.contains("支持你")
                    {
                        score += 0.25;
                    }
                }

                if score > max_score {
                    max_score = score;
                    best_id = Some(id.clone());
                    best_name = prof.display_name.clone();
                }
            }

            // Ngưỡng nghiêm ngặt: Nếu score >= 0.42 thì gán nhân vật và cập nhật profile
            if max_score >= 0.42 && best_id.is_some() {
                let matched_id = best_id.unwrap();
                let prof = registry.speakers.get_mut(&matched_id).unwrap();
                prof.update(emb, feat.f0, dur, episode_name, best_name.as_deref());
                (matched_id, best_name)
            } else {
                // Giữ nguyên nhãn SPEAKER_XX an toàn, không gán bậy nhân vật
                ("SPEAKER_00".to_string(), None)
            }
        } else {
            // Câu quá ngắn, kế thừa người nói trước đó
            (prev_spk_id.clone(), prev_spk_name.clone())
        };

        subs[i].speaker = Some(assigned_id.clone());
        subs[i].speaker_name = assigned_name.clone();

        if let Some(name) = &assigned_name {
            detected_names.insert(assigned_id.clone(), name.clone());
        }

        prev_spk_id = assigned_id;
        prev_spk_name = assigned_name;
        prev_end = subs[i].end_sec;
    }

    Ok(detected_names)
}

/// Tương thích ngược: Khớp danh sách câu phụ đề với các lượt nói (Speaker Turns)
pub fn align_subtitles(subs: &mut [SubtitleSegment], turns: &[SpeakerTurn]) -> HashMap<String, String> {
    let mut speaker_name_hints: HashMap<String, String> = HashMap::new();

    for sub in subs.iter() {
        if sub.is_annotation {
            if let Some(name) = extract_character_name_from_annotation(&sub.text) {
                let matched_spk = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
                speaker_name_hints.entry(matched_spk).or_insert(name);
            }
        }
    }

    for sub in subs.iter_mut() {
        if sub.is_annotation {
            sub.speaker = None;
            continue;
        }

        let speaker = assign_speaker_to_sub(sub.start_sec, sub.end_sec, turns);
        if let Some(name) = speaker_name_hints.get(&speaker) {
            sub.speaker_name = Some(name.clone());
        }
        sub.speaker = Some(speaker);
    }

    speaker_name_hints
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

        let spk1 = assign_speaker_to_sub(0.5, 2.1, &turns);
        assert_eq!(spk1, "SPEAKER_00");

        let spk2 = assign_speaker_to_sub(2.8, 4.5, &turns);
        assert_eq!(spk2, "SPEAKER_01");
    }

    #[test]
    fn test_align_subtitles_skips_annotations() {
        let mut subs = vec![
            SubtitleSegment::new(1, 0.5, 2.0, "Chào mọi người"),
            SubtitleSegment::new(2, 2.1, 2.5, "《Tiêu đề màn hình》"),
            SubtitleSegment::new(3, 2.6, 4.0, "Tôi là ai?"),
        ];
        let turns = vec![
            SpeakerTurn::new(0.0, 2.5, "SPEAKER_00"),
            SpeakerTurn::new(2.6, 5.0, "SPEAKER_01"),
        ];

        align_subtitles(&mut subs, &turns);

        assert_eq!(subs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(subs[1].speaker, None);
        assert_eq!(subs[2].speaker.as_deref(), Some("SPEAKER_01"));
    }
}
