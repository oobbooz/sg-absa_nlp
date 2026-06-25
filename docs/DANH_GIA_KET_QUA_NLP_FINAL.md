# Đánh giá kết quả ABSA trên Restaurant tiếng Việt

Tài liệu này tổng hợp kết quả từ notebook `nlp-res-vi.ipynb` trên dataset `dataset/rest_vi`.
Pipeline dùng multilingual encoder `microsoft/mdeberta-v3-base`, giữ nguyên official test,
loại overlap train-test, chia validation theo câu, chạy ablation trên 3 seed và chọn cấu hình theo:

```text
Robust Macro-F1 = Mean Macro-F1 - Standard Deviation
```

## 1. Dữ liệu sau làm sạch

| Tập | Số câu | Số mẫu aspect | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 1.979 | 3.606 | 807 (22,38%) | 637 (17,67%) | 2.162 (59,95%) |
| Test | 600 | 1.120 | 196 (17,50%) | 196 (17,50%) | 728 (65,00%) |

Từ log của notebook, 2 mẫu train bị loại do trùng trực tiếp với official test.

## 2. Ablation validation

Bảng dưới đây là kết quả trung bình trên 3 validation seed.

| Cấu hình | Mean Macro-F1 | Std | Robust Macro-F1 | Mean Accuracy | Mean ECE |
|---|---:|---:|---:|---:|---:|
| hybrid + term | **69,46%** | **0,17%** | **69,30%** | **77,89%** | **4,82%** |
| aspect attention + neutral 1.25 | 70,06% | 2,48% | 67,57% | 77,22% | 5,94% |
| aspect attention + term | 69,18% | 1,65% | 67,53% | 77,03% | 5,00% |
| baseline CLS | 68,19% | 0,60% | 67,60% | 75,99% | 6,91% |
| CLS + neutral 1.25 | 68,01% | 0,44% | 67,57% | 76,78% | 6,16% |
| CLS + neutral 1.50 | 66,92% | 0,27% | 66,65% | 75,37% | 6,83% |

Cấu hình thắng theo tiêu chí robust là `hybrid_term`.
Điểm mạnh của cấu hình này là độ ổn định rất cao giữa các seed, dù Mean Macro-F1 không cao nhất tuyệt đối.

## 3. Chọn cấu hình

Notebook chọn cấu hình theo bảng xếp hạng validation:

| Trial | Mean Macro-F1 | Std | Robust Macro-F1 | Mean Accuracy |
|---|---:|---:|---:|---:|
| hybrid_term | 69,46% | 0,17% | **69,30%** | 77,89% |
| aspect_attention_neutral125 | 70,06% | 2,48% | 67,57% | 77,22% |
| aspect_attention_term | 69,18% | 1,65% | 67,53% | 77,03% |
| baseline_cls | 68,19% | 0,60% | 67,60% | 75,99% |
| cls_neutral125 | 68,01% | 0,44% | 67,57% | 76,78% |
| cls_neutral150 | 66,92% | 0,27% | 66,65% | 75,37% |

Notebook ghi nhận best configuration là:

```json
{
  "pooling_strategy": "hybrid",
  "input_format": "term",
  "encoder_layer_pooling": "last",
  "encoder_learning_rate": 8e-06,
  "head_learning_rate": 2e-05,
  "dropout": 0.3,
  "class_weighting": "none",
  "neutral_weight_multiplier": 1.0,
  "label_smoothing": 0.02,
  "loss_type": "cross_entropy",
  "scheduler": "linear",
  "warmup_ratio": 0.1,
  "max_length": 128
}
```

## 4. Kết quả official test

### 4.1. Theo seed

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC | Log-loss | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 2 | 79,64% | 75,99% | 64,21% | 68,13% | 77,89% | 90,48% | 0,5372 | 5,82% |
| 42 | 6 | **83,66%** | **77,51%** | **73,60%** | **74,63%** | **82,65%** | **92,83%** | **0,4655** | **3,81%** |
| 3407 | 3 | 80,80% | 73,64% | 69,15% | 68,86% | 78,83% | 91,52% | 0,4865 | 2,37% |

### 4.2. Trung bình và độ lệch chuẩn

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | **81,37%** | 1,69% |
| Macro Precision | **75,71%** | 1,59% |
| Macro Recall | **68,99%** | 3,84% |
| Macro-F1 | **70,54%** | 2,91% |
| Weighted-F1 | **79,79%** | 2,06% |
| ROC-AUC | **91,61%** | 0,96% |
| Log-loss | **0,4964** | 0,0301 |
| ECE | **4,00%** | 1,41% |

Seed 42 là tốt nhất trên test, nhưng độ dao động giữa 3 seed vẫn đáng kể. Điều này cho thấy
pipeline còn nhạy với split/seed dù đã dùng criterion robust ở validation.

## 5. Seen và unseen aspect

Theo log của notebook:

| Nhóm | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Seen aspect | 766 | 85,25% | 76,39% |
| Unseen aspect | 354 | 80,23% | 70,65% |

Unseen aspect làm giảm khoảng 5 điểm Accuracy và gần 6 điểm Macro-F1.
Đây là tín hiệu rõ ràng rằng mô hình vẫn phụ thuộc nhiều vào aspect đã gặp trong train.

## 6. Đọc kết quả

- `hybrid_term` là cấu hình tổng quát tốt nhất theo robust validation.
- `aspect_attention_neutral125` có Mean Macro-F1 cao nhất nhưng độ lệch chuẩn lớn hơn, nên không thắng theo robust score.
- Test Accuracy cao hơn Macro-F1 khá rõ, phản ánh lệch lớp positive trong dataset.
- Calibration đã được cải thiện sau temperature scaling, ECE trung bình ở mức 4,00%.
- Unseen aspect vẫn là điểm nghẽn chính của pipeline.

## 7. Kết luận

Pipeline trên `rest_vi` đạt:

- `81,37%` Accuracy trung bình test
- `70,54%` Macro-F1 trung bình test

Cấu hình `hybrid_term` là lựa chọn ổn định nhất trong notebook này.
Nếu muốn cải thiện tiếp, ưu tiên rõ ràng nhất là xử lý unseen aspect, kiểm tra thêm class-wise errors,
và thử các chiến lược canonical hóa aspect hoặc bổ sung description tiếng Việt trước khi mở rộng search space.
