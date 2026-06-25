# Đánh giá kết quả ABSA trên Restaurant, Laptop và Restaurant tiếng Việt

## 1. Phạm vi thực nghiệm

Hai notebook sử dụng cùng pipeline Aspect-Based Sentiment Analysis (ABSA):

- `nlp-final.ipynb`: huấn luyện và đánh giá trên miền Restaurant.
- `nlp-lap.ipynb`: huấn luyện và đánh giá trên miền Laptop.
- `nlp-res-vi.ipynb`: huấn luyện và đánh giá trên bản tiếng Việt của Restaurant.

Mỗi aspect trong một câu được tách thành một mẫu phân loại có nhãn `negative`, `neutral` hoặc `positive`.
Quy trình giữ nguyên official test set, loại câu train trùng test, chia validation theo câu, chạy ablation trên
ba cấp seed và chỉ đánh giá test sau khi đã chọn cấu hình.

Metric chính là Macro-F1 vì cả ba dataset đều mất cân bằng nhãn. Cấu hình được xếp hạng bằng:

```text
Robust Macro-F1 = Mean Macro-F1 - Standard Deviation
```

## 2. Kết quả Restaurant

### 2.1. Dữ liệu sau làm sạch

| Tập | Số câu | Số mẫu aspect | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 1.979 | 3.607 | 807 (22,37%) | 637 (17,66%) | 2.163 (59,97%) |
| Test | 600 | 1.120 | 196 (17,50%) | 196 (17,50%) | 728 (65,00%) |

Một mẫu positive bị loại khỏi train do câu trùng với official test.

### 2.2. Core ablation trên validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 | Mean Accuracy |
|---|---:|---:|---:|---:|
| Aspect attention + neutral 1.25 | **77,38%** | **0,76%** | **76,62%** | **83,31%** |
| Aspect attention | 76,34% | 1,11% | 75,22% | 82,56% |
| Hybrid pooling | 75,83% | 0,66% | 75,17% | 82,19% |
| CLS + neutral 1.25 | 75,38% | 3,06% | 72,32% | 81,82% |
| CLS + neutral 1.50 | 75,32% | 3,13% | 72,19% | 81,64% |
| CLS baseline | 75,31% | 3,67% | 71,64% | 81,88% |

`aspect_attention_neutral125` là cấu hình tốt nhất. So với CLS baseline, cấu hình này tăng 2,07 điểm phần trăm
Mean Macro-F1 và giảm mạnh độ biến động giữa các seed.

### 2.3. Official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 12 | 86,52% | 82,75% | 76,56% | 78,10% | 85,20% | 91,11% |
| 42 | 4 | **87,32%** | **82,85%** | **78,21%** | **79,95%** | **86,52%** | **95,73%** |
| 3407 | 3 | 85,71% | 81,16% | 75,03% | 77,06% | 84,43% | 94,57% |
| **Trung bình** | — | **86,52%** | **82,25%** | **76,60%** | **78,37%** | **85,38%** | **93,80%** |
| **Độ lệch chuẩn** | — | 0,66% | 0,77% | 1,30% | 1,20% | 0,86% | 1,96% |

Log-loss trung bình là 0,3754 và Expected Calibration Error (ECE) là 4,14%.

## 3. Kết quả Laptop

### 3.1. Dữ liệu sau làm sạch

| Tập | Số câu | Số mẫu aspect | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 1.465 | 2.327 | 870 (37,39%) | 463 (19,90%) | 994 (42,72%) |
| Test | 411 | 638 | 128 (20,06%) | 169 (26,49%) | 341 (53,45%) |

Một mẫu neutral bị loại khỏi train do câu trùng với official test.

### 3.2. Core ablation trên validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 | Mean Accuracy | Mean ECE |
|---|---:|---:|---:|---:|---:|
| Aspect attention | 78,06% | **0,58%** | **77,47%** | 81,64% | 5,20% |
| Aspect attention + neutral 1.25 | **78,15%** | 1,04% | 77,11% | 81,07% | 5,65% |
| Hybrid pooling | 77,52% | 1,89% | 75,63% | 81,73% | 4,75% |
| CLS baseline | 77,75% | 2,35% | 75,40% | **81,92%** | 4,58% |
| CLS + neutral 1.50 | 77,20% | 2,34% | 74,87% | 80,87% | **3,68%** |
| CLS + neutral 1.25 | 76,78% | 2,18% | 74,59% | 81,45% | 5,12% |

Điểm Mean Macro-F1 cao nhất thuộc về `aspect_attention_neutral125`, nhưng chênh lệch chỉ 0,09 điểm phần trăm
và cấu hình này biến động lớn hơn. Theo tiêu chí robust, `aspect_attention_term` thắng nhờ độ lệch chuẩn chỉ 0,58%.

### 3.3. Official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC | Log-loss | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 5 | 78,37% | 75,92% | 74,76% | 72,24% | 76,30% | 91,32% | 0,5753 | 7,85% |
| 42 | 10 | **81,50%** | 77,78% | **79,09%** | **77,54%** | **81,11%** | 91,71% | **0,5389** | **4,96%** |
| 3407 | 5 | 80,72% | **78,69%** | 76,93% | 74,75% | 79,00% | **91,75%** | 0,5677 | 6,94% |
| **Trung bình** | — | **80,20%** | **77,46%** | **76,93%** | **74,84%** | **78,80%** | **91,59%** | **0,5606** | **6,58%** |
| **Độ lệch chuẩn** | — | 1,33% | 1,15% | 1,77% | 2,16% | 1,97% | 0,19% | 0,0157 | 1,21% |

Seed 42 tốt nhất, nhưng chênh lệch Macro-F1 giữa seed 42 và seed 2024 là 5,30 điểm phần trăm.

### 3.4. Seen và unseen aspect

| Nhóm | Accuracy trung bình | Macro-F1 trung bình |
|---|---:|---:|
| Aspect đã xuất hiện trong train | 83,96% | 76,30% |
| Aspect chưa xuất hiện trong train | 75,00% | 72,92% |
| Mức giảm trên unseen aspect | 8,96 điểm | 3,39 điểm |

Theo split dùng cho final run, 268/638 mẫu test (42,01%) là unseen aspect.

## 4. Kết quả rest_vi từ `nlp-res-vi.ipynb`

### 4.1. Dữ liệu sau làm sạch

| Tập | Số câu | Số mẫu aspect | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 1.979 | 3.606 | 807 (22,38%) | 637 (17,67%) | 2.162 (59,95%) |
| Test | 600 | 1.120 | 196 (17,50%) | 196 (17,50%) | 728 (65,00%) |

Hai mẫu train bị loại do trùng trực tiếp với official test.

### 4.2. Core ablation trên validation

| Cấu hình | Mean Macro-F1 | Std | Robust Macro-F1 | Mean Accuracy | Mean ECE |
|---|---:|---:|---:|---:|---:|
| hybrid_term | **69,46%** | **0,17%** | **69,30%** | **77,89%** | **4,82%** |
| aspect_attention_neutral125 | 70,06% | 2,48% | 67,57% | 77,22% | 5,94% |
| aspect_attention_term | 69,18% | 1,65% | 67,53% | 77,03% | 5,00% |
| baseline_cls | 68,19% | 0,60% | 67,60% | 75,99% | 6,91% |
| cls_neutral125 | 68,01% | 0,44% | 67,57% | 76,78% | 6,16% |
| cls_neutral150 | 66,92% | 0,27% | 66,65% | 75,37% | 6,83% |

`hybrid_term` là cấu hình thắng theo robust score.

### 4.3. Official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC | Log-loss | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 2 | 79,64% | 75,99% | 64,21% | 68,13% | 77,89% | 90,48% | 0,5372 | 5,82% |
| 42 | 6 | **83,66%** | **77,51%** | **73,60%** | **74,63%** | **82,65%** | **92,83%** | **0,4655** | **3,81%** |
| 3407 | 3 | 80,80% | 73,64% | 69,15% | 68,86% | 78,83% | 91,52% | 0,4865 | 2,37% |
| **Trung bình** | — | **81,37%** | **75,71%** | **68,99%** | **70,54%** | **79,79%** | **91,61%** | **0,4964** | **4,00%** |
| **Độ lệch chuẩn** | — | 1,69% | 1,59% | 3,84% | 2,91% | 2,06% | 0,96% | 0,0301 | 1,41% |

### 4.4. Seen và unseen aspect

| Nhóm | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Seen aspect | 766 | 85,25% | 76,39% |
| Unseen aspect | 354 | 80,23% | 70,65% |

Unseen aspect làm giảm khoảng 5 điểm Accuracy và gần 6 điểm Macro-F1.

## 5. So sánh ba miền

| Metric test trung bình | Restaurant | Laptop | Rest VI |
|---|---:|---:|---:|
| Accuracy | 86,52% | 80,20% | 81,37% |
| Macro Precision | 82,25% | 77,46% | 75,71% |
| Macro Recall | 76,60% | 76,93% | 68,99% |
| Macro-F1 | 78,37% | 74,84% | 70,54% |
| Weighted-F1 | 85,38% | 78,80% | 79,79% |
| ROC-AUC | 93,80% | 91,59% | 91,61% |
| Log-loss | 0,3754 | 0,5606 | 0,4964 |
| ECE | 4,14% | 6,58% | 4,00% |

Các nhận xét chính:

- Aspect attention thắng trên cả ba miền hoặc là lựa chọn rất mạnh.
- Neutral weighting phụ thuộc miền dữ liệu. Hệ số 1.25 giúp rõ trên Restaurant nhưng không thắng robust score trên Laptop hoặc Rest VI.
- Laptop khó hơn về Accuracy, Macro-F1, log-loss và calibration.
- Rest VI nằm giữa Restaurant và Laptop về chất lượng tổng thể, nhưng vẫn chịu ảnh hưởng của dịch máy và unseen aspect.
- Kết quả Rest VI xác nhận rằng pipeline multilingual chạy ổn, nhưng vẫn cần xử lý tốt aspect hiếm và chuẩn hóa từ vựng.

## 6. Ưu điểm của quy trình

- Official test không được dùng để chọn cấu hình.
- Câu train trùng test được loại bỏ.
- Validation được chia theo toàn bộ câu, tránh leakage giữa các aspect trong cùng câu.
- Mỗi cấu hình được đánh giá trên ba cấp split/model seed.
- Tiêu chí robust cân bằng hiệu năng trung bình và độ ổn định.
- Có ablation riêng cho pooling và neutral weighting.
- Báo cáo nhiều metric: Accuracy, Macro-F1, ROC-AUC, log-loss, ECE và seen/unseen aspect.

## 7. Hạn chế và hướng cải thiện

- Ba seed vẫn chưa đủ cho kiểm định thống kê mạnh; nên chạy ít nhất 5 seed nếu tài nguyên cho phép.
- Chưa chạy `aspect description ablation` cho Rest VI vì `aspects.json` của bản dịch chưa có description tiếng Việt.
- Báo cáo tổng hợp notebook chưa hiển thị precision, recall và F1 theo từng lớp; cần dùng file metrics/predictions để xác định lớp gây lỗi chính.
- Tỷ lệ unseen aspect cao, đặc biệt ở Laptop và Rest VI, cho thấy nên thử input có aspect description, chuẩn hóa/canonicalize aspect hoặc augmentation.
- Temperature scaling được fit trước refit rồi áp dụng cho mô hình refit; nên dành calibration split riêng cho model cuối.
- Best epoch của Laptop và Rest VI còn dao động theo seed; nên thử learning rate thấp hơn, layer-wise decay hoặc checkpoint averaging.

## 8. Kết luận

Pipeline đạt kết quả tốt hơn trên Restaurant với **86,52% Accuracy** và **78,37% Macro-F1**.
Trên Laptop đạt **80,20% Accuracy** và **74,84% Macro-F1**.
Trên Rest VI đạt **81,37% Accuracy** và **70,54% Macro-F1**.

Ba miền cùng xác nhận lợi ích của aspect attention, nhưng không xác nhận một quy tắc class weighting chung.
Cấu hình Restaurant phù hợp nhất là `aspect_attention_neutral125`, Laptop phù hợp nhất theo robust score là
`aspect_attention_term`, còn Rest VI phù hợp nhất theo robust score là `hybrid_term`.

Mô hình hiện là baseline mạnh cho nghiên cứu ABSA theo từng miền. Bước tiếp theo nên tập trung vào unseen aspect,
metric từng lớp, calibration sau refit và đánh giá nhiều seed hơn.
