# Đánh giá kết quả ABSA trên Restaurant và Laptop

## 1. Phạm vi thực nghiệm

Hai notebook sử dụng cùng pipeline Aspect-Based Sentiment Analysis (ABSA):

- `nlp-final.ipynb`: huấn luyện và đánh giá trên miền Restaurant.
- `nlp-lap.ipynb`: huấn luyện và đánh giá trên miền Laptop.

Mỗi aspect trong một câu được tách thành một mẫu phân loại có nhãn `negative`, `neutral` hoặc `positive`. Quy trình giữ nguyên official test set, loại câu train trùng test, chia validation theo câu, chạy ablation trên ba cặp seed và chỉ đánh giá test sau khi đã chọn cấu hình.

Metric chính là Macro-F1 vì cả hai dataset đều mất cân bằng nhãn. Cấu hình được xếp hạng bằng:

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

`aspect_attention_neutral125` là cấu hình tốt nhất. So với CLS baseline, cấu hình này tăng 2,07 điểm phần trăm Mean Macro-F1 và giảm mạnh độ biến động giữa các seed. Trên Restaurant, tăng trọng số neutral chỉ có tác dụng rõ khi kết hợp với aspect attention.

### 2.3. Official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 12 | 86,52% | 82,75% | 76,56% | 78,10% | 85,20% | 91,11% |
| 42 | 4 | **87,32%** | **82,85%** | **78,21%** | **79,95%** | **86,52%** | **95,73%** |
| 3407 | 3 | 85,71% | 81,16% | 75,03% | 77,06% | 84,43% | 94,57% |
| **Trung bình** | — | **86,52%** | **82,25%** | **76,60%** | **78,37%** | **85,38%** | **93,80%** |
| **Độ lệch chuẩn** | — | 0,66% | 0,77% | 1,30% | 1,20% | 0,86% | 1,96% |

Log-loss trung bình là 0,3754 và Expected Calibration Error (ECE) là 4,14%.

Accuracy cao hơn Macro-F1 8,15 điểm phần trăm, phản ánh ưu thế của lớp positive và hiệu năng chưa đồng đều giữa ba lớp. Macro Recall thấp hơn Macro Precision cho thấy mô hình vẫn bỏ sót một phần mẫu negative và neutral.

### 2.4. Seen và unseen aspect

| Nhóm | Accuracy trung bình | Macro-F1 trung bình |
|---|---:|---:|
| Aspect đã xuất hiện trong train | 88,70% | 80,84% |
| Aspect chưa xuất hiện trong train | 81,78% | 73,70% |
| Mức giảm trên unseen aspect | 6,92 điểm | 7,14 điểm |

Khoảng 31,52% mẫu test thuộc nhóm unseen aspect sau khi làm sạch. Mức giảm hơn 7 điểm Macro-F1 cho thấy mô hình Restaurant còn phụ thuộc đáng kể vào aspect đã gặp trong train.

## 3. Kết quả Laptop từ `nlp-lap.ipynb`

### 3.1. Dữ liệu sau làm sạch

| Tập | Số câu | Số mẫu aspect | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Train | 1.465 | 2.327 | 870 (37,39%) | 463 (19,90%) | 994 (42,72%) |
| Test | 411 | 638 | 128 (20,06%) | 169 (26,49%) | 341 (53,45%) |

Một mẫu neutral bị loại khỏi train do câu trùng với official test. So với Restaurant, train Laptop cân bằng hơn giữa negative và positive, nhưng phân phối nhãn thay đổi khá mạnh từ train sang test.

### 3.2. Core ablation trên validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 | Mean Accuracy | Mean ECE |
|---|---:|---:|---:|---:|---:|
| Aspect attention | 78,06% | **0,58%** | **77,47%** | 81,64% | 5,20% |
| Aspect attention + neutral 1.25 | **78,15%** | 1,04% | 77,11% | 81,07% | 5,65% |
| Hybrid pooling | 77,52% | 1,89% | 75,63% | 81,73% | 4,75% |
| CLS baseline | 77,75% | 2,35% | 75,40% | **81,92%** | 4,58% |
| CLS + neutral 1.50 | 77,20% | 2,34% | 74,87% | 80,87% | **3,68%** |
| CLS + neutral 1.25 | 76,78% | 2,18% | 74,59% | 81,45% | 5,12% |

Điểm Mean Macro-F1 cao nhất thuộc về `aspect_attention_neutral125`, nhưng chênh lệch chỉ 0,09 điểm phần trăm và cấu hình này biến động lớn hơn. Theo tiêu chí robust, `aspect_attention_term` thắng nhờ độ lệch chuẩn chỉ 0,58%.

Kết quả này khác Restaurant: tăng trọng số neutral không giúp cấu hình cuối ổn định hơn trên Laptop. Aspect attention vẫn là thay đổi có giá trị nhất, còn CLS đạt Accuracy validation cao nhưng Macro-F1 và độ ổn định kém hơn.

### 3.3. Official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC | Log-loss | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 5 | 78,37% | 75,92% | 74,76% | 72,24% | 76,30% | 91,32% | 0,5753 | 7,85% |
| 42 | 10 | **81,50%** | 77,78% | **79,09%** | **77,54%** | **81,11%** | 91,71% | **0,5389** | **4,96%** |
| 3407 | 5 | 80,72% | **78,69%** | 76,93% | 74,75% | 79,00% | **91,75%** | 0,5677 | 6,94% |
| **Trung bình** | — | **80,20%** | **77,46%** | **76,93%** | **74,84%** | **78,80%** | **91,59%** | **0,5606** | **6,58%** |
| **Độ lệch chuẩn** | — | 1,33% | 1,15% | 1,77% | 2,16% | 1,97% | 0,19% | 0,0157 | 1,21% |

Seed 42 tốt nhất, nhưng chênh lệch Macro-F1 giữa seed 42 và seed 2024 là 5,30 điểm phần trăm. Độ lệch chuẩn test Macro-F1 2,16% cao hơn validation của cấu hình thắng, cho thấy bước refit và số epoch được chọn vẫn nhạy với seed.

Macro Precision và Macro Recall trung bình khá gần nhau, nhưng Macro-F1 thấp hơn cả hai. Đây là dấu hiệu hiệu năng giữa các lớp không đồng đều và không thể kết luận chất lượng chỉ từ các macro average riêng lẻ.

### 3.4. Seen và unseen aspect

| Nhóm | Accuracy trung bình | Macro-F1 trung bình |
|---|---:|---:|
| Aspect đã xuất hiện trong train | 83,96% | 76,30% |
| Aspect chưa xuất hiện trong train | 75,00% | 72,92% |
| Mức giảm trên unseen aspect | 8,96 điểm | 3,39 điểm |

Theo split dùng cho final run, 268/638 mẫu test (42,01%) là unseen aspect. Đây là tỷ lệ cao hơn Restaurant. Accuracy giảm gần 9 điểm trên unseen aspect, nhưng Macro-F1 chỉ giảm 3,39 điểm; cần đọc hai metric cùng nhau vì phân phối nhãn của hai nhóm có thể khác nhau.

## 4. So sánh Restaurant và Laptop

| Metric test trung bình | Restaurant | Laptop | Laptop so với Restaurant |
|---|---:|---:|---:|
| Accuracy | 86,52% | 80,20% | -6,32 điểm |
| Macro Precision | 82,25% | 77,46% | -4,79 điểm |
| Macro Recall | 76,60% | 76,93% | +0,33 điểm |
| Macro-F1 | 78,37% | 74,84% | -3,53 điểm |
| Weighted-F1 | 85,38% | 78,80% | -6,58 điểm |
| ROC-AUC | 93,80% | 91,59% | -2,21 điểm |
| Log-loss | 0,3754 | 0,5606 | +0,1852 |
| ECE | 4,14% | 6,58% | +2,44 điểm |

Các nhận xét chính:

- Aspect attention thắng trên cả hai miền, củng cố giả thuyết rằng biểu diễn tập trung trực tiếp vào aspect phù hợp hơn CLS thuần cho ABSA.
- Neutral weighting phụ thuộc miền dữ liệu. Hệ số 1.25 giúp rõ trên Restaurant nhưng không thắng robust score trên Laptop.
- Laptop khó hơn về Accuracy, Macro-F1, log-loss và calibration. Nguyên nhân hợp lý gồm tập train nhỏ hơn 35,49%, tỷ lệ unseen aspect test cao hơn và độ lệch phân phối nhãn train–test lớn hơn.
- Macro Recall của Laptop tương đương Restaurant, nhưng Precision và F1 thấp hơn. Mô hình Laptop không chỉ bỏ sót mẫu mà còn tạo nhiều dự đoán sai hơn khi gán nhãn.
- Kết quả Laptop nhạy seed hơn: độ lệch chuẩn Macro-F1 test là 2,16%, so với 1,20% trên Restaurant.
- Không nên dùng chung một cấu hình class weighting cho mọi miền. Pooling có khả năng tổng quát tốt hơn class weighting, còn trọng số lớp cần tuning riêng theo dataset.

## 5. Ưu điểm của quy trình

- Official test không được dùng để chọn cấu hình.
- Câu train trùng test được loại bỏ.
- Validation được chia theo toàn bộ câu, tránh leakage giữa các aspect trong cùng câu.
- Mỗi cấu hình được đánh giá trên ba cặp split/model seed.
- Tiêu chí robust cân bằng hiệu năng trung bình và độ ổn định.
- Có ablation riêng cho pooling và neutral weighting.
- Báo cáo nhiều metric: Accuracy, Macro-F1, ROC-AUC, log-loss, ECE và seen/unseen aspect.
- `nlp-lap.ipynb` giữ nguyên pipeline Restaurant, nhờ đó phép so sánh hai miền có ý nghĩa hơn.

## 6. Hạn chế và hướng cải thiện

- Ba seed vẫn chưa đủ cho kiểm định thống kê mạnh; nên chạy ít nhất 5 seed nếu tài nguyên cho phép.
- Chưa chạy `aspect description ablation`, nên chưa đánh giá đóng góp thực tế của `aspects.json`.
- Báo cáo tổng hợp notebook chưa hiển thị precision, recall và F1 theo từng lớp; cần dùng các file metrics/predictions để xác định lớp gây lỗi chính.
- Tỷ lệ unseen aspect cao, đặc biệt ở Laptop, cho thấy nên thử input có aspect description, chuẩn hóa/canonicalize aspect hoặc augmentation.
- Temperature scaling được fit trước refit rồi áp dụng cho mô hình refit. Nên dành calibration split riêng cho chính mô hình cuối.
- Best epoch Laptop dao động từ 5 đến 10 và test Macro-F1 biến động đáng kể; nên thử learning rate thấp hơn, layer-wise decay hoặc checkpoint averaging.
- Với Laptop, nên ưu tiên cải thiện robustness trước khi tăng độ phức tạp: chạy thêm seed, phân tích lỗi từng lớp và kiểm tra các nhóm unseen aspect phổ biến.

## 7. Kết luận

Pipeline đạt kết quả tốt hơn trên Restaurant với **86,52% Accuracy** và **78,37% Macro-F1**. Trên Laptop, cùng quy trình đạt **80,20% Accuracy** và **74,84% Macro-F1**.

Kết quả hai miền cùng xác nhận lợi ích của aspect attention, nhưng không xác nhận một quy tắc class weighting chung. Cấu hình Restaurant phù hợp nhất là `aspect_attention_neutral125`, trong khi Laptop phù hợp nhất theo robust score là `aspect_attention_term`.

Mô hình hiện là baseline mạnh cho nghiên cứu ABSA theo từng miền. Bước tiếp theo nên tập trung vào unseen aspect, metric từng lớp, calibration sau refit và đánh giá nhiều seed hơn.
