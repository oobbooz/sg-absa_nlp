# Đánh giá và tổng kết kết quả `nlp-final`

## 1. Phạm vi thực nghiệm

Notebook `nlp-final.ipynb` huấn luyện mô hình Aspect-Based Sentiment Analysis trên tập dữ liệu Restaurant. Nhiệm vụ của mô hình là xác định cảm xúc `negative`, `neutral` hoặc `positive` đối với từng aspect được nhắc đến trong câu.

Tập train sạch có 3.607 mẫu aspect:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 807 | 22,37% |
| Neutral | 637 | 17,66% |
| Positive | 2.163 | 59,97% |

Official test set có 1.120 mẫu:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 196 | 17,50% |
| Neutral | 196 | 17,50% |
| Positive | 728 | 65,00% |

Dữ liệu mất cân bằng khá rõ, do đó Macro-F1 có ý nghĩa đánh giá cao hơn việc chỉ sử dụng Accuracy.

## 2. Kết quả ablation trên validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 | Mean Accuracy |
|---|---:|---:|---:|---:|
| Aspect attention + neutral 1.25 | **77,38%** | **0,76%** | **76,62%** | **83,31%** |
| Aspect attention | 76,34% | 1,11% | 75,22% | 82,56% |
| Hybrid pooling | 75,83% | 0,66% | 75,17% | 82,19% |
| CLS + neutral 1.25 | 75,38% | 3,06% | 72,32% | 81,82% |
| CLS + neutral 1.50 | 75,32% | 3,13% | 72,19% | 81,64% |
| CLS baseline | 75,31% | 3,67% | 71,64% | 81,88% |

Cấu hình `aspect_attention_neutral125` đạt kết quả tốt nhất. So với CLS baseline, cấu hình này tăng khoảng 2,07 điểm phần trăm Mean Macro-F1 và giảm đáng kể độ biến động giữa các seed.

Kết quả cũng cho thấy việc chỉ tăng trọng số neutral trên CLS không tạo ra cải thiện đáng kể. Neutral weighting phát huy hiệu quả rõ hơn khi kết hợp với aspect attention.

## 3. Kết quả cuối trên official test

| Seed | Best epoch | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 12 | 86,52% | 82,75% | 76,56% | 78,10% | 85,20% | 91,11% |
| 42 | 4 | **87,32%** | **82,85%** | **78,21%** | **79,95%** | **86,52%** | **95,73%** |
| 3407 | 3 | 85,71% | 81,16% | 75,03% | 77,06% | 84,43% | 94,57% |
| **Trung bình** | — | **86,52%** | **82,25%** | **76,60%** | **78,37%** | **85,38%** | **93,80%** |
| **Độ lệch chuẩn** | — | 0,66% | 0,77% | 1,30% | 1,20% | 0,86% | 1,96% |

Các metric xác suất trung bình:

| Metric | Kết quả |
|---|---:|
| Log-loss | 0,3754 |
| Expected Calibration Error | 4,14% |

Accuracy trung bình đạt 86,52%, nhưng Macro-F1 chỉ đạt 78,37%. Khoảng cách này phản ánh ảnh hưởng của mất cân bằng nhãn và cho thấy mô hình xử lý lớp positive tốt hơn các lớp thiểu số.

Macro Precision đạt 82,25%, cao hơn Macro Recall 76,60%. Mô hình có độ chính xác tương đối tốt khi dự đoán một lớp, nhưng vẫn bỏ sót một phần mẫu negative hoặc neutral.

## 4. Kết quả trên seen và unseen aspect

| Nhóm | Accuracy trung bình | Macro-F1 trung bình |
|---|---:|---:|
| Aspect đã xuất hiện trong train | 88,70% | 80,84% |
| Aspect chưa xuất hiện trong train | 81,78% | 73,70% |
| Mức giảm trên unseen aspect | 6,92 điểm | 7,14 điểm |

Sau khi refit, 353 trong tổng số 1.120 mẫu test thuộc nhóm unseen aspect, tương đương khoảng 31,52%. Mức giảm hơn 7 điểm Macro-F1 cho thấy mô hình còn phụ thuộc đáng kể vào các aspect đã gặp trong quá trình huấn luyện.

## 5. Ưu điểm

- Quy trình tách test rõ ràng; official test không được dùng để chọn cấu hình.
- Loại câu train trùng với test, giảm nguy cơ rò rỉ dữ liệu.
- Validation được chia theo câu, tránh để các aspect của cùng một câu nằm ở hai partition.
- Đánh giá trên ba seed thay vì chỉ báo cáo một lần chạy thuận lợi.
- Tiêu chí `Mean Macro-F1 - Standard Deviation` ưu tiên cả hiệu năng và độ ổn định.
- Aspect attention phù hợp với bản chất ABSA vì representation được xây dựng trực tiếp quanh aspect.
- Có ablation riêng cho pooling và neutral weighting.
- Cấu hình tốt nhất có độ lệch chuẩn validation thấp, chỉ khoảng 0,76%.
- Báo cáo nhiều metric: Accuracy, Macro-F1, ROC-AUC, log-loss, ECE và kết quả seen/unseen.
- Có temperature scaling, giúp giảm hiện tượng mô hình tự tin quá mức.
- Sinh prediction và biểu đồ, thuận lợi cho phân tích lỗi sau huấn luyện.

## 6. Hạn chế

- Macro-F1 thấp hơn Accuracy hơn 8 điểm phần trăm, cho thấy hiệu năng giữa các lớp chưa đồng đều.
- Macro Recall thấp hơn Macro Precision, nghĩa là mô hình còn bỏ sót nhiều mẫu thuộc lớp khó.
- Hiệu năng giảm rõ rệt trên unseen aspect.
- Dữ liệu chỉ thuộc miền nhà hàng, chưa chứng minh khả năng tổng quát sang miền khác hoặc ngôn ngữ khác.
- Chỉ sử dụng ba seed, chưa đủ cho một kiểm định thống kê mạnh.
- Best epoch dao động từ 3 đến 12 giữa các seed, cho thấy fine-tuning vẫn nhạy với khởi tạo ngẫu nhiên.
- Notebook chưa chạy nhóm `aspect description ablation`, nên chưa đánh giá được đóng góp thực tế của `aspects.json`.
- Bảng tổng hợp notebook chưa đưa ra precision, recall và F1 của từng lớp, vì vậy chưa xác định chính xác neutral hay negative là điểm nghẽn lớn nhất.
- Fast tokenizer phát cảnh báo không hỗ trợ đầy đủ byte fallback, có thể ảnh hưởng đến token hiếm.
- Quá trình cài thư viện trên Kaggle tạo cảnh báo xung đột với một số package CUDA có sẵn.
- File ZIP cuối khoảng 1,6 GB dù cấu hình không lưu checkpoint trực tiếp. Các ZIP artifact con có thể vẫn chứa checkpoint.
- Temperature được fit trên mô hình validation trước khi mô hình được khởi tạo lại và refit. Việc dùng temperature đó cho mô hình refit chưa hoàn toàn chặt chẽ về phương pháp. Nên hiệu chỉnh bằng một calibration split riêng của chính mô hình cuối.

## 7. Tổng kết

Thực nghiệm `nlp-final` xây dựng được một pipeline ABSA có kiểm soát dữ liệu và có khả năng tái lập tương đối tốt. Mô hình cuối đạt trung bình **86,52% Accuracy**, **78,37% Macro-F1** và **93,80% ROC-AUC** trên official test.

Kết quả ablation chứng minh aspect attention mang lại cải thiện rõ ràng so với CLS baseline. Việc tăng trọng số lớp neutral lên 1.25 tiếp tục cải thiện mô hình khi kết hợp với aspect attention, nhưng gần như không có tác dụng nếu chỉ áp dụng trên CLS.

Mô hình phù hợp làm baseline mạnh hoặc kết quả nghiên cứu cho miền Restaurant ABSA. Tuy nhiên, trước khi triển khai thực tế cần ưu tiên cải thiện recall của lớp thiểu số, khả năng xử lý unseen aspect, đánh giá trên dữ liệu ngoài miền và sửa quy trình temperature calibration cho mô hình sau refit.

