# Tóm tắt quy trình huấn luyện NLP

## Bảng các bước thực hiện

| Bước | Công việc | Nội dung thực hiện |
|---:|---|---|
| 1 | Chuẩn bị môi trường | Chạy notebook trên Kaggle với GPU Tesla T4 và cài các thư viện trong `requirements.txt`. |
| 2 | Kiểm tra clean-start | Xóa kết quả cũ và xác nhận mã nguồn không chứa checkpoint, cache hoặc artifact từ lần huấn luyện trước. |
| 3 | Đọc dữ liệu | Sử dụng tập Restaurant gồm 1.980 câu train và 600 câu test. Mỗi aspect trong một câu được tách thành một mẫu phân loại riêng. |
| 4 | Làm sạch dữ liệu | Giữ nguyên official test set và loại khỏi train những câu trùng với test. Có một mẫu positive bị loại do trùng lặp. |
| 5 | Chia tập validation | Dùng 15% dữ liệu train làm validation. Việc chia được thực hiện theo nhóm câu để các aspect của cùng một câu không xuất hiện đồng thời trong train và validation. |
| 6 | Chuẩn bị đầu vào | Mỗi đầu vào là cặp `sentence + aspect term`, được tokenize với độ dài tối đa 128 token. |
| 7 | Xây dựng mô hình | Dùng `microsoft/deberta-v3-base` làm encoder và classification head để dự đoán ba lớp `negative`, `neutral`, `positive`. |
| 8 | Xây dựng representation | So sánh ba phương pháp pooling: CLS, aspect attention và hybrid pooling. |
| 9 | Xử lý mất cân bằng | Thử tăng trọng số riêng cho lớp neutral với hệ số 1.25 và 1.50. |
| 10 | Chạy ablation | So sánh sáu cấu hình core, không sử dụng aspect description để tránh trộn nhiều thay đổi trong cùng một thí nghiệm. |
| 11 | Tuning nhiều seed | Mỗi cấu hình được đánh giá trên ba cặp split/model seed: `2024:2024`, `42:42`, `3407:3407`. |
| 12 | Huấn luyện | Dùng cross-entropy, label smoothing 0.02, dropout 0.3, linear scheduler, warmup 10%, mixed precision và gradient checkpointing. |
| 13 | Early stopping | Theo dõi validation Macro-F1 với patience bằng 3. Checkpoint tốt nhất được chọn theo Macro-F1. |
| 14 | Chọn cấu hình | Xếp hạng bằng `Mean Macro-F1 - Standard Deviation` nhằm ưu tiên cả hiệu năng và độ ổn định giữa các seed. |
| 15 | Refit | Khởi tạo lại mô hình và huấn luyện trên toàn bộ tập train sạch trong số epoch đã được chọn từ validation. |
| 16 | Đánh giá cuối | Chỉ sử dụng official test set sau khi đã chọn xong cấu hình. Chạy final evaluation với ba seed 2024, 42 và 3407. |
| 17 | Hiệu chỉnh xác suất | Áp dụng temperature scaling và tính thêm ECE, log-loss, Brier score và confidence. |
| 18 | Xuất artifact | Sinh JSON, CSV metric, prediction, learning curve, confusion matrix, ROC curve, calibration diagram và file ZIP. |

## Cấu hình được lựa chọn

| Thành phần | Giá trị |
|---|---|
| Tên cấu hình | `aspect_attention_neutral125` |
| Encoder | `microsoft/deberta-v3-base` |
| Pooling | Aspect attention |
| Input | Sentence + aspect term |
| Encoder learning rate | `8e-6` |
| Head learning rate | `2e-5` |
| Dropout | 0.3 |
| Neutral weight multiplier | 1.25 |
| Label smoothing | 0.02 |
| Loss | Cross-entropy |
| Scheduler | Linear |
| Warmup ratio | 0.1 |
| Max length | 128 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Validation ratio | 0.15 |
| Selection metric | Mean Macro-F1 − Standard Deviation |

## Luồng xử lý tổng quát

```text
Dữ liệu JSON
    → Tách một mẫu cho mỗi aspect
    → Loại câu train trùng với test
    → Chia train/validation theo câu
    → Tokenize sentence + aspect
    → Fine-tune DeBERTa-v3-base
    → So sánh các cấu hình ablation trên ba seed
    → Chọn cấu hình ổn định nhất
    → Refit trên toàn bộ train sạch
    → Đánh giá official test
    → Xuất metric và artifact
```

