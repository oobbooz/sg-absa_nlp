# Tổng kết bài toán phân tích cảm xúc theo khía cạnh

## 1. Bài toán là gì?

Đề tài thực hiện **Aspect-Based Sentiment Analysis (ABSA)**, tức phân tích cảm xúc theo từng khía cạnh cụ thể được nhắc đến trong câu.

Trong phân tích cảm xúc thông thường, toàn bộ câu hoặc đoạn văn chỉ nhận một nhãn cảm xúc. Cách làm đó không đủ chính xác khi một câu đồng thời nhận xét nhiều đối tượng. Ví dụ:

> The food was excellent, but the service was slow.

Câu này có hai khía cạnh:

- `food`: cảm xúc tích cực (`positive`);
- `service`: cảm xúc tiêu cực (`negative`).

Nếu chỉ phân loại cảm xúc chung cho cả câu, mô hình khó biểu diễn đúng hai ý kiến trái ngược. Vì vậy, ABSA nhận đầu vào là cặp:

```text
Câu + aspect cần phân tích
```

và dự đoán một trong ba nhãn:

- `negative`: tiêu cực;
- `neutral`: trung tính;
- `positive`: tích cực.

Trong đề tài này, aspect đã được xác định sẵn trong dữ liệu. Phạm vi chính là **Aspect Sentiment Classification**, không bao gồm bước tự động trích xuất aspect từ văn bản mới.

## 2. Mục tiêu của đề tài

Đề tài có các mục tiêu chính:

1. Xây dựng pipeline huấn luyện mô hình ABSA dựa trên Transformer.
2. So sánh các phương pháp tạo biểu diễn cho câu và aspect.
3. Kiểm tra tác động của việc tăng trọng số lớp neutral.
4. Đánh giá mô hình trên hai miền dữ liệu:
   - đánh giá nhà hàng (`Restaurant`);
   - đánh giá máy tính xách tay (`Laptop`).
5. Đánh giá không chỉ hiệu năng trung bình mà còn độ ổn định qua nhiều seed.
6. Phân tích khả năng xử lý aspect đã gặp và chưa gặp trong quá trình huấn luyện.

## 3. Dataset

### 3.1. Cấu trúc dữ liệu

Mỗi record JSON chứa:

- câu đánh giá;
- danh sách token;
- một hoặc nhiều aspect;
- vị trí aspect trong câu;
- nhãn cảm xúc của từng aspect;
- một số thông tin cú pháp như dependency head, dependency relation và POS tag.

Ví dụ rút gọn:

```json
{
  "sentence": "The battery life is very good.",
  "aspects": [
    {
      "term": ["battery", "life"],
      "polarity": "positive"
    }
  ]
}
```

Mỗi aspect được bung thành một mẫu phân loại độc lập. Nếu một câu có ba aspect thì câu đó tạo thành ba mẫu, mỗi mẫu có cùng nội dung câu nhưng aspect mục tiêu khác nhau.

### 3.2. Dataset Restaurant

| Split | Số câu | Số mẫu aspect |
|---|---:|---:|
| Train thô | 1.980 | 3.608 |
| Train sạch | 1.979 | 3.607 |
| Test | 600 | 1.120 |

Phân phối nhãn của train sạch:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 807 | 22,37% |
| Neutral | 637 | 17,66% |
| Positive | 2.163 | 59,97% |

Phân phối nhãn của test:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 196 | 17,50% |
| Neutral | 196 | 17,50% |
| Positive | 728 | 65,00% |

Restaurant mất cân bằng khá mạnh về lớp positive. Các aspect phổ biến gồm `food`, `service`, `prices`, `menu`, `staff` và `atmosphere`.

Trung bình mỗi câu train có khoảng 1,82 aspect. Gần một nửa số câu train chứa nhiều hơn một aspect.

### 3.3. Dataset Laptop

| Split | Số câu | Số mẫu aspect |
|---|---:|---:|
| Train thô | 1.466 | 2.328 |
| Train sạch | 1.465 | 2.327 |
| Test | 411 | 638 |

Phân phối nhãn của train sạch:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 870 | 37,39% |
| Neutral | 463 | 19,90% |
| Positive | 994 | 42,72% |

Phân phối nhãn của test:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 128 | 20,06% |
| Neutral | 169 | 26,49% |
| Positive | 341 | 53,45% |

Laptop cân bằng hơn Restaurant trên train, nhưng có sự thay đổi phân phối khá lớn giữa train và test. Tỷ lệ negative giảm mạnh trong test, trong khi neutral và positive tăng.

Các aspect Laptop phân tán trên nhiều nhóm như `screen`, `price`, `battery life`, `keyboard`, `software`, `warranty`, `performance` và hệ điều hành.

### 3.4. So sánh hai dataset

| Đặc điểm | Restaurant | Laptop |
|---|---:|---:|
| Mẫu train sạch | 3.607 | 2.327 |
| Mẫu test | 1.120 | 638 |
| Unique aspect train sạch | 1.201 | 949 |
| Tỷ lệ unseen aspect trong test | 31,52% | 42,01% |
| Độ dài câu train trung bình | 20,30 token | 22,34 token |
| Số aspect trung bình/câu train | 1,82 | 1,59 |

Restaurant có nhiều dữ liệu hơn nhưng mất cân bằng nhãn mạnh hơn. Laptop có ít dữ liệu, aspect phân tán và tỷ lệ aspect chưa xuất hiện trong train cao hơn. Đây là những nguyên nhân khiến Laptop khó tổng quát hơn.

### 3.4. Dataset Restaurant tiếng Việt

Đề tài còn có thêm một biến thể tiếng Việt của Restaurant, dùng trong notebook `nlp-res-vi.ipynb` và tập `dataset/rest_vi`.

| Split | Số câu | Số mẫu aspect |
|---|---:|---:|
| Train thô | 1.980 | 3.608 |
| Train sạch | 1.979 | 3.606 |
| Test | 600 | 1.120 |

Phân phối nhãn của train sạch:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 807 | 22,38% |
| Neutral | 637 | 17,67% |
| Positive | 2.162 | 59,95% |

Phân phối nhãn của test:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 196 | 17,50% |
| Neutral | 196 | 17,50% |
| Positive | 728 | 65,00% |

Restaurant tiếng Việt giữ gần như nguyên phân phối của Restaurant gốc vì đây là bản dịch máy của cùng bộ dữ liệu. Khác biệt đáng chú ý là số mẫu train sạch giảm còn 3.606 do loại 2 câu overlap với official test.

Các aspect phổ biến nhất trong bản tiếng Việt gồm `đồ ăn`, `dịch vụ`, `giá`, `món ăn`, `địa điểm`, `thức ăn`, `nhân viên`, `pizza`, `không khí` và `bàn`.

| Đặc điểm | Restaurant | Laptop | Rest VI |
|---|---:|---:|---:|
| Train sạch | 3.607 | 2.327 | 3.606 |
| Test | 1.120 | 638 | 1.120 |
| Tỷ lệ unseen aspect trong test | 31,52% | 42,01% | 32,32% |

Rest VI về bản chất là cùng miền với Restaurant nhưng chạy qua pipeline tiếng Việt. Vì vậy, nó hữu ích để kiểm tra khả năng tổng quát hóa đa ngôn ngữ và mức độ ổn định của aspect attention khi aspect được dịch sang tiếng Việt.

## 4. Tiền xử lý và kiểm soát dữ liệu

### 4.1. Bung dữ liệu theo aspect

Mỗi aspect trong một câu được chuyển thành một mẫu riêng:

```text
sentence = "The screen is beautiful but the battery is weak."

mẫu 1: sentence + "screen"  → positive
mẫu 2: sentence + "battery" → negative
```

Cách tổ chức này giúp mô hình học cảm xúc tương ứng với đúng aspect thay vì dự đoán cảm xúc chung của câu.

### 4.2. Loại train–test overlap

Official test được giữ nguyên. Những câu train xuất hiện trong test bị loại hoàn toàn khỏi train:

- Restaurant loại một mẫu positive;
- Laptop loại một mẫu neutral.

Bước này giảm nguy cơ mô hình ghi nhớ câu test và làm kết quả cao giả tạo.

### 4.3. Chia validation theo câu

Validation chiếm khoảng 15% train sạch. Việc chia được thực hiện theo nhóm câu, không chia ngẫu nhiên từng aspect riêng lẻ.

Nếu các aspect của cùng một câu xuất hiện ở cả train và validation, mô hình đã nhìn thấy gần như toàn bộ nội dung câu validation trong lúc train. Chia theo câu giúp ngăn dạng leakage này.

### 4.4. Tokenization

Mô hình nhận cặp `sentence + aspect term`. Dữ liệu được tokenize bằng tokenizer tương ứng với DeBERTa-v3-base, với độ dài tối đa 128 token.

Độ dài câu lớn nhất của hai dataset vẫn dưới 128 token theo token gốc. Vì vậy, nguy cơ mất nhiều nội dung do cắt câu là tương đối thấp, dù số subword sau tokenizer có thể lớn hơn số token gốc.

## 5. Mô hình sử dụng

Encoder chính là:

```text
microsoft/deberta-v3-base
```

Đây là mô hình Transformer đã được pretrain trên lượng lớn văn bản. Sau encoder là classification head dự đoán ba lớp cảm xúc.

Các cấu hình chính:

| Thành phần | Giá trị |
|---|---|
| Encoder | DeBERTa-v3-base |
| Input | Sentence + aspect term |
| Số lớp đầu ra | 3 |
| Encoder learning rate | `8e-6` |
| Head learning rate | `2e-5` |
| Dropout | 0,3 |
| Label smoothing | 0,02 |
| Loss | Cross-entropy |
| Scheduler | Linear |
| Warmup ratio | 0,1 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Max length | 128 |
| Early stopping patience | 3 |

Mixed precision và gradient checkpointing được sử dụng để giảm nhu cầu bộ nhớ GPU.

## 6. Các phương pháp biểu diễn được so sánh

### 6.1. CLS pooling

CLS pooling sử dụng biểu diễn token đầu tiên của Transformer làm biểu diễn chung cho toàn bộ input.

Ưu điểm:

- đơn giản;
- ít thành phần bổ sung;
- chi phí tính toán thấp hơn.

Nhược điểm:

- biểu diễn thiên về toàn câu;
- có thể không tập trung đủ vào aspect mục tiêu;
- khó xử lý câu có nhiều aspect mang cảm xúc khác nhau.

### 6.2. Aspect attention

Phương pháp này tạo biểu diễn cho aspect, sau đó sử dụng aspect làm truy vấn attention trên các token của câu.

Mục tiêu là chọn ra phần ngữ cảnh liên quan trực tiếp đến aspect. Ví dụ khi phân tích `service`, mô hình cần tập trung vào từ `slow`, không để nhận xét tích cực về `food` chi phối kết quả.

Đây là phương pháp phù hợp với bản chất của ABSA và đã thắng trên cả hai dataset.

### 6.3. Hybrid pooling

Hybrid pooling kết hợp:

- biểu diễn CLS;
- biểu diễn aspect;
- attentive context.

Phương pháp này cung cấp nhiều thông tin hơn nhưng cũng tăng số tham số và độ phức tạp. Kết quả thực nghiệm không cho thấy hybrid ổn định hơn aspect attention thuần.

### 6.4. Neutral weighting

Do neutral là lớp ít mẫu, thí nghiệm tăng riêng trọng số neutral lên:

- 1,25;
- 1,50.

Mục tiêu là làm loss phạt mạnh hơn khi dự đoán sai neutral. Tuy nhiên, tăng trọng số quá mức có thể làm mô hình dự đoán neutral nhiều hơn cần thiết và giảm hiệu năng của các lớp khác.

## 7. Quy trình huấn luyện và lựa chọn mô hình

Mỗi cấu hình được đánh giá trên ba cặp split seed và model seed:

```text
2024:2024
42:42
3407:3407
```

Quy trình gồm:

1. Làm sạch train và giữ riêng official test.
2. Chia train–validation theo câu.
3. Huấn luyện từng cấu hình tối đa 12 epoch.
4. Theo dõi validation Macro-F1.
5. Dừng sớm nếu không cải thiện trong ba epoch.
6. Lặp lại với ba seed.
7. Tính Mean Macro-F1 và độ lệch chuẩn.
8. Chọn cấu hình theo robust score.
9. Khởi tạo lại mô hình.
10. Refit trên toàn bộ train sạch với số epoch đã chọn.
11. Đánh giá official test trên ba model seed.

Robust score được tính:

```text
Robust Macro-F1 = Mean Macro-F1 - Standard Deviation
```

Tiêu chí này không chỉ ưu tiên cấu hình có điểm trung bình cao mà còn phạt cấu hình biến động mạnh giữa các seed.

## 8. Metric đánh giá

### Accuracy

Tỷ lệ tổng số dự đoán đúng. Metric này dễ hiểu nhưng có thể gây hiểu lầm khi dữ liệu mất cân bằng.

### Macro-F1

F1 được tính riêng cho từng lớp rồi lấy trung bình không trọng số. Mỗi lớp có vai trò ngang nhau dù số mẫu khác nhau.

Macro-F1 là metric chính vì đề tài cần đánh giá cả negative, neutral và positive, không chỉ lớp phổ biến.

### Weighted-F1

F1 của mỗi lớp được tính trọng số theo số lượng mẫu. Metric này phản ánh hiệu năng tổng thể nhưng vẫn chịu ảnh hưởng của lớp lớn.

### ROC-AUC

Đánh giá khả năng mô hình xếp hạng xác suất đúng cho từng lớp theo one-vs-rest.

### Log-loss và ECE

- Log-loss đánh giá chất lượng phân phối xác suất.
- Expected Calibration Error đo mức chênh lệch giữa confidence và độ chính xác thực tế.

Hai mô hình có Accuracy tương đương vẫn có thể khác nhau về độ tin cậy của xác suất.

## 9. Kết quả Restaurant

### 9.1. Ablation validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 |
|---|---:|---:|---:|
| Aspect attention + neutral 1.25 | **77,38%** | 0,76% | **76,62%** |
| Aspect attention | 76,34% | 1,11% | 75,22% |
| Hybrid pooling | 75,83% | **0,66%** | 75,17% |
| CLS + neutral 1.25 | 75,38% | 3,06% | 72,32% |
| CLS + neutral 1.50 | 75,32% | 3,13% | 72,19% |
| CLS baseline | 75,31% | 3,67% | 71,64% |

Cấu hình được chọn là:

```text
aspect_attention_neutral125
```

Aspect attention kết hợp neutral weight 1.25 tăng 2,07 điểm Mean Macro-F1 so với CLS baseline và ổn định hơn rõ rệt.

### 9.2. Official test

| Metric | Trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Accuracy | **86,52%** | 0,66% |
| Macro Precision | 82,25% | 0,77% |
| Macro Recall | 76,60% | 1,30% |
| Macro-F1 | **78,37%** | 1,20% |
| Weighted-F1 | 85,38% | 0,86% |
| ROC-AUC | 93,80% | 1,96% |
| Log-loss | 0,3754 | — |
| ECE | 4,14% | — |

Kết quả cho thấy mô hình phân loại tốt trên Restaurant. Tuy nhiên, Accuracy cao hơn Macro-F1 8,15 điểm, chứng tỏ hiệu năng giữa các lớp chưa đồng đều và lớp positive đóng góp lớn vào Accuracy.

### 9.3. Seen và unseen aspect

| Nhóm | Accuracy | Macro-F1 |
|---|---:|---:|
| Seen aspect | 88,70% | 80,84% |
| Unseen aspect | 81,78% | 73,70% |

Macro-F1 giảm 7,14 điểm trên unseen aspect. Mô hình vẫn phụ thuộc đáng kể vào các aspect đã gặp trong train.

## 10. Kết quả Laptop

### 10.1. Ablation validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 |
|---|---:|---:|---:|
| Aspect attention | 78,06% | **0,58%** | **77,47%** |
| Aspect attention + neutral 1.25 | **78,15%** | 1,04% | 77,11% |
| Hybrid pooling | 77,52% | 1,89% | 75,63% |
| CLS baseline | 77,75% | 2,35% | 75,40% |
| CLS + neutral 1.50 | 77,20% | 2,34% | 74,87% |
| CLS + neutral 1.25 | 76,78% | 2,18% | 74,59% |

Cấu hình được chọn là:

```text
aspect_attention_term
```

Bản neutral 1.25 có Mean Macro-F1 cao hơn 0,09 điểm, nhưng biến động lớn hơn. Aspect attention không tăng trọng số neutral có robust score cao nhất.

### 10.2. Official test

| Metric | Trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Accuracy | **80,20%** | 1,33% |
| Macro Precision | 77,46% | 1,15% |
| Macro Recall | 76,93% | 1,77% |
| Macro-F1 | **74,84%** | 2,16% |
| Weighted-F1 | 78,80% | 1,97% |
| ROC-AUC | 91,59% | 0,19% |
| Log-loss | 0,5606 | 0,0157 |
| ECE | 6,58% | 1,21% |

Laptop có kết quả thấp hơn Restaurant và biến động Macro-F1 lớn hơn. Seed tốt nhất đạt 77,54% Macro-F1, trong khi seed thấp nhất đạt 72,24%.

### 10.3. Seen và unseen aspect

| Nhóm | Accuracy | Macro-F1 |
|---|---:|---:|
| Seen aspect | 83,96% | 76,30% |
| Unseen aspect | 75,00% | 72,92% |

Accuracy giảm 8,96 điểm trên unseen aspect. Khoảng 42,01% test Laptop chứa aspect không xuất hiện trong toàn bộ train sạch.

## 11. Kết quả Restaurant tiếng Việt

### 11.1. Ablation validation

| Cấu hình | Mean Macro-F1 | Độ lệch chuẩn | Robust Macro-F1 |
|---|---:|---:|---:|
| hybrid_term | **69,46%** | **0,17%** | **69,30%** |
| aspect_attention_neutral125 | 70,06% | 2,48% | 67,57% |
| aspect_attention_term | 69,18% | 1,65% | 67,53% |
| baseline_cls | 68,19% | 0,60% | 67,60% |
| cls_neutral125 | 68,01% | 0,44% | 67,57% |
| cls_neutral150 | 66,92% | 0,27% | 66,65% |

Cấu hình được chọn là `hybrid_term`. Điểm mạnh của cấu hình này là độ ổn định rất cao giữa các seed, dù Mean Macro-F1 không cao nhất tuyệt đối.

### 11.2. Official test

| Metric | Trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Accuracy | **81,37%** | 1,69% |
| Macro Precision | **75,71%** | 1,59% |
| Macro Recall | **68,99%** | 3,84% |
| Macro-F1 | **70,54%** | 2,91% |
| Weighted-F1 | **79,79%** | 2,06% |
| ROC-AUC | **91,61%** | 0,96% |
| Log-loss | 0,4964 | 0,0301 |
| ECE | **4,00%** | 1,41% |

Kết quả cho thấy Rest VI thấp hơn Restaurant gốc nhưng vẫn là một baseline mạnh. Accuracy vẫn cao hơn Macro-F1 rõ rệt, phản ánh mất cân bằng nhãn positive và độ khó của unseen aspect.

### 11.3. Seen và unseen aspect

| Nhóm | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Seen aspect | 766 | 85,25% | 76,39% |
| Unseen aspect | 354 | 80,23% | 70,65% |

Unseen aspect làm giảm khoảng 5 điểm Accuracy và gần 6 điểm Macro-F1. Đây vẫn là điểm nghẽn chính của bản tiếng Việt.

## 12. So sánh kết quả ba miền

| Metric | Restaurant | Laptop | Chênh lệch Laptop |
|---|---:|---:|---:|
| Accuracy | 86,52% | 80,20% | -6,32 điểm |
| Macro Precision | 82,25% | 77,46% | -4,79 điểm |
| Macro Recall | 76,60% | 76,93% | +0,33 điểm |
| Macro-F1 | 78,37% | 74,84% | -3,53 điểm |
| Weighted-F1 | 85,38% | 78,80% | -6,58 điểm |
| ROC-AUC | 93,80% | 91,59% | -2,21 điểm |
| ECE | 4,14% | 6,58% | +2,44 điểm |

Rest VI có cùng xu hướng với Restaurant về lợi thế aspect attention, nhưng kết quả tổng thể thấp hơn đôi chút do bản dịch tiếng Việt và độ khó của unseen aspect.

Aspect attention thắng trên cả hai miền. Điều này cho thấy việc tập trung biểu diễn vào aspect mục tiêu có tính tổng quát tốt hơn so với chỉ dùng CLS.

Neutral weighting không có kết luận chung:

- có lợi trên Restaurant;
- không cải thiện robust score trên Laptop.
- với Rest VI, cấu hình tốt nhất theo robust score là `hybrid_term`, cho thấy bản tiếng Việt cần cân bằng giữa pooling đa nguồn và độ ổn định seed.

Do đó, class weighting phụ thuộc phân phối dữ liệu và cần được tuning riêng cho từng miền.

## 13. Vì sao Restaurant tốt hơn Laptop?

### 13.1. Restaurant có nhiều dữ liệu hơn

Restaurant có 3.607 mẫu train sạch, nhiều hơn Laptop 1.280 mẫu, tương đương khoảng 55%. Nhiều dữ liệu hơn giúp mô hình học ngữ cảnh cảm xúc và giảm overfitting.

### 13.2. Laptop có tỷ lệ unseen aspect cao hơn

- Restaurant: 31,52%;
- Laptop: 42,01%.

Laptop yêu cầu mô hình tổng quát đến nhiều tên linh kiện, phần mềm, hệ điều hành và thuộc tính chưa gặp.

### 13.3. Aspect Laptop phân tán hơn

Restaurant tập trung mạnh vào các khái niệm lặp lại như `food` và `service`. Laptop có nhiều cách gọi chi tiết cho linh kiện, phiên bản phần mềm và tính năng. Mỗi aspect trung bình có ít ví dụ hơn.

### 13.4. Phân phối Laptop thay đổi giữa train và test

Tỷ lệ negative trong Laptop giảm từ khoảng 37% ở train xuống 20% ở test. Neutral và positive tăng. Sự dịch chuyển này làm phân phối mô hình học được không hoàn toàn giống dữ liệu đánh giá.

### 13.5. Laptop nhạy với seed hơn

Độ lệch chuẩn test Macro-F1:

- Restaurant: 1,20%;
- Laptop: 2,16%.

Tập nhỏ và đa dạng khiến quá trình fine-tuning phụ thuộc nhiều hơn vào khởi tạo ngẫu nhiên và số epoch được chọn.

## 14. Ưu điểm của đề tài

- Giải quyết đúng bài toán ABSA thay vì sentiment classification toàn câu.
- Dùng Transformer pretrained mạnh.
- Official test được giữ riêng trong quá trình chọn mô hình.
- Loại câu train trùng test.
- Chia validation theo câu để tránh leakage.
- Có ablation để kiểm tra đóng góp của pooling và class weighting.
- Chạy ba seed thay vì chỉ báo cáo một lần chạy thuận lợi.
- Chọn cấu hình dựa trên cả hiệu năng và độ ổn định.
- Refit trên toàn bộ train sạch sau khi chọn cấu hình.
- Đánh giá bằng nhiều metric.
- Có phân tích seen/unseen aspect.
- Thực nghiệm trên ba biến thể dữ liệu giúp kiểm tra khả năng áp dụng của pipeline.

## 15. Hạn chế

- Chỉ thực hiện phân loại cảm xúc khi aspect đã được cung cấp; chưa tự động trích xuất aspect.
- Ba seed chưa đủ cho kiểm định thống kê mạnh.
- Chưa chạy ablation sử dụng aspect description.
- Chưa tổng hợp Precision, Recall và F1 từng lớp trong báo cáo chính.
- Tỷ lệ unseen aspect còn cao.
- Chưa chuẩn hóa các aspect gần nghĩa hoặc biến thể số ít/số nhiều.
- Kết quả Laptop còn nhạy với seed.
- Temperature scaling được fit trên mô hình trước refit rồi áp dụng cho mô hình sau refit. Điều này ảnh hưởng độ chặt chẽ của ECE và log-loss, dù không làm thay đổi nhãn dự đoán, Accuracy hay Macro-F1.
- Mới đánh giá trên dữ liệu tiếng Anh và một bản dịch tiếng Việt của Restaurant; chưa chứng minh khả năng tổng quát sang ngôn ngữ hoặc lĩnh vực khác.
- Mô hình DeBERTa-v3-base tương đối nặng, chi phí huấn luyện và suy luận cao hơn các mô hình nhỏ.

## 16. Hướng cải thiện

1. Báo cáo confusion matrix và metric từng lớp.
2. Chạy từ 5 đến 10 seed để đánh giá ổn định tốt hơn.
3. Thử aspect description cho các aspect hiếm và unseen.
4. Chuẩn hóa alias, số ít/số nhiều và các aspect cùng khái niệm.
5. Thử augmentation cho neutral và aspect hiếm.
6. Dùng calibration split riêng sau khi refit.
7. Thử learning rate thấp hơn hoặc layer-wise learning-rate decay.
8. Thử checkpoint averaging để giảm biến động seed.
9. Bổ sung bước tự động trích xuất aspect để tạo pipeline ABSA hoàn chỉnh.
10. Đánh giá cross-domain, chẳng hạn train trên một miền và kiểm tra khả năng chuyển sang miền còn lại.

## 17. Kết luận

Đề tài đã xây dựng được pipeline phân tích cảm xúc theo khía cạnh trên hai miền Restaurant và Laptop, đồng thời kiểm tra thêm biến thể tiếng Việt của Restaurant. Kết quả chứng minh rằng aspect attention phù hợp với ABSA hơn biểu diễn CLS thuần vì mô hình có thể tập trung vào ngữ cảnh liên quan trực tiếp đến aspect.

Cấu hình tốt nhất trên Restaurant là `aspect_attention_neutral125`, đạt trung bình:

- **86,52% Accuracy**;
- **78,37% Macro-F1**;
- **93,80% ROC-AUC**.

Cấu hình tốt nhất trên Laptop là `aspect_attention_term`, đạt trung bình:

- **80,20% Accuracy**;
- **74,84% Macro-F1**;
- **91,59% ROC-AUC**.

Restaurant đạt kết quả cao hơn nhờ tập train lớn và nhiều aspect phổ biến được lặp lại. Laptop khó hơn do dữ liệu ít, aspect đa dạng, tỷ lệ unseen cao và có sự thay đổi phân phối nhãn giữa train và test.

Rest VI cho thấy pipeline có thể chạy ổn trên bản dịch tiếng Việt, nhưng vẫn bị ảnh hưởng bởi mất cân bằng nhãn và unseen aspect. Cấu hình tốt nhất theo robust score là `hybrid_term`, còn kết quả test trung bình ở mức **81,37% Accuracy** và **70,54% Macro-F1**.

Kết quả phù hợp để sử dụng trong báo cáo môn học vì quy trình có kiểm soát leakage, có ablation, đánh giá nhiều seed và phân tích hạn chế. Tuy nhiên, các con số nên được trình bày như kết quả thực nghiệm trên dataset cụ thể, không nên khẳng định mô hình đã giải quyết hoàn toàn bài toán ABSA hoặc có thể tổng quát tốt cho mọi miền dữ liệu.
