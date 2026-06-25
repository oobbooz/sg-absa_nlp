# Phân tích dataset Restaurant tiếng Việt

Tài liệu này tổng hợp thống kê của `dataset/rest_vi` trong package `nlp-res-vi.ipynb`.
Dataset là bản dịch máy của Restaurant ABSA sang tiếng Việt, giữ nguyên nhãn sentiment và
chỉ dùng các trường cần cho pipeline hiện tại: `sentence`, `tokens`, `aspects[].term`,
`aspects[].from`, `aspects[].to`, `aspects[].polarity`.

## 1. Quy mô dữ liệu

| Split | Số câu | Số mẫu aspect | Unique aspect |
|---|---:|---:|---:|
| Train | 1.980 | 3.608 | 1.224 |
| Test | 600 | 1.120 | 524 |

Mỗi câu có thể chứa nhiều aspect, nên số mẫu aspect lớn hơn số câu rất nhiều.

## 2. Phân phối nhãn

### 2.1. Train

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| Negative | 807 | 22,37% |
| Neutral | 637 | 17,66% |
| Positive | 2.164 | 59,98% |

### 2.2. Test

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| Negative | 196 | 17,50% |
| Neutral | 196 | 17,50% |
| Positive | 728 | 65,00% |

Dataset lệch mạnh về positive ở cả train lẫn test. Điều này giải thích vì sao
Accuracy có thể cao hơn Macro-F1 nếu mô hình thiên về lớp positive.

## 3. Aspect phổ biến

### 3.1. Train

| Aspect | Số mẫu |
|---|---:|
| đồ ăn | 226 |
| dịch vụ | 222 |
| giá | 92 |
| món ăn | 87 |
| địa điểm | 64 |
| thức ăn | 62 |
| nhân viên | 60 |
| pizza | 54 |
| không khí | 53 |
| bàn | 53 |

### 3.2. Test

| Aspect | Số mẫu |
|---|---:|
| đồ ăn | 90 |
| dịch vụ | 74 |
| giá | 29 |
| nhân viên | 24 |
| món ăn | 21 |
| địa điểm | 18 |
| người phục vụ | 18 |
| món khai vị | 18 |
| đồ uống | 17 |
| bầu không khí | 17 |

Hai aspect chi phối mạnh nhất vẫn là `đồ ăn` và `dịch vụ`, nên đây là miền ABSA khá “cổ điển” với
các target chung, đồng thời vẫn có nhiều biến thể từ vựng như `người phục vụ`, `bầu không khí`,
`món khai vị`.

## 4. Unseen aspect trong pipeline

Notebook `nlp-res-vi.ipynb` dùng cách chia validation theo câu và loại bỏ overlap train-test.
Từ log thực nghiệm của notebook:

| Split dùng cho training | Số mẫu | Unseen aspect samples | Unseen ratio |
|---|---:|---:|---:|
| Train | 3.064 | 148 | 4,10% |
| Validation | 542 | 148 | 27,31% |
| Test | 1.120 | 362 | 32,32% |

Trên tập train đầy đủ:

| Trạng thái | Số mẫu |
|---|---:|
| Mẫu train gốc | 3.606 |
| Mẫu bị loại do overlap train-test | 2 |

Điểm đáng chú ý là tỷ lệ unseen ở validation và test khá cao. Với ABSA, đây là nguồn khó chính vì mô hình
không chỉ phải học sentiment mà còn phải tổng quát hóa sang aspect chưa gặp.

## 5. Độ đa dạng và rủi ro dữ liệu

- Tập train có 1.224 unique aspect, cho thấy vocabulary target khá rộng.
- Nhiều aspect xuất hiện với các cách gọi gần nghĩa, ví dụ `nhân viên`, `người phục vụ`, `dịch vụ`.
- Tập test vẫn thiên về positive, nên một mô hình chỉ bám bias lớp có thể có Accuracy nhìn khá ổn nhưng Macro-F1 thấp.
- Phân phối unseen aspect ở test cao hơn validation, vì vậy kết quả validation không nên được diễn giải quá lạc quan.

## 6. Hệ quả cho mô hình

- Nên ưu tiên Macro-F1 thay vì chỉ nhìn Accuracy.
- `hybrid` và `aspect_attention` có lý do hợp lý để thử, vì target aspect đóng vai trò quan trọng hơn CLS thuần.
- Không nên giả định class weighting chung sẽ tối ưu cho mọi miền; với bản tiếng Việt này, cân bằng giữa bias positive
  và khả năng nhận diện negative/neutral vẫn cần tuning riêng.
- Nên kiểm tra riêng seen/unseen aspect khi phân tích kết quả cuối.

## 7. Kết luận

`rest_vi` là dataset ABSA tiếng Việt có quy mô vừa, lệch mạnh về positive, và có tỷ lệ unseen aspect cao ở
validation lẫn test. Đây là một miền đủ khó để đánh giá năng lực tổng quát hóa của pipeline, đặc biệt ở các lớp
negative và neutral.
