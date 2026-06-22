# Phân tích dataset Restaurant và Laptop

## 1. Cách thống kê

Thống kê được tính trực tiếp từ:

- `dataset/rest/train.multiple.json`
- `dataset/rest/test.multiple.json`
- `dataset/lap/train.multiple.json`
- `dataset/lap/test.multiple.json`

Một câu có thể chứa nhiều aspect. Vì vậy cần phân biệt:

- **Số câu**: số record JSON.
- **Số mẫu aspect**: tổng số aspect sau khi bung mỗi aspect thành một mẫu ABSA.
- **Train sạch**: train sau khi loại toàn bộ câu xuất hiện trong official test.

## 2. Quy mô dữ liệu

| Dataset | Split | Số câu | Mẫu aspect thô | Aspect/câu | Câu nhiều aspect |
|---|---|---:|---:|---:|---:|
| Restaurant | Train | 1.980 | 3.608 | 1,82 | 971 (49,04%) |
| Restaurant | Test | 600 | 1.120 | 1,87 | 315 (52,50%) |
| Laptop | Train | 1.466 | 2.328 | 1,59 | 548 (37,38%) |
| Laptop | Test | 411 | 638 | 1,55 | 152 (36,98%) |

Restaurant lớn hơn Laptop:

- Train nhiều hơn 514 câu và 1.280 mẫu aspect.
- Test nhiều hơn 189 câu và 482 mẫu aspect.
- Train Restaurant có nhiều mẫu aspect hơn khoảng 54,98%.
- Câu Restaurant thường chứa nhiều aspect hơn, làm nhiệm vụ tách sentiment theo từng target quan trọng hơn.

## 3. Phân phối nhãn

### 3.1. Train thô

| Dataset | Negative | Neutral | Positive | Tổng |
|---|---:|---:|---:|---:|
| Restaurant | 807 (22,37%) | 637 (17,66%) | 2.164 (59,98%) | 3.608 |
| Laptop | 870 (37,37%) | 464 (19,93%) | 994 (42,70%) | 2.328 |

Restaurant mất cân bằng mạnh về positive. Laptop cân bằng hơn giữa positive và negative, nhưng neutral vẫn là lớp nhỏ nhất.

Tỷ lệ lớp lớn nhất/lớp nhỏ nhất:

- Restaurant: `positive / neutral = 3,40`.
- Laptop: `positive / neutral = 2,14`.

Do đó Accuracy có nguy cơ đánh giá quá lạc quan trên Restaurant nhiều hơn Laptop.

### 3.2. Official test

| Dataset | Negative | Neutral | Positive | Tổng |
|---|---:|---:|---:|---:|
| Restaurant | 196 (17,50%) | 196 (17,50%) | 728 (65,00%) | 1.120 |
| Laptop | 128 (20,06%) | 169 (26,49%) | 341 (53,45%) | 638 |

Restaurant test còn thiên positive hơn train. Laptop cũng tăng positive trong test, nhưng thay đổi đáng chú ý nhất là:

- Negative giảm từ 37,37% xuống 20,06%.
- Neutral tăng từ 19,93% lên 26,49%.
- Positive tăng từ 42,70% lên 53,45%.

Sự dịch chuyển phân phối train–test của Laptop lớn hơn Restaurant và có thể làm class weighting học từ train kém phù hợp với test.

## 4. Độ dài câu

Độ dài được tính bằng số token có sẵn trong trường `tokens`.

| Dataset | Split | Trung bình | Trung vị | P95 | Lớn nhất |
|---|---|---:|---:|---:|---:|
| Restaurant | Train | 20,30 | 19 | 40 | 79 |
| Restaurant | Test | 19,40 | 18 | 40 | 70 |
| Laptop | Train | 22,34 | 20 | 47 | 83 |
| Laptop | Test | 18,57 | 16 | 42 | 78 |

Với `max_length=128`, phần lớn câu không bị cắt theo độ dài token gốc. Laptop train có câu dài và phân tán hơn Restaurant train, nhưng vẫn cách khá xa ngưỡng 128.

## 5. Độ đa dạng aspect

| Dataset | Split | Unique aspect | Mẫu/unique aspect |
|---|---|---:|---:|
| Restaurant | Train | 1.202 | 3,00 |
| Restaurant | Test | 520 | 2,15 |
| Laptop | Train | 949 | 2,45 |
| Laptop | Test | 389 | 1,64 |

Laptop có ít dữ liệu hơn nhưng aspect vẫn rất đa dạng. Số mẫu trên mỗi unique aspect thấp hơn Restaurant, nghĩa là nhiều aspect Laptop chỉ xuất hiện rất ít lần.

Các aspect phổ biến nhất trong train:

| Restaurant | Số mẫu | Laptop | Số mẫu |
|---|---:|---|---:|
| food | 361 | screen | 61 |
| service | 225 | price | 56 |
| prices | 63 | use | 54 |
| place | 59 | battery life | 53 |
| menu | 56 | keyboard | 50 |
| dinner | 56 | battery | 47 |
| staff | 55 | programs | 37 |
| pizza | 51 | features | 35 |
| atmosphere | 46 | software | 33 |
| price | 41 | warranty | 32 |

Restaurant tập trung mạnh vào một số aspect tổng quát như `food` và `service`. Laptop phân tán hơn giữa linh kiện, hệ điều hành, phần mềm, hiệu năng và thuộc tính sử dụng.

## 6. Train–test overlap và unseen aspect

Mỗi dataset có đúng một câu train trùng official test:

| Dataset | Mẫu bị loại | Nhãn |
|---|---:|---|
| Restaurant | 1 | Positive |
| Laptop | 1 | Neutral |

Sau khi loại overlap:

| Dataset | Mẫu train sạch | Unique aspect train sạch |
|---|---:|---:|
| Restaurant | 3.607 | 1.201 |
| Laptop | 2.327 | 949 |

Nếu so aspect test trực tiếp với toàn bộ train sạch:

| Dataset | Unseen test samples | Tỷ lệ | Unique unseen aspect |
|---|---:|---:|---:|
| Restaurant | 353/1.120 | 31,52% | 334 |
| Laptop | 268/638 | 42,01% | 235 |

Phân phối nhãn trong nhóm unseen:

| Dataset | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Restaurant | 45 (12,75%) | 82 (23,23%) | 226 (64,02%) |
| Laptop | 64 (23,88%) | 94 (35,07%) | 110 (41,04%) |

Nhóm unseen Laptop không chỉ lớn hơn mà còn cân bằng hơn và chứa nhiều neutral hơn. Đây là nhóm khó, ít được hưởng lợi từ bias positive và có thể giải thích một phần chênh lệch Accuracy giữa hai miền.

Lưu ý: notebook có thể báo tỷ lệ unseen khác giữa các validation split vì tập tham chiếu là phần train của từng split. Con số 31,52% và 42,01% ở đây dùng toàn bộ train sạch làm tập tham chiếu và cùng cách so khớp aspect của pipeline.

## 7. Chất lượng và rủi ro dữ liệu

### Restaurant

- Quy mô lớn hơn và có nhiều ví dụ lặp lại cho các aspect phổ biến.
- Mất cân bằng positive mạnh, vì vậy Accuracy dễ cao hơn chất lượng thực ở lớp thiểu số.
- Một số aspect gần nghĩa nhưng được giữ thành chuỗi khác nhau, ví dụ `price` và `prices`.
- Unseen aspect vẫn chiếm gần một phần ba test.

### Laptop

- Phân phối train cân bằng hơn, phù hợp để học negative tốt hơn.
- Tập nhỏ hơn nhưng vocabulary aspect rộng, làm số ví dụ trên mỗi aspect thấp.
- Có domain shift nhãn đáng kể giữa train và test.
- Unseen aspect chiếm hơn 42% test.
- Nhiều biến thể tên gọi có thể cùng chỉ một khái niệm, ví dụ hệ điều hành, phiên bản Windows, pin và battery life.

## 8. Hàm ý cho mô hình và thí nghiệm

- Dùng Macro-F1 làm metric chính; không chọn mô hình chỉ theo Accuracy.
- Giữ split theo câu vì tỷ lệ câu nhiều aspect cao ở cả hai dataset.
- Tuning class weight riêng cho từng dataset. Kết quả thực nghiệm đã cho thấy neutral weight 1.25 thắng trên Restaurant nhưng không thắng robust score trên Laptop.
- Thử chuẩn hóa aspect theo concept, số ít/số nhiều và alias để giảm unseen rate.
- Chạy aspect-description ablation, đặc biệt trên Laptop, vì mô tả có thể hỗ trợ aspect hiếm hoặc chưa gặp.
- Báo cáo metric từng lớp và confusion matrix để xác định neutral hay negative là điểm nghẽn.
- Phân tích riêng seen/unseen và frequent/rare aspect thay vì chỉ dùng một metric tổng.
- Khi so sánh hai miền, giữ nguyên pipeline nhưng không giả định hyperparameter tối ưu có thể chuyển trực tiếp từ Restaurant sang Laptop.

## 9. Kết luận

Restaurant có lợi thế về quy mô nhưng mất cân bằng positive mạnh. Laptop cân bằng hơn trong train, song khó hơn do tập nhỏ, aspect phân tán, dịch chuyển nhãn train–test và tỷ lệ unseen aspect cao.

Các đặc tính này phù hợp với kết quả mô hình: Laptop có Accuracy và Macro-F1 thấp hơn, calibration kém hơn và nhạy seed hơn. Ưu tiên cải thiện cho Laptop nên là biểu diễn unseen aspect, chuẩn hóa aspect và tăng độ ổn định qua nhiều seed.
