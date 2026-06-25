# Restaurant ABSA tiếng Việt

Đây là bản dịch máy toàn bộ dataset Restaurant từ tiếng Anh sang tiếng Việt.

## Các file chính

- `train.multiple.json`: 1.980 câu, 3.608 mẫu aspect.
- `test.multiple.json`: 600 câu, 1.120 mẫu aspect.
- `aspects.json`: thống kê các aspect tiếng Việt và term nguồn.
- `translation_metadata.json`: thông tin quy mô và phương thức chuyển đổi.
- `train.preview.json`: bản preview thủ công ban đầu, không dùng để train.

## Cấu trúc

Mỗi record giữ các trường cần cho pipeline:

```json
{
  "index": 0,
  "sentence": "nhưng nhân viên đối xử với chúng tôi thật kinh khủng .",
  "tokens": [
    "nhưng",
    "nhân",
    "viên",
    "đối",
    "xử",
    "với",
    "chúng",
    "tôi",
    "thật",
    "kinh",
    "khủng",
    "."
  ],
  "aspects": [
    {
      "term": ["nhân", "viên"],
      "polarity": "negative",
      "from": 1,
      "to": 3
    }
  ]
}
```

`["nhân", "viên"]` là một aspect duy nhất có nội dung `nhân viên`.

## Nguyên tắc chuyển đổi

- Dịch câu theo ngữ cảnh bằng Google Translate.
- Đánh dấu aspect bằng thẻ trong lúc dịch để giữ ranh giới.
- Giữ nguyên số aspect và nhãn sentiment.
- Tính lại `tokens`, `from` và `to` theo câu tiếng Việt.
- Giữ `source_sentence` và `source_term` để có thể đối chiếu bản tiếng Anh.
- Loại `heads`, `deprels` và `tags`, vì cú pháp tiếng Anh không còn đúng sau
  khi dịch.

Pipeline hiện tại chỉ sử dụng `sentence`, `aspects[].term` và
`aspects[].polarity`; các trường nguồn bổ sung không ảnh hưởng huấn luyện.

## Lưu ý chất lượng

Đây là dataset dịch máy, không phải corpus tiếng Việt được gán nhãn trực tiếp
bởi con người. Marker có thể bị dịch vụ dịch chuyển trong một số câu; script đã
có cơ chế khôi phục và toàn bộ vị trí aspect đã được kiểm tra tự động. Tuy
nhiên, trước khi dùng làm kết quả nghiên cứu chính thức vẫn nên rà soát thủ
công một mẫu ngẫu nhiên và các câu có nhiều aspect.

Script tái tạo dataset:

```powershell
python translate_rest_to_vi.py
```
