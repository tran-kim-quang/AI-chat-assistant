## Tổng quan
AI-chatbot-assistant có các chức năng:
- Đọc CSV từ file upload hoặc URL
- Phân tích và tóm tắt dataset
- Trả lời câu hỏi về dữ liệu, ảnh
- Hiển thị biểu đồ và thống kê
- Trò chuyện thường ngày

Dự án sử dụng LLM được cung cấp bởi [groq](https://groq.com/) - Nền tảng cung cấp API để sử dụng các LLM opensource với chi phí rẻ và tốc độ nhanh.

## Cài đặt Local

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Keys

Tạo file `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key
TEXT_MODEL="GROQ_MODEL1"
PICTURE_MODEL="GROQ_MODEL2"
```

### 3. Chạy ứng dụng

```bash
python app_multiturn.py
```

Truy cập: http://localhost:5000

## Giới hạn

- **Kích thước file**: Tối đa 10MB
- **Encoding**: Hỗ trợ UTF-8, Latin1, ISO-8859-1, CP1252
- **Format**: Chỉ chấp nhận file .csv



