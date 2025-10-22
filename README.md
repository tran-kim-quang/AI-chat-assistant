## [Demo Video](https://drive.google.com/file/d/1PaP79yMtC3y9cKsg8UHjEa4uPZg3322v/view?usp=sharing)
## Tổng quan
AI-chatbot-assistant có các chức năng:
- Đọc CSV từ file upload hoặc URL
- Phân tích và tóm tắt dataset
- Trả lời câu hỏi về dữ liệu, ảnh
- Hiển thị biểu đồ và thống kê
- Trò chuyện thường ngày

Dự án sử dụng LLM được cung cấp bởi [groq](https://groq.com/) - Nền tảng cung cấp API để sử dụng các LLM opensource với chi phí rẻ và tốc độ nhanh.

Cuộc trò chuyện sẽ được đánh dấu theo từng session mỗi lần chạy và lưu vào [conversations.db](conversations.db)

Nội dung người dùng upload khi sử dụng chatbot sẽ được lưu vào thư mục [uploads](uploads)

## Cài đặt Local

### 1. Cài đặt dependencies
Tạo môi trường ảo venv
```bash
# Linux/Ubuntu
python3.12 -m venv venv
source venv/bin/activate
# Window
py -3.12 -m venv venv
venv\Scripts\activate.ps1
```
Tải các thư viện cần thiết
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
- **Context**: Chỉ nhớ được lịch sử 4 nội dung gần nhất gần nhất của cuộc trò chuyện (vẫn giữ được nội dung cần phân tích với file CSV hoặc ảnh được upload gần nhất)



