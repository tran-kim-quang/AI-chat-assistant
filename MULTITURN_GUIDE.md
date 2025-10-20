# Hướng dẫn Hệ thống Multi-turn Conversation

## Tổng quan

Hệ thống AI Chat Assistant với khả năng multi-turn conversation thông minh, sử dụng **AI để phân loại intent** (KHÔNG dùng keyword matching).

### Đặc điểm chính

✅ **AI Intent Classification**: Dùng LLM để hiểu ý định người dùng  
✅ **Multi-turn Conversation**: Duy trì ngữ cảnh qua nhiều lượt hội thoại  
✅ **Session Management**: Quản lý session với SQLite database  
✅ **Context Awareness**: Nhớ files/ảnh đã upload, lịch sử chat  
✅ **Auto-routing**: Tự động chuyển đến handler phù hợp  

## Kiến trúc Hệ thống

```
┌─────────────┐
│   User      │
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────┐
│   Flask App (app_multiturn.py)     │
│   - Session init                    │
│   - File upload                     │
│   - Message routing                 │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ ConversationHandler                 │
│  - Process message                  │
│  - Coordinate components            │
└───────┬───────────┬─────────────────┘
        │           │
        ▼           ▼
┌───────────┐ ┌───────────────┐
│  Intent   │ │   Session     │
│Classifier │ │   Manager     │
│           │ │               │
│- AI-based │ │- SQLite DB    │
│- No       │ │- History      │
│ keywords  │ │- Context      │
└─────┬─────┘ └───────┬───────┘
      │               │
      ▼               ▼
┌──────────────────────────┐
│   Process Modules        │
│  - csv_process()         │
│  - img_process()         │
│  - text_process()        │
└──────────────────────────┘
```

## Cài đặt

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Keys

File `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
OPENAI_API_KEY=sk_your_key_here
TEXT_MODEL=llama-3.3-70b-versatile
PICTURE_MODEL=gpt-4o-mini
```

### 3. Chạy ứng dụng

```bash
python app_multiturn.py
```

Truy cập: http://localhost:5000

## Cách hoạt động

### Flow xử lý tin nhắn

```
1. User gửi message
         ↓
2. IntentClassifier phân tích với AI
   - Gửi message + context summary đến LLM
   - LLM trả về: intent, confidence, reasoning
         ↓
3. ConversationHandler routing
   - csv_analysis → Xử lý với csv_process()
   - image_analysis → Xử lý với img_process()
   - general_chat → Chat với LLM
   - clarification → Hỏi lại user
         ↓
4. Lưu vào SessionManager
   - Message history
   - Context updates
         ↓
5. Trả response cho user
```

### Intent Classification

Sử dụng LLM prompt engineering:

```python
Phân tích tin nhắn: "Ai có lương cao nhất?"

NGỮ CẢNH:
- File CSV: employees.csv (10 dòng, 5 cột)
- Lịch sử: User vừa upload file CSV

→ AI phân loại:
{
    "intent": "csv_analysis",
    "confidence": 0.95,
    "reasoning": "Câu hỏi về dữ liệu trong CSV vừa upload",
    "suggested_action": "Analyze CSV with question"
}
```

**Ưu điểm so với keyword matching:**
- Hiểu ngữ nghĩa phức tạp
- Xem xét ngữ cảnh (context)
- Linh hoạt với cách diễn đạt khác nhau
- Không bị giới hạn bởi keywords cố định

## Các Module chính

### 1. SessionManager (`config/session_manager.py`)

Quản lý session và lưu trữ:

```python
from config.session_manager import SessionManager

session_mgr = SessionManager()

# Tạo session
session_id = session_mgr.create_session()

# Thêm message
session_mgr.add_message(
    session_id=session_id,
    role='user',
    content='Hello',
    intent='general_chat'
)

# Lấy lịch sử
history = session_mgr.get_conversation_history(session_id)

# Thêm context (file uploaded)
session_mgr.add_context(
    session_id=session_id,
    context_type='csv',
    context_data={'filename': 'data.csv', 'rows': 100}
)
```

**Database Schema:**

- `sessions`: session_id, created_at, last_activity, metadata
- `messages`: id, session_id, role, content, intent, context, timestamp
- `context_storage`: id, session_id, context_type, context_data, created_at

### 2. IntentClassifier (`config/intent_classifier.py`)

Phân loại intent bằng AI:

```python
from config.intent_classifier import get_intent_classifier

classifier = get_intent_classifier()

result = classifier.classify(
    user_message="Tóm tắt dataset này",
    context_summary="File CSV: data.csv đã upload"
)

print(result)
# {
#     'intent': 'csv_analysis',
#     'confidence': 0.9,
#     'reasoning': '...',
#     'suggested_action': '...'
# }
```

**Các Intent được hỗ trợ:**

- `csv_analysis`: Phân tích/hỏi về CSV
- `image_analysis`: Phân tích/hỏi về ảnh
- `general_chat`: Trò chuyện thông thường
- `upload_request`: Yêu cầu upload file
- `clarification`: Cần làm rõ

### 3. ConversationHandler (`config/conversation_handler.py`)

Xử lý multi-turn conversation:

```python
from config.conversation_handler import get_conversation_handler

handler = get_conversation_handler()

# Xử lý message
result = handler.process_message(
    session_id=session_id,
    user_message="Ai có lương cao nhất?"
)

print(result['response'])
```

**Chức năng:**
- Route theo intent
- Duy trì context
- Tích hợp process modules
- Trả lời intelligent

### 4. Flask App (`app_multiturn.py`)

Web interface với API endpoints:

**Endpoints:**

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Trang chat |
| `/api/init_session` | POST | Khởi tạo session |
| `/api/chat` | POST | Gửi message |
| `/api/upload_file` | POST | Upload file |
| `/api/upload_url` | POST | Upload từ URL |
| `/api/history` | GET | Lấy lịch sử |
| `/api/clear_session` | POST | Xóa session |
| `/api/session_info` | GET | Thông tin session |

## Ví dụ Sử dụng

### Scenario 1: Phân tích CSV

```
User: Xin chào
Bot: Xin chào! Tôi có thể giúp gì cho bạn?
     [Intent: general_chat, Confidence: 0.95]

User: [Upload employees.csv]
Bot: File CSV đã được upload! (10 dòng, 5 cột)
     Bạn có thể hỏi tôi về dữ liệu.
     [Intent: file_uploaded]

User: Có bao nhiêu người ở Hà Nội?
Bot: Dựa trên dữ liệu, có 4 người làm việc ở Hà Nội.
     [Intent: csv_analysis, Confidence: 0.92]

User: Ai có lương cao nhất?
Bot: Hoàng Văn E có lương cao nhất: 40,000,000 VNĐ
     [Intent: csv_analysis, Confidence: 0.95]
```

### Scenario 2: Phân tích Ảnh

```
User: Tôi muốn phân tích một bức ảnh
Bot: Vui lòng upload ảnh để tôi phân tích.
     [Intent: upload_request, Confidence: 0.88]

User: [Upload cat.jpg]
Bot: Ảnh đã được upload! Bạn có thể hỏi về ảnh này.
     [Intent: file_uploaded]

User: Mô tả ảnh này
Bot: Bức ảnh cho thấy một con mèo đang nằm trên ghế sofa...
     [Intent: image_analysis, Confidence: 0.91]

User: Màu gì là chủ đạo?
Bot: Màu chủ đạo trong ảnh là màu cam và trắng...
     [Intent: image_analysis, Confidence: 0.87]
```

### Scenario 3: Mixed Conversation

```
User: Bạn có thể làm gì?
Bot: Tôi có thể:
     1. Phân tích dữ liệu CSV
     2. Phân tích ảnh
     3. Trò chuyện thông thường
     [Intent: clarification, Confidence: 0.93]

User: [Upload data.csv]
Bot: File CSV đã upload! (100 dòng)
     [Intent: file_uploaded]

User: Thống kê cơ bản
Bot: [Hiển thị stats của các cột số]
     [Intent: csv_analysis, Confidence: 0.96]

User: Cảm ơn bạn
Bot: Rất vui được giúp đỡ! Có gì khác tôi có thể hỗ trợ không?
     [Intent: general_chat, Confidence: 0.89]
```

## Context Awareness

### Context được lưu trữ

1. **File Context**:
   - Filename
   - Filepath/URL
   - Metadata (rows, columns cho CSV)
   - Upload timestamp

2. **Conversation History**:
   - Role (user/assistant)
   - Content
   - Intent
   - Timestamp

3. **Session Metadata**:
   - User agent
   - IP address
   - Created time
   - Last activity

### Context Summary

```python
context_summary = session_mgr.get_recent_context_summary(session_id)

# Output:
"""
Context hiện tại:
- File CSV: employees.csv

Lịch sử hội thoại gần đây:
user: Xin chào
assistant: Xin chào! Tôi có thể giúp gì?
user: Upload employees.csv
assistant: File đã được upload!
user: Ai có lương cao nhất?
"""
```

Context này được gửi kèm cho IntentClassifier để phân loại chính xác hơn.

## Tối ưu hóa

### 1. Cache Intent Results

```python
# TODO: Implement caching cho similar messages
# Giảm số lần gọi API
```

### 2. Batch Processing

```python
# TODO: Batch multiple messages
# Xử lý nhiều message cùng lúc
```

### 3. Context Pruning

```python
# Giới hạn context size
# Chỉ giữ N messages gần nhất
context_summary = get_recent_context_summary(
    session_id, 
    limit=10  # 10 messages gần nhất
)
```

## Testing

Chạy test:

```bash
python test_multiturn_system.py
```

Test cases:
- Intent classification accuracy
- Multi-turn context retention
- Session persistence
- File upload và context update
- Error handling

## Troubleshooting

### Lỗi: "IntentClassifier failed"
**Nguyên nhân**: GROQ_API_KEY không hợp lệ hoặc rate limit  
**Giải pháp**: Kiểm tra API key, dùng fallback classification

### Lỗi: "Session not found"
**Nguyên nhân**: Session đã bị xóa hoặc expire  
**Giải pháp**: Init session mới

### Intent classification không chính xác
**Nguyên nhân**: Context summary không đủ thông tin  
**Giải pháp**: Tăng history limit, cải thiện prompt

## Roadmap

- [ ] Support streaming responses
- [ ] Add conversation branching
- [ ] Implement RAG for long-term memory
- [ ] Multi-modal input (voice, video)
- [ ] Fine-tune intent classifier
- [ ] Add analytics dashboard
- [ ] Export conversation history
- [ ] Multi-language support

## License

MIT License

