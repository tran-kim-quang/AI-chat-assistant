# Hướng dẫn sử dụng AI CSV Analyzer

## Tổng quan

AI CSV Analyzer là ứng dụng web phân tích dữ liệu CSV thông minh sử dụng AI để:
- Đọc CSV từ file upload hoặc URL
- Phân tích và tóm tắt dataset
- Trả lời câu hỏi về dữ liệu
- Hiển thị biểu đồ và thống kê

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Keys

Tạo file `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key
OPENAI_API_KEY=sk_your_openai_api_key
TEXT_MODEL=llama-3.3-70b-versatile
PICTURE_MODEL=gpt-4o-mini
```

### 3. Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

## Sử dụng Web Interface

### Bước 1: Upload CSV

Có 2 cách upload:

#### Cách 1: Upload file từ máy tính
1. Click vào vùng "Upload File"
2. Chọn file CSV (tối đa 10MB)
3. Hệ thống sẽ hiển thị preview dữ liệu

#### Cách 2: Nhập URL
1. Chuyển sang tab "URL (GitHub)"
2. Nhập URL CSV (ví dụ: GitHub raw link)
3. Click "Tải từ URL"

**Ví dụ URL:**
```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

### Bước 2: Phân tích dữ liệu

Sau khi upload thành công, bạn có thể:

#### A. Phân tích nhanh

Click các nút:
- **Tóm tắt Dataset**: Xem thông tin tổng quan (số dòng, cột, kiểu dữ liệu)
- **Thống kê cơ bản**: Xem min, max, mean, median của các cột số
- **Missing Values**: Tìm các cột có giá trị bị thiếu
- **Histogram**: Vẽ biểu đồ phân bố cho cột số

#### B. Hỏi AI

Nhập câu hỏi vào ô "Đặt câu hỏi" và click "Hỏi AI".

**Ví dụ câu hỏi:**
- "Ai có lương cao nhất?"
- "Có bao nhiêu người ở Hà Nội?"
- "Lương trung bình là bao nhiêu?"
- "Nghề nào phổ biến nhất?"
- "So sánh lương giữa các thành phố"

## Sử dụng qua Code

### 1. Load CSV từ file

```python
from config.process import load_csv_from_source, csv_process

# Load từ file
df = load_csv_from_source("data.csv")

# Hoặc phân tích với câu hỏi
result = csv_process("data.csv", "Tóm tắt dataset")
print(result)
```

### 2. Load CSV từ URL

```python
# Load từ GitHub raw link
url = "https://raw.githubusercontent.com/.../data.csv"
df = load_csv_from_source(url)

# Hoặc trực tiếp
result = csv_process(url, "Show basic stats")
print(result)
```

### 3. Các hàm phân tích chuyên biệt

```python
from config.process import (
    summarize_dataset,
    basic_stats,
    find_missing_values,
    plot_histogram
)

# Tóm tắt
summary = summarize_dataset(df)
print(summary)

# Thống kê
stats = basic_stats(df)
print(stats)

# Missing values
missing = find_missing_values(df)
print(missing)

# Histogram
hist = plot_histogram(df, "Lương")
print(hist)
```

## Các loại câu hỏi được hỗ trợ

### 1. Tóm tắt dataset
Keywords: `summarize`, `tóm tắt`, `overview`, `tổng quan`

**Ví dụ:**
- "Summarize the dataset"
- "Tóm tắt dataset"
- "Cho tôi tổng quan về dữ liệu"

### 2. Thống kê cơ bản
Keywords: `basic stats`, `thống kê`, `statistics`, `stats`

**Ví dụ:**
- "Show basic stats"
- "Thống kê các cột số"
- "Hiển thị statistics"

### 3. Missing values
Keywords: `missing`, `null`, `nan`, `thiếu`

**Ví dụ:**
- "Which column has missing values?"
- "Cột nào có giá trị thiếu?"
- "Tìm null values"

### 4. Histogram
Keywords: `histogram`, `plot`, `biểu đồ`

**Ví dụ:**
- "Plot a histogram of Lương"
- "Vẽ biểu đồ cho cột Tuổi"
- "Histogram of price"

### 5. Câu hỏi tùy chỉnh
Bất kỳ câu hỏi nào khác sẽ được AI phân tích.

**Ví dụ:**
- "Ai có lương cao nhất?"
- "What is the average age?"
- "So sánh giữa các nhóm"
- "Xu hướng như thế nào?"

## API Endpoints

Nếu muốn tích hợp vào ứng dụng khác:

### POST /upload
Upload file CSV

**Request:**
```
Content-Type: multipart/form-data
file: CSV file
```

**Response:**
```json
{
    "success": true,
    "filename": "data.csv",
    "preview": "<table>...</table>",
    "summary": "..."
}
```

### POST /load_url
Load CSV từ URL

**Request:**
```json
{
    "url": "https://..."
}
```

**Response:**
```json
{
    "success": true,
    "url": "https://...",
    "preview": "<table>...</table>",
    "summary": "..."
}
```

### POST /analyze
Phân tích với câu hỏi

**Request:**
```json
{
    "question": "Ai có lương cao nhất?"
}
```

**Response:**
```json
{
    "success": true,
    "result": "..."
}
```

### POST /quick_analysis/{type}
Phân tích nhanh

**Types:** `summarize`, `stats`, `missing`, `histogram`

**Response:**
```json
{
    "success": true,
    "result": "..."
}
```

## Giới hạn

- **Kích thước file**: Tối đa 10MB
- **Số dòng**: Mặc định 10,000 dòng (có thể điều chỉnh)
- **Encoding**: Hỗ trợ UTF-8, Latin1, ISO-8859-1, CP1252
- **Format**: Chỉ chấp nhận file .csv

## Troubleshooting

### Lỗi: "File quá lớn"
**Giải pháp:**
- Chia nhỏ file hoặc lọc dữ liệu trước
- Giới hạn hiện tại: 10MB

### Lỗi: "Không thể đọc file"
**Kiểm tra:**
- File có đúng định dạng CSV?
- Có ký tự đặc biệt không?
- Thử encoding khác

### Lỗi: "Không tải được từ URL"
**Kiểm tra:**
- URL có public không?
- Có kết nối internet?
- GitHub raw link phải dạng: `https://raw.githubusercontent.com/...`

### AI không trả lời đúng
**Lưu ý:**
- Đảm bảo có GROQ_API_KEY trong .env
- Câu hỏi càng rõ ràng càng tốt
- AI chỉ dựa trên dữ liệu mẫu (head) nên có thể không chính xác 100%

## Ví dụ hoàn chỉnh

### Phân tích dataset nhân viên

1. **Upload file** `test_data.csv`
2. **Click "Tóm tắt Dataset"** → Xem tổng quan
3. **Click "Thống kê cơ bản"** → Xem min/max/mean của Tuổi, Lương
4. **Click "Missing Values"** → Kiểm tra dữ liệu thiếu
5. **Click "Histogram"** → Chọn cột "Lương" → Xem phân bố
6. **Hỏi AI:** "Ai có lương cao nhất?" → AI trả lời dựa trên dữ liệu
7. **Hỏi AI:** "Lương trung bình ở Hà Nội?" → AI phân tích và trả lời

## Tips

- Sử dụng GitHub raw links để share dataset dễ dàng
- Đặt câu hỏi rõ ràng, cụ thể
- Kiểm tra preview trước khi phân tích
- Với dataset lớn, hãy lọc trước khi upload
- Sử dụng "Thống kê cơ bản" để hiểu dữ liệu trước khi hỏi AI

## Tích hợp vào dự án

Bạn có thể tích hợp CSV Analyzer vào dự án của mình:

```python
from config.process import csv_process

# Trong Flask route
@app.route('/analyze_data')
def analyze_data():
    result = csv_process('data.csv', 'Tóm tắt dataset')
    return render_template('result.html', data=result)
```

## License

MIT License

