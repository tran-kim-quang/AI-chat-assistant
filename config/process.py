import openai
import os
import base64
import pandas as pd
import requests
import io
from typing import Union, Dict, Any
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model_text = os.getenv('TEXT_MODEL')
model_img = os.getenv('PICTURE_MODEL')

# Process text
def text_process(content: str) -> str:
    """Xử lý văn bản với LLM"""
    try:
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )

        response = client.chat.completions.create(
            model=model_text,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo thân thiện"},
                {"role": "user", "content": content}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi xử lý văn bản: {str(e)}"

# Process img
def encode_image_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Xác định loại ảnh
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"
    
    return f"data:{mime_type};base64,{encoded}"

def img_process(path: str, user_prompt: str = None) -> str:
    try:
        # Encode ảnh
        base64_image = encode_image_to_base64(path)
        
        # Sử dụng prompt tùy chỉnh hoặc mặc định
        prompt = user_prompt if user_prompt else "Mô tả bức ảnh này"
        
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )

        response = client.chat.completions.create(
            model=model_img,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
    except FileNotFoundError as e:
        return f"Lỗi: {str(e)}"
    except Exception as e:
        return f"Lỗi xử lý ảnh: {str(e)}"

# CSV Helper Functions
def load_csv_from_source(source: Union[str, io.BytesIO], max_rows: int = 10000) -> pd.DataFrame:
    """
    Load CSV từ nhiều nguồn: file path, URL, hoặc file upload
    
    Args:
        source: Đường dẫn file, URL, hoặc file object
        max_rows: Số dòng tối đa để đọc
    
    Returns:
        pandas DataFrame
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
    
    # Nếu là URL
    if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
        try:
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            
            # Thử decode với nhiều encoding
            for encoding in encodings:
                try:
                    csv_content = response.content.decode(encoding)
                    df = pd.read_csv(io.StringIO(csv_content), nrows=max_rows)
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            raise ValueError("Không thể đọc CSV từ URL với các encoding phổ biến")
        
        except requests.RequestException as e:
            raise ValueError(f"Không thể tải CSV từ URL: {str(e)}")
    
    # Nếu là file path
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Không tìm thấy file: {source}")
        
        # Kiểm tra kích thước file
        file_size = os.path.getsize(source)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise ValueError(f"File quá lớn ({file_size / (1024*1024):.2f}MB). Giới hạn 10MB.")
        
        # Đọc với nhiều encoding
        for encoding in encodings:
            try:
                df = pd.read_csv(source, encoding=encoding, nrows=max_rows)
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Không thể đọc file với các encoding phổ biến")
    
    # Nếu là file object (uploaded file)
    else:
        for encoding in encodings:
            try:
                source.seek(0)  # Reset file pointer
                df = pd.read_csv(source, encoding=encoding, nrows=max_rows)
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Không thể đọc file với các encoding phổ biến")


def summarize_dataset(df: pd.DataFrame) -> str:
    """Tóm tắt thông tin dataset"""
    summary = f"""
📊 TỔNG QUAN DATASET
{'='*50}
Thông tin cơ bản:
  - Số dòng: {len(df):,}
  - Số cột: {len(df.columns)}
  - Tên các cột: {', '.join(df.columns.tolist())}

Kiểu dữ liệu:
{df.dtypes.value_counts().to_string()}

Bộ nhớ sử dụng: {df.memory_usage(deep=True).sum() / 1024:.2f} KB

Các cột số: {', '.join(df.select_dtypes(include=['int64', 'float64']).columns.tolist())}
Các cột text: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}
"""
    return summary


def basic_stats(df: pd.DataFrame) -> str:
    """Thống kê cơ bản cho các cột số"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_cols.columns) == 0:
        return "Dataset không có cột số nào để thống kê."
    
    stats = f"""
THỐNG KÊ CÁC CỘT SỐ
{'='*50}
{numeric_cols.describe().to_string()}

PHÂN BỐ:
"""
    
    for col in numeric_cols.columns:
        stats += f"\n{col}:"
        stats += f"\n  - Min: {numeric_cols[col].min()}"
        stats += f"\n  - Max: {numeric_cols[col].max()}"
        stats += f"\n  - Mean: {numeric_cols[col].mean():.2f}"
        stats += f"\n  - Median: {numeric_cols[col].median():.2f}"
        stats += f"\n  - Std Dev: {numeric_cols[col].std():.2f}"
    
    return stats


def find_missing_values(df: pd.DataFrame) -> str:
    """Tìm các cột có missing values"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Cột': missing.index,
        'Số missing': missing.values,
        'Tỷ lệ (%)': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Số missing'] > 0].sort_values('Số missing', ascending=False)
    
    if len(missing_df) == 0:
        return "Dataset không có giá trị missing!"
    
    result = f"""
PHÂN TÍCH MISSING VALUES
{'='*50}
{missing_df.to_string(index=False)}

Cột có nhiều missing nhất: {missing_df.iloc[0]['Cột']} ({missing_df.iloc[0]['Tỷ lệ (%)']:.2f}%)
"""
    return result


def plot_histogram(df: pd.DataFrame, column: str) -> str:
    """Tạo histogram dạng text cho một cột số"""
    if column not in df.columns:
        return f"Không tìm thấy cột '{column}'"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Cột '{column}' không phải là cột số"
    
    # Loại bỏ NaN
    data = df[column].dropna()
    
    if len(data) == 0:
        return f"Cột '{column}' không có dữ liệu"
    
    # Tạo histogram text-based
    hist, bins = pd.cut(data, bins=10, retbins=True, duplicates='drop')
    counts = hist.value_counts().sort_index()
    
    max_count = counts.max()
    bar_width = 50
    
    result = f"""
HISTOGRAM: {column}
{'='*60}
Min: {data.min():.2f} | Max: {data.max():.2f} | Mean: {data.mean():.2f}
{'='*60}
"""
    
    for interval, count in counts.items():
        bar_len = int((count / max_count) * bar_width)
        bar = '█' * bar_len
        result += f"\n{str(interval):30s} | {bar} {count}"
    
    return result


# Process CSV - Main Function
def csv_process(source: Union[str, io.BytesIO], user_question: str = None, max_rows: int = 10000) -> str:
    try:
        # Load CSV từ source
        df = load_csv_from_source(source, max_rows)
        
        # Kiểm tra empty
        if len(df) == 0:
            return "❌ File CSV không có dữ liệu"
        
        # Nếu không có câu hỏi, trả về tổng quan
        if not user_question:
            return summarize_dataset(df) + "\n\n" + df.head(10).to_string()
        
        # Xử lý câu hỏi theo keyword
        question_lower = user_question.lower()
        
        # Summarize dataset
        if any(word in question_lower for word in ['summarize', 'tóm tắt', 'overview', 'tổng quan']):
            return summarize_dataset(df) + "\n\n📋 DỮ LIỆU MẪU:\n" + df.head().to_string()
        
        # Basic stats
        elif any(word in question_lower for word in ['basic stats', 'thống kê', 'statistics', 'stats']):
            return basic_stats(df)
        
        # Missing values
        elif any(word in question_lower for word in ['missing', 'null', 'nan', 'thiếu']):
            return find_missing_values(df)
        
        # Histogram/plot
        elif 'histogram' in question_lower or 'plot' in question_lower or 'biểu đồ' in question_lower:
            # Tìm tên cột trong câu hỏi
            for col in df.columns:
                if col.lower() in question_lower:
                    return plot_histogram(df, col)
            
            # Nếu không tìm thấy cột, hỏi user
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                return f"Vui lòng chỉ định cột cần vẽ histogram. Các cột số: {', '.join(numeric_cols)}"
            else:
                return "Dataset không có cột số nào để vẽ histogram"
        
        # Câu hỏi tùy chỉnh - dùng AI
        else:
            return analyze_with_ai(df, user_question)
        
    except (FileNotFoundError, ValueError) as e:
        return f"❌ Lỗi: {str(e)}"
    except pd.errors.EmptyDataError:
        return "❌ File CSV trống"
    except pd.errors.ParserError:
        return "❌ Không thể đọc file CSV. Vui lòng kiểm tra định dạng."
    except Exception as e:
        return f"❌ Lỗi xử lý CSV: {str(e)}"


def analyze_with_ai(df: pd.DataFrame, question: str) -> str:
    """Sử dụng AI để trả lời câu hỏi về dataset"""
    try:
        # Tạo context về dataset
        context = f"""
Thông tin dataset:
- Số dòng: {len(df)}
- Số cột: {len(df.columns)}
- Các cột: {', '.join(df.columns.tolist())}

Dữ liệu mẫu (5 dòng đầu):
{df.head().to_string()}

Thống kê cơ bản:
{df.describe().to_string()}
"""
        
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
        
        response = client.chat.completions.create(
            model=model_text,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu. Trả lời câu hỏi dựa trên dữ liệu được cung cấp một cách chi tiết và chính xác."},
                {"role": "user", "content": f"{context}\n\nCâu hỏi: {question}"}
            ]
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ Lỗi phân tích với AI: {str(e)}"
