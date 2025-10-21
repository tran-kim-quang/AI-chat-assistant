import openai
import os
import base64
import pandas as pd
import requests
import io
from typing import Union, Dict, Any
from dotenv import load_dotenv
from tabulate import tabulate
from config.rich_message import RichMessage
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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
TỔNG QUAN DATASET
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


def plot_histogram(df: pd.DataFrame, column: str) -> Dict:
    """
    Tạo histogram cho một cột số và trả về base64 image
    
    Returns:
        Dict với 'text' và 'chart_base64'
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    if column not in df.columns:
        return {
            'text': f"Không tìm thấy cột '{column}'",
            'chart_base64': None
        }
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            'text': f"Cột '{column}' không phải là cột số",
            'chart_base64': None
        }
    
    # Loại bỏ NaN
    data = df[column].dropna()
    
    if len(data) == 0:
        return {
            'text': f"Cột '{column}' không có dữ liệu",
            'chart_base64': None
        }
    
    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram của {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Tần suất', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Thêm thống kê
    stats_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}\nMean: {data.mean():.2f}\nMedian: {data.median():.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Text description
    text_desc = f"""
 **Histogram của cột '{column}'**

Thống kê:
- Min: {data.min():.2f}
- Max: {data.max():.2f}  
- Mean: {data.mean():.2f}
- Median: {data.median():.2f}
- Std: {data.std():.2f}
"""
    
    return {
        'text': text_desc,
        'chart_base64': chart_base64,
        'chart_type': 'histogram',
        'column': column
    }


# Pandas Agent Helper
def create_csv_agent(df: pd.DataFrame):
    """Tạo pandas agent để phân tích CSV"""
    llm = ChatGroq(
        model=model_text,
        api_key=GROQ_API_KEY,
        temperature=0.0
    )
    
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        max_iterations=50,
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix="""Bạn là trợ lý phân tích dữ liệu chuyên nghiệp, thông minh và hiểu rõ ý định người dùng.

NHIỆM VỤ CỦA BẠN:
- Trả lời câu hỏi dựa trên dữ liệu thực tế một cách CHÍNH XÁC và NGẮN GỌN
- Sử dụng describe(), value_counts(), groupby(), agg() để phân tích
- Nếu người dùng hỏi về bảng: trả về kết quả dạng markdown table
- Nếu người dùng hỏi về số liệu: trả về con số cụ thể kèm giải thích ngắn
- KHÔNG giải thích code, CHỈ đưa ra KẾT QUẢ
- Format kết quả rõ ràng, dễ đọc

QUAN TRỌNG: Trả lời bằng Tiếng Việt một cách tự nhiên, chuyên nghiệp.
"""
    )
    
    return agent


# Process CSV - Main Function
def csv_process(source: Union[str, io.BytesIO], user_question: str = None, max_rows: int = 10000) -> Union[str, Dict]:
    try:
        # Load CSV từ source
        df = load_csv_from_source(source, max_rows)
        
        # Kiểm tra empty
        if len(df) == 0:
            return RichMessage.create_text_message(
                "File CSV không có dữ liệu",
                intent="error"
            )
        
        # Nếu không có câu hỏi, trả về tổng quan
        if not user_question or user_question.strip() == "":
            summary_text = summarize_dataset(df)
            return RichMessage.create_mixed_message(
                text=summary_text + "\n\n**Dữ liệu mẫu (10 dòng đầu):**",
                table=df.head(10),
                intent="data_overview"
            )
        
        # Tạo pandas agent
        agent = create_csv_agent(df)
        
        # Phân tích câu hỏi để xem có cần visualization không
        viz_request = RichMessage.detect_visualization_request(user_question)
        
        # Nếu người dùng yêu cầu vẽ biểu đồ
        if viz_request['needs_visualization']:
            column_to_plot = None
            
            # Tìm tên cột trực tiếp từ câu hỏi
            # Ưu tiên tìm các từ sau "của", "column", "cột"
            import re
            
            # Pattern 1: "của cột X" hoặc "của X"
            matches = re.findall(r'(?:của\s+cột\s+|của\s+|column\s+|cột\s+)([A-Z_][A-Z0-9_]*)', user_question, re.IGNORECASE)
            if matches:
                potential_col = matches[0].upper()
                # Kiểm tra xem có trong df không
                for col in df.columns:
                    if col.upper() == potential_col:
                        column_to_plot = col
                        break
            
            # Pattern 2: Tìm tất cả các từ viết hoa liên tiếp (tên cột)
            if not column_to_plot:
                for col in df.columns:
                    if col.upper() in user_question.upper():
                        column_to_plot = col
                        break
            
            # Nếu vẫn không tìm thấy, chọn cột phù hợp dựa vào chart type
            if not column_to_plot:
                chart_type = viz_request['chart_type']
                
                # Histogram cần cột số
                if chart_type == 'histogram':
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if numeric_cols:
                        column_to_plot = numeric_cols[0]
                    else:
                        return RichMessage.create_text_message(
                            "Dataset không có cột số nào để vẽ histogram. Thử 'bar chart' hoặc 'pie chart' cho dữ liệu categorical.",
                            intent="error"
                        )
                # Bar/Pie có thể dùng cột categorical hoặc số
                elif chart_type in ['bar', 'pie']:
                    # Ưu tiên cột categorical (object/string)
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                    if categorical_cols:
                        column_to_plot = categorical_cols[0]
                    elif numeric_cols:
                        column_to_plot = numeric_cols[0]
                    else:
                        return RichMessage.create_text_message(
                            "Dataset không có cột phù hợp để vẽ biểu đồ.",
                            intent="error"
                        )
                # Các loại khác (line, scatter) cần cột số
                else:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if numeric_cols:
                        column_to_plot = numeric_cols[0]
                    else:
                        return RichMessage.create_text_message(
                            "Dataset không có cột số nào để vẽ biểu đồ này.",
                            intent="error"
                        )
            
            # Kiểm tra cột tồn tại
            if column_to_plot not in df.columns:
                return RichMessage.create_text_message(
                    f"Không tìm thấy cột '{column_to_plot}' trong dataset",
                    intent="error"
                )
            
            # Kiểm tra loại dữ liệu và chart type
            is_numeric = pd.api.types.is_numeric_dtype(df[column_to_plot])
            chart_type = viz_request['chart_type']
            
            # Histogram chỉ cho cột số
            if chart_type == 'histogram' and not is_numeric:
                return RichMessage.create_text_message(
                    f"Histogram yêu cầu cột số. Cột '{column_to_plot}' là cột categorical. Thử dùng 'bar chart' hoặc 'pie chart' thay thế.",
                    intent="error"
                )
            
            # Bar chart và pie chart có thể dùng cho cột categorical
            # Tự động chuyển sang bar nếu cột là categorical
            if not is_numeric and chart_type in ['histogram', 'line', 'scatter']:
                chart_type = 'bar'  # Auto fallback cho categorical
            
            # Vẽ biểu đồ
            chart_message = RichMessage.create_chart_message(
                df=df,
                chart_type=chart_type,
                column=column_to_plot,
                description=f"{chart_type.title()} của cột '{column_to_plot}'",
                bins=20 if chart_type == 'histogram' else None
            )
            
            return chart_message
        
        # Nếu không cần visualization, dùng agent để trả lời
        try:
            response = agent.invoke(user_question)
            answer = response['output']
            
            # Kiểm tra xem câu trả lời có chứa bảng không (dựa vào markdown table syntax)
            if '|' in answer and '\n' in answer:
                # Có thể là bảng markdown
                return RichMessage.create_text_message(
                    content=answer,
                    intent="data_analysis"
                )
            else:
                # Text thông thường
                return RichMessage.create_text_message(
                    content=answer,
                    intent="data_analysis"
                )
        
        except Exception as e:
            return RichMessage.create_text_message(
                f"Lỗi khi phân tích: {str(e)}",
                intent="error"
            )
        
    except (FileNotFoundError, ValueError) as e:
        return RichMessage.create_text_message(
            f"Lỗi: {str(e)}",
            intent="error"
        )
    except pd.errors.EmptyDataError:
        return RichMessage.create_text_message(
            "File CSV trống",
            intent="error"
        )
    except pd.errors.ParserError:
        return RichMessage.create_text_message(
            "Không thể đọc file CSV. Vui lòng kiểm tra định dạng.",
            intent="error"
        )
    except Exception as e:
        return RichMessage.create_text_message(
            f"Lỗi xử lý CSV: {str(e)}",
            intent="error"
        )


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
        return f"Lỗi phân tích với AI: {str(e)}"
