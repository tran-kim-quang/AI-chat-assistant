import openai
import os
import base64
import pandas as pd
import requests
import io
from typing import Union, Dict, Any
from dotenv import load_dotenv
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


# Pandas Agent Helper - ĐÂY LÀ CORE, AGENT TỰ XỬ LÝ MỌI THỨ
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
    """
    Xử lý CSV với pandas agent - ĐỂ AGENT TỰ PHÂN TÍCH
    Chỉ can thiệp khi cần visualization (để trả về chart base64)
    """
    try:
        # Load CSV từ source
        df = load_csv_from_source(source, max_rows)
        
        # Kiểm tra empty
        if len(df) == 0:
            return RichMessage.create_text_message(
                "File CSV không có dữ liệu",
                intent="error"
            )
        
        # Nếu không có câu hỏi, tạo quick summary
        if not user_question or user_question.strip() == "":
            summary = f"""Dataset có **{len(df)} dòng** và **{len(df.columns)} cột**.

**Các cột:** {', '.join(df.columns.tolist())}

Bạn có thể hỏi tôi về dữ liệu này!"""
            
            return RichMessage.create_mixed_message(
                text=summary,
                table=df.head(10),
                intent="data_overview"
            )
        
        # Tạo pandas agent
        agent = create_csv_agent(df)
        
        # Phân tích câu hỏi để xem có cần visualization không
        viz_request = RichMessage.detect_visualization_request(user_question)
        
        # Nếu người dùng yêu cầu vẽ biểu đồ
        if viz_request['needs_visualization']:
            # HỎI AGENT tìm cột phù hợp thay vì dùng regex cứng nhắc
            try:
                column_question = f"""Dựa vào câu hỏi: "{user_question}"
                
Hãy trả về CHÍNH XÁC TÊN MỘT CỘT (không giải thích) phù hợp nhất để vẽ biểu đồ.
Nếu không rõ cột nào, chọn cột có ý nghĩa nhất cho loại biểu đồ "{viz_request['chart_type']}".

Chỉ trả về TÊN CỘT, không có text khác."""
                
                agent_response = agent.invoke(column_question)
                suggested_col = agent_response['output'].strip()
                
                # Tìm cột trong danh sách columns
                column_to_plot = None
                for col in df.columns:
                    if col in suggested_col or col.lower() in suggested_col.lower():
                        column_to_plot = col
                        break
                
                # Nếu không tìm thấy, lấy cột đầu tiên phù hợp với chart type
                if not column_to_plot:
                    chart_type = viz_request['chart_type']
                    if chart_type in ['bar', 'pie']:
                        # Ưu tiên categorical
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        column_to_plot = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
                    else:
                        # Cần numeric
                        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        column_to_plot = num_cols[0] if num_cols else None
                
                if not column_to_plot:
                    return RichMessage.create_text_message(
                        f"Không tìm thấy cột phù hợp để vẽ {viz_request['chart_type']}. Các cột: {', '.join(df.columns.tolist())}",
                        intent="error"
                    )
            
            except Exception as e:
                # Fallback: tìm cột đơn giản
                for col in df.columns:
                    if col.upper() in user_question.upper():
                        column_to_plot = col
                        break
                
                if not column_to_plot:
                    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    column_to_plot = num_cols[0] if num_cols else df.columns[0]
            
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
        
        # Nếu không cần visualization, ĐỂ AGENT TỰ TRẢ LỜI
        try:
            response = agent.invoke(user_question)
            answer = response['output']
            
            # Trả về text answer (agent tự format markdown nếu cần)
            return RichMessage.create_text_message(
                content=answer,
                intent="data_analysis"
            )
        
        except Exception as e:
            return RichMessage.create_text_message(
                f"❌ Lỗi khi phân tích: {str(e)}",
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
