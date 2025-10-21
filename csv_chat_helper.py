import os
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')  # Backend cho server, không cần GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO

load_dotenv()


class CSVChatAgent:
    """
    Class để xử lý CSV data chat
    
    Usage:
        csv_agent = CSVChatAgent()
        
        # Load từ file
        csv_agent.load_from_file("data.csv")
        
        # Hoặc load từ URL
        csv_agent.load_from_url("https://raw.githubusercontent.com/.../data.csv")
        
        # Chat
        response = csv_agent.chat("Tóm tắt dataset")
        
        # Lấy bảng
        table_html = csv_agent.get_table_html()
    """
    
    def __init__(self):
        self.df = None
        self.agent = None
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.BASE_MODEL = os.getenv("TEXT_MODEL")
        
    def load_from_file(self, file_path):
        """Load CSV từ file path"""
        try:
            self.df = pd.read_csv(file_path)
            self._init_agent()
            return {
                "success": True,
                "message": f"Đã load thành công {len(self.df)} dòng và {len(self.df.columns)} cột",
                "rows": len(self.df),
                "columns": list(self.df.columns)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Lỗi khi load CSV: {str(e)}"
            }
    
    def load_from_url(self, url):
        """Load CSV từ URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            self.df = pd.read_csv(StringIO(response.text))
            self._init_agent()
            
            return {
                "success": True,
                "message": f"Đã load thành công từ URL: {len(self.df)} dòng và {len(self.df.columns)} cột",
                "rows": len(self.df),
                "columns": list(self.df.columns)
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"Lỗi khi tải CSV từ URL: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Lỗi khi parse CSV: {str(e)}"
            }
    
    def load_from_upload(self, file_content):
        """Load CSV từ file upload (file content as string or bytes)"""
        try:
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            self.df = pd.read_csv(StringIO(file_content))
            self._init_agent()
            
            return {
                "success": True,
                "message": f"Đã load thành công {len(self.df)} dòng và {len(self.df.columns)} cột",
                "rows": len(self.df),
                "columns": list(self.df.columns)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Lỗi khi parse CSV: {str(e)}"
            }
    
    def _init_agent(self):
        """Khởi tạo LangChain agent"""
        llm = ChatGroq(
            model=self.BASE_MODEL,
            api_key=self.GROQ_API_KEY,
            temperature=0.0
        )
        
        self.agent = create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=False,  # Set False cho production
            agent_type="tool-calling",
            allow_dangerous_code=True,
            max_iterations=50,
            agent_executor_kwargs={"handle_parsing_errors": True},
            prefix="""Bạn là trợ lý phân tích dữ liệu chuyên nghiệp.
Hãy trả lời ngắn gọn, chính xác với KẾT QUẢ thực tế từ dữ liệu.
- Nếu hỏi về bảng: dùng df.to_markdown() hoặc tabulate để format đẹp
- Nếu hỏi về thống kê: dùng describe(), info(), value_counts()
- KHÔNG giải thích code, CHỈ đưa ra kết quả.
- Format kết quả dạng markdown table nếu có thể.
"""
        )
    
    def chat(self, question):
        """
        Chat với CSV data
        
        Returns:
            dict: {
                "success": bool,
                "question": str,
                "answer": str,
                "error": str (nếu có lỗi)
            }
        """
        if self.df is None or self.agent is None:
            return {
                "success": False,
                "question": question,
                "answer": "",
                "error": "Chưa load CSV. Hãy load file trước khi chat."
            }
        
        try:
            response = self.agent.invoke(question)
            return {
                "success": True,
                "question": question,
                "answer": response['output'],
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "answer": "",
                "error": str(e)
            }
    
    def get_summary(self):
        """Lấy tóm tắt dataset"""
        if self.df is None:
            return None
        
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict() if not self.df.select_dtypes(include='number').empty else {}
        }
    
    def get_table_html(self, max_rows=100):
        """Lấy bảng dưới dạng HTML"""
        if self.df is None:
            return None
        
        return self.df.head(max_rows).to_html(classes='table table-striped', index=False)
    
    def get_table_markdown(self, max_rows=100):
        """Lấy bảng dưới dạng Markdown"""
        if self.df is None:
            return None
        
        return tabulate(self.df.head(max_rows), headers='keys', tablefmt='pipe', showindex=False)
    
    def get_basic_stats(self):
        """Lấy thống kê cơ bản dạng text"""
        if self.df is None:
            return None
        
        stats = []
        stats.append(f"📊 **Thống kê cơ bản**\n")
        stats.append(f"- Số dòng: {len(self.df)}")
        stats.append(f"- Số cột: {len(self.df.columns)}")
        stats.append(f"- Các cột: {', '.join(self.df.columns)}\n")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            stats.append("**Missing values:**")
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.df) * 100)
                stats.append(f"- {col}: {count} ({pct:.1f}%)")
        else:
            stats.append("✅ Không có missing values")
        
        # Numeric columns stats
        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            stats.append("\n**Thống kê các cột số:**")
            stats.append(tabulate(self.df[numeric_cols].describe(), headers='keys', tablefmt='pipe'))
        
        return "\n".join(stats)
    
    def plot_histogram(self, column_name, bins=10):
        """
        Vẽ histogram cho một cột
        
        Returns:
            str: Base64 encoded image hoặc None nếu lỗi
        """
        if self.df is None:
            return None
        
        if column_name not in self.df.columns:
            return None
        
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(self.df[column_name].dropna(), bins=bins, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.3)
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error plotting histogram: {e}")
            return None


# Ví dụ sử dụng cho Flask/FastAPI endpoint
def example_flask_route():
    """
    Ví dụ về cách tích hợp vào Flask route
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    csv_agent = CSVChatAgent()
    
    @app.route('/api/csv/upload', methods=['POST'])
    def upload_csv():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        result = csv_agent.load_from_upload(content)
        return jsonify(result)
    
    @app.route('/api/csv/load-url', methods=['POST'])
    def load_url():
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        
        result = csv_agent.load_from_url(url)
        return jsonify(result)
    
    @app.route('/api/csv/chat', methods=['POST'])
    def chat():
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        result = csv_agent.chat(question)
        return jsonify(result)
    
    @app.route('/api/csv/summary', methods=['GET'])
    def get_summary():
        summary = csv_agent.get_summary()
        if summary is None:
            return jsonify({"error": "No CSV loaded"}), 400
        return jsonify(summary)
    
    return app


if __name__ == "__main__":
    # Test
    print("Testing CSVChatAgent...")
    
    agent = CSVChatAgent()
    
    # Test load from file
    result = agent.load_from_file("test_data.csv")
    print(f"\n1. Load result: {result}")
    
    # Test get summary
    print(f"\n2. Summary: {agent.get_summary()}")
    
    # Test chat
    response = agent.chat("Có bao nhiêu người ở mỗi thành phố?")
    print(f"\n3. Chat response: {response}")
    
    # Test get table
    print(f"\n4. Table (markdown):\n{agent.get_table_markdown()}")
    
    print("\n✅ Test completed!")

