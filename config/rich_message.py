import base64
from typing import Dict, List, Union, Literal
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO


MessageType = Literal['text', 'table', 'chart', 'image', 'mixed']


class RichMessage:
    """
    Class để tạo rich message với nhiều loại content
    
    Format:
    {
        "type": "text" | "table" | "chart" | "image" | "mixed",
        "content": "...",  # Text content chính
        "data": {
            "table_html": "...",  # HTML table nếu type là table
            "chart_base64": "...",  # Base64 image nếu type là chart
            "chart_type": "histogram" | "bar" | "line",
            "image_url": "...",  # URL hoặc path của image
            ...
        },
        "metadata": {
            "intent": "...",
            "confidence": ...
        }
    }
    """
    
    @staticmethod
    def create_text_message(content: str, intent: str = None, confidence: float = None) -> Dict:
        """Tạo text message đơn giản"""
        return {
            "type": "text",
            "content": content,
            "data": {},
            "metadata": {
                "intent": intent,
                "confidence": confidence
            }
        }
    
    @staticmethod
    def create_table_message(
        df: pd.DataFrame, 
        description: str = None,
        max_rows: int = 50,
        intent: str = None
    ) -> Dict:
        """Tạo message với bảng dữ liệu"""
        
        # Tạo HTML table
        table_html = df.head(max_rows).to_html(
            classes='table table-sm table-striped table-hover',
            index=False,
            border=0
        )
        
        # Content text
        if description is None:
            description = f"Bảng dữ liệu ({len(df)} dòng × {len(df.columns)} cột)"
        
        return {
            "type": "table",
            "content": description,
            "data": {
                "table_html": table_html,
                "rows": len(df),
                "columns": list(df.columns),
                "showing_rows": min(max_rows, len(df))
            },
            "metadata": {
                "intent": intent,
                "confidence": None
            }
        }
    
    @staticmethod
    def create_chart_message(
        df: pd.DataFrame,
        chart_type: str,
        column: str = None,
        description: str = None,
        **plot_kwargs
    ) -> Dict:
        """
        Tạo message với biểu đồ
        
        Args:
            df: DataFrame
            chart_type: 'histogram', 'bar', 'line', 'scatter', 'pie'
            column: Tên cột để vẽ
            description: Mô tả biểu đồ
            **plot_kwargs: Các tham số cho matplotlib
        """
        
        try:
            # Tạo figure
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'histogram':
                if column is None or column not in df.columns:
                    return RichMessage.create_text_message(
                        f"Lỗi: Cột '{column}' không tồn tại",
                        intent="error"
                    )
                
                bins = plot_kwargs.get('bins', 20)
                plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title(f'Histogram của {column}', fontsize=14, fontweight='bold')
                plt.xlabel(column, fontsize=12)
                plt.ylabel('Tần suất', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                
            elif chart_type == 'bar':
                if column is None or column not in df.columns:
                    return RichMessage.create_text_message(
                        f"Lỗi: Cột '{column}' không tồn tại",
                        intent="error"
                    )
                
                value_counts = df[column].value_counts().head(20)
                plt.bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                plt.title(f'Phân bố của {column}', fontsize=14, fontweight='bold')
                plt.xlabel(column, fontsize=12)
                plt.ylabel('Số lượng', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                
            elif chart_type == 'line':
                if column is None or column not in df.columns:
                    return RichMessage.create_text_message(
                        f"Lỗi: Cột '{column}' không tồn tại",
                        intent="error"
                    )
                
                plt.plot(df.index, df[column], marker='o', linestyle='-', linewidth=2, markersize=4)
                plt.title(f'Biểu đồ đường của {column}', fontsize=14, fontweight='bold')
                plt.xlabel('Index', fontsize=12)
                plt.ylabel(column, fontsize=12)
                plt.grid(True, alpha=0.3)
                
            elif chart_type == 'scatter':
                x_col = plot_kwargs.get('x_column')
                y_col = plot_kwargs.get('y_column', column)
                
                if x_col not in df.columns or y_col not in df.columns:
                    return RichMessage.create_text_message(
                        "Lỗi: Cột không tồn tại để vẽ scatter plot",
                        intent="error"
                    )
                
                plt.scatter(df[x_col], df[y_col], alpha=0.6, c='purple', edgecolors='black')
                plt.title(f'Scatter: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                plt.xlabel(x_col, fontsize=12)
                plt.ylabel(y_col, fontsize=12)
                plt.grid(True, alpha=0.3)
            
            elif chart_type == 'pie':
                if column is None or column not in df.columns:
                    return RichMessage.create_text_message(
                        f"Lỗi: Cột '{column}' không tồn tại",
                        intent="error"
                    )
                
                value_counts = df[column].value_counts().head(10)
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Phân bố của {column}', fontsize=14, fontweight='bold')
                plt.axis('equal')
            
            else:
                plt.close()
                return RichMessage.create_text_message(
                    f"Loại biểu đồ '{chart_type}' chưa được hỗ trợ",
                    intent="error"
                )
            
            plt.tight_layout()
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Content description
            if description is None:
                description = f"Biểu đồ {chart_type} của cột '{column}'"
            
            return {
                "type": "chart",
                "content": description,
                "data": {
                    "chart_base64": chart_base64,
                    "chart_type": chart_type,
                    "column": column
                },
                "metadata": {
                    "intent": "data_visualization",
                    "confidence": None
                }
            }
            
        except Exception as e:
            plt.close()
            return RichMessage.create_text_message(
                f"Lỗi khi tạo biểu đồ: {str(e)}",
                intent="error"
            )
    
    @staticmethod
    def create_mixed_message(
        text: str = None,
        table: pd.DataFrame = None,
        chart_config: Dict = None,
        intent: str = None
    ) -> Dict:
        """
        Tạo message kết hợp nhiều loại content
        
        Args:
            text: Text content
            table: DataFrame để hiển thị
            chart_config: {
                'df': DataFrame,
                'chart_type': 'histogram',
                'column': 'column_name',
                ...
            }
        """
        
        content_parts = []
        data = {}
        
        if text:
            content_parts.append(text)
        
        if table is not None:
            table_html = table.head(50).to_html(
                classes='table table-sm table-striped table-hover',
                index=False,
                border=0
            )
            data['table_html'] = table_html
            data['table_rows'] = len(table)
        
        if chart_config:
            chart_msg = RichMessage.create_chart_message(**chart_config)
            if chart_msg['type'] == 'chart':
                data['chart_base64'] = chart_msg['data']['chart_base64']
                data['chart_type'] = chart_msg['data']['chart_type']
        
        return {
            "type": "mixed",
            "content": "\n\n".join(content_parts) if content_parts else "",
            "data": data,
            "metadata": {
                "intent": intent,
                "confidence": None
            }
        }
    
    @staticmethod
    def detect_visualization_request(message: str) -> Dict:
        """
        Phát hiện xem user có yêu cầu vẽ biểu đồ không
        
        Returns:
            {
                'needs_visualization': bool,
                'chart_type': str,
                'column': str (nếu xác định được)
            }
        """
        
        message_lower = message.lower()
        
        # Keywords cho các loại biểu đồ
        chart_keywords = {
            'histogram': ['histogram', 'phân phối', 'distribution', 'hist'],
            'bar': ['bar chart', 'biểu đồ cột', 'bar', 'count', 'đếm'],
            'line': ['line chart', 'biểu đồ đường', 'line', 'trend', 'xu hướng'],
            'scatter': ['scatter', 'tương quan', 'correlation', 'scatter plot'],
            'pie': ['pie chart', 'biểu đồ tròn', 'pie', 'phần trăm']
        }
        
        # Từ khóa chung cho visualization
        viz_keywords = ['plot', 'vẽ', 'biểu đồ', 'chart', 'graph', 'visualize', 'show', 'hiển thị']
        
        needs_viz = any(kw in message_lower for kw in viz_keywords)
        
        if not needs_viz:
            return {
                'needs_visualization': False,
                'chart_type': None,
                'column': None
            }
        
        # Xác định loại chart
        chart_type = None
        for ctype, keywords in chart_keywords.items():
            if any(kw in message_lower for kw in keywords):
                chart_type = ctype
                break
        
        # Mặc định là histogram nếu không xác định được
        if chart_type is None:
            chart_type = 'histogram'
        
        return {
            'needs_visualization': True,
            'chart_type': chart_type,
            'column': None  # Sẽ được xác định bởi agent hoặc user
        }


# Utility function để format response từ backend
def format_response_for_frontend(response_data: Dict) -> Dict:
    """
    Format rich message để gửi về frontend
    Đảm bảo tương thích với format hiện tại
    """
    
    if response_data.get('type') == 'text' or response_data.get('type') is None:
        # Text message - giữ nguyên format cũ
        return {
            'success': True,
            'response': response_data.get('content', ''),
            'intent': response_data.get('metadata', {}).get('intent'),
            'confidence': response_data.get('metadata', {}).get('confidence'),
            'message_type': 'text'
        }
    
    else:
        # Rich message - thêm thông tin chi tiết
        return {
            'success': True,
            'response': response_data.get('content', ''),
            'intent': response_data.get('metadata', {}).get('intent'),
            'confidence': response_data.get('metadata', {}).get('confidence'),
            'message_type': response_data.get('type'),
            'data': response_data.get('data', {})
        }

