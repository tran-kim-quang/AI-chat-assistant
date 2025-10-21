from flask import Flask, render_template, request, jsonify, session as flask_session, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid

from config.conversation_handler import get_conversation_handler
from config.session_manager import SessionManager
from config.process import load_csv_from_source
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize handlers
conversation_handler = get_conversation_handler()
session_manager = SessionManager()

ALLOWED_EXTENSIONS = {'csv', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('chat.html')


@app.route('/api/init_session', methods=['POST'])
def init_session():
    """Khởi tạo session mới"""
    try:
        # Tạo session ID
        session_id = session_manager.create_session({
            'user_agent': request.headers.get('User-Agent'),
            'ip': request.remote_addr
        })
        
        # Lưu vào Flask session
        flask_session['chat_session_id'] = session_id
        
        # Gửi welcome message
        welcome_msg = """Xin chào! Tôi là AI Assistant thông minh.

Tôi có thể giúp bạn:
- Phân tích dữ liệu CSV
- Phân tích và mô tả ảnh
- Trò chuyện thông thường

Bạn muốn làm gì hôm nay?"""
        
        session_manager.add_message(
            session_id=session_id,
            role='assistant',
            content=welcome_msg,
            intent='greeting'
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'welcome_message': welcome_msg
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint chính xử lý chat
    AI sẽ tự động phân loại intent và route đến handler phù hợp
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Tin nhắn trống'}), 400
        
        # Lấy hoặc tạo session ID
        session_id = flask_session.get('chat_session_id')
        if not session_id:
            session_id = session_manager.create_session()
            flask_session['chat_session_id'] = session_id
        
        # Xử lý message với ConversationHandler
        result = conversation_handler.process_message(session_id, user_message)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'intent': result['intent'],
            'confidence': result['confidence'],
            'suggested_actions': result.get('suggested_actions', []),
            'message_type': result.get('message_type', 'text'),
            'data': result.get('data', {})
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500


@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    """Upload file (CSV hoặc ảnh)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File không hợp lệ. Chỉ chấp nhận .csv, .jpg, .jpeg, .png'}), 400
        
        # Lấy session ID
        session_id = flask_session.get('chat_session_id')
        if not session_id:
            session_id = session_manager.create_session()
            flask_session['chat_session_id'] = session_id
        
        # Lưu file
        filename = secure_filename(file.filename)
        unique_filename = f"{session_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Xác định loại file
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            # CSV file
            df = load_csv_from_source(filepath, max_rows=100)
            
            # Thêm vào context
            conversation_handler.add_file_context(
                session_id=session_id,
                file_type='csv',
                file_data={
                    'filename': filename,
                    'filepath': filepath,
                    'rows': len(df),
                    'columns': df.columns.tolist()
                }
            )
            
            preview = df.head(10).to_html(classes='table table-sm table-striped')
            
            # Tự động gửi message thông báo
            auto_message = f"File CSV '{filename}' đã được upload thành công!\n\nDữ liệu có {len(df)} dòng và {len(df.columns)} cột.\nCác cột: {', '.join(df.columns.tolist())}\n\nBạn có thể hỏi tôi về dữ liệu này!"
            
            session_manager.add_message(
                session_id=session_id,
                role='assistant',
                content=auto_message,
                intent='file_uploaded'
            )
            
            return jsonify({
                'success': True,
                'file_type': 'csv',
                'filename': filename,
                'preview': preview,
                'auto_message': auto_message,
                'stats': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                }
            })
        
        else:
            # Image file
            conversation_handler.add_file_context(
                session_id=session_id,
                file_type='image',
                file_data={
                    'filename': filename,
                    'filepath': filepath
                }
            )
            
            auto_message = f"Ảnh '{filename}' đã được upload!\n\nBạn có thể hỏi tôi về ảnh này."
            
            session_manager.add_message(
                session_id=session_id,
                role='assistant',
                content=auto_message,
                intent='file_uploaded'
            )
            
            return jsonify({
                'success': True,
                'file_type': 'image',
                'filename': filename,
                'filepath': filepath,
                'auto_message': auto_message
            })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi upload: {str(e)}'}), 500


@app.route('/api/upload_url', methods=['POST'])
def upload_url():
    """Upload CSV từ URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL trống'}), 400
        
        # Lấy session ID
        session_id = flask_session.get('chat_session_id')
        if not session_id:
            session_id = session_manager.create_session()
            flask_session['chat_session_id'] = session_id
        
        # Load CSV từ URL
        df = load_csv_from_source(url, max_rows=100)
        
        # Thêm vào context
        conversation_handler.add_file_context(
            session_id=session_id,
            file_type='csv',
            file_data={
                'url': url,
                'rows': len(df),
                'columns': df.columns.tolist()
            }
        )
        
        preview = df.head(10).to_html(classes='table table-sm table-striped')
        
        auto_message = f"File CSV từ URL đã được tải thành công!\n\nDữ liệu có {len(df)} dòng và {len(df.columns)} cột.\nBạn có thể hỏi tôi về dữ liệu!"
        
        session_manager.add_message(
            session_id=session_id,
            role='assistant',
            content=auto_message,
            intent='file_uploaded'
        )
        
        return jsonify({
            'success': True,
            'preview': preview,
            'auto_message': auto_message,
            'stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi tải URL: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Lấy lịch sử hội thoại"""
    try:
        session_id = flask_session.get('chat_session_id')
        if not session_id:
            return jsonify({'history': []})
        
        history = session_manager.get_conversation_history(session_id, limit=50)
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Xóa session hiện tại"""
    try:
        session_id = flask_session.get('chat_session_id')
        if session_id:
            # Xóa files uploaded
            contexts = session_manager.get_context(session_id)
            for ctx in contexts:
                if 'filepath' in ctx['data']:
                    filepath = ctx['data']['filepath']
                    if os.path.exists(filepath):
                        os.remove(filepath)
            
            # Xóa session
            session_manager.clear_session(session_id)
        
        # Clear Flask session
        flask_session.clear()
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session_info', methods=['GET'])
def session_info():
    """Lấy thông tin session"""
    try:
        session_id = flask_session.get('chat_session_id')
        if not session_id:
            return jsonify({'error': 'Chưa có session'}), 404
        
        summary = conversation_handler.get_session_summary(session_id)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files (images)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  AI MULTI-TURN CONVERSATION SYSTEM                           ║
    ║  Powered by Intent Classification (NO KEYWORDS)              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Features:
    - AI-based intent classification
    - Multi-turn conversation with context
    - Session management
    - File upload (CSV & Images)
    - Conversation history retrieval
    
    Access: http://localhost:5000
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

