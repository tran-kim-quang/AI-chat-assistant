from flask import Flask, render_template, request, jsonify, session
import os
import io
from werkzeug.utils import secure_filename
from config.process import csv_process, load_csv_from_source
from config.process import summarize_dataset, basic_stats, find_missing_values, plot_histogram
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload CSV file"""
    try:
        # Kiểm tra có file không
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Chỉ chấp nhận file .csv'}), 400
        
        # Lưu file tạm
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Lưu path vào session
        session['csv_file'] = filepath
        
        # Load và hiển thị preview
        df = load_csv_from_source(filepath, max_rows=100)
        preview = df.head(10).to_html(classes='table table-striped table-bordered')
        
        summary = summarize_dataset(df)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'preview': preview,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi upload file: {str(e)}'}), 500


@app.route('/load_url', methods=['POST'])
def load_url():
    """Load CSV từ URL"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL không được để trống'}), 400
        
        # Kiểm tra URL hợp lệ
        if not url.startswith('http://') and not url.startswith('https://'):
            return jsonify({'error': 'URL không hợp lệ'}), 400
        
        # Lưu URL vào session
        session['csv_url'] = url
        
        # Load và hiển thị preview
        df = load_csv_from_source(url, max_rows=100)
        preview = df.head(10).to_html(classes='table table-striped table-bordered')
        
        summary = summarize_dataset(df)
        
        return jsonify({
            'success': True,
            'url': url,
            'preview': preview,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi load URL: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Phân tích CSV với câu hỏi"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        # Lấy source từ session
        csv_file = session.get('csv_file')
        csv_url = session.get('csv_url')
        
        if not csv_file and not csv_url:
            return jsonify({'error': 'Chưa upload file hoặc nhập URL'}), 400
        
        source = csv_url if csv_url else csv_file
        
        # Xử lý câu hỏi
        result = csv_process(source, question)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi phân tích: {str(e)}'}), 500


@app.route('/quick_analysis/<analysis_type>', methods=['POST'])
def quick_analysis(analysis_type):
    """Phân tích nhanh: summarize, stats, missing, histogram"""
    try:
        # Lấy source từ session
        csv_file = session.get('csv_file')
        csv_url = session.get('csv_url')
        
        if not csv_file and not csv_url:
            return jsonify({'error': 'Chưa upload file hoặc nhập URL'}), 400
        
        source = csv_url if csv_url else csv_file
        df = load_csv_from_source(source)
        
        # Xử lý theo loại
        if analysis_type == 'summarize':
            result = summarize_dataset(df)
            result += "\n\n📋 DỮ LIỆU MẪU:\n" + df.head().to_string()
        
        elif analysis_type == 'stats':
            result = basic_stats(df)
        
        elif analysis_type == 'missing':
            result = find_missing_values(df)
        
        elif analysis_type == 'histogram':
            # Lấy column từ request
            data = request.get_json()
            column = data.get('column')
            
            if not column:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                return jsonify({
                    'error': f'Vui lòng chọn cột. Các cột số: {", ".join(numeric_cols)}'
                }), 400
            
            result = plot_histogram(df, column)
        
        else:
            return jsonify({'error': 'Loại phân tích không hợp lệ'}), 400
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi phân tích: {str(e)}'}), 500


@app.route('/get_columns', methods=['GET'])
def get_columns():
    """Lấy danh sách columns"""
    try:
        csv_file = session.get('csv_file')
        csv_url = session.get('csv_url')
        
        if not csv_file and not csv_url:
            return jsonify({'error': 'Chưa upload file'}), 400
        
        source = csv_url if csv_url else csv_file
        df = load_csv_from_source(source, max_rows=10)
        
        columns = {
            'all': df.columns.tolist(),
            'numeric': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'text': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return jsonify(columns)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_session():
    """Xóa session và file tạm"""
    try:
        # Xóa file tạm nếu có
        csv_file = session.get('csv_file')
        if csv_file and os.path.exists(csv_file):
            os.remove(csv_file)
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

