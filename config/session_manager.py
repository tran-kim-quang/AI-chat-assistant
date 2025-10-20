"""
Session Manager - Quản lý session và lịch sử hội thoại
"""
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

class SessionManager:
    """Quản lý session và conversation history"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Khởi tạo database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                last_activity TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Bảng messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                intent TEXT,
                context TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Bảng context (files, images uploaded)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_storage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                context_type TEXT,
                context_data TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, metadata: Dict = None) -> str:
        """Tạo session mới"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, created_at, last_activity, metadata)
            VALUES (?, ?, ?, ?)
        """, (session_id, now, now, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Lấy thông tin session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, created_at, last_activity, metadata
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'session_id': row[0],
                'created_at': row[1],
                'last_activity': row[2],
                'metadata': json.loads(row[3])
            }
        return None
    
    def add_message(self, session_id: str, role: str, content: str, 
                   intent: str = None, context: Dict = None):
        """Thêm message vào history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cập nhật last_activity
        cursor.execute("""
            UPDATE sessions SET last_activity = ?
            WHERE session_id = ?
        """, (datetime.now(), session_id))
        
        # Thêm message
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, intent, context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, intent, json.dumps(context or {}), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Lấy lịch sử hội thoại"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, intent, context, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in reversed(rows):  # Đảo ngược để có thứ tự cũ -> mới
            messages.append({
                'role': row[0],
                'content': row[1],
                'intent': row[2],
                'context': json.loads(row[3]) if row[3] else {},
                'timestamp': row[4]
            })
        
        return messages
    
    def add_context(self, session_id: str, context_type: str, context_data: Dict):
        """Thêm context (file uploaded, image, etc.)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO context_storage (session_id, context_type, context_data, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, context_type, json.dumps(context_data), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_context(self, session_id: str, context_type: str = None) -> List[Dict]:
        """Lấy context của session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if context_type:
            cursor.execute("""
                SELECT context_type, context_data, created_at
                FROM context_storage
                WHERE session_id = ? AND context_type = ?
                ORDER BY created_at DESC
            """, (session_id, context_type))
        else:
            cursor.execute("""
                SELECT context_type, context_data, created_at
                FROM context_storage
                WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        contexts = []
        for row in rows:
            contexts.append({
                'type': row[0],
                'data': json.loads(row[1]),
                'created_at': row[2]
            })
        
        return contexts
    
    def clear_session(self, session_id: str):
        """Xóa session và tất cả dữ liệu liên quan"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM context_storage WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_recent_context_summary(self, session_id: str) -> str:
        """Tạo tóm tắt context gần đây để AI hiểu ngữ cảnh"""
        contexts = self.get_context(session_id)
        history = self.get_conversation_history(session_id, limit=10)
        
        summary = ""
        
        # Context về files/images
        if contexts:
            summary += "Context hiện tại:\n"
            for ctx in contexts[:3]:  # 3 context gần nhất
                if ctx['type'] == 'csv':
                    summary += f"- File CSV: {ctx['data'].get('filename', 'unknown')}\n"
                elif ctx['type'] == 'image':
                    summary += f"- Ảnh: {ctx['data'].get('filename', 'unknown')}\n"
            summary += "\n"
        
        # Lịch sử hội thoại
        if history:
            summary += "Lịch sử hội thoại gần đây:\n"
            for msg in history[-5:]:  # 5 messages gần nhất
                summary += f"{msg['role']}: {msg['content'][:100]}...\n"
        
        return summary

