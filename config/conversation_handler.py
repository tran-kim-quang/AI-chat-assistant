import openai
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv

from config.session_manager import SessionManager
from config.intent_classifier import get_intent_classifier
from config.process import text_process, img_process, csv_process
from config.code_executor import get_code_executor

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL_TEXT = os.getenv('TEXT_MODEL')


class ConversationHandler:
    """Xử lý hội thoại multi-turn với context awareness"""
    
    def __init__(self, session_manager: SessionManager = None):
        self.session_manager = session_manager or SessionManager()
        self.intent_classifier = get_intent_classifier()
        self.code_executor = get_code_executor()
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
    
    def process_message(self, session_id: str, user_message: str) -> Dict:

        
        # Lấy context summary
        context_summary = self.session_manager.get_recent_context_summary(session_id)
        
        # Phân loại intent
        intent_result = self.intent_classifier.classify(user_message, context_summary)
        
        # Lưu user message vào history
        self.session_manager.add_message(
            session_id=session_id,
            role='user',
            content=user_message,
            intent=intent_result['intent'],
            context={'confidence': intent_result['confidence']}
        )
        
        # Xử lý theo intent
        response = self._route_by_intent(
            session_id=session_id,
            user_message=user_message,
            intent_result=intent_result
        )
        
        # Lưu assistant response vào history
        self.session_manager.add_message(
            session_id=session_id,
            role='assistant',
            content=response['response'],
            intent=intent_result['intent']
        )
        
        return response
    
    def _route_by_intent(self, session_id: str, user_message: str, 
                        intent_result: Dict) -> Dict:
        """Route tin nhắn đến handler phù hợp dựa trên intent"""
        
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        
        # OVERRIDE: Kiểm tra CSV context và visualization request
        # Nếu có CSV và yêu cầu vẽ biểu đồ → LUÔN route về csv_analysis
        csv_contexts = self.session_manager.get_context(session_id, 'csv')
        if csv_contexts:
            message_lower = user_message.lower()
            viz_keywords = ['biểu đồ', 'chart', 'histogram', 'plot', 'vẽ', 'graph', 'visualize']
            if any(kw in message_lower for kw in viz_keywords):
                # Override intent → csv_analysis
                intent = 'csv_analysis'
                intent_result['intent'] = 'csv_analysis'
        
        # Nếu confidence thấp hoặc cần clarification
        if self.intent_classifier.needs_more_context(intent_result):
            clarification = self.intent_classifier.get_clarification_message(intent_result)
            return {
                'response': clarification,
                'intent': intent,
                'confidence': confidence,
                'context_updated': False,
                'suggested_actions': ['upload_csv', 'upload_image', 'chat']
            }
        
        # Route theo intent
        if intent == 'csv_analysis':
            return self._handle_csv_analysis(session_id, user_message, intent_result)
        
        elif intent == 'image_analysis':
            return self._handle_image_analysis(session_id, user_message, intent_result)
        
        elif intent == 'code_generation':
            return self._handle_code_generation(session_id, user_message, intent_result)
        
        elif intent == 'upload_request':
            return self._handle_upload_request(session_id, user_message, intent_result)
        
        elif intent == 'general_chat':
            return self._handle_general_chat(session_id, user_message, intent_result)
        
        else:
            # Default: general chat
            return self._handle_general_chat(session_id, user_message, intent_result)
    
    def _handle_csv_analysis(self, session_id: str, user_message: str, 
                            intent_result: Dict) -> Dict:
        
        # Lấy CSV context
        csv_contexts = self.session_manager.get_context(session_id, 'csv')
        
        if not csv_contexts:
            return {
                'response': 'Bạn chưa upload file CSV nào. Vui lòng upload file CSV để tôi có thể phân tích.',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['upload_csv']
            }
        
        # Lấy CSV file gần nhất
        latest_csv = csv_contexts[0]['data']
        csv_source = latest_csv.get('filepath') or latest_csv.get('url')
        
        # Phân tích với csv_process
        try:
            result = csv_process(csv_source, user_message)
            
            # Kiểm tra result là Dict (rich message) hay str
            if isinstance(result, dict):
                # Rich message với chart/table/mixed
                return {
                    'response': result.get('content', ''),
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence'],
                    'context_updated': False,
                    'suggested_actions': ['ask_more', 'upload_new'],
                    'message_type': result.get('type'),
                    'data': result.get('data', {})
                }
            else:
                # Text response thông thường
                return {
                    'response': result,
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence'],
                    'context_updated': False,
                    'suggested_actions': ['ask_more', 'upload_new']
                }
        
        except Exception as e:
            return {
                'response': f'Lỗi khi phân tích CSV: {str(e)}',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['upload_new']
            }
    
    def _handle_image_analysis(self, session_id: str, user_message: str, 
                              intent_result: Dict) -> Dict:
        
        # Lấy image context
        image_contexts = self.session_manager.get_context(session_id, 'image')
        
        if not image_contexts:
            return {
                'response': 'Bạn chưa upload ảnh nào. Vui lòng upload ảnh để tôi có thể phân tích.',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['upload_image']
            }
        
        # Lấy ảnh gần nhất
        latest_image = image_contexts[0]['data']
        image_path = latest_image.get('filepath')
        try:
            # Nếu có câu hỏi cụ thể, trả lời theo yêu cầu
            if len(user_message) > 10:
                prompt = f"Trả lời câu hỏi sau về ảnh này một cách đầy đủ và ngắn gọn: {user_message}"
                response_text = img_process(image_path, prompt)
            else:
                # Mô tả chung
                response_text = img_process(image_path)
            
            return {
                'response': response_text,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['ask_more', 'upload_new']
            }
        
        except Exception as e:
            return {
                'response': f'Lỗi khi phân tích ảnh: {str(e)}',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['upload_new']
            }
    
    def _handle_code_generation(self, session_id: str, user_message: str,
                                intent_result: Dict) -> Dict:
        """Xử lý yêu cầu sinh code"""
        
        try:
            # Generate và execute code
            result = self.code_executor.generate_and_execute(user_message)
            
            # Format output - KHÔNG hiển thị code, chỉ hiển thị kết quả
            formatted_result = self.code_executor.format_result(result, show_code=False)
            
            return {
                'response': formatted_result,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': ['modify_code', 'run_again', 'ask_more']
            }
        
        except Exception as e:
            return {
                'response': f'Lỗi khi sinh và thực thi code: {str(e)}',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': []
            }
    
    def _handle_upload_request(self, session_id: str, user_message: str, 
                              intent_result: Dict) -> Dict:
        """Xử lý yêu cầu upload"""
        
        response = """Bạn có thể upload file theo 2 cách:

1. Upload file CSV:
   - Click vào vùng upload
   - Hoặc kéo thả file CSV vào

2. Upload từ URL:
   - Nhập URL của file CSV (ví dụ: GitHub raw link)

Sau khi upload, bạn có thể hỏi tôi về dữ liệu!"""
        
        return {
            'response': response,
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'context_updated': False,
            'suggested_actions': ['show_upload_ui']
        }
    
    def _handle_general_chat(self, session_id: str, user_message: str, 
                           intent_result: Dict) -> Dict:
        """Xử lý trò chuyện thông thường"""
        
        # Lấy lịch sử để có context
        history = self.session_manager.get_conversation_history(session_id, limit=5)
        
        # Tạo messages cho API
        messages = [
            {"role": "system", "content": "Bạn là trợ lý AI thân thiện, giúp người dùng phân tích CSV và ảnh. Trò chuyện tự nhiên và hữu ích."}
        ]
        
        # Thêm history
        for msg in history[-4:]:  # 4 messages gần nhất
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Thêm user message hiện tại
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Gọi API
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=messages
            )
            
            response_text = response.choices[0].message.content
            
            return {
                'response': response_text,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': []
            }
        
        except Exception as e:
            return {
                'response': f'Xin lỗi, tôi gặp lỗi: {str(e)}',
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'context_updated': False,
                'suggested_actions': []
            }
    
    def add_file_context(self, session_id: str, file_type: str, file_data: Dict):
        self.session_manager.add_context(
            session_id=session_id,
            context_type=file_type,  # 'csv' hoặc 'image'
            context_data=file_data
        )
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Lấy tóm tắt session"""
        session_info = self.session_manager.get_session(session_id)
        history = self.session_manager.get_conversation_history(session_id)
        contexts = self.session_manager.get_context(session_id)
        
        return {
            'session_info': session_info,
            'message_count': len(history),
            'contexts': contexts,
            'last_messages': history[-5:] if history else []
        }


# Singleton instance
_conversation_handler = None

def get_conversation_handler() -> ConversationHandler:
    """Lấy singleton instance của ConversationHandler"""
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler()
    return _conversation_handler

