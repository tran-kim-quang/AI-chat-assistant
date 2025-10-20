"""
Intent Classifier - Phân loại ý định người dùng bằng AI
KHÔNG SỬ DỤNG KEYWORD MATCHING
"""
import openai
import os
from typing import Dict, Optional
from dotenv import load_dotenv
import json

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL_TEXT = os.getenv('TEXT_MODEL', 'llama-3.3-70b-versatile')


class IntentClassifier:
    INTENTS = {
        'csv_analysis': 'Người dùng muốn phân tích/hỏi về dữ liệu CSV',
        'image_analysis': 'Người dùng muốn phân tích/hỏi về ảnh',
        'code_generation': 'Người dùng muốn tạo/viết code Python để giải quyết vấn đề',
        'general_chat': 'Trò chuyện thông thường, không liên quan CSV hay ảnh',
        'upload_request': 'Người dùng muốn upload file hoặc ảnh',
        'clarification': 'Người dùng cần làm rõ hoặc hỏi về khả năng của hệ thống'
    }
    
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
    
    def classify(self, user_message: str, context_summary: str = "") -> Dict:
        
        # Tạo prompt cho LLM
        prompt = self._create_classification_prompt(user_message, context_summary)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là chuyên gia phân tích intent của người dùng trong hệ thống AI Assistant.
Hệ thống hỗ trợ các chức năng chính:
1. Phân tích CSV data
2. Phân tích ảnh (image vision)
3. Sinh code Python và thực thi (code generation)
4. Trò chuyện thông thường

Nhiệm vụ: Phân tích tin nhắn và xác định intent chính xác dựa trên ngữ nghĩa và ngữ cảnh.
QUAN TRỌNG: Trả về kết quả dưới dạng JSON hợp lệ."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Giảm temperature để kết quả ổn định hơn
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Nếu không parse được, extract từ text
                result = self._extract_intent_from_text(result_text)
            
            # Validate và bổ sung defaults
            return self._validate_result(result)
        
        except Exception as e:
            print(f"Lỗi phân loại intent: {e}")
            # Fallback: phân loại dựa trên context
            return self._fallback_classification(user_message, context_summary)
    
    def _create_classification_prompt(self, user_message: str, context_summary: str) -> str:
        intents_desc = "\n".join([f"- {key}: {desc}" for key, desc in self.INTENTS.items()])        
        prompt = f"""
Phân tích tin nhắn sau và xác định intent:

TIN NHẮN: "{user_message}"

NGỮ CẢNH HIỆN TẠI:
{context_summary if context_summary else "Chưa có context (session mới)"}

CÁC INTENT KHẢ DỤ:
{intents_desc}

Hãy phân tích và trả về JSON với format:
{{
    "intent": "<tên intent>",
    "confidence": <0.0-1.0>,
    "reasoning": "<giải thích ngắn gọn>",
    "suggested_action": "<hành động cụ thể>"
}}

LƯU Ý:
- Xem xét cả ngữ nghĩa và ngữ cảnh
- Nếu có file CSV trong context và câu hỏi liên quan dữ liệu → csv_analysis
- Nếu có ảnh trong context và câu hỏi liên quan ảnh → image_analysis
- Nếu người dùng muốn "viết code", "tạo hàm", "code để...", "tính toán..." → code_generation
- Nếu người dùng hỏi "bạn có thể làm gì?" → clarification
- Nếu người dùng nói "phân tích file này", "xem ảnh này" → cần kiểm tra context
- Trò chuyện bình thường như "xin chào", "cảm ơn" → general_chat

QUAN TRỌNG với code_generation:
- "Viết code tính tổng 1 đến 100" → code_generation
- "Tạo hàm sắp xếp list" → code_generation
- "Code để tính fibonacci" → code_generation
- "Làm sao để...[vấn đề lập trình]" → code_generation

Trả về CHÍNH XÁC format JSON trên, không thêm text nào khác.
"""
        
        return prompt
    
    def _extract_intent_from_text(self, text: str) -> Dict:
        """Extract intent từ text response khi không parse được JSON"""
        text_lower = text.lower()
        
        # Tìm intent trong text
        intent = 'general_chat'  # default
        for intent_name in self.INTENTS.keys():
            if intent_name in text_lower or intent_name.replace('_', ' ') in text_lower:
                intent = intent_name
                break
        
        return {
            'intent': intent,
            'confidence': 0.6,
            'reasoning': 'Extracted from text response',
            'suggested_action': 'Process based on intent'
        }
    
    def _validate_result(self, result: Dict) -> Dict:
        """Validate và bổ sung defaults cho result"""
        validated = {
            'intent': result.get('intent', 'general_chat'),
            'confidence': float(result.get('confidence', 0.5)),
            'reasoning': result.get('reasoning', ''),
            'suggested_action': result.get('suggested_action', '')
        }
        
        # Validate intent value
        if validated['intent'] not in self.INTENTS:
            validated['intent'] = 'general_chat'
            validated['confidence'] = 0.3
        
        # Clamp confidence
        validated['confidence'] = max(0.0, min(1.0, validated['confidence']))
        
        return validated
    
    def needs_more_context(self, intent_result: Dict) -> bool:
        """Kiểm tra xem có cần thêm context không"""
        return (
            intent_result['confidence'] < 0.5 or
            intent_result['intent'] == 'clarification'
        )
    
    def get_clarification_message(self, intent_result: Dict) -> str:
        """Tạo câu hỏi để làm rõ intent"""
        
        if intent_result['intent'] == 'clarification':
            return """Tôi có thể giúp bạn:
1. Phân tích dữ liệu CSV (upload file hoặc URL)
2. Phân tích và mô tả ảnh
3. Sinh code Python và thực thi (tính toán, xử lý dữ liệu, etc.)
4. Trò chuyện thông thường

Bạn muốn làm gì?"""
        
        elif intent_result['confidence'] < 0.5:
            return f"""Tôi chưa rõ bạn muốn làm gì. 
Bạn có thể:
- Upload file CSV và hỏi về dữ liệu
- Upload ảnh để phân tích
- Yêu cầu viết code Python
- Hoặc đơn giản là trò chuyện

Bạn muốn tôi giúp gì?"""
        
        return ""


# Singleton instance
_intent_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """Lấy singleton instance của IntentClassifier"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier

