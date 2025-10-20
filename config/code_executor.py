import openai
import os
from typing import Dict, Any, Optional
import sys
from io import StringIO
import contextlib
import signal
import traceback
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL_TEXT = os.getenv('TEXT_MODEL')


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timeout")


class CodeExecutor:
    """Thực thi code Python trong môi trường an toàn"""
    
    # Các modules được phép import
    ALLOWED_IMPORTS = {
        'math', 'random', 'datetime', 'collections', 'itertools',
        'functools', 're', 'json', 'statistics', 'decimal',
        'fractions', 'operator', 'string', 'time'
    }
    
    # Các built-in functions bị cấm
    FORBIDDEN_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 
        'input', 'raw_input', 'file', 'execfile', 'reload',
        'exit', 'quit', 'help', 'license', 'copyright', 'credits'
    }
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
    
    def generate_code(self, user_request: str, language: str) -> Dict[str, Any]:
        
        prompt = f"""
Bạn là chuyên gia lập trình. Hãy viết code theo ngôn ngữ được yêu cầu để thực hiện yêu cầu của người dùng:

NGÔN NGỮ: {language}
YÊU CẦU: {user_request}

QUY TẮC:
1. Chỉ sử dụng {language} standard library
2. Code phải hoàn chỉnh và có thể chạy ngay
3. Sử dụng hàm main() để tổ chức code
4. In kết quả ra console với print()
5. Không sử dụng input(), file I/O, hoặc network operations
6. Code phải ngắn gọn, hiệu quả
7. Thêm comments giải thích

Trả về JSON format:
{{
    "code": "<code {language} hoàn chỉnh>",
    "explanation": "<giải thích ngắn gọn code làm gì>",
    "language": "python",
    "ready_to_run": true
}}

QUAN TRỌNG: Trả về CHÍNH XÁC format JSON, không thêm text nào khác.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {
                        "role": "system",
                        "content": f"Bạn là chuyên gia {language}, viết code sạch, hiệu quả và an toàn."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            
            # Xử lý nếu có markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Validate
            if 'code' not in result:
                raise ValueError("Missing 'code' in response")
            
            return {
                'code': result.get('code', ''),
                'explanation': result.get('explanation', ''),
                'language': result.get('language', 'python'),
                'ready_to_run': result.get('ready_to_run', True)
            }
        
        except Exception as e:
            return {
                'code': f'# Error generating code: {str(e)}',
                'explanation': f'Lỗi khi sinh code: {str(e)}',
                'language': language,
                'ready_to_run': False
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Kiểm tra code có an toàn để thực thi không
        
        Returns:
            {
                'is_safe': bool,
                'issues': List[str],
                'warnings': List[str]
            }
        """
        issues = []
        warnings = []
        
        # Kiểm tra forbidden keywords
        dangerous_patterns = [
            'os.system', 'subprocess', 'eval(', 'exec(',
            '__import__', 'open(', 'file(', 'input(',
            'socket', 'urllib', 'requests', 'http'
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                issues.append(f"Phát hiện pattern nguy hiểm: {pattern}")
        
        # Kiểm tra imports
        import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        
        for imp_line in import_lines:
            for module in imp_line.split():
                if module not in self.ALLOWED_IMPORTS and module not in ['import', 'from', 'as']:
                    if not any(allowed in imp_line for allowed in self.ALLOWED_IMPORTS):
                        warnings.append(f"Import không trong whitelist: {imp_line}")
        
        is_safe = len(issues) == 0
        
        return {
            'is_safe': is_safe,
            'issues': issues,
            'warnings': warnings
        }
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Thực thi code trong môi trường sandbox
        
        Returns:
            {
                'success': bool,
                'output': str,
                'error': str,
                'execution_time': float
            }
        """
        import time
        
        # Validate trước
        validation = self.validate_code(code)
        
        if not validation['is_safe']:
            return {
                'success': False,
                'output': '',
                'error': 'Code không an toàn để thực thi:\n' + '\n'.join(validation['issues']),
                'execution_time': 0
            }
        
        # Tạo restricted globals
        safe_builtins = {
            k: v for k, v in __builtins__.items() 
            if k not in self.FORBIDDEN_BUILTINS
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) 
            for k in dir(__builtins__) 
            if k not in self.FORBIDDEN_BUILTINS
        }
        
        restricted_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None
        }
        
        # Capture output
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        start_time = time.time()
        
        try:
            # Setup timeout (chỉ hoạt động trên Unix)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            # Redirect stdout và stderr
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec(code, restricted_globals)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            execution_time = time.time() - start_time
            
            output = output_buffer.getvalue()
            error = error_buffer.getvalue()
            
            if error:
                return {
                    'success': False,
                    'output': output,
                    'error': error,
                    'execution_time': execution_time
                }
            
            return {
                'success': True,
                'output': output if output else 'Code chạy thành công (không có output)',
                'error': '',
                'execution_time': execution_time
            }
        
        except TimeoutException:
            return {
                'success': False,
                'output': output_buffer.getvalue(),
                'error': f'Code execution timeout ({self.timeout}s)',
                'execution_time': self.timeout
            }
        
        except Exception as e:
            return {
                'success': False,
                'output': output_buffer.getvalue(),
                'error': f'{type(e).__name__}: {str(e)}\n{traceback.format_exc()}',
                'execution_time': time.time() - start_time
            }
        
        finally:
            # Cleanup
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            output_buffer.close()
            error_buffer.close()
    
    def generate_and_execute(self, user_request: str) -> Dict[str, Any]:
        """
        Sinh code và thực thi luôn
        
        Returns:
            {
                'code': str,
                'explanation': str,
                'execution_result': Dict,
                'success': bool
            }
        """
        # Generate code
        gen_result = self.generate_code(user_request)
        
        if not gen_result['ready_to_run']:
            return {
                'code': gen_result['code'],
                'explanation': gen_result['explanation'],
                'execution_result': {
                    'success': False,
                    'output': '',
                    'error': 'Code không sẵn sàng để chạy',
                    'execution_time': 0
                },
                'success': False
            }
        
        # Execute code
        exec_result = self.execute_code(gen_result['code'])
        
        return {
            'code': gen_result['code'],
            'explanation': gen_result['explanation'],
            'execution_result': exec_result,
            'success': exec_result['success']
        }
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format kết quả để hiển thị cho user"""
        
        output = f"""
CODE GENERATED:
{'='*60}
{result['code']}

EXPLANATION:
{'='*60}
{result['explanation']}

EXECUTION RESULT:
{'='*60}
"""
        
        if result['success']:
            output += f"Status: SUCCESS\n"
            output += f"Output:\n{result['execution_result']['output']}\n"
            output += f"Execution time: {result['execution_result']['execution_time']:.4f}s\n"
        else:
            output += f"Status: FAILED\n"
            output += f"Error:\n{result['execution_result']['error']}\n"
        
        return output


# Singleton instance
_code_executor = None

def get_code_executor() -> CodeExecutor:
    """Lấy singleton instance"""
    global _code_executor
    if _code_executor is None:
        _code_executor = CodeExecutor()
    return _code_executor

