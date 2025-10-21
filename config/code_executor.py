import openai
import os
from typing import Dict, Any, Optional
import sys
from io import StringIO
import contextlib
import signal
import traceback
import threading
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
        'eval', 'exec', 'compile', 'open', 
        'input', 'raw_input', 'file', 'execfile', 'reload',
        'exit', 'quit', 'help', 'license', 'copyright', 'credits'
    }
    # Lưu ý: __import__ sẽ được thay bằng safe_import trong execute_code()
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
    
    def generate_code(self, user_request: str, language: str = 'python') -> Dict[str, Any]:
        
        allowed_modules_str = ', '.join(sorted(self.ALLOWED_IMPORTS))
        
        prompt = f"""
Bạn là chuyên gia lập trình. Hãy viết code theo ngôn ngữ được yêu cầu để thực hiện yêu cầu của người dùng:

NGÔN NGỮ: {language}
YÊU CẦU: {user_request}

QUY TẮC:
1. Chỉ được import các modules sau: {allowed_modules_str}
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
            # Parse module name từ import statement
            # "import math" → "math"
            # "from collections import Counter" → "collections"
            parts = imp_line.split()
            if len(parts) >= 2:
                module_name = parts[1].split('.')[0]  # Lấy base module
                if module_name not in self.ALLOWED_IMPORTS:
                    warnings.append(f"Import module '{module_name}' không trong whitelist: {imp_line}")
        
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
        
        # Tạo safe import function chỉ cho phép modules trong whitelist
        original_import = __import__
        def safe_import(name, *args, **kwargs):
            # Lấy module name gốc (trước dấu chấm đầu tiên)
            base_module = name.split('.')[0]
            if base_module in self.ALLOWED_IMPORTS:
                return original_import(name, *args, **kwargs)
            raise ImportError(f"Import module '{name}' không được phép. Chỉ cho phép: {', '.join(self.ALLOWED_IMPORTS)}")
        
        # Thêm safe_import vào builtins
        safe_builtins['__import__'] = safe_import
        
        restricted_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None
        }
        
        # Capture output
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        start_time = time.time()
        
        # Kiểm tra xem có phải main thread không
        is_main_thread = threading.current_thread() == threading.main_thread()
        use_signal_timeout = hasattr(signal, 'SIGALRM') and is_main_thread
        
        try:
            # Setup timeout (chỉ hoạt động trên Unix và trong main thread)
            if use_signal_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            # Redirect stdout và stderr
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec(code, restricted_globals)
            
            # Cancel timeout
            if use_signal_timeout:
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
            if use_signal_timeout:
                signal.alarm(0)
            output_buffer.close()
            error_buffer.close()
    
    def generate_and_execute(self, user_request: str, language: str = 'python') -> Dict[str, Any]:
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
        gen_result = self.generate_code(user_request, language)
        
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
    
    def format_result(self, result: Dict[str, Any], show_code: bool = False) -> str:
        """
        Format kết quả để hiển thị cho user
        
        Args:
            result: Kết quả từ generate_and_execute()
            show_code: Nếu True, hiển thị code. Nếu False, chỉ hiển thị output
        """
        
        if not show_code:
            # Chỉ hiển thị output, không hiển thị code
            if result['success']:
                return result['execution_result']['output']
            else:
                return f"Lỗi: {result['execution_result']['error']}"
        
        # Format đầy đủ với code (cho debug)
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

