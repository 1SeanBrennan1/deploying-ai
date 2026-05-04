# tools_custom.py
from langchain.tools import tool
from datetime import datetime
import math
import numexpr
from utils.logger import get_logger

_logs = get_logger(__name__)


@tool
def get_current_time() -> str:
    """
    Returns the current date and time.
    
    Use this tool when a user asks:
    - "What time is it?"
    - "What's today's date?"
    - "What day is it?"
    
    Returns:
        A formatted string with the current date and time
    """
    _logs.info('Getting current time')
    now = datetime.now()
    return now.strftime("It's currently %I:%M %p on %A, %B %d, %Y.")


@tool
def calculate(expression: str) -> str:
    """
    Performs mathematical calculations safely using numexpr.
    
    Use this tool for ANY math problem including:
    - Basic arithmetic: "2 + 2", "10 * 5"
    - Powers and roots: "2**10", "sqrt(49)"
    - Using constants: "pi * 2", "e ** 3"
    - Parentheses: "(3 + 4) * 2"
    
    Args:
        expression: A mathematical expression as a string
    
    Returns:
        The calculated result with the original expression
    """
    _logs.info(f'Calculating: {expression}')
    
    try:
        local_dict = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "abs": abs,
            "round": round,
            "pow": pow
        }
        
        result = numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict
        )
        
        result = result.item() if hasattr(result, 'item') else result
        
        if isinstance(result, float):
            result = round(result, 4)
        
        return f"The result of {expression} is {result}."
    
    except ZeroDivisionError:
        return "You can't divide by zero! That's like trying to split a pizza among no friends."
    
    except Exception as e:
        _logs.error(f'Calculation error: {str(e)}')
        return f"I ran into a problem calculating that. Could you rephrase the expression? Error: {str(e)}"