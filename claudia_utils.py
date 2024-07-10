from collections import abc
from colorama import init, Fore, Style

# Initialize colorama
init()
# Color constants
USER_COLOR = Fore.WHITE
CLAUDE_COLOR = Fore.BLUE
TOOL_COLOR = Fore.GREEN
RESULT_COLOR = Fore.YELLOW
ERROR_COLOR = Fore.RED
REMINDER_COLOR = Fore.MAGENTA
ASK_COLOR = Fore.CYAN

def print_colored(text, color):
    print(f"{color}{text}{Style.RESET_ALL}")