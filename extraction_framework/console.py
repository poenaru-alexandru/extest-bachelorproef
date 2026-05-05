import sys
import builtins
from colorama import init, Fore, Style

# Initialize colorama to ensure ANSI colors work on Windows
init(autoreset=True)

_original_print = builtins.print
_use_color = sys.stdout.isatty()

def colored_print(*args, **kwargs):
    if not args:
        _original_print(*args, **kwargs)
        return

    text = str(args[0])

    if not _use_color:
        _original_print(text, *args[1:], **kwargs)
        return
    
    # Categorize by keywords
    if "ERROR" in text or "Error" in text or "❌" in text or "failed" in text.lower():
        # Errors in Red
        text = f"{Fore.RED}{text}{Style.RESET_ALL}"
    elif "Warning" in text or "WARNING" in text:
        # Warnings in Yellow
        text = f"{Fore.YELLOW}{text}{Style.RESET_ALL}"
    elif "✓" in text or "success" in text.lower() or "completed" in text.lower() or "Found" in text or "Loaded" in text or "Saved" in text:
        # Success/Positive info in Green
        text = f"{Fore.GREEN}{text}{Style.RESET_ALL}"
    elif "[API]" in text:
        # General API logs in Cyan
        text = f"{Fore.CYAN}{text}{Style.RESET_ALL}"
    elif "BATCH RUN" in text or "Testing:" in text or "[LlamaCpp]" in text or "[ModelLoader]" in text or "[Scoring]" in text:
        # Framework stages in Magenta
        text = f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"
    elif "=====" in text or "#####" in text:
        # Dividers in Blue
        text = f"{Fore.BLUE}{text}{Style.RESET_ALL}"
    else:
        # Default general info
        pass

    # Pass the colored text and any remaining arguments to the original print
    _original_print(text, *args[1:], **kwargs)

# Override the built-in print globally
builtins.print = colored_print
