from pathlib import Path
import re

def get_next_run_id(base_dir: Path, pattern: str = r".*?(\d+).*") -> int:
    """
    Finds the highest existing run_XXXX number in the base_dir and returns next number.
    
    Example: if directories run_003, run_007, run_042 exist → returns 43
    """
    if not base_dir.exists():
        return 1

    existing_numbers = []

    # Look for all subdirectories that match the pattern
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
            
        match = re.match(pattern, item.name)
        if match:
            try:
                num = int(match.group(1))
                existing_numbers.append(num)
            except ValueError:
                continue

    if not existing_numbers:
        return 1
        
    return max(existing_numbers) + 1