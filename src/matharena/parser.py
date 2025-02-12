import regex
from typing import Optional
from loguru import logger

def find_last_boxed_content(text: str) -> Optional[str]:
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    return last_match.group(2)


def extract_boxed_answer(text: str) -> Optional[str]:
    answer = find_last_boxed_content(text)
    if answer is not None and "=" in answer:
        answer = answer.split("=")[-1]
    if answer is not None:
        return answer
    else:
        return None


def extract_boxed_int_answer(text: str) -> Optional[int]:
    answer = extract_boxed_answer(text)
    if answer is not None:
        try:
            return int(answer)
        except:
            logger.warning(f"Could not parse answer {answer} as integer")
            return None
    return None

def extract_last_integer(text: str) -> Optional[int]:
    pattern = r"\b\d+\b"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None
    return int(matches[-1].group())


def extract_answer(text: str, strict_parsing: bool = True):
    answer = extract_boxed_int_answer(text)
    if answer is not None or strict_parsing:
        return answer
    
    return extract_last_integer(text)
