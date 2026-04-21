import re

from math_verify import parse


def extract_final_answer(response: str) -> str:
    """Extract a concise final answer from a raw model response."""
    text = response.strip()

    def _clean_answer(answer: str) -> str:
        """Clean extracted answer by removing prefixes, symbols, and normalizing decimals."""
        answer = answer.strip()
        # Remove "ANSWER:" prefix (case insensitive)
        answer = re.sub(r'^ANSWER:\s*', '', answer, flags=re.IGNORECASE)
        # Remove brackets
        answer = answer.strip('[](){}')
        # Remove currency symbols
        answer = re.sub(r'^[$€£¥]', '', answer)
        # Normalize: remove trailing .00 for whole numbers
        answer = re.sub(r'\.00$', '', answer)
        # Normalize: remove trailing .0 for whole numbers
        answer = re.sub(r'\.0$', '', answer)
        # Remove text after numbers (e.g., "72 clips" -> "72")
        # Match number at start, then optional whitespace and text
        match = re.match(r'^([-\d\.]+)', answer)
        if match:
            answer = match.group(1)
        return answer.strip()

    def _parsed_to_text(parsed: object) -> str:
        if parsed is None:
            return ""
        if isinstance(parsed, (list, tuple)):
            for item in reversed(parsed):
                item_text = str(item).strip()
                if item_text:
                    return _clean_answer(item_text)
            return ""
        return _clean_answer(str(parsed).strip())

    # DeepSeek-style traces often end with ``` then final answer text.
    if "```" in text:
        text = text.split("```", 1)[1].strip()

    # Prefer the last boxed answer when present.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        candidate = boxed[-1].strip()
        try:
            parsed_candidate = parse(candidate)
            parsed_text = _parsed_to_text(parsed_candidate)
            if parsed_text:
                return parsed_text
        except Exception:
            pass
        return _clean_answer(candidate)

    # Common plain-text markers.
    markers = ["Final Answer:", "Answer:", "The answer is"]
    for marker in markers:
        if marker in text:
            candidate = text.split(marker, 1)[1].strip()
            try:
                parsed_candidate = parse(candidate)
                parsed_text = _parsed_to_text(parsed_candidate)
                if parsed_text:
                    return parsed_text
            except Exception:
                pass
            return _clean_answer(candidate)

    # Last resort: try parsing the full remaining text only if explicit answer
    # markers were not present. If parsing fails, do NOT return arbitrary text.
    try:
        parsed_full = parse(text)
        parsed_text = _parsed_to_text(parsed_full)
        if parsed_text:
            return parsed_text
    except Exception:
        pass

    # Could not extract a reliable answer.
    return ""
