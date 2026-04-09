import re

from math_verify import parse


def extract_final_answer(response: str) -> str:
    """Extract a concise final answer from a raw model response."""
    text = response.strip()

    def _parsed_to_text(parsed: object) -> str:
        if parsed is None:
            return ""
        if isinstance(parsed, (list, tuple)):
            for item in reversed(parsed):
                item_text = str(item).strip()
                if item_text:
                    return item_text
            return ""
        return str(parsed).strip()

    # DeepSeek-style traces often end with </think> then final answer text.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

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
        return candidate

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
            return candidate

    # Last resort: try parsing the full remaining text only if explicit answer
    # markers were not present.
    try:
        parsed_full = parse(text)
        parsed_text = _parsed_to_text(parsed_full)
        if parsed_text:
            return parsed_text
    except Exception:
        pass

    return text
