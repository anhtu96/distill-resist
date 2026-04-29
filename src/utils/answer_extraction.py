import re

from math_verify import parse


def _extract_last_boxed(text: str) -> str | None:
    """Return content of last \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def extract_final_answer(response: str) -> str:
    """Extract a concise final answer from a raw model response."""
    text = response.strip()

    def _clean_answer(answer: str) -> str:
        """Clean extracted answer. Only strip numeric-only noise; preserve math expressions."""
        answer = answer.strip()
        answer = re.sub(r'^ANSWER:\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^[$€£¥]', '', answer)
        # Only strip surrounding plain brackets when the content has no LaTeX
        if not any(c in answer for c in ('\\', '{', '}')):
            answer = answer.strip('[](){}')
        # Normalize trailing .00 / .0 only for bare numbers
        if re.fullmatch(r'-?\d+\.00', answer):
            answer = answer[:-3]
        elif re.fullmatch(r'-?\d+\.0', answer):
            answer = answer[:-2]
        # Strip trailing word units only when the answer is a plain number
        # (e.g. "72 clips" → "72"), but leave math expressions untouched.
        if re.match(r'^-?[\d\.]+\s+[A-Za-z]', answer):
            m = re.match(r'^(-?[\d\.]+)', answer)
            if m:
                answer = m.group(1)
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

    # Prefer the last boxed answer when present (nested-brace-aware).
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        try:
            parsed_candidate = parse(boxed)
            parsed_text = _parsed_to_text(parsed_candidate)
            if parsed_text:
                return parsed_text
        except Exception:
            pass
        return _clean_answer(boxed)

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

    return ""
