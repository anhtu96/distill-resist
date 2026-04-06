import re


def extract_final_answer(response: str) -> str:
    """Extract a concise final answer from a raw model response."""
    text = response.strip()

    # DeepSeek-style traces often end with </think> then final answer text.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Prefer the last boxed answer when present.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Common plain-text markers.
    markers = ["Final Answer:", "Answer:", "The answer is"]
    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()

    return text
