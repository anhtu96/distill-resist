"""Answer accuracy evaluation for holdout datasets."""

import re

from src.config import ANSWER_FORCE_STRING, EVAL_MODE
from math_verify import parse, verify


class AnswerEvaluator:
    """Evaluates answer accuracy by comparing final_answer to ground_truth."""

    @staticmethod
    def _get_trace_text(sample: dict) -> str:
        """Return the best available generated text to evaluate."""
        return str(
            sample.get("response")
            or sample.get("trace")
            or sample.get("final_answer")
            or ""
        )

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Return content of the last \\boxed{...} in text, or text stripped."""
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        return matches[-1].strip() if matches else text.strip()

    @staticmethod
    def _extract_letter(text: str) -> str:
        """Extract a single answer letter (A-E) from text, handling formats like
        '**B**', 'Final Answer: (C)', '\\boxed{D}', 'E)**', etc.
        Returns the first matching letter found, or empty string."""
        # Try \boxed{X} first
        m = re.search(r'\\boxed\{([A-Ea-e])\}', text)
        if m:
            return m.group(1).upper()
        # Try "answer is X", "answer: X", "(X)", "**X**"
        m = re.search(r'(?:answer(?:\s+is)?[:\s]+\(?([A-Ea-e])\)?|\(([A-Ea-e])\)|\*\*([A-Ea-e])\*\*)', text, re.IGNORECASE)
        if m:
            return (m.group(1) or m.group(2) or m.group(3)).upper()
        # Last resort: first standalone letter A-E in text
        m = re.search(r'\b([A-Ea-e])\b', text)
        return m.group(1).upper() if m else ""

    @staticmethod
    def _is_correct_letter(sample: dict, ground_truth: str) -> bool:
        fa = str(sample.get("final_answer", "")).strip()
        trace = AnswerEvaluator._get_trace_text(sample)
        # Try final_answer first (strip junk like 'E)**'), then full trace
        candidate = AnswerEvaluator._extract_letter(fa) if fa else ""
        if not candidate:
            candidate = AnswerEvaluator._extract_letter(trace)
        return candidate.upper() == ground_truth.strip().upper()

    @staticmethod
    def _is_correct_string(sample: dict, ground_truth: str) -> bool:
        fa = str(sample.get("final_answer", "")).strip()
        trace = AnswerEvaluator._get_trace_text(sample)
        candidate = fa if fa else AnswerEvaluator._extract_boxed(trace)
        return candidate.strip().lower() == ground_truth.strip().lower()

    @staticmethod
    def evaluate_accuracy(samples: list[dict], dataset: str | None = None) -> dict:
        """Evaluate accuracy of samples against ground truth.

        Strategy:
        1) If `final_answer` exists, try verifying it first (most precise).
        2) If that fails or is missing, fall back to verifying the full trace,
           with special handling for `ANSWER_FORCE_STRING` splits.

        Returns a dict with keys: accuracy, correct, total, mismatches.
        """
        correct = 0
        mismatches = []
        mode = EVAL_MODE.get(dataset or "", "math_verify")

        for idx, sample in enumerate(samples, start=1):
            trace = AnswerEvaluator._get_trace_text(sample)
            ground_truth = str(sample.get("ground_truth", sample.get("solution", "")))

            if mode == "exact_letter":
                is_correct = AnswerEvaluator._is_correct_letter(sample, ground_truth)
            elif mode == "exact_string":
                is_correct = AnswerEvaluator._is_correct_string(sample, ground_truth)
            else:
                try:
                    truth = parse(ground_truth)
                    # Prefer checking the extracted final answer first when present
                    fa = str(sample.get("final_answer", "")).strip()
                    is_correct = False
                    if fa:
                        try:
                            is_correct = verify(truth, parse(fa))
                        except Exception:
                            is_correct = False

                    # Fall back to trace-based verification if needed
                    if not is_correct:
                        if ANSWER_FORCE_STRING in trace:
                            parts = trace.split(ANSWER_FORCE_STRING)
                            alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
                            alt_ans2 = parts[-1]
                            is_correct = any(
                                verify(truth, parse(ans))
                                for ans in [trace, alt_ans1, alt_ans2]
                            )
                        else:
                            is_correct = verify(truth, parse(trace))
                except Exception:
                    is_correct = False

            if is_correct:
                correct += 1
            else:
                mismatches.append(
                    {
                        "idx": idx,
                        "trace": trace,
                        "final_answer": sample.get("final_answer", ""),
                        "ground_truth": ground_truth,
                    }
                )

        total = len(samples)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mismatches": mismatches,
        }
