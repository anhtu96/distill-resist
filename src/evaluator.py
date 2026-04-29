"""Answer accuracy evaluation for holdout datasets."""

from src.config import ANSWER_FORCE_STRING
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
    def evaluate_accuracy(samples: list[dict]) -> dict:
        """Evaluate accuracy of samples against ground truth.

        Strategy:
        1) If `final_answer` exists, try verifying it first (most precise).
        2) If that fails or is missing, fall back to verifying the full trace,
           with special handling for `ANSWER_FORCE_STRING` splits.

        Returns a dict with keys: accuracy, correct, total, mismatches.
        """
        correct = 0
        mismatches = []

        for idx, sample in enumerate(samples, start=1):
            trace = AnswerEvaluator._get_trace_text(sample)
            ground_truth = str(sample.get("ground_truth", sample.get("solution", "")))

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
