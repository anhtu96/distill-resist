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
        """Evaluate accuracy using the same trace-based logic as gentraces.py.

        Prefers the full generated response/trace when available and handles
        answer-forcing strings by trying multiple candidate splits.

        Args:
            samples: List of dicts with generated text and ground-truth fields.

        Returns:
            dict with keys:
                - accuracy: float (0.0 to 1.0)
                - correct: int (number of correct answers)
                - total: int (total samples)
                - mismatches: list of dicts with indices and mismatched pairs
        """
        correct = 0
        mismatches = []

        for idx, sample in enumerate(samples):
            trace = AnswerEvaluator._get_trace_text(sample)
            ground_truth = str(sample.get("ground_truth", sample.get("solution", "")))

            try:
                truth = parse(ground_truth)
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
