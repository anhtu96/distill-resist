"""Answer accuracy evaluation for holdout datasets."""


class AnswerEvaluator:
    """Evaluates answer accuracy by comparing final_answer to ground_truth."""

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer for comparison: strip whitespace, lowercase."""
        return answer.strip().lower()

    @staticmethod
    def evaluate_accuracy(samples: list[dict]) -> dict:
        """Evaluate accuracy by comparing final_answer to ground_truth.
        
        Args:
            samples: List of dicts with 'final_answer' and 'ground_truth' keys.
        
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
            final = AnswerEvaluator._normalize_answer(sample.get("final_answer", ""))
            truth = AnswerEvaluator._normalize_answer(sample.get("ground_truth", ""))

            if final == truth:
                correct += 1
            else:
                mismatches.append(
                    {
                        "idx": idx,
                        "final_answer": sample.get("final_answer", ""),
                        "ground_truth": sample.get("ground_truth", ""),
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
