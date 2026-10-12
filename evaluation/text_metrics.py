from rouge_score import rouge_scorer

class TextEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def evaluate_summary(self, generated_summary: str, reference_summary: str) -> dict:
        scores = self.scorer.score(reference_summary, generated_summary)
        return {
            "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
        }