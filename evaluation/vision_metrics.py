from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure tokenizer models are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

class VisionEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smoother = SmoothingFunction().method1

    def compute_metrics(self, prediction: str, reference: str) -> dict:
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())

        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoother)
        rouge_scores = self.rouge.score(reference, prediction)

        return {
            "bleu": round(bleu_score, 4),
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4)
        }