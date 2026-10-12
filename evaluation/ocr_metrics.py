import jiwer
from preprocessing.text.normalization import normalize_for_eval

class OCREvaluator:
    @staticmethod
    def evaluate_text_match(predicted_text: str, ground_truth_text: str) -> dict:
        clean_pred = normalize_for_eval(predicted_text)
        clean_gt = normalize_for_eval(ground_truth_text)

        cer = jiwer.cer(clean_gt, clean_pred)
        wer = jiwer.wer(clean_gt, clean_pred)

        # Accuracy percentage bounded between 0 and 100
        char_accuracy = max(0.0, 1.0 - cer) * 100

        return {
            "char_accuracy_pct": round(char_accuracy, 2),
            "cer": round(cer, 4),
            "wer": round(wer, 4)
        }