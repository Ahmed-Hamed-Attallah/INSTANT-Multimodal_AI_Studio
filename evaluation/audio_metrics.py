import jiwer
from preprocessing.text.normalization import normalize_for_eval

class AudioEvaluator:
    @staticmethod
    def compute_wer(prediction: str, reference: str) -> dict:
        # Normalize text to remove formatting discrepancies
        norm_pred = normalize_for_eval(prediction)
        norm_ref = normalize_for_eval(reference)

        if not norm_ref:
            return {"wer": 0.0, "cer": 0.0}

        wer = jiwer.wer(norm_ref, norm_pred)
        cer = jiwer.cer(norm_ref, norm_pred)

        return {
            "wer": round(wer, 4),
            "cer": round(cer, 4)
        }