import time
from typing import Dict, List, Tuple


def calculate_levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Computes Levenshtein edit distance between two token sequences."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (WER) for ASR and OCR."""
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    dist = calculate_levenshtein(ref_tokens, hyp_tokens)
    return round(min(1.0, dist / len(ref_tokens)), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER) for OCR and ASR."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = calculate_levenshtein(ref_chars, hyp_chars)
    return round(min(1.0, dist / len(ref_chars)), 4)


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Exact Match (EM) score for QA."""
    return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0


def compute_token_f1(reference: str, hypothesis: str) -> float:
    """Token-level F1 score for QA and captioning."""
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()
    if not ref_tokens or not hyp_tokens:
        return 1.0 if ref_tokens == hyp_tokens else 0.0

    common = set(ref_tokens) & set(hyp_tokens)
    if not common:
        return 0.0

    prec = sum(hyp_tokens.count(t) for t in common) / len(hyp_tokens)
    rec = sum(ref_tokens.count(t) for t in common) / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return round(2 * (prec * rec) / (prec + rec), 4)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """LCS-based ROUGE-L approximation for captioning and summarization."""
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return round(2 * (prec * rec) / (prec + rec), 4)