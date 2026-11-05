"""Reference for M11-ex02: answer and refusal rates, judge agreement with invalid outputs, Cohen's kappa."""

from collections import Counter


def answer_rates(records):
    records = list(records)
    if not records:
        raise ValueError("need at least one record")
    for row in records:
        if any(type(row.get(key)) is not bool for key in ("answered", "correct", "answerable")):
            raise ValueError("answered, correct and answerable must be explicit booleans")
        if row["correct"] and not row["answered"]:
            raise ValueError("a refusal cannot be a correct answer")
    n = len(records)
    answered = sum(row["answered"] for row in records)
    correct = sum(row["correct"] for row in records)
    refusals = n - answered
    appropriate = sum(not row["answered"] and not row["answerable"] for row in records)
    return {
        "coverage": answered / n,
        "accuracy_answered": correct / answered,  # quality among answers only: refusals are not in it
        "correct_total": correct / n,
        "appropriate_refusals": appropriate / refusals,
        "success": (correct + appropriate) / n,  # an appropriate refusal is a right outcome too
    }


def judge_agreement(human, judge):
    human, judge = list(human), list(judge)
    if not human or len(human) != len(judge):
        raise ValueError("need two non-empty lists of equal length")
    for label in human + [v for v in judge if v is not None]:
        if type(label) is not int or label not in (0, 1):
            raise ValueError("labels must be 0 or 1; None marks an invalid judge output")
    n = len(human)
    valid = [i for i, label in enumerate(judge) if label is not None]
    matches = sum(judge[i] == human[i] for i in valid)
    return {
        "coverage": len(valid) / n,
        "agreement_valid": matches / len(valid),
        "correct_over_all": matches / n,  # an invalid output is not a correct judgement
        "false_approvals": [i + 1 for i in valid if judge[i] == 1 and human[i] == 0],
    }


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    if not a or len(a) != len(b):
        raise ValueError("need two non-empty lists of equal length")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    share_a, share_b = Counter(a), Counter(b)
    # Agreement expected if each rater kept their own label shares but labelled at random.
    p_e = sum(share_a[label] * share_b[label] for label in share_a) / (n * n)
    return (p_o - p_e) / (1 - p_e)
