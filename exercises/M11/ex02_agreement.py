def answer_rates(records):
    """Rates of a run where the system may answer or refuse.

    records: list of {"answered": bool, "correct": bool, "answerable": bool}; a refusal has correct = False.
    Return {"coverage": answered / all, "accuracy_answered": correct / answered, "correct_total": correct / all,
    "appropriate_refusals": refusals of unanswerable questions / refusals,
    "success": (correct answers + appropriate refusals) / all}.
    The checker never passes a run without answers or without refusals: what a rate with a zero
    denominator means is your call (variant C of M11-L01).
    Reject an empty list, non-bool fields and correct = True for a refusal.
    """
    raise NotImplementedError("Implement answer_rates")


def judge_agreement(human, judge):
    """Compare judge labels with independent human labels (0 or 1); None marks an invalid judge output.

    Return {"coverage": valid / all, "agreement_valid": matches / valid, "correct_over_all": matches / all,
    "false_approvals": 1-based positions where the judge gave 1 and the human 0}.
    Invalid outputs stay in the denominator of coverage and correct_over_all. The checker never passes
    a run without a valid judge label.
    Reject empty or unequal lists, labels other than 0 and 1 (bool included) and None among human labels.
    """
    raise NotImplementedError("Implement judge_agreement")


def cohen_kappa(a, b):
    """Cohen's kappa of two raters over the same cases: (p_o - p_e) / (1 - p_e).

    p_o is the share of cases with equal labels; p_e is the sum over labels of the product of the shares
    of that label in a and in b. Labels are any hashable values: 0, 1, 2 or strings.
    The checker never passes inputs with p_e = 1 (variant C of M11-L02).
    Reject empty or unequal lists.
    """
    raise NotImplementedError("Implement cohen_kappa")
