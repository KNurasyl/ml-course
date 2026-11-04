def sign_test_p(wins, losses, ties=0):
    """Two-sided exact sign test p-value under H0: each non-tied pair is won by either side with probability 0.5.

    Ties are dropped. Return min(1, 2 * P(X >= max(wins, losses))) for X ~ Binomial(wins + losses, 0.5).
    Reject negative or non-integer counts and wins + losses == 0.
    """
    raise NotImplementedError("Implement sign_test_p")


def binary_log_loss(y_true, p):
    """Mean of -ln(probability of the observed class): -ln p for y = 1 and -ln(1 - p) for y = 0.

    Reject empty or unequal lists, labels other than 0 and 1, p outside [0, 1]
    and a zero probability for the observed class (the loss would be infinite).
    """
    raise NotImplementedError("Implement binary_log_loss")
