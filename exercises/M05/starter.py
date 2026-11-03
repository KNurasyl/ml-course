def split_records(df, train_before=144, validation_before=192):
    """Return train/validation/test/excluded row-ID lists. Keep groups whole;
    exclude a group spanning a boundary, preserve chronology."""
    raise NotImplementedError("Implement split_records")

def build_pipeline(C=1.):
    """Return sklearn pipeline with FitAudit first, train-only imputer/scaler,
    one-hot unknown-category handling, logistic classifier. Allowed FEATURES only.
    FitAudit and FEATURES may be imported from labs.M05.pipeline."""
    raise NotImplementedError("Implement build_pipeline")

def choose_threshold(y, probabilities, thresholds=(.25,.5,.75), fn_cost=2):
    """Minimize (fn_cost*FN+FP)/N using >=, tie -> smallest threshold."""
    raise NotImplementedError("Implement choose_threshold")
