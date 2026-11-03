def prepare(train, test):
    """Lists of finite numbers or None; fit mean on train only. Return
    {'mean':float,'train':list,'test':list}; reject empty/all-missing train."""
    raise NotImplementedError("Implement prepare")

def deduplicate(rows):
    """Rows have session_id/value. Equal copies collapse; conflicting copy raises."""
    raise NotImplementedError("Implement deduplicate")
