"""Lists of IDs, answer includes EOS. Padding is right-sided; no silent truncation."""
def training_example(prefix,answer,max_length,pad_id=0):
    """Return input_ids, attention_mask, labels of length max_length. Ignore index=-100."""
    raise NotImplementedError('Mask prefix/padding, preserve assistant targets and reject overflow/empty answer')
