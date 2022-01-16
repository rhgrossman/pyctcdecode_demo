def greedy_decode(logits, labels):
    """Decode argmax of logits and squash in CTC fashion."""
    label_dict = {n: c for n, c in enumerate(labels)}
    prev_c = None
    out = []
    for n in logits.argmax(axis=1):
        c = label_dict.get(n, "")  # if not in labels, then assume it's ctc blank char
        if c != prev_c:
            out.append(c)
        prev_c = c
    return "".join(out)