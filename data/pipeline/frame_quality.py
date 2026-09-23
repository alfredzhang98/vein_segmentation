"""Resolve human quality decisions without mistaking missing labels for bad images."""


def frame_is_valid(row, automatically_valid=True):
    review = row.get("quality_review", "").lower()
    if review == "valid":
        return True
    if review == "invalid" or row.get("frame_valid", "").lower() == "false":
        return False
    return bool(automatically_valid)


def reviewed_validity(sequences, rows):
    """Copy auto-screened flags, then apply decisions at original frame indices."""
    flags = {seq: list(record["valid"]) for seq, record in sequences.items()}
    for row in rows:
        seq, frame = row["sequence_id"], int(row["frame_index"])
        if seq not in flags or not 0 <= frame < len(flags[seq]):
            raise ValueError(f"Quality decision has unknown source frame: {seq}/{frame}")
        flags[seq][frame] = frame_is_valid(row, flags[seq][frame])
    return flags
