import hashlib


def get_sequence_hash(sequence_str: str) -> str:
    """Generates a SHA-256 hash for the given sequence string."""
    hasher = hashlib.sha256()
    hasher.update(sequence_str.encode("utf-8"))
    return hasher.hexdigest()
