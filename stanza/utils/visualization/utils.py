def find_nth(haystack, needle, n):
    """
    Returns the starting index of the nth occurrence of the substring 'needle' in the string 'haystack'.
    """
    pos = -1
    start = 0
    for _ in range(n):
        pos = haystack.find(needle, start)
        if pos == -1:
            return -1
        start = pos + len(needle)
    return pos


def round_base(num, base=10):
    """
    Rounding a number to its nearest multiple of the base. round_base(49.2, base=50) = 50.
    """
    return base * round(num / base)