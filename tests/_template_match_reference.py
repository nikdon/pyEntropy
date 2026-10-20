def _brute_match_counts(series, sample_length, tolerance):
    """Count matching pairs of full windows: (prefix of length L - 1, full length L)."""
    n = len(series)
    n_prefix = 0
    n_full = 0
    for i in range(n - sample_length):
        for j in range(i + 1, n - sample_length + 1):
            prefix_close = all(abs(series[i + k] - series[j + k]) < tolerance for k in range(sample_length - 1))
            if prefix_close:
                n_prefix += 1
                if abs(series[i + sample_length - 1] - series[j + sample_length - 1]) < tolerance:
                    n_full += 1
    return n_prefix, n_full
