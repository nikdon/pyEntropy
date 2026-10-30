"""Template matching: pairs of delay-embedded windows whose Chebyshev distance is within a tolerance.

Two exact strategies produce the same matches. The banded one sorts the windows by their first value: only
windows within the tolerance in that value can match, and after sorting they are neighbours, so only a band of
pairs is compared. It is chosen when that band is small and the series is long enough; otherwise every pair is
compared. Matches are produced block by block, so callers can count them in total or per window.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# the sorted band search pays off from about this many windows (its setup costs more on short series)
BANDED_MIN_WINDOWS = 512
# and below this estimated cost (band fraction * template length, relative to comparing all pairs);
# the measured break-even is about 1
BANDED_MAX_COST = 0.75


def match_blocks(x, length, tolerance, inclusive=False):
    """Find the matching pairs of windows, block by block.

    Parameters
    ----------
    x : np.ndarray
        1D float series.
    length : int
        Window (template) length L; windows start at 0 .. len(x) - L.
    tolerance : float
        Two windows match on a prefix of length k when every one of their first k values differs by
        less than the tolerance (or at most the tolerance, with inclusive=True).
    inclusive : bool, optional
        Use d <= tolerance instead of d < tolerance.

    Returns
    -------
    order : np.ndarray or None
        Window index of each row and column position used by the blocks (the sort order), or None
        when positions are window indices.
    blocks : iterator of (k, rows, cols, matches, count)
        For each block and prefix length k = 1 .. L while any pair still matches: ``matches[r, c]`` is
        True when the windows at positions ``rows.start + r`` and ``cols.start + c`` match on their first k
        values, and ``count`` is the number of True entries. Every unordered pair of distinct windows
        appears in exactly one block, with the row before the column. The array is reused: consume it
        before advancing the iterator.

    """
    n_win = len(x) - length + 1
    if n_win < 2:
        return None, iter(())
    if n_win >= BANDED_MIN_WINDOWS:
        windows = sliding_window_view(x, length)[:n_win]
        order = np.argsort(windows[:, 0], kind="stable")
        columns = [np.ascontiguousarray(windows[order, k]) for k in range(length)]
        blocks = band_blocks(columns[0], tolerance)
        starts, stops, ends = blocks
        band_pairs = float(np.sum((stops - starts) * (ends - starts - 1)))
        if band_pairs * length < BANDED_MAX_COST * n_win * (n_win - 1) / 2:
            return order, banded_blocks(columns, tolerance, blocks, inclusive)
    return None, dense_blocks(x, length, tolerance, inclusive)


def band_blocks(first, tolerance, block_size=64):
    """Split sorted first values into blocks of rows and the end of the column band of each block.

    Row p can only match columns p + 1 .. end - 1, where end is the first value more than the
    tolerance above row p. The block's end is the one of its last row, which covers all its rows.

    Returns
    -------
    starts, stops, ends : np.ndarray
        Rows start .. stop - 1 of a block are compared with columns start + 1 .. end - 1.

    """
    n_win = len(first)
    # a superset of the band: the slack covers rounding in first + tolerance, the exact test decides
    slack = 4 * np.finfo(float).eps * (np.max(np.abs(first)) + tolerance)
    band_end = np.searchsorted(first, first + (tolerance + slack), side="right")
    starts = np.arange(0, n_win - 1, block_size)
    stops = np.minimum(starts + block_size, n_win - 1)
    ends = np.maximum(band_end[stops - 1], starts + 1)
    return starts, stops, ends


def banded_blocks(columns, tolerance, blocks, inclusive):
    """Yield the matches of windows sorted by their first value, comparing only the band."""
    compare = np.less_equal if inclusive else np.less
    starts, stops, ends = blocks
    rows_max = int(np.max(stops - starts))
    width_max = int(np.max(ends - starts - 1))
    if width_max <= 0:
        return

    size = rows_max * width_max
    diff_buf, hit_buf, cur_buf = np.empty(size), np.empty(size, dtype=np.bool_), np.empty(size, dtype=np.bool_)
    # triu[r, c] is True when c >= r, i.e. when column position start + 1 + c comes after row position start + r
    triu = np.triu(np.ones((rows_max, rows_max), dtype=np.bool_))

    for start, stop, end in zip(starts.tolist(), stops.tolist(), ends.tolist(), strict=True):
        rows, width = stop - start, end - start - 1
        if width <= 0:
            continue
        diff = diff_buf[: rows * width].reshape(rows, width)
        hit = hit_buf[: rows * width].reshape(rows, width)
        cur = cur_buf[: rows * width].reshape(rows, width)
        row_range, col_range = slice(start, stop), slice(start + 1, end)

        # cur[r, c]: positions start + r and start + 1 + c match on every value compared so far
        for k, column in enumerate(columns):
            np.subtract(column[start:stop, None], column[None, start + 1 : end], out=diff)
            np.abs(diff, out=diff)
            if k == 0:
                compare(diff, tolerance, out=cur)
                sq = min(rows - 1, width)
                if sq > 0:
                    cur[:, :sq] &= triu[:rows, :sq]  # each unordered pair once, no self-matches
            else:
                compare(diff, tolerance, out=hit)
                cur &= hit
            count = np.count_nonzero(cur)
            if count == 0:
                break
            yield k + 1, row_range, col_range, cur, count


def dense_blocks(x, length, tolerance, inclusive, chunk_size=32):
    """Yield the matches of all pairs of windows, comparing every pair once."""
    compare = np.less_equal if inclusive else np.less
    n_win = len(x) - length + 1

    # triu_full[r, c] is True when c >= r, i.e. when the second window starts after the first
    triu_full = np.triu(np.ones((chunk_size + length - 1,) * 2, dtype=np.bool_))

    # windows start .. end - 1 (rows) are compared with all later windows (columns)
    for start in range(0, n_win - 1, chunk_size):
        end = min(start + chunk_size, n_win - 1)
        b_len = end - start
        w_len = n_win - start - 1
        row_range, col_range = slice(start, end), slice(start + 1, n_win)

        # B[r, c] is True when |x[start + r] - x[start + 1 + c]| is within the tolerance. Entry (r + t, c + t)
        # is then the match of the pair (start + r, start + 1 + c) at offset t inside the windows
        diff = np.subtract(x[start : end + length - 1, None], x[None, start + 1 :])
        np.abs(diff, out=diff)
        B = compare(diff, tolerance)

        # keep only pairs whose second window starts after the first (no self- or double-matches)
        sq = min(b_len + length - 2, B.shape[1])
        if sq > 0:
            B[:, :sq] &= triu_full[: B.shape[0], :sq]

        cur = B[:b_len, :w_len].copy()
        # extend the matches one value at a time; `cur` can only shrink
        for k in range(1, length + 1):
            if k > 1:
                cur &= B[k - 1 : b_len + k - 1, k - 1 : w_len + k - 1]
            count = np.count_nonzero(cur)
            if count == 0:
                break
            yield k, row_range, col_range, cur, count


def count_matches(x, length, tolerance, inclusive=False):
    """Count matching unordered pairs of distinct windows, for each prefix length.

    Returns
    -------
    counts : np.ndarray
        counts[k] is the number of matching pairs on the first k values, k = 1 .. length (counts[0] is 0).

    """
    counts = np.zeros(length + 1)
    _, blocks = match_blocks(x, length, tolerance, inclusive)
    for k, _rows, _cols, _matches, count in blocks:
        counts[k] += count
    return counts


def count_matches_per_window(x, length, tolerance, inclusive=False):
    """Count, for every window, the other windows it matches, for each prefix length.

    Returns
    -------
    counts : np.ndarray
        Integer array of shape (length, n_windows): counts[k - 1, i] is the number of windows j != i that
        match window i on the first k values.

    """
    n_win = max(len(x) - length + 1, 0)
    counts = np.zeros((length, n_win), dtype=np.int64)
    order, blocks = match_blocks(x, length, tolerance, inclusive)
    for k, rows, cols, matches, _count in blocks:
        counts[k - 1, rows] += np.count_nonzero(matches, axis=1)
        counts[k - 1, cols] += np.count_nonzero(matches, axis=0)
    if order is not None:
        # positions refer to the sorted windows; put the counts back in window order
        unsorted = np.empty_like(counts)
        unsorted[:, order] = counts
        counts = unsorted
    return counts
