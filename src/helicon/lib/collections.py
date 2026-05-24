from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import pandas as pd

__all__ = [
    "unique",
    "assign_to_groups",
    "flatten",
    "order_by_unique_counts",
    "split_array",
    "unique_attr_name",
    "all_matched_attrs",
    "first_matched_attr",
    "DotDict",
]


def unique(inputList: list) -> list:
    """Return unique elements from a list while preserving order.

    Parameters
    ----------
    inputList : list
        Input list.

    Returns
    -------
    list
        List with duplicates removed, order preserved.
    """
    ret = []
    for v in inputList:
        if v not in ret:
            ret.append(v)
    return ret


def assign_to_groups(numbers: Iterable, group_size: int) -> dict:
    """Sort values and assign them to groups of a given size.

    Parameters
    ----------
    numbers : Iterable
        Values to assign to groups.
    group_size : int
        Maximum number of values per group.

    Returns
    -------
    dict
        Mapping from value to group ID.
    """
    from collections import defaultdict

    sorted_numbers = sorted(numbers)

    # Group duplicate values
    value_groups = defaultdict(list)
    for i, num in enumerate(sorted_numbers):
        value_groups[num].append(i)

    result = {}
    group_id = 1
    current_group = []
    current_group_size = 0

    # Group the numbers
    for num, indices in value_groups.items():
        if current_group_size + len(indices) > group_size:
            # If adding this set of duplicates exceeds the group size,
            # finalize the current group and start a new one
            if current_group:
                for value in current_group:
                    result[value] = group_id
                group_id += 1
            current_group = [num] * len(indices)
            current_group_size = len(indices)
        else:
            # Add the duplicates to the current group
            current_group.extend([num] * len(indices))
            current_group_size += len(indices)

        # If the group is full, finalize it
        if current_group_size == group_size:
            for value in current_group:
                result[value] = group_id
            group_id += 1
            current_group = []
            current_group_size = 0

    # Handle the last group
    if current_group:
        if len(current_group) < group_size // 2 and len(result) > 0:
            # Merge with the previous group if it's less than half the group size
            prev_group_id = max(result.values())
            for value in current_group:
                result[value] = prev_group_id
        else:
            # Add as a new group
            for value in current_group:
                result[value] = group_id

    return result


# flatten multiple level list or tuple
# taken from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
def flatten(l, ltypes: tuple = (list, tuple)) -> list | tuple:
    """Flatten a nested list or tuple into a single level.

    Recursively expands any elements that are instances of *ltypes*.

    Parameters
    ----------
    l : list or tuple
        Nested collection.
    ltypes : tuple, optional
        Container types to flatten. Defaults to
        ``(list, tuple)``.

    Returns
    -------
    list or tuple
        Flattened collection (type matches input).
    """
    ltype = type(l)
    if ltype not in ltypes:
        ltype = list
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i : i + 1] = l[i]
        i += 1
    return ltype(l)


def order_by_unique_counts(
    labels: np.ndarray | list, ignoreNegative: bool = True
) -> list:
    """Reorder labels by their frequency (most frequent first).

    Parameters
    ----------
    labels : array-like
        Input labels.
    ignoreNegative : bool, optional
        If True, negative labels are
        preserved as-is and placed after positive ones. Defaults to True.

    Returns
    -------
    list
        Labels reordered by decreasing frequency.
    """
    if ignoreNegative:
        labels_pos = labels[labels >= 0]
        unique, counts = np.unique(labels_pos, return_counts=True)
        order = np.argsort(counts)[::-1]
        mapping = {unique[v]: i for i, v in enumerate(order)}
        labels_neg = labels[labels < 0]
        mapping.update({v: v for v in np.unique(labels_neg)})
    else:
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        mapping = {unique[v]: i for i, v in enumerate(order)}
    ret = [mapping[v] for v in labels]
    return ret


def split_array(arr: list) -> tuple[list, list]:
    """Split an array into two groups minimizing the difference of their sums.

    Parameters
    ----------
    arr : list
        Input array of numeric values.

    Returns
    -------
    tuple of (list, list)
        Indices of elements in group 1 and group 2.
    """
    total_sum = sum(arr)
    target_sum = total_sum // 2
    n = len(arr)

    # Create a 2D DP table
    dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]

    # Initialize the first column
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if arr[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    # Find the largest sum <= target_sum that can be achieved
    achieved_sum = 0
    for j in range(target_sum, -1, -1):
        if dp[n][j]:
            achieved_sum = j
            break

    # Backtrack to find the elements in the first group
    group1 = []
    i, j = n, achieved_sum
    while i > 0 and j > 0:
        if not dp[i - 1][j]:
            group1.append(i - 1)
            j -= arr[i - 1]
        i -= 1

    # The second group consists of all elements not in the first group
    group2 = [i for i in range(n) if i not in group1]

    return group1, group2


def unique_attr_name(data, attr_prefix: str) -> str:
    """Return an attribute name that is not already present in *data*.

    Appends incrementing integers to the prefix until a free name is found.

    Parameters
    ----------
    data : container
        Container supporting ``__contains__`` (e.g. dict,
        list, DataFrame columns).
    attr_prefix : str
        Desired prefix for the attribute name.

    Returns
    -------
    str
        Unique attribute name.
    """
    if attr_prefix not in data:
        return attr_prefix
    attr_i = 2
    attr = f"{attr_prefix}{attr_i}"
    while attr in data:
        attr_i += 1
        attr = f"{attr_prefix}{attr_i}"
    return attr


def all_matched_attrs(data: pd.DataFrame | Any, query_str: str) -> list:
    """Return all column/field names containing a query string.

    Parameters
    ----------
    data : pd.DataFrame or cryosparc.tools.Dataset
        Data object.
    query_str : str
        Substring to search for in column names.

    Returns
    -------
    list of str
        Matching column names.

    Raises
    ------
    TypeError
        If *data* is not a DataFrame or Dataset.
    """
    import pandas as pd
    from cryosparc.tools import Dataset

    if isinstance(data, pd.DataFrame):
        cols = data.columns
    elif isinstance(data, Dataset):
        cols = list(data.keys())
    else:
        raise TypeError(
            f"first_matched_atrrs(data, query_str): data is a {type(data)} but it must be a pandas dataframe or a cryosparc.tools.Dataset"
        )

    ret = [col for col in cols if col.find(query_str) != -1]
    return ret


def first_matched_attr(data: Any, attrs: list) -> str | None:
    """Return the first attribute from a list that exists in *data*.

    Parameters
    ----------
    data : container
        Container supporting ``__contains__``.
    attrs : list of str
        Candidate attribute names.

    Returns
    -------
    str or None
        The first matching attribute, or None.
    """
    ret = None
    for attr in attrs:
        if attr in data:
            ret = attr
            break
    return ret


class DotDict(dict):
    """A dictionary subclass that supports attribute-style access.

    Items can be get/set via ``d.key`` in addition to ``d["key"]``.
    """

    def __getattr__(self, name):
        """Get an item via attribute-style access."""
        return self[name]

    def __setattr__(self, name, value):
        """Set an item via attribute-style access."""
        self[name] = value
