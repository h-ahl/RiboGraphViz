"""Copied from ToyFold 1D utils (Rhiju Das' Matlab code originally)."""

import math
from collections import Counter

import numpy as np

LEFT_DELIMITERS = ["(", "{", "[", "<"]
RIGHT_DELIMITERS = [")", "}", "]", ">"]


def find_all(text: str, query: str | list[str]) -> list[int]:
    """Finds all indices of characters in a string `s` that match any character in `ch`.

    Args:
        text: The string to search within.
        query (str or list of str): A character or a list of characters to find in the string.

    Returns:
        list of int: A list of indices where the characters in `ch` are found in `s`.
    """
    if isinstance(query, str):
        query = [query]  # Convert a single character to a list for uniform processing

    return [i for i, ltr in enumerate(text) if ltr in query]


def pairs_from_dotbracket(structure: str) -> list[list[int, int]]:
    """Returns a list of base pairs from an RNA secondary structure in dot-bracket notation.

    Args:
        structure: RNA secondary structure in dot-bracket notation.

    Returns:
        A list of base pairs, each represented as a list of two indices [i, j].
    """

    other_delimiters = [k for k in Counter(structure) if k not in RIGHT_DELIMITERS + LEFT_DELIMITERS + ["."]]

    bps = []
    for delimiter in other_delimiters:
        pos = find_all(structure, delimiter)

        n = int(len(pos) / 2)
        i, j = pos[:n], pos[n:]

        for ind in range(n):
            bps.append([i[ind], j[-1 - ind]])

    for left_delimiter, right_delimiter in list(zip(LEFT_DELIMITERS, RIGHT_DELIMITERS, strict=True)):
        left_indices = []
        for i, char in enumerate(structure):
            if char == left_delimiter:
                left_indices.append(i)
            elif char == right_delimiter and left_indices:
                bps.append([left_indices.pop(), i])

    return bps


def stems_from_pairs(pairs: list[list[int, int]]) -> list[list[list[int]]]:
    """Extracts stems from a list of base pairs. A stem is a sequence of continuous base pairs.

    Args:
        List of base pairs [i, j].

    Returns:
        List of stems, where each stem is a list of base pairs.

    Examples:
        >>> stems_from_pairs(pairs_from_dotbracket("((.))"))  # [[0,4],[1,3]]
        >>> stems_from_pairs(pairs_from_dotbracket('((.)).((.))'))  # [[[0,4],[1,3]],[[6,10],[7,9]]]
    """

    if len(pairs) == 0:
        return []

    # Find the maximum index value in pairs for boundary checks
    nres = np.max(pairs)
    stems = []

    while pairs:
        bp = pairs.pop(0)  # Take the first base pair
        stem = [bp]

        # Check outward
        bp_next = bp.copy()
        while bp_next[0] > 0:
            bp_next = [bp_next[0] - 1, bp_next[1] + 1]
            matching_pairs = [p for p in pairs if p[0] == bp_next[0] and p[1] == bp_next[1]]
            if matching_pairs:
                stem.append(bp_next)
                pairs.remove(matching_pairs[0])
            else:
                break

        # Check inward
        bp_next = bp.copy()
        while bp_next[0] < nres:
            bp_next = [bp_next[0] + 1, bp_next[1] - 1]
            matching_pairs = [p for p in pairs if p[0] == bp_next[0] and p[1] == bp_next[1]]
            if matching_pairs:
                stem.insert(0, bp_next)
                pairs.remove(matching_pairs[0])
            else:
                break

        stems.append(stem)

    return stems


def get_stem_assignment(structure: str) -> np.ndarray:
    """Returns an array indicating stem assignments for each bead in the structure.

    Args:
        structure: Dot-bracket notation or a partner vector.

    Returns:
        Array of stem assignments (1 to max(N_stems)), or 0 if not in a stem.
    """
    if structure[0].isdigit():
        # Convert partner vector to base pairs
        partner = [int(c) for c in structure]
        pairs = [[i, p] for i, p in enumerate(partner) if p > i]
    else:
        pairs = pairs_from_dotbracket(structure)

    # Extract stems and create assignment array
    stems = stems_from_pairs(pairs)
    stem_assignment = np.zeros(len(structure), dtype=int)

    for i, stem in enumerate(stems):
        stem_id = i + 1
        for bp in stem:
            stem_assignment[bp[0]] = stem_id
            stem_assignment[bp[1]] = stem_id

    return stem_assignment


def get_pairmap(structure: str) -> list[int]:
    """Generates a list containing pair mappings, where each position in the list holds the index for it'text paired
    base. If no pair exists, the value is -1.

    Args:
        Structure in dot-bracket notation.

    Returns:
        List with pair mappings.

    Note:
        Taken from draw_rna by Rhiju Das.
    """

    pair_stack = []
    end_stack = []
    pairs_array = [-1 for _ in range(len(structure))]  # -1 meaning no pair

    # Assign pairs based on structure
    for ii in range(len(structure)):
        if structure[ii] in LEFT_DELIMITERS:
            pair_stack.append(ii)
        elif structure[ii] in RIGHT_DELIMITERS:
            if not pair_stack:
                end_stack.append(ii)
            else:
                index = pair_stack.pop()
                pairs_array[index] = ii
                pairs_array[ii] = index

    if len(pair_stack) == len(end_stack):
        n = len(pair_stack)
        for ii in range(n):
            pairs_array[pair_stack[ii]] = end_stack[-ii]
            pairs_array[end_stack[-ii]] = pair_stack[ii]

    return pairs_array


def flip_helix(x, y, left_indices, right_indices):
    """Reflects points over the center line between two groups.

    Args:
        x: x-coordinates of points.
        y: y-coordinates of points.
        left_indices: Indices of points in the left group.
        right_indices: Indices of points in the right group.

    Returns:
        Updated x and y-coordinates of reflected points.
    """

    class1 = []
    class2 = []
    labels1 = []
    labels2 = []
    for i in left_indices:
        class1.append(np.array([x[i], y[i]]))
        labels1.append(-1)
    for i in right_indices:
        class2.append(np.array([x[i], y[i]]))
        labels1.append(1)
    class1 = np.array(class1)
    class2 = np.array(class2)
    x = np.vstack((class1, class2))
    x = np.array([np.ones(len(x)), x[:, 0], x[:, 1]]).T
    y = np.concatenate((labels1, labels2)).T
    beta = np.linalg.inv(x.T @ x) @ (x.T @ y)

    # Reflect all points over center line
    for i in left_indices + right_indices:
        temp = -2 * (beta[0] + beta[1] * x[i] + beta[2] * y[i]) / (beta[1] ** 2 + beta[2] ** 2)
        x[i] = temp * beta[1] + x[i]
        y[i] = temp * beta[2] + y[i]
    return x, y


def translate_group(x: list[float], y: list[float], offset: tuple[float, float], group: list[int]) -> tuple[
    list[float], list[float]]:
    """Moves points in `group` by the specified `offset`.

    Args:
        x: x-coordinates of points.
        y: y-coordinates of points.
        offset: (x, y) offset for movement.
        group: Indices of points to move.

    Returns:
        tuple of list of float: Updated x and y-coordinates of points.
    """

    for i in group:
        x[i] += offset[0]
        y[i] += offset[1]
    return x, y


def rotate_group(x: list[float], y: list[float], angle: float, group: list[int]) -> tuple[list[float], list[float]]:
    """Rotates points in `group` around their centroid by `angle` degrees.

    Args:
        x: x-coordinates of points.
        y: y-coordinates of points.
        angle: Rotation angle in degrees.
        group: Indices of points to rotate.

    Returns:
        Updated x and y-coordinates of points.
    """
    rotation = math.radians(angle)
    sum_x = 0
    sum_y = 0
    num_points = len(group)
    for i in group:
        sum_x += x[i]
        sum_y += y[i]
    centroid = [sum_x / num_points, sum_y / num_points]
    for i in group:
        x_orig = x[i]
        x[i] = centroid[0] + math.cos(rotation) * (x[i] - centroid[0]) - math.sin(rotation) * (y[i] - centroid[1])
        y[i] = centroid[1] + math.sin(rotation) * (x_orig - centroid[0]) + math.cos(rotation) * (y[i] - centroid[1])
    return x, y
