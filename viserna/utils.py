"""Copied from ToyFold 1D utils (Rhiju Das' Matlab code originally)."""

from collections import Counter

import numpy as np

LEFT_DELIMITERS = ["(", "{", "[", "<"]
RIGHT_DELIMITERS = [")", "}", "]", ">"]


def find_all(text: str, query: str | list[str]) -> list[int]:
    """Finds all indices of characters in a string `s` that match any character in `ch`.

    Args:
        text: The string to search within.
        query: A character or a list of characters to find in the string.

    Returns:
        list of int: A list of indices where the characters in `ch` are found in `s`.
    """
    if isinstance(query, str):
        query = [query]

    return [i for i, ltr in enumerate(text) if ltr in query]


def pairs_from_dotbracket(structure: str) -> list[list[int, int]]:
    """Returns a list of base pairs from an RNA secondary structure in dot-bracket notation.

    Args:
        structure: RNA secondary structure in dot-bracket notation.

    Returns:
        A list of base pairs, each represented as a list of two indices [i, j].
    """

    other_delimiters = [k for k in Counter(structure) if k not in RIGHT_DELIMITERS + LEFT_DELIMITERS + ["."]]

    pairs = []
    for delimiter in other_delimiters:
        index = find_all(structure, delimiter)

        n = int(len(index) / 2)
        i, j = index[:n], index[n:]

        for ind in range(n):
            pairs.append([i[ind], j[-1 - ind]])

    for left_delimiter, right_delimiter in list(zip(LEFT_DELIMITERS, RIGHT_DELIMITERS, strict=True)):
        left_indices = []
        for i, char in enumerate(structure):
            if char == left_delimiter:
                left_indices.append(i)
            elif char == right_delimiter and left_indices:
                pairs.append([left_indices.pop(), i])

    return pairs  # type: ignore


def stems_from_pairs(pairs: list[list[int, int]]) -> list[list[list[int]]]:
    """Extracts _stems from a list of base pairs. A stem is a sequence of continuous base pairs.

    Args:
        pairs: List of base pairs [i, j].

    Returns:
        List of _stems, where each stem is a list of base pairs.

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

            if not matching_pairs:
                break

            stem.append(bp_next)
            pairs.remove(matching_pairs[0])

        # Check inward
        bp_next = bp.copy()
        while bp_next[0] < nres:
            bp_next = [bp_next[0] + 1, bp_next[1] - 1]
            matching_pairs = [p for p in pairs if p[0] == bp_next[0] and p[1] == bp_next[1]]

            if not matching_pairs:
                break

            stem.insert(0, bp_next)
            pairs.remove(matching_pairs[0])

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

    # Extract _stems and create assignment array
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


def get_strand_id(char: str) -> str:
    """Determine strand id based on the delimiter.

    Args:
        char: Delimiter character.

    Returns:
        Strand id. "a" if the first strand, "b" if the second one.
    """

    if char in LEFT_DELIMITERS:
        return "a"
    elif char in RIGHT_DELIMITERS:
        return "b"
    else:
        raise ValueError(f"Unexpected character in structure: {char}")
