import re

import numpy as np

from viserna.graph import RNAGraph


def extract_stacks(sequence: str, structure: str, data: list | None = None, stack_size: int = 0):
    """Extracts stack motifs from a given RNA secondary structure and sequence. Optionally, extracts associated
    per-nucleotide data values if provided.

    Args:
        sequence: The RNA sequence.
        structure: The RNA secondary structure in dot-bracket notation.
        data: A list of per-nucleotide data values corresponding to the sequence.
        stack_size: The size of the stacks to be extracted.

    Returns:
        - A list of stack motifs if `data` is None. Each motif is represented as a string in the
              format 'SEQ1 SEQ2,... STR1 STR2,...', where SEQ1, SEQ2, ... are sequences of the stack segments
              and STR1, STR2, ... are corresponding structural elements.

        - A dictionary if `data` is provided. Keys are stack motifs in the format 'SEQ1 SEQ2,... STR1 STR2,...',
              and values are lists of associated per-nucleotide data arrays.

    Raises:
        AssertionError: If the length of the sequence and structure do not match.
    """

    assert len(sequence) == len(structure)
    mdl = RNAGraph(structure, sequence=sequence)
    struct = "(" * stack_size + " " + ")" * stack_size

    full_list_motifs = []
    full_list_data = []

    for stem in mdl._stems:
        for stack_ind in range(len(stem) - stack_size + 1):
            stack = stem[stack_ind : stack_ind + stack_size]

            side_1, side_2 = np.array(stack).T
            side_1 = list(reversed(side_1))
            side_2 = list(side_2)

            if data is not None:
                dat_vec = [data[x] for x in side_1] + [-1] + [data[x] for x in side_2]

            seq_vec = "".join([mdl.sequence[x] for x in side_1] + [" "] + [mdl.sequence[x] for x in side_2])
            full_list_motifs.append(f"{seq_vec},{struct}")

            if data is not None:
                full_list_data.append(np.array(dat_vec))

    if not data:
        return full_list_motifs

    returning_dct = {}
    for i, k in enumerate(full_list_motifs):
        if k in returning_dct:
            returning_dct[k].append(full_list_data[i])
        else:
            returning_dct[k] = [full_list_data[i]]
    return returning_dct


def extract_loops(sequence: str, structure: str, data: list | None = None, n_neighbors: int = 0):
    """Extracts loop motifs from a given RNA secondary structure and sequence. Optionally, extracts associated
    per-nucleotide data values if provided.

    Args:
        sequence: The RNA sequence.
        structure: The RNA secondary structure in dot-bracket notation.
        data: A list of per-nucleotide data values corresponding to the sequence.
        n_neighbors: Number of neighboring base pairs to include around each loop motif.

    Returns:
        - A list of loop motifs if `data` is None. Each motif is represented as a string in the
              format 'SEQ1 SEQ2,... STR1 STR2,...', where SEQ1, SEQ2, ... are sequences of the loop segments
              and STR1, STR2, ... are corresponding structural elements.

        - A dictionary if `data` is provided. Keys are loop motifs in the format 'SEQ1 SEQ2,... STR1 STR2,...',
              and values are lists of associated per-nucleotide data arrays.

    Raises:
        AssertionError: If the length of the sequence and structure do not match.
    """

    assert len(sequence) == len(structure)
    rg = RNAGraph(structure, sequence=sequence)

    nodes = [n for n in list(rg.graph.nodes) if not isinstance(n, str)]
    string_assignment = "".join(["%d" % int(x) for x in rg._stem_assignment])
    full_list_motifs = []
    full_list_data = []

    for i in range(1, max(nodes) + 1):
        loop_motif = []
        loop_seq = []
        loop_data = []

        children = list(rg.graph[i])
        lst = [i] + children + [i]

        for j in range(len(lst) - 1):
            start_stretch, end_stretch = lst[j], lst[j + 1]
            if start_stretch == end_stretch:
                obj = re.search(r"%d00*%d" % (start_stretch, end_stretch), string_assignment)  # in a hairpin
            else:
                obj = re.search(r"%d0*%d" % (start_stretch, end_stretch), string_assignment)

            start_ind, end_ind = obj.start(), obj.end()

            loop_motif.append(rg.structure[(start_ind - n_neighbors + 1): (end_ind + n_neighbors - 1)])
            loop_seq.append(rg.sequence[(start_ind - n_neighbors + 1): (end_ind + n_neighbors - 1)])

            if data is not None:
                loop_data.extend(data[(start_ind - n_neighbors + 1): (end_ind + n_neighbors - 1)])
                loop_data.extend([-1])

        motif = " ".join(loop_seq) + "," + " ".join(loop_motif)
        if data is not None:
            del loop_data[-1]

            assert len(" ".join(loop_seq)) == len(loop_data)
        assert len(" ".join(loop_seq)) == len(" ".join(loop_motif))

        full_list_motifs.append(motif)

        if data is not None:
            full_list_data.append(np.array(loop_data))

    if not data:
        return full_list_motifs

    returning_dct = {}
    for i, k in enumerate(full_list_motifs):
        if k not in returning_dct:
            returning_dct[k] = [full_list_data[i]]
            continue
        returning_dct[k].append(full_list_data[i])

    return returning_dct
