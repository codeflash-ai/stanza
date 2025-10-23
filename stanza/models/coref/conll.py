""" Contains functions to produce conll-formatted output files with
predicted spans and their clustering """

from collections import defaultdict
from contextlib import contextmanager
import os
from typing import List, TextIO

from stanza.models.coref.config import Config
from stanza.models.coref.const import Doc, Span


# pylint: disable=too-many-locals
def write_conll(doc: Doc,
                clusters: List[List[Span]],
                heads: List[int],
                f_obj: TextIO):
    """ Writes span/cluster information to f_obj, which is assumed to be a file
    object open for writing """
    placeholder = list("\t_" * 7)
    # the nth token needs to be a number
    placeholder[9] = "0"
    placeholder = "".join(placeholder)

    doc_id = doc["document_id"].replace("-", "_").replace("/", "_").replace(".", "_")
    words = doc["cased_words"]
    part_id = doc["part_id"]
    sents = doc["sent_id"]

    # If unused, consider removing; left for behavioral parity
    max_word_len = max(len(w) for w in words)

    # Use list as default directly to avoid repeated lambda: []
    starts = defaultdict(list)
    ends = defaultdict(list)
    single_word = defaultdict(list)

    clusters_len = len(clusters)  # avoid multiple global lookups

    # Bulk populate span lookup tables
    for cluster_id, cluster in enumerate(clusters):
        h = heads[cluster_id]
        if len(h) != len(cluster):
            continue
        for cluster_part, (start, end) in enumerate(cluster):
            if end - start == 1:
                single_word[start].append((cluster_part, cluster_id))
            else:
                starts[start].append((cluster_part, cluster_id))
                ends[end - 1].append((cluster_part, cluster_id))

    f_obj.write(f"# newdoc id = {doc_id}\n# global.Entity = eid-head\n")

    # Move helpers outside loop for speed
    # Pre-compile str.split and int for tight loop
    _split_dash = str.split
    _int = int

    def compare_sort(x):
        split = _split_dash(x, "-")
        if len(split) > 1: 
            return _int(split[-1].replace(")", "").strip())  
        else: 
            # we want everything that's a closer to be first
            return float("inf")

    # Pre-store clusters/heads as locals for tight loop
    clusters_local = clusters
    heads_local = heads

    word_number = 0
    sent_id = 0

    # Hoist all .append and inner accesses
    starts_get = starts.get
    single_word_get = single_word.get
    ends_get = ends.get

    words_len = len(words)
    sents_local = sents

    write = f_obj.write

    for word_id in range(words_len):
        word = words[word_id]

        cluster_info_lst = []

        for part, cluster_marker in starts_get(word_id, ()):
            # local lookups, reduce attribute access
            s, e = clusters_local[cluster_marker][part]
            cluster_info_lst.append(f"(e{cluster_marker}-{min(heads_local[cluster_marker][part], e-s)}")
        for part, cluster_marker in single_word_get(word_id, ()):
            s, e = clusters_local[cluster_marker][part]
            cluster_info_lst.append(f"(e{cluster_marker}-{min(heads_local[cluster_marker][part], e-s)})")
        for part, cluster_marker in ends_get(word_id, ()):
            cluster_info_lst.append(f"e{cluster_marker})")

        # Only sort if >1
        if len(cluster_info_lst) > 1:
            cluster_info_lst.sort(key=compare_sort, reverse=True)
        cluster_info = "".join(cluster_info_lst) if cluster_info_lst else "_"

        if word_id == 0 or sents_local[word_id] != sents_local[word_id - 1]:
            write(f"# sent_id = {doc_id}-{sent_id}\n")
            word_number = 0
            sent_id += 1

        if cluster_info != "_":
            cluster_info = f"Entity={cluster_info}"

        write(f"{word_id}\t{word}{placeholder}\t{cluster_info}\n")
        word_number += 1

    write("\n")


@contextmanager
def open_(config: Config, epochs: int, data_split: str):
    """ Opens conll log files for writing in a safe way. """
    base_filename = f"{config.section}_{data_split}_e{epochs}"
    conll_dir = config.conll_log_dir
    kwargs = {"mode": "w", "encoding": "utf8"}

    os.makedirs(conll_dir, exist_ok=True)

    with open(os.path.join(  # type: ignore
            conll_dir, f"{base_filename}.gold.conll"), **kwargs) as gold_f:
        with open(os.path.join(  # type: ignore
                conll_dir, f"{base_filename}.pred.conll"), **kwargs) as pred_f:
            yield (gold_f, pred_f)
