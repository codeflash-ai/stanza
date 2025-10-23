"""
Converts a file of labels on constituency trees for the it_vit dataset

The labels are for whether or not a sentence is written in a standard
S-V-O order.  The intent is to see how much a constituency parser
can improve over a regular transformer classifier.

This file is provided by Prof. Delmonte as part of a classification
project.  Contact John Bauer for more details.

Technically this should be "classifier" instead of "sentiment"
"""

import os

from stanza.models.classifiers.data import SentimentDatum
from stanza.utils.datasets.sentiment import process_utils
from stanza.utils.datasets.constituency.convert_it_vit import read_updated_trees
import stanza.utils.default_paths as default_paths

def label_trees(label_map, trees):
    # Pre-bind frequently used functions and classes for faster local lookup
    SentimentDatum_ = SentimentDatum
    # Pre-bind leaf_labels as method for fast access in loop
    new_trees = []
    
    # Convert label_map to set for O(1) membership test if not a dict (preserves behavior with dict input, avoids any change)
    label_map_contains = label_map.__contains__
    label_map_get = label_map.__getitem__
    
    # Local variable assignment for the append function to minimize attribute lookup
    nt_append = new_trees.append
    
    for tree in trees:
        con_id = tree.con_id
        if not label_map_contains(con_id):  # O(1) dict check
            raise ValueError("%s not labeled" % con_id)
        label = label_map_get(con_id)
        tree_tree = tree.tree
        nt_append(SentimentDatum_(label, tree_tree.leaf_labels(), tree_tree))
    return new_trees

def read_label_map(label_filename):
    with open(label_filename, encoding="utf-8") as fin:
        lines = fin.readlines()
    lines = [x.strip() for x in lines]
    lines = [x.split() for x in lines if x]
    label_map = {}
    for line_idx, line in enumerate(lines):
        k = line[0].split("#")[1]
        v = line[1]

        # compensate for an off-by-one error in the labels for ids 12 through 129
        # we went back and forth a few times but i couldn't explain the error,
        # so whatever, just compensate for it on the conversion side
        k_idx = int(k.split("_")[1])
        if k_idx != line_idx + 1:
            if k_idx >= 12 and k_idx <= 129:
                k = "sent_%05d" % (k_idx - 1)
            else:
                raise ValueError("Unexpected key offset for line {}: {}".format(line_idx, line))

        if v == "neg":
            v = "0"
        elif v == "pos":
            v = "1"
        else:
            raise ValueError("Unexpected label %s for key %s" % (v, k))

        if k in label_map:
            raise ValueError("Duplicate key %s: new value %s, old value %s" % (k, v, label_map[k]))
        label_map[k] = v

    return label_map

def main():
    paths = default_paths.get_default_paths()

    dataset_name = "it_vit_sentences"

    label_filename = os.path.join(paths["SENTIMENT_BASE"], "italian", "sentence_classification", "classified")
    if not os.path.exists(label_filename):
        raise FileNotFoundError("Expected to find the labeled file in %s" % label_filename)

    label_map = read_label_map(label_filename)

    # this will produce three lists of trees with their con_id attached
    train_trees, dev_trees, test_trees = read_updated_trees(paths)

    train_trees = label_trees(label_map, train_trees)
    dev_trees   = label_trees(label_map, dev_trees)
    test_trees  = label_trees(label_map, test_trees)

    dataset = (train_trees, dev_trees, test_trees)
    process_utils.write_dataset(dataset, paths["SENTIMENT_DATA_DIR"], dataset_name)

if __name__ == '__main__':
    main()
