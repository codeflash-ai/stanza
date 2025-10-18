from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

_RE_SPLIT = re.compile(r'([()])')

def output_entities(sentence):
    # Slightly faster: use local variable lookups
    words = sentence.words
    for word in words:
        misc = word.misc
        if misc is None:
            continue

        # Only split if present and once per non-None
        for piece in misc.split("|"):
            if piece.startswith("Entity="):
                # Partition is slightly faster than split with maxsplit=1
                entity = piece.partition("=")[2]
                print("  " + entity)
                break

def extract_single_sentence(sentence):
    # Avoid repeated attribute lookups in loops
    words_in = sentence.words
    current_entity = []
    words = []

    # Minor optimization: move pieces and closes to be initialized inside loop only when needed
    for word in words_in:
        text = word.text
        misc = word.misc

        if misc is None:
            pieces = []
        else:
            pieces = misc.split("|")

        closes = []
        first_entity = False
        for piece in pieces:
            if piece.startswith("Entity="):
                # Use partition and precompiled regex
                entity = piece.partition("=")[2]
                entity_pieces = _RE_SPLIT.split(entity)
                # Use filter to avoid creating large intermediate lists if possible
                entity_pieces = list(filter(None, entity_pieces))  # remove blanks from re.split
                entity_idx = 0
                len_ep = len(entity_pieces)
                append_ce = current_entity.append
                while entity_idx < len_ep:
                    p = entity_pieces[entity_idx]
                    if p == '(':
                        if len_ep <= entity_idx + 1:
                            raise AssertionError("Opening an unspecified entity")
                        if not current_entity:
                            first_entity = True
                        append_ce(entity_pieces[entity_idx + 1])
                        entity_idx += 2
                    elif p == ')':
                        if entity_idx == 0:
                            raise AssertionError("Closing an unspecified entity")
                        closes.append(entity_pieces[entity_idx - 1])
                        entity_idx += 1
                    else:
                        entity_idx += 1

        if not current_entity:
            entity = 'O'
        else:
            entity_val = current_entity[0]
            entity = "B-" + entity_val if first_entity else "I-" + entity_val
        words.append((text, entity))

        if len(current_entity) < len(closes):
            raise AssertionError("Too many closes for the current open entities")
        # Elide TODO comment (unchanged logic)
        for close_entity in closes:
            if close_entity != current_entity[-1]:
                raise AssertionError(
                    "Closed the wrong entity: %s vs %s" % (close_entity, current_entity[-1])
                )
            # Avoid creating new lists: just use pop() for efficiency
            current_entity.pop()
    return words

def extract_sentences(doc):
    sentences = []
    sents = doc.sentences
    for sentence in sents:
        try:
            words = extract_single_sentence(sentence)
            sentences.append(words)
        except AssertionError as e:
            print("Skipping sentence %s  ... %s" % (sentence.sent_id, str(e)))
            output_entities(sentence)

    return sentences

def convert_iahlt(udbase, output_dir, short_name):
    shards = ("train", "dev", "test")
    ud_datasets = ["UD_Hebrew-IAHLTwiki", "UD_Hebrew-IAHLTknesset"]
    base_filenames = ["he_iahltwiki-ud-%s.conllu", "he_iahltknesset-ud-%s.conllu"]
    datasets = defaultdict(list)

    for ud_dataset, base_filename in zip(ud_datasets, base_filenames):
        ud_dataset_path = os.path.join(udbase, ud_dataset)
        for shard in shards:
            filename = os.path.join(ud_dataset_path, base_filename % shard)
            doc = CoNLL.conll2doc(filename)
            sentences = extract_sentences(doc)
            print("Read %d sentences from %s" % (len(sentences), filename))
            datasets[shard].extend(sentences)

    datasets = [datasets[x] for x in shards]
    write_dataset(datasets, output_dir, short_name)

def main():
    paths = default_paths.get_default_paths()

    udbase = paths["UDBASE_GIT"]
    output_directory = paths["NER_DATA_DIR"]
    convert_iahlt(udbase, output_directory, "he_iahlt")

if __name__ == '__main__':
    main()
