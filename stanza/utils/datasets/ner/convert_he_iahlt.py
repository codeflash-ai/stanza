from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

_ENTITY_SPLIT_PATTERN = re.compile(r"([()])")

def output_entities(sentence):
    for word in sentence.words:
        misc = word.misc
        if misc is None:
            continue

        pieces = misc.split("|")
        for piece in pieces:
            if piece.startswith("Entity="):
                entity = piece.split("=", maxsplit=1)[1]
                print("  " + entity)
                break

def extract_single_sentence(sentence):
    current_entity = []
    words = []
    # Minor micro-optimization: bind local variables for methods
    append_word = words.append
    split_entity = _ENTITY_SPLIT_PATTERN.split

    for word in sentence.words:
        text = word.text
        misc = word.misc

        # Avoid unnecessary list allocation if misc is None
        if misc is None:
            pieces = ()
        else:
            pieces = misc.split("|")

        closes = []
        first_entity = False
        for piece in pieces:
            if piece.startswith("Entity="):
                # Split only once for max performance
                entity = piece[7:] if piece.startswith("Entity=") else piece.split("=", maxsplit=1)[1]
                # Use pre-compiled regex, replacing re.split with pre-bound split_entity
                entity_pieces = split_entity(entity)
                # Remove blanks from re.split, can use list comprehension (no change)
                entity_pieces = [x for x in entity_pieces if x]

                # Use index iteration as before, but avoid function call overhead in len() per loop
                entity_pieces_len = len(entity_pieces)
                entity_idx = 0
                while entity_idx < entity_pieces_len:
                    piece_val = entity_pieces[entity_idx]
                    if piece_val == '(':
                        # assert unchanged
                        assert entity_pieces_len > entity_idx + 1, "Opening an unspecified entity"
                        if not current_entity:
                            first_entity = True
                        current_entity.append(entity_pieces[entity_idx + 1])
                        entity_idx += 2
                    elif piece_val == ')':
                        # assert unchanged
                        assert entity_idx != 0, "Closing an unspecified entity"
                        closes.append(entity_pieces[entity_idx - 1])
                        entity_idx += 1
                    else:
                        entity_idx += 1

        if not current_entity:
            entity = 'O'
        else:
            entity_val = current_entity[0]
            entity = "B-" + entity_val if first_entity else "I-" + entity_val
        append_word((text, entity))

        assert len(current_entity) >= len(closes), "Too many closes for the current open entities"
        # Avoid building an intermediate list for closes if empty
        if closes:
            # Use reversed order to match the original LIFO behavior, but semantics is unchanged here as it iterates the closes as produced.
            for close_entity in closes:
                # assert unchanged
                assert close_entity == current_entity[-1], (
                    "Closed the wrong entity: %s vs %s" % (close_entity, current_entity[-1])
                )
                # Use pop() instead of slicing for much faster removal from end of list
                current_entity.pop()
    return words

def extract_sentences(doc):
    sentences = []
    for sentence in doc.sentences:
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
