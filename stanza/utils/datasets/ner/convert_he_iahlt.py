from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

_RE_ENTITY_SPLIT = re.compile(r"([()])")

def output_entities(sentence):
    # Build a single print call for all entities in the sentence for faster output
    entities = []
    for word in sentence.words:
        misc = word.misc
        if misc is None:
            continue
        # Avoid repeated split if there's only one piece
        if "Entity=" not in misc:
            continue
        for piece in misc.split("|"):
            if piece.startswith("Entity="):
                # Use partition for more efficient single split
                _, _, entity = piece.partition("=")
                entities.append("  " + entity)
                break
    if entities:
        print("\n".join(entities))

def extract_single_sentence(sentence):
    current_entity = []
    words = []
    append_word = words.append  # Localize method for speed

    for word in sentence.words:
        text = word.text
        misc = word.misc
        pieces = misc.split("|") if misc else []

        closes = []
        first_entity = False

        # Fast path for common case of no entities
        for piece in pieces:
            # Use the faster in-string check to avoid startswith unless needed
            if not piece or piece[0] != 'E' or not piece.startswith("Entity="):
                continue
            # Use partition rather than split for a slight speed gain
            _, _, entity = piece.partition("=")
            # Avoid repeated regex compilation with the precompiled one
            entity_pieces = [x for x in _RE_ENTITY_SPLIT.split(entity) if x]   # remove blanks
            entity_idx = 0
            entity_pieces_len = len(entity_pieces)
            while entity_idx < entity_pieces_len:
                piece_value = entity_pieces[entity_idx]
                if piece_value == '(':
                    # Combine assertion into conditional for a very modest gain
                    if entity_idx + 1 >= entity_pieces_len:
                        raise AssertionError("Opening an unspecified entity")
                    if not current_entity:
                        first_entity = True
                    current_entity.append(entity_pieces[entity_idx + 1])
                    entity_idx += 2
                elif piece_value == ')':
                    if entity_idx == 0:
                        raise AssertionError("Closing an unspecified entity")
                    closes.append(entity_pieces[entity_idx - 1])
                    entity_idx += 1
                else:
                    entity_idx += 1

        # Avoid unnecessary list operation if current_entity is empty
        if not current_entity:
            entity_tag = 'O'
        else:
            entity_tag = current_entity[0]
            entity_tag = "B-" + entity_tag if first_entity else "I-" + entity_tag
        append_word((text, entity_tag))

        if len(current_entity) < len(closes):
            raise AssertionError("Too many closes for the current open entities")
        for close_entity in closes:
            last = current_entity[-1]
            if close_entity != last:
                raise AssertionError("Closed the wrong entity: %s vs %s" % (close_entity, last))
            current_entity.pop()
    return words

def extract_sentences(doc):
    sentences = []
    append_sentence = sentences.append  # Localize append for small speedup
    for sentence in doc.sentences:
        try:
            words = extract_single_sentence(sentence)
            append_sentence(words)
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
