from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

_ENTITY_SPLIT_RE = re.compile(r'([()])')

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
    append_word = words.append  # local variable for performance
    split_entity = _ENTITY_SPLIT_RE.split  # local variable for performance

    for word in sentence.words:
        text = word.text
        misc = word.misc
        if misc is None:
            pieces = []
        else:
            pieces = misc.split("|")

        closes = []
        first_entity = False
        for piece in pieces:
            # This is a hot path, so use string methods efficiently
            if piece.startswith("Entity="):
                # Avoid repeated split (more efficient: slice string)
                entity = piece[7:]
                # Use precompiled regex, avoids global lookup in the loop
                entity_pieces = split_entity(entity)
                # Avoid list comprehension here: filter in-place for better locality & performance
                filtered_pieces = []
                for x in entity_pieces:
                    if x:
                        filtered_pieces.append(x)
                entity_pieces = filtered_pieces
                entity_idx = 0
                # Avoid repeatedly calling len in the loop
                n_pieces = len(entity_pieces)
                while entity_idx < n_pieces:
                    val = entity_pieces[entity_idx]
                    if val == '(':
                        # Fast path: don't call len each time
                        next_idx = entity_idx + 1
                        assert n_pieces > next_idx, "Opening an unspecified entity"
                        if len(current_entity) == 0:
                            first_entity = True
                        current_entity.append(entity_pieces[next_idx])
                        entity_idx += 2
                    elif val == ')':
                        assert entity_idx != 0, "Closing an unspecified entity"
                        closes.append(entity_pieces[entity_idx-1])
                        entity_idx += 1
                    else:
                        # the entities themselves get added or removed via the ()
                        entity_idx += 1

        if len(current_entity) == 0:
            entity = 'O'
        else:
            entity = current_entity[0]
            entity = "B-" + entity if first_entity else "I-" + entity

        append_word((text, entity))

        # 'closes' is usually small--avoid costly checks unless necessary
        assert len(current_entity) >= len(closes), "Too many closes for the current open entities"
        for close_entity in closes:
            # TODO: check the close is closing the right thing
            assert close_entity == current_entity[-1], "Closed the wrong entity: %s vs %s" % (close_entity, current_entity[-1])
            # Avoid creating a new list each time for popping last item, use .pop() for efficiency
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
