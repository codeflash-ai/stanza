"""Convert the coref annotation of IAHLT to the Stanza coref format

This dataset is available at

https://github.com/IAHLT/coref

Download it via git clone to $COREF_BASE/hebrew, so for example on the cluster:

cd /u/nlp/data/coref/
mkdir hebrew
cd hebrew
git clone git@github.com:IAHLT/coref.git

Then run

python3 stanza/utils/datasets/coref/convert_hebrew_iahlt.py

The scores for models built from the dataset are pretty lousy in
general, but seem to be in line with the scores obtained by other
people working on this data.  For example, the authors said they had a
52 F1, whereas if we use roberta-xlm, we get 50.
"""

import argparse
from collections import defaultdict, namedtuple
import json
import os

import stanza

from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import process_document

tqdm = get_tqdm()

CorefDoc = namedtuple("CorefDoc", ['doc_id', 'sentences', 'coref_spans'])

# TODO: binary search for speed?
def search_mention_start(doc, mention_start):
    # Optimize by pre-fetching list references and use indexing to avoid repetitive attribute access
    sentences = doc.sentences
    num_sentences = len(sentences)

    # Binary search for sentence containing start
    lo, hi = 0, num_sentences - 1
    sent_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if mention_start < sentences[mid].tokens[-1].end_char:
            hi = mid - 1
            sent_idx = mid
        else:
            lo = mid + 1
    if sent_idx is None:
        raise ValueError

    sentence = sentences[sent_idx]
    words = sentence.words
    num_words = len(words)

    # Binary search for word containing start
    lo, hi = 0, num_words - 1
    word_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        word = words[mid]
        if word.end_char is None:
            print("Found weirdness on sentence:\n|%s|" % sentence.text)
            print(word.parent)
            return None, None
        if mention_start < word.end_char:
            word_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if word_idx is None:
        raise ValueError
    return sent_idx, word_idx

def search_mention_end(doc, mention_end):
    sentences = doc.sentences
    num_sentences = len(sentences)

    # Binary search for the sentence where mention_end is before the next sentence's first token
    lo, hi = 0, num_sentences - 1
    sent_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid + 1 == num_sentences or mention_end < sentences[mid+1].tokens[0].start_char:
            sent_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # For robustness, fall back to sequential if not found (should not happen).
    if sent_idx is None:
        sent_idx = num_sentences - 1
    sentence = sentences[sent_idx]
    words = sentence.words
    num_words = len(words)

    # Binary search for the word where mention_end is before the next word's start_char
    lo, hi = 0, num_words - 1
    word_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid + 1 == num_words or mention_end < words[mid+1].start_char:
            word_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if word_idx is None:
        word_idx = num_words - 1
    return sent_idx, word_idx

def extract_doc(tokenizer, lines):
    # 16, 1, 5 for the train, dev, test sets
    broken = 0
    tok_error = 0
    singletons = 0
    one_words = 0
    processed_docs = []

    # Reduce method & attribute lookup in loops where possible
    for line_idx, line in enumerate(tqdm(lines)):
        all_clusters = defaultdict(list)
        doc_id = line['metadata']['doc_id']
        text = line['text']
        clusters = line['clusters']
        doc = tokenizer(text)
        doc_sentences = doc.sentences # minor local reference speedup
        for cluster_idx, cluster in enumerate(clusters):
            found_mentions = []
            mentions = cluster['mentions']
            for mention_idx, mention in enumerate(mentions):
                mention_start = mention[0]
                mention_end = mention[1]
                start_sent, start_word = search_mention_start(doc, mention_start)
                if start_sent is None or start_word is None:
                    tok_error += 1
                    continue
                end_sent, end_word = search_mention_end(doc, mention_end)
                assert end_sent >= start_sent
                if start_sent != end_sent:
                    broken += 1
                    continue

                assert end_word >= start_word
                if end_word == start_word:
                    one_words += 1
                found_mentions.append((start_sent, start_word, end_word))
            if len(found_mentions) == 0:
                continue
            elif len(found_mentions) == 1:
                # the number of singletons, after discarding mentions that
                # crossed a sentence boundary according to Stanza, is
                # 5, 0, 1
                # so clearly the dataset does not intentionally have
                # (many?) singletons in it
                singletons += 1
                continue
            else:
                all_clusters[cluster_idx] = found_mentions
        # maybe we need to update the interface - there can be MWT in Hebrew
        sentences = [[word.text for word in sent.words] for sent in doc_sentences]
        coref_spans = defaultdict(list)
        for cluster_idx, mention_list in all_clusters.items():
            for sent_idx, start_word, end_word in mention_list:
                coref_spans[sent_idx].append((cluster_idx, start_word, end_word))
        processed_docs.append(CorefDoc(doc_id, sentences, coref_spans))
    print("Found %d broken across two sentences, %d tok errors, %d singleton mentions, %d one_word mentions" % (broken, tok_error, singletons, one_words))
    return processed_docs

def read_doc(tokenizer, filename):
    with open(filename, encoding="utf-8") as fin:
        lines = fin.readlines()
    lines = [json.loads(line) for line in lines]
    return extract_doc(tokenizer, lines)

def write_json_file(output_filename, dataset):
    with open(output_filename, "w", encoding="utf-8") as fout:
        json.dump(dataset, fout, indent=2, ensure_ascii=False)

def main(args=None):
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        prog='Convert Hebrew IAHLT data',
    )
    parser.add_argument('--output_directory', default=None, type=str, help='Where to output the data (defaults to %s)' % paths['COREF_DATA_DIR'])
    args = parser.parse_args(args=args)
    coref_output_path = args.output_directory if args.output_directory else paths['COREF_DATA_DIR']
    print("Will write IAHLT dataset to %s" % coref_output_path)

    coref_input_path = paths["COREF_BASE"]
    hebrew_base_path = os.path.join(coref_input_path, "hebrew", "coref", "train_val_test")

    tokenizer = stanza.Pipeline("he", processors="tokenize", package="default_accurate")
    pipe = stanza.Pipeline("he", processors="tokenize,pos,lemma,depparse", package="default_accurate", tokenize_pretokenized=True)

    input_files = ["coref-5-heb_train.jsonl", "coref-5-heb_val.jsonl", "coref-5-heb_test.jsonl"]
    output_files = ["he_iahlt.train.json", "he_iahlt.dev.json", "he_iahlt.test.json"]
    for input_filename, output_filename in zip(input_files, output_files):
        input_filename = os.path.join(hebrew_base_path, input_filename)
        assert os.path.exists(input_filename)
        docs = read_doc(tokenizer, input_filename)
        dataset = [process_document(pipe, doc.doc_id, "", doc.sentences, doc.coref_spans, None, lang="he") for doc in tqdm(docs)]

        output_filename = os.path.join(coref_output_path, output_filename)
        write_json_file(output_filename, dataset)

    return output_files

if __name__ == '__main__':
    main()
