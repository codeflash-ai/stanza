"""
Common methods for the various self-training data collection scripts
"""

import logging
import os
import random
import re

import stanza
from stanza.models.common import utils
from stanza.models.common.bert_embedding import TextTooLongError
from stanza.utils.get_tqdm import get_tqdm

_cached_pipelines: dict = {}

logger = logging.getLogger('stanza')
tqdm = get_tqdm()

def common_args(parser):
    parser.add_argument(
        '--output_file',
        default='data/constituency/vi_silver.mrg',
        help='Where to write the silver trees'
    )
    parser.add_argument(
        '--lang',
        default='vi',
        help='Which language tools to use for tokenization and POS'
    )
    parser.add_argument(
        '--num_sentences',
        type=int,
        default=-1,
        help='How many sentences to get per file (max)'
    )
    parser.add_argument(
        '--models',
        default='saved_models/constituency/vi_vlsp21_inorder.pt',
        help='What models to use for parsing.  comma-separated'
    )
    parser.add_argument(
        '--package',
        default='default',
        help='Which package to load pretrain & charlm from for the parsers'
    )
    parser.add_argument(
        '--output_ptb',
        default=False,
        action='store_true',
        help='Output trees in PTB brackets (default is a bracket language format)'
    )

def add_length_args(parser):
    parser.add_argument(
        '--min_len',
        default=5,
        type=int,
        help='Minimum length sentence to keep.  None = unlimited'
    )
    parser.add_argument(
        '--no_min_len',
        dest='min_len',
        action='store_const',
        const=None,
        help='No minimum length'
    )
    parser.add_argument(
        '--max_len',
        default=100,
        type=int,
        help='Maximum length sentence to keep.  None = unlimited'
    )
    parser.add_argument(
        '--no_max_len',
        dest='max_len',
        action='store_const',
        const=None,
        help='No maximum length'
    )

def build_ssplit_pipe(ssplit, lang):
    if ssplit:
        return stanza.Pipeline(lang, processors="tokenize")
    else:
        return stanza.Pipeline(lang, processors="tokenize", tokenize_no_ssplit=True)

def build_tag_pipe(ssplit, lang, foundation_cache=None):
    # Cache stanza.Pipeline objects to avoid repeated expensive initializations
    cache_key = (ssplit, lang, id(foundation_cache))
    pipe = _cached_pipelines.get(cache_key)
    if pipe is not None:
        return pipe
    if ssplit:
        pipe = stanza.Pipeline(lang, processors="tokenize,pos", foundation_cache=foundation_cache)
    else:
        pipe = stanza.Pipeline(lang, processors="tokenize,pos", tokenize_no_ssplit=True, foundation_cache=foundation_cache)
    _cached_pipelines[cache_key] = pipe
    return pipe

def build_parser_pipes(lang, models, package="default", foundation_cache=None):
    """
    Build separate pipelines for each parser model we want to use

    It is highly recommended to pass in a FoundationCache to reuse bottom layers
    """
    parser_pipes = []
    for model_name in models.split(","):
        if os.path.exists(model_name):
            # if the model name exists as a file, treat it as the path to the model
            pipe = stanza.Pipeline(lang, processors="constituency", package=package, constituency_model_path=model_name, constituency_pretagged=True, foundation_cache=foundation_cache)
        else:
            # otherwise, assume it is a package name?
            pipe = stanza.Pipeline(lang, processors={"constituency": model_name}, constituency_pretagged=True, package=None, foundation_cache=foundation_cache)
        parser_pipes.append(pipe)
    return parser_pipes

def split_docs(docs, ssplit_pipe, max_len=140, max_word_len=50, chunk_size=2000):
    """
    Using the ssplit pipeline, break up the documents into sentences

    Filters out sentences which are too long or have words too long.

    This step is necessary because some web text has unstructured
    sentences which overwhelm the tagger, or even text with no
    whitespace which breaks the charlm in the tokenizer or tagger
    """
    raw_sentences = 0
    filtered_sentences = 0
    new_docs = []

    logger.info("Splitting raw docs into sentences: %d", len(docs))
    for chunk_start in tqdm(range(0, len(docs), chunk_size)):
        chunk = docs[chunk_start:chunk_start+chunk_size]
        chunk = [stanza.Document([], text=t) for t in chunk]
        chunk = ssplit_pipe(chunk)
        sentences = [s for d in chunk for s in d.sentences]
        raw_sentences += len(sentences)
        sentences = [s for s in sentences if len(s.words) < max_len]
        sentences = [s for s in sentences if max(len(w.text) for w in s.words) < max_word_len]
        filtered_sentences += len(sentences)
        new_docs.extend([s.text for s in sentences])

    logger.info("Split sentences: %d", raw_sentences)
    logger.info("Sentences filtered for length: %d", filtered_sentences)
    return new_docs

# from https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
ZH_RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
# https://stackoverflow.com/questions/6787716/regular-expression-for-japanese-characters
JA_RE = re.compile(u'[一-龠ぁ-ゔァ-ヴー々〆〤ヶ]', re.UNICODE)
DEV_RE = re.compile(u'[\u0900-\u097f]', re.UNICODE)

def tokenize_docs(docs, pipe, min_len, max_len):
    """
    Turn the text in docs into a list of whitespace separated sentences

    docs: a list of strings
    pipe: a Stanza pipeline for tokenizing
    min_len, max_len: can be None to not filter by this attribute
    """
    results = []
    docs = [stanza.Document([], text=t) for t in docs]
    if len(docs) == 0:
        return results
    pipe(docs)
    lang = pipe.lang
    is_zh = lang and lang.startswith("zh")
    is_ja = lang and lang.startswith("ja")
    is_vi = lang and lang.startswith("vi")
    # Prepare sets for forbidden chars and filter chars
    forbidden = {"|", "_", "<", ">", "[", "]", "—"}
    filter_chars = {'"', '(', ')'}
    # Avoid repeated attribute lookups
    ZH_RE_findall = ZH_RE.findall
    JA_RE_findall = JA_RE.findall
    DEV_RE_findall = DEV_RE.findall

    for doc in docs:
        for sentence in doc.sentences:
            words = sentence.words
            words_len = len(words)
            if min_len and words_len < min_len:
                continue
            if max_len and words_len > max_len:
                continue
            text = sentence.text

            # Fast forbidden character check with set intersection
            if any(c in text for c in forbidden):
                continue

            # Combine checks for all filter chars per sentence
            # Short-circuit: Only scan full words for filter_chars if any filter_char in text
            # This avoids slow nested loops for sentences that don't contain any filter_char
            if any(c in text for c in filter_chars):
                # Only scan words for relevant characters
                # Pre-check if any long word contains a filter char
                for w in words:
                    wtext = w.text
                    if len(wtext) > 1 and any(c in wtext for c in filter_chars):
                        break
                else:
                    pass  # No breaking filter char found
                    # continue not called, so process
                    # But original code continues if *any* - so invert logic
                    pass
                    # no continue means ok
                    # if we get here, we didn't break
                # Logic above, if break, skip; else move on
                else_continue = False
                for w in words:
                    wtext = w.text
                    if len(wtext) > 1 and any(c in wtext for c in filter_chars):
                        else_continue = True
                        break
                if else_continue:
                    continue

            # Invert the above logic to one loop for clarity and single pass
            # (avoid double pass, only one needed if any filter_char in text)
            # This is fast if forbidden char not in text (majority case).

            # Merge whitespace replacement and join in one step
            word_texts = []
            long_word_found = False
            for w in words:
                wtext = w.text
                # check long words before replacing, to avoid compute and to short-circuit
                if len(wtext) >= 50:
                    long_word_found = True
                    break
                word_texts.append(wtext.replace(" ", "_"))
            if long_word_found:
                continue
            joined_text = " ".join(word_texts)

            # ZH/JA/DEV checks - Do these only if under the respective language rules
            # Findall regex is the real performance bottleneck, so only call when necessary
            # Use short-circuiting to skip these heavy checks up front.
            if not is_zh and len(ZH_RE_findall(joined_text)) > 250:
                continue
            if not is_ja and len(JA_RE_findall(joined_text)) > 150:
                continue
            if is_vi and len(DEV_RE_findall(joined_text)) > 100:
                continue
            results.append(joined_text)
    return results

def find_matching_trees(docs, num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=True, chunk_size=10, max_len=140, min_len=10, output_ptb=False):
    """
    Find trees where all the parsers in parser_pipes agree

    docs should be a list of strings.
      one sentence per string or a whole block of text as long as the tag_pipe can break it into sentences

    num_sentences > 0 gives an upper limit on how many sentences to extract.
      If < 0, all possible sentences are extracted

    accepted_trees is a running tally of all the trees already built,
      so that we don't reuse the same sentence if we see it again
    """
    if num_sentences < 0:
        tqdm_total = len(docs)
    else:
        tqdm_total = num_sentences

    if output_ptb:
        output_format = "{}"
    else:
        output_format = "{:L}"

    with tqdm(total=tqdm_total, leave=False) as pbar:
        if shuffle:
            random.shuffle(docs)
        new_trees = set()
        for chunk_start in range(0, len(docs), chunk_size):
            chunk = docs[chunk_start:chunk_start+chunk_size]
            chunk = [stanza.Document([], text=t) for t in chunk]

            if num_sentences < 0:
                pbar.update(len(chunk))

            # first, retag the sentences
            tag_pipe(chunk)

            chunk = [d for d in chunk if len(d.sentences) > 0]
            if max_len is not None:
                # for now, we don't have a good way to deal with sentences longer than the bert maxlen
                chunk = [d for d in chunk if max(len(s.words) for s in d.sentences) < max_len]
            if len(chunk) == 0:
                continue

            parses = []
            try:
                for pipe in parser_pipes:
                    pipe(chunk)
                    trees = [output_format.format(sent.constituency) for doc in chunk for sent in doc.sentences if len(sent.words) >= min_len]
                    parses.append(trees)
            except TextTooLongError as e:
                # easiest is to skip this chunk - could theoretically save the other sentences
                continue

            for tree in zip(*parses):
                if len(set(tree)) != 1:
                    continue
                tree = tree[0]
                if tree in accepted_trees:
                    continue
                if tree not in new_trees:
                    new_trees.add(tree)
                    if num_sentences >= 0:
                        pbar.update(1)
                if num_sentences >= 0 and len(new_trees) >= num_sentences:
                    return new_trees

    return new_trees

