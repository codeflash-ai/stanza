from collections import defaultdict
from functools import lru_cache

class DynamicDepth():
    """
    Implements a cache + dynamic programming to find the relative depth of every word in a subphrase given the head word for every word.
    """
    def get_parse_depths(self, heads, start, end):
        """Return the relative depth for every word

        Args:
            heads (list): List where each entry is the index of that entry's head word in the dependency parse
            start (int): starting index of the heads for the subphrase
            end (int): ending index of the heads for the subphrase

        Returns:
            list: Relative depth in the dependency parse for every word
        """
        self.heads = heads[start:end]
        self.relative_heads = [h - start if h else -100 for h in self.heads] # -100 to deal with 'none' headwords

        depths = [self._get_depth_recursive(h) for h in range(len(self.relative_heads))]

        return depths

    @lru_cache(maxsize=None)
    def _get_depth_recursive(self, index):
        """Recursively get the depths of every index using a cache and recursion

        Args:
            index (int): Index of the word for which to calculate the relative depth

        Returns:
            int: Relative depth of the word at the index
        """
        # if the head for the current index is outside the scope, this index is a relative root
        if self.relative_heads[index] >= len(self.relative_heads) or self.relative_heads[index] < 0:
            return 0
        return self._get_depth_recursive(self.relative_heads[index]) + 1

def find_cconj_head(heads, upos, start, end):
    """
    Finds how far each word is from the head of a span, then uses the closest CCONJ to the head as the new head

    If no CCONJ is present, returns None
    """
    # use head information to extract parse depth
    dynamicDepth = DynamicDepth()
    depth = dynamicDepth.get_parse_depths(heads, start, end)
    depth_limit = 2

    # Combine filtering and collection into single loop for efficiency
    for i in range(end - start):
        if upos[i + start] == 'CCONJ' and depth[i] < depth_limit:
            return i + start
    return None

def process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers, use_cconj_heads=True, lang=None):
    """
    doc_id: a string naming the document
    part_id: if the document has a particular subpart (can be blank)
    sentences: a list of list of string representing the raw text

    coref_spans: a list of lists
    one list per sentence
    each sentence has a list of spans, where each span is (span_index, span_start, span_end)
    the indices are relative to 0 for that particular sentence, and if the span is exactly 1 word long, span_start == span_end

    sentence_speakers: a list of list of string representing who said each word.  can all be blank if there are no known speakers
    """
    sentence_lens = [len(x) for x in sentences]

    # Optimize speaker flattening with list expansion and list comprehension
    if sentence_speakers is None:
        sentence_speakers = [" " for _ in sentences]
    if all(isinstance(x, list) for x in sentence_speakers):
        # This flatten is optimal for list of lists
        speaker = [item for sublist in sentence_speakers for item in sublist]
    else:
        # Replace inner list build with multiplication and list extend for efficiency
        speaker = []
        for sp, slen in zip(sentence_speakers, sentence_lens):
            speaker.extend([sp] * slen)

    cased_words = [y for x in sentences for y in x]

    # More efficient sent_id construction without repeated list creation
    sent_id = []
    for idx, sent_len in enumerate(sentence_lens):
        sent_id.extend([idx] * sent_len)

    doc = pipe(sentences)
    word_total = 0
    heads = []
    deprel = []
    # Single-pass loop for heads and deprel extraction per word, using local vars for minor speed up
    for sentence in doc.sentences:
        sentence_word_count = len(sentence.words)
        for word in sentence.words:
            deprel.append(word.deprel)
            if word.head == 0:
                heads.append("null")
            else:
                heads.append(word.head - 1 + word_total)
        word_total += sentence_word_count

    # Avoid defaultdict overhead by using dict/list directly until final sort
    span_clusters = defaultdict(list)
    word_clusters = defaultdict(list)
    head2span = []
    word_total = 0
    for sent_idx, (parsed_sentence, ontonotes_words) in enumerate(zip(doc.sentences, sentences)):
        # Precompute sentence values outside inner span loop for efficiency
        sentence_words = parsed_sentence.words
        sentence_upos = [x.upos for x in sentence_words]
        sentence_heads = [x.head - 1 if x.head > 0 else None for x in sentence_words]
        for span in coref_spans[sent_idx]:
            # input is expected to be start word, end word + 1
            # counting from 0
            # whereas the OntoNotes coref_span is [start_word, end_word] inclusive
            span_start = span[1] + word_total
            span_end = span[2] + word_total + 1
            candidate_head = find_cconj_head(sentence_heads, sentence_upos, span[1], span[2]+1) if use_cconj_heads else None
            if candidate_head is None:
                # Optimize candidate_head selection loop by using local references and range directly
                for candidate_head in range(span[1], span[2] + 1):
                    word_head = sentence_words[candidate_head].head - 1
                    # treat the head of the phrase as the first word that has a head outside the phrase
                    if word_head < span[1] or word_head > span[2]:
                        break
                else:
                    # if none have a head outside the phrase (circular??)
                    # then just take the first word
                    candidate_head = span[1]
            candidate_head += word_total
            span_clusters[span[0]].append((span_start, span_end))
            word_clusters[span[0]].append(candidate_head)
            head2span.append((candidate_head, span_start, span_end))
        word_total += len(ontonotes_words)
    # Perform sorting after dict->list extraction for memory and speed
    span_clusters = [sorted(span_clusters[k]) for k in sorted(span_clusters)]
    word_clusters = [sorted(word_clusters[k]) for k in sorted(word_clusters)]
    head2span.sort()

    processed = {
        "document_id": doc_id,
        "part_id": part_id,
        "cased_words": cased_words,
        "sent_id": sent_id,
        "speaker": speaker,
        #"pos": pos,
        "deprel": deprel,
        "head": heads,
        "span_clusters": span_clusters,
        "word_clusters": word_clusters,
        "head2span": head2span,
    }
    if part_id is not None:
        processed["part_id"] = part_id
    if lang is not None:
        processed["lang"] = lang
    return processed
