import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

logger = logging.getLogger('stanza')

BERT_ARGS = {
    "vinai/phobert-base": { "use_fast": True },
    "vinai/phobert-large": { "use_fast": True },
}

class TextTooLongError(ValueError):
    """
    A text was too long for the underlying model (possibly BERT)
    """
    def __init__(self, length, max_len, line_num, text):
        super().__init__("Found a text of length %d (possibly after tokenizing).  Maximum handled length is %d  Error occurred at line %d" % (length, max_len, line_num))
        self.line_num = line_num
        self.text = text


def update_max_length(model_name, tokenizer):
    if model_name in ('hf-internal-testing/tiny-bert',
                      'google/muril-base-cased',
                      'google/muril-large-cased',
                      'airesearch/wangchanberta-base-att-spm-uncased',
                      'camembert/camembert-large',
                      'hfl/chinese-electra-180g-large-discriminator',
                      'hfl/chinese-macbert-large',
                      'NYTK/electra-small-discriminator-hungarian'):
        tokenizer.model_max_length = 512

def load_tokenizer(model_name, tokenizer_kwargs=None, local_files_only=False):
    if model_name:
        # note that use_fast is the default
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers library for BERT support! Try `pip install transformers`.")
        bert_args = BERT_ARGS.get(model_name, dict())
        if not model_name.startswith("vinai/phobert"):
            bert_args["add_prefix_space"] = True
        if tokenizer_kwargs:
            bert_args.update(tokenizer_kwargs)
        bert_args['local_files_only'] = local_files_only
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **bert_args)
        update_max_length(model_name, bert_tokenizer)
        if model_name == 'princeton-nlp/Sheared-LLaMA-1.3B':
            bert_tokenizer.pad_token = bert_tokenizer.eos_token
            logger.debug("Tokenizer does not have a pad_token - setting to %s (%s)", bert_tokenizer.pad_token, bert_tokenizer.eos_token)
        return bert_tokenizer
    return None

def load_bert(model_name, tokenizer_kwargs=None, local_files_only=False):
    if model_name:
        # such as: "vinai/phobert-base"
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install transformers library for BERT support! Try `pip install transformers`.")
        bert_model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        bert_tokenizer = load_tokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs, local_files_only=local_files_only)
        return bert_model, bert_tokenizer
    return None, None

def tokenize_manual(model_name, sent, tokenizer):
    """
    Tokenize a sentence manually, using for checking long sentences and PHOBert.
    """
    #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
    tokenized = [word.replace("\xa0","_").replace(" ", "_") for word in sent] if model_name.startswith("vinai/phobert") else [word.replace("\xa0"," ") for word in sent]

    #concatenate to a sentence
    sentence = ' '.join(tokenized)

    #tokenize using AutoTokenizer PhoBERT
    tokenized = tokenizer.tokenize(sentence)

    #convert tokens to ids
    sent_ids = tokenizer.convert_tokens_to_ids(tokenized)

    #add start and end tokens to sent_ids
    tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]

    return tokenized, tokenized_sent

def filter_data(model_name, data, tokenizer = None, log_level=logging.DEBUG):
    """
    Filter out the (NER, POS) data that is too long for BERT model.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name) 
    filtered_data = []
    #eliminate all the sentences that are too long for bert model
    for sent in data:
        sentence = [word if isinstance(word, str) else word[0] for word in sent]
        _, tokenized_sent = tokenize_manual(model_name, sentence, tokenizer)
        
        if len(tokenized_sent) > tokenizer.model_max_length - 2:
            continue

        filtered_data.append(sent)

    logger.log(log_level, "Eliminated %d of %d datapoints because their length is over maximum size of BERT model.", (len(data)-len(filtered_data)), len(data))
    
    return filtered_data

def needs_length_filter(model_name):
    """
    TODO: we were lazy and didn't implement any form of length fudging for models other than bert/roberta/electra
    """
    if 'bart' in model_name or 'xlnet' in model_name:
        return True
    if model_name.startswith("vinai/phobert"):
        return True
    return False

def cloned_feature(feature, num_layers, detach=True):
    """
    Clone & detach the feature, keeping the last N layers (or averaging -2,-3,-4 if not specified)

    averaging 3 of the last 4 layers worked well for non-VI languages
    """
    # in most cases, need to call with features.hidden_states
    # bartpho is different - it has features.decoder_hidden_states
    # feature[2] is the same for bert, but it didn't work for
    # older versions of transformers for xlnet
    if num_layers is None:
        feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
    else:
        feature = torch.stack(feature[-num_layers:], axis=3)
    if detach:
        return feature.clone().detach()
    else:
        return feature

def extract_bart_word_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    """
    Handles vi-bart.  May need testing before using on other bart

    https://github.com/VinAIResearch/BARTpho
    """
    processed = [] # final product, returns the list of list of word representation

    sentences = [" ".join([word.replace(" ", "_") for word in sentence]) for sentence in data]
    tokenized = tokenizer(sentences, return_tensors='pt', padding=True, return_attention_mask=True)
    input_ids_all = tokenized['input_ids'].to(device)
    attention_mask_all = tokenized['attention_mask'].to(device)

    batch_size = 128
    num_batches = math.ceil(len(sentences) / batch_size)
    for i in range(num_batches):
        start_sentence = i * batch_size
        end_sentence = min(start_sentence + batch_size, len(sentences))
        input_ids = input_ids_all[start_sentence:end_sentence]
        attention_mask = attention_mask_all[start_sentence:end_sentence]

        if detach:
            with torch.no_grad():
                features = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                feats = cloned_feature(features.decoder_hidden_states, num_layers, detach)
        else:
            features = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            feats = cloned_feature(features.decoder_hidden_states, num_layers, detach)

        # Use torch.split to avoid iteration through features, improving slicing efficiency
        for feature, sentence in zip(feats, data[start_sentence:end_sentence]):
            # +2 for the endpoints
            sliced = feature[:len(sentence)+2]
            if not keep_endpoints:
                sliced = sliced[1:-1]
            processed.append(sliced)

    return processed

def extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    """
    Extract transformer embeddings using a method specifically for phobert

    Since phobert doesn't have the is_split_into_words / tokenized.word_ids(batch_index=0)
    capability, we instead look for @@ to denote a continued token.
    data: list of list of string (the text tokens)
    """
    processed = [] # final product, returns the list of list of word representation
    tokenized_sents = [] # list of sentences, each is a torch tensor with start and end token
    list_tokenized = [] # list of tokenized sentences from phobert

    # Precompute all tokenizations and tensors, eliminating unnecessary list-building inside batch
    for idx, sent in enumerate(data):
        tokenized, tokenized_sent = tokenize_manual(model_name, sent, tokenizer)
        list_tokenized.append(tokenized)
        if len(tokenized_sent) > tokenizer.model_max_length:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(tokenized_sent), data[idx])
            raise TextTooLongError(len(tokenized_sent), tokenizer.model_max_length, idx, " ".join(data[idx]))
        tokenized_sents.append(torch.tensor(tokenized_sent).detach())
        processed.append([])

    size = len(tokenized_sents)
    # Padding the inputs just once
    tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents, batch_first=True, padding_value=tokenizer.pad_token_id)
    features = []

    batch_size = 128
    num_batches = math.ceil(size / batch_size)
    max_len = tokenized_sents_padded.shape[1]

    # Precompute mask for the full batch, then slice per batch
    pad_mask = torch.arange(max_len).expand(size, max_len)
    lens = torch.tensor([len(sent) for sent in tokenized_sents], device=pad_mask.device).unsqueeze(1)
    attention_mask_global = (pad_mask < lens).long().to(device)

    for i in range(num_batches):
        start_sentence = i * batch_size
        end_sentence = min(start_sentence + batch_size, size)
        padded_input = tokenized_sents_padded[start_sentence:end_sentence].to(device)
        attention_mask = attention_mask_global[start_sentence:end_sentence]

        if detach:
            with torch.no_grad():
                feature = model(padded_input.clone().detach(), attention_mask=attention_mask, output_hidden_states=True)
                feats = cloned_feature(feature.hidden_states, num_layers, detach)
        else:
            feature = model(padded_input, attention_mask=attention_mask, output_hidden_states=True)
            feats = cloned_feature(feature.hidden_states, num_layers, detach)

        features.extend(feats)

    assert len(features) == size
    assert len(features) == len(processed)

    # Compute offsets efficiently up front
    offsets = []
    for idx, token_list in enumerate(list_tokenized):
        off = [idx2 + 1 for idx2, _ in enumerate(token_list)
               if (idx2 > 0 and not token_list[idx2-1].endswith("@@")) or (idx2 == 0)]
        if keep_endpoints:
            off = [0] + off + [-1]
        offsets.append(off)

    # Optimize by using torch.stack where possible
    processed = [feature[offset] for feature, offset in zip(features, offsets)]

    # This is a list of tensors
    # Each tensor holds the representation of a sentence extracted from phobert
    return processed

BAD_TOKENIZERS = ('bert-base-german-cased',
                  # the dbmdz tokenizers turn one or more types of characters into empty words
                  # for example, from PoSTWITA:
                  #   ewww 󾓺 — in viaggio Roma
                  # the character which may not be rendering properly is 0xFE4FA
                  # https://github.com/dbmdz/berts/issues/48
                  'dbmdz/bert-base-german-cased',
                  'dbmdz/bert-base-italian-xxl-cased',
                  'dbmdz/bert-base-italian-cased',
                  'dbmdz/electra-base-italian-xxl-cased-discriminator',
                  # each of these (perhaps using similar tokenizers?)
                  # does not digest the script-flip-mark \u200f
                  'avichr/heBERT',
                  'onlplab/alephbert-base',
                  'imvladikon/alephbertgimmel-base-512',
                  # these indonesian models fail on a sentence in the Indonesian GSD dataset:
                  # 'Tak', 'dapat', 'disangkal', 'jika', '\u200e', 'kemenangan', ...
                  # weirdly some other indonesian models (even by the same group) don't have that problem
                  'cahya/bert-base-indonesian-1.5G',
                  'indolem/indobert-base-uncased',
                  'google/muril-base-cased',
                  'l3cube-pune/marathi-roberta')

def fix_blank_tokens(tokenizer, data):
    """Patch bert tokenizers with missing characters

    There is an issue that some tokenizers (so far the German ones identified above)
    tokenize soft hyphens or other unknown characters into nothing
    If an entire word is tokenized as a soft hyphen, this means the tokenizer
    simply vaporizes that word.  The result is we're missing an embedding for
    an entire word we wanted to use.

    The solution we take here is to look for any words which get vaporized
    in such a manner, eg `len(token) == 2`, and replace it with a regular "-"

    Actually, recently we have found that even the Bert / Electra tokenizer
    can do this in the case of "words" which are one special character long,
    so the easiest thing to do is just always run this function
    """
    new_data = []
    for sentence in data:
        tokenized = tokenizer(sentence, is_split_into_words=False).input_ids
        new_sentence = [word if len(token) > 2 else "-" for word, token in zip(sentence, tokenized)]
        new_data.append(new_sentence)
    return new_data

def extract_llama_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    # will calculate attention masks ourselves later
    tokenized = tokenizer(data, is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)

    list_offsets = []
    # Precompute offsets as a single sweep to improve locality
    for idx in range(len(data)):
        converted_offsets = convert_to_position_list(data[idx], tokenized.word_ids(batch_index=idx))
        list_offsets.append(converted_offsets)

    if any(any(x is None for x in converted_offsets) for converted_offsets in list_offsets):
        raise ValueError("OOPS, hit None when preparing to use transformer at idx {}\ndata[idx]: {}\nlist_offsets[idx]: {}\ntokenizer output: {}".format(idx, data[idx], list_offsets[idx], tokenized))

    features = []
    batch_size = 128
    num_batches = math.ceil(len(data) / batch_size)
    input_ids_all = tokenized['input_ids']
    for i in range(num_batches):
        batch_ids = [id_row + [tokenizer.eos_token_id] for id_row in input_ids_all[batch_size*i:batch_size*i+batch_size]]
        max_id_len = max(len(x) for x in batch_ids)
        attention_tensor = torch.zeros((len(batch_ids), max_id_len), dtype=torch.long, device=device)
        for idx, id_row in enumerate(batch_ids):
            attention_tensor[idx, :len(id_row)] = 1
            if len(id_row) < max_id_len:
                id_row.extend([0] * (max_id_len - len(id_row)))
        id_tensor = torch.tensor(batch_ids, device=device)

        if detach:
            with torch.no_grad():
                feats = build_cloned_features(model, tokenizer, attention_tensor, id_tensor, num_layers, detach, device)
        else:
            feats = build_cloned_features(model, tokenizer, attention_tensor, id_tensor, num_layers, detach, device)
        features.extend(feats)

    #process the output
    processed = []
    if not keep_endpoints:
        #remove the bos and eos tokens
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed


def extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    # using attention masks makes contextual embeddings much more useful for downstream tasks
    tokenized = tokenizer(data, is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
    #tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)

    list_offsets = []
    input_ids_all = []
    for idx, sentence in enumerate(data):
        offsets = tokenized.word_ids(batch_index=idx)
        offsets_list = [None]*(len(sentence)+2)
        offsets_list[0] = 0
        for pos, offset in enumerate(offsets):
            if offset is None:
                break
            offsets_list[offset+1] = pos + 1
        offsets_list[-1] = offsets_list[-2] + 1
        if any(x is None for x in offsets_list):
            raise ValueError("OOPS, hit None when preparing to use Bert\ndata[idx]: {}\noffsets: {}\nlist_offsets[idx]: {}".format(sentence, offsets, offsets_list, tokenized))

        if len(offsets) > tokenizer.model_max_length - 2:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), sentence)
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(sentence))

        list_offsets.append(offsets_list)
        ids = tokenized['input_ids'][idx]
        input_ids_all.append([tokenizer.bos_token_id] + ids[:-2] + [tokenizer.eos_token_id])

    features = []
    batch_size = 128
    num_batches = math.ceil(len(data) / batch_size)
    for i in range(num_batches):
        input_ids = input_ids_all[batch_size*i:batch_size*i+batch_size]
        max_len = max(len(x) for x in input_ids)
        attention_mask = torch.zeros(len(input_ids), max_len, dtype=torch.long, device=device)
        for idx, input_row in enumerate(input_ids):
            attention_mask[idx, :len(input_row)] = 1
            if len(input_row) < max_len:
                input_row.extend([tokenizer.pad_token_id] * (max_len - len(input_row)))
        id_tensor = torch.tensor(input_ids, device=device)
        if detach:
            with torch.no_grad():
                feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
                feats = cloned_feature(feature.hidden_states, num_layers, detach)
        else:
            feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
            feats = cloned_feature(feature.hidden_states, num_layers, detach)
        features.extend(feats)

    processed = []
    if not keep_endpoints:
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed

def build_cloned_features(model, tokenizer, attention_tensor, id_tensor, num_layers, detach, device):
    """
    Extract an embedding from the given transformer for a certain attention mask and tokens range

    In the event that the tokens are longer than the max length
    supported by the model, the range is split up into overlapping
    sections and the overlapping pieces are connected.  No idea if
    this is actually any good, but at least it returns something
    instead of horribly failing

    TODO: at least two upgrades are very relevant
      1) cut off some overlap at the end as well
      2) use this on the phobert, bart, and xln versions as well
    """
    if attention_tensor.shape[1] <= tokenizer.model_max_length:
        features = model(id_tensor, attention_mask=attention_tensor, output_hidden_states=True)
        features = cloned_feature(features.hidden_states, num_layers, detach)
        return features

    slices = []
    slice_len = max(tokenizer.model_max_length - 20, tokenizer.model_max_length // 2)
    prefix_len = tokenizer.model_max_length - slice_len
    if slice_len < 5:
        raise RuntimeError("Really tiny tokenizer!")
    remaining_attention = attention_tensor
    remaining_ids = id_tensor
    while True:
        attention_slice = remaining_attention[:, :tokenizer.model_max_length]
        id_slice = remaining_ids[:, :tokenizer.model_max_length]
        features = model(id_slice, attention_mask=attention_slice, output_hidden_states=True)
        features = cloned_feature(features.hidden_states, num_layers, detach)
        if len(slices) > 0:
            features = features[:, prefix_len:, :]
        slices.append(features)
        if remaining_attention.shape[1] <= tokenizer.model_max_length:
            break
        remaining_attention = remaining_attention[:, slice_len:]
        remaining_ids = remaining_ids[:, slice_len:]
    slices = torch.cat(slices, axis=1)
    return slices


def convert_to_position_list(sentence, offsets):
    """
    Convert a transformers-tokenized sentence's offsets to a list of word to position
    """
    # +2 for the beginning and end
    list_offsets = [None] * (len(sentence) + 2)
    for pos, offset in enumerate(offsets):
        if offset is None:
            continue
        # this uses the last token piece for any offset by overwriting the previous value
        list_offsets[offset+1] = pos
    list_offsets[0] = 0
    for offset in list_offsets[-2::-1]:
        # count backwards in case the last position was
        # a word or character that got erased by the tokenizer
        # this loop should eventually find something...
        # after all, we just set the first one to be 0
        if offset is not None:
            list_offsets[-1] = offset + 1
            break
    return list_offsets

def extract_base_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach):
    #add add_prefix_space = True for RoBerTa-- error if not
    # using attention masks makes contextual embeddings much more useful for downstream tasks
    tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)
    input_ids_all = tokenized['input_ids']
    attention_mask_all = tokenized['attention_mask']
    list_offsets = []
    for idx in range(len(data)):
        converted_offsets = convert_to_position_list(data[idx], tokenized.word_ids(batch_index=idx))
        list_offsets.append(converted_offsets)

    # If missing offsets, fix blank tokens and retokenize just like original
    if any(any(x is None for x in converted_offsets) for converted_offsets in list_offsets):
        data = fix_blank_tokens(tokenizer, data)
        tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)
        input_ids_all = tokenized['input_ids']
        attention_mask_all = tokenized['attention_mask']
        list_offsets = []
        for idx in range(len(data)):
            converted_offsets = convert_to_position_list(data[idx], tokenized.word_ids(batch_index=idx))
            list_offsets.append(converted_offsets)

    if any(any(x is None for x in converted_offsets) for converted_offsets in list_offsets):
        raise ValueError("OOPS, hit None when preparing to use transformer at idx {}\ndata[idx]: {}\nlist_offsets[idx]: {}\ntokenizer output: {}".format(idx, data[idx], list_offsets[idx], tokenized))

    features = []
    batch_size = 128
    num_batches = math.ceil(len(data) / batch_size)
    for i in range(num_batches):
        attention_tensor = torch.tensor(attention_mask_all[batch_size*i:batch_size*i+batch_size], device=device)
        id_tensor = torch.tensor(input_ids_all[batch_size*i:batch_size*i+batch_size], device=device)
        if detach:
            with torch.no_grad():
                feats = build_cloned_features(model, tokenizer, attention_tensor, id_tensor, num_layers, detach, device)
        else:
            feats = build_cloned_features(model, tokenizer, attention_tensor, id_tensor, num_layers, detach, device)
        features.extend(feats)

    processed = []
    if not keep_endpoints:
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed

def extract_bert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers=None, detach=True, peft_name=None):
    """
    Extract transformer embeddings using a generic roberta extraction

    data: list of list of string (the text tokens)
    num_layers: how many to return.  If None, the average of -2, -3, -4 is returned
    """
    # TODO: can maybe cache this value for a model and save some time
    # TODO: too bad it isn't thread safe, but then again, who does?
    if peft_name is None:
        if model._hf_peft_config_loaded:
            model.disable_adapters()
    else:
        model.enable_adapters()
        model.set_adapter(peft_name)

    if model_name.startswith("vinai/phobert"):
        return extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)

    if 'bart' in model_name:
        # this should work with "vinai/bartpho-word"
        # not sure this works with any other Bart
        return extract_bart_word_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)

    if isinstance(data, tuple):
        data = list(data)

    if "xlnet" in model_name:
        return extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)

    if "LLaMA" in model_name:
        return extract_llama_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)

    return extract_base_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)

