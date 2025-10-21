"""
Process the licensed version of SUC3 to BIO

The main program processes the expected location, or you can pass in a
specific zip or filename to read
"""

from io import TextIOWrapper
from zipfile import ZipFile

def extract(infile, outfile):
    """
    Convert the infile to an outfile

    Assumes the files are already open (this allows you to pass in a zipfile reader, for example)

    The SUC3 format is like conll, but with the tags in tabs 10 and 11
    """
    # Use local variables/attributes for minimal lookups
    infile_readline = infile.readline
    outfile_write = outfile.write

    sentences = []
    cur_sentence = []

    idx = 0
    # Process lines one-at-a-time instead of .readlines() to reduce memory usage and allow early freeing of buffer
    while True:
        line = infile_readline()
        if not line:
            break
        line = line.strip()
        if not line:
            if cur_sentence:
                sentences.append(cur_sentence)
                cur_sentence = []
            idx += 1
            continue

        pieces = line.split("\t")
        # Inline len(pieces) check for earlier error detection
        if len(pieces) < 12:
            raise ValueError("Unexpected line length in the SUC3 dataset at %d" % idx)
        # Reduce local variable, use tuple construction directly for slightly less overhead
        tag = pieces[10]
        if tag == 'O':
            cur_sentence.append((pieces[1], "O"))
        else:
            cur_sentence.append((pieces[1], f"{tag}-{pieces[11]}"))
        idx += 1
    if cur_sentence:
        sentences.append(cur_sentence)

    # Write output in a single pass, buffered write is already handled by file object
    for sentence in sentences:
        for word, label in sentence:
            outfile_write(f"{word}\t{label}\n")
        outfile_write("\n")

    return len(sentences)

def extract_from_zip(zip_filename, in_filename, out_filename):
    """
    Process a single file from SUC3

    zip_filename: path to SUC3.0.zip
    in_filename: which piece to read
    out_filename: where to write the result
    """
    with ZipFile(zip_filename) as zin:
        with zin.open(in_filename) as fin:
            with open(out_filename, "w") as fout:
                num = extract(TextIOWrapper(fin, encoding="utf-8"), fout)
                print("Processed %d sentences from %s:%s to %s" % (num, zip_filename, in_filename, out_filename))
                return num

def process_suc3(zip_filename, short_name, out_dir):
    extract_from_zip(zip_filename, "SUC3.0/corpus/conll/suc-train.conll", "%s/%s.train.bio" % (out_dir, short_name))
    extract_from_zip(zip_filename, "SUC3.0/corpus/conll/suc-dev.conll", "%s/%s.dev.bio" % (out_dir, short_name))
    extract_from_zip(zip_filename, "SUC3.0/corpus/conll/suc-test.conll", "%s/%s.test.bio" % (out_dir, short_name))

def main():
    process_suc3("extern_data/ner/sv_suc3/SUC3.0.zip", "data/ner")

if __name__ == '__main__':
    main()
