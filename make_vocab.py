import codecs
import os
from collections import Counter
import regex

def make_vocab(fpath, f_name):
    text = codecs.open(fpath, "r", "utf-8").read()
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists("data/preprocessed"):
        os.mkdir("data/preprocessed")
    with codecs.open("data/preprocessed/{}".format(f_name), "w", "utf-8") as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
                "<PAD>", "<UNK>", "<S>", "</S>"
            )
        )
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == "__main__":
    make_vocab("data/train/cn.txt", "cn.txt.vocab.tsv")
    make_vocab("data/train/en.txt", "en.txt.vocab.tsv")
    make_vocab("data/test/cn.txt", "cn.txt.vocab.tsv")
    make_vocab("data/test/en.txt", "en.txt.vocab.tsv")
    print("Done")