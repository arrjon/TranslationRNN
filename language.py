from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS and UNK
        self.cutoff_point = -1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        return

    def addWord(self, word):
        if self.word2count[word] > self.cutoff_point:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                # self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            # else:
            #    self.word2count[word] += 1
        return

    def countWords(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2count:
                self.word2count[word] = 1
            else:
                self.word2count[word] += 1

    def createCutoff(self, max_vocab_size):
        word_freq = list(self.word2count.values())
        word_freq.sort(reverse=True)
        if len(word_freq) > max_vocab_size:
            self.cutoff_point = word_freq[max_vocab_size]
        return

    def getIndex(self, word, UNK_token=2):
        try:
            index = self.word2index[word]
        except KeyError:
            index = UNK_token
        return index


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, max_length, lang_prefixes=None):
    if lang_prefixes:
        return len(p[0].split(' ')) < max_length and \
            len(p[1].split(' ')) < max_length and \
            p[0].startswith(lang_prefixes)
    else:
        return len(p[0].split(' ')) < max_length and \
            len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length, lang_prefixes=None):
    return [pair for pair in pairs if filterPair(pair, max_length, lang_prefixes)]


def prepareData(lang1, lang2, max_length, max_vocab_size=20000, lang_prefixes=None, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs, max_length, lang_prefixes)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.countWords(pair[0])
        output_lang.countWords(pair[1])

    input_lang.createCutoff(max_vocab_size)
    output_lang.createCutoff(max_vocab_size)

    print("Create vocabulary...")
    clean_pairs = []
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        clean_pairs.append([pair[0], pair[1]])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, clean_pairs
