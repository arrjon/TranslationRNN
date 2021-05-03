from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from language import normalizeString
from tqdm import tqdm
import pickle

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
# eng_prefixes = None
if eng_prefixes is None:
    path = 'data/long/'
else:
    path = 'data/short/'

file = open(path+'input.lang', 'rb')
input_lang = pickle.load(file)
file = open(path+'output.lang', 'rb')
output_lang = pickle.load(file)
file = open(path+'pairs.p', 'rb')
pairs = pickle.load(file)


def cal_bleu_score_google(test_pairs):
    translator = Translator()

    score4 = []
    score3 = []
    score2 = []
    score1 = []
    count = 0

    for pair in tqdm(test_pairs):
        count += 1
        # use google translator API
        try:
            translation = translator.translate(pair[0], dest='de')
        except AttributeError:
            print(count)
            break
        # clean and tokenize the words
        predicted_words = [normalizeString(word) for word in translation.text.split(' ')]
        # compute BLEU score
        target = [word for word in pair[1].split(' ')]
        score4.append(sentence_bleu([predicted_words], target, weights=[0.25] * 4))
        score3.append(sentence_bleu([predicted_words], target, weights=[1 / 3] * 3))
        score2.append(sentence_bleu([predicted_words], target, weights=[0.5] * 2))
        score1.append(sentence_bleu([predicted_words], target, weights=[1] * 1))

    print('BLEU-4 Score:', round(sum(score4)/count, 2))
    print('BLEU-3 Score:', round(sum(score3)/count, 2))
    print('BLEU-2 Score:', round(sum(score2)/count, 2))
    print('BLEU-1 Score:', round(sum(score1)/count, 2))
    return


cal_bleu_score_google(pairs)
