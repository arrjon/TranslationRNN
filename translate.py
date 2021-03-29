# Jonas Arruda

from language import normalizeString
from training import evaluate

from model import EncoderRNN, AttnDecoderRNN
import torch
import pickle

"""eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)"""
eng_prefixes = None

max_length = 10
hidden_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Loading model...')
file = open('data/input.lang', 'rb')
input_lang = pickle.load(file)
file = open('data/output.lang', 'rb')
output_lang = pickle.load(file)

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=max_length, device=device).to(device)

encoder1.load_state_dict(torch.load('encoder.pth', map_location=device))
attn_decoder1.load_state_dict(torch.load('decoder.pth', map_location=device))
print('Model loaded! \n')

test_sentence = normalizeString(input("English sentence to be translated: "))
if eng_prefixes:
    if not test_sentence.startswith(eng_prefixes):
        print('This sentence is not supported...')
        exit()
if len(test_sentence.split(' ')) < max_length:
    print('Translating...')
    try:
        output_words, attentions = evaluate(encoder1, attn_decoder1, input_lang, output_lang, test_sentence,
                                            max_length, device=device)
        output_sentence = ' '.join(output_words)
        print(output_sentence)
    except KeyError:
        print('Word not in dictionary...')
else:
    print('This sentence is not supported...')
