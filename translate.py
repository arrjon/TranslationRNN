# Jonas Arruda

from language import normalizeString
from training import evaluate, evaluateRandomly

from model import EncoderRNN, AttnDecoderRNN
import torch
import pickle

enter = input('Press enter to translate your own sentences...')
if enter == '':
    randomEval = False
else:
    randomEval = True

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

max_length = 10
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('Loading model...')
file = open(path+'input.lang', 'rb')
input_lang = pickle.load(file)
file = open(path+'output.lang', 'rb')
output_lang = pickle.load(file)
file = open(path+'pairs.p', 'rb')
pairs = pickle.load(file)


encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=max_length, device=device).to(device)

encoder1.load_state_dict(torch.load(path+'encoder.pth', map_location=device))
attn_decoder1.load_state_dict(torch.load(path+'decoder.pth', map_location=device))
print('Model loaded! \n')

if randomEval:
    evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, pairs, max_length, device=device)
    exit()

print('Prefixes are: I am..., he/she is..., you/we/they are...')
while True:
    try:
        test_sentence = normalizeString(input("English sentence to be translated: "))
        if len(test_sentence.split(' ')) < max_length:
            output_words, attentions = evaluate(encoder1, attn_decoder1, input_lang, output_lang, test_sentence,
                                                max_length, device=device)
            output_sentence = ' '.join(output_words[:-1])
            print('>>', output_sentence)
        else:
            print('This sentence is too long...')
        print('\n')
        enter = input('Press enter to continue...')
        if enter != '':
            break
        print('\n')
    except KeyboardInterrupt:
        break
