# Jonas Arruda

from language import prepareData
from model import EncoderRNN, AttnDecoderRNN
from training import train, evaluateRandomly, cal_bleu_score

import torch
import random
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# parameters of the model
max_length = 10
hidden_size = 256
lr = 0.01
teacher_forcing_ratio = 0.5

# define prefixes
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

# create vocabulary
input_lang, output_lang, pairs = prepareData('eng', 'deu', max_length, lang_prefixes=eng_prefixes)
iterations = len(pairs)*7

# save vocabulary
file = open(path+'pairs.p', 'wb')
pickle.dump(pairs, file)
file.close()
file = open(path+'input.lang', 'wb')
pickle.dump(input_lang, file)
file.close()
file = open(path+'output.lang', 'wb')
pickle.dump(output_lang, file)
file.close()

# split data into training and test data
train_pairs = []
test_pairs = []
random_split = 0.1
random.seed(1)  # to get same test data every time
for pair in pairs:
    if random.random() < random_split:
        test_pairs.append(pair)
    else:
        train_pairs.append(pair)

# create model
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=max_length, device=device).to(device)

# train model
train(encoder1, attn_decoder1, train_pairs, input_lang, output_lang, n_iterations=iterations,
      learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio, max_length=max_length, device=device)

# save model
torch.save(encoder1.state_dict(), path+'encoder.pth')
torch.save(attn_decoder1.state_dict(), path+'decoder.pth')

# test model
print('Scores on training data:')
cal_bleu_score(encoder1, attn_decoder1, input_lang, output_lang, train_pairs, max_length, device=device)
print('Scores on test data:')
cal_bleu_score(encoder1, attn_decoder1, input_lang, output_lang, test_pairs, max_length, device=device)
evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, test_pairs, max_length, device=device)
