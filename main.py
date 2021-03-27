# Jonas Arruda

from language import prepareData
from model import EncoderRNN, AttnDecoderRNN
from training import train, evaluateRandomly

import torch
import random
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 10
hidden_size = 256
lr = 0.01
teacher_forcing_ratio = 0.5
iterations = 75000

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

input_lang, output_lang, pairs = prepareData('eng', 'deu', max_length, eng_prefixes)

file = open('data/input.lang', 'wb')
pickle.dump(input_lang, file)
file.close()
file = open('data/output.lang', 'wb')
pickle.dump(output_lang, file)
file.close()

# split data into training and test data
train_pairs = []
test_pairs = []
random_split = 0.33
for pair in pairs:
    if random.random() < random_split:
        test_pairs.append(pair)
    else:
        train_pairs.append(pair)


encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=max_length, device=device).to(device)

train(encoder1, attn_decoder1, train_pairs, input_lang, output_lang, n_iterations=iterations,
      learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio, max_length=max_length, device=device)

torch.save(encoder1.state_dict(), 'encoder.pth')
torch.save(attn_decoder1.state_dict(), 'decoder.pth')

evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, test_pairs, max_length, device=device)
