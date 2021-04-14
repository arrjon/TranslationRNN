from utils import tensorsFromPair, showPlot, timeSince, tensorFromSentence

import random
import time

import torch
import torch.nn as nn
from torch import optim

from nltk.translate.bleu_score import sentence_bleu
import warnings

import matplotlib.pyplot as plt

plt.switch_backend('agg')
warnings.filterwarnings("ignore")  # to catch warning from sentence_bleu


def batch_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, teacher_forcing_ratio, max_length, SOS_token, EOS_token, device):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, decoder, pairs, input_lang, output_lang, n_iterations, print_every=1000, plot_every=100,
          learning_rate=0.01, teacher_forcing_ratio=0.5, max_length=10, SOS_token=0, EOS_token=1, device='cpu'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, device=device)
                      for j in range(n_iterations)]
    criterion = nn.NLLLoss()

    for i in range(1, n_iterations + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = batch_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                           criterion, teacher_forcing_ratio, max_length, SOS_token, EOS_token, device)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iterations),
                                         i, i / n_iterations * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=10,
             SOS_token=0, EOS_token=1, device='cpu'):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length, device=device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def cal_bleu_score(encoder, decoder, input_lang, output_lang, test_pairs, max_length=10, device='cpu'):
    score4 = []
    score3 = []
    score2 = []
    score1 = []

    for i, pair in enumerate(test_pairs):
        predicted_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0],
                                               max_length, device=device)
        target = [word for word in pair[1].split(' ')]
        score4.append(sentence_bleu([predicted_words[:-1]], target, weights=[0.25] * 4))
        score3.append(sentence_bleu([predicted_words[:-1]], target, weights=[1 / 3] * 3))
        score2.append(sentence_bleu([predicted_words[:-1]], target, weights=[0.5] * 2))
        score1.append(sentence_bleu([predicted_words[:-1]], target, weights=[1] * 1))

    print('BLEU-4 Score:', round(sum(score4)/len(test_pairs), 2))
    print('BLEU-3 Score:', round(sum(score3)/len(test_pairs), 2))
    print('BLEU-2 Score:', round(sum(score2)/len(test_pairs), 2))
    print('BLEU-1 Score:', round(sum(score1)/len(test_pairs), 2))
    return


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length=10, n=10, device='cpu'):
    score4 = []
    score3 = []
    score2 = []
    score1 = []
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0],
                                            max_length, device=device)
        output_sentence = ' '.join(output_words[:-1])
        print('<', output_sentence)
        print('')

        target = [word for word in pair[1].split(' ')]
        score4.append(sentence_bleu([output_words[:-1]], target, weights=[0.25] * 4))
        score3.append(sentence_bleu([output_words[:-1]], target, weights=[1 / 3] * 3))
        score2.append(sentence_bleu([output_words[:-1]], target, weights=[0.5] * 2))
        score1.append(sentence_bleu([output_words[:-1]], target, weights=[1] * 1))

    print('BLEU-4 Score:', round(sum(score4)/n, 2))
    print('BLEU-3 Score:', round(sum(score3)/n, 2))
    print('BLEU-2 Score:', round(sum(score2)/n, 2))
    print('BLEU-1 Score:', round(sum(score1)/n, 2))
    return
