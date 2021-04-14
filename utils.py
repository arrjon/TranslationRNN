# Jonas Arruda

import time
import math

import torch

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def indexesFromSentence(lang, sentence):
    return [lang.getIndex(word) for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, EOS_token=1, device='cpu'):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, device='cpu'):
    input_tensor = tensorFromSentence(input_lang, pair[0], device=device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device=device)
    return input_tensor, target_tensor


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("loss-plot.png")
    plt.close()
    return
