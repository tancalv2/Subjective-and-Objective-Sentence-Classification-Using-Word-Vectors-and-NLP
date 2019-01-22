"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""
import torch
import torchtext
from torchtext import data
import spacy

import argparse

# Part 1
text = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
label = data.Field(sequential=False, use_vocab=False)
train_data, val_data, test_data = data.TabularDataset.splits(
                                    path='./data/', train='train.tsv', validation='val.tsv',
                                    test='test.tsv', skip_header=True, format='tsv',
                                    fields=[('text', text), ('label', label)])
text.build_vocab(train_data)
glove_emb = torchtext.vocab.GloVe(name="6B", dim=100)
text.vocab.load_vectors(glove_emb)

# Part 2
model_baseline = torch.load('./models/model_baseline.pt')
model_rnn = torch.load('./models/model_rnn.pt')
model_cnn = torch.load('./models/model_cnn.pt')

# Part 3
def tokenizer(string):
    nlp = spacy.load('en')
    return nlp(string)


while True:
    print("Enter a sentence")
    string = input()

    # Part 4
    list_string = tokenizer(string)
    list_int = []
    for i in range(len(list_string)):
        list_int.append(text.vocab.stoi[str(list_string[i])])

    # Part 5
    sentence_tensor = torch.LongTensor([list_int]).permute(1, 0)

    # Part 6
    sentence_length = torch.LongTensor([len(list_int)])

    baseline = model_baseline(sentence_tensor, sentence_length)
    rnn = model_rnn(sentence_tensor, sentence_length)
    cnn = model_cnn(sentence_tensor, sentence_length)

    # Part 7
    value_b = int(baseline.detach().numpy()*1000)/1000
    value_r = int(rnn.detach().numpy()*1000)/1000
    value_c = int(cnn.detach().numpy()*1000)/1000

    if value_b > 0.5:
        print('Model baseline: subjective ({})'.format(value_b))
    else:
        print('Model baseline: objective ({})'.format(value_b))
    if value_r > 0.5:
        print('Model rnn: subjective ({})'.format(value_r))
    else:
        print('Model rnn: objective ({})'.format(value_r))
    if value_c > 0.5:
        print('Model cnn: subjective ({})'.format(value_c))
    else:
        print('Model cnn: objective ({})'.format(value_c))
    print('\n')

