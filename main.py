import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import torchtext
from torchtext import data
import spacy

import argparse
import os


from models import *


def load_model(lr, arg_model, embedding_dim, vocab, hidden_dim, n_filters, filter_sizes):

    loss_fnc = torch.nn.BCELoss()
    if arg_model == 'baseline':
        model = Baseline(embedding_dim, vocab)
    elif arg_model == 'rnn':
        model = RNN(embedding_dim, vocab, hidden_dim)
    else:
        model = CNN(embedding_dim, vocab, n_filters, filter_sizes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def evaluate(model, data_iter, loss_fnc):

    accum_loss = 0
    total_corr = 0
    for i, texts in enumerate(data_iter):
        words, length = texts.text
        labels = texts.label
        prediction = model(words, length)
        batch_loss = loss_fnc(input=prediction.squeeze(), target=labels.float())
        corr = (prediction > 0.5).squeeze().long() == labels
        accum_loss += batch_loss
        total_corr += int(corr.sum())
    loss = float(accum_loss) / (i + 1)

    return float(total_corr)/len(data_iter.dataset), loss


# =================================== VISUALIZE TRAINING AND VALIDATION =========================================== #


def plot_data(train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, model):
    ######

    # Accuracy
    plt.figure()
    plt.title("Accuracy vs. Epochs ({}) ".format(model), fontsize=14)
    plt.plot(train_acc, label="Training")
    plt.plot(valid_acc, label="Validation")
    plt.plot(test_acc, label="Testing")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc=4, fontsize=12)  # lower right
    plt.savefig("./plots/acc_vs_epoch_model_{}.png".format(model))
    plt.show()

    # Loss
    plt.figure()
    plt.title("Loss vs. Epochs ({}) ".format(model), fontsize=14)
    plt.plot(train_loss, label="Training")
    plt.plot(valid_loss, label="Validation")
    plt.plot(test_loss, label="Testing")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc=4, fontsize=12)  # lower right
    plt.savefig("./plots/loss_vs_epoch_model_{}.png".format(model))
    plt.show()

    #######


def main(args):
    ######

    # 3.2 Processing of the data

    ######

    text = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    label = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
                                        path='./data/', train='train.tsv', validation='val.tsv',
                                        test='test.tsv', skip_header=True, format='tsv',
                                        fields=[('text', text), ('label', label)])

    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                     sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
    val_iter = data.BucketIterator(val_data, batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
    test_iter = data.BucketIterator(test_data, batch_size=args.batch_size,
                                    sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)

    # Section 7.3.(c)
    # Remember to:
    # use RNN
    # comment pack_padded_sequence of RNN
    # train_iter = data.Iterator(train_data, batch_size=args.batch_size,
    #                            sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
    # val_iter = data.Iterator(val_data, batch_size=args.batch_size,
    #                          sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
    # test_iter = data.Iterator(test_data, batch_size=args.batch_size,
    #                           sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)

    text.build_vocab(train_data)

    ######

    # 4.1 Loading GloVe Vector

    ######

    glove_emb = torchtext.vocab.GloVe(name="6B", dim=100)
    text.vocab.load_vectors(glove_emb)

    ######

    # 5 Training and Evaluation

    filter_sizes = (2, 4)   # k = [2, 4]
    model, loss_fnc, optimizer = load_model(args.lr, args.model, args.emb_dim, text.vocab,
                                            args.rnn_hidden_dim, args.num_filt, filter_sizes)

    MaxEpoch = args.epochs
    # start_time = time()

    # variables for every epoch
    train_acc = [0]
    train_loss = [0]
    valid_acc = [0]
    valid_loss = [0]
    test_acc = [0]
    test_loss = [0]

    for epoch in range(MaxEpoch):
        accum_loss = 0
        total_corr = 0
        for i, texts in enumerate(train_iter):
            words, length = texts.text
            labels = texts.label
            optimizer.zero_grad()
            prediction = model(words, length)
            batch_loss = loss_fnc(input=prediction.squeeze(), target=labels.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            corr = (prediction > 0.5).squeeze().long() == labels
            total_corr += int(corr.sum())

        train_acc.append(float(total_corr) / len(train_iter.dataset))
        train_loss.append(accum_loss / 100)

        val = evaluate(model, val_iter, loss_fnc)
        valid_acc.append(val[0])
        valid_loss.append(val[1])

        test = evaluate(model, test_iter, loss_fnc)
        test_acc.append(test[0])
        test_loss.append(test[1])

        print("Epoch: {}| Train acc: {} | Train loss: {} |  Valid acc: {} |  Valid loss: {}"
              .format(epoch + 1, train_acc[epoch + 1], train_loss[epoch + 1],
                      valid_acc[epoch + 1], valid_loss[epoch + 1]))

    #  training complete
    results = ["{} model".format(args.model),
               "Train Accuracy: " + str(train_acc[-1]), "Train Loss: " + str(float(train_loss[-1])),
               "Valid Accuracy: " + str(valid_acc[-1]), "Valid Loss: " + str(valid_loss[-1]),
               "Test Accuracy: " + str(test_acc[-1]), "Test Loss: " + str(test_loss[-1])]
    np.savetxt('./results/results_{}.txt'.format(args.model), results, fmt='%s')
    plot_data(train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, args.model)
    # torch.save(model, './models/model_{}.pt'.format(args.model))

    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    # parser.add_argument('--model', type=str, default='cnn',
    #                     help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--model', type=str, default='rnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    # parser.add_argument('--model', type=str, default='baseline',
    #                     help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    # All three models
    # type = ['baseline', 'rnn', 'cnn']
    # for model in type:
    #     args.model = model
    #     print('{}'.format(model))
    #     main(args)
    main(args)
