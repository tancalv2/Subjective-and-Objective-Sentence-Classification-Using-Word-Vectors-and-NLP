Subjective/Objective Sentence Classification Using Word Vectors and NLP
Dataset: 10,000 sentences (5,000 subjective, 5,000 objective) from https://www.cs.cornell.edu/people/pabo/movie-review-data/ based on Rotten Tomatoes Movie Reviews and IMDb plot summaries
Goal: Two parts
 1. Build three Natural Language Processing (NLP) models: baseline, Recurrent Neural Network (RNN), and Convolutional Neural Network (CNN) to predict a 2-class dataset using Pytorch torchtext library and pre-trained word embeddings
 2. Create a command-line interface to test "subjectivity" of a sentence

Hyperparameters
- Learning Rate = 0.001
- Batch Size = 64
- Epochs = 25
- Adam Optimizer (Default settings, as described in PyTorch Documentation)
- Loss Function = Binary Cross Entropy

Note:
GloVe model will be downloaded if not already present. Required space of 822 MB
only 'glove.6B.100d.txt' file required of GloVe model

install the following packages:
torchtext, spacy
download 'en' from spacy

Train:
- ensure all files are downloaded
- run main.py

Subjective Sentence Command-Line Tester:
- ensure models and data folder and subjective_bot.py are downloaded
- run subjective_bot.py