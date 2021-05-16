from urllib.request import urlretrieve
import os
if not os.path.isfile('datasets/mini.h5'):
    print("Downloading Conceptnet Numberbatch word embeddings...")
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
    urlretrieve(conceptnet_url, 'datasets/mini.h5')
    
# Load the file and pull out words and embeddings
import h5py

with h5py.File('datasets/mini.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]
    
print("all_words dimensions: {}".format(len(all_words)))
print("all_embeddings dimensions: {}".format(all_embeddings.shape))

print("Random example word: {}".format(all_words[1337]))

# Restrict our vocabulary to just the English words
english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
english_embeddings = all_embeddings[english_word_indices]

print("Number of English words in all_words: {0}".format(len(english_words)))
print("english_embeddings dimensions: {0}".format(english_embeddings.shape))

print(english_words[1337])


#========================================================================================================================================================================================
        #normalizing the word vectors
#========================================================================================================================================================================================
import numpy as np

norms = np.linalg.norm(english_embeddings, axis=1) 
#norms will be a one dimensional array (list) containing norm of each of the 150875 vectors
#we reshape it to get a 2D column vector

#converting to unit vectors 
normalized_embeddings = english_embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1]) 
print(normalized_embeddings.shape)

#========================================================================================================================================================================================
        #Making a dictonary of words and vectors (mapping words and vectors)
#========================================================================================================================================================================================
#mapping each word to it's vector
index = {word: i for i, word in enumerate(english_words)}

#========================================================================================================================================================================================
        #defining similarity score between two vectors
#========================================================================================================================================================================================

def similarity_score(w1, w2):
    score = np.dot(normalized_embeddings[index[w1], :], normalized_embeddings[index[w2], :])
    return score

# A word is as similar with itself as possible:
print('cat\tcat\t', similarity_score('cat', 'cat'))

# Closely related words still get high scores:
print('cat\tfeline\t', similarity_score('cat', 'feline'))
print('cat\tdog\t', similarity_score('cat', 'dog'))

# Unrelated words, not so much
print('cat\tmoo\t', similarity_score('cat', 'moo'))
print('cat\tfreeze\t', similarity_score('cat', 'freeze'))

# Antonyms are still considered related, sometimes more so than synonyms
print('antonym\topposite\t', similarity_score('antonym', 'opposite'))
print('antonym\tsynonym\t', similarity_score('antonym', 'synonym'))

#========================================================================================================================================================================================
        #getting words closer to a given word
#========================================================================================================================================================================================

def closest_to_vector(v, n):
    all_scores = np.dot(normalized_embeddings, v)
    best_words = list(map(lambda i: english_words[i], reversed(np.argsort(all_scores))))
    return best_words[:n]

def most_similar(w, n):
    return closest_to_vector(normalized_embeddings[index[w], :], n)

print(most_similar('cat', 10))
print(most_similar('dog', 10))
print(most_similar('duke', 10))

#========================================================================================================================================================================================
        #getting a word analogus to three given words
#========================================================================================================================================================================================

def solve_analogy(a1, b1, a2):
    b2 = normalized_embeddings[index[b1], :] - normalized_embeddings[index[a1], :] + normalized_embeddings[index[a2], :]
    return closest_to_vector(b2, 1)

print(solve_analogy("man", "brother", "woman"))
print(solve_analogy("man", "husband", "woman"))
print(solve_analogy("spain", "madrid", "france"))


#========================================================================================================================================================================================
        #using word embeddings in deep models
#========================================================================================================================================================================================
# We'll perform sentiment analysis on a set of movie reviews:
# in particular, we will attempt to classify a movie review as positive or negative based on its text.
# We will use a Simple Word Embedding Model (SWEM, Shen et al. 2018) to do so.
# We will represent a review as the mean of the embeddings of the words in the review.
# Then we'll train a two-layer MLP (a neural network) to classify the review as positive or negative.
# As you might guess, using just the mean of the embeddings discards a lot of the information in a sentences,
# but for tasks like sentiment analysis, it can be surprisingly effective.
#========================================================================================================================================================================================

import string
remove_punct=str.maketrans('','',string.punctuation)

# This function converts a line of our data file into
# a tuple (x, y), where x is 300-dimensional representation
# of the words in a review, and y is its label.
def convert_line_to_example(line):
    # Pull out the first character: that's our label (0 or 1)
    y = int(line[0])
    
    # Split the line into words using Python's split() function
    words = line[2:].translate(remove_punct).lower().split()
    
    # Look up the embeddings of each word, ignoring words not
    # in our pretrained vocabulary.
    embeddings = [normalized_embeddings[index[w]] for w in words
                  if w in index]
    #embeddings is a list of m dimensional 1D arrays which represent word vectors
    
    # Take the mean of the embeddings
    x = np.mean(np.vstack(embeddings), axis=0)
    return x, y

# Apply the function to each line in the file.
xs = []
ys = []
with open("datasets/movie-simple.txt", "r", encoding='utf-8', errors='ignore') as f:
    for l in f.readlines():
        x, y = convert_line_to_example(l)
        xs.append(x)
        ys.append(y)

# Concatenate all examples into a numpy array
xs = np.vstack(xs)
ys = np.vstack(ys)
print("Shape of inputs: {}".format(xs.shape))
print("Shape of labels: {}".format(ys.shape))

num_examples = xs.shape[0]

#========================================================================================================================================================================================
        #Shuffling the data before training it
#========================================================================================================================================================================================
print("First 20 labels before shuffling: {0}".format(ys[:20, 0]))

shuffle_idx = np.random.permutation(num_examples)
xs = xs[shuffle_idx, :]
ys = ys[shuffle_idx, :]

print("First 20 labels after shuffling: {0}".format(ys[:20, 0]))

#========================================================================================================================================================================================
        #let's create a TensorDataset and DataLoader as we've used in the past for MNIST.
#========================================================================================================================================================================================
import torch

num_train = 4*num_examples // 5

x_train = torch.tensor(xs[:num_train])
y_train = torch.tensor(ys[:num_train], dtype=torch.float32)

x_test = torch.tensor(xs[num_train:])
y_test = torch.tensor(ys[num_train:], dtype=torch.float32)

reviews_train = torch.utils.data.TensorDataset(x_train, y_train)
reviews_test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(reviews_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(reviews_test, batch_size=100, shuffle=False)

#========================================================================================================================================================================================
        #Building the SWEM(simple word embedding model)
#========================================================================================================================================================================================
import torch.nn as nn
import torch.nn.functional as F

class SWEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#========================================================================================================================================================================================
        #Training the SWEM(simple word embedding model)
#========================================================================================================================================================================================

# Instantiate model
model = SWEM()

# Binary cross-entropy (BCE) Loss and Adam Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(250):
    correct = 0
    num_examples = 0
    for inputs, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y = model(inputs)
        loss = criterion(y, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_examples += len(inputs)
    
    # Print training progress
    if epoch % 25 == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

## Testing
correct = 0
num_test = 0

with torch.no_grad():
    # Iterate through test set minibatchs 
    for inputs, labels in test_loader:
        # Forward pass
        y = model(inputs)
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))

#========================================================================================================================================================================================
        # Check some words
#========================================================================================================================================================================================

words_to_test = ["exciting", "hated", "boring", "loved"]

for word in words_to_test:
    x = torch.tensor(normalized_embeddings[index[word]].reshape(1, 300))
    print("Sentiment of the word '{0}': {1}".format(word, torch.sigmoid(model(x))))
    
#========================================================================================================================================================================================
        # Learning the word embeddings within the model
#========================================================================================================================================================================================
VOCAB_SIZE = 5000
EMBED_DIM = 300

embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
print(embedding.weight.size())

class SWEMWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
model = SWEMWithEmbeddings(
    vocab_size = 5000,
    embedding_size = 300, 
    hidden_dim = 64, 
    num_outputs = 1,
)
print(model)    



















