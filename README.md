# Movie-Review-Classification

Describing the dataset:
      The numeral 0 (for negative) or the numeral 1 (for positive), followed by a tab (the whitespace character), and then the review itself. Total number of examples are 1411 which is divided into 1128 training examples and 283 test examples to calculate accuracy for our model. Since our dataset is small I used word embeddings from the h5py library directly.

Describing the model:
       The model used is Simple Word Embedding Model (SWEM). In this model we represent review as mean of embeddings of the words in the review. In this way an entire paragraph is represented by a single word vector. Word embeddings used in the model are derived from the h5py library. Alternately we can learn the Word embeddings also when we have a large amount of training data. Model with learning Word embeddings is given at the end of the code. The 300 dimensional word vector is passed through two fully connected layers. The hidden layer has 64 neurons and the output layer has 1 neuron representing the probability of the review being positive.

convert_line_to_example function is defined in such a way that it takes review as an input and returns the corresponding word vector of the review. In this way our dataset is converted into 1411x300 shape where 300 is the dimension of each word vector.

Loss function - Binary Cross Entropy loss (BCE)
optimizer - Adam


Accuracy:
       Number of epochs used = 255
       Minibatch size=100
       Accuracy = 0.95053 (defined as number of correct prediction / total number of examples)
