# digits_neural_net
Program uses a neural network to classify hand written digits.  


 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Project 5: Machine Learning (due 5/3 at 4:00pm) Version 1.000. Last Updated: 02/16/2019.
Table of Contents
Introduction
Setup
Provided Code (Part I)
Q1: Perceptron
Neural Network Tips Provided Code (Part II) Example: Linear Regression Q2: (Non-linear) Regression Q3: Digit Classification
Q4: Language Identification
In this project you will build a neural network to classify digits, and more!
Introduction
This project will be an introduction to machine learning.
The code for this project contains the following files, available as a zip archive (assets/files/machinelearning.zip).
 https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 1/14

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Note: In this project you only need to submit the token generated by submission_autograder.py . See Submission for details.
Files you'll edit:
Files you should read but NOT edit:
   models.py
  Perceptron and neural network models for a variety of applications
    nn.py
Files you will not edit:
autograder.py backend.py
data submission_autograder.py
Neural network mini-library
Project autograder
Backend code for various machine learning tasks
Datasets for digit classification and language identification Submission autograder (generates tokens for submission)
             Files to Edit and Submit: You will fill in portions of models.py during the assignment. Please do not change the other files in this distribution.
Note: You only need to submit machinelearning.token , generated by running submission_autograder.py . It contains the evaluation results from your local autograder,
and a copy of all your code. You do not need to submit any other files.
Evaluation: Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's judgements -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.
Academic Dishonesty: We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.
Proper Dataset Use: Part of your score for this project will depend on how well the models you train perform on the test set included with the autograder. We do not provide any APIs for you to access the test set directly. Any attempts to bypass this separation or to use the testing data during training will be considered cheating.
Getting Help: You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, section, and the discussion forum are there for your support; please use them. If you can't make our office hours, let us know and we will
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 2/14

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 schedule more. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.
Discussion: Please be careful not to post spoilers. Installation
For this project, you will need to install the following two libraries:
numpy (http://www.numpy.org/) , which provides support for large multi-dimensional arrays - installation instructions (https://docs.scipy.org/doc/numpy- 1.13.0/user/install.html)
matplotlib (https://matplotlib.org/) , a 2D plotting library - installation instructions (https://matplotlib.org/users/installing.html)
You will not be using these libraries directly, but they are required in order to run the provided code and autograder.
To test that everything has been installed, run:
If numpy and matplotlib are installed correctly, you should see a window pop up where a line segment spins in a circle:
Provided Code (Part I)
 python autograder.py --check-dependencies
 https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 3/14

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
For this project, you have been provided with a neural network mini-library ( nn.py ) and a collection of datasets ( backend.py ).
The library in nn.py defines a collection of node objects. Each node represents a real number or a matrix of real numbers. Operations on Node objects are optimized to work faster than using Python's built-in types (such as lists).
Here are a few of the provided node types:
nn.Constant represents a matrix (2D array) of floating point numbers. It is typically used to represent input features or target outputs/labels. Instances of this type will be provided to you by other functions in the API; you will not need to construct them directly
nn.Parameter represents a trainable parameter of a perceptron or neural network nn.DotProduct computes a dot product between its inputs
Additional provided functions:
nn.as_scalar can extract a Python floating-point number from a node.
When training a perceptron or neural network, you will be passed a dataset object. You can retrieve batches of training examples by calling dataset.iterate_once(batch_size) :
For example, let's extract a batch of size 1 (i.e. a single training example) from the perceptron training data:
 for x, y in dataset.iterate_once(batch_size): ...
 >>>
>>>
...
...
...
...
<Constant shape=1x3 at 0x11a8856a0> <Constant shape=1x1 at 0x11a89efd0>
batch_size = 1
for x, y in dataset.iterate_once(batch_size):
print(x)
print(y)
break
The input features x and the correct label y are provided in the form of nn.Constant nodes. The shape of x will be batch_size x num_features , and the shape of y is
batch_size x num_outputs . Here is an example of computing a dot product of x with itself, first as a node and then as a Python number.
 >>> nn.DotProduct(x, x)
<DotProduct shape=1x1 at 0x11a89edd8> >>> nn.as_scalar(nn.DotProduct(x, x)) 1.9756581717465536
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 4/14

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Question 1 (6 points): Perceptron
Before starting this part, be sure you have numpy and matplotlib installed!
In this part, you will implement a binary perceptron. Your task will be to complete the
implementation of the PerceptronModel class in models.py .
For the perceptron, the output labels will be either or , meaning that data points (x, y) from the dataset will have y be a nn.Constant node that contains either or as its entries.
We have already initialized the perceptron weights self.w to be a 1 x dimensions parameter node. The provided code will include a bias feature inside x when needed, so you will not need a separate parameter for the bias.
Your tasks are to:
Implement the run(self, x) method. This should compute the dot product of the stored weight vector and the given input, returning an nn.DotProduct object.
Implement get_prediction(self, x) , which should return if the dot product is non- negative or otherwise. You should use nn.as_scalar to convert a scalar Node into a Python floating-point number.
Write the train(self) method. This should repeatedly loop over the data set and make updates on examples that are misclassified. Use the update method of the
nn.Parameter class to update the weights. When an entire pass over the data set is completed without making any mistakes, 100% training accuracy has been achieved, and training can terminate.
In this project, the only way to change the value of a parameter is by calling parameter.update(direction, multiplier) , which will perform the update to the
weights:
The direction argument is a Node with the same shape as the parameter, and the
multiplier argument is a Python scalar.
To test your implementation, run the autograder:
Note: the autograder should take at most 20 seconds or so to run for a correct implementation. If the autograder is taking forever to run, your code probably has a bug.
 python autograder.py -q q1
Neural Network Tips
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 5/14
1− 1
reilpitlum ⋅ noitcerid + sthgiew ← sthgiew
1
1− 1
1−

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 In the remaining parts of the project, you will implement the following models:
Q2: Regression
Q3: Handwritten Digit Classification Q4: Language Identification
Building Neural Nets
Throughout the applications portion of the project, you'll use the framework provided in nn.py to create neural networks to solve a variety of machine learning problems. A
simple neural network has layers, where each layer performs a linear operation (just like perceptron). Layers are separated by a non-linearity, which allows the network to approximate general functions. We'll use the ReLU operation for our non-linearity, defined as . For example, a simple two-layer neural network for mapping
an input row vector to an output vector
where we have parameter matrices and
learn during gradient descent. will be an
our input vectors , and is the hidden layer size.
to choose any value we want for the hidden size (we will just need to make sure the dimensions of the other matrices and vectors agree so that we can perform the operations). Using a larger hidden size will usually make the network more powerful (able to fit more training data), but can make the network harder to train (since it adds more parameters to all the matrices and vectors we need to learn), or can lead to overfitting on the training data.
We can also create deeper networks by adding more layers, for example a three-layer net:
Note on Batching
For efficiency, you will be required to process whole batches of data at once rather than a single example at a time. This means that instead of a single input row vector with size
, you will be presented with a batch of inputs represented as a matrix . We provide an example for linear regression to demonstrate how a linear layer can be implemented in the batched setting.
Note on Randomness
The parameters of your neural network will be randomly initialized, and data in some tasks will be presented in shuffled order. Due to this randomness, it's possible that you will still occasionally fail some tasks even with a strong architecture -- this is the problem of local optima! This should happen very rarely, though -- if when testing your code you fail the autograder twice in a row for a question, you should explore other architectures.
Practical tips
would be given by the function:
and parameter vectors and to matrix, where is the dimension of
will be a size vector. We are free
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 6/14
2b
1b
2W 1W
Xi×bbi x
3b + 3W ⋅ )2b + 2W ⋅ )1b + 1W ⋅ x(uler(uler = )x(f
h1b hx i h×i 1W
2b + 2W ⋅ )1b + 1W ⋅ x(uler = )x(f
)x(f x
)0 ,x(xam = )x(uler

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Designing neural nets can take some trial and error. Here are some tips to help you along the way:
Be systematic. Keep a log of every architecture you've tried, what the hyperparameters (layer sizes, learning rate, etc.) were, and what the resulting performance was. As you try more things, you can start seeing patterns about which parameters matter. If you find a bug in your code, be sure to cross out past results that are invalid due to the bug.
Start with a shallow network (just two layers, i.e. one non-linearity). Deeper networks have exponentially more hyperparameter combinations, and getting even a single one wrong can ruin your performance. Use the small network to find a good learning rate and layer size; afterwards you can consider adding more layers of similar size.
If your learning rate is wrong, none of your other hyperparameter choices matter. You can take a state-of-the-art model from a research paper, and change the learning rate such that it performs no better than random. A learning rate too low will result in the model learning too slowly, and a learning rate too high may cause loss to diverge to infinity. Begin by trying different learning rates while looking at how the loss decreases over time. Smaller batches require lower learning rates. When experimenting with different batch sizes, be aware that the best learning rate may be different depending on the batch size. Refrain from making the network too wide (hidden layer sizes too large) If you keep making the network wider accuracy will gradually decline, and computation time will increase quadratically in the layer size -- you're likely to give up due to excessive slowness long before the accuracy falls too much. The full autograder for all parts of the project takes 2-12 minutes to run with staff solutions; if your code is taking much longer you should check it for efficiency.
If your model is returning Infinity or NaN, your learning rate is probably too high for your current architecture.
Recommended values for your hyperparameters:
Hidden layer sizes: between 10 and 400
Batch size: between 1 and the size of the dataset. For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
Learning rate: between 0.001 and 1.0
Number of hidden layers: between 1 and 3
Provided Code (Part II)
Here is a full list of nodes available in nn.py . You will make use of these in the remaining parts of the assignment:
nn.Constant represents a matrix (2D array) of floating point numbers. It is typically used to represent input features or target outputs/labels. Instances of this type will be provided to you by other functions in the API; you will not need to construct them directly
nn.Parameter represents a trainable parameter of a perceptron or neural network. All parameters must be 2-dimensional.
Usage: nn.Parameter(n, m) constructs a parameter with shape nn.Add adds matrices element-wise
Usage: nn.Add(x, y) accepts two nodes of shape batch_size x num_features and
constructs a node that also has shape batch_size x num_features nn.AddBias adds a bias vector to each feature vector
     https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 7/14
m×n

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Usage: nn.AddBias(features, bias) accepts features of shape
batch_size x num_features and bias of shape 1 x num_features , and constructs a
node that has shape batch_size x num_features .
nn.Linear applies a linear transformation (matrix multiplication) to the input
Usage: nn.Linear(features, weights) accepts features of shape
batch_size x num_input_features and weights of shape
num_input_features x num_output_features , and constructs a node that has shape batch_size x num_output_features .
nn.ReLU applies the element-wise Rectified Linear Unit nonlinearity
. This nonlinearity replaces all negative entries in its input with
zeros.
Usage: nn.ReLU(features) , which returns a node with the same shape as the
input .
nn.SquareLoss computes a batched square loss, used for regression problems
Usage: nn.SquareLoss(a, b) , where a and b both have shape batch_size x num_outputs .
nn.SoftmaxLoss computes a batched softmax loss, used for classification problems Usage: nn.SoftmaxLoss(logits, labels) , where logits and labels both have shape batch_size x num_classes . The term "logits" refers to scores produced by a model, where each entry can be an arbitrary real number. The labels, however, must be non-negative and have each row sum to 1. Be sure not to swap the order of the arguments!
Do not use nn.DotProduct for any model other than the perceptron The following methods are available in nn.py :
nn.gradients computes gradients of a loss with respect to provided parameters.
Usage: nn.gradients(loss, [parameter_1, parameter_2, ..., parameter_n]) will return a list [gradient_1, gradient_2, ..., gradient_n] , where each element is an nn.Constant containing the gradient of the loss with respect to a parameter.
nn.as_scalar can extract a Python floating-point number from a loss node. This can be useful to determine when to stop training.
Usage: nn.as_scalar(node) , where node is either a loss node or has shape . The datasets provided also have two additional methods:
dataset.iterate_forever(batch_size) yields an infinite sequences of batches of examples.
dataset.get_validation_accuracy() returns the accuracy of your model on the validation set. This can be useful to determine when to stop training.
Example: Linear Regression
As an example of how the neural network framework works, let's fit a line to a set of data points. We'll start four points of training data constructed using the function
. In batched form, our data is:
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 8/14
1×1
3 + 1x8 + 0x7 = y
)0 ,x(xam = )x(uler

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 and
Suppose the data is provided to us in the form of nn.Constant nodes:
Let's construct and train a model of the form
correctly, we should be able to learn than , , and .
First, we create our trainable parameters. In matrix form, these are: and
Which corresponds to the following code:
Printing them gives:
Next, we compute our model's predictions for y:
. If done
 >>> x
<Constant shape=4x2 at 0x10a30fe80> >>> y
<Constant shape=4x1 at 0x10a30fef0>
 m = nn.Parameter(2, 1) b = nn.Parameter(1, 1)
 >>> m
<Parameter shape=2x1 at 0x112b8b208> >>> b
<Parameter shape=1x1 at 0x112b8beb8>
 xm = nn.Linear(x, m) predicted_y = nn.AddBias(xm, b)
Our goal is to have the predicted y-values match the provided data. In linear regression we do this by minimizing the square loss:
We construct a loss node:
 loss = nn.SquareLoss(predicted_y, y)
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 9/14
))x(f−y()y,x(∑N2 =L 21
3 = b 8 = 1m 7 = 0m b + 1m ⋅ 1x + 0m ⋅ 0x = )x(f
⎦81⎣
⎦1 1⎣
] b [ = B
] 1m [ = M 0m
⎥ 01 ⎢ = Y ⎥11⎢
⎥ 0 1 ⎢ = X ⎥1 0⎢
⎤3⎡
⎤00⎡

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 In our framework, we provide a method that will return the gradients of the loss with respect to the parameters:
Printing the nodes used gives:
 grad_wrt_m, grad_wrt_b = nn.gradients(loss, [m, b])
 >>> xm
<Linear shape=4x1 at 0x11a869588> >>> predicted_y
<AddBias shape=4x1 at 0x11c23aa90> >>> loss
<SquareLoss shape=() at 0x11c23a240> >>> grad_wrt_m
<Constant shape=2x1 at 0x11a8cb160> >>> grad_wrt_b
<Constant shape=1x1 at 0x11a8cb588>
We can then use the update method to update our parameters. Here is an update for m , assuming we have already initialized a multiplier variable based on a suitable learning rate of our choosing:
If we also include an update for b and add a loop to repeatedly perform gradient updates, we will have the full training procedure for linear regression.
Question 2 (6 points): Non-linear Regression
For this question, you will train a neural network to approximate over .
You will need to complete the implementation of the RegressionModel class in models.py . For this problem, a relatively simple architecture should suffice (see Neural
Network Tips for architecture tips.) Use nn.SquareLoss as your loss. Your tasks are to:
Implement RegressionModel.__init__ with any needed initialization
Implement RegressionModel.run to return a batch_size x 1 node that represents your model's prediction.
Implement RegressionModel.get_loss to return a loss for given inputs and target outputs.
Implement RegressionModel.train , which should train your model using gradient-based updates.
 m.update(grad_wrt_m, multiplier)
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 10/14
]π2 ,π2−[ )x(nis

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 There is only a single dataset split for this task, i.e. there is only training data and no validation data or test set. Your implementation will receive full points if it gets a loss of 0.02 or better, averaged across all examples in the dataset. You may use the training loss to determine when to stop training (use nn.as_scalar to convert a loss node to a Python number). Note that it should take the model a few minutes to train.
Question 3 (6 points): Digit Classification
For this question, you will train a network to classify handwritten digits from the MNIST dataset.
Each digit is of size pixels, the values of which are stored in a -dimensional vector of floating point numbers. Each output we provide is a 10-dimensional vector which has zeros in all positions, except for a one in the position corresponding to the correct class of the digit.
Complete the implementation of the DigitClassificationModel class in models.py . The return value from DigitClassificationModel.run() should be a batch_size x 10 node containing scores, where higher scores indicate a higher probability of a digit belonging to a particular class (0-9). You should use nn.SoftmaxLoss as your loss. Do not put a ReLU activation after the last layer of the network.
For both this question and Q4, in addition to training data there is also validation data and a test set. You can use dataset.get_validation_accuracy() to compute validation accuracy for your model, which can be useful when deciding whether to stop training. The test set will be used by the autograder.
To receive points for this question, your model should achieve an accuracy of at least 97% on the test set. For reference, our staff implementation consistently achieves an accuracy of 98% on the validation data after training for around 5 epochs. Note that the test grades you on test accuracy, while you only have access to validation accuracy - so if your validation accuracy meets the 97% threshold, you may still fail the test if your test accuracy does not meet the threshold. Therefore, it may help to set a slightly higher stopping threshold on validation accuracy, such as 97.5% or 98%.
To test your implementation, run the autograder:
Question 4 (7 points): Language Identification
 python autograder.py -q q2
 python autograder.py -q q3
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 11/14
487 82 × 82

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 Language identification is the task of figuring out, given a piece of text, what language the text is written in. For example, your browser might be able to detect if you've visited a page in a foreign language and offer to translate it for you. Here is an example from Chrome (which uses a neural network to implement this feature):
In this project, we're going to build a smaller neural network model that identifies language for one word at a time. Our dataset consists of words in five languages, such as the table below:
   Word
discussed
eternidad
itseänne
paleis
mieszkać
Language
English
Spanish
Finnish
Dutch
Polish
       Different words consist of different numbers of letters, so our model needs to have an architecture that can handle variable-length inputs. Instead of a single input (like in the previous questions), we'll have a separate input for each character in the word:
where is the length of the word. We'll start by applying a network that is just like the feed-forward networks in the previous problems. It accepts its
input and computes an output vector of dimensionality :
Next, we'll combine the output of the previous step with the next letter in the word, generating a vector summary of the the first two letters of the word. To do this, we'll apply a sub-network that accepts a letter and outputs a hidden state, but now also depends on the previous hidden state . We denote this sub-network as .
This pattern continues for all letters in the input word, where the hidden state at each step summarizes all the letters the network has processed thus far:
Throughout these computations, the function is the same piece of neural network and uses the same trainable parameters; will also share some of the same parameters as . In this way, the parameters used when processing words of different length are all shared. You can implement this using a for loop over the provided inputs xs , where each iteration of the loop computes either or .
The technique described above is called a Recurrent Neural Network (RNN). A schematic diagram of the RNN is shown below:
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 12/14
x
f laitinif
laitinif )⋅,⋅(f
... )2x,2h(f = 3h
)1x,1h(f = 2h
f 1h
)0x(laitinif = 1h
d 1h 0x laitinif
)⋅,⋅(f
L 1−Lx,...,1x,0x

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
  Here, an RNN is used to encode the word "cat" into a fixed-size vector .
After the RNN has processed the full length of the input, it has encoded the arbitrary- length input word into a fixed-size vector , where is the length of the word. This vector summary of the input word can now be fed through additional output layers to generate classification scores for the word's language identity.
Batching
Although the above equations are in terms of a single word, in practice you must use batches of words for efficiency reasons. For simplicity, our code in the project ensures that all words within a single batch have the same length. In batched form, a hidden state
is replaced with the matrix of dimensionality . Design Tips
The design of the recurrent function is the primary challenge for this task. Here are some tips:
Start with a feed-forward architecture of your choice, as long as it has at least one non-linearity.
You should use the following method of constructing given . The first layer of will begin by multiplying the vector by some weight matrix to produce . For subsequent letters, you should replace this computation with
using an nn.Add operation. In other words, you should replace a computation of the form z = nn.Linear(x, W) with a computation of the form z =
nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden)) .
If done correctly, the resulting function will be non-linear in both and
The hidden size should be sufficiently large
Start with a shallow network for , and figure out good values for the hidden size and learning rate before you make the network deeper. If you start with a deep network right away you will have exponentially more hyperparameter combinations, and getting any single hyperparameter wrong can cause your performance to suffer dramatically.
Your task
Complete the implementation of the LanguageIDModel class.
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 13/14
3h
f hxf
d
W )x(laitinif
0x
W ⋅ 0x = 0z laitinif
)x ,h(f
)x(laitinif )x,h(f
d × ezis_hctab iH
ih
L Lh
neddihWih + Wix = iz

 4/21/2019 Project 5 - Machine Learning - CS 188: Introduction to Artificial Intelligence, Spring 2019
 To receive full points on this problem, your architecture should be able to achieve an accuracy of at least 81% on the test set.
To test your implementation, run the autograder:
Disclaimer: this dataset was generated using automated text processing. It may contain errors. It has also not been filtered for profanity. However, our reference implementation can still correctly classify over 89% of the validation set despite the limitations of the data. Our reference implementation takes 10-20 epochs to train.
Submission
Submit machinelearning.token , generated by running submission_autograder.py , to Project 5 on Gradescope.
The full project autograder takes 2-12 minutes to run for the staff reference solutions to the project. If your code takes significantly longer, consider checking your implementations for efficiency.
Note: You only need to submit machinelearning.token , generated by running submission_autograder.py . It contains the evaluation results from your local autograder,
and a copy of all your code. You do not need to submit any other files.
Please specify any partner you may have worked with and verify that both you and your partner are associated with the submission after submitting.
Congratulations! This was the last project for CS 188!
 python autograder.py -q q4
 python3 submission_autograder.py
https://inst.eecs.berkeley.edu/~cs188/sp19/project5.html 14/14
