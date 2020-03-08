# Neural Network from Scratch

## Introduction
A neural network model is implemented from scratch, using only NumPy, to solve a regression prediction problem. The data comes from the UCI Machine Learning Database https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

## Problem
Statement: Neural Network for predicting Bike Sharing Rides. Here NN will predict how many bikes a company needs because if they have too few they are losing money from potential riders and if they have too many they are wasting money on bikes that are just sitting around. So NN will predict from the hisitorical data how many bikes they will need in the near future.

Network Description: The network has two layers, a hidden layer and an output layer. The hidden layer uses the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is f(x)=xf(x)=x . A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called forward propagation. We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called backpropagation.

## Results
- All the code in the notebook runs in Python 3 without failing, and all unit tests pass.
- The number of epochs is chosen such the network is trained well enough to accurately make predictions but is not overfitting to the training data.
- The number of hidden units is chosen such that the network is able to accurately predict the number of bike riders, is able to generalize, and is not overfitting.
- The learning rate is chosen such that the network successfully converges, but is still time efficient.
- The number of output nodes is properly selected to solve the desired problem.
- The training loss is below 0.09 and the validation loss is below 0.18.

## Conclusion
The predictions given by the model are quite accurate. However, the model overestimes bike ridership in December because it hasn't had sufficient holiday season training examples. The model predicts well except for Thanksgiving and Christmas. We don't have enough examples of holidays for the network to learn about them (and the holiday variable is incorrectly labeled for the days around Christmas and Thanksgiving). For dealing with holidays with the small amount of training data available over the holiday time periods, we could use a RNN, time-lagged features or fix the variable for holiday/not holiday (they only label the 22nd and 25th of December a holiday). There's also the possibility of oversampling the data on Christmas and Thanksgiving holidays. The network also over-predicts for Thanksgiving if you look back further into the validation set.
