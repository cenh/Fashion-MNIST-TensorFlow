# Pruning Fashion MNIST in TensorFlow
This repository is used for solving the task of pruning a neural network in TensorFlow. The example is set up using the Fashion MNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
People have a tendency to simply create larger and larger neural networks to solve their problems. They do not realize that this comes at a cost, computation time and storage. The higher the amount of weights, the more storage your model is going to take up, and it is going to take longer to run and use the model.
This is were pruning comes in handy, with pruning you are able to cut away the weights and not use them anymore. This project shows you can take a model that achieves 88% accuracy and prune away 90% of the weights and still keep a high accuracy (85%).
The idea is very simple, pre-train a large model until it reaches a high accuracy, then prune away unimportant weights and re-train the model over multiple epochs.

## Installation
To run this program you simply need a python 3.x (that works with TensorFlow).

### Python
    pip install -r requirements.txt


## Usage
    python main.py

There are multiple hyperparameters that can be set in the program, some of them are set from the main function, others are set as default values in the functions.

## Results
Using the Fashion MNIST dataset, a model is first trained for 2500 epochs. Then that same model is re-trained with 10% of the weights pruned.

### Pre-trained model
The accuracy of the pre-trained model over epochs.
![alt text](https://github.com/cenh/Fashion-MNIST-TensorFlow/blob/master/results/Unpruned_Fashion_MNIST.png?raw=true "Unpruned model accuracy")

### Pruned model
The pre-trained model, after being trained with 10%, 25%, 50%, and 90% sparsity respectively:
![alt text](https://github.com/cenh/Fashion-MNIST-TensorFlow/blob/master/results/Fashion_MNIST_Sparsity_10.png?raw=true "Model with 10% sparsity")
![alt text](https://github.com/cenh/Fashion-MNIST-TensorFlow/blob/master/results/Fashion_MNIST_Sparsity_25.png?raw=true "Model with 25% sparsity")
![alt text](https://github.com/cenh/Fashion-MNIST-TensorFlow/blob/master/results/Fashion_MNIST_Sparsity_50.png?raw=true "Model with 50% sparsity")
![alt text](https://github.com/cenh/Fashion-MNIST-TensorFlow/blob/master/results/Fashion_MNIST_Sparsity_90.png?raw=true "Model with 90% sparsity")

The red vertical line denotes when the pre-training stop and pruning begins.

## License
The package is Open Source Software released under the [APACHE](LICENSE) license.
