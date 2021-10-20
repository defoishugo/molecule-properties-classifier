# Molecule Classifier

MoleculeClassification is a molecule properties classifier based on neural networks. The application provides a CLI and REST interface and a Docker image ready to be deployed.

## Table of contents
1. [Models](#models)
    1. [Evaluation metric & loss function](#evaluation-metric--loss-function)
    2. [Data Augmentation](#data-augmentation)
    1. [Balancing the dataset](#balancing-the-dataset)
    1. [Cross-Validation](#cross-validation)
    1. [Model #1 (Fully-Connected Neural Network)](#model-1-fully-connected-neural-network)
    1. [Model #2 (Convolutional Neural Network)](#model-2-convolutional-neural-network)
3. [Setup](#setup)
4. [CLI Usage](#cli-usage)
5. [API Usage](#api-usage)
5. [Docker Usage](#docker-usage)

![A molecule](https://www.pnglib.com/wp-content/uploads/2020/01/molecule_5e19a406b2242.png)
## Models

Two models have been imagined and implemented with [Keras](https://keras.io/).

The first model exploits a vector of characteristics of molecules and the second model exploits a convolutional neuron network.

⚠️ For computational power constraints, the selected neural networks are shallow and the number of parameters to be trained is intended to be small. The goal of models is to be able to train a classifier quickly without a GPU.

### Evaluation metric & loss function

The primary evaluation metric chosen is accuracy. To evaluate the model, we also evaluate the recall and precision. 

The minimized loss function is the binary cross-entropy.

### Data Augmentation

If the number of samples is very low, we apply a data augmentation strategy. 

For each molecule defined in the training set, other SMILES are generated which represent the same molecule.

### Balancing the dataset

When the dataset is badly balanced, a low-sampling strategy is applied (we select as many elements in the majority class as the minority class).

### Cross-Validation

The training data set is split into 4 folds. They will be used for hyperparameter research.

To avoid any bias, the augmented SMILES from the same initial molecule are grouped together within the same fold.

![enter image description here](https://i.ibb.co/RQ5b44y/Untitled-Diagram-drawio-2.png)

For each set of hyperparameters, the model is trained 4 times using 3 folds for training (in yellow) and one fold for validation (in green). The accuracy of the model is given by the average accuracy of the validation sets.

### Model #1 (Fully-Connected Neural Network)

The model exploits a fully-connected neural network that learns the relationship between the ECFP characteristics of molecules and the presence of a certain property.

The ECFP characteristics are defined on a binary vector of 2048 columns. Only columns that take more than one value are kept for dimensionnality-reduction purpose.

#### Neural Network Architecture

The proposed neural network offers three layers of non-linear activations. Bayesian optimization selects the size of the 2 hidden layers (between 4 and 64 neurons) and the activation function (either relu or swish).

A dropout is added after activations of hidden layers to reduce overfitting. The dropout rate is also found by automated tuning.

![The neural network architecture](https://i.ibb.co/YDKQq1j/Untitled-Diagram-drawio-6.png)

#### Hypermodel learning

Hyperparameters are searched by Bayesian optimization using [keras-tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner). Hyperparameters are selected on the accuracy of consolidated models.

#### Accuracy results

Depending on the number of hyperparameter trials, the model scores between 60-70%. This accuracy should be improved, and this is the objective of the second model.

### Model #2 (Convolutional Neural Network)

The convolutional neural network model is inspired by the spatial organization of the molecule to predict the property. 

#### Vectorization 

We represent a molecule by a matrix of fixed size. For this, we one-hot-encode the SMILES thanks to [smiles2vec](https://arxiv.org/abs/1712.02034). 

For the two SMILES here (they are not real molecules), we create this representation:

![smile2vec in action](https://i.ibb.co/d2Z2XsN/Untitled-Diagram-drawio-3.png)
The matrix has as many columns as the vocabulary of SMILES (counting atoms, parentheses or even chemical bonds) and as many rows as the size of the largest SMILES in the training dataset. 

The matrix is used to find convolution filters that detect structures within molecules.

Here is an example of a filter (with stride = 1 and filter size = 2) we move on one vectorized input:

![smiles2vec convolution](https://i.ibb.co/tpHGLgq/Untitled-Diagram-drawio-4.png)

The filter is only applied over the lines with the full width.

#### Neural network architecture

TODO

#### Hypermodel learning

The filters hyperparameters such as the filter size and the number of kernels is learned with bayesian optimization, the same way it's done in the model #1.


## Setup

### Building from sources

The installation of the application is done using pip from the root of the repository:

    pip install .

For development purpose:

    pip install -e .

### Building a Docker image

A Dockerfile is defined and allows the user to create a docker image:

    docker build . -t servier

## CLI Usage

When the application is installed, it can be used by means of the "servier" command.

Train a model:

    servier train --input-path dataset_single.csv --model 1

Evaluate the model:

    servier evaluate --input-path dataset_single_test.csv --model 1

Make predictions:

    servier predict --input-path dataset_single_test.csv --model 1

## API Usage

A REST API exposes the prediction module. For this, the HTTP server must be instantiated:

    servier train --input-path dataset_single.csv --model 1
    servier serve
 
The exposed endpoint is the following:

    GET /predict?smiles=YOUR_ENCODED_SMILES_HERE

Note: as for any URL request, you should encode the SMILES:

| Before encoding | After encoding  |
|--|--|
| Nc1ccc(C(=O)O)c(O)c1 | Nc1ccc%28C%28%3DO%29O%29c%28O%29c1 |

## Docker Usage

TODO: add the container to use the docker image