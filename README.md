# Molecule Classifier

MoleculeClassification is a molecule properties classifier based on neural networks. The application provides a CLI and REST interface and a Docker image ready to be deployed. The repository implements two predictive models and the README gives a benchmark of performance and computing time.

## Table of contents
1. [Models](#models)
    1. [Evaluation metric & loss function](#evaluation-metric--loss-function)
    2. [Data Augmentation](#data-augmentation)
    3. [Balancing the dataset](#balancing-the-dataset)
    4. [Cross-Validation](#cross-validation)
    5. [Model #1 (Fully-Connected Neural Network)](#model-1-fully-connected-neural-network)
    6. [Model #2 (Convolutional Neural Network)](#model-2-convolutional-neural-network)
    7. [Benchmark](#benchmark)
2. [Setup](#setup)
	3. [Building from sources](#building-from-sources)
	4. [Building a docker image](#building-a-docker-image)
3. [CLI Usage](#cli-usage)
4. [API Usage](#api-usage)
5. [Docker Usage](#docker-usage)
	1. [Shared Directory](#shared-directory)
	2. [Environnement variables](#environnement-variables)
	3. [Starting the container](#starting-the-container)

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

![The cross validation strategy in image](https://i.ibb.co/RQ5b44y/Untitled-Diagram-drawio-2.png)

For each set of hyperparameters, the model is trained 4 times using 3 folds for training (in yellow) and one fold for validation (in green). The accuracy of the model is given by the average accuracy of the validation sets.

### Model #1 (Fully-Connected Neural Network)

The model exploits a fully-connected neural network that learns the relationship between the ECFP characteristics of molecules and the presence of a certain property.

The ECFP characteristics are defined on a binary vector of 2048 columns. Only columns that take more than one value are kept for dimensionnality-reduction purpose.

#### Neural Network Architecture

The proposed neural network offers three layers of non-linear activations. Bayesian optimization selects the size of the 2 hidden layers (between 4 and 64 neurons) and the activation function (either relu or swish).

A dropout is added after activations of hidden layers to reduce overfitting. The dropout rate is also found by automated tuning.

![The neural network architecture](https://i.ibb.co/bswdFRG/Untitled-Diagram-drawio-8.png)

#### Hypermodel learning

Hyperparameters are searched by Bayesian optimization using [keras-tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner). Hyperparameters are selected on the accuracy of consolidated models.

#### Accuracy results

Depending on the number of hyperparameter trials, the model scores between 60-70%. This accuracy should be improved, and this is the objective of the second model.

### Model #2 (Convolutional Neural Network)

The convolutional neural network model is inspired by the spatial organization of the molecule to predict the property. 

#### Vectorization 

We represent a molecule by a matrix of fixed size. For this, we one-hot-encode the SMILES thanks to [smiles2vec](https://arxiv.org/abs/1712.02034). 

For the two SMILES here (they are not real molecules), we create this representation:

![smile2vec in action](https://i.ibb.co/SRpxpbY/Untitled-Diagram-drawio-9.png)

The matrix has as many columns as the vocabulary of SMILES (counting atoms, parentheses or even chemical bonds) and as many rows as the size of the largest SMILES in the training dataset. 

The matrix is used to find convolution filters that detect structures within molecules.

Here is an example of a filter (with stride = 1 and filter size = 2) we move on one vectorized input:

![smiles2vec convolution](https://i.ibb.co/TRZrP56/Untitled-Diagram-drawio-7.png)

The filter is only applied over the lines with the full width.

#### Neural network architecture

The neural network is similar to the previous one. The final activation function is a sigmoid and there is a fully-connected part (hidden layer 3). The big difference is the succession of two convolutional layers that learn full-width convolution filters. 

![The CNN model](https://i.ibb.co/MR819M9/Untitled-Diagram-drawio-11.png)

The input has a volume (V, C, 1) with :

- V: the size of the vector, defined by default to 90. In the training set used, the longest vector was 75. 90 was defined to support predictions on longer SMILES.
- C: the SMILES vocabulary

#### Hypermodel learning

The filters hyperparameters such as the filter size and the number of kernels is learned with bayesian optimization, the same way it's done in the model #1.

### Benchmark

The predictive performance of the models is tested with a private CSV dataset of 4999 lines. Each line represents a molecule and gives:

- The SMILES of the molecule, in textual format (column "smiles")
- The presence or not of the property to predict (column "P1")

The dataset is randomly divided into two parts:

- The training set contains 4499 rows (90%) of the initial dataset
- The test set contains 500 rows (10%) of the initial dataset

The cross-validation strategy is applied on the training set. During all the training, the model remains agnostic of the test data. The test set is used only when evaluating the model performance, with the "evaluate" module.

The benchmark is performed on the following configuration:

- **CPU:** Intel(R) Core(TM) i7-10610U CPU @ 1.80GHz   2.30 GHz
- **RAM:** 16,0 Go
- **System type:** 64-bit operating system, x64 processor

The results of the benchmark are as follows:

| Model | Predict time | Training accuracy | Validation accuracy | Test accuracy 
|--|--|--|--|--|
| Dummy Classifier |  |
| #1 (ECFP-FCNN) |  |
| #2 (Smiles-CNN) |  |

| Model | Test F-score | Test Precision | Test Recall 
|--|--|--|--|--|
| Dummy Classifier |  |
| #1 (ECFP-FCNN) |  |
| #2 (Smiles-CNN) |  |



A few notes about the model choices:

- Research shows very good performances with recurrent neural networks, especially of the LTSM type. It would be interesting to use this type of neural networks to replace the proposed model #2. 

- The models are trained without GPU and without using cloud computing resources. The models were chosen to be fast to tune with a single CPU. 

- The models were designed and implemented in approximately 30 hours, under high time constraints. More time would be needed to improve the performance of the models.

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

    servier --input-table dataset_single.csv --model 1 train

Evaluate the model:

    servier --input-table dataset_single_test.csv --model 1 evaluate

Make predictions:

    servier --input-table dataset_single_test.csv --model 1 predict

## API Usage

A REST API exposes the prediction module. For this, the HTTP server must be instantiated:

    servier --input-table dataset_single.csv --model 1 train
    servier serve
 
The exposed endpoint is the following:

    GET /predict?smiles=YOUR_ENCODED_SMILES_HERE

Note: as for any URL request, you should encode the SMILES:

| Before encoding | After encoding  |
|--|--|
| Nc1ccc(C(=O)O)c(O)c1 | Nc1ccc%28C%28%3DO%29O%29c%28O%29c1 |

Only the model 1 is supported for the API usage.

## Docker Usage

⚠️ The docker usage doesn't supports GPU processing unit, since it is untested. For GPU acceleration, please use the CLI (see above).

The Docker image provides a distribution that allows you to run the application. A docker-compose interface allows you to instantiate a container that can perform the same commands described above (CLI section).

### Shared directory

The image does not contain the inputs. A shared mount point is dynamically achieved between the host and the container. Another shared mount point is achieved for the models parameters save.

The input mount point must be created by the user at the root of the repository:

    mkdir data
    mv train.csv data/
    mv test.csv data/

The model mount point is automatically created.

### Environnement variables

The following environment variables should be defined by the user:

| Variable | Description |
|--|--|
| COMMAND | Command to execute (either train, predict, evaluate or serve) |
| CSV | Input table name to use for the training, the evaluation or the prediction. The file should be defined in the *data* directory |
| MODEL | The model to use (either 1 or 2) |
| SMILES | The smiles to predict |

The COMMAND variable must be defined for any execution. Other variables are functions of the COMMAND.

### Starting the container

The container is created the following way:

    # Creating the container
    docker-compose build

Then, it is possible to use it with the following Linux compatible commands:

    # Training from input_train.csv
    COMMAND=train CSV=input_train.csv MODEL=1  docker-compose up
    
    # Evaluating from input_test.csv
    COMMAND=evaluate CSV=input_test.csv MODEL=1  docker-compose up
    
    # Serving
    COMMAND=serve MODEL=1  docker-compose up

