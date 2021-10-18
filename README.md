# Molecule Classifier

MoleculeClassification is a molecule properties classifier based on neural networks. The application provides a CLI and REST interface and a Docker image ready to be deployed.

![A molecule](https://www.pnglib.com/wp-content/uploads/2020/01/molecule_5e19a406b2242.png)

## Models

Two models have been imagined and implemented with [Keras](https://keras.io/).

The first model exploits a vector of characteristics of molecules and the second model exploits a convolutional neuron network.

For computational power constraints, the selected neural networks are shallow and the number of parameters to be trained is intended to be small.

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

### Model #1 (Fully-Connected Neural Network)

The model exploits a fully-connected neural network that learns the relationship between the ECFP characteristics of molecules and the presence of a certain property.

The ECFP characteristics are defined on a binary vector of 2048 columns. Only columns that take more than one value are kept for dimensionnality-reduction purpose.

#### Neural Network Architecture

The proposed neural network offers three canapes of non-linear activations. Bayesian optimization selects the size of the 2 hidden layers (between 4 and 64 neurons) and the activation function (either relu or swish).

A dropout is added after activations of hidden layers to reduce overfitting. The dropout rate is also found by automated tuning.

![The neural network architecture](https://i.ibb.co/Kskkhby/Untitled-Diagram-drawio.png)

#### Hypermodel learning

Hyperparameters are searched by Bayesian optimization using [keras-tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner). Hyperparameters are selected on the accuracy of consolidated models.

### Model #2 (Convolutional Neural Network)

Le modèle de réseaux de neuronnes convolutionnels s'inspire de l'organisation spatiale de la molécule pour prédire la propriété.

#### Vectorization 



## Setup

The installation of the application is done using pip from the root of the repository:

    pip install .

## CLI Usage

When the application is installed, it can be used by means of the "servier" command.

Train a model:

    servier train --input-path dataset_single.csv --model 1

Evaluate the model:

    servier evaluate --input-path dataset_single_test.csv --model 1

Make predictions:

    servier predict --input-path dataset_single_test.csv --model 1

## API Usage

Une API REST expose le module de prédiction. Pour cela, il faut instancier le serveur HTTP:

    servier train --input-path dataset_single.csv --model 1
    servier serve
 
 L'endpoint exposé est le suivant:

    GET /predict?smiles=YOUR_ENCODED_SMILES_HERE

You should encode the SMILES:

| Before encoding | After encoding  |
|--|--|
| Nc1ccc(C(=O)O)c(O)c1 | Nc1ccc%28C%28%3DO%29O%29c%28O%29c1 |
