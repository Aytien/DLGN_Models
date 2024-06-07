# DLGN Models
This Repository contains code to train and evaluate all three DLGN models : Value Network(VN), Value Tensor(VT), Kernel. 

## Code Structure
- `training_methods.py/` : Contains code to train and evaluate all three DLGN models.
- `data_gen.py/` : Contains code to generate synthetic data for training and evaluation.
- `DLGN_VN.py/` : Contains code for the Value Network model.
- `DLGN_VT.py/` : Contains code for the Value Tensor model.
- `DLGN_Kernel.py/` : Contains code for the Kernel model.
- `DLGN_enums.py/` : Contains enums used in the code.
- `/data/` : Contains synthetic datasets for training and evaluation, these are used for my experiments.

## Training
To train the models, import `training_methods.py` and call the `train_model` function with the following parameters:
- `data` : The dataset to train on. This is a dictionary. The dataset should have the following keys:
    - `train_data` : The training data.
    - `train_labels` : The training labels.
    - `test_data` : The testing data.
    - `test_labels` : The testing labels.
- `config` : The configuration for the model. These are dictionaries/namespaces and can be copy pasted from the file configs.txt. Appropriately set the values according to each model and the data. These help in setting the hyperparameters for the model.

This function returns the trained model and the training loss.

An example use case is present in train_models.ipynb.
