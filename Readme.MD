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

## Configs for different Models
### Kernel
- `device` : 'cuda:0'(Based on available devices)
- `model_type` : ModelTypes.KERNEL
- `num_data` : Number of training datapoints
- `dim_in` : Input data dimension
- `width` : Width of a layer
- `depth` : Number of layers
- `beta` : Gating parameter, multiplied  with the gating score before performing sigmoid
- `alpha_init` : Initial values for alpha: ['zero','random']
- `BN` : Enable batch norm, boolean
- `feat` : Decides how features are defined(composit/shallow): ['cf','sf']
- `weight_decay` : Weight decay for the gating network
- `train_method` : ['KernelTrainMethod.PEGASOS', 'KernelTrainMethod.SVC','KernelTrainMethod.GD']
- `reg` : For ['KernelTrainMethod.PEGASOS', 'KernelTrainMethod.SVC'], use as regularization for fitting, has different meaning for both
- `loss_fn_type` : ['LossTypes.HINGE', 'LossTypes.CE']
- `optimizer_type` : ['Optim.ADAM', 'Optim.SGD']
- `gates_lr` : Is the learning rate for gating network
- `alpha_lr` : Only enabled for 'KernelTrainMethod.GD', used for alpha updates
- `epochs` : Number of epochs for training
- `value_freq`: Frequency of value tensor updates
- `num_iter` : For ['KernelTrainMethod.PEGASOS', 'KernelTrainMethod.SVC'], number of optimising iterations
- `threshold` : Threshold for checking proximity of features to dths. Generally use 0.3
- `use_wandb` : For using wandb, boolean

### VN
- `device` : 'cuda:0'(Based on available devices)
- `model_type` : ModelTypes.VN
- `num_data` : Number of training datapoints
- `dim_in` : Input data dimension
- `num_hidden_nodes` : Should be a list of the form: [Width]*Depth
- `beta` : Gating parameter, multiplied  with the gating score before performing sigmoid
- `BN` : Enable batch norm, boolean
- `mode` : "pwc"
- `reg` : For ['KernelTrainMethod.PEGASOS', 'KernelTrainMethod.SVC'], use as regularization for fitting, has different meaning for both
- `loss_fn_type` : 'LossTypes.CE'
- `optimizer_type` : ['Optim.ADAM', 'Optim.SGD']
- `lr` : Is the learning rate for gating network
- `lr_ratio` : Changes value network updates by a factor: value_lr = lr/lr_ratio 
- `epochs` : Number of epochs for training                
- `use_wandb` : For using wandb, boolean

### VT
- `device` : 'cuda:0'(Based on available devices)
- `model_type` : ModelTypes.VT
- `num_data` : Number of training datapoints
- `dim_in` : Input data dimension
- `num_hidden_nodes` : Should be a list of the form: [Width]*Depth
- `beta` : Gating parameter, multiplied  with the gating score before performing sigmoid
- `value_scale` : Scales randn function for tensor initialization
- `BN` : Enable batch norm, boolean
- `mode` : "pwc"
- `prod` : Determine how the gating network output is computed: ['op','ip']
- `vt_fit` : ['KernelTrainMethod.PEGASOS', 'KernelTrainMethod.SVC','KernelTrainMethod.LOGISTIC','KernelTrainMethod.PEGASOSKERNEL', 'KernelTrainMethod.NPKSVC','KernelTrainMethod.LINEARSVC']
- `reg` : For different fit methods, use as regularization for fitting, has different meaning so check the example configs
- `feat` : Decides how features are defined(composit/shallow): ['cf','sf']
- `loss_fn_type` : ['LossTypes.HINGE', 'LossTypes.CE']
- `optimizer_type` : ['Optim.ADAM', 'Optim.SGD']
- `lr` : Is the learning rate for gating network
- `epochs` : Number of epochs for training
- `value_freq`: Frequency of value tensor updates
- `save_freq` : Frequency of saving the model to disk
- `use_wandb` : For using wandb, boolean