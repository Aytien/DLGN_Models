# This is an example configuration file for the VN model.
# Copy this dictionary and pass it as an argument to the train_model function in the main script.
config = {
    'device' : device,
    'model_type' : ModelTypes.VN,
    'num_data' : len(data['train_data']),
    'dim_in' : data_config.dim_in,
    'num_hidden_nodes' : [500]*4, 
    'beta' : 10,
    'loss_fn_type' : LossTypes.CE,
    'optimizer_type' : Optim.ADAM,
    'mode' : 'pwc',
    'lr_ratio' : 10000,
    'log_features' : False,
    'lr' : 0.001,
    'epochs' : 1000,
    'use_wandb' : False
}