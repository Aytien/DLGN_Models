# This is an example configuration file for the VT model.
# This uses the Logistic training method and the CE loss function.
# Copy this dictionary and pass it as an argument to the train_model function in the main script.
config = {
    'device' : device,
    'model_type' : ModelTypes.VT,
    'num_data' : len(data['train_data']),
    'dim_in' : data_config.dim_in,
    'num_hidden_nodes' : [8]*4,
    'beta' : 30,
    'mode' : 'pwc',
    'value_scale' : 100,
    'BN' : False,
    'prod' : 'op',
    'feat' : 'sf', 
    'loss_fn_type' : LossTypes.CE,
    'optimizer_type' : Optim.ADAM,
    'epochs' : 200,
    'save_freq' : 100,
    'value_freq' : 100,
    'lr': 0.001,
    'vt_fit' : VtFit.LOGISTIC,
    'reg' : 1,
    'use_wandb' : False,
}