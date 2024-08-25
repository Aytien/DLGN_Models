# This is an example configuration file for the Kernel model.
# This uses the PEGASOS training method and the Hinge loss function.
# Copy this dictionary and pass it as an argument to the train_model function in the main script.
{
    "device" : device,
    "model_type" : ModelTypes.KERNEL,
    "num_data" : len(data["train_data"]),
    "dim_in" : data_config.dim_in,
    "width" : 128,
    "depth" : 4,
    "beta" : 30,
    "alpha_init" : None,
    "BN" : True,
    "feat" : "cf",
    "weight_decay" : 0.01,
    "train_method" : KernelTrainMethod.PEGASOS,
    "reg" : 0.001,
    "loss_fn_type" : LossTypes.HINGE,
    "optimizer_type" : Optim.ADAM,
    "log_features" : False,
    "gates_lr" : 0.01,
    "alpha_lr" : 0.1,
    "epochs" : 1000,
    "value_freq": 100,
    "num_iter" : 50000,
    "threshold" : 0.3,
    "use_wandb" : False,
}