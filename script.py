from training_methods import train_model
from DLGN_kernel import *
from argparse import Namespace
from data_gen import *
from sklearn.model_selection import train_test_split
from DLGN_enums import *
import wandb

if __name__ == "__main__":
    # Set the random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set the device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    DATASET = "dataset3b"

    DATA_DIR = 'data/' + DATASET
    data = {}
    data['train_data'] = torch.tensor(np.load(DATA_DIR + '/x_train.npy'))
    data['train_labels'] = torch.tensor(np.load(DATA_DIR + '/y_train.npy'))
    data['test_data'] = torch.tensor(np.load(DATA_DIR + '/x_test.npy'))
    data['test_labels'] = torch.tensor(np.load(DATA_DIR + '/y_test.npy'))
    data_config = np.load(DATA_DIR + '/config.npy', allow_pickle=True).item()

    WANDB_NOTEBOOK_NAME = 'DLGN_Kernel'
    WANDB_PROJECT_NAME = 'DLGN_KERNEL_BTP'
    WANDB_ENTITY = 'cs20b004'
    wandb.login()

    sweep_config = {
        "name": "KernelPegasos_D1_value_freq",
        "method": "grid",
        "parameters": {
            "depth": {
                "values": [5]
            },
            "width": {
                "values": [128]
            },
            "beta": {
                "values": [30]
            },
            "alpha_init": {
                "values": [None]
            },
            "log_features": {
                "values": [False]
            },
            "BN": {
                "values": [True]
            },
            "gates_lr": {
                "values": [0.001]
            },
            "epochs":{
                "values": [1000]
            },
            "reg": {
                "values": [0.0005]
            },
            "value_freq": {
                "values": [25]
            },
            "num_iter": {
                "values": [5e4]
            },
            "weight_decay": {
                "values": [0.01]
            },
            "threshold":{
                "values": [0.3]
            },
            "use_wandb": {
                "values": [True]
            },
            "feat": {
                "values": ['cf']
            },
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME)
    const_config = {
        "device" : device,
        "model_type" : ModelTypes.KERNEL,                    
        "loss_fn_type" : LossTypes.HINGE,
        "optimizer_type" : Optim.ADAM,
        "train_method" :KernelTrainMethod.PEGASOS,
        "num_data" : len(data['train_data']),
        "dim_in": data_config.dim_in,
    }
    def wb_sweep_sf():
        run = wandb.init()
        config = wandb.config
        filename_suffx = str(config.value_freq) 
        run.name = filename_suffx
        config = {**config, **const_config}
        config = Namespace(**config)
        model = train_model(data, config)
        run.finish()
        torch.cuda.empty_cache()
        return
    wandb.agent(sweep_id, wb_sweep_sf, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME)
    wandb.finish()


