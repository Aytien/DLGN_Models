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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATASET = "dataset3b"

    DATA_DIR = 'data/' + DATASET
    data = {}
    data['train_data'] = torch.tensor(np.load(DATA_DIR + '/x_train.npy'))
    data['train_labels'] = torch.tensor(np.load(DATA_DIR + '/y_train.npy'))
    data['test_data'] = torch.tensor(np.load(DATA_DIR + '/x_test.npy'))
    data['test_labels'] = torch.tensor(np.load(DATA_DIR + '/y_test.npy'))
    data_config = np.load(DATA_DIR + '/config.npy', allow_pickle=True).item()

    WANDB_NOTEBOOK_NAME = 'DLGN_VT'
    WANDB_PROJECT_NAME = 'DLGN_KERNEL_BTP'
    WANDB_ENTITY = 'cs20b004'
    wandb.login()

    sweep_config = {
        "name": "VT_solvers",
        "method": "grid",
        "parameters": {
            "num_hidden_nodes": {
                "values": [[10]*4]
            },
            "beta": {
                "values": [50]
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
            "prod":{
                "values":['op']
            },
            "lr": {
                "values": [0.001]
            },
            "epochs":{
                "values": [1000]
            },
            "reg": {
                "values": [0.1]
            },
            "value_freq": {
                "values": [100]
            },
            "num_iter": {
                "values": [5e4]
            },
            "weight_decay": {
                "values": [0.01]
            },
            "use_wandb": {
                "values": [True]
            },
            "feat": {
                "values": ['cf']
            },
            "vt_fit1": {
                "values": [2,3,4,5]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME)
    const_config = {
        "device" : device,
        "model_type" : ModelTypes.VT,                    
        "loss_fn_type" : LossTypes.HINGE,
        "value_scale" : 500,
        "optimizer_type" : Optim.ADAM,
        "num_data" : len(data['train_data']),
        "dim_in" : data_config.dim_in,
        "mode" : "pwc",
        "save_freq" : 100,
    }
    def wb_sweep_sf():
        run = wandb.init()
        config = wandb.config

        if config.vt_fit1 == 1:
            const_config["vt_fit"] = VtFit.LOGISTIC
            config.reg = 0.1
        elif config.vt_fit1 == 2:
            const_config["vt_fit"] = VtFit.LINEARSVC
            const_config["loss_fn_type"] = LossTypes.CE
            config.reg = 0.1
        elif config.vt_fit1 == 3:
            const_config["vt_fit"] = VtFit.PEGASOS
            const_config["loss_fn_type"] = LossTypes.HINGE
            config.reg = 0.1
        elif config.vt_fit1 == 4:
            const_config["vt_fit"] = VtFit.NPKSVC
            const_config["loss_fn_type"] = LossTypes.HINGE
            config.reg = 0.001
        elif config.vt_fit1 == 5:
            const_config["vt_fit"] = VtFit.PEGASOSKERNEL
            const_config["loss_fn_type"] = LossTypes.HINGE
            config.reg = 0.1
        config = {**config, **const_config}
        config = Namespace(**config)
        filename_suffx = str(config.vt_fit1)   
        run.name = filename_suffx
        model = train_model(data, config)
        run.finish()
        torch.cuda.empty_cache()
        return
    wandb.agent(sweep_id, wb_sweep_sf, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME)
    wandb.finish()


