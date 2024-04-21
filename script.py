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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim_in = 20
    tree_depth = 3
    num_points = 10000

    X,Y = gen_spherical_data(depth=tree_depth, dim_in=dim_in, type_data='spherical', num_points=num_points, feat_index_start=0,radius=1)


    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )

    data = {}
    data['train_data'] = x_train
    data['train_labels'] = y_train 
    data['test_data'] = x_test
    data['test_labels'] = y_test

    WANDB_NOTEBOOK_NAME = 'DLGN_Kernel'
    WANDB_PROJECT_NAME = 'DLGN_KERNEL_BTP'
    WANDB_ENTITY = 'cs20b004'
    wandb.login()

    sweep_config = {
        "name": "KernelCompleteBackprop",
        "method": "random",
        "parameters": {
            "num_data": {
                "values": [len(data['train_data'])]
            },
            "dim_in": {
                "values": [dim_in]
            },
            "depth": {
                "values": [4,5]
            },
            "width": {
                "values": [10,15,20]
            },
            "beta": {
                "values": [10,20]
            },
            "alpha_init": {
                "values": [None]
            },
            "log_features": {
                "values": [False]
            },
            "gates_lr": {
                "values": [0.1,0.5]
            },
            "alpha_lr": {
                "values": [0.1,0.5]
            },
            "weight_decay": {
                "values": [0,0.1,0.01]
            },
            "use_wandb": {
                "values": [True]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME)
    const_config = {
        "device" : device,
        "model_type" : ModelTypes.KERNEL,                    
        "loss_fn_type" : LossTypes.BCE,
        "optimizer_type" : None,
        "train_method" : KernelTrainMethod.VANILA,
    }
    def wb_sweep_sf():
        run = wandb.init()
        config = wandb.config
        filename_suffx = str(config.depth) + '_' + str(config.width) + '_' + str(config.beta) + '_' + format(config.gates_lr,".1e") + '_' + format(config.alpha_lr,".1e")  
        run.name = filename_suffx
        config = {**config, **const_config}
        config = Namespace(**config)
        model = train_model(data, config)
        run.finish()
        torch.cuda.empty_cache()
        return
    wandb.agent(sweep_id, wb_sweep_sf, entity=WANDB_ENTITY, project=WANDB_PROJECT_NAME, count=40)
    wandb.finish()

    # config = {
    # "device" : device,
    # "model_type" : ModelTypes.KERNEL,
    # "num_data" : len(data['train_data']),
    # "dim_in" : dim_in,
    # "width" : 4,
    # "depth" : 4,
    # "beta" : 10,
    # "alpha_init" : None,
    # "loss_fn_type" : LossTypes.BCE,
    # "optimizer_type" : None,
    # "train_method" : KernelTrainMethod.VANILA,
    # "log_features" : False,
    # "gates_lr" : 0.1,
    # "alpha_lr" : 0.01,
    # 'weight_decay' : 0.01,
    # "use_wandb" : False,
    # }
    # config = Namespace(**config)
    # print(type(config.model_type))
    # model = train_model(data,config)
