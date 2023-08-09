from src.trainer_v1.model import MNISTNetConfig
from src.utilities import ConvParam

default_model_config = MNISTNetConfig(
    image_width=28,
    image_height=28,
    image_channels=1,
    conv_params=[
        ConvParam(in_channels=1, out_channels=32, kernel_size=3, padding=0, stride=1, pooling=None),
        ConvParam(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=1, pooling=None)
    ],
    dropout=0.25,
    dropout2=0.5,
    fc1_out_features=128,
    fc2_out_features=10
)

default_data_config = {}

default_trainer_config = {
    "cpu_only": False,
    "max_epochs": 14,
    "batch_size": 64,
    "test_batch_size": 1000,
    "optimizer": "adadelta",
    "lr": 1,
    "lr_decay": 0.5,
    "gamma": 0.7,
    "lr_scheduler": "step",
}

default_wandb_config = {
    "project": "mnist",
    "entity": "mdai",
    "group": "MNIST",
    "job_type": "train",
    "sweep": False,
    "sweep_count": 100,
}


default_sweep_config = {
    "method": "grid",  # "bayes",
    "metric": {
        "name": "Test Loss",
        "goal": "minimize"
    },
    "run_cap": 10,
    # "early_terminate": {
    #     "type": "hyperband",
    #     "s": 2 if max_epochs // 9 > 2 else 1,
    #     "max_iter": max_epochs,
    # },
    "parameters": {
        "lr": {
            "values": [1, 0.1, 0.01, 0.001, 0.0001]
        },
        "batch_size": {
            "values": [64, 128, 256, 512]
        },
        "optimizer": {
            "values": ["adadelta", "adam"]
        },
    }
}

