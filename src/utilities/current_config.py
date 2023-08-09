import json
import os
import sys


default_root = os.path.realpath(__file__).split("slurm-pytorch-ddp-boilerplate")[0]
default_project_root = os.path.join(default_root, "slurm-pytorch-ddp-boilerplate")
if default_project_root not in sys.path:
    sys.path.append(default_project_root)

from src.utilities.cryptography import hash_sha256
from src.ddp.ddp_utils import dprint


def get_keys_recursive(dictionary):
    if hasattr(dictionary, "__dict__"):
        dictionary = dictionary.__dict__
    keys = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            keys += get_keys_recursive(v)
        else:
            keys.append(k)
    return keys


def get_repeated_keys(d, exclude=None):
    if hasattr(d, "__dict__"):
        d = d.__dict__
    if not isinstance(d, dict):
        raise ValueError("d must be a dictionary")

    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        if isinstance(exclude, str):
            exclude = [exclude]
        else:
            raise ValueError("exclude must be a list of strings")

    found_keys = []
    repeated_keys = []
    for k, v in d.items():
        if k in exclude:
            continue
        if hasattr(v, "__dict__"):
            v = v.__dict__
        if isinstance(v, dict):
            keys = get_keys_recursive(v)
            for key in keys:
                if key in found_keys:
                    repeated_keys.append(key)
                found_keys.append(key)
    return repeated_keys


class CurrentConfig:
    """
    This class is a singleton that stores the current configuration of the project.
    It is useful to avoid passing around the configuration dictionary to different functions in DDP.

    It also allows for easy access to the configuration dictionary using the [] operator whatever the nesting level.
    Which makes it easy to update when you are doing a wandb sweep (flat config dictionary).
    Due to this, if you have repeated keys in the config, the behaviour of [] operator is not safe.

    It stores explicitly the configuration for the following:
        - root: root directory of the project
        - project_root: project root directory of the project
        - data_config: data configuration
        - model_config: model configuration
        - wandb_config: wandb configuration
        - trainer_config: trainer configuration
        - sweep_config: sweep configuration that is passed to wandb sweep (not the same as the wandb.run.config)

    """
    _instance = None
    _initialized = False

    def __init__(self,
                 root=None,
                 project_root=None,
                 data_config=None,
                 model_config=None,
                 wandb_config=None,
                 trainer_config=None,
                 sweep_config=None,
                 check_repeated_keys=True,
                 stdout=False,
                 **kwargs):
        if not CurrentConfig._initialized:
            if root is None:
                root = default_root
            if project_root is None:
                project_root = default_project_root
            self.root = root
            self.project_root = project_root
            self.data_config = data_config or {}
            self.model_config = model_config or {}
            self.wandb_config = wandb_config or {}
            self.trainer_config = trainer_config or {}
            self.sweep_config = sweep_config or {}
            self.stdout = stdout

            for k, v in kwargs.items():
                if k not in self.__dict__:
                    raise KeyError(f"Key {k} not found in config")
                self.__dict__[k] = v

            # Check for repeated keys
            if check_repeated_keys:
                repeated_keys = get_repeated_keys(self.__dict__, exclude=["sweep_config"])
                if len(repeated_keys) > 0:
                    dprint(
                        "\033[33m" +
                        f"Warning: CurrentConfig class's behaviour is unsafe when repeated keys are present in the config." +
                        f"Do not use direct setitem or getitem methods with those keys e.g. config[key] = v. " +
                        f"Repeated keys in config: {repeated_keys}"
                        + "\033[0m"
                    )

            CurrentConfig._initialized = True

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(CurrentConfig, cls).__new__(cls)
        return cls._instance

    def get_recursive(self, dictionary, key):
        if not isinstance(dictionary, dict):
            # check if it has a __dict__ attribute
            if hasattr(dictionary, "__dict__"):
                dictionary = dictionary.__dict__
        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                if k == key:
                    return v
                if hasattr(v, "__dict__"):
                    v = v.__dict__
                if isinstance(v, dict):
                    result = self.get_recursive(v, key)
                    if result is not None:
                        return result

    def __getitem__(self, item):
        return self.get_recursive(self.__dict__, item)

    def get(self, item, default=None):
        result = self.get_recursive(self.__dict__, item)
        if result is None:
            return default
        return result

    def set_recursive(self, path, key, value):
        dictionary = self.__dict__
        if path != "":
            for p in path.split("."):
                dictionary = dictionary[p]
        if hasattr(dictionary, "__dict__"):
            dictionary = dictionary.__dict__
        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                if k == "sweep_config":
                    # Don't set sweep_config here
                    continue
                if k == key:
                    # set self.__dict__[path][path2]...[key] = value
                    dictionary[k] = value
                    return
                if hasattr(v, "__dict__"):
                    v = v.__dict__
                if isinstance(v, dict):
                    if path == "":
                        new_path = k
                    else:
                        new_path = path + "." + k
                    self.set_recursive(new_path, key, value)

    def __setitem__(self, key, value):
        splits = [k for k in key.split(".") if k != ""]
        path = ".".join(splits[:-1])
        key = splits[-1]
        self.set_recursive(path, key, value)

    def update(self, dictionary, path=""):
        if hasattr(dictionary, "__dict__"):
            dictionary = dictionary.__dict__
        if path is None or not isinstance(path, str):
            raise ValueError("path must be a string")
        if path != "":
            splits = [k for k in path.split(".") if k != ""]
            path = ".".join(splits[:-1])
            if path != "":
                path += "."
        for k, v in dictionary.items():
            self[path + k] = v

    def load(self, file_path, path=""):
        with open(file_path, "r") as f:
            if ".json" in file_path:
                dictionary = json.load(f)
            elif ".yml" in file_path or ".yaml" in file_path:
                import yaml
                dictionary = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError("File extension not supported")
        self.update(dictionary, path)

    def reset(self, **kwargs):
        CurrentConfig._initialized = False
        self.__init__(**kwargs)

    def __str__(self):
        return json.dumps({
            "root": self.root,
            "project_root": self.project_root,
            "data_config": self.data_config,
            "model_config": self.model_config,
            "wandb_config": self.wandb_config,
            "trainer_config": self.trainer_config,
            "sweep_config": self.sweep_config
        }, indent=4)

    def __repr__(self):
        return self.__str__()

    def print(self):
        print(self.__str__())

    @property
    def hash(self):
        return hash_sha256(json.dumps(self.__dict__, sort_keys=True))


current_config = CurrentConfig()
