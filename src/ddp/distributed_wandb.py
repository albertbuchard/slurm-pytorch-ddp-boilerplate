import os

from dotenv import load_dotenv
from torch import distributed as dist

from src.ddp.ddp_utils import dist_identity, dprint, distribute_str


class DistributedWandb:
    """
    Singleton class to handle distributed wandb.
    Most methods are passed through to wandb, the key difference is that
    only rank 0 will emit to wandb, unless every_process is set to True.

    Also handles wandb sweep and agent with distributed support.
    """
    _instance = None
    _initialized = False

    def __init__(self, every_process=False, key=None, **kwargs):
        if not DistributedWandb._initialized:
            if key is not None:
                os.environ["WANDB_API_KEY"] = key
            self._every_process = every_process
            self.has_logged_in = False
            self.is_emitting = False
            self.project = None
            self.group = None
            if os.getenv("WANDB_API_KEY") is None:
                dprint("WANDB_API_KEY not found in environment. Check your .env file. Disabling Wandb.", flush=True)
            self.setup(**kwargs)

            DistributedWandb._initialized = True

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DistributedWandb, cls).__new__(cls)
        return cls._instance

    def setup(self, project=None, group=None, **kwargs) -> None:
        if dist_identity.rank == 0 or self._every_process:
            if os.getenv("WANDB_API_KEY") is None:
                return
            dprint(f"Initializing wandb with kwargs: {kwargs}", flush=True)
            load_dotenv()
            import wandb as _wandb
            load_dotenv()
            _wandb.login(host=os.getenv("WANDB_BASE_URL", "https://api.wandb.ai"),
                         key=os.getenv("WANDB_API_KEY", None))
            self.has_logged_in = True
            if project is None:
                project = "default"
            if group is None:
                group = f"DDP-{dist_identity.job_id}"
            self.project = project
            self.group = group
            self.is_emitting = True

    def init(self, project=None, group=None, **kwargs) -> 'DistributedWandb':
        if not self.has_logged_in:
            self.setup(project=project, group=group, **kwargs)
        if self.has_logged_in and (dist_identity.rank == 0 or self._every_process):
            dprint(f"Reinitializing wandb with kwargs: {kwargs}", flush=True)
            import wandb as _wandb
            if project is None:
                project = self.project
            else:
                self.project = project
            if group is None:
                group = self.group
            else:
                self.group = group
            _wandb.init(project=project, group=group, **kwargs)
            self.is_emitting = True
        return self

    def finish(self):
        if self.is_emitting:
            import wandb as _wandb
            _wandb.finish()
            self.is_emitting = False
            self.has_logged_in = False

    @property
    def config(self):
        if self.is_emitting:
            import wandb as _wandb
            return _wandb.config
        else:
            return None

    def sweep(self, sweep, project=None, **kwargs):
        sweep_id = None
        if self.is_emitting:
            if project is None:
                project = self.project
            import wandb as _wandb
            sweep_id = _wandb.sweep(sweep=sweep, project=project, **kwargs)
        return distribute_str(sweep_id)

    def agent(self, sweep_id, function, **kwargs):
        if self.is_emitting:
            import wandb as _wandb
            _wandb.agent(sweep_id, function=function, **kwargs)
            dprint("After agent ends")
            dist.destroy_process_group()
        else:
            while True:
                function()
                dprint("Calling function again")

    @property
    def every_process(self):
        return self._every_process

    @every_process.setter
    def every_process(self, value):
        if not isinstance(value, bool):
            raise ValueError("every_process must be a boolean")
        self._every_process = value
        if self._every_process:
            self.setup()

    def log(self, *args, **kwargs):
        if self.is_emitting:
            import wandb as _wandb
            _wandb.log(*args, **kwargs)

    def watch(self, *args, **kwargs):
        if self.is_emitting:
            import wandb as _wandb
            _wandb.watch(*args, **kwargs)

    @property
    def run_dir(self):
        if self.is_emitting:
            import wandb as _wandb
            return _wandb.run.dir
        else:
            return None

    def __getattr__(self, name):
        if self.is_emitting:
            import wandb as _wandb
            if hasattr(_wandb, name):
                return getattr(_wandb, name)
            else:
                raise AttributeError(f"{name} not found in {self.__class__.__name__}")
        return None

    def __setattr__(self, key, value):
        if "is_emitting" in self.__dict__ and self.is_emitting:
            dprint(f"Setting wandb[{key}] to {value}")
            import wandb as _wandb
            if hasattr(_wandb, key):
                setattr(_wandb, key, value)
                return
        super().__setattr__(key, value)

    def __str__(self):
        return f"DistributedWandb({self._initialized})"

    def __repr__(self):
        return self.__str__()

