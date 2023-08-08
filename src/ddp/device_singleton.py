import os

import torch


class DeviceSingleton:
    _instance = None
    _device = None

    def __new__(cls, device="cpu", *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DeviceSingleton, cls).__new__(cls, *args, **kwargs)
            cls._instance.set(device)
        return cls._instance

    def set(self, device):
        if isinstance(device, torch.device):
            self._device = device
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            raise ValueError("Invalid device. Must be an instance of torch.device or a string.")
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        if self._device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def get(self):
        return self._device

    def is_cpu(self):
        return self._device.type == "cpu"

    def is_cuda(self):
        return self._device.type == "cuda"

    def is_mps(self):
        return self._device.type == "mps"

    def print(self):
        print(f"{self._device}")

    def __str__(self):
        return f"{self._device}"

    def __repr__(self):
        return self.__str__()


device = DeviceSingleton("cpu")
