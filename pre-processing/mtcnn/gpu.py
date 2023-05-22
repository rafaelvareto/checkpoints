"""gpu nn utils"""

from typing import List, Dict, Tuple
from torch.backends import cudnn
from torch import nn
import torch


def num_devices() -> int:
    """num_devices"""
    return torch.cuda.device_count()

def devices_capabilities(devices: List[int] = None) -> Dict[int, Tuple[int, int]]:
    """devices_capabilities"""
    devices = devices if devices is not None else list(range(0, num_devices()))
    capabilities: Dict[int, Tuple[int, int]] = dict()
    for device in devices:
        capabilities[device] = torch.cuda.get_device_capability(device)
    return capabilities

def cuda_is_available() -> bool:
    """cuda_is_available"""
    return torch.cuda.is_available()

def get_device(gpu_enable: bool = True, gpu_device: int = 0) -> torch.device:
    """get device"""
    gpu_enable = gpu_enable and torch.cuda.is_available()
    return torch.device(f'cuda:{gpu_device}' if gpu_enable else "cpu")

def init(gpu_enable: bool = True, gpu_device: int = 0) -> torch.device:
    """init"""
    torch.manual_seed(1337)
    gpu_enable = gpu_enable and torch.cuda.is_available()
    if gpu_enable:
        torch.cuda.manual_seed(1337)
        cudnn.enabled = True
        cudnn.benchmark = True
    return get_device(gpu_enable, gpu_device)

def multi_gpu_model(model: nn.Module, gpu_enable: bool = True):
    """multi_gpu_model"""
    gpu_enable = gpu_enable and torch.cuda.is_available()
    if gpu_enable and num_devices() > 1:
        return nn.DataParallel(model)
    return model
