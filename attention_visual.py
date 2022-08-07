import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import DCGenerator

torch.manual_seed(69)
fixed_z = torch.randn(1, 512)

