import os
import lightning.pytorch as pl
from torch import optim, nn
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from lightning.pytorch.callbacks import Callback