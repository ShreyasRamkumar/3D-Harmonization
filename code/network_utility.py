from torch import optim, nn, ones, sqrt, mean, std, abs
import torch.nn.functional as F
from tqdm import tqdm

class Network_Utility():
    @staticmethod
    def conv_3d(in_c, out_c):
        run = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.BatchNorm3d(num_features=out_c)
        )
        return run

    @staticmethod
    def down_conv_3d(in_c, out_c):
        run = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_c)
        )
        return run
        
    @staticmethod
    def up_conv_3d(in_c, out_c):
        run = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_c),
        )
        return run

    @staticmethod
    def final_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=1),
            nn.ReLU()
        )
        return run
    
    @staticmethod
    def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]
    
    @staticmethod
    def create_data_splits(dataset_len):
        training_len = int(dataset_len * 0.8)
        validation_len = int((dataset_len - training_len) / 2)
        return [training_len, validation_len, validation_len]