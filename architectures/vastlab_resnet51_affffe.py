import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision._internally_replaced_utils import load_state_dict_from_url

from .utils import model_urls


_weights_dict = dict()

def load_weights(state_file):
    if state_file == None:
        return

    try:
        weights_dict = numpy.load(state_file, allow_pickle=True).item()
    except:
        weights_dict = numpy.load(state_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class Resnet51_afffe(nn.Module):
    def __init__(self, state_dict=None, state_file=None):
        super(Resnet51_afffe, self).__init__()
        global _counter_var
        global _weights_dict

        if state_dict is None and state_file is None:
            raise ValueError(f'You must provide either state_dict or state_file for the constructor.')

        if   state_dict is not None:
            _weights_dict = state_dict
        elif state_file is not None:
            _weights_dict = load_weights(state_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.bn_conv1 = self.__batch_normalization(2, 'bn_conv1', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch1 = self.__conv(2, name='res2a_branch1', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.res2a_branch2a = self.__conv(2, name='res2a_branch2a', in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch1 = self.__batch_normalization(2, 'bn2a_branch1', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.bn2a_branch2a = self.__batch_normalization(2, 'bn2a_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2b = self.__conv(2, name='res2a_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2b = self.__batch_normalization(2, 'bn2a_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2c = self.__conv(2, name='res2a_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2c = self.__batch_normalization(2, 'bn2a_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2a = self.__conv(2, name='res2b_branch2a', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2a = self.__batch_normalization(2, 'bn2b_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2b = self.__conv(2, name='res2b_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2b = self.__batch_normalization(2, 'bn2b_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2c = self.__conv(2, name='res2b_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2c = self.__batch_normalization(2, 'bn2b_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2a = self.__conv(2, name='res2c_branch2a', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2a = self.__batch_normalization(2, 'bn2c_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2b = self.__conv(2, name='res2c_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2b = self.__batch_normalization(2, 'bn2c_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2c = self.__conv(2, name='res2c_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2c = self.__batch_normalization(2, 'bn2c_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch1 = self.__conv(2, name='res3a_branch1', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res3a_branch2a = self.__conv(2, name='res3a_branch2a', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn3a_branch1 = self.__batch_normalization(2, 'bn3a_branch1', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.bn3a_branch2a = self.__batch_normalization(2, 'bn3a_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2b = self.__conv(2, name='res3a_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2b = self.__batch_normalization(2, 'bn3a_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2c = self.__conv(2, name='res3a_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2c = self.__batch_normalization(2, 'bn3a_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2a = self.__conv(2, name='res3b_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2a = self.__batch_normalization(2, 'bn3b_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2b = self.__conv(2, name='res3b_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2b = self.__batch_normalization(2, 'bn3b_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2c = self.__conv(2, name='res3b_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2c = self.__batch_normalization(2, 'bn3b_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2a = self.__conv(2, name='res3c_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2a = self.__batch_normalization(2, 'bn3c_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2b = self.__conv(2, name='res3c_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2b = self.__batch_normalization(2, 'bn3c_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2c = self.__conv(2, name='res3c_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2c = self.__batch_normalization(2, 'bn3c_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2a = self.__conv(2, name='res3d_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2a = self.__batch_normalization(2, 'bn3d_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2b = self.__conv(2, name='res3d_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2b = self.__batch_normalization(2, 'bn3d_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2c = self.__conv(2, name='res3d_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2c = self.__batch_normalization(2, 'bn3d_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch1 = self.__conv(2, name='res4a_branch1', in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res4a_branch2a = self.__conv(2, name='res4a_branch2a', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn4a_branch1 = self.__batch_normalization(2, 'bn4a_branch1', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.bn4a_branch2a = self.__batch_normalization(2, 'bn4a_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2b = self.__conv(2, name='res4a_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2b = self.__batch_normalization(2, 'bn4a_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2c = self.__conv(2, name='res4a_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2c = self.__batch_normalization(2, 'bn4a_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2a = self.__conv(2, name='res4b_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2a = self.__batch_normalization(2, 'bn4b_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2b = self.__conv(2, name='res4b_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2b = self.__batch_normalization(2, 'bn4b_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2c = self.__conv(2, name='res4b_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2c = self.__batch_normalization(2, 'bn4b_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2a = self.__conv(2, name='res4c_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2a = self.__batch_normalization(2, 'bn4c_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2b = self.__conv(2, name='res4c_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2b = self.__batch_normalization(2, 'bn4c_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2c = self.__conv(2, name='res4c_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2c = self.__batch_normalization(2, 'bn4c_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2a = self.__conv(2, name='res4d_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2a = self.__batch_normalization(2, 'bn4d_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2b = self.__conv(2, name='res4d_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2b = self.__batch_normalization(2, 'bn4d_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2c = self.__conv(2, name='res4d_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2c = self.__batch_normalization(2, 'bn4d_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2a = self.__conv(2, name='res4e_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2a = self.__batch_normalization(2, 'bn4e_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2b = self.__conv(2, name='res4e_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2b = self.__batch_normalization(2, 'bn4e_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2c = self.__conv(2, name='res4e_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2c = self.__batch_normalization(2, 'bn4e_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2a = self.__conv(2, name='res4f_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2a = self.__batch_normalization(2, 'bn4f_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2b = self.__conv(2, name='res4f_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2b = self.__batch_normalization(2, 'bn4f_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2c = self.__conv(2, name='res4f_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2c = self.__batch_normalization(2, 'bn4f_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res5a_branch1 = self.__conv(2, name='res5a_branch1', in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res5a_branch2a = self.__conv(2, name='res5a_branch2a', in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn5a_branch1 = self.__batch_normalization(2, 'bn5a_branch1', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.bn5a_branch2a = self.__batch_normalization(2, 'bn5a_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5a_branch2b = self.__conv(2, name='res5a_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2b = self.__batch_normalization(2, 'bn5a_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5a_branch2c = self.__conv(2, name='res5a_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2c = self.__batch_normalization(2, 'bn5a_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2a = self.__conv(2, name='res5b_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2a = self.__batch_normalization(2, 'bn5b_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2b = self.__conv(2, name='res5b_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2b = self.__batch_normalization(2, 'bn5b_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2c = self.__conv(2, name='res5b_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2c = self.__batch_normalization(2, 'bn5b_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2a = self.__conv(2, name='res5c_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2a = self.__batch_normalization(2, 'bn5c_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2b = self.__conv(2, name='res5c_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2b = self.__batch_normalization(2, 'bn5c_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2c = self.__conv(2, name='res5c_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2c = self.__batch_normalization(2, 'bn5c_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.fc1000_1 = self.__dense(name = 'fc1000_1', in_features = 2048, out_features = 1000, bias = True)
        self.features_1 = self.__dense(name = 'features_1', in_features = 1000, out_features = 1000, bias = True)

    def forward(self, x):
        conv1_pad       = F.pad(x, (3, 3, 3, 3))
        conv1           = self.conv1(conv1_pad)
        bn_conv1        = self.bn_conv1(conv1)
        conv1_relu      = F.relu(bn_conv1)
        pool1_pad       = F.pad(conv1_relu, (0, 1, 0, 1), value=float('-inf'))
        pool1, pool1_idx = F.max_pool2d(pool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        res2a_branch1   = self.res2a_branch1(pool1)
        res2a_branch2a  = self.res2a_branch2a(pool1)
        bn2a_branch1    = self.bn2a_branch1(res2a_branch1)
        bn2a_branch2a   = self.bn2a_branch2a(res2a_branch2a)
        res2a_branch2a_relu = F.relu(bn2a_branch2a)
        res2a_branch2b_pad = F.pad(res2a_branch2a_relu, (1, 1, 1, 1))
        res2a_branch2b  = self.res2a_branch2b(res2a_branch2b_pad)
        bn2a_branch2b   = self.bn2a_branch2b(res2a_branch2b)
        res2a_branch2b_relu = F.relu(bn2a_branch2b)
        res2a_branch2c  = self.res2a_branch2c(res2a_branch2b_relu)
        bn2a_branch2c   = self.bn2a_branch2c(res2a_branch2c)
        res2a           = bn2a_branch1 + bn2a_branch2c
        res2a_relu      = F.relu(res2a)
        res2b_branch2a  = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a   = self.bn2b_branch2a(res2b_branch2a)
        res2b_branch2a_relu = F.relu(bn2b_branch2a)
        res2b_branch2b_pad = F.pad(res2b_branch2a_relu, (1, 1, 1, 1))
        res2b_branch2b  = self.res2b_branch2b(res2b_branch2b_pad)
        bn2b_branch2b   = self.bn2b_branch2b(res2b_branch2b)
        res2b_branch2b_relu = F.relu(bn2b_branch2b)
        res2b_branch2c  = self.res2b_branch2c(res2b_branch2b_relu)
        bn2b_branch2c   = self.bn2b_branch2c(res2b_branch2c)
        res2b           = res2a_relu + bn2b_branch2c
        res2b_relu      = F.relu(res2b)
        res2c_branch2a  = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a   = self.bn2c_branch2a(res2c_branch2a)
        res2c_branch2a_relu = F.relu(bn2c_branch2a)
        res2c_branch2b_pad = F.pad(res2c_branch2a_relu, (1, 1, 1, 1))
        res2c_branch2b  = self.res2c_branch2b(res2c_branch2b_pad)
        bn2c_branch2b   = self.bn2c_branch2b(res2c_branch2b)
        res2c_branch2b_relu = F.relu(bn2c_branch2b)
        res2c_branch2c  = self.res2c_branch2c(res2c_branch2b_relu)
        bn2c_branch2c   = self.bn2c_branch2c(res2c_branch2c)
        res2c           = res2b_relu + bn2c_branch2c
        res2c_relu      = F.relu(res2c)
        res3a_branch1   = self.res3a_branch1(res2c_relu)
        res3a_branch2a  = self.res3a_branch2a(res2c_relu)
        bn3a_branch1    = self.bn3a_branch1(res3a_branch1)
        bn3a_branch2a   = self.bn3a_branch2a(res3a_branch2a)
        res3a_branch2a_relu = F.relu(bn3a_branch2a)
        res3a_branch2b_pad = F.pad(res3a_branch2a_relu, (1, 1, 1, 1))
        res3a_branch2b  = self.res3a_branch2b(res3a_branch2b_pad)
        bn3a_branch2b   = self.bn3a_branch2b(res3a_branch2b)
        res3a_branch2b_relu = F.relu(bn3a_branch2b)
        res3a_branch2c  = self.res3a_branch2c(res3a_branch2b_relu)
        bn3a_branch2c   = self.bn3a_branch2c(res3a_branch2c)
        res3a           = bn3a_branch1 + bn3a_branch2c
        res3a_relu      = F.relu(res3a)
        res3b_branch2a  = self.res3b_branch2a(res3a_relu)
        bn3b_branch2a   = self.bn3b_branch2a(res3b_branch2a)
        res3b_branch2a_relu = F.relu(bn3b_branch2a)
        res3b_branch2b_pad = F.pad(res3b_branch2a_relu, (1, 1, 1, 1))
        res3b_branch2b  = self.res3b_branch2b(res3b_branch2b_pad)
        bn3b_branch2b   = self.bn3b_branch2b(res3b_branch2b)
        res3b_branch2b_relu = F.relu(bn3b_branch2b)
        res3b_branch2c  = self.res3b_branch2c(res3b_branch2b_relu)
        bn3b_branch2c   = self.bn3b_branch2c(res3b_branch2c)
        res3b           = res3a_relu + bn3b_branch2c
        res3b_relu      = F.relu(res3b)
        res3c_branch2a  = self.res3c_branch2a(res3b_relu)
        bn3c_branch2a   = self.bn3c_branch2a(res3c_branch2a)
        res3c_branch2a_relu = F.relu(bn3c_branch2a)
        res3c_branch2b_pad = F.pad(res3c_branch2a_relu, (1, 1, 1, 1))
        res3c_branch2b  = self.res3c_branch2b(res3c_branch2b_pad)
        bn3c_branch2b   = self.bn3c_branch2b(res3c_branch2b)
        res3c_branch2b_relu = F.relu(bn3c_branch2b)
        res3c_branch2c  = self.res3c_branch2c(res3c_branch2b_relu)
        bn3c_branch2c   = self.bn3c_branch2c(res3c_branch2c)
        res3c           = res3b_relu + bn3c_branch2c
        res3c_relu      = F.relu(res3c)
        res3d_branch2a  = self.res3d_branch2a(res3c_relu)
        bn3d_branch2a   = self.bn3d_branch2a(res3d_branch2a)
        res3d_branch2a_relu = F.relu(bn3d_branch2a)
        res3d_branch2b_pad = F.pad(res3d_branch2a_relu, (1, 1, 1, 1))
        res3d_branch2b  = self.res3d_branch2b(res3d_branch2b_pad)
        bn3d_branch2b   = self.bn3d_branch2b(res3d_branch2b)
        res3d_branch2b_relu = F.relu(bn3d_branch2b)
        res3d_branch2c  = self.res3d_branch2c(res3d_branch2b_relu)
        bn3d_branch2c   = self.bn3d_branch2c(res3d_branch2c)
        res3d           = res3c_relu + bn3d_branch2c
        res3d_relu      = F.relu(res3d)
        res4a_branch1   = self.res4a_branch1(res3d_relu)
        res4a_branch2a  = self.res4a_branch2a(res3d_relu)
        bn4a_branch1    = self.bn4a_branch1(res4a_branch1)
        bn4a_branch2a   = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a_relu = F.relu(bn4a_branch2a)
        res4a_branch2b_pad = F.pad(res4a_branch2a_relu, (1, 1, 1, 1))
        res4a_branch2b  = self.res4a_branch2b(res4a_branch2b_pad)
        bn4a_branch2b   = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b_relu = F.relu(bn4a_branch2b)
        res4a_branch2c  = self.res4a_branch2c(res4a_branch2b_relu)
        bn4a_branch2c   = self.bn4a_branch2c(res4a_branch2c)
        res4a           = bn4a_branch1 + bn4a_branch2c
        res4a_relu      = F.relu(res4a)
        res4b_branch2a  = self.res4b_branch2a(res4a_relu)
        bn4b_branch2a   = self.bn4b_branch2a(res4b_branch2a)
        res4b_branch2a_relu = F.relu(bn4b_branch2a)
        res4b_branch2b_pad = F.pad(res4b_branch2a_relu, (1, 1, 1, 1))
        res4b_branch2b  = self.res4b_branch2b(res4b_branch2b_pad)
        bn4b_branch2b   = self.bn4b_branch2b(res4b_branch2b)
        res4b_branch2b_relu = F.relu(bn4b_branch2b)
        res4b_branch2c  = self.res4b_branch2c(res4b_branch2b_relu)
        bn4b_branch2c   = self.bn4b_branch2c(res4b_branch2c)
        res4b           = res4a_relu + bn4b_branch2c
        res4b_relu      = F.relu(res4b)
        res4c_branch2a  = self.res4c_branch2a(res4b_relu)
        bn4c_branch2a   = self.bn4c_branch2a(res4c_branch2a)
        res4c_branch2a_relu = F.relu(bn4c_branch2a)
        res4c_branch2b_pad = F.pad(res4c_branch2a_relu, (1, 1, 1, 1))
        res4c_branch2b  = self.res4c_branch2b(res4c_branch2b_pad)
        bn4c_branch2b   = self.bn4c_branch2b(res4c_branch2b)
        res4c_branch2b_relu = F.relu(bn4c_branch2b)
        res4c_branch2c  = self.res4c_branch2c(res4c_branch2b_relu)
        bn4c_branch2c   = self.bn4c_branch2c(res4c_branch2c)
        res4c           = res4b_relu + bn4c_branch2c
        res4c_relu      = F.relu(res4c)
        res4d_branch2a  = self.res4d_branch2a(res4c_relu)
        bn4d_branch2a   = self.bn4d_branch2a(res4d_branch2a)
        res4d_branch2a_relu = F.relu(bn4d_branch2a)
        res4d_branch2b_pad = F.pad(res4d_branch2a_relu, (1, 1, 1, 1))
        res4d_branch2b  = self.res4d_branch2b(res4d_branch2b_pad)
        bn4d_branch2b   = self.bn4d_branch2b(res4d_branch2b)
        res4d_branch2b_relu = F.relu(bn4d_branch2b)
        res4d_branch2c  = self.res4d_branch2c(res4d_branch2b_relu)
        bn4d_branch2c   = self.bn4d_branch2c(res4d_branch2c)
        res4d           = res4c_relu + bn4d_branch2c
        res4d_relu      = F.relu(res4d)
        res4e_branch2a  = self.res4e_branch2a(res4d_relu)
        bn4e_branch2a   = self.bn4e_branch2a(res4e_branch2a)
        res4e_branch2a_relu = F.relu(bn4e_branch2a)
        res4e_branch2b_pad = F.pad(res4e_branch2a_relu, (1, 1, 1, 1))
        res4e_branch2b  = self.res4e_branch2b(res4e_branch2b_pad)
        bn4e_branch2b   = self.bn4e_branch2b(res4e_branch2b)
        res4e_branch2b_relu = F.relu(bn4e_branch2b)
        res4e_branch2c  = self.res4e_branch2c(res4e_branch2b_relu)
        bn4e_branch2c   = self.bn4e_branch2c(res4e_branch2c)
        res4e           = res4d_relu + bn4e_branch2c
        res4e_relu      = F.relu(res4e)
        res4f_branch2a  = self.res4f_branch2a(res4e_relu)
        bn4f_branch2a   = self.bn4f_branch2a(res4f_branch2a)
        res4f_branch2a_relu = F.relu(bn4f_branch2a)
        res4f_branch2b_pad = F.pad(res4f_branch2a_relu, (1, 1, 1, 1))
        res4f_branch2b  = self.res4f_branch2b(res4f_branch2b_pad)
        bn4f_branch2b   = self.bn4f_branch2b(res4f_branch2b)
        res4f_branch2b_relu = F.relu(bn4f_branch2b)
        res4f_branch2c  = self.res4f_branch2c(res4f_branch2b_relu)
        bn4f_branch2c   = self.bn4f_branch2c(res4f_branch2c)
        res4f           = res4e_relu + bn4f_branch2c
        res4f_relu      = F.relu(res4f)
        res5a_branch1   = self.res5a_branch1(res4f_relu)
        res5a_branch2a  = self.res5a_branch2a(res4f_relu)
        bn5a_branch1    = self.bn5a_branch1(res5a_branch1)
        bn5a_branch2a   = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a_relu = F.relu(bn5a_branch2a)
        res5a_branch2b_pad = F.pad(res5a_branch2a_relu, (1, 1, 1, 1))
        res5a_branch2b  = self.res5a_branch2b(res5a_branch2b_pad)
        bn5a_branch2b   = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b_relu = F.relu(bn5a_branch2b)
        res5a_branch2c  = self.res5a_branch2c(res5a_branch2b_relu)
        bn5a_branch2c   = self.bn5a_branch2c(res5a_branch2c)
        res5a           = bn5a_branch1 + bn5a_branch2c
        res5a_relu      = F.relu(res5a)
        res5b_branch2a  = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a   = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a_relu = F.relu(bn5b_branch2a)
        res5b_branch2b_pad = F.pad(res5b_branch2a_relu, (1, 1, 1, 1))
        res5b_branch2b  = self.res5b_branch2b(res5b_branch2b_pad)
        bn5b_branch2b   = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b_relu = F.relu(bn5b_branch2b)
        res5b_branch2c  = self.res5b_branch2c(res5b_branch2b_relu)
        bn5b_branch2c   = self.bn5b_branch2c(res5b_branch2c)
        res5b           = res5a_relu + bn5b_branch2c
        res5b_relu      = F.relu(res5b)
        res5c_branch2a  = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a   = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a_relu = F.relu(bn5c_branch2a)
        res5c_branch2b_pad = F.pad(res5c_branch2a_relu, (1, 1, 1, 1))
        res5c_branch2b  = self.res5c_branch2b(res5c_branch2b_pad)
        bn5c_branch2b   = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b_relu = F.relu(bn5c_branch2b)
        res5c_branch2c  = self.res5c_branch2c(res5c_branch2b_relu)
        bn5c_branch2c   = self.bn5c_branch2c(res5c_branch2c)
        res5c           = res5b_relu + bn5c_branch2c
        res5c_relu      = F.relu(res5c)
        pool5           = F.avg_pool2d(res5c_relu, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        fc1000_0        = pool5.view(pool5.size(0), -1)
        fc1000_1        = self.fc1000_1(fc1000_0)
        features_0      = fc1000_1.view(fc1000_1.size(0), -1)
        features_1      = self.features_1(features_0)
        
        return features_1


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(_weights_dict[name + '.weight'])
        if (name + '.bias') in _weights_dict:
            layer.state_dict()['bias'].copy_(_weights_dict[name + '.bias'])
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        # if (name + '.scale') in _weights_dict:
        #     layer.state_dict()['weight'].copy_(_weights_dict[name + '.scale'])
        if (name + '.weight') in _weights_dict:
            layer.state_dict()['weight'].copy_(_weights_dict[name + '.weight'])
        else:
            layer.weight.data.fill_(1)

        if (name + '.bias') in _weights_dict:
            layer.state_dict()['bias'].copy_(_weights_dict[name + '.bias'])
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(_weights_dict[name + '.running_mean'])
        layer.state_dict()['running_var'].copy_(_weights_dict[name + '.running_var'])
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(_weights_dict[name + '.weight'])
        if (name + '.bias') in _weights_dict:
            layer.state_dict()['bias'].copy_(_weights_dict[name + '.bias'])
        return layer


def resnet51_afffe(pretrained=False, progress=True, weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    arch = 'vastlab_resnet51_afffe'
    if   weights_path is not None:
        state_dict = torch.load(weights_path)
    elif pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f'No checkpoint is available for model type {arch}')
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model = Resnet51_afffe(state_dict=state_dict, state_file=None)
    return model


if __name__ == '__main__':
    model = resnet51_afffe(pretrained=True, progress=True)
