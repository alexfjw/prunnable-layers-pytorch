import torch.nn as nn
from torch.autograd import Variable
import torch


class PConv2d(nn.Conv2d):
    """
    Exactly like a Conv2d, but saves the activation of the last forward pass
    This allows calculation of the taylor estimate in https://arxiv.org/abs/1611.06440
    Includes convenience functions for feature map pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.__recent_activations = None
        self.taylor_estimates = None
        self.register_backward_hook(self.__estimate_taylor_importance)

    def forward(self, x):
        output = super().forward(x)
        self.__recent_activations = output.clone()
        return output

    def __estimate_taylor_importance(self, _, grad_input, grad_output):
        # skip dim=1, its the dim for feature maps
        n_batch, _, n_x, n_y = self.__recent_activations.size()
        n_dimensions = n_batch * n_x * n_y

        estimates = self.__recent_activations.mul_(grad_output[0]) \
            .sum(dim=3) \
            .sum(dim=2) \
            .sum(dim=0) \
            .div_(n_dimensions)

        # normalization
        self.taylor_estimates = torch.abs(estimates) / torch.sqrt(torch.sum(estimates * estimates))
        del estimates, self.__recent_activations
        self.__recent_activations = None

    def prune_feature_map(self, map_index):
        is_cuda = self.weight.is_cuda

        indices = Variable(torch.LongTensor([i for i in range(self.out_channels) if i != map_index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
        self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
        self.out_channels -= 1

    def drop_input_channel(self, index):
        """
        Use when a convnet earlier in the chain is pruned. Reduces input channel count
        :param index:
        :return:
        """
        is_cuda = self.weight.is_cuda

        indices = Variable(torch.LongTensor([i for i in range(self.in_channels) if i != index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(1, indices).data)
        self.in_channels -= 1


class PLinear(nn.Linear):

    def drop_inputs(self, input_shape, index, dim=0):
        """
        Previous layer is expected to be a convnet which just underwent pruning
        Drop cells connected to the pruned layer of the convnet
        :param input_shape: shape of inputs before flattening, should exclude batch_size
        :param index: index to drop
        :param dim: dimension where index is dropped, w.r.t input_shape
        :return:
        """
        is_cuda = self.weight.is_cuda

        reshaped = self.weight.view(-1, *input_shape)
        dim_length = input_shape[dim]
        indices = Variable(torch.LongTensor([i for i in range(dim_length) if i != index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(
            reshaped.index_select(dim+1, indices)
                .data
                .view(self.out_features, -1)
        )
        self.in_features = self.weight.size()[1]


class PBatchNorm2d(nn.BatchNorm2d):

    def drop_input_channel(self, index):
        if self.affine:
            is_cuda = self.weight.is_cuda
            indices = Variable(torch.LongTensor([i for i in range(self.num_features) if i != index]))
            indices = indices.cuda() if is_cuda else indices

            self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
            self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
            self.running_mean = self.running_mean.index_select(0, indices.data)
            self.running_var = self.running_var.index_select(0, indices.data)

        self.num_features -= 1


