import torch
from torch.autograd import Variable
import torch.nn as nn
import src.nn.prunable_nn as pnn
import torch.utils.model_zoo as model_zoo
from torchvision import models
from operator import itemgetter


class VGG(models.VGG):

    def prune(self):
        # gather all modules & their indices. (excluding classifier)
        # gather all talyor_estimate_lists & pair with the indices
        # gather all talyor_estimates & pair with their list index & module index
        # reduce to the minimum in the list
        # grab the module with the minimum & prune
        # grab the PBatchNorm & adjust
        # adjust the next layer too

        feature_list = list(enumerate(self.features))
        # grab the taylor estimates of PConv2ds & pair with the module's index in self.features
        taylor_estimates_by_module = [(module.taylor_estimates, module_idx) for module_idx, module in feature_list
                                      if issubclass(type(module), pnn.PConv2d) and module.out_channels > 1]
        taylor_estimates_by_feature_map = \
            [(estimate, map_idx, module_idx)
             for estimates_by_map, module_idx in taylor_estimates_by_module
             for map_idx, estimate in enumerate(estimates_by_map)]

        _, min_map_idx, min_module_idx = min(taylor_estimates_by_feature_map, key=itemgetter(0))

        p_conv2d = self.features[min_module_idx]
        p_conv2d.prune_feature_map(min_map_idx)

        p_batchnorm = self.features[min_module_idx+1]
        p_batchnorm.drop_input_channel(min_map_idx)

        offset = 3 # batchnorm, relu, maxpool
        is_last_conv2d = (len(feature_list)-1)-offset == min_module_idx
        is_double_conv2d_layer = min_module_idx == 8 or min_module_idx == 15 or min_module_idx == 22
        if is_last_conv2d:
            first_p_linear = self.classifier[0]
            shape = (first_p_linear.in_features//49, 7, 7) # the input is always ?x7x7
            first_p_linear.drop_inputs(shape, min_map_idx)
        elif is_double_conv2d_layer:
            # no max pool,
            next_p_conv2d = self.features[min_module_idx+offset]
            next_p_conv2d.drop_input_channel(min_map_idx)
        else:
            next_p_conv2d = self.features[min_module_idx+offset+1]
            next_p_conv2d.drop_input_channel(min_map_idx)

def vgg_model(num_classes):

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            return nn.Sequential(*layers)

    cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],}
    model_url = 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
    model = VGG(make_layers(cfg['A'], batch_norm=True))
    model.load_state_dict(model_zoo.load_url(model_url), strict=False)

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model, 'vgg11_bn'


def chinese_model(num_classes):
    return ChineseNet(num_classes), 'chinese_net'


def chinese_pruned_80(num_classes):
    config = [26, 'M', 39, 'M', 52, 'M', 75, 93, 'M', 88, 95, 'M']
    return ChineseNet(num_classes, config), 'chinese_net_80'


def chinese_pruned_90(num_classes):
    config = [15, 'M', 14, 'M', 20, 'M', 27, 31, 'M', 28, 30, 'M']
    return ChineseNet(num_classes, config), 'chinese_net_90'


class ChineseNet(nn.Module):
    # inspired by https://arxiv.org/abs/1702.07975, used for chinese ocr
    def __init__(self, num_classes, config=None):
        super(ChineseNet, self).__init__()
        self.config = [96, 'M', 128, 'M', 160, 'M', 256, 256, 'M', 384, 384, 'M'] if config is None else config
        self.features = self.make_layers()
        self.classifier = nn.Sequential(
            # input is 96x96, output from features section should always be 2x2
            pnn.PLinear(self.config[-2]*2*2, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        self.convert_to_onnx = False

    def make_layers(self):
        layers = []
        in_channels = 1
        for v in self.config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                conv2d = pnn.PConv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, pnn.PBatchNorm2d(v), nn.PReLU()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.convert_to_onnx:
            x = self.classifier[0](x)

            # manually perform 1d batchnorm, caffe2 currently requires a resize,
            # which is hard to squeeze into the exported network
            bn_1d = self.classifier[1]
            numerator = (x - Variable(bn_1d.running_mean))
            denominator = Variable(torch.sqrt(bn_1d.running_var + bn_1d.eps))
            x = numerator/denominator*Variable(bn_1d.weight.data) + Variable(bn_1d.bias.data)

            x = self.classifier[2](x)
            x = self.classifier[3](x)
            x = self.classifier[4](x)
            return x
        else:
            x = self.classifier(x)
            return x

    def prune(self):
        # gather all modules & their indices. (excluding classifier)
        # gather all talyor_estimate_lists & pair with the indices
        # gather all talyor_estimates & pair with their list index & module index
        # reduce to the minimum in the list
        # grab the module with the minimum & prune
        # grab the PBatchNorm & adjust
        # adjust the next layer too

        feature_list = list(enumerate(self.features))
        # grab the taylor estimates of PConv2ds & pair with the module's index in self.features
        taylor_estimates_by_module = [(module.taylor_estimates, module_idx) for module_idx, module in feature_list
                                      if issubclass(type(module), pnn.PConv2d) and module.out_channels > 1]

        taylor_estimates_by_feature_map = \
            [(estimate, map_idx, module_idx)
             for estimates_by_map, module_idx in taylor_estimates_by_module
             for map_idx, estimate in enumerate(estimates_by_map)]

        _, min_map_idx, min_module_idx = min(taylor_estimates_by_feature_map, key=itemgetter(0))

        p_conv2d = self.features[min_module_idx]
        p_conv2d.prune_feature_map(min_map_idx)

        p_batchnorm = self.features[min_module_idx+1]
        p_batchnorm.drop_input_channel(min_map_idx)

        offset = 3 # batchnorm & prelu & maxpool
        is_last_conv2d = (len(feature_list)-1)-offset == min_module_idx
        is_double_conv2d_layer = min_module_idx == 12 or min_module_idx == 19
        if is_last_conv2d:
            first_p_linear = self.classifier[0]
            shape = (first_p_linear.in_features//4, 2, 2) # the input is always ?x2x2
            first_p_linear.drop_inputs(shape, min_map_idx)
        elif is_double_conv2d_layer:
            # no max pool, -1
            next_p_conv2d = self.features[min_module_idx+offset]
            next_p_conv2d.drop_input_channel(min_map_idx)
        else:
            next_p_conv2d = self.features[min_module_idx+offset+1]
            next_p_conv2d.drop_input_channel(min_map_idx)

