from models import *
import prunable_nn as pnn
import torch
from torch.autograd import Variable, backward
import unittest

class TestChineseNet(unittest.TestCase):

    def setUp(self):
        self.model = ChineseNet(10).cuda()
        self.inputs = Variable(torch.FloatTensor(4, 1, 96, 96).cuda())
        self.grad = Variable(torch.FloatTensor(4, 10).cuda())

    def test_PruneModel_PruneFirstFeatureMapOfFirstModule(self):
        output = self.model(self.inputs)
        backward(output, self.grad)

        # rig gradients, set all to 1, except first module's first map
        pConv2ds = (module for module in self.model.modules() if issubclass(type(module), pnn.PConv2d))
        for idx, pConv2d in enumerate(pConv2ds):
            pConv2d.taylor_estimates = torch.ones(pConv2d.taylor_estimates.size())
            if idx == 0:
                pConv2d.taylor_estimates[0] = 0.1

        expected_conv2d_out_channels = self.model.features[0].out_channels - 1
        expected_batchnorm_num_features = self.model.features[1].num_features - 1
        next_conv2d_in_channels = self.model.features[4].in_channels - 1
        self.model.prune()

        # being a little lazy here, since prunable_nn_test covered weight checking
        # check first conv2d's input
        # check first batchnorm's input
        # check 2nd conv2d's input
        self.assertEqual(self.model.features[0].out_channels, expected_conv2d_out_channels)
        self.assertEqual(self.model.features[1].num_features, expected_batchnorm_num_features)
        self.assertEqual(self.model.features[4].in_channels, next_conv2d_in_channels)

        # run again, ensure no bugs with modules
        self.model(self.inputs)

    def test_PruneModel_PruneFirstFeatureMapOfLastModule(self):
        output = self.model(self.inputs)
        backward(output, self.grad)

        # rig gradients, set all to 1, except last module's first map
        pConv2ds = [module for module in self.model.modules() if issubclass(type(module), pnn.PConv2d)]
        last_idx = len(pConv2ds) - 1
        for idx, pConv2d in enumerate(pConv2ds):
            pConv2d.taylor_estimates = torch.ones(pConv2d.taylor_estimates.size())
            if idx == last_idx:
                pConv2d.taylor_estimates[0] = 0.1

        old_linear_in_features = self.model.classifier[0].in_features
        self.model.prune()

        # only check linear's input size
        self.assertTrue(self.model.classifier[0].in_features < old_linear_in_features)

        # run again, ensure no bugs with modules
        self.model(self.inputs)

class TestVGG(unittest.TestCase):

    def setUp(self):
        model, _ = vgg_model(10)
        self.model = model.cuda()
        self.inputs = Variable(torch.FloatTensor(4, 3, 224, 224).cuda())
        self.grad = Variable(torch.FloatTensor(4, 10).cuda())


    def test_PruneModel_ShouldBePrunedInRightPlace(self):
        output = self.model(self.inputs)
        backward(output, self.grad)

        # rig gradients, set all to 1, except first module's first map
        pConv2ds = (module for module in self.model.modules() if issubclass(type(module), pnn.PConv2d))
        for idx, pConv2d in enumerate(pConv2ds):
            pConv2d.taylor_estimates = torch.ones(pConv2d.taylor_estimates.size())
            if idx == 0:
                pConv2d.taylor_estimates[0] = 0.1

        expected_conv2d_out_channels = self.model.features[0].out_channels - 1
        expected_batchnorm_num_features = self.model.features[1].num_features - 1
        next_conv2d_in_channels = self.model.features[4].in_channels - 1
        self.model.prune()

        # being a little lazy here, since prunable_nn_test covered weight checking
        # check first conv2d's input
        # check first batchnorm's input
        # check 2nd conv2d's input
        self.assertEqual(self.model.features[0].out_channels, expected_conv2d_out_channels)
        self.assertEqual(self.model.features[1].num_features, expected_batchnorm_num_features)
        self.assertEqual(self.model.features[4].in_channels, next_conv2d_in_channels)

        # run again, ensure no bugs with modules
        self.model(self.inputs)

    def test_PruneModel_PruneFirstFeatureMapOfLastModule(self):
        output = self.model(self.inputs)
        backward(output, self.grad)

        # rig gradients, set all to 1, except last module's first map
        pConv2ds = [module for module in self.model.modules() if issubclass(type(module), pnn.PConv2d)]
        last_idx = len(pConv2ds) - 1
        for idx, pConv2d in enumerate(pConv2ds):
            pConv2d.taylor_estimates = torch.ones(pConv2d.taylor_estimates.size())
            if idx == last_idx:
                pConv2d.taylor_estimates[0] = 0.1

        old_linear_in_features = self.model.classifier[0].in_features
        self.model.prune()

        # only check linear's input size
        self.assertTrue(self.model.classifier[0].in_features < old_linear_in_features)

        # run again, ensure no bugs with modules
        self.model(self.inputs)

if __name__ == '__main__':
    unittest.main()