import unittest
import prunable_nn as pnn
import torch
import numpy as np
from torch.autograd import Variable


class TestPrunableConv2d(unittest.TestCase):

    def setUp(self):
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)

        self.module = pnn.PConv2d(5, 5, 3, padding=1).cuda()
        self.module.train()

        self.input_shape = (1, 5, 50, 50)
        self.input = Variable(torch.rand(*self.input_shape).cuda(), requires_grad=True)
        self.upstream_gradient = torch.rand(*self.input_shape).cuda()

    def test_getTaylorEstimates_ShouldGiveValidValueAndSize(self):
        output = self.module(self.input)
        torch.autograd.backward(output, self.upstream_gradient)

        # ensure input and output are different
        self.assertFalse(np.array_equal(self.input.data.cpu().numpy(), output.data.cpu().numpy()))

        estimates = self.module.taylor_estimates.data.cpu()
        size = estimates.size()

        # ensure sane size
        self.assertEqual(size, torch.FloatTensor(self.input_shape[1]).size())
        # ensure not zero
        self.assertFalse(np.array_equal(estimates.numpy(), torch.zeros(size).numpy()))

    def test_pruneFeatureMap_ShouldPruneRightParams(self):
        dropped_index = 0
        output = self.module(self.input)
        torch.autograd.backward(output, self.upstream_gradient)

        old_weight_size = self.module.weight.size()
        old_bias_size = self.module.bias.size()
        old_out_channels = self.module.out_channels
        old_weight_values = self.module.weight.data.cpu().numpy()

        # ensure that the chosen index is dropped
        self.module.prune_feature_map(dropped_index)

        # check bias size
        self.assertEqual(self.module.bias.size()[0], (old_bias_size[0]-1))
        # check output channels
        self.assertEqual(self.module.out_channels, old_out_channels-1)

        _, *other_old_weight_sizes = old_weight_size
        # check weight size
        self.assertEqual(self.module.weight.size(), (old_weight_size[0]-1, *other_old_weight_sizes))
        # check weight value
        expected = np.delete(old_weight_values, dropped_index , 0)
        self.assertTrue(np.array_equal(self.module.weight.data.cpu().numpy(), expected))

    def test_dropInputChannel_ShouldDropRightValues(self):
        dropped_index = 0

        old_weight_values = self.module.weight.data.cpu().numpy()

        # ensure that the chosen index is dropped
        self.module.drop_input_channel(dropped_index)
        expected = np.delete(old_weight_values, dropped_index, 1)
        self.assertTrue(np.array_equal(self.module.weight.data.cpu().numpy(), expected))


class TestDropInputClasses(unittest.TestCase):

    def test_PLinearDropInputs_ShouldDropRightParams(self):
        dropped_index = 0

        # assume input is 2x2x2, 2 layers of 2x2
        input_shape = (2, 2, 2)
        module = pnn.PLinear(8, 10)

        old_num_features = module.in_features
        old_weight = module.weight.data.cpu().numpy()
        resized_old_weight = np.resize(old_weight, (module.out_features, *input_shape))

        module.drop_inputs(input_shape, dropped_index)
        new_shape = module.weight.size()

        # ensure that the chosen index is dropped
        expected_weight = np.resize(np.delete(resized_old_weight, dropped_index, 1), new_shape)
        output = module.weight.data.cpu().numpy()
        self.assertTrue(np.array_equal(output, expected_weight))

        # ensure num features is reduced
        self.assertTrue(module.in_features, old_num_features-1)

    def test_PBatchNorm2dDropInputChannel_ShouldDropRightParams(self):
        dropped_index = 0
        module = pnn.PBatchNorm2d(2)

        old_num_features = module.num_features
        old_bias = module.bias.data.cpu().numpy()
        old_weight = module.weight.data.cpu().numpy()

        module.drop_input_channel(dropped_index)

        # ensure that the chosen index is dropped
        expected_weight = np.delete(old_weight, dropped_index, 0)
        self.assertTrue(np.array_equal(module.weight.data.cpu().numpy(), expected_weight))
        expected_bias = np.delete(old_bias, dropped_index, 0)
        self.assertTrue(np.array_equal(module.bias.data.cpu().numpy(), expected_bias))
        # ensure num features is reduced
        self.assertTrue(module.num_features, old_num_features-1)


if __name__ == '__main__':
    unittest.main()