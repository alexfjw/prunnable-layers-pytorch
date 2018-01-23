## Prunable neural network layers for Pytorch
The code in this repo has been extracted from my ML udacity capstone project [here](https://github.com/alexfjw/jp-ocr-prunned-cnn),
where I experimented with CNNs for Japanese OCR.

Prunable versions of common CNN layers can be found in this repo.
We use taylor expansion to estimate the importance of feature maps, as described in [1].

I went with an OOP approach to make the code more readable.
The work done by Jacob was a great reference for writing this code.
His blogpost on pruning can be found [here](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

### Dependencies
- Python 3.6.3
- Pytorch 0.4
- torchvision (for vgg example)

## Description of code
The layers in `prunable_nn.py` are plug and play. However, they only compute the importance of each feature map. (pruning is a concept linked to models, and not individual layers).
Filtering to select the smallest feature map, and to dropping inputs for the immediate layers has to be done manually.

PConv2d is a Conv2d layer which registers a backward hook during gradient calulation to weigh the importance of each feature map.

Feature maps are stored after every forward operation, and talyor estimates for feature map importance is calculated after every backward operation.

Removing a feature map reduces the outputs of the layer. The next immediate layer has to take in fewer inputs. (drop the number of inputs)
PLinear and PBatchNorm are coded to perform this task.

## Examples
There are two examples of models adapted to support pruning in `models.py`.
Pruning with ChineseNet has been thoroughly tested.
VGG11_BN requires too much memory for pruning, but can still be used as a reference. (Seems to need more than 6GB of VRAM)

The tests in `./tests/` may be useful for testing your prunable models.

For examples of the full training & pruning procedure, refer to my ML udacity capstone project [here](https://github.com/alexfjw/jp-ocr-prunned-cnn).
The project also contains benchmarks on pruning ChineseNet.

## Possible Extensions
Making PLinear prunable - shouldn't be too difficult, implementation should be identical to PConv2d.


## References

[1] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017, June 08). Pruning Convolutional Neural Networks for Resource Efficient Inference. Retrieved December 10, 2017, from https://arxiv.org/abs/1611.06440

[2] Link to Jacob's blogpost
https://jacobgil.github.io/deeplearning/pruning-deep-learning
