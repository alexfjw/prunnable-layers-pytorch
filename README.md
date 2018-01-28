## Prunable neural network layers for Pytorch
The code in this repo has been extracted from my ML udacity capstone project [here](https://github.com/alexfjw/jp-ocr-prunned-cnn),
where I experimented with CNNs for Japanese OCR.

Prunable versions of common CNN layers can be found in this repo.
We use taylor expansion to estimate the importance of feature maps, as described by Molchanov, P. et al[1].  
A good summary of this approach can be found [here](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

Pruning & taylor estimation was written with an OOP approach for readability.

### Dependencies
- Python 3.6.3
- Pytorch 0.4
- torchvision (for vgg example)

## Description of code
The layers in `prunable_nn.py` are plug and play. However, they only compute the importance of each feature map. (pruning is a concept linked to models, and not individual layers).
Filtering to select the smallest feature map and dropping inputs for layer next to the pruned layer has to be done manually.

PConv2d is a Conv2d layer which registers a backward hook during gradient calulation to weigh the importance of each feature map.

Feature maps are stored after every forward call, and talyor estimates for feature map importance is calculated after every backward call.

As mentioned, removing a feature map reduces the outputs of the layer. 
The next immediate layer has to take in fewer inputs.
PLinear and PBatchNorm have been written to perform this.

## Usage Examples
Simply use `PLinear(..), PConv2(..) & PBatchNorm(...)` in place of the originals.  
Also modify your models to add a prune method.  
See `models.py` for further information.

The tests in `./tests/` may be useful for testing your prunable models.  

For benchmarks & examples of the full training & pruning procedure, refer to my ML udacity capstone project [here](https://github.com/alexfjw/jp-ocr-prunned-cnn).

## Possible Extensions
Making PLinear prunable - shouldn't be too difficult, implementation should be identical to PConv2d.


## References

[1] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017, June 08). Pruning Convolutional Neural Networks for Resource Efficient Inference. Retrieved December 10, 2017, from https://arxiv.org/abs/1611.06440

