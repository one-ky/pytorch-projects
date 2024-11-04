'''torch.auotgrad is pytorchs automatic differentiation engine.
- weights and biases in pytorch as stored as tensors
- trainning a NN has two steps
- forward prop the nn makes its best guesss about the correct output
- backprop, the nn adjusts parameters proportinatio to the error or its guess. it traverses backwards
and collects the derivatives of the error with respect to the parameter functions (gradients)'''

'''example of a single training step. pretrained resnet18 model from torchvision. create a random data tensor to represent a singel image with 3 channels
and height & width of 64, and its corresponding label initialized to some random values. label in pretrained models has shape(1, 1000)'''

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)
