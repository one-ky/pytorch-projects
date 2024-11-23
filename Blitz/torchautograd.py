'''torch.auotgrad is pytorchs automatic differentiation engine.
- weights and biases in pytorch as stored as tensors
- trainning a NN has two steps
- forward prop the nn makes its best guesss about the correct output
- backprop, the nn adjusts parameters proportinatio to the error or its guess. it traverses backwards
and collects the derivatives of the error with respect to the parameter functions (gradients)'''

'''example of a single training step. pretrained resnet18 model from torchvision. create a random data tensor to represent a single image with 3 channels
and height & width of 64, and its corresponding label initialized to some random values. label in pretrained models has shape(1, 1000)
it is just a noisy image to be used as an example. it is not an image that represents anything'''

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT) # pretrained model
data = torch.rand(1,3,64,64) # data tensor to represnet a single image. 1 image, 3 channels (RGB), height 64 pixels width 64 pixels
labels = torch.rand(1,1000) # this is arbitrary. used to show the calculations of the backprop. in real life if the image was of an object this would be an array of probabilities of the image being each class
# the random labels are only present because to demonstrate backprop we need some target to calculate loss against
# to calculate loss, you need something to compare the models output to
# to show .backward() you need a loss value to propogate back

''' next we run the forward pass
takes the input data through all layers of the resnet18
each layers operations are automatically tracked by autograd
creates a computational graph for automatic differnetiation
returns predictions across all 1000 imagenet classes'''

prediction = model(data) # forward pass where model tries to classify this static

''' use the models prediction and corresponding label to calcualte error (loss). the next step is to backpropogate this error through the network
backward propagation is kicked off we call .backward() on the error tensor. autograd then calculates and store teh gradients for each model parameter in the parameters .grad attribute'''
# loss = prediction based on randomly generated pixels that. prediction is the prob of being in each of the 1000 classes of image net
loss = (prediction - labels).sum()
loss.backward() # backprop

# next, we load an optimizer. in this case, an SGD with a learning rate of 0.01 and momentum of 0.9
''' after loss.backward() calculates the gradients, the optimizer updates the models parameters. it decides the gradients to improve the model'''
# SGD is Stochastic Gradient Descent, a method for adjusting the models parameters based on gradients
# taking small steps looking for the steepest descent\
# momentum helps the optimizer move more efficiently by considering the previous updates, it can help with small increased when looking for the largest decrese
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# finally we call .step() to initiate gradient descent. the optimizer adjusts each parameter by its gradients stored in .grad
optim.step()



# DIFFERENTIATION IN AUTOGRAD ########################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################

# this gets deeper into the mathematics
'''lets learn how autograd collects gradients. create two tensors and every operation is tracked'''

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# create another tensor Q from a and b, Q is a vector with 2 elements, a and b
Q = 3*a**3 - b**2
'''lets assume a and b to be parameters of an NN, and Q to be the error. in NN training, we want gradients fo the error w.r.t parameters, ie
    dQ/da = 9a**2
    dQ/db = -2b
when we call .backward() on Q, autograd calculates these gradients and stores them in the respective tensors' .grad attribute
we need to explicitly pass a gradient argument in Q.backward because it is a vector. gradient is a tensor of the shape as Q, and it represents the gradient of Q w.r.t itself
    dQ/dQ = 1
we can use Q.sum().backward() to sum the gradients for each component, where the gradients are the weights that are multiplying the derivatives. summing them would give each component equal contribution
the point in summing the gradients is to 
'''
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
#gradients are now deposited in a.grad and b.grad
print(9*a**2 == a.grad)
print(-2*b == b.grad)



### simple example ###
''' detour to get it through my head
1. Understand our variables
a and b are tensors with 2 elements [2,3] and [6,4] respectivly

    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)

2. Understand Q
Q is a tensor with two elements a and b
    Q = 3*a**3 - b**2
Q[0]=-12 and Q[1]=65

3. 
if we look at this like a normal Neural Net, a and b could be our parameters(weights)
Q could be our predictions for two different examples
if our target values were [0,0], Q represents our errors where Q[0]=-12, Q(a=2, b=6),  means our first prediction is 12 units too low and Q[1]=65, Q(a=3, b=4), vmeans our second prediction was 65 units too high
if we use 
    external_gard=torch.tensor([1.,1.])
    Q.backward(gradient=external_gradient)
this calculates the chain rule to find the derivaties of Q withrespect to a and respect to b
then we plug in the values for a[0], a[1] to get the first and second component of the derivaties with respect to a
do the same for b
so we have 4 values, ∂Q/∂a[0] = 36, ∂Q/∂a[1] = 81, ∂Q/∂b[0] = -12, ∂Q/∂b[1] = -8
this would result in the total gradient for a to be [36,81] and the total gradient for b to be [-12,8]
finally, for the opimization step, we multiply the learning rate by the output of this calculation and subtract it from the original parameter. so
    new_a[0] = a[0] - learning_rate * 36 == 1.64
    new_a[1] = a[1] - learning_rate * 81 == 2.19
    new_b[0] = b[0] - learning_rate * -12 == 6.12
    new_b[1] = b[1] - learning_rate * -8 == 4.08

4. summary of variables roles
a,b are parameters were trying to optimize (like weights in a neural net)
Q is the output of our function (like prediction in a neural net)
Q.sum is the total error across all components
Gradients (a.grad, b.grad) tell us how to change each parameter to reduce error
learning_rate: how big a step to take in the direction of the gradients

Key points:
1. Gradients tell us the direction of the steepest increase for each parameter
2. The magnitude of the gradient tells us how sensitive Q is to that parameter
3. we move opposite to the gradient to minimize our error
4. learning rate controls the size of the step

Fundementals:
Forward Pass - Make Prediction (Q)
Compare to targets to get error
Backward pass: Calculate gradients
Update Parameters untils gradients reduce error
Repeat until predictions are good enough (usually max amount of epochs)
'''


### NOTES ON VECTOR CALCULUS
''' torch.autograd is an engine for computing vector-Jacobian product
conceptually, autograd keeps a record of data (tensors), and all executed operations (along with the resulting new tensors)

'''

# frozen parameters example
''' in a NN, parameters that don't compute gradients are usually called frozen parameters. we freeze parameters if we know we wont need their gradients. 
in finetuning, we freeze most of the mdoel and typically modify the classifier layes to make predictions on new labels. below is an example
'''
from torch import nn, optim
model = resnet18(weights=ResNet18_Weights.DEFAULT)


# freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad=False

'''lets say we want to fine tune our model on a new dataset with 10 labels. in resnet, the classifier is the last linear layer model.fc.
we can simply replace it with a new linear layer that acts as our classifier
'''
model.fc = nn.Linear(512, 10)
''' now all parameters in the model, except the parameters of model.fc are frozen. the only parameters that compute gradients are the wights and bias of model.fc'''
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

''' the mental image is 
Input Image --> [Frozen Layers] --> Features --> [Trainable FC layers] --> Predictions
                - no gradients                   - Gradients
                  or updates                       updates                  '''                 


