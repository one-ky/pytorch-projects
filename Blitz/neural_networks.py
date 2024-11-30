'''nn can be constructed with torch.nn
nn depends on autograd to define models and differentiate them'''

# typical training procedure for nn is as follows"

# 1: define the nn tha has some learnabel parameters (or weights)
# 2: iterate over the dataet of inputs
# 3: pricess input through the network
# 4: compute the loss (how far is the output from being correct)
# 5: propogate gradients back into the networks parameters
# 6: update the weights of the network, typically using a simple update rule: weight = weight - learning rate * gradient

'''lets define the network'''

# import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

'''a class is a template that defines 
what properties an object has
what actions an object can perform
'''
class Net(nn.Module): # nn.Module is the base class for all nn in pytorch

    def __init__(self): # init is a constructor that runs automaticallly to set up the initial state of the new object, self refers to the specific 
        #instance of the class being created
        super(Net, self).__init__() # calls the parent class's constructor, gets all the basic features from the Net and adds special features
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # add our specific neural network layers
        # nn.module provides essential functionality such as parameter management, gpu/cpu device movement, save/load functionality, traning mode switches 
        self.conv1 = nn.Conv2d(1, 6, 5) # conv1 will look for basic shapes that are represented often in specific outputs
        self.conv2 = nn.Conv2d( 6, 16, 5)
        # and affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 
        self.fc2 = nn.Linear(120, 84) # 
        self.fc3 = nn.Linear(84, 10) # fc3 makes the final decision
        '''
        this is a digit recognition nn
        each of these are a layer that have their own job
        - conv1 will learn to detect maybe horizontal lines, diagonal lines, and loops that are important for distinguishing digits from eachother
        it scans the image in small 5x5 pixel chunks looking for basic patterns
        - conv2 is like the second look that will take the patterns from conv1 and combine them together,like combining a curve and line to make a 6
        - fc1 looks at all the information and assignes importance, like loop at bottom important for for 6 and loop at top important for 9
        - fc2 combines importance again looking for stronger evidence
        - fc3 make the final probability for each digit (0-9), output for 6 could be somthing like [0.01, 0.01, 0.02, 0.01, 0.01, 0.9, 0.01, 0.01, 0.01]
        so we are 90% sure that the digit is the number 6
        '''

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channelrs
        # 5x5 square convolution, it used RELU activation function, and 
        # outputs a Tensor with size (N, 6, 28, 28) where N is the size of the batch
        c1 = F.relu(self.conv(input))
        #Subsampling layer S2: 2x2 grid, purley functional,
        # this layer does not have any parameter, and ouputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it used RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not ahve any parameter, and outputs a (N, 16, 5, 5) tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and 
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output

net = Net()
print(net)
