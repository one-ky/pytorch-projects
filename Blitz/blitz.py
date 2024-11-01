import torch
import numpy as np

'''
tensors are a datastructure similar to arrays. in pytorch, we use tensors to encode the inputs and outputs of a model, as well as model parameters. 
tensors are similar to to Numpys ndarrays, except that tensors run on GPU
'''

# tensors can be completed directly from data, this datatype is automatically inferred
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# tensors can be created from numpy arrays
np_array = np.array(data)
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the dtype of the x_data
print(f"Random Tensor: \n {x_rand} \n")

'''torch tensors support automatic gradient computation '''
'''shape defines the structure of the tensor'''

# shape is a tuple of tensors dimensions. in the fucntion below, it detemines the dimensionality of the output tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random tensor: \n {rand_tensor} \n")
print(f"Ones tensor: \n {ones_tensor} \n")
print(f"Zeros tensor: \n {zeros_tensor} \n")

'''tensor operations on the gpu'''
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# move the tensor to the gpu

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is sotred on: {tensor.device}")