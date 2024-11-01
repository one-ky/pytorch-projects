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

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

vector = torch.tensor([1,2,3,4])
matrix = torch.tensor([[1,2],
                      [3,4]])
cube = torch.tensor([[[1,2], [3,4]], [[5,6], [7, 8]]])
''' pytorch tensors can have any number of dimensions
they can run on gpus for faster calculations
they automatically track gradients
'''

# you can use torch.cat to concatenate a sequence of tensors along a given dimension
t1 = torch.cat([tensor, tensor, tensor, tensor], dim=1)
print(f'\n{t1}')

# tensor element wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)}\n")
print(f"tensor * tensor \n {tensor* tensor}\n") # alternate syntax

# matrix multiplication

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor@ tensor.T \n {tensor @ tensor.T} \n")

a = torch.tensor([[2,3], [4,1]])
b = torch.tensor([[5,1], [2,6]])
'''take the second matrix first column, transpose to row, multiple first matrix first row element wise, that is the first 
row first col value of the new matrix product. so 2*5 + 3*2 for top left value. 4*5 + 2*1 for bottom left value'''
c = a @ b
print(a)
print(b)
print(c)

# in place opereators that have a _ suffix are in-place.
print(tensor, '\n')
tensor.add_(5)
print(tensor)
'''in place operators can save memory but when compouting derivaties it causes an immediate loss of history so they are not used commonly'''

# bridge with numpy
'''tensors on the cpu and numpy arrays can shate their underlying memory locations, and changeing one will change the other'''
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
# numpy array to tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")