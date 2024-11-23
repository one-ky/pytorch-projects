import torch

# tensors (single number) with gradient tracking, to track operations input and output
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# simple neural network
# y = wx + b
def forward_pass():

    # each operation creates nodes in computational graph. the graph records the operation 
    # preformed, input tensor, and resulting output tensor
    layer1 = w * x
    output = layer1 + b
    return output
'''one single forward pass will show us layer1 = 3*2 + 1, which will be 7'''

y = forward_pass()
print(f"Output: {y}")

# compute gradients
y.backward()

# computing the derivaties with respect to the variables
print(f"dx/dy: {x.grad}")
print(f"dw/dy: {w.grad}")
print(f"db/dy: {b.grad}")

# derivative recap
x = torch.tensor(3.0, requires_grad=True) # x is a scaler number 3
y = x*x # y equals x times x, or 3*3
print(f"x={x}") # x is equal to 3
print(f"f(x) = {y}") # y is equal to 9

y.backward()
print(f"derivative = {x.grad}") # the derivative of x*x is 2x, 2*x is 2*3 is 6
'''this means a change in x causes 6 times the change in y when x is equal to 3'''

# simple example
parameter =  torch.tensor(2.0, requires_grad=True) # lets say our initial guess is multiplying by 2

true_answer = 15.0 # if the input is 5 the answer should be 20

for step in range(5):
    # forward pass: make a prediction using out current parameter
    prediction  = 5 * parameter # if this is the first pass, 2 will be the parameter and the prediction will be 10

    loss = (prediction - true_answer)**2 # the loss function here is the squared error

    print(f"\n Step {step + 1}")
    print(f"\nCurrent Parameter: {parameter.item():.4f}")
    print(f"Prediction: {prediction.item(): .4f}")
    print(f"\nHow wrong we are (loss): {loss.item():.4f}")

    # backward pass, shows us which direction to adjust the parameter
    loss.backward()
    print(f"Gradient: {parameter.grad.item():.4f}")

    # update parameter to make prediction better
    with torch.no_grad(): # temporarily turn off gradient tracking
        parameter -= 0.01 * parameter.grad # adjust parameter using gradient, move the parameter in the opposite direction fo the gradient
        parameter.grad.zero_() # reset gradient for next step

'''pytorch will keep track of all of the calculations done with the torch.tensor'''
''' using .backward() tells python this is the calculation we need to examine to see how to change our parameter'''

''' understanding loss'''
# 1. loss = (prediction - true)^2
# 2. prediction = 5 * parameter

# substitute #2 into #1 
# loss = (5*parameter - true)^2

# using the chain rule to find the derivative with respect to the parameter
# d(loss)/d(parameter) = 2 * (5*parameter - true) * 5

# in our example
# = 2 * (5*2 -15) *5
# = 2 * (-5) * 5
# = -50


# simplify
''' a tensor is a number that is being tracked, if it is ever used, it will be tracked every step'''
'''pytorch only cares about the calculation that is connected with the loss'''

parameter = torch.tensor(2.0, requires_grad=True) # this is our initial parameter
true_answer = 15 # the number we want to get 

# forward pass, make a prediction
prediction = 5 * parameter

#loss
loss = (prediction - true_answer)**2
# (10 - 15)**2
# (-5)**2
# = -25

# backward pass, calculate the gradient
loss.backward()
# the derivative of our loss function:
# starting with: (5*parameter - 15)**2 # 5*parameter is the initail prediction of 10 for the first pass
# derivative using the chain rule (5*parameter -15)**2 is
# chain rule is dy/dx = (dy/du) * (du/dx)
# chain rule inner function: u = 5*x -15, dy/du = 5
# chain rule outer function: y = u**2, du/dx = 2x
# dy/dx = (dy/du) * (du/dx) == 5 * 2x == 5*2*5

# update the parameter
parameter -= 0.01 * parameter.grad
# parameter = 2.0 - (0.01 * -50)
# parameter = 2.0 + 0.5
# parameter becomes 2.5

# reset the gradient
parameter.grad.zero_() # clear the gradient for the next pass

# Started with initial parameter value of 2
# prediction was 10
# target value is 15
# loss was 25
# gradient is 50
# new parameter is 2.5