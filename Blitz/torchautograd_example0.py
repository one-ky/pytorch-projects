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

# example. we want to teach a model how to multiply by 4
parameter =  torch.tensor(2.0, requires_grad=True) # lets say our initial guess is multiplying by 2

true_answer = 15.0 # if the input is 5 the answer should be 20

for step in range(5):
    # forward pass: make a prediction using out current parameter
    prediction  = 5 * parameter # if this is the first pass, 2 will be the parameter and the prediction will be 10

    loss = (prediction - true_answer)**2 

    print(f"\n Step {step + 1}")
    print(f"\nCurrent Parameter: {parameter.item():.4f}")
    print(f"Prediction: {prediction.item(): .4f}")
    print(f"\nHow wrong we are (loss): {loss.item():.4f}")

    # backward pass, shows us which direction to adjust the parameter
    loss.backward()
    print(f"Gradient: {parameter.grad.item():.4f}")

    # update parameter to make prediction better
    with torch.no_grad(): # temporarily turn off gradient tracking
        parameter -= 0.01 * parameter.grad # adjust parameter using gradient
        parameter.grad.zero_() # reset gradient for next step


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
