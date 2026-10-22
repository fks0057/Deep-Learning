import torch
import torch.nn as nn

#Forward pass
x = torch.tensor([
    [1.0, 2.0, 3.0]
])

layer = nn.Linear(3, 2)

output = layer(x)

print("Input:")
print(x)

print("Output:")
print(output)

print("Weights:")
print(layer.weight)

print("Bias:")
print(layer.bias)