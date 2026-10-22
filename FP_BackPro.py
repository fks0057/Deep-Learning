import torch
import torch.nn as nn

torch.manual_seed(42)

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([[1.0, 0.0]])

layer = nn.Linear(3, 2)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    output = layer(x)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}, "
            f"Output: {output.detach().numpy()}"
        )