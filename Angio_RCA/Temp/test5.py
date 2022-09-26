import numpy as np
import torch

a = np.arange(30).reshape((15, 2))

a = torch.from_numpy(a).type(torch.DoubleTensor)

a.requires_grad = True

print(a, a.requires_grad)

norm = torch.norm(a[1:, :] - a[:-1, :], dim=-1)

print(norm)

loss = torch.sum(norm - 2)

# loss.backward()

external_grad = 0.1 * torch.arange(14).type(torch.DoubleTensor)
print(norm.shape, external_grad)

print(a.grad)

norm.backward(gradient=external_grad)
# norm[11].backward(retain_graph=True)

print(a.grad[10, 0], a.grad)

# norm[11].backward(retain_graph=True)
# print(a.grad[10, 0], a.grad)
