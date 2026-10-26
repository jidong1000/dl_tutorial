import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = torch.relu(x)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x.detach(), y.detach(), color='purple')
ax[0].set_title('relu(x)')

y.sum().backward()
ax[1].plot(x.detach(), x.grad, color='purple')
ax[1].set_title("relu'(x)")

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_position('zero')
ax[0].spines['bottom'].set_position('zero')

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_position('zero')
ax[1].spines['bottom'].set_position('zero')
plt.show()