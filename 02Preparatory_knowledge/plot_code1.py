import matplotlib.pyplot as plt
import torch

plt.plot([1, 2, 3, 4, 5], [2, 3, 7, 10, 6], '^--r', label="A")
plt.plot([1, 2, 3, 4, 5], [4, 8, 9, 11, 2], '*-.y', label="B")
plt.title("Demo")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, 7)
plt.xticks(torch.arange(0, 7, 1))
plt.legend()
plt.show()
