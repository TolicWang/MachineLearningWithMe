import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.rand(100) * 5
y1 = np.array([1 for _ in x1])
color = [1 if i > 1.25 and i < 3.75 else 0 for i in x1]
x2 = x1
y2 = (x2 - 2.5) ** 2
plt.scatter(x1,y1,c = color)
plt.scatter(x2, y2, c=color)
plt.show()
