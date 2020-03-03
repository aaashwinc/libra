import numpy as np
import matplotlib.pyplot as plt

d = np.zeros((21,21))
for i in range(0,21):
  d[i,10] = 1
# d[10,10] = 1
# d[10,9] = -1
D = np.fft.fft2(d)

plt.matshow(np.abs(D))
plt.matshow(d)
plt.show()