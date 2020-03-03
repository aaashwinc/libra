import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.ndimage import gaussian_filter

np.convolve([1, 2, 3], [0, 1, 0.5])

fig = plt.figure()
ax = plt.axes()



gaussian = np.linspace(0,0,201)
gaussian[100] = 1
gaussian = gaussian_filter(gaussian, sigma=3)

print(gaussian)

xs = np.linspace(0, 10, 100)
y0 = np.linspace(0,0,100)
y0[35:65] = 1
y1 = np.convolve(y0, gaussian)
ax.plot(xs, y0)
ax.plot(xs, y1[100:-100])
# ax.plot(xs, y2[5:105])

plt.show()