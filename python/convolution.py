import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
from scipy.optimize import minimize_scalar

np.convolve([1, 2, 3], [0, 1, 0.5])

fig = plt.figure()
ax = plt.axes()



gaussian = np.linspace(0,0,201)
gaussian[100] = 1
gaussian = gaussian_filter(gaussian, sigma=3)

# print(gaussian)

xs = np.linspace(0, 10, 100)
y0 = np.linspace(0,0,100)
y0[35:65] = 1
y1 = np.convolve(y0, gaussian)
# ax.plot(xs, y0)
# ax.plot(xs, y1[100:-100])


def diff(x):
  xs = np.linspace(0,1,1000)
  ys = np.zeros((1000,1))
  ys[450:550]=1
  yss = -ndimage.gaussian_laplace(ys, sigma=x)
  center = len(yss)//2
  v = yss[center] - 0.5*(yss[center-1] + yss[center+1])
  print(x, v, center, yss[center])
  if v > 0.0000000001:
    return np.nan
  return -v



xs = np.linspace(0,1,1000)
ys = np.zeros((1000,1))
ys[450:550]=1
center = -1

sig = minimize_scalar(diff, bounds=(28,30), method='bounded', tol=1e-30).x

yss = -ndimage.gaussian_laplace(ys, sigma=sig)
center = len(yss)//2
print(yss[center] - 0.5*(yss[center-1] + yss[center+1]))

ax.plot(xs,yss)
# ax.plot(xs,zs)
# ax.plot(xs, y2[5:105])

plt.show()