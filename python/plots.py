import matplotlib.pyplot as plt
import numpy as np

xs = np.arange(-3,3,0.01)
ys1 = np.exp(-xs**2.)
ys2 = np.exp(-xs**4.)
ys3 = np.exp(-(abs(xs)**3.))
ys4 = np.exp(-(abs(xs)**8))
# ys5 = np.exp(-(abs(xs)**1.0))
plt.plot(xs, ys1)
plt.plot(xs, ys2)
plt.plot(xs, ys3)
plt.plot(xs, ys4)
# plt.plot(xs, ys5)
# plt.title('generalized multivariate gaussian')
plt.show()