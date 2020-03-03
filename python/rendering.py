import numpy as np
import nrrd
import matplotlib.pyplot as plt
from PIL import Image
import random

from scipy import signal
from scipy.ndimage import gaussian_filter
import scipy.stats as st

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration
import scipy

# filename = 'testdata.nrrd'
data = nrrd.read('../../../data/16-05-05/000.nrrd')
data = np.array(data)[0]
data = data.astype('float')
print(data.shape)

MIP = data[0,0:,0:,150]
MIP /= np.max(MIP)

# MIP = np.random.rand(MIP.shape[0], MIP.shape[1])

print(MIP.shape)
# for i in range(0,200):
#   MIP[100,i] = 2635
  # MIP[100,i+1] = 2635
# for i in range (450):
#   slice = data[0,:,:,i]
#   MIP += slice
print(MIP.shape)
print(MIP)
print(np.max(MIP))

a = np.arange(50, step=2).reshape((5,5))
gaussian = gaussian_filter(a, sigma=1)


def gkern(kernlen=11, nsig=1):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

print(gkern())

# recovered, remainder = signal.deconvolve(MIP, gkern())

def deconvolve(image, sigma):
  return restoration.richardson_lucy(image, gkern(nsig=sigma), iterations=30)
# deconvolved1 = 
# deconvolved2 = restoration.richardson_lucy(image, gkern(nsig=3), iterations=30)


def laplace(image, sigma):
  laplace = MIP
  laplace = scipy.ndimage.gaussian_filter(laplace, sigma)
  laplace = scipy.ndimage.laplace(laplace)
  laplace = -np.clip(laplace, None, 0)
  laplace = np.power(laplace, 0.5)
  laplace = scipy.ndimage.gaussian_filter(laplace, 1)
  return laplace
# img = Image.fromarray(MIP)
# print(np.asarray(img))
# plt.imshow(deconvolved)
fig, axes = plt.subplots(4, 2)
axes[0,0].imshow(MIP)
axes[1,0].imshow(deconvolve(MIP, 1))
axes[2,0].imshow(deconvolve(MIP, 3))
axes[3,0].imshow(deconvolve(MIP, 5))
axes[0,1].imshow(MIP)
axes[1,1].imshow(   laplace(MIP, 1))
axes[2,1].imshow(   laplace(MIP, 3))
axes[3,1].imshow(   laplace(MIP, 5))


for ax in axes.flatten():
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

fig.tight_layout()
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)


plt.show()


# import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
# from matplotlib.path import Path
# from matplotlib.patches import PathPatch

# delta = 0.025
# x = y = np.arange(-3.0, 3.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2

# fig, ax = plt.subplots()
# im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
#                origin='lower', extent=[-3, 3, -3, 3],
#                vmax=abs(Z).max(), vmin=-abs(Z).max())

# plt.show()