from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax = plt.axes(projection='3d')

# Data for a three-dimensional line

def load_paths():
  f= open("../rsc/store/s-home-ashwin-data-miniventral2-000.nrrd.paths.txt","r")
  # f= open("../rsc/store/s-home-ashwin-data-16-05-05-000.nrrd.paths.txt","r")
  contents = f.read()
  lines = contents.split('\n')
  lines = [line.split('; ') for line in lines]
  lines = [[vector.split(' ') for vector in line] for line in lines]

  # xs = [[]]
  # ys = [[]]
  # zs = [[]]

  output = []
  # for line in lines:
  #   for vector in line:
  #     print(vector)


  for line in lines:
    line1 = []
    for vector in line:
      if vector != ['']:
        # print('vector2', float(vector[0]), float(vector[1]), float(vector[2]))
        vector1 = [float(vector[0]), float(vector[1]), float(vector[2])]
        # print(vector1)
        line1 += [vector1]
    if line1 != []:
      output += [np.array(line1)]
      # print(np.array(line1))
    # output += [line1]
    # output += [array]
    # xss = np.array()
    # yss = np.array()
    # zss = np.array()
    # for vector in line:
    #   if(vector[0] != ''):
    #     xss += [float(vector[0])]
    #     yss += [float(vector[1])]
    #     zss += [float(vector[2])]
    # xs += [xss]
    # ys += [yss]
    # zs += [zss]

  # xs = np.array(xs[1:-1])
  # ys = np.array(ys[1:-1])
  # zs = np.array(zs[1:-1])
  return (output)

# def sphere_paths():



# paths = load_paths()
# print(paths)
# print(np.array([[[1,2,3]]]).shape)
# exit()

# print(xs[-2], ys[-2], zs[-2])
# linexs = 
# 

# print([x for x in [1,2,3,4]])

# n = len(xs) - 1

# print('n = ', n)

def sample_spherical(npoints, ndim=3):
  vec = np.random.randn(ndim, npoints)
  vec /= np.linalg.norm(vec, axis=0)
  return vec

# def line(center, direction):


zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)

# transformation: only first and last points
def transform1():
  for i in range(1,n):
    print(i, '=', xs[i])
    xs[i]=[xs[i][0], xs[i][-1]]
    ys[i]=[ys[i][0], ys[i][-1]]
    zs[i]=[zs[i][0], zs[i][-1]]
    print(i, '>', xs[i])

# transformation: line, but centered at 0
def transform2():
  for i in range(1,n):
    print(i, '=', xs[i])
    xs[i]=[0, xs[i][-1] - xs[i][0]]
    ys[i]=[0, ys[i][-1] - ys[i][0]]
    zs[i]=[0, zs[i][-1] - zs[i][0]]
    print(i, '>', xs[i])

def smooth(path, sigma=2):
  xs = path[:,0]
  ys = path[:,1]
  zs = path[:,2]

  xs = gaussian_filter(xs, sigma=sigma)
  ys = gaussian_filter(ys, sigma=sigma)
  zs = gaussian_filter(zs, sigma=sigma)

  return np.array([xs,ys,zs]).T

# transform1()

# sphere = sample_spherical(100)
# ax.scatter(sphere[0], sphere[1], sphere[2], s=100, c='r', zorder=10)
# for i in range(len(paths)):
  # path = paths[i]
  # path = smooth(path, sigma=15)
  # print('pathxs', path[:,0])
  # ax.plot3D(path[:,0], path[:,1], path[:,2])
  # ax.plot3D(xs[-3], ys[-3], zs[-3])

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

# plt.show()