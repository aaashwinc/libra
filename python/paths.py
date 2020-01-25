from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

# ax = plt.axes(projection='3d')

# Data for a three-dimensional line


f= open("../rsc/store/s-home-ashwin-data-16-05-05-000.nrrd.paths.txt","r")

contents = f.read()
lines = contents.split('\n')
lines = [line.split('; ') for line in lines]
lines = [[vector.split(' ') for vector in line] for line in lines]
# lines = [[]]
# lines = [[vector.split(' ') for vector in line] for line in lines]

xs = [[]]
ys = [[]]
zs = [[]]
for line in lines:
  for vector in line:
    print(vector)


for line in lines:
  xss = []
  yss = []
  zss = []
  for vector in line:
    if(vector[0] != ''):
      xss += [float(vector[0])]
      yss += [float(vector[1])]
      zss += [float(vector[2])]
  xs += [xss]
  ys += [yss]
  zs += [zss]

print(xs[-2], ys[-2], zs[-2])
# linexs = 
# 

print([x for x in [1,2,3,4]])

n = len(xs) - 1

print('n = ', n)

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

# transform1()

sphere = sample_spherical(100)
# ax.scatter(sphere[0], sphere[1], sphere[2], s=100, c='r', zorder=10)
for i in range(1,n):
  ax.plot3D(xs[i], ys[i], zs[i])
  # ax.plot3D(xs[-3], ys[-3], zs[-3])

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

plt.show()