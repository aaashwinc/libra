from mpl_toolkits import mplot3d

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE, MDS
from sklearn import manifold
import sklearn

from pathsio import load_paths

def sample_spherical(npoints, ndim=3):
  vec = np.random.randn(ndim, npoints)
  vec /= np.linalg.norm(vec, axis=0)
  return vec

def random_vector():
  vec = np.random.randn(3)
  vec /= np.linalg.norm(vec, axis=0)
  return vec


def random_point(boundmax):
  vec = np.random.randn(3)
  vec[0] *= boundmax[0]
  vec[1] *= boundmax[1]
  vec[2] *= boundmax[2]
  print(vec)
  # vec /= np.linalg.norm(vec, axis=0)
  return vec

def line(center, vector, length):
  print('center=', center, 'vector=', vector)

  xs = center[0] + np.linspace(0, length, 2)*vector[0]
  ys = center[1] + np.linspace(0, length, 2)*vector[1]
  zs = center[2] + np.linspace(0, length, 2)*vector[2]
  print(xs, ys, zs)
  return [xs,ys,zs]

def spring(center, vector, radius, loops=1, height=0, ovalu=1, ovalv=1, nsample=100, noise=0.05, orthovector=random_vector()):

  # mutually orthogonal normal vectors
  vector /= np.linalg.norm(vector)
  vector2 = orthovector
  vector2 = np.cross(vector, vector2)
  vector3 = np.cross(vector, vector2)

  vector2 /= np.linalg.norm(vector2)
  vector3 /= np.linalg.norm(vector3)

  iline = np.linspace(0, 3.141592*2*loops, nsample)
  uline = np.sin(iline)
  vline = np.cos(iline)
  hline = np.linspace(0, height, nsample)

  # print(iline)
  # print(uline)
  # print(vline)
  # print("~~~")

  line = []
  for (i,u,v,h) in zip(iline, uline, vline, hline):
    p = center + u*ovalu*vector2 + v*ovalv*vector3 + h*vector
    p += noise*random_vector()
    # print(p)
    line += [p]
  # for (x,y,z) in xline,yline,zline:
    # pass

  # zline = np.linspace(0,0,10)

  # line = np.array([xline, yline, zline])
  # line = line.reshape(3,-1)
  line = np.array(line)
  # print(line)
  xs = (line[:,0])
  ys = (line[:,1])
  zs = (line[:,2])

  # print(vector, vector2, vector3)
  # print(np.dot(vector, vector3))

  # xs =

  # xs = np.linspace
  # xs = center[0] + np.linspace(0, length, 2)*vector[0]
  # ys = center[1] + np.linspace(0, length, 2)*vector[1]
  # zs = center[2] + np.linspace(0, length, 2)*vector[2]

  # print(xs, ys, zs)
  # pass
  # xs = []
  # ys = []
  # zs = []

  # return [[0, vector[0], 0, vector2[0], 0, vector3[0]],
  #         [0, vector[1], 0, vector2[1], 0, vector3[1]],
  #         [0, vector[2], 0, vector2[2], 0, vector3[2]]]

  return np.array([xs,ys,zs]).T

def smooth(path, sigma=2):
  xs = path[:,0]
  ys = path[:,1]
  zs = path[:,2]

  xs = gaussian_filter(xs, sigma=sigma)
  ys = gaussian_filter(ys, sigma=sigma)
  zs = gaussian_filter(zs, sigma=sigma)

  return np.array([xs,ys,zs]).T

from scipy.interpolate import interp1d

def normalize(path, n=100):
  xs = interp1d(np.linspace(0,n,len(path[:,0])), path[:,0], kind='linear')
  ys = interp1d(np.linspace(0,n,len(path[:,1])), path[:,1], kind='linear')
  zs = interp1d(np.linspace(0,n,len(path[:,2])), path[:,2], kind='linear')

  return np.array([xs(np.arange(0,n)), ys(np.arange(0,n)), zs(np.arange(0,n))]).T

# a set of transformations that convert a path to a point.
# think of this as being similar to a kernel. 


def to_point_naive(path):
  return path.flatten()

def to_point_firstlast(path):
  return np.array([path[0][0],  path[0][1],  path[0][2],
                   path[-1][0], path[-1][1], path[-1][2]])

def to_point_lastminusfirst(path):
  return np.array([path[-1,0] - path[0,0],
                   path[-1,1] - path[0,1],
                   path[-1,2] - path[0,2]])

def to_point_endpoint(path):
  return path[-1,:]

def to_point_startpoint(path):
  return path[0,:]

def to_point_fishingline(path):
  path0 = path[0]
  path1 = path[-1]

  center = np.linspace(
    [path0[0], path0[1], path0[2]],
    [path1[0], path1[1], path1[2]],
    path.shape[0]
  )

  fishingline = np.linalg.norm(path-center, axis=1)

  return fishingline

def to_point_fishingline_axial(path):

  mean = path.mean(axis=0)
  uu, dd, vv = np.linalg.svd(path - mean)
  v = vv[0]
  
  # print(mean)
  # print(v)

  path0 = path[0]
  path1 = path[-1]
  # print('~~')
  # print(path0)
  # print(path1)

  path0 = mean + v*((path0 - mean).dot(v))
  path1 = mean + v*((path1 - mean).dot(v))

  # print(path0)
  # print(path1)
  # print('~~')

  center = np.linspace(
    [path0[0], path0[1], path0[2]],
    [path1[0], path1[1], path1[2]],
    path.shape[0]
  )

  # for i in range(path.shape[0]):
  #   print(path[i], center[i], np.linalg.norm(path[i]-center[i]))

  fishinglineaxial = np.linalg.norm(path-center, axis=1)

  return fishinglineaxial
  # print(fishingline)


def to_point_dotproduct_axial(path):

  path = path[1:-1] - path[0:-2]
  path /= np.linalg.norm(path, ord=2, axis=1, keepdims=True)
  # np.normalized(path, axis=2)

  mean = path.mean(axis=0)
  uu, dd, vv = np.linalg.svd(path - mean)
  v = vv[0]
  
  # print(mean)
  # print(v)

  path0 = path[0]
  path1 = path[-1]

  
  dotproductaxial = []

  for i in range(path.shape[0]):
    # print(path[i], v, np.dot(v, path[i]))
    dotproductaxial += [np.dot(v, path[i])]

  # dotproductaxial = np.tensordot(v, path)

  return np.array(dotproductaxial)

fig = plt.figure()
ax  = fig.add_subplot(3, 3, 1, projection='3d')
ax2 = fig.add_subplot(3, 3, 2, projection='3d')
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7, projection='3d')

def plot(ax, path):
  ax.plot3D(path[:,0], path[:,1], path[:,2])

# def to_point
paths=[None]*3
paths[0] = np.array([[10, 20], [10, 20], [10, 20]]).T
paths[0] = normalize(paths[0])
# print(to_point_firstlast(paths[0]))
# print(to_point_lastminusfirst(paths[0]))
# print(to_point_startpoint(paths[0]))
# print(to_point_endpoint(paths[0]))
# print(normalize(path0))

paths[1] = spring([10,10,10], [0,0,1], radius=1.0, loops=0.5, height=10, noise=0, nsample=200)
paths[1] = smooth(paths[1])
# print(to_point_fishingline(paths[1]))

paths[2] = spring([15,10,10], [0,0,1], radius=2.0, loops=3, height=10, noise=0, nsample=200)

# print("fishingline transformation")
# print(to_point_fishingline_axial(paths[0]))
# print(to_point_fishingline_axial(paths[1]))
# print(to_point_fishingline_axial(paths[2]))

# print("dotproduct_axial transformation")
# print(to_point_dotproduct_axial(paths[0]))
# print(to_point_dotproduct_axial(paths[1]))
# print(to_point_dotproduct_axial(paths[2]))

# print(path1)


# sphere = sample_spherical(100)
# ax.scatter(sphere[0], sphere[1], sphere[2], s=100, c='r', zorder=10)

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)




# path = line([2,2,2], [1,1,1], 5)
# for i in range(0,100):
#   # path = spring(random_point([10,10,10]), random_vector(), radius=random.uniform(4,10), loops=random.uniform(0.5,4), height=0)
#   path = spring([10,10,10], random_vector(), radius=1.0)
#   ax.plot3D(path[0], path[1], path[2], 'gray')

# ax.plot3D(path1[:,0], path1[:,1], path1[:,2])
# print('add')
# for path in paths:
#   plot(ax, path)
# plot(ax, path0)
# plot(ax, path1)
##### generate orbits on/around sphere:

print('show')
# plt.ion()
# plt.show()

circles   = []
circlespt = []

def gen_paths_longitude(center=[0,0,0], orient = 1):
  paths = []
  for i in range(0,30):
    path = spring(center, [np.sin(i),np.cos(i),0], orthovector=[0,0,orient], loops=0.5, radius=1.0, noise=0)
    paths += [path]
  return paths

  # plot(ax, path)

# for i in range(0,30):
#   path = spring([10,10,10], [np.sin(i),np.cos(i),0], orthovector=[0,0,-1], loops=0.5, radius=1.0, noise=0)
#   circles += [path]

def gen_paths_torus(center=[0,0,0], radius1=4, radius2=1):
  paths = []
  for i in range(0,30):
    center  = [np.sin(i)*radius1,np.cos(i)*radius1,0]
    path = spring(center, [-np.cos(i), np.sin(i), 0], orthovector=[0,0,1], radius = radius2, noise=0)
    path += np.array(center)
    paths += [path]
  return paths

circles += gen_paths_torus()
print(to_point_fishingline_axial(circles[-1]))
circles += gen_paths_longitude(orient=1)
print(to_point_fishingline_axial(circles[-1]))
plot(ax2, circles[-1])
circles += gen_paths_longitude(orient=-1)

  # plot(ax, path)

# for i in range(0,10):
#   path = spring([15,10,10], random_vector(), radius=1.0, noise=0)
#   circles += [path]
#   plot(ax, path)


circles = load_paths()

for circle in circles:
  circle = normalize(circle)
  circle = smooth(circle, sigma=5)
  circlespt += [to_point_naive(circle)]
  plot(ax, circle)

# print(circlespt[-1])

print(np.array(circlespt).shape)
# print(circles)


# plt.show()
# plt.draw()
# plt.pause(0.001)

print('tsne')
X = np.array(circlespt)
print(X.shape)
X1 = manifold.MDS(n_components=2).fit_transform(X)
X2 = manifold.Isomap(n_components=2).fit_transform(X)
X3 = manifold.LocallyLinearEmbedding(n_components=2).fit_transform(X)
X4 = sklearn.decomposition.PCA(n_components=2).fit_transform(X)
X5 = sklearn.decomposition.PCA(n_components=3).fit_transform(X)


# def hover(event):
  # if event.inaxes == ax3:
    # print(event.ind)



axes = [ax, ax2, ax3, ax4, ax5, ax6, ax7]

def on_pick(event):
  # print(event)
  # print('picked!')
  # print(event.ind)
  toplot = []
  for c in event.ind:
    toplot += [circles[c]]
  ax2.clear()
  for line in toplot:
    plot(ax2, line)

  plt.ion()
  plt.draw()
  plt.pause(0.001)
  plt.ioff()
  # print(event.inaxes)
  # if event.inaxes == ax3:
  #   print(event.ind)

# fig.canvas.mpl_connect("motion_notify_event", hover)

# print(X4)
# V = pca.components_
# X3 = manifold.TSNE(n_components=2).fit_transform(X)

ax3.scatter(X1[:,0], X1[:,1], picker=True)
ax4.scatter(X2[:,0], X2[:,1], picker=True)
ax5.scatter(X3[:,0], X3[:,1], picker=True)
ax6.scatter(X4[:,0], X4[:,1], picker=True)
ax7.scatter(X5[:,0], X5[:,1], X5[:,2], picker=True)
print('done')
# line([2,2,2],[1,-1,0], 5)

fig.canvas.mpl_connect('pick_event', on_pick)

# plt.draw()
# plt.pause(0.001)
# plt.ioff()
plt.show()