from mpl_toolkits import mplot3d

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE, MDS
from sklearn import manifold
import sklearn

from pathsio import load_paths
from matplotlib.widgets import CheckButtons

from scipy.spatial.transform import Rotation as R

import lasso

from sklearn.cluster import AgglomerativeClustering

# np.random.seed(0)


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

  xs = center[0] + np.linspace(0, length, 3)*vector[0]
  ys = center[1] + np.linspace(0, length, 3)*vector[1]
  zs = center[2] + np.linspace(0, length, 3)*vector[2]
  return np.array([xs,ys,zs]).T

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

def normalize(path, n=100, nth=1):
  path = path[::nth]
  xs = interp1d(np.linspace(0,n,len(path[:,0])), path[:,0], kind='linear')
  ys = interp1d(np.linspace(0,n,len(path[:,1])), path[:,1], kind='linear')
  zs = interp1d(np.linspace(0,n,len(path[:,2])), path[:,2], kind='linear')

  return np.array([xs(np.arange(0,n)), ys(np.arange(0,n)), zs(np.arange(0,n))]).T

# a set of transformations that convert a path to a point.
# think of this as being similar to a kernel. 


def to_point_naive(path):
  return path.flatten()

def transform_naive_minus_start(path):
  return (path - path[0])

def to_point_naive_minus_start(path):
  return (path - path[0]).flatten()

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
  print(path)
  path = path[1:-1] - path[0:-2]
  path /= np.linalg.norm(path, ord=2, axis=1, keepdims=True)
  # np.normalized(path, axis=2)

  mean = path.mean(axis=0)
  # print(mean)
  uu, dd, vv = np.linalg.svd(path - mean)
  v = vv[0]
  
  # print(mean)
  # print(v)

  # path0 = path[0]
  # path1 = path[-1]

  
  dotproductaxial = []

  for i in range(path.shape[0]):
    # print(path[i], v, np.dot(v, path[i]))
    dotproductaxial += [np.dot(v, path[i])]

  # dotproductaxial = np.tensordot(v, path)

  return np.array(dotproductaxial)

def to_point_length(path):
  length = 0
  for i in range(len(path)-1):
    length += np.linalg.norm(path[i] - path[i+1])
  return np.array([length])

def to_point_cog(path):
  point = np.average(path, axis=0)
  # print(path)
  return point

def to_point_norm(path):
  length = np.linalg.norm(path[-1] - path[0])
  return np.array([length])

def to_point_gradient(path):
  path = path[1:-1] - path[0:-2]
  path /= np.linalg.norm(path, ord=2, axis=1, keepdims=True)
  return path.flatten()

def to_point_curvature(path):
  curvature = 0
  length    = 0
  for i in range(len(path)-1):
    length += np.linalg.norm(path[i] - path[i+1])

  for i in range(len(path)-2):
    v1 = path[i+1] - path[i]
    v2 = path[i+2] - path[i+1]
    dot = v1.dot(v2)/np.linalg.norm(v1)
    curvature += dot;
  curvature /= length
  # print(curvature, length)
  return [curvature,0,0]


def transform_normalized_2d_pca(linein):
  line2 = linein.copy()
  line = linein.copy()
  # line[:,2] = 0
  # line = line[:,:]
  pca = sklearn.decomposition.PCA(n_components=2)
  line2 = pca.fit_transform(line)
  line2 -= line2[0]



  angle = np.arctan2(line2[-1][1], line2[-1][0])
  rotmatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  yavg = 0
  for point in line2:
    point2 = point.dot(rotmatrix)
    point[0] = point2[0]
    point[1] = point2[1]
    yavg += point[1]
  yavg /= len(line2)
  if(yavg < 0):
    line2[:,1] *= -1

  return line2

def to_point_normalized_2d_pca(line):
  return transform_normalized_2d_pca(line).flatten()

def metric_mindist(path1, path2):
  path1 = path1.reshape(-1,3)
  path2 = path2.reshape(-1,3)
  # path1 = normalize(path1, 5)
  # path2 = normalize(path2, 5)
  # print(path1)
  # print(path1)
  mindistsq = np.inf
  for x in path1[::20]:
    for y in path2[::20]:
      d0 = x[0]-y[0]
      d1 = x[1]-y[1]
      d2 = x[2]-y[2]

      distsq = d0*d0 + d1*d1 + d2*d2

      if distsq < mindistsq:
        mindistsq = distsq
  return np.sqrt(mindistsq)

def metric_simple(path1, path2):
  path1 = path1.reshape(-1,3)
  path2 = path2.reshape(-1,3)
  path1 -= path1[0]
  path2 -= path2[0]

  distances = np.linalg.norm(path2-path1, axis=1)
  distance = np.sum(distances)
  return distance

def metric_simple3(path1, path2):
  # print(path1)
  path1 = path1.reshape(-1,3)
  path2 = path2.reshape(-1,3)
  path1 -= path1[0]
  path2 -= path2[0]

  distances = np.linalg.norm(path2-path1, axis=1)
  distance = np.sum(distances)
  return distance

def metric_simple2(path1, path2):
  # print(path1)
  path1 = path1.reshape(-1,2)
  path2 = path2.reshape(-1,2)
  path1 -= path1[0]
  path2 -= path2[0]

  distances = np.linalg.norm(path2-path1, axis=1)
  distance = np.sum(distances)
  return distance

def metric_simple_2d(path1, path2):
  path1 = path1.reshape(-1,3)
  path2 = path2.reshape(-1,3)
  path1 = sklearn.decomposition.PCA(n_components=3).fit_transform(path1)
  path2 = sklearn.decomposition.PCA(n_components=3).fit_transform(path2)
  path1 -= path1[0]
  path2 -= path2[0]

  distances = np.linalg.norm(path2-path1, axis=1)
  distance = np.sum(distances)
  return distance



fig = plt.figure()
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
ax2 = fig.add_subplot(3, 4, 2, projection='3d')
ax3 = fig.add_subplot(3, 4, 3)
ax4 = fig.add_subplot(3, 4, 4)
ax5 = fig.add_subplot(3, 4, 5)
ax6 = fig.add_subplot(3, 4, 6)
ax7 = fig.add_subplot(3, 4, 7)
ax8 = fig.add_subplot(3, 4, 8)
ax9 = fig.add_subplot(3, 4, 9)
ax10 = fig.add_subplot(3, 4, 10)
ax11 = fig.add_subplot(3, 4, 11)
ax12 = fig.add_subplot(3, 4, 12)
plaxes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

def plot(ax, path):
  ax.plot3D(path[:,0], path[:,1], path[:,2])


# def to_point
paths=[None]*3
paths[0] = np.array([[10, 20], [10, 20], [10, 20]]).T
# paths[0] = normalize(paths[0])
# print(to_point_curvature(paths[0]))
# print(to_point_lastminusfirst(paths[0]))
# print(to_point_startpoint(paths[0]))
# print(to_point_endpoint(paths[0]))
# print(normalize(path0))

# paths[1] = spring([10,10,10], [0,0,1], radius=1.0, loops=10, height=10, noise=0, nsample=200)
# paths[1] = smooth(paths[1])
# print(to_point_curvature(paths[1]))

# paths[2] = spring([15,10,10], [0,0,1], radius=2.0, loops=3, height=10, noise=0, nsample=200)

# print(paths[1])
# for i in range(8000):
#   x = metric_mindist(to_point_naive(paths[1]), to_point_naive(paths[2]))
# print(x)
# exit(0)
# y = metric_simple_2d(to_point_naive(paths[1]), to_point_naive(paths[2]))
# print(x,y)
# print(paths[1])
# exit()

# plot(ax1, paths[1])
# plot(ax1, paths[2])
# plt.show()
# exit(0)
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

# plt.ion()
# plt.show()

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

# display helper functions
def color_of_line(line):
  color = line[-1] - line[0]
  # print('norm = ', line[-1], line[0], color)
  color /= np.linalg.norm(color)
  color = (1.0+color)/2.0
  return color  


# paths += gen_paths_torus()
# print(to_point_fishingline_axial(circles[-1]))
# paths += gen_paths_longitude(orient=1)
# print(to_point_fishingline_axial(circles[-1]))
# plot(ax2, circles[-1])
# circles += gen_paths_longitude(orient=-1)

  # plot(ax, path)

# for i in range(0,10):
#   path = spring([15,10,10], random_vector(), radius=1.0, noise=0)
#   circles += [path]
#   plot(ax, path)

print('show')

paths = load_paths()
paths += [normalize(line([0,50,50], [1,0,0], 100))]
# print(line([0,50,50], [1,0,0], 100))
print('loaded ', len(paths), ' paths')
paths2d     = []


for path in paths:
  paths2d += [path[:,0:2]]

# print(paths[0])
# print(paths2d[0])
# print(paths)
# print(paths[0])
# print(to_point_dotproduct_axial(transverse))
# print(paths)
# paths = np.append(paths, transverse)
# print(paths)
pathcolors = []
for path in paths:
  pathcolors += [color_of_line(path)]
pathspt    = []
pathspt2d  = []

for i in range(len(paths)):
  path = paths[i]
  path2d = paths2d[i]
  path = normalize(path)
  path = smooth(path, sigma=5)
  # path = normalize(path, nth=10)

  # pathspt += [to_point_length(path)+to_point_fishingline(path)]
  pathspt += [to_point_naive_minus_start(path)]

  color = path[-1] - path[1]
  color /= np.linalg.norm(color)
  color = (1.0+color)/2.0
  # print(color)
  ax1.plot3D(path[:,0], path[:,1], path[:,2], color=color)
# for path in paths2d:
  # ax3.plot( path[:,0], path[:,1], color=color)
  # ax7.plot( path[:,0], path[:,2], color=color)
  # ax11.plot(path[:,1], path[:,2], color=color)

  paths[i] = path
  paths2d[i] = path2d

# paths = paths2
# print(circlespt[-1])

# print(np.array(circlespt).shape)
# print(circles)


# plt.show()
# plt.draw()
# plt.pause(0.001)

# def norm1(x, y):
#   return np.sum(np.abs(x-y))

# print('statistics')
X = np.array(pathspt)
# print(X.shape)
# print(X)
# print('MDS')
# X1 = manifold.MDS(n_components=2).fit_transform(X)
# print('Isomap')
# X2 = manifold.Isomap(n_components=2).fit_transform(X)
# print('LLE')
# X3 = manifold.LocallyLinearEmbedding(n_components=2).fit_transform(X)
# random.seed(0)
# print('tsne perp=20')
# print(X.reshape(-1,3))
# X4 = manifold.TSNE(n_components=2, n_jobs=-1, n_iter=1000, perplexity=20).fit_transform(X)
# random.seed(0)
print('tsne perp=10')
X5 = manifold.TSNE(n_components=2, n_jobs=-1, n_iter=1000, perplexity=10).fit_transform(X)
# X5 = manifold.TSNE(n_components=2).fit_transform(X)


# def hover(event):
  # if event.inaxes == ax3:
    # print(event.ind)
linecolors = []

def on_pick(event):
  # print(event)
  # print('picked!')
  # print(event)
  # print(event.ind)
  toplot = []
  toplot_zero = []
  for c in event.ind:
    toplot += [paths[c]]
    toplot_zero += [transform_naive_minus_start(paths[c])]

  average = None
  if(len(toplot) > 0):
    toplot_array = np.array(toplot_zero)
    average = np.average(toplot_array, axis=0)
    print(average)
    print(average.shape)
    print(toplot[0].shape)
    # average = np.array(np.zeros(toplot[0].shape))
    # for line in toplot:
    #   average
    toplot += [average]

  ax2.clear()
  ax4.clear()
  ax8.clear()
  ax12.clear()
  ax5.clear()
  ax3.set_aspect('equal', adjustable='box')
  ax4.set_aspect('equal', adjustable='box')
  ax7.set_aspect('equal', adjustable='box')
  ax8.set_aspect('equal', adjustable='box')
  ax11.set_aspect('equal', adjustable='box')
  ax12.set_aspect('equal', adjustable='box')
  ax5.set_aspect('equal', adjustable='box')
  for line in paths:
    ax2.plot3D(line[:,0], line[:,1], line[:,2], color=[0.8,0.8,0.8,0.8])
  # for line in paths:
  #   ax4.plot(line[:,0], line[:,1], color=[0.8,0.8,0.8,0.8])
  #   ax8.plot(line[:,0], line[:,2], color=[0.8,0.8,0.8,0.8])
  #   ax12.plot(line[:,1], line[:,2], color=[0.8,0.8,0.8,0.8])
  # for line in paths2d:
  #   ax5.plot(line[:,0], )
  for line in toplot:
    
    ax2.plot3D(line[:,0], line[:,1], line[:,2], color=color_of_line(line))
    ax4.plot(line[:,0], line[:,1], color=color_of_line(line))
    ax8.plot(line[:,0], line[:,2], color=color_of_line(line))
    ax12.plot(line[:,1], line[:,2], color=color_of_line(line))
    
    line2 = transform_normalized_2d_pca(line)
 
    
    ax5.plot(line2[:,0], line2[:,1], color=color_of_line(line))
  # ax8.clear()

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

# ax3.scatter(X1[:,0], X1[:,1], picker=True, color=pathcolors)
# ax5.scatter(X2[:,0], X2[:,1], picker=True, color=pathcolors)
# ax6.scatter(X3[:,0], X3[:,1], picker=True, color=pathcolors)
# ax9.scatter(X4[:,0], X4[:,1], picker=True, color=pathcolors, s=10)
pts = ax10.scatter(X5[:,0], X5[:,1], color=pathcolors, s=10)
selector = lasso.SelectFromCollection(ax10, pts, on_pick)

clustering = AgglomerativeClustering(distance_threshold=None, n_clusters=20).fit(X5)
print(clustering.n_clusters, 'clusters')

get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
hexcolors = get_colors(1000)

# print(hexcolors)
linecolors = []
for l in clustering.labels_:
  linecolors += [hexcolors[l]]

# print(colors)

# print(X5.shape, len(colors))

pts = ax6.scatter(X5[:,0], X5[:,1], color=linecolors, picker=True, s=10)
selector = lasso.SelectFromCollection(ax6, pts, on_pick)
# selector = lasso.SelectFromCollection(ax6, pts, on_pick)

ax8.label = 'tsne'
print('done')
# line([2,2,2],[1,-1,0], 5)

fig.canvas.mpl_connect('pick_event', on_pick)

# plt.draw()
# plt.pause(0.001)
# plt.ioff()
plt.show()