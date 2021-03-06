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
from matplotlib.widgets import RadioButtons


from scipy.spatial.transform import Rotation as R

import lasso

from sklearn.cluster import AgglomerativeClustering

from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import convolve1d

import scipy as scipy

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

  return np.array([xs(np.linspace(0,n,n)), ys(np.linspace(0,n,n)), zs(np.linspace(0,n,n))]).T


def normalize_avg(path, n=100, nth=1):
  path = path[::nth]
  window_size = max(1, path.shape[0] / n)
  window_size = 2
  window = [1./window_size] * int(window_size)

  # path[:,0] = convolve1d(path[:,0], window, mode='nearest')
  # path[:,1] = convolve1d(path[:,1], window, mode='nearest')
  # path[:,2] = convolve1d(path[:,2], window, mode='nearest')

  print('win', window)
  # path[:,0] = np.convolve(path[:,0],window, mode='same')
  # path[:,1] = np.convolve(path[:,1],window, mode='same')
  # path[:,2] = np.convolve(path[:,2],window, mode='same')
  xs = interp1d(np.linspace(0,n,len(path[:,0])), path[:,0], kind='linear')
  ys = interp1d(np.linspace(0,n,len(path[:,1])), path[:,1], kind='linear')
  zs = interp1d(np.linspace(0,n,len(path[:,2])), path[:,2], kind='linear')

  return np.array([xs(np.linspace(0,n-1,n)), ys(np.linspace(0,n-1,n)), zs(np.linspace(0,n-1,n))]).T



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

def splinify(path, knots=4):
  n = len(path)
  path = smooth(path, sigma=2)
  path = normalize(path, n=knots)
  t = np.arange(0, path.shape[0])
  cs = CubicSpline(t, path, axis=0, bc_type='clamped')
  return cs(np.linspace(0, path.shape[0]-1, n))



def to_point_spline3(path):
  return to_point_naive_minus_start(path)
  # # print('r', path)
  # # path = np.array([0,0,0,0,1,0,0,0,0] * 3).reshape(-1,3)
  # # print('r', path)
  # # print('n', normalize(path, n=6))
  # # print('a', normalize_avg(path, n=6))
  # # path = normalize_avg(path, n=6)
  path = smooth(path, sigma=2)
  path = normalize(path, n=10)
  path = path - path[0]
  t = np.arange(0, path.shape[0])
  cs = CubicSpline(t, path, axis=0, bc_type='clamped')
  # # print('coef',      cs.c)
  # # print('shap', cs.c.shape)
  # # print('coef_flat', cs.c.reshape((-1)))
  # # print('path', path)
  # # print('coef_resh', cs.c.reshape((4, -1, 3)))
  # # print('xs', cs.x.reshape((-1)))
  # # print('xs', np.arange(0, cs.c.reshape((-1)).shape[0]/12 + 1))
  # # print('xxs', cs.c.reshape((-1)).shape[0], cs.c.reshape((-1)).shape[0]/12)


  # print(path)
  # print(path[:,0])
  # knots = (np.linspace(0, 1, 15, endpoint=True))

  # tt = np.linspace(0,1,len(path[:,0]))
  # path = np.linspace(0,1,len(path[:,0]))

  # timeList = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
  # signal = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
  # myKnots= [.05, .3, .6, .95]

  # spl = scipy.interpolate.splprep([timeList, signal], t=myKnots, k=2)
# 
  # print(tt)
  # print(knots)
  # # tck,u = scipy.interpolate.splprep([tt, tt, tt], k=2, t=knots)
  # res = scipy.interpolate.splrep( timeList, signal, t=myKnots, k = 2 )
  # print(res)
  # print('```')
  # print(res[0])
  # # print(tck[0])
  # # print('hello!')
  # exit(0)

  # tt = np.linspace(0,1,len(path[:,0]))
  # dydx = smooth(path[1:], sigma=2) - smooth(path[:-1], sigma=2)
  # print(dydx)
  # dydx = np.vstack((np.array([0,0,0]), dydx))
  # print(dydx)

  # chs = CubicHermiteSpline(tt, path, dydx)

  # print(chs)

  # exit(0)

  # return [0,0,0,0,0]
  # print('spline', cs.c.reshape(-1))
  return cs.c.reshape((-1))

# def point_spline3_to_path(coef):
#   coef = coef.reshape((4, -1, 4))
#   xs   = np.arange(0, cs.c.reshape((-1)).shape[0]/12 + 1)
#   axis = 0

#   cs = CubicSpline()

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
# paths=[None]*3
# paths[0] = np.array([[10, 20], [10, 20], [10, 20]]).T
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

# paths_raw = load_paths("../rsc/store/s-home-ashwin-data-miniventral2-000.nrrd.paths.txt")
# paths_raw = load_paths("../rsc/store/s-home-ashwin-data2-16-05-05-000.nrrd.paths.txt")
# paths_raw = load_paths("/home/ashwin/repo/tracks/build/tracking--home-ashwin-data-16-05-05-tracking-GMEMfinalResult_frame0-0-219-.xml-paths.txt")
paths_raw = load_paths("/home/ashwin/repo/tracks/build/tracking--home-ashwin-data2-tracking-17-05-01-000000-0-219-.xml-paths.txt")
# paths_raw = load_paths("../rsc/store/s-home-ashwin-data-17-05-01-small-100.nrrd.paths.txt")
paths = []
for path in paths_raw:
  # print(path.shape)
  if path.shape[0] > 99:
    paths += [path]


paths += [normalize(line([50,50,50], [1,0,0], 100))]
paths += [normalize(line([50,50,50], [0,1,0], 100))]
paths += [normalize(line([50,50,50], [0,0,1], 100))]


paths = paths[6:]


#TEST


# paths = []

# for i in range(20):
#   paths += [normalize(line([50,random.uniform(40,60),random.uniform(40,60)], [1,0,0], 100))]

#   paths += [normalize(line([50,random.uniform(40,60),random.uniform(40,60)], [2,0,0], 100))]


#/TEST
# print(line([50,50,50], [1,0,0], 100))
# print(line([50,50,50], [0,1,0], 100))
# print(line([50,50,50], [0,0,1], 100))
# print('loaded ', len(paths), ' paths')


## todo put this in paper


## calculate error:
print(np.max(np.array(paths[:-3]), axis=(0,1)))
# exit()
for k in range(2, 15):
  error = 0.0
  ppn = 0.0
  for q in paths[:-3]:
    p = q*10
    e = 0.0
    pn = 0.0
    z = path*0
    s = splinify(p, k)
    # s = smooth(path, 0)
    # s = path
    for t in range(len(p)):
      # print(p[t], z[t], s[t])
      e += np.linalg.norm(p[t]-s[t])
      pn = pn+1
    if pn>0:
      error += e / pn
    ppn = ppn+1
    # print('')
  error /= ppn
  print('k,error =', k, (error))
exit(0)




# print(paths[0])
# paths2d     = []

# paths = gen_paths_torus()
# for path in paths:
#   paths2d += [path[:,0:2]]

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
  # path2d = paths2d[i]
  path = normalize(path)
  path = smooth(path, sigma=1)
  # path = normalize(path, nth=10)

  # pathspt += [to_point_length(path)+to_point_fishingline(path)]
  # pathspt += [to_point_naive_minus_start(path)]
  pathspt += [to_point_spline3(path)]

  # print(color)
  ax1.plot3D(path[:,0], path[:,1], path[:,2], color=color_of_line(path))

# for path in paths2d:
  # ax3.plot( path[:,0], path[:,1], color=color)
  # ax7.plot( path[:,0], path[:,2], color=color)
  # ax11.plot(path[:,1], path[:,2], color=color)

  paths[i] = path
  # paths2d[i] = path2d

# exit(0)

xs= []
ys= []
zs= []
cs= []
for i in range(len(paths)):
  path = paths[i]
  xs = [path[0,0]]
  ys = [path[0,1]]
  zs = [path[0,2]]
  # cs += [color_of_line(path)]
  ax2.scatter(xs, ys, zs, color=color_of_line(path), s=5)
# print(xs)



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
print('LLE')
X1 = manifold.LocallyLinearEmbedding(n_components=2).fit_transform(X)
print('Isomap')
X2 = manifold.Isomap(n_components=2).fit_transform(X)
X6 = sklearn.decomposition.PCA(n_components=2).fit_transform(X)
# X2 = X6
# X1 = X2
# print('LLE')
# X3 = manifold.LocallyLinearEmbedding(n_components=2).fit_transform(X)
# random.seed(0)
# print('tsne perp=20')
# print(X.reshape(-1,3))
# X4 = manifold.TSNE(n_components=2, n_jobs=-1, n_iter=1000, perplexity=20).fit_transform(X)
# random.seed(0)
# X = X[0:10]
# print(X.shape)
print('tsne perp=10')
X5 = manifold.TSNE(n_components=2, n_iter=1000, perplexity=20, n_jobs=-1).fit_transform(X)
print('done')


ax2.set_yticklabels([])
ax2.set_xticklabels([])
# X5 = manifold.TSNE(n_components=2).fit_transform(X)


# def hover(event):
  # if event.inaxes == ax3:
    # print(event.ind)
linecolors = []

pcascatter = None
pcalines   = None
clustering = None
# def on_hover(event):
#   if event.inaxes == ax9:
#     cont, ind = pcascatter.contains(event)
#     print(ind)
#     ax5.clear()
#     for i in ind['ind']:
#       print(i)
#       path = pcalines[i]

def render_paths(inds):
  toplot = []
  toplot_zero = []
  for c in inds:
    # print('pick', c)
    toplot += [paths[c]]
    toplot_zero += [transform_naive_minus_start(paths[c])]

  average = None
  if(len(toplot) > 0):
    toplot_array = np.array(toplot_zero)
    average = np.average(toplot_array, axis=0)
    # print(average)
    # print(average.shape)
    # print(toplot[0].shape)
    # average = np.array(np.zeros(toplot[0].shape))
    # for line in toplot:
    #   average
    toplot += [average]
    # aspoints = []
    # for i in toplot:
    #   aspoints += [to_point_naive_minus_start(i)]
    # print(np.array(aspoints))
    # print(np.array(aspoints).shape)
    # X2 = sklearn.decomposition.PCA(n_components=1).fit_transform(np.array(aspoints))
    # ax9.clear()
    # global pcascatter
    # global pcalines
    # pcalines   = toplot
    # ax5.clear()
    # for path in toplot:
      # print('path')
      # ax5.plot(path[:,0], path[:,1], color=color_of_line(path))
    
    # ax9.scatter(X2[:,0], np.zeros(X2.shape[0]), picker=True, s=10)
    # print('pcascatter', pcascatter)

  ax2.clear()
  ax4.clear()
  ax8.clear()
  ax12.clear()
  ax5.clear()
  # ax3.set_aspect('equal', adjustable='box')
  ax4.set_aspect('equal', adjustable='box')
  # ax7.set_aspect('equal', adjustable='box')
  ax8.set_aspect('equal', adjustable='box')
  # ax11.set_aspect('equal', adjustable='box')
  ax12.set_aspect('equal', adjustable='box')
  # ax5.set_aspect('equal', adjustable='box')
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

def on_pick(event):
  render_paths(event.ind)

check_buttons = None

def on_check(label):
  # print('select', label)
  index = label.index(label)

  checks = check_buttons.get_status()
  labels = np.where(np.array(checks) == True)[0]
  # labels = [int(label)]

  # print('search for ', labels)

  # print(label, index, type(label), type(index))
  # print(clustering.labels_)
  indices = [x for x in range(len(clustering.labels_)) if clustering.labels_[x] in labels]
  render_paths(indices)
  # print('indices', indices)
  # print(label)
  # print(event)
  # print('picked!')
  # print(event)
  # print(event.ind)

  # print(event.inaxes)
  # if event.inaxes == ax3:
  #   print(event.ind)

hexcolors = None





def draw_phenotypes(clustering):
  n = clustering.n_clusters
  nn = n
  fig, axs = plt.subplots(int(np.ceil(nn/4)), int(np.ceil(nn/(nn/4))), subplot_kw={'projection': '3d'})
  axes = axs.flat

  paths_sample = random.sample(paths,min(len(paths), 500))

  for i in range(n):    # enumerate axes = cluster
    ax = axes[i]
    ax.view_init(elev=ax1.elev, azim=ax1.azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for line in paths_sample:
      ax.plot3D(line[:,0], line[:,1], line[:,2], color=[0.8,0.8,0.8,0.8])
    for p in range(len(paths)):   # enumerate paths
      if clustering.labels_[p] == i:
        path = paths[p]
        # path = splinify(path)
        ax.plot3D(path[:,0], path[:,1], path[:,2], color=color_of_line(paths[p]))

  # for i in range(2):
  #   ax = axes[i]
  #   ax.view_init(elev=ax1.elev, azim=ax1.azim)
  #   ax.set_xticks([])
  #   ax.set_yticks([])
  #   ax.set_zticks([])
  #   for p in range(len(paths)):   # enumerate paths
  #     # if clustering.labels_[p] == i:
  #     path = paths[p]
  #     if i%2 is 0:
  #       path = splinify(path)
  #     ax.plot3D(path[:,0], path[:,1], path[:,2], color=color_of_line(paths[p]))

  # def on_move(event):
  #   if event.inaxes == axes[n]:
  #     print('hi')
  #     for ax in axes:
  #       ax.view_init(elev=axes[n].elev, azim=axes[n].azim)

  # c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)




# fig.canvas.mpl_connect("motion_notify_event", hover)

# print(X4)
# V = pca.components_
# X3 = manifold.TSNE(n_components=2).fit_transform(X)

# ax3.scatter(X1[:,0], X1[:,1], picker=True, color=pathcolors)
ax3.scatter(X1[:,0], X1[:,1], picker=True, color=pathcolors, s=10)
ax7.scatter(X2[:,0], X2[:,1], picker=True, color=pathcolors, s=10)
ax11.scatter(X6[:,0], X6[:,1], picker=True, color=pathcolors, s=10)
# selector = lasso.SelectFromCollection(ax11, pts2, on_pick)
# ax6.scatter(X3[:,0], X3[:,1], picker=True, color=pathcolors)
# ax9.scatter(X4[:,0], X4[:,1], picker=True, color=pathcolors, s=10)
pts = ax10.scatter(X5[:,0], X5[:,1], color=pathcolors, s=10)
selector = lasso.SelectFromCollection(ax10, pts, on_pick)

clustering = AgglomerativeClustering(distance_threshold=None, n_clusters=16).fit(X)
# clustering = sklearn.cluster.KMeans(n_clusters=16).fit(X)
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
# fig.canvas.mpl_connect("motion_notify_event", on_hover)

cluster_names = []

for i in range(clustering.n_clusters):
  cluster_names += [i]


check_buttons = CheckButtons(ax9, cluster_names, [False]*clustering.n_clusters)
for i, rect in zip(range(len(check_buttons.rectangles)), check_buttons.rectangles):
  rect.fill = True
  # rect.color = hexcolors[i]
  rect.set_facecolor(hexcolors[i])
  rect.set_width(0.1)
check_buttons.on_clicked(on_check)
# plt.draw()
# plt.pause(0.001)
plt.draw()
# plt.show()

draw_phenotypes(clustering)

plt.show()