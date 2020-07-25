import numpy as np

from scipy.ndimage.filters import gaussian_filter

from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert

import numpy as np

def get_sphere_distribution(n, dmin, Ls, maxiter=1e6, allow_wall=True):
  """Get random points in a box with given dimensions and minimum separation.
  
  Parameters:
    
  - n: number of points
  - dmin: minimum distance
  - Ls: dimensions of box, shape (3,) array 
  - maxiter: maximum number of iterations.
  - allow_wall: whether to allow points on wall; 
     (if False: points need to keep distance dmin/2 from the walls.)
    
  Return:
    
  - ps: array (n, 3) of point positions, 
    with 0 <= ps[:, i] < Ls[i]
  - n_iter: number of iterations
  - dratio: average nearest-neighbor distance, divided by dmin.
  
  Note: with a fill density (sphere volume divided by box volume) above about
  0.53, it takes very long. (Random close-packed spheres have a fill density
  of 0.64).
  
  Author: Han-Kwang Nienhuys (2020)
  Copying: BSD, GPL, LGPL, CC-BY, CC-BY-SA
  See Stackoverflow: https://stackoverflow.com/a/62895898/6228891 
  """
  Ls = np.array(Ls).reshape(2)
  if not allow_wall:
    Ls -= dmin
  
  # filling factor; 0.64 is for random close-packed spheres
  # This is an estimate because close packing is complicated near the walls.
  # It doesn't work well for small L/dmin ratios.
  # sphere_vol = np.pi/6*dmin**3
  # box_vol = np.prod(Ls + 0.5*dmin)
  # fill_dens = n*sphere_vol/box_vol
  # if fill_dens > 0.64:
  #   msg = f'Too many to fit in the volume, density {fill_dens:.3g}>0.64'
  #   raise ValueError(msg)
  
  # initial try   
  ps = np.random.uniform(size=(n, 2)) * Ls
  
  # distance-squared matrix (diagonal is self-distance, don't count)
  dsq = ((ps - ps.reshape(n, 1, 2))**2).sum(axis=2)
  dsq[np.arange(n), np.arange(n)] = np.infty

  for iter_no in range(int(maxiter)):
    # find points that have too close neighbors
    close_counts = np.sum(dsq < dmin**2, axis=1)  # shape (n,)
    n_close = np.count_nonzero(close_counts)
    if n_close == 0:
      break
    
    # Move the one with the largest number of too-close neighbors
    imv = np.argmax(close_counts)
    
    # new positions
    newp = np.random.uniform(size=2)*Ls
    ps[imv]= newp
    
    # update distance matrix
    new_dsq_row = ((ps - newp.reshape(1, 2))**2).sum(axis=-1)
    dsq[imv, :] = dsq[:, imv] = new_dsq_row
    dsq[imv, imv] = np.inf
  else:
    raise RuntimeError(f'Failed after {iter_no+1} iterations.')

  if not allow_wall:
    ps += dmin/2
  
  dratio = (np.sqrt(dsq.min(axis=1))/dmin).mean()
  return ps, iter_no+1, dratio

def psdmatrix():
  while True:
    M = (np.random.rand(2,2)*2 - 0.5) 
    M = M.dot(M.T)
    M /= np.linalg.norm(M)
    M *= (0.05 + np.random.rand()*0.1)
    if np.linalg.cond(M) < 3:
      return M


def dataset2d_real(ncells):
  ncells = 30
  
  image = np.zeros((100,100))

  for i in range(ncells):
    img = np.zeros(image.shape)
    p = np.random.rand(2)*image.shape
    for i in range(4):
      pp = p + (np.random.rand(2)-0.5)*20
      pp = np.clip(pp, 0, 100-0.0001)
      print(pp)
      img[int(pp[0]), int(pp[1])] = 1.0
    img = convex_hull_image(img).astype('float')

    img = gaussian_filter(img, sigma=2)
    img[img>0.4]=1
    img[img<0.4]=0
    img = gaussian_filter(img, sigma=2)

    diff = 2
    image += img
  return image

def dataset2d(ncells):
  # return dataset2d_real(ncells)

  image = np.zeros((100,100))
  M    = []
  beta = []
  c    = []

  for i in range(ncells):
    M += [psdmatrix()]
    beta += [0.7 + np.random.rand() * 2]
    # c += [np.random.rand(2)*image.shape]

  c = get_sphere_distribution(ncells, 15, image.shape, maxiter=1e4, allow_wall=False)[0]

  print(c[3])

  # print(M, beta, c)

  for px in range(image.shape[0]):
    for py in range(image.shape[1]):
      p = np.array([px, py])

      for i in range(ncells):
        x = p-c[i]
        v = np.exp(-0.5 * np.power(x.dot(M[i]).dot(x), beta[i]))
        image[p[0], p[1]] += np.random.rand()*v

      # image[p[0],p[1]] = np.random.rand()

  image /= np.sum(image)
  # image += np.random.rand(image.shape[0], image.shape[1])*0.0004

  # # # axes[1].imshow(image)
  image = gaussian_filter(image, sigma=2)
  # axes[2].imshow(image)
  # image /= np.sum(image)
  image += np.random.rand(image.shape[0], image.shape[1])*0.0001
  # # # # axes[3].imshow(image)
  image /= np.sum(image)

  return image
  
def cell_image_small(offcenter, rmin=5.0, rmax=24.0):
  image = np.zeros((32,32))
  M = np.zeros((2,2))
  while True:
    M = (np.random.rand(2,2)*2 - 0.5) 
    M = M.dot(M.T)
    M /= np.linalg.norm(M)
    M *= (0.05 + np.random.rand()*0.1)
    if np.linalg.cond(M) < 3:
      break
  beta = 0.7 + np.random.rand() * 2

  center = np.zeros((2))
  if offcenter:
    center = np.random.rand((2))*2-1
    center /= np.linalg.norm(center)
    center *= rmin + np.random.rand()*(rmax-rmin)

  for p in np.ndindex(image.shape):
    x = np.array(p) - np.array(np.array(image.shape)/2.0)
    x = x-center
    image[p[0], p[1]] = np.exp(-0.5 * np.power(x.dot(M).dot(x), beta)) * np.random.rand()

  # image = gaussian_filter(image, sigma=1)
  return image
