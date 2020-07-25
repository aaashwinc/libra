import numpy as np
from scipy.ndimage.filters import gaussian_filter


def psdmatrix():
  while True:
    M = (np.random.rand(2,2)*2 - 0.5) 
    M = M.dot(M.T)
    M /= np.linalg.norm(M)
    M *= (0.05 + np.random.rand()*0.1)
    if np.linalg.cond(M) < 3:
      return M

def synth(T=6, ncells=10):
  # return dataset2d_real(ncells)

  image = np.zeros((T,120,120))

  M      = []
  beta   = []
  center = []
  veloc  = []

  for i in range(ncells):
    M += [psdmatrix()]
    beta += [1.5 + np.random.rand() * 1.5]
    # c += [np.random.rand(2)*image.shape]

  center = np.multiply(np.random.rand(ncells, 2) , image.shape[1:])
  veloc  = (np.random.rand(ncells, 2)-0.5)

  # print(center)

  # gravitycoef = 1e-8

  for t in range(T):
    # evolve system
    for tt in range(5):
      for i in range(ncells):
        for j in range(ncells):
          if i != j:
            away = center[i] - center[j]
            if np.linalg.norm(away) < 20:
              veloc[i] += away * 1/(np.abs(np.linalg.norm(away)))
            # veloc[i] += gravity * 1e-6*(np.linalg.norm(gravity)**2.0)
        gravity = np.array(image.shape[1:])/2.0 - center[i]
        veloc[i] += gravity*0.004
        # veloc[i] = np.array([0,3])
        if np.linalg.norm(veloc[i]) > 1.5:
          veloc[i] /= np.linalg.norm(veloc[i])
        # veloc[i] += np.random.rand(2)*0.1
        center[i] += veloc[i]

    # cell divisions:
    for i in range(ncells):
      if np.random.rand() < 0.03:
        print('cell division!', t, i, center[i])
        center = np.append(center, (center[i]+np.random.rand(2)*3).reshape(1,2), axis=0)
        veloc  = np.append(veloc,  veloc[i].reshape(1,2), axis=0)
        M += [psdmatrix()]
        beta += [0.7 + np.random.rand() * 2]
        # print(center)
        ncells = ncells+1

    # print(center)

    # draw system
    for x in range(image.shape[1]):
      for y in range(image.shape[2]):
        for i in range(ncells):
          p = np.array([x,y]).astype('float') - center[i]
          v = np.exp(-0.5 * np.power(p.dot(M[i]).dot(p), beta[i])) * np.random.rand()
          if(v<1e-10):
            v=0
          image[t,x,y] += v
    image[t] = gaussian_filter(image[t], sigma=1)
    image[t] /= np.sum(image)
    image[t] += np.random.rand(image.shape[1], image.shape[2])*0.0001
    image[t] /= np.sum(image)
  return image
  