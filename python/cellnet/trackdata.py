from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import gen as gen
import generator

class TrackDataset(Dataset):

  def __init__(self, mode, generate=False):
    self.samples = []
    self.labels  = []

    # plt.ion()

    try:
      if generate:
        raise Error
      self.samples = np.load('cellnet/data/track_samples.npy')
      self.labels = np.load('cellnet/data/track_labels.npy')
    except:
      for i in range(0, 512):
        print('construct ', i)
        image = np.zeros((2,32,32))
        label = np.zeros((32, 32))
        nc = np.random.randint(4)
        center = np.array([16,16]).astype('float')
        M = []
        beta = []
        if nc>0:
          others = gen.get_sphere_distribution(nc, 14, image.shape[1:], maxiter=1e4, allow_wall=True)[0]
          center = np.vstack([center, others])
        else:
          center = np.array([center])

        maxvel = [3, 1.8, 1.5, 1.5][nc]
        veloc  = (np.random.rand(len(center), 2)-0.5)
        veloc *= np.random.rand()*maxvel/np.linalg.norm(veloc)

        for i in range(len(center)):
          M += [generator.psdmatrix()]
          beta += [0.7 + np.random.rand() * 2]

        # print(veloc)

        # print(center)


        nc = len(center)

        # print(veloc)
        # print(center)
        for c in range(2):
          for x in range(32):
            for y in range(32):
              for i in range(len(center)):
                p = np.array([x,y]).astype('float') - center[i]
                # M = np.array([[0.1,0],[0,0.1]])
                v = np.exp(-0.5 * np.power(p.dot(M[i]).dot(p), beta[i])) * np.random.rand()
                # v = np.exp(-0.5 * np.power(p.dot(M).dot(p), 2.0))
                image[c,x,y] += v
          image[c] /= np.sum(image[c])
          image[c] = gaussian_filter(image[c], sigma=1.68)
          image[c] += np.random.rand(32, 32)*0.001
          image[c] /= np.sum(image[c])

          if c is 1:
            # print('c', center[0].astype('int'))
            label[int(center[0,0]*0.99999), int(center[0,1]*0.99999)] = 1.0

          for t in range(5):
            for i in range(nc):
              for j in range(nc):
                if i != j:
                  away = center[i] - center[j]
                  veloc[i] += 4 * away * 1/(np.linalg.norm(away)**2.0)
              if np.linalg.norm(veloc[i]) > maxvel:
                veloc[i] /= np.linalg.norm(veloc[i])
              center[i] += veloc[i]


        self.samples += [image]
        self.labels += [label]
      self.samples = np.array(self.samples)
      self.labels = np.array(self.labels)
      np.save('cellnet/data/track_samples.npy', self.samples)
      np.save('cellnet/data/track_labels.npy', self.labels)

    if mode == 'training':
      self.samples = self.samples[:len(self.samples)//2]
      self.labels = self.labels[:len(self.labels)//2]
    else:
      self.samples = self.samples[len(self.samples)//2:]
      self.labels = self.labels[len(self.labels)//2:]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return (self.samples[idx], self.labels[idx])