from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import gen as gen

class CellsDataset(Dataset):

  def __init__(self, mode, generate=False):
    self.samples = []
    self.labels  = []

    # plt.ion()

    try:
      if generate:
        raise Error
      with open('cellnet/data/cellnet_data.npy', 'rb') as f:
        self.samples = np.load(f)
        self.labels =  np.load(f)

      # self.samples, self.labels = np.load('cellnet/data/cellnet_data.npy')
    except:
      for i in range(0, 1024):
        print('construct ', i)
        image = np.zeros((32,32))
        r = np.random.rand()

        noise = False
        label = None

        # 0.1: random noise
        # 0.2: random noise with random blur
        # 0.6: [1-8] random off-center cells
        # 0.7: 

        # 0.1: random noise
        if   r<0.1:   
          label = 2
          image = np.random.rand(image.shape[0], image.shape[1])
          image /= np.sum(image)
          # noise = True
        
        # 0.2: random noise with random gaussian blur
        elif r<0.2:  
          label = 2 
          image = np.random.rand(image.shape[0], image.shape[1])
          image = gaussian_filter(image, sigma=np.random.rand()*3)
          image /= np.sum(image)
          # noise = True

        # 0.5: [1-8] random off-center cells
        elif r<0.6:
          label = 1
          for i in range(np.random.randint(1,8)):
            image += gen.cell_image_small(offcenter=True)

        # 0.7: single on-center cell
        elif r<0.7:
          label = 0
          image += gen.cell_image_small(offcenter=False)

        # 1.0: [1-8] off-center cells, 1 on-center cell, [0-2] more bright off-center cells.
        elif r<1.0:
          label = 0  
          for i in range(np.random.randint(1,8)):
            image += gen.cell_image_small(offcenter=True)
          nm = np.max(image)
          image += gen.cell_image_small(offcenter=False) * 2.0 * np.max(image)
          for i in range(np.random.randint(0,3)):
            image += gen.cell_image_small(offcenter=True, rmin=10) * (2.0*np.random.rand() * nm)

        # if the image has cells, postprocess it.
        if label is not 2:
          image /= np.sum(image)
          image += np.random.rand(image.shape[0], image.shape[1])*0.005
          image = gaussian_filter(image, sigma=2+np.random.rand()*3)
          image += np.random.rand(image.shape[0], image.shape[1])*0.0005
          image /= np.sum(image)

        self.samples += [image]
        self.labels += [label]
      self.samples = np.array(self.samples)
      self.labels = np.array(self.labels)
      with open('cellnet/data/cellnet_data.npy', 'wb') as f:
        np.save(f, self.samples)
        np.save(f, self.labels)

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