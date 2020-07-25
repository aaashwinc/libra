import matplotlib.pyplot as plt
import numpy as np


def format_axes(ax, im):
  ax.set_xticks(np.arange(0, im.shape[0], 8))
  ax.set_yticks(np.arange(0, im.shape[1], 8))
  ax.set_xticks(np.arange(0, im.shape[0], 4), minor=True)
  ax.set_yticks(np.arange(0, im.shape[1], 4), minor=True)
  ax.grid(which='major', alpha=0.6, c='gray', ls='-')
  ax.grid(which='minor', alpha=0.3, c='gray', ls='-')
  ax.xaxis.set_ticklabels([])
  ax.yaxis.set_ticklabels([])
  ax.xaxis.set_ticks_position('none')
  ax.yaxis.set_ticks_position('none')


def imshow(img,labels):
  # img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()
  # printimshow1('shape', npimg.shape)
  fig, axes=plt.subplots(1,len(img))
  axes = np.atleast_1d(axes)
  for i in range(len(img)):
    axes[i].imshow(npimg[i])
    if labels is not None:
      format_axes(axes[i], npimg[i])
      axes[i].set_title(labels[i])
  plt.show()

def imshow1(imgs):
  # img = img / 2 + 0.5   # unnormalize
  # npimg = img.numpy()
  # print('shape', npimg.shape)
  fig, axes=plt.subplots(1,len(imgs))
  axes = np.atleast_1d(axes)
  for i in range(len(imgs)):
    axes[i].imshow(imgs[i])
    format_axes(axes[i], imgs[i])
  plt.show()

def imshowfull(image, label, output):
  fig, axes=plt.subplots(4,len(image))
  axes = np.atleast_1d(axes)
  # image += np.random.rand(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
  for i in range(len(image)):
    im = np.transpose(np.vstack([image[i]/np.max(image.numpy()[i]), image[np.newaxis,i,0]]), (1,2,0))
    # print(np.min(im), np.max(im))
    axes[0,i].imshow(im)
    # print('display ---')
    # print(np.transpose(np.vstack([image[i], np.zeros((1,32,32))]), (1,2,0)).shape)
    # print(np.transpose(np.vstack([image[i], np.zeros((1,32,32))]), (1,2,0)))
    # exit(0)
    axes[1,i].imshow(label[i])
    axes[2,i].imshow(output[i])
    onlymax = output[i]
    onlymax[onlymax!=np.max(onlymax.numpy())] = 0
    axes[3,i].imshow(np.abs(onlymax+label[i]))
    for j in range(4):
      format_axes(axes[j,i], image[0][0])  
  plt.show()
  # exit(0)


def imshow1d(image,label):
  fig, axes=plt.subplots(2,len(image))
  axes = np.atleast_1d(axes)
  for i in range(len(image)):
    axes[0,i].imshow(image[i])
    axes[1,i].imshow(label[i])
  plt.show()

def imshow_2_1(image,label):
  # print(image.shape)
  fig, axes=plt.subplots(4,len(image))
  axes = np.atleast_2d(axes)
  # print((image).shape, (label).shape, (axes).shape)
  for i in range(len(image)):
    # print('-0', i, image[i][0].shape)
    # print('-1', i, image[i][1].shape)
    # print(axes.shape,image.shape)
    axes[i,0].imshow(image[i][0])
    axes[i,1].imshow(image[i][1])
    axes[i,2].imshow(label[i]   )
    # print(label[i][np.newaxis,...].shape)
    axes[i,3].imshow(np.transpose(np.vstack([image[i],label[i][np.newaxis,...]]), (1,2,0)))

    for a in axes[:,i]:
      format_axes(a, image[i][0])
    # if labels is not None:
    #   axes[i].set_title(labels[i])
  plt.show()

def imshow_images(images):
  fig, axes=plt.subplots(1,len(images))
  for i in range(len(axes)):
    axes[i].imshow(images[i])
  plt.show()