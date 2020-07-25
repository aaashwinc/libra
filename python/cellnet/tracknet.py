# import cifar
import celldata
import gen

import torch
import torchvision
import torchvision.transforms as transforms
# from celldata import TrackDataset
from trackdata import TrackDataset

import sys

from optparse import OptionParser

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import display

# input:
# channel 1: image at t0 centered at c0
# channel 2: image at t1 centered at c0
# output:    distribution of tracking locations (c1)
class TrackNN(nn.Module):
  def __init__(self):
    super(TrackNN, self).__init__()
    self.pool = nn.MaxPool2d(2, 2)

    self.conv1 = nn.Conv2d(2, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.fc1 = nn.Linear(16 * 5 * 5, 32 * 32)
    # self.fc2 = nn.Linear(16 * 5 * 5, 32 * 32)

  def forward(self, x):
    prelu = nn.PReLU()
    x = self.pool(prelu(self.conv1(x)))
    x = self.pool(prelu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = prelu(self.fc1(x))
    # x = prelu(self.fc2(x))
    # x = prelu(self.fc2(x))
    # x = self.fc3(x)
    x = x.view(-1, 32, 32)

    return x

class TrackNet():
  def __init__(self, display=False, batch_size=4):
    self.net = TrackNN()
    self.batch_size = batch_size
    self.display = display
    pass

  def load(self):
    self.net = TrackNN()
    self.net.load_state_dict(torch.load('data/tracknet.nn'))

  def train(self):

    if torch.cuda.is_available():
      device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
      print("Running on the GPU")
    else:
      device = torch.device("cpu")
      print("Running on the CPU")

    trainset = TrackDataset('training')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    self.net = TrackNN()

    criterion = nn.MSELoss()

    optimizer = optim.Adagrad(self.net.parameters(), lr=0.01)

    i = 0
    for epoch in range(400):  # loop over the dataset multiple times
      running_loss = 0.0
      nnn = 0
      for ii, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.float()
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        nnn = nnn + 1
        i=i+1

      print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / nnn))

      # if(running_loss / nnn < 0.01):
      #   break
      running_loss = 0.01,2,0
      nnn = 0
        # i = i+1

    print('Finished Training')

    torch.save(self.net.state_dict(), 'data/tracknet.nn')

  def test(self):

    testset = TrackDataset('test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    correct = 0
    total   = 0
    with torch.no_grad():
      for data in testloader:
        image, label = data
        image = image.float()
        label = label.float()
        # display.imshow_2_1(image, label)
        # print(image.shape)
        out   = self.net(image)
        show = False

        # print(np.min(image.numpy()), np.sum(image.numpy()))

        for i in range(len(image)):
          indmax1 = np.unravel_index(np.argmax(label[i].numpy(), axis=None), label[i].numpy().shape)
          indmax2 = np.unravel_index(np.argmax(out[i].numpy(), axis=None), out[i].numpy().shape)
          if(np.linalg.norm(np.array(indmax1) - np.array(indmax2)) < 5):
            correct += 1
          elif self.display:
            show = True
          total += 1
        if show:
          display.imshowfull(image, label, np.exp(out))
      print('Accuracy = %d / %d = %.4f' % (correct, total, correct/total))
  
  def gendata(self):
    _ = TrackDataset('test', generate=True)
    pass

  def displaydata(self):
    trainset = TrackDataset('training')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    display.imshow_2_1(images, labels)

  def track(self, region):
    ten = torch.from_numpy(region.reshape(1,2,32,32)).float()
    out    = self.net(ten)
    out = out.detach().numpy()
    indmax = np.unravel_index(np.argmax(out[0], axis=None), out[0].shape)
    indmax = np.array(indmax)
    v =  indmax - np.array(region[0].shape)/2.0
    if self.display:
      display.imshow_2_1(region.reshape(1,2,32,32), out.reshape(1,32,32))
      # print(v)
    return v

if __name__ == "__main__":
  usage = "usage: %prog [options] arg"
  parser = OptionParser(usage)

  parser.add_option("-g", "--generate-data",
                    action="store_true", dest="generate", default=False)
  parser.add_option("-t", "--train-model",
                    action="store_true", dest="train", default=False)
  parser.add_option("-a", "--apply-model",
                    action="store_true", dest="apply", default=False)
  parser.add_option("-d", "--display-data",
                    action="store_true", dest="display", default=False)
  parser.add_option("-b", "--batch-size",
                     dest="bs", default='4')
  (options, args) = parser.parse_args()

  net = TrackNet(display = options.display, batch_size = int(options.bs))

  if options.generate:
    net.gendata()

    if options.display:
      net.displaydata()

  if options.train:
    net.train()

  if options.apply:
    net.load()
    net.test()
    # if options.display:
    #   data = generator.synth(T=1)
    #   net.find_blobs(data[0])


# usage = "usage: %prog [options] arg"
# parser = OptionParser(usage)

# parser.add_option("-g", "--generate-data",
#                   action="store_true", dest="generate", default=False)
# parser.add_option("-t", "--train-model",
#                   action="store_true", dest="train", default=False)
# parser.add_option("-a", "--apply-model",
#                   action="store_true", dest="apply", default=False)
# parser.add_option("-d", "--display-data",
#                   action="store_true", dest="display", default=False)
# parser.add_option("-b", "--batch-size",
#                    dest="bs", default='4')
# (options, args) = parser.parse_args()
# print(options)

# init TrackNet:


# gather data
# batch_size = int(options.bs)

# trainset = TrackDataset('training', options.generate)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = TrackDataset('test')
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# display data

# def imshow(image,label):
#   fig, axes=plt.subplots(4,len(image))
#   axes = np.atleast_1d(axes)
#   for i in range(len(image)):
#     axes[0,i].imshow(image[i][0])
#     axes[1,i].imshow(image[i][1])
#     axes[2,i].imshow(label[i]   )
#     # print(label[i][np.newaxis,...].shape)
#     axes[3,i].imshow(np.transpose(np.vstack([image[i],label[i][np.newaxis,...]]), (1,2,0)))
#     # if labels is not None:
#     #   axes[i].set_title(labels[i])
#   plt.show()


# if options.display:
#   dataiter = iter(trainloader)
#   images, labels = dataiter.next()
#   imshow(images.numpy(), labels.numpy())



      