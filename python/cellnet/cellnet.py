
import torch
import torchvision
import torchvision.transforms as transforms
from celldata import CellsDataset

import gen as gen
import sys

from optparse import OptionParser
import torch.nn as nn
import torch.nn.functional as F

import display
import torch.optim as optim
import scipy as scipy
import generator
import numpy as np
import matplotlib.pyplot as plt
import time

class CellNN(nn.Module):
  def __init__(self):
    super(CellNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 40)
    self.fc3 = nn.Linear(40, 3)
    # self.lin = nn.Linear(8*5*5, 3)

  def forward(self, x):
    prelu = nn.PReLU()
    x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
    # x = self.pool(F.relu(self.conv1(x)))
    # x = self.pool(F.relu(self.conv2(x)))
    # x = x.view(-1, 16 * 5 * 5)
    # x = F.relu(self.fc1(x))
    # x = F.relu(self.fc2(x))
    # x = self.fc3(x)
    # print(x.shape)
    # print(x)
    # x = x.view(x.shape[0], -1)
    x = self.pool(prelu(self.conv1((x))))
    x = self.pool(prelu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = prelu(self.fc1(x))
    x = prelu(self.fc2(x))
    x = self.fc3(x)
    # print(x)
    # x = F.sigmoid(x)
    # print(x.shape)
    return x


#############################################
############### CellNet #####################
#############################################


# class SaveFeatures():
#   def __init__(self, module):
#     self.hook = module.register_forward_hook(self.hook_fn)
#   def hook_fn(self, module, input, output):
#     self.features = torch.tensor(output,requires_grad=True)
#   def close(self):
#     self.hook.remove()

class CellNet():
  def __init__(self, display=False, batch_size=4):
    self.net = CellNN()
    self.batch_size = batch_size
    self.display = display
    pass

  def load(self):
    self.net = CellNN()
    self.net.load_state_dict(torch.load('data/cellnet.nn'))

  def train(self):
    trainset = CellsDataset('training')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    # self.net = CellNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(self.net.parameters(), lr=0.01)

    i=0
    for epoch in range(500):
      epoch_loss = 0.0
      epochn     = 0.0
      total = 0
      correct = 0
      for ii, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = inputs.float()
        labels = labels.long()

        optimizer.zero_grad()

        outputs = self.net(inputs)
        # print(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics

        epoch_loss += loss.item()
        epochn = epochn + 1

      print('[%d, %5d] loss: %.6f' % (epoch, i, epoch_loss / epochn))
      print('accuracy = %d/%d = %.4f' % (correct, total, correct/total))
      if(epoch_loss / epochn < 0.01 or (total-correct < min(10, 1+0.02*total))):
        break
      epoch_loss = 0.0
      epochn = 0
    torch.save(self.net.state_dict(), 'data/cellnet.nn')

  def test(self):
    classes = ['center', 'skew', 'noise']
    testset = CellsDataset('test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    images = images.float()
    labels = labels.long()

    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    correct = 0
    total = 0

    bad_images = []
    bad_labels = []

    with torch.no_grad():
      for data in testloader:
        images, labels = data
        images = images.float()
        labels = labels.long()
        outputs = self.net(images)

        _, predicted = torch.max(outputs.data, 1)

        c = (predicted == labels).squeeze()
        for i in range(len(images)):
          label = labels[i]
          class_correct[label] += c[i].item()
          class_total[label] += 1

          if c[i].item() == 0:
            bad_labels += [predicted[i]]
            bad_images += [images[i]]

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('test accuracy: %d/%d = %.4f' % (correct, total, correct/total))
    for i in range(3):
      print('class accuracy of %6s: %.3f' % (classes[i], (class_correct[i]) / (class_total[i]+1e-100)))

    if self.display:
      display.imshow(torchvision.utils.make_grid(bad_images), [classes[label] for label in bad_labels])

  def gendata(self):
    _ = CellsDataset('test', generate=True)
    pass

  def displaydata(self):
    classes = ['center', 'skew', 'noise']
    trainset = CellsDataset('training')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(self.batch_size)))
    display.imshow(torchvision.utils.make_grid(images), [classes[labels[j]] for j in range(self.batch_size)])


  # def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
  #   self.size, self.upscaling_steps, self.upscaling_factor = 32, 400, 2

  #   sz = self.size
  #   img = np.random.rand(1,32,32)
  #   display.imshow1([img[0]])
  #   activations = SaveFeatures(list(self.net.children())[layer])  # register hook

  #   for _ in range(100):  # scale the image up upscaling_steps times
  #     # train_tfms, val_tfms = tfms_from_model(vgg16, sz)
  #     img_var = torch.as_tensor(img).requires_grad_(True)  # convert image to Variable that requires grad
  #     optimizer = torch.optim.Adam([img_var], lr=1.0)
  #     for n in range(opt_steps):  # optimize pixel values for opt_steps times
  #       optimizer.zero_grad()
  #       self.net(img_var.float())
  #       loss = -activations.features[0, 0].mean()
  #       print(loss)
  #       # loss.backward()
  #       # optimizer.step()
  #       # img_var /= 2
  #       print(img_var)
  #     # img = val_var.detach().numpy()
  #     self.output = img_var.detach().numpy()
  #     sz = int(self.upscaling_factor * sz)  # calculate new image size
  #     # img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
  #     # if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
  #   # self.save(layer, filter)
  #   activations.close()
  #   display.imshow1([img[0]])



  def find_blobs(self, data):



    # display.imshow1([data[0]])

    proc, comp, netd, blob = None, None, None, None

    proc = np.zeros(data.shape)

    if self.display:
      comp = np.zeros(data.shape)
      netd = np.zeros(data.shape)
      blob = np.zeros(data.shape)

      fig, ax = plt.subplots(1, 1)
      pyimg = ax.imshow(np.random.rand(32,32))
      plt.ion()

    m = nn.Softmax
    with torch.no_grad():
      for x in range(0,data.shape[0]):
        for y in range(0,data.shape[1]):
          if data.shape[0]-x > 33 and data.shape[1]-y > 33:
            region = data[x:x+32, y:y+32].copy()
            region = region/np.sum(region)
            region = torch.from_numpy(np.array([region])).float()
            outputs = self.net(region)
            if outputs[0,0] > outputs[0,1] and outputs[0,0] > outputs[0,2]:
              proc[x+16, y+16] = 1
              if self.display:
                pyimg.set_data(region[0])
                pyimg.set_clim(np.min(region[0].numpy()), np.max(region[0].numpy()))
                ax.set_title(str([x,y]))
                plt.draw()
                plt.pause(0.0001)
              # print(F.softmax(outputs))
            if self.display:
              comp[x+16,y+16] = data[x+16,y+16] * proc[x+16,y+16]
              # print(F.softmax(outputs.data)[0])
              # print(region)
              netd[x+16,y+16] = (F.softmax(outputs.data))[0,0]

    label, nf = scipy.ndimage.label(proc)
    plt.ioff()  

    # print(label, nf)
    blobs = np.array(scipy.ndimage.center_of_mass(proc, label, np.arange(1, nf+1)))
    blobs = blobs.reshape(-1,2)
    # print(blobs)
    if self.display:
      for b in blobs:
        # print(b)
        blob[int(b[0]), int(b[1])] = 1
      display.imshow1([data, proc, comp, netd, blob])

    return blobs

# parser = OptionParser()

# parser.add_option()

# actions: compute data; train model; test model on full image.




if __name__ == "__main__":
  usage = "usage: %prog [options] arg"
  parser = OptionParser(usage)

  parser.add_option("-g", "--generate-data", action="store_true", dest="generate", default=False)
  parser.add_option("-t", "--train-model",   action="store_true", dest="train",    default=False)
  parser.add_option("-r", "--resume",        action="store_true", dest="resume",   default=False)
  parser.add_option("-a", "--apply-model",   action="store_true", dest="apply",    default=False)
  parser.add_option("-e", "--exec-model",    action="store_true", dest="exec",     default=False)
  parser.add_option("-d", "--display-data",  action="store_true", dest="display",  default=False)
  parser.add_option("-v", "--vis",           action="store_true", dest="vis",      default=False)

  parser.add_option("-b", "--batch-size", dest="bs", default='4')
  (options, args) = parser.parse_args()

  net = CellNet(display = options.display, batch_size = int(options.bs))

  if options.vis:
    net.visualize(0, 0)

  if options.generate:
    net.gendata()

  # if options.display:
  #   net.displaydata()

  if options.train:
    if options.resume:
      net.load()
    net.train()

  if options.apply:
    net.load()
    net.test()

  if options.exec:
    data = generator.synth(T=1)
    net.find_blobs(data[0])



# if options.train:

#   net = CellNN()




#   i = 0
#   for epoch in range(1200):  # loop over the dataset multiple times

#       running_loss = 0.0
#       nnn = 0
#       for ii, data in enumerate(trainloader, 0):
#           # get the inputs; data is a list of [inputs, labels]
#           inputs, labels = data

#           inputs = inputs.float()
#           labels = labels.long()

#           # zero the parameter gradients
#           optimizer.zero_grad()

#           # forward + backward + optimize
#           outputs = net(inputs)
#           # outputs[0,0] = 1
#           # outputs[1,0] = 1
#           # outputs[2,0] = 1
#           # outputs[3,0] = 1
#           # outputs[0,1] = 0
#           # outputs[1,1] = 0
#           # outputs[2,1] = 0
#           # outputs[3,1] = 0
#           loss = criterion(outputs, labels)
#           # print('```')
#           # print(outputs, labels)
#           # print(loss.item())
#           loss.backward()
#           optimizer.step()
#           # print('c', outputs[0:2], net(1+inputs*2)[0:2])

#           # print statistics
#           running_loss += loss.item()
#           # print(loss.item())
#           # if i % 20 == 19:    # print every 2000 mini-batches
#           nnn = nnn + 1
#       print('[%d, %5d] loss: %.6f' %
#             (epoch + 1, i + 1, running_loss / nnn))

#       if(running_loss / nnn < 0.01):
#         break
#       running_loss = 0.0
#       nnn = 0
#           # i = i+1

#   print('Finished Training')

#   PATH = './cellnet.nn'
#   torch.save(net.state_dict(), PATH)

#   dataiter = iter(testloader)
#   images, labels = dataiter.next()

#   images = images.float()
#   labels = labels.long()

#   # print images
#   # imshow(torchvision.utils.make_grid(images))
#   print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
#   print('Predicted: ', net(images))

#   net = CellNN()
#   net.load_state_dict(torch.load('./cellnet.nn'))


#   outputs = net(images)
#   _, predicted = torch.max(outputs, 1)  

#   correct = 0
#   total = 0
#   with torch.no_grad():
#       for data in testloader:
#           # print(data)
#           images, labels = data
#           images = images.float()
#           labels = labels.long()
#           outputs = net(images)
#           # print('888888', images, labels, outputs)
#           _, predicted = torch.max(outputs.data, 1)
#           total += labels.size(0)
#           # print('~~~~~', predicted, labels)
#           correct += (predicted == labels).sum().item()

#   print('Accuracy of the network on the 10000 test images: %d %%' % (
#       100 * correct / total))

#   class_correct = list(0. for i in range(10))
#   class_total = list(0. for i in range(10))
#   with torch.no_grad():
#       for data in testloader:
#           images, labels = data
#           images = images.float()
#           labels = labels.long()
#           outputs = net(images)
#           _, predicted = torch.max(outputs, 1)
#           c = (predicted == labels).squeeze()
#           for i in range(len(images)):
#               label = labels[i]
#               class_correct[label] += c[i].item()
#               class_total[label] += 1


#   for i in range(2):
#       print('Accuracy of %5s : %2d %%' % (
#           classes[i], 100 * class_correct[i] / class_total[i]))


# # machine learning model:

# if options.apply:
#   net = CellNN()
#   net.load_state_dict(torch.load('./cellnet.nn'))

#   print('dataset')
#   data = gen.dataset2d(20)
#   proc = np.zeros(data.shape)
#   comp = np.zeros(data.shape)
#   netd = np.zeros(data.shape)

#   m = nn.Softmax()

#   for x in range(0,data.shape[0]):
#     for y in range(0,data.shape[1]):
#       if data.shape[0]-x > 33 and data.shape[1]-y > 33:
#         region = data[x:x+32, y:y+32].copy()
#         region /= np.sum(region)
#         region = torch.from_numpy(np.array([region])).float()
#         outputs = net(region)
#         if outputs[0,1] > outputs[0,0]:
#           proc[x+16, y+16] = 1
#         comp[x+16,y+16] = data[x+16,y+16] * proc[x+16,y+16]
#         # print(outputs)
#         # print(m(outputs.data)[0])
#         netd[x+16,y+16] = m(outputs.data)[0,1]
#         # print(outputs)



#   imshow1([data, proc, comp, netd])
#   exit(0)