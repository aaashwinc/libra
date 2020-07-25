import torch
import torchvision
import torchvision.transforms as transforms
from cellnet.celldata import CellsDataset
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                     download=True, transform=transform)

trainset = CellsDataset()

# print(trainset)
# exit(0)
# trainset = 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                      shuffle=True, num_workers=2)

# testset = CellsDataset()

testset = trainset

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                     shuffle=False, num_workers=2)

classes = ('center', 'skew')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img, l):
  # img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()
  # print('shape', npimg.shape)
  fig, axes=plt.subplots(1,4)
  for i in range(4):
    axes[i].imshow(npimg[i])
    axes[i].set_title(l[i].numpy())
  plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(images,labels)

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# show images
imshow(torchvision.utils.make_grid(images), labels)

# exit(0)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 7 * 7, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 2)

  def forward(self, x):
    x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 7 * 7)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x




net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(20):  # loop over the dataset multiple times

  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]

    # print(data)
    inputs, labels = data

    labels = labels.float()
    inputs = inputs.float()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    # print(outputs, labels[:,0])
    loss = criterion(outputs, labels[:,0])
    # print(loss)
    # print(outputs)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 20 == 19:  # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

PATH = './data/cifar_net.pth'
torch.save(net.state_dict(), PATH)

# testitr = itr(testloader)
# print(list(net.parameters()))
# print(list(net.parameters())[0].data.numpy())

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
  100 * correct / total))