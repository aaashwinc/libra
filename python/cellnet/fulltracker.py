import tracknet as tracknet
import cellnet as cellnet
import generator as gen
import display as disp
import numpy as np

from skimage.draw import line_aa
from skimage.draw import line

import matplotlib.pyplot as plt


tracker = tracknet.TrackNet()
finder  = cellnet.CellNet(display=False)

tracker.load()
finder.load()

if tracker.net is None or finder.net is None:
  print("TrackNet or CellNet not found.")
  exit(0)

print('synth')

data = None

try:
  raise Error
  data = np.load('data/gen.npy')
except:
  data = gen.synth(8)
  np.save('data/gen.npy', data)

class Cell():
  def __init__(self, x, t):
    self.x = np.copy(x)
    self.t = np.copy(t)
    self.pred = None
    self.succ = []

    self.prevp = None

  def id(self):
    return id(self) % 1000000
  def __str__(self):
    return "(%s %s | %s > %s > %s)" % (str(self.t), str(self.x), str((self.pred.id() if self.pred else -1)), str((self.id())), " ".join([str((x.id())) for x in self.succ]))
    # return '%s: %s %s <- %s\n    %s\n    %s\n' % (str(id(self)), str(self.t), str(self.x),str(self.prevp), str(self.pred), str(self.succ))
    # return '[' + str(id(self)) + ', ' +str(self.x)+', '+str(self.prevp)+', '+str(self.t)+', '+str(self.pred)+', '+str(self.succ)+']'
    # self.None = None
    # self.None = None
# [46.0952381  72.57142857] | 83448 > 8148
  # def __id__(self):
  #   return super().__id__() % 1000

cells = [None] * len(data)    # list of cells at frame t
blobs = [None] * len(data)

finder.display = False

disp.imshow1(data)

# finder.display = True
# tracker.display = True


for t in np.arange(len(data)-1, 0, -1):
  print('t=', t)
  if cells[t] is None:
    cells[t] = []
    blobs[t] = finder.find_blobs(data[t])
    for i in range(len(blobs[t])):
      cells[t] += [Cell(x=blobs[t][i],t=t)]
  
  if cells[t-1] is None and t-1 >= 0:
    cells[t-1] = []
    blobs[t-1] = finder.find_blobs(data[t-1])
    for i in range(len(blobs[t-1])):
      cells[t-1] += [Cell(x=blobs[t-1][i],t=t-1)]
    # for x in cells[t-1]:
    #   print (x)
  print('timestep ', t)
  print('b[t]:  ', blobs[t])
  print('b[t-1]:', blobs[t-1])
  for c in cells[t]:
    if c.x[0]>=16 and c.x[1]>=16 and c.x[0]+16<data.shape[1] and c.x[1]+16 < data.shape[2]:
      regiont0 = np.copy(data[t,   int(c.x[0]-16):int(c.x[0]+16), int(c.x[1]-16):int(c.x[1]+16)])
      regiont0 /= np.sum(regiont0) 
      regiont1 = np.copy(data[t-1, int(c.x[0]-16):int(c.x[0]+16), int(c.x[1]-16):int(c.x[1]+16)])
      regiont1 /= np.sum(regiont1)


      # regiont0 at timestep t
      # regiont1 at timestep t-1
      # delta is relative motion of cell from timestep t to timestep t-1
      # c.prevp is predicted image coordinates of cell at timestep t-1
      region = np.stack([regiont0, regiont1])
      # tracker.display = True
      delta = tracker.track(region)
      c.prevp = c.x + delta

      distsq = np.sum((blobs[t-1] - c.prevp)**2.0, axis=1)

      closesti = np.argmin(distsq)
      if distsq[closesti] < 100:
        c.pred = cells[t-1][closesti]
        cells[t-1][closesti].succ += [c]


      print('tracked ', c.x, '\t| predicted origin= ', c.prevp, '\t| cell location=', c.pred.x if c.pred is not None else None)
      print(' ', distsq)
      # print('  distance = %.2f'%(np.linalg.norm(c.prevp - c.pred.x)))
      # print('  d <<  %.2f'%(c.prevp-blobs[t-1]))
      # visualize:
      #   blobs at t
      #   blobs at t-1
      #   for each blob in t:
      #     region around blob[t]  [i]
      #     region around blob[t-1][i]
      #     tracking results

      # point = np.zeros(blobs[t].shape)
      # point[np.clip(int(delta[0])+16, 0,31), np.clip(int(delta[1])+16,0,31)] = 1

      # disp.imshow_images([regiont0, regiont1, point])
      # cells[t-1][closesti].succ += [c]
      # print('\n\n        studying cell')
      # print(c)
      # print('appending to ', cells[t-1][closesti])
      
      # print(c)
      # print('closest cell at index', closesti)
      # print('closest cell is ', id(cells[t-1][closesti]))
      # print(cells[t-1][closesti])
      # print(distsq)
      # print(c)

# display results:
display = np.zeros((data.shape[1], data.shape[2], 3))


allcells = [cell for celllist in cells for cell in celllist]
# allcells = [cell for cell in celllist for celllist in cells]

print('all cells ~~~~')
for i in allcells:
  print(i)

drawn = {}

def drawlineages(cell, color, display):
  # print(display.shape)
  # print(display[int(cell.x[0]), int(cell.x[1])])
  # print(color)

  # if drawn[cell]:
  #   return

  drawn[cell] = True

  display[int(cell.x[0]), int(cell.x[1])] = color
  if cell.succ is not None:
    for c in cell.succ:
      # display[line(int(cell.x[0]), int(cell.x[1]), int(c.x[0]), int(c.x[1]))] = color
      drawlineages(c, color, display)



for t in range(len(cells)):
  for cell in cells[t]:
    if (cell not in drawn) or (not drawn[cell]):
      drawn[cell] = True
      drawlineages(cell, np.random.rand(3), display)
  

# plt.ion()
for t in range(len(data)):
  for i in range(len(blobs[t])):
    data[t, int(blobs[t][i][0]), int(blobs[t][i][1])] = np.max(data[t])

disp.imshow1(data)
disp.imshow_images([data[0], data[-1], display])
# plt.ioff()