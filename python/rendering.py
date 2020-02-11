import numpy as np
import nrrd
import matplotlib.pyplot as plt
from PIL import Image

# filename = 'testdata.nrrd'
data = nrrd.read('../../../data/16-05-05/000.nrrd')
data = np.array(data)[0]
print(data.shape)

MIP = data[0,0:200,0:200,0]

print(MIP.shape)
# for i in range(0,200):
#   MIP[100,i] = 2635
  # MIP[100,i+1] = 2635
# for i in range (450):
#   slice = data[0,:,:,i]
#   MIP += slice
print(MIP.shape)
print(MIP)
print(np.max(MIP))

img = Image.fromarray(MIP)

fig, axes = plt.subplots(1, 1)
axes.imshow(img)

plt.show()