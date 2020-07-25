from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter


image = np.random.normal(0,1,size=[40,40])

# gaus2d.pdf([1,0])

# def gaussian(x, mu, sig):
  # return gaus2d.pdf(x)
    # return np.exp(-np.power(np.linalg.norm(x - mu), 2.) / (2 * np.power(sig, 2.)))


def gaus(mean=[20,20], cov=[[20,5],[5,20]]):
  image = np.zeros([40,40])
  gaus2d = multivariate_normal(mean, cov)
  for index in np.ndindex(40, 40):
    image[index] = gaus2d.pdf(np.array(index))
    # print(image[index])
  return image


def square(s):
  img = np.zeros((40,40))
  img[20:30, 20:30] = 1
  img = gaussian_filter(img, sigma=s)
  return img



def add_noise(img, m):
  # for index in np.ndindex(40, 40):
  #   img[index] += index[0]*0.0002
  # print(m)
  img += np.random.normal(0, 1, img.shape) * m

  img += -np.min(img)
  img /= np.sum(img)

  return img


# square(2)
# add_noise(image, 0.0004)

  # both methods are similar
  # error_quick =  0.00014529950815378098
  # error_slow =  9.450906999235192e-05

# gaus()
# add_noise(image, 0.0004)

  # also similar
  # error_quick =  0.00013392393490772552
  # error_slow =  6.864952898551591e-05

# image = square(1)
# image = add_noise(image, m=0.05)

  # errors are similar
  # error_quick =  0.0004573592863102129
  # error_slow =  0.0002918144249957604
  # but the slow method's shape fits better, because
  # the fast method overfits the noise

# gaus()

  # when the shape is a gaussian, the fit is perfect
  # error_quick =  1.5989253858631154e-08
  # error_slow =  1.2032931292841745e-08

image = gaus(mean=[35,35], cov=[[100,0],[0,100]])
# image = add_noise(image, 0.0001)


# print('image', image)



# fig, axs = plt.subplots(4, 1, squeeze=True)
# plt.ion()
# plt.draw()
# plt.pause(0.001)
# plt.show()




def imageof(p):
  gaus = multivariate_normal(mean=[p[0], p[1]], cov=[[p[2], p[3]],[p[3],p[4]]])
  test = np.zeros(shape=[40,40])
  for index in np.ndindex(40, 40):
    test[index] = gaus.pdf(np.array(index))
  return test * p[5]

def render(p, ax):
  ax.imshow(p)
  plt.pause(0.001)

def lsq(p, image):
  error = np.sum(np.square((image-imageof(p))))
  return error

def percent_error(source, image):
  error = np.sum(np.abs(source - image)) / np.sum(source)
  return error

def average_squared_error(image1, image2):
  error = np.sqrt(np.sum(np.square(image1 - image2)))
  return error

def lsq_deriv(p, image):
  f0 = np.sum(np.square((image-imageof(p))))
  deriv = [0]*len(p)
  delta = 0.001
  for i in range(len(p)):
    change = [0]*len(p)
    change[i] = 1*delta
    print('change', change)
    f1 = np.sum(np.square((image-imageof(p+change))))
    deriv[i] = (f1 - f0) / delta
  print('d', deriv)
  return deriv


def lsq_only_mu_cov10(p):
  error = np.sum(np.square((image-imageof([p[0], p[1], 10, 0, 10, 1]))))
  return error


def lsq_mag(mag, image, p):
  # print(image)
  test = imageof([p[0], p[1], p[2], p[3], p[4], mag])
  error = np.sum(np.square((image-test)))
  # print('mag', mag, p, error)
  # print('test', np.max(image), np.max(test))
  return error


def weighted_cov(obs, wts, mu):
  cov = np.array([[0.0,0],[0,0]])
  n = np.sum(wts)
  for i in range(len(obs)):
    xx = obs[i][0] - mu[0]
    yy = obs[i][1] - mu[1]
    # print('w', wts[i])
    # print(obs[i], mu, xx, yy, n)
    cov[0,0] += xx*xx * wts[i]
    cov[1,0] += yy*xx * wts[i]
    cov[0,1] += xx*yy * wts[i]
    cov[1,1] += yy*yy * wts[i]
    # print('c00', cov[0,0])
  # exit(0)
  return cov / n

def quick_estim(image):
  sum = np.sum(image)
  mu = np.array([0,0]).astype(float)
  cov = np.array([1,0,1])


  indices = np.arange(0,40)
  obs = np.transpose([np.tile(indices, len(indices)), np.repeat(indices, len(indices))])
  wts = []
  for i in obs:
    # print(i)
    mu += i * image[i[0],i[1]]/sum
    wts += [image[i[0], i[1]]]
  wts = np.array(wts)
  # print(obs)
  # print(wts)

  # print(len(obs))
  # print(obs)
  # mu = np.sum(obs * wts[:,np.newaxis], axis=0) / sum
  cov = weighted_cov(obs, wts, mu)
  # cov = np.cov(obs.T, aweights=(wts))
  # print(mu)
  # print(cov)

  # print(cov.shape)
  # print(mu)
  # # obs = np.array([[0],[0]])
  # # print(obs)
  # # wts = np.array([])

  # # print(np.ndindex(40,40))

  # for index in np.ndindex(40, 40):
  #   ix = np.array(index)
  #   # print('ix', ix)
  #   # print(image[index]/sum)
  #   mu += image[index]/sum * np.array(ix).astype(float) 
  #   # print( np.array([ix]))
  #   obs = np.append(obs, np.array([ix]).T)
  #   # print(obs)
  #   wts = np.append(wts, image[index]/sum)
  # # print(obs)
  # # print(wts[0:20])
  xq = [mu[0], mu[1], cov[0,0], cov[0,1], cov[1,1], 1]
  xq[5] = minimize_scalar(lsq_mag, args=(image, xq)).x
  return xq

def callback(x):
  print('callback', x)

def bfgs_estim(image):
  xq = quick_estim(image)
  # res = minimize(lsq, xq, jac=None, method='Nelder-Mead', tol=1e-3, args=(image), callback=callback)
  res = minimize(lsq, xq, jac=None, method='BFGS', tol=1e-8, args=(image), callback=callback)
  return res.x

def callback(xk):
  pass





# add_noise(image)



# axs[0].imshow(image)
# plt.pause(0.001)


# x0 = [25, 25, 10, 0, 10, 1]
# xq = quick_estim(image)
# xq[5] = minimize_scalar(lsq_mag, args=(image, xq)).x
# print(xq)
# render(xq, axs[1])

# # plt.ioff()
# # plt.show()
# # exit(0)

# res = minimize(lsq, xq, method='Nelder-Mead', tol=1e-4, callback=callback)
# print(res)


# render(res.x, axs[2])


# ### fit a 



# print('error_quick = ', lsq(xq))
# print('error_slow = ', lsq(res.x))

# plt.ioff()

# plt.show()


def quad(x, A):
  # print('~~', A, x, (np.dot(x.T ,np.dot(A, x))))
  return (np.dot(x.T ,np.dot(A, x)))[0,0]

def bell(x):
  return np.exp(-0.5 * np.power(x, 2))

def mgsn(mu, sigma, p, gamma=1.0):
  image = np.zeros((40,40))
  pi = 3.1415926535979323846264
  d = 2
  for index in np.ndindex(40, 40):
    v = 0
    x = np.array([[index[0]-20],[index[1]-20]])
    for k in range(1, 8):
      a1 = p*((1-p)**(k-1))
      a2 = ((2*pi)**(d/2)) * np.sqrt(np.linalg.norm(sigma, ord='fro')) * k**(d/2)
      a3 = -(1/(2*k)) * quad(x-k*mu, np.linalg.inv(sigma))**gamma
      v += ((a1/a2) * np.exp(a3))
    image[index] = v
  image *= 1/(np.sum(image))
  return image


def gaussian(mu, sigma):
  image = np.zeros((40,40))
  pi = 3.1415926535979323846264
  d = 2
  sigma = np.linalg.inv(sigma)
  for index in np.ndindex(40, 40):
    x = np.array([[index[0]-20],[index[1]-20]])
    v = (x-mu).T.dot(sigma).dot(x-mu)
    # print(v)
    v = np.exp(-0.5 * v)
    image[index] = v
  image *= 1/(np.sum(image))
  return image

def truncate(image, x):
  image[:,x:] = 0
  return image

X = np.linspace(0, 40, 40)
Y = np.linspace(0, 40, 40)

fig, axsa = plt.subplots(4, 9)
# fig, axs = plt.subplots(4, 1, squeeze=True)
plt.ion()
plt.draw()
plt.pause(0.001)
plt.show()

print('constructing...')
image_list = [
  gaussian(np.array([[20], [0]]), np.array([[40,0],[0,40]])),
  gaussian(np.array([[-0.5], [-0.5]]), np.array([[40,0],[0,40]])),
  mgsn(np.array([[3], [0]]), np.array([[40,0],[0,40]]), 0.01),
  mgsn(np.array([[3], [3]]), np.array([[40,5],[5,40]]), 0.003),
  mgsn(np.array([[3], [0]]), np.array([[40,0],[0,40]]), 0.015),
  mgsn(np.array([[3], [3]]), np.array([[20,4],[4,20]]), 0.2),
  mgsn(np.array([[3], [0]]), np.array([[40,0],[0,40]]), 0.015, gamma=2),
  mgsn(np.array([[3], [3]]), np.array([[20,4],[4,20]]), 0.2, gamma=2),
  ]

print('starting...')
for i in range(len(image_list)):
  images = {}
  image = image_list[i]
  # print('shape', image.shape)
  print('0')
  images[0] = image
  print('1')
  images[1] = np.zeros(image.shape)
  print('2')
  qe = quick_estim(image)
  print(qe)
  images[2] = imageof(qe)
  print('3')
  # be = bfgs_estim(image)
  # print(be)
  images[3] = imageof(be)
  # images[3] = imageof(qe)

  axs = axsa[:, i]

  for i in range(len(images)):
    print('draw ', i)
    ax = axs[i]
    ax.set_xticks(np.arange(0, 40, 12))
    ax.set_yticks(np.arange(0, 40, 12))
    ax.set_xticks(np.arange(0, 40, 4), minor=True)
    ax.set_yticks(np.arange(0, 40, 4), minor=True)
    ax.grid(which='major', alpha=0.6, c='gray', ls='-')
    ax.grid(which='minor', alpha=0.3, c='gray', ls='-')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.imshow(images[i])
    ax.contour(X,Y,images[i])
    ax.set_xlabel('error = %.7f' % (average_squared_error(images[i], images[0])))
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    plt.show()


print('We compare the results of our approximated Gaussian parameters to what we would expect from least-squares fitting of the parameters in the same window (ie. when the Laplacian of the Gaussian is positive).')
print('When the signal to be fit is a perfect Gaussian, the approximation provides a perfect fit, which is to be expected.')
print('When the signal to be fit is a Gaussian with a skew parameter, the approximation is up to 10\% different form the least squares fit.')
print('When the signal to be fit is a Gaussian with a kurtosis parameter, the approximation is up to 10\% different form the least squares fit.')

plt.ioff()
plt.show()