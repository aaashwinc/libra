from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import chisquare


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



def add_noise(img2, m):
  # for index in np.ndindex(40, 40):
  #   img[index] += index[0]*0.0002
  # print(m)
  img = img2.copy()
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

image1 = gaus(mean=[35,35], cov=[[100,0],[0,100]])
image2 = add_noise(image1, 0.0001)
# image = add_noise(image, 0.0001)


# print('image', image)



fig, axs = plt.subplots(4, 1, squeeze=True)
plt.ion()
plt.draw()
plt.pause(0.001)
plt.show()




def imageof(p):
  try:
  # print(p)
    gaus = multivariate_normal(mean=[p[0], p[1]], cov=[[p[2], p[3]],[p[3],p[4]]])
    test = np.zeros(shape=[40,40])
    for index in np.ndindex(40, 40):
      test[index] = gaus.pdf(np.array(index)) * p[5]
    return test
  except:
    return np.inf

def render(p, ax):
  ax.imshow(imageof(p))
  plt.pause(0.001)

def lsq(p):
  error = np.sum(np.square((image-imageof(p))))
  return error


def lsq_only_mu_cov10(p):
  error = np.sum(np.square((image-imageof([p[0], p[1], 10, 0, 10, 1]))))
  return error

def quick_estim(image):
  sum = np.sum(image)
  mu = np.array([0,0]).astype(float)
  cov = np.array([1,0,1])


  indices = np.arange(0,40)
  obs = np.transpose([np.tile(indices, len(indices)), np.repeat(indices, len(indices))])
  wts = []
  for i in obs:
    # print(i)
    wts += [image[i[0], i[1]]]
  wts = np.array(wts)
  print(obs)
  print(wts)

  # print(len(obs))

  mu = np.sum(obs * wts[:,np.newaxis], axis=0) / sum
  cov = np.cov(obs.T, aweights=(wts))
  print(mu)
  print(cov)

  print(cov.shape)
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
  return [mu[0], mu[1], cov[0,0], cov[0,1], cov[1,1], 1]

def callback(xk):
  pass



def lsq_mag(mag, image, p):
  # print(image)
  test = imageof([p[0], p[1], p[2], p[3], p[4], mag])
  error = np.sum(np.square((image-test)))
  print('mag', mag, p, error)
  print('test', np.max(image), np.max(test))
  return error



# add_noise(image)


def goodnessfit(observed, expected):
  observed = observed.flatten()
  expected = expected.flatten()

  expected *= np.sum(observed) / np.sum(expected)

  observed *= 50000
  expected *= 50000


  # observed *= 30
  print(np.sum(observed), np.sum(expected))
  # expected = expected/np.sum(expected)
  # observed = observed/np.sum(observed)
  chi2stat = np.sum ( np.power(expected - observed, 2) / expected )
  chisq, p = chisquare(f_obs=observed, f_exp=expected)

  print('````````````````````')
  print(chi2stat, chisq, p)

def likelihood(img, model):
  pass


# image1 = gaus(mean=[35,35], cov=[[100,0],[0,100]])
# image1 = np.zeros(image1.shape) + 1
# image2 = np.zeros(image1.shape) + 1
# image2 = add_noise(image1, 0.01)
# image2 = gaus(mean=[35,35], cov=[[100,50],[50,100]])
# image2 = image2

# goodnessfit(expected=image1, observed=image2)


image1 = np.random.normal(size=(35, 35))
image2 = np.random.uniform(size=(35, 35))

axs[0].imshow(image1)
axs[1].imshow(image2)
plt.pause(0.001)




# x0 = [25, 25, 10, 0, 10, 1]
# xq = quick_estim(image)
# xq[5] = minimize_scalar(lsq_mag, args=(image, xq)).x
# print(xq)
# render(xq, axs[1])

# plt.ioff()
# plt.show()
# exit(0)

# res = minimize(lsq, xq, method='Nelder-Mead', tol=1e-4, callback=callback)
# print(res)


# render(res.x, axs[2])


# print('error_quick = ', lsq(xq))
# print('error_slow = ', lsq(res.x))

plt.ioff()

plt.show()