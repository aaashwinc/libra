import numpy as np
from numpy import sqrt
import scipy.optimize
import matplotlib.pyplot as plt

def ell(A,x):
  # print('A', A)
  # print('x', x)
  # print('ell', x.T @ A @ x)
  return x.T @ A @ x

def g(A, x):
  return x * 1.0/np.sqrt(ell(A, x))

# objective(x) for ellipsoid (A, 0) DISTANCE p
def f(A, x, p):
  # return (g(A,x)-p).T @ (g(A,x)-p)
  return (1.0/(x.T@A@x) * (x.T@x)) - (1.0/np.sqrt(x.T@A@x))*2.0*x.T@p

def df(A, x, p):
  xAx = np.asscalar(ell(A, x))
  VxAx = np.asscalar(sqrt(xAx))
  Ax  = A@x
  xp  = np.asscalar(x.T@p)
  xTx = np.asscalar(x.T.dot(x))
  xTAAT = (x.T @ (A + A.T)).T
  # print('hi', x.T, (A+A.T), xTAAT)
  # print('hi', VxAx);
  # print('hi', xTx, xTx/(2.0*VxAx), Ax)
  # return 0
  return (2.0*x/(xAx)) + (-xTx/(xAx**2.0) * xTAAT) - (((1.0/VxAx)*2*p) + (2*xp)*(-xTAAT/(2.0*(VxAx**3.0))))
  # return (x/(xAx)) + (x.T@x/(2.0*VxAx) * Ax) - (((1.0/VxAx)*2*p) + (2*xp)*(-Ax/(VxAx**3.0)))
  # return (x/(xAx)) + (x.T@x/(2.0*VxAx) * Ax) - (((1.0/VxAx)*2*p) + (2*xp)*(-Ax/(VxAx**3.0)))

def df2(A,x,p):
  pass

def gradient_descent(A, x, p):
  itr = 0
  while True:
    ++itr
    fx  = f(A,x,p)
    dfx = -df(A,x,p)
    x = x + 0.5*dfx
    # print(g(A,x).T, np.asscalar(fx), np.linalg.norm(dfx))
    if(np.asscalar(dfx.T @ dfx) < 0.1):
      break
  # print('iterations ',itr)
  return x

def ellipsoid_point():
  pass


def quad_to_linear(A):
  return scipy.linalg.sqrtm(np.linalg.inv(A))
def linear_to_quad(A):
  return np.linalg.inv(A@A)

def distance(p0, A0, p1, A1):
  p0 = np.array([p0]).T
  p1 = np.array([p1]).T
  # print(p0)
  # distance between ellipsoids.
  p1 -= p0;
  p0 -= p0;
  inv_A0 = np.linalg.inv(A0)
  A0 = inv_A0 @ A0
  A1 = inv_A0 @ A1
  p1 = inv_A0 @ p1
  
  # now, p0 = 0, A0 = I.
  # A = np.linalg.inv(A1)
  A = A1
  Aq = linear_to_quad(A)
  x = g(A, -p1)
  p = -p1
  x = scipy.optimize.minimize(
        fun=lambda x: np.asscalar(f(Aq,x,-p1)),
        x0=x,
        method='BFGS',
        jac=lambda x: np.array(df(Aq,x,p).T.tolist()[0]),
        options={'gtol':0.0001, 'disp':True})
  x = x.x
  # print(x)
  x = g(Aq, x)
  # print(A)
  # print(x)
  # print(x@A@x.T)
  distance = np.linalg.norm(x.T - (-p1))
  print('distance', distance)
  # print('closest points', x, -p1)
  return (A1, Aq, -p1, x)

# A0 = np.matrix('0.0025 0; 0 1')
# A0 = np.matrix('0.0025 0; 0 1')

# print(ell(A, np.array([[1,3]]).T))

# x = np.array([[0,30]]).T
# x = g(A,x)
# p = np.array([[40,0]]).T

# print('find closest point to', p.T, 'on ellipsoid:\n')
# print(A)
# print()

# # print(gradient_descent(A,x,p)
# print('min', x.x)


# Fixing random state for reproducibility
np.random.seed(19680801)


def ellipse(p, A, c=None, mode='linear'):
  # print('A', A)
  xs = []
  ys = []
  for theta in np.arange(0, 2*3.14159265, 0.1):
    pp = np.array([np.cos(theta), np.sin(theta)]).reshape(2,1)
    # print('shape', pp)
    if mode == 'quadratic':
      pp  =g(A, pp)
    else:
      pp = A @ pp
    # pp = pp.flatten()
    # print('list', np.concatenate(pp).flatten())
    # print('pp', pp.shape, pp, pp.reshape(2,1).flatten().shape)
    xs += [p[0] + pp[0,0]]
    ys += [p[1] + pp[1,0]]
  plt.scatter(xs, ys, c=c, alpha=0.3)

A0 = np.matrix('2 1; 1 2')
p0 = np.array([-1,0])
A1 = np.matrix('1 0; 0 1')
p1 = np.array([-2.8,0.1])
plt.gca().set_aspect('equal', adjustable='box')

A0L = quad_to_linear(A0)
# print('AAA\n', A0L, '\n', A0)

ellipse(p0, A0, c='red',   mode='linear')
# ellipse(p0, linear_to_quad((A0)), c='blue',   mode='quadratic')


ellipse(p1, A1, c='red',  mode='linear')

(A, Aq, p, x) = distance(p0, A0, p1, A1)

ellipse(np.array([0,0]), A, c='blue')
# ellipse(np.array([0,0]), Aq, c='green', mode='quadratic')
ellipse(p, np.matrix('1 0; 0 1'), c='blue')

plt.scatter([p[0,0], x[0,0]], [p[1,0], x[0,1]], c='green')


if False:

  ellipse(p0,A0, c='red')
  ellipse(p1,A1, c='red')

  (A, p, x) = distance(p0, A0, p1, A1)

  # print(x[0,0], x[0,1])
  ellipse(np.array([0,0]), A, c='blue')
  ellipse(p, np.matrix('1 0; 0 1'), c='blue')
  # print(p[0,0], p[1,0])
  plt.scatter([p[0,0], x[0,0]], [p[1,0], x[0,1]])
# plt.scatter(x, y)
# plt.show()