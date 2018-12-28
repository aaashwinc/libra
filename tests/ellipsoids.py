import numpy as np
from numpy import sqrt
import scipy.optimize
import matplotlib.pyplot as plt

def ell(A,x):
  # print('ell', A, x)
  return x.T @ A @ x

def g(A, x):
  return x * 1.0/np.sqrt(ell(A, x))

# objective(x) for ellipsoid (A, 0) DISTANCE p
def f(A, x, p):
  # return (g(A,x)-p).T @ (g(A,x)-p)
  v = (1.0/(x.T@A@x) * (x.T@x)) - (1.0/np.sqrt(x.T@A@x))*2.0*x.T@p
  # print('f', v)
  return v

def df(A, x, p):
  xAx = np.asscalar(ell(A, x))
  VxAx = np.asscalar(sqrt(xAx))
  Ax  = A@x
  xp  = np.asscalar(x.T@p)
  xTx = np.asscalar(x.T.dot(x))
  xTAAT = (x.T @ (A + A.T)).T
  df =  (2.0*x/(xAx)) + (-xTx/(xAx**2.0) * xTAAT) - (((1.0/VxAx)*2*p) + (2*xp)*(-xTAAT/(2.0*(VxAx**3.0))))
  # print('df', df)
  return df
def gradient_descent(A, x, p):
  print('gradient descent')
  print(x)
  print(p)
  itr = 0
  while True:
    ++itr
    fx  = f(A,x,p)
    dfx = -df(A,x,p)
    x = x + 0.5*dfx
    print('itr', itr, g(A,x), np.asscalar(fx), np.linalg.norm(dfx))
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

def distance(ap0, aA0, ap1, aA1):
  p0 = ap0
  A0 = aA0
  p1 = ap1
  A1 = aA1

  # distance between ellipsoids.

  VA0 = scipy.linalg.sqrtm(aA0)
  VA1 = scipy.linalg.sqrtm(aA1)

  VA0i = np.linalg.inv(VA0)
  VA1i = np.linalg.inv(VA1)

  VA0A1i  = VA0 @ VA1i
  VA1A0iv = VA1 @ VA0i

  print('VA0 = \n'     , VA0);
  print('VA1 = \n'     , VA1);
  print('VA0i = \n'    , VA0i);
  print('VA1i = \n'    , VA1i);
  print('VA0A1i = \n'  , VA0A1i);
  print('VA1A0iv = \n' , VA1A0iv);
  
  p1 =  VA0 @ (p0 - p1)
  x = g(VA0A1i, p1)
  Aq = VA1A0iv @ VA1A0iv


  print('p1 = \n', p1);
  print('x  = \n', x);
  print('Aq = \n', Aq);

  # p1 = np.array([-4.82686, -4.9731, 0.161859])
  # x  = np.array([0.863359, -0.889517, 0.028951])
  # Aq = np.matrix('1.23411 0.834364 0.297033; 0.61754 2.0786 0.380122; 0.154424 0.261383 0.931259')
  # Aq = np.matrix('1.23411 0.834364 0.297033; 0.61754 2.0786 0.380122; 0.154424 0.261383 0.931259')
  # print('p1 =\n', p1)
  # print('x  =\n', x)
  # print('Aq =\n', Aq)

  print('g = ', g(Aq,x))
  # gradient_descent(Aq, x, p1)

  x = scipy.optimize.minimize(
        fun=lambda x: (f(Aq,x,p1)),
        x0=x,
        method='BFGS',
        jac=lambda x: (df(Aq,x,p1)),
        options={'gtol':0.0001, 'disp':False})
  x = x.x
  print('x = ', x)

  print('c++ solution', g(Aq, np.array([-6.176, -3.018, 2.833])))
  print('python solution', g(Aq, x))
  print("f(x') = ", f(Aq, np.array([-6.176, -3.018, 2.833]), p1))   
  print("f(x) = ", f(Aq, x, p1))   
  # print(x)
  x = g(Aq, x)
  # print(x)
  # print(p1)
  # print(x.T - (-p1))
  distance = np.linalg.norm(x.T - (p1))
  print('distance', distance)
  return (A1, Aq, p1, x)

# Fixing random state for reproducibility
np.random.seed(19680801)


def ellipse(p, A, c=None, mode='linear'):
  xs = []
  ys = []
  for theta in np.arange(0, 2*3.14159265, 0.1):
    pp = np.array([np.cos(theta), np.sin(theta)]).reshape(2,1)
    if mode == 'quadratic':
      pp  =g(A, pp)
    else:
      pp = A @ pp
    xs += [p[0] + pp[0,0]]
    ys += [p[1] + pp[1,0]]
  plt.scatter(xs, ys, c=c, alpha=0.3)

A0 = np.matrix('1 0 0; 0 3 0; 0 0 0.25')
p0 = np.array([0,0,0])
A1 = np.matrix('2 0 0; 0 1 0; 0 0 1')
p1 = np.array([4,0,0])

p0 = np.array([46.251, 3.784, 2.895])
p1 = np.array([48.919, 7.445, 2.841])
A0 = np.matrix('3.12201  0.0960595  -0.051745; 0.0960595 1.78576 -0.0372085; -0.051745 -0.0372085 1.01729')
A1 = np.matrix(' 3.83764  1.80666 0.339901; 1.80666  3.79705 0.396492; 0.339901 0.396492 0.943025')

plt.gca().set_aspect('equal', adjustable='box')


# print(A1, quad_to_linear(linear_to_quad(A1)))

(A, Aq, p, x) = distance(
  p0,
  ((A0)),
  p1,
  ((A1)))

# (A, Aq, p, x) = distance(p0, A0, p1, A1)

# ellipse(p0, A0, c='red', mode='linear')
# ellipse(p1, A1, c='red', mode='linear')
# ellipse(np.array([0,0]), A, c='blue')
# ellipse(p, np.matrix('1 0; 0 1'), c='blue')

# plt.scatter([p[0,0], x[0,0]], [p[1,0], x[0,1]], c='green')

# plt.show()