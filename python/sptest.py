import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep
import scipy

x = np.linspace( 0, 1, 120 )
timeList = np.linspace( 0, 1, 15 )
signal = np.fromiter( ( np.sin( 2 * np.pi * t +.3 ) for t in timeList ), np.float )
signal[ -1 ] -= .15
myKnots=np.linspace( .05, .95, 8 )  
spl = splrep( timeList, signal, t=myKnots, k=2 )
print(spl)
fit = splev(x,spl)

fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.plot( timeList, signal, marker='o' )
ax.plot( x, fit , 'r' )
for i in myKnots:
    ax.axvline( i )
plt.show()