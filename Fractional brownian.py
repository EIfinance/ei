import numpy as np
from fbm import fbm, times
import matplotlib.pyplot as plt

nu,X0,alpha=0.3,-5,5*10**(-4)
m=X0


def log_vol(T,lag,H):
    delta=lag/252
    N=int(T/delta)
    X=np.zeros(N)
    X[0]=X0
    W = fbm(N-1, H, length=T, method='daviesharte')
    for i in range(len(W)-1):
        X[i+1]=X[i]+(W[i+1]-W[i])*nu+alpha*(m-X[i])*delta
    return(X)

H=0.5
X=log_vol(10000,1,H)
plt.subplot(2, 2, 1)
plt.hist(np.diff(X[::1]),100,label="1 lag",density=True)
plt.xlabel("frequency")
plt.ylabel("X-X0")
plt.legend()
plt.title("The distribution of the log-volatility increments for H = "+str(H))

plt.subplot(2,2,2)
plt.hist(np.diff(X[::5]),100,label="5 lag",density=True)
plt.xlabel("frequency")
plt.ylabel("X-X0")
plt.legend()
plt.title("The distribution of the log-volatility increments for H = "+str(H))

plt.subplot(2, 2, 3)
plt.hist(np.diff(X[::25]),100,label="25 lag",density=True)
plt.xlabel("frequency")
plt.ylabel("X-X0")
plt.legend()
plt.title("The distribution of the log-volatility increments for H = "+str(H))

plt.subplot(2, 2, 4)
plt.hist(np.diff(X[::125]),100,label="125 lag",density=True)
plt.xlabel("frequency")
plt.ylabel("X-X0")
plt.legend()
plt.title("The distribution of the log-volatility increments for H = "+str(H))
plt.show()