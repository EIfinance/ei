import numpy as np
from fbm import fbm, times
import matplotlib.pyplot as plt
from scipy import stats

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

def M(X,q,lag):
    a=np.mean(abs(np.diff(X[::int(lag)]))**q)
    return(a)

def H_exp(h):
    X=log_vol(1,10**-3,h)
    Delta=np.logspace(1,4,100)
    Q=[0.5,1,1.5,2,3]
    Y=np.zeros(len(Delta))
    h_exp=0
    for q in Q:
        i=0
        for d in Delta:
            Y[i]=M(X,q,d)
            i+=1
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(Delta),np.log10(Y))
        x=[1,4]
        y=[1*slope+intercept,4*slope+intercept]
        plt.plot(x,y)
        h_exp+=slope/q/len(Q)
        plt.scatter(np.log10(Delta),np.log10(Y),marker=".",s=10)
    plt.title("H theoric = "+str(h)+", H experimental = "+str(h_exp))
    plt.xlabel(r"$\log(\Delta)$")
    plt.ylabel(r"$\log(m(q,\Delta))$")
    plt.show()

H_exp(0.8)
H_exp(0.5)
H_exp(0.1)