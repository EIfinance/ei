import numpy as np
from fbm import fbm, times
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

d=pd.read_csv("RealizedVarianceData.csv")["Realized Variance (5-minute)"].tolist()[1:]

d=np.array([np.log10(float(x)) for x in d])
Mvar=np.mean(np.array([x for x in d if x!=np.nan]))

delta=1
def M(X,q,lag):
    a=np.mean(abs(np.diff(X[::int(lag)]))**q)
    return(a)

def H_exp(X):

    Delta=np.logspace(1,2.1,100)
    Q=[0.5,0.75,1,1.25,1.5,2]
    Y=np.zeros(len(Delta))   
    h_exp=0
    for q in Q:
        i=0
        for d in Delta:   
            Em=M(X,q,d)
            Y[i]=Em
            i+=1
        Y=np.log10(Y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(Delta),Y)
        x=[1,2.1]
        y=[1*slope+intercept,2.1*slope+intercept]
        plt.plot(x,y)
        print(slope/q)
        h_exp+=slope/q/len(Q)
        plt.scatter(np.log10(Delta),Y,marker=".",s=10)
    plt.title("H experimental = "+str(h_exp))
    plt.xlabel(r"$\log(\Delta)$")
    plt.ylabel(r"$\log(m(q,\Delta))$")
    plt.show()
    return(h_exp)


for k in range(len(d)):
    if str(d[k])=="nan":
        d[k]=d[k-1]
    

def histo(X):
    plt.subplot(2, 2, 1)
    plt.hist(np.diff(X[::1]),30,label="1 lag",density=True)
    plt.xlabel("frequency")
    plt.ylabel("X")
    plt.legend()
    plt.title("The distribution of the log-volatility increments")

    plt.subplot(2,2,2)
    plt.hist(np.diff(X[::5]),30,label="3 lag",density=True)
    plt.xlabel("frequency")
    plt.ylabel("X")
    plt.legend()
    plt.title("The distribution of the log-volatility increments")

    plt.subplot(2, 2, 3)
    plt.hist(np.diff(X[::25]),30,label="7 lag",density=True)
    plt.xlabel("frequency")
    plt.ylabel("X")
    plt.legend()
    plt.title("The distribution of the log-volatility increments")

    plt.subplot(2, 2, 4)
    plt.hist(np.diff(X[::125]),30,label="15 lag",density=True)
    plt.xlabel("frequency")
    plt.ylabel("X")
    plt.legend()
    plt.title("The distribution of the log-volatility increments")
    plt.show()


histo(d)
h_exp=H_exp(d)