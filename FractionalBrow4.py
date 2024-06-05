import numpy as np
from fbm import fbm, times
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

d=pd.read_csv("RealizedVarianceData.csv")["Realized Variance (5-minute)"].tolist()[1:]
d=np.array([np.log10(float(x)) for x in d])
for k in range(len(d)):
    if str(d[k])=="nan":
        d[k]=d[k-1]
Mvar=np.mean([x for x in d if str(x)!="nan"])
N=len(d)


H=0.147402801965937
delta=1

def pred_sigma(log_sigmasquare,delta,H):
    pred = 0
    coeff=((delta**(H+1/2))*np.cos(H*np.pi)/np.pi)
    for k in range(len(log_sigmasquare)):
        d=(k+delta+1/2)*((k+1/2)**(H+1/2))
        a=log_sigmasquare[-k-1]/d
        pred+=a
    return(pred*coeff)

def P(delta):
    a,b=0,0
    for k in range(1,N-delta):
        a+=(pred_sigma(d[:k+1],delta,H)-d[k+delta])**2
        b+=(d[k+delta]-Mvar)**2
    return(a/b)


X=[pred_sigma(d[:k+1],1,H) for k in range(len(d))]
plt.plot(d,label="Real")
plt.plot(X,label="Pred")
plt.legend()
plt.show()