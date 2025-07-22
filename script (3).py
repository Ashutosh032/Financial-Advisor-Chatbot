import numpy as np, pandas as pd
P = 500000
annual_rate=8
r=annual_rate/12/100
n=15*12
emi = P*r*(1+r)**n/((1+r)**n -1)
months=np.arange(1,n+1)
balance=P
balances=[]
for m in months:
    interest=balance*r
    principal=emi-interest
    balance-=principal
    balances.append(balance)

df=pd.DataFrame({'Month':months,'Outstanding Balance':balances})
df_head=df.head()
df_tail=df.tail()
df.shape