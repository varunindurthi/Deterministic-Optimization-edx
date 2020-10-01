"""
Portfolio optimization with CVXPY
See examples at http://cvxpy.org
Author: Shabbir Ahmed
"""

import pandas as pd
import numpy as np
from cvxpy import Variable, quad_form, Problem , Minimize

# read monthly_prices.csv
mp = pd.read_csv("monthly_prices.csv",index_col=0)

ret_cols = []
for col in mp.columns:
    mp['next_'+col] = mp[col].shift(-1)
    mp['ret_'+col] = (mp['next_'+col] - mp[col])/mp[col]
    ret_cols.append('ret_'+col)
mr = mp[ret_cols].dropna()
        
# get symbol names
symbols = mp.columns[:3]

# convert monthly return data frame to a numpy matrix
return_data = mr.values.T

# compute mean return
r = np.asarray(np.mean(return_data, axis=1))

# covariance
C = np.asmatrix(np.cov(return_data))

# print out expected return and std deviation
print ("----------------------")
for j in range(len(symbols)):
    print ('%s: Exp ret = %f, Risk = %f' %(symbols[j],r[j], C[j,j]**0.5))
   

# set up optimization model
n = len(symbols)
x = Variable(n)

req_return = 0.02
ret = r.T@x
risk = quad_form(x, C)

prob = Problem(Minimize(risk), 
               [sum(x) == 1, ret >= req_return,
                x >= 0])

# solve problem and write solution
prob.solve()

result_dict = {}
for symbol in range(len(symbols)):
    result_dict[symbols[symbol]] = x.value[symbol]

# print results
print(result_dict)
print("Total return:{}".format(ret.value))
print("Total risk:{}".format((risk.value)**0.5))
