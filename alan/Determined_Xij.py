
import numpy as np
import cvxpy as cp
import math
#max cost = 100
inf = 101
nc = np.array([[inf,2,7,9], [2,inf,1,6], [7,1,inf,10],[9,6,10,inf]],int)
ncnp = 4
ncn = 4
def D_Xij(cn,cnp,c):


    #set cost
    #c = np.array([[1,2,3], [4,5,6], [7,8,9]],int)
    #c = np.array([[inf,2,3], [2,inf,2], [3,2,inf]])
    #c = np.array([[inf,1,2,3], [1,inf,4,5], [2,4,inf,6],[3,5,6,inf]],int)

    x = cp.Variable(shape=(cn,cnp),boolean = True)
    #x = cp.Variable(shape=(cn,cnp),nonpos = True)
    #x = cp.Variable(shape=(cn,cnp))
    constraints_list = []

    # set constraints
    constraints_list.append(cp.sum(x,axis=1) <= 1)
    constraints_list.append(cp.sum(x,axis=0) <= 1)
    constraints_list.append(cp.sum(x,axis=1) >= 0)
    #constraints_list.append(cp.sum(x,axis=0) >= 0)
    #constraints_list.append(x >= 0)
    constraints_list.append(cp.sum(x) == min(cnp, cn))

    print(x)

    #
    obj = cp.Minimize(cp.sum(cp.multiply(x,c)))
    #obj = cp.Maximize(cp.sum(cp.multiply(x,c)))

    print(obj)
    print(constraints_list)
    prob = cp.Problem(obj,constraints_list)
    prob.solve(solver=cp.MOSEK)
    return(x.value)

"""
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)
    
"""
    

#D_Xij(ncn,ncnp,nc)
