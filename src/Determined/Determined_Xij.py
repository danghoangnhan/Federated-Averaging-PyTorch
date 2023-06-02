
import numpy as np
import cvxpy as cp
import math
#max cost = 100
inf = 101
nc = np.array([[inf,2,7,9], [2,inf,1,6], [7,1,inf,10],[9,6,10,inf]],int)
ncnp = 4
ncn = 4
print (cp.installed_solvers())
def D_Xij(cn,cnp,c):
    ##將nan 轉成無限大
    print("cnp:",cnp)
    for i in range(np.size(c,0)):
        for j in range(np.size(c,1)):
            #print("c[",i,"][",j,"]: ",c[i][j])
            if(np.isnan(c[i][j])):
                c[i][j] = inf
    #print("c: ",c)

    #set cost
    #c = np.array([[1,2,3], [4,5,6], [7,8,9]],int)
    #c = np.array([[inf,2,3], [2,inf,2], [3,2,inf]])
    #c = np.array([[inf,1,2,3], [1,inf,4,5], [2,4,inf,6],[3,5,6,inf]],int)

    x = cp.Variable(shape=(cn,cn),boolean = True)
    #x = cp.Variable(shape=(cn,cnp),integer = True)
    #x = cp.Variable(shape=(cn,cnp),nonpos = True)
    #x = cp.Variable(shape=(cn,cnp))
    constraints_list = []

    # set constraints
    constraints_list.append(cp.sum(x,axis=1) <= 1)
    constraints_list.append(cp.sum(x,axis=0) <= 1)
    constraints_list.append(cp.sum(x,axis=1) >= 0)
    constraints_list.append(cp.sum(x,axis=0) >= 0)
    constraints_list.append(x >= 0)
    #constraints_list.append(x >= 0)
    constraints_list.append(cp.sum(x) == min(cnp, cn))
    #constraints_list.append(sum_entries(x))
    #print("x:",x)

    #
    obj = cp.Minimize(cp.sum(cp.multiply(x,c)))
    #obj = cp.Maximize(cp.sum(cp.multiply(x,c)))

    #print("obj: ",obj)
    #print("constraints_list: ",constraints_list)
    prob = cp.Problem(obj,constraints_list)
    prob.solve(solver=cp.MOSEK)
    """
    prob.solve(solver=cp.MOSEK)
    prob.solve(solver=cp.MOSEK)
    prob.solve(solver=cp.MOSEK)
    prob.solve(solver=cp.MOSEK)
    """
    return(x.value)

"""
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)
    
"""
    

#D_Xij(ncn,ncnp,nc)
