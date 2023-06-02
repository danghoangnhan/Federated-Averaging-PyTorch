
import numpy as np
import cvxpy as cp
import math
from Determined_Xij import D_Xij
#max cost = 100
inf = 101
cnp = 4
cn = 4
number_client = 4
#c = np.array([[inf,2,7,9], [2,inf,1,6], [7,1,inf,10],[9,6,10,inf]],int)
c = np.zeros((cn,cnp))
c[c == 0]=inf
####計算cost
##Define GL
#GL = np.array([0,1,2,3,4,5,6,7,8,9])
GL = np.array([0,1,2,3])

##Define OL
#OL = np.array([0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9])
OL = np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]])

##Define TL
#TL = np.zeros((number_client,GL.size),int)
TL =np.array(OL)
##Define LL
LL =np.array(TL)
#LL = np.zeros((4,4))


for i in range(np.size(TL,0)):
    for j in range(np.size(TL,1)):
        if(TL[i][j] == 1):
            LL[i][j] = 0
        else:
            LL[i][j] = 1


TTL = np.array(TL.T)
n_L = np.array(OL@LL.T)
for i in range(np.size(n_L,0)):
    for j in range(np.size(n_L,1)):
        n_L[i][j] = np.size(GL)- n_L[i][j]  

print("c:\n",c)
print("GL:\n",GL)
print("OL:\n",OL)
print("TL:\n",TL)
print("LL:\n",LL)

print("n_L:\n",n_L)
test=np.array(D_Xij(cn,cnp,n_L))
print("test:\n",test) 