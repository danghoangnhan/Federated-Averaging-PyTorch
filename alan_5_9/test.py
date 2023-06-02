
import numpy as np
import cvxpy as cp

m = 3
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
print("s0:",s0)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
p = c.T@x
print(c)
print(cp.Minimize(p))
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()


# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

