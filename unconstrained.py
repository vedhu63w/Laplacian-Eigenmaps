from scipy import optimize
import scipy
from functools import partial

from numpy				import array as np_array,\
						sum as np_sum,\
						square as np_square,\
						diagonal as np_diagonal,\
						dot as np_dot,\
						transpose as np_trans,\
						eye as np_eye
from numpy.linalg		import norm as np_linalg_norm
from numpy.random		import rand as np_rand


def f(x,A,n,dim,D,c):
    ans = 0.0
    x=x.reshape(n,dim)
    for i in range(n):
    	for j in range(n):
    		ans = ans + (np_linalg_norm(x[i,:]-x[j,:])**2)*A[i,j] #\
    		                   # - (np.linalg.norm(x[i,:]-x[j,:])**2)*(1-A[i,j]) 
	lst_eq = []
	# for i in range(dim):
	# 	for j in range(dim):
	# 		Mij = 0
	# 		for k in range(n):
	# 			Mij = Mij + D[k]*x[k,i]*x[k,j]
	# 		if i == j: lst_eq.append(Mij - 1)
	# 		else: lst_eq.append(Mij)
	# constraint = np.array(lst_eq) 
	constraint = np_sum(np_square(np_diagonal(np_dot(x,np_trans(x)) - np_eye(n))) )
	return ans + c*constraint


# def f_jac(x,A,n,dim,D,c):
# 	ans = 0.0
# 	jac = np_zeros((dim,1))
# 	jac[] = np_dot(A,x)


A = [[1,1,0,0],[1,1,1,1],[0,1,1,0],[0,1,0,1]]
A_dense = np_array(A)
D = np_sum(A_dense, axis=0)
A = scipy.sparse.csr_matrix(A_dense)
# A = A_dense

A_sq = np_dot(A, A)

dim = 1
n = 4

res = optimize.minimize(partial(f,A=A,n=n,dim=dim,D=D,c=1), np_rand(n*dim))

print res.x