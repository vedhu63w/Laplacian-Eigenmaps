# import matplotlib.pyplot as plt
from argparse               import ArgumentParser
from functools              import partial
from numpy                  import savetxt as np_savetxt,\
                                    save as np_save
from numpy				    import array as np_array,\
						    sum as np_sum,\
						    square as np_square,\
						    diagonal as np_diagonal,\
						    dot as np_dot,\
						    transpose as np_trans,\
						    eye as np_eye,\
                            zeros_like as np_zeros_like
from numpy.linalg		    import norm as np_linalg_norm
from numpy.random		    import rand as np_rand
from scipy                  import optimize
from scipy.sparse           import csr_matrix
from time                   import time

import networkx as nx
import numpy as np

from my_utils               import writable



def num_lap(x, A, n, dim, D, c):
    ans = 0.0
    x=x.reshape(n,dim)
    for i in range(n):
    	for j in range(n):
    		ans = ans + (np_linalg_norm(x[i,:]-x[j,:])**2)*A[i,j] #\
    		                   # - (np.linalg.norm(x[i,:]-x[j,:])**2)*(1-A[i,j]) 
	lst_eq = []
	constraint = np_sum(np_square(np_diagonal(np_dot(x,np_trans(x)) - np_eye(n))) )
	return ans + c*constraint



def lap_jac(x, A, n, dim, c):
    # x = x.reshape(n,dim)
    x_grad = np_zeros_like(x)
    for i in range(n):
        for j in range(dim):
            for k in range(n):
                x_grad[i*dim+j] = 4*(x[i*dim+j] - x[k*dim+j])*A[i,k]\
                                        + 2*c*x[i*dim+j]
    return x_grad


def main(opts):
    dataset = opts.dataset
    embed_dim = int(opts.dimension)
    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_f = 'Data/%s.edgelist' % dataset
    
    G = nx.read_edgelist(edge_f)
    A = nx.adjacency_matrix(G)
    num_points = A.shape[0]
    D = np.sum(A, axis=0)
    D = np.squeeze(np.asarray(D))
    A = csr_matrix(A)
    res = optimize.minimize(partial(num_lap,A=A,n=num_points,dim=embed_dim,D=D,c=1), 
                                            np.random.rand(num_points*embed_dim),
                                            jac=partial(lap_jac,A=A,n=num_points,
                                                dim=embed_dim,c=1))
    print (res.fun)
    t1 = time()
    Y = res.x.reshape(num_points, embed_dim)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    np_save(writable("Embedding_Results", dataset+str(embed_dim)+opts.cost), Y)



if __name__ == "__main__":
    parser = ArgumentParser(description='Laplacian Eigenmaps Experiments with different Cost functions')
    parser.add_argument('--dataset', help='name of the dataset in Data Directory', 
                                                                required=True)
    parser.add_argument('--dimension', help='dimension of the embedding', default=128)
    parser.add_argument('--cost', help='cost function to use', default="num_lap")
    opts = parser.parse_args()
    main(opts)
