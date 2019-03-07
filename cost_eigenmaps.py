# import matplotlib.pyplot as plt
from argparse               import ArgumentParser
from functools              import partial
from numpy                  import savetxt as np_savetxt,\
                                    save as np_save
from scipy                  import optimize
from scipy.sparse           import csr_matrix
from time                   import time

import networkx as nx
import numpy as np

from my_utils               import writable



def num_lap(x, A, n, dim):
    ans = 0.0
    x=x.reshape(n,dim)
    for i in range(n):
        for j in range(n):
            ans = ans + (np.linalg.norm(x[i,:]-x[j,:])**2)*A[i,j]
    return ans



def num_lap_neg(x, A, n, dim):
    ans = 0.0
    x=x.reshape(n,dim)
    for i in range(n):
        for j in range(n):
            ans = ans + (np.linalg.norm(x[i,:]-x[j,:])**2)*A[i,j] - \
                            (np.linalg.norm(x[i,:]-x[j,:])**2)*(1-A[i,j])
    return ans



def constraint(x, D, n, dim):
    x = x.reshape(n,dim)
    lst_eq = []
    for i in range(dim):
        for j in range(dim):
            Mij = 0
            for k in range(n):
                Mij = Mij + D[k]*x[k,i]*x[k,j]
            if i == j: lst_eq.append(Mij - 1)
            else: lst_eq.append(Mij)
    return np.array(lst_eq)



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
    if opts.cost == "num_lap":
        res = optimize.minimize(partial(num_lap, A=A, n=num_points, dim=embed_dim), 
                                            np.random.rand(num_points*embed_dim), 
                                            method="SLSQP", constraints={"fun": 
                                            partial(constraint, D=D, n=num_points, 
                                                    dim=embed_dim), "type": "eq"})
    elif opts.cost == "num_lap_neg":
        res = optimize.minimize(partial(num_lap_neg, A=A, n=num_points, dim=embed_dim), 
                                            np.random.rand(num_points*embed_dim), 
                                            method="SLSQP", constraints={"fun": 
                                            partial(constraint, D=D, n=num_points, 
                                                    dim=embed_dim), "type": "eq"})
    print (res.fun)
    t1 = time()
    Y = res.x.reshape(num_points, embed_dim)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    np_save(writable("Embedding_Results", dataset+str(embed_dim)+opts.cost), Y)



if __name__ == "__main__":
    parser = ArgumentParser(description='LaplacianEigenmaps Experiments with different Cost functions')
    parser.add_argument('--dataset', help='name of the dataset in Data Directory', 
                                                                required=True)
    parser.add_argument('--dimension', help='dimension of the embedding', default=128)
    parser.add_argument('--cost', help='cost function to use', default="num_lap")
    opts = parser.parse_args()
    main(opts)
