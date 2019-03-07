# import matplotlib.pyplot as plt

from argparse               import ArgumentParser
from numpy                  import savetxt as np_savetxt,\
                                    save as np_save
from time                   import time

from gem.utils              import graph_util, plot_util
from gem.evaluation         import evaluate_graph_reconstruction as gr
from gem.embedding.lap      import LaplacianEigenmaps
from my_utils               import writable



def main(opts):
    dataset = opts.dataset
    embed_dim = int(opts.dimension)
    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_f = 'Data/%s.edgelist' % dataset
 
    # Specify whether the edges are directed
    # isDirected = True
    
    print "Loading Dataset"
    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    #G = G.to_directed()

    embedding = LaplacianEigenmaps(d=embed_dim)
    
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), 
                                            G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    print "Starting Embedding"
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, 
                            is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    np_save(writable("Embedding_Results", "jac_"+dataset+str(embed_dim)), Y)



if __name__ == "__main__":
    parser = ArgumentParser(description='LaplacianEigenmaps Experiments')
    parser.add_argument('--dataset', help='name of the dataset in Data Directory', 
                                                                required=True)
    parser.add_argument('--dimension', help='dimension of the embedding', default=128)
    opts = parser.parse_args()
    main(opts)
