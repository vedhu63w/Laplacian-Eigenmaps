import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
from argparse import ArgumentParser
#from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.preprocessing import normalize

dataset = "flickr"
parser = ArgumentParser()
parser.add_argument('-input', type=str, default="Data/%s.mat"%dataset)
parser.add_argument('-output', type=str, default="Embedding_Results/num_%s"%dataset)
parser.add_argument('-embed_size', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lr1', type=float, default=1e-2)
parser.add_argument('-lr2', type=float, default=1e-2)
parser.add_argument('-opt', choices=['sgd', 'adam', 'RMSProp'], default='adam')
parser.add_argument('-epochs', type=int, default=200)
parser.add_argument('-k', type=int, default=40)
parser.add_argument('-print_every', type=int, default=1)
args = parser.parse_args()

args.output = args.output+str(args.embed_size)

def sample(pdist):
    return pdist.indices[np.argmax(np.random.multinomial(1, pdist.data))]


# TODO convert data to CSR
similarity_matrix = normalize(sp.io.loadmat(args.input)['network'].tocsr(), norm='l1', axis=1)
adjmat = sp.io.loadmat(args.input)['network'].tocsr().toarray()
comp_adjmat = np.ones(adjmat.shape) - adjmat
eps = 0.01
degree = np.diag(np.sum(adjmat,axis=1)) #+ eps 
#laplacian = (np.eye(adjmat.shape[0]) - np.matmul((1.0/degree),adjmat)).astype(np.float32)
laplacian = (degree - adjmat).astype(np.float32)
comp_laplacian = (degree - comp_adjmat).astype(np.float32)
pool = mp.cpu_count()
num_nodes = similarity_matrix.shape[0]
embed_size = args.embed_size
batch_size = args.batch_size
max_epochs = args.epochs
k = args.k

init_range = 4*np.sqrt(6.0/(2*embed_size))
embed_matrix = tf.Variable(tf.random_uniform([num_nodes, embed_size], -1*init_range, init_range, dtype=tf.float32))

'''concerned_nodes = tf.placeholder(tf.int32, shape=batch_size)
positive_nodes = tf.placeholder(tf.int32, shape=None)
negative_nodes = tf.placeholder(tf.int32, shape=batch_size*k)

nce_weight = embed_matrix
bias = tf.Variable(tf.zeros([num_nodes]))

con_embed = tf.nn.embedding_lookup(embed_matrix, concerned_nodes)
pos_embed = tf.nn.embedding_lookup(embed_matrix, positive_nodes)
neg_embed = tf.reshape(tf.nn.embedding_lookup(embed_matrix, negative_nodes), (k, -1, embed_size))

nce_bias = tf.nn.embedding_lookup(bias, concerned_nodes)

# tmp_tf = con_embed
# tmp_tf = tf.reduce_sum(tf.reduce_sum(tf.log(tf.sigmoid(-1*(nce_bias + tf.reduce_sum(tf.multiply(tf.expand_dims(con_embed, axis=0), neg_embed), axis=2)))), axis=0))
pos_sim = -1*tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(nce_bias + tf.reduce_sum(tf.multiply(con_embed, pos_embed), axis=1)), 1e-7, 1-1e-7)), axis=0)
neg_sim = -1*tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(tf.sigmoid(-1*(tf.expand_dims(nce_bias, axis=0) + tf.reduce_sum(tf.multiply(tf.expand_dims(con_embed, axis=0), neg_embed), axis=2))), 1e-7, 1-1e-7)), axis=0))
#labels = tf.concat([tf.zeros(batch_size,1),tf.ones(batch_size,k)],axis=1)

loss = (pos_sim + 1*neg_sim)'''
#embed_matrix = tf.Variable(tf.random_uniform([num_nodes, embed_size], -1*init_range, init_range, dtype=tf.float32))
#lap = 
lambda_reg = 100
reg = tf.reduce_mean(tf.square(tf.diag_part(tf.matmul(embed_matrix,tf.transpose(embed_matrix) ) - tf.eye(adjmat.shape[0]) )))
pos_loss = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(embed_matrix),laplacian),embed_matrix))) + lambda_reg* reg
neg_loss = -1*tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(embed_matrix),comp_laplacian),embed_matrix)))
loss = pos_loss + neg_loss
optimizer_1 = tf.train.AdamOptimizer(args.lr1).minimize(pos_loss)
optimizer_2 = tf.train.AdamOptimizer(args.lr2).minimize(neg_loss)

# Get embeddings
get_embed = tf.nn.l2_normalize(embed_matrix, axis=1)

num_epochs = 0
sess = tf.Session()
sess.run(tf.global_variables_initializer())


n_batches = int(num_nodes/batch_size)
for epoch_id in range(max_epochs):
    epoch_loss = 0
    pos_e_loss = 0
    neg_e_loss = 0
    for num_iter in range(1):
        '''sampled_nodes = np.random.randint(0, num_nodes, batch_size)
        concerned_similarities = similarity_matrix[sampled_nodes]
        pos_nodes = [sample(concerned_similarities[i]) for i in range(batch_size)]
        neg_nodes = np.random.randint(num_nodes, size=batch_size*k)
        #print(sess.run([tf.shape(tf.multiply(con_embed, pos_embed)),tf.shape(con_embed),tf.shape(tf.reduce_sum(tf.multiply(con_embed, pos_embed), axis=1))],
        #                         feed_dict={concerned_nodes:sampled_nodes, positive_nodes:pos_nodes,
        #                                    negative_nodes:neg_nodes}))
        # negative_nodes = positive_nodes[np.random.uniform(batch_size, size=batch_size * k)]
        '''
        loss_value, _,pos_val,neg_val= sess.run([pos_loss, optimizer_1,pos_loss,neg_loss])

        
        epoch_loss += loss_value
        pos_e_loss += pos_val
        neg_e_loss += neg_val
        loss_value, _,pos_val,neg_val= sess.run([pos_loss, optimizer_2,pos_loss,neg_loss])

        epoch_loss += loss_value
        pos_e_loss += pos_val
        neg_e_loss += neg_val
    print('Epoch: ', epoch_id, ' loss: ', epoch_loss/n_batches, '+loss: ', pos_e_loss/n_batches, '-loss: ', neg_e_loss/n_batches)
embeddings = sess.run(get_embed)
np.save(args.output, embeddings)
# np.savetxt("tmp.csv", embeddings)
