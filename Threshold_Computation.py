import pickle
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csgraph
import math
import argparse
import logging
import theano
import networkx as nx
from theano import tensor as T
logger = logging.getLogger(__name__)

def _create_D_inverse(graph):
    """
    Create the D^{-1} matrix of given graph for GraRep matrix construction.
    """
    """
    Arg types:
        * **graph** *(NetworkX graph)* - The graph being embedded.
    """
    index = np.arange(graph.number_of_nodes())
    values = np.array(
        [1.0 / graph.degree[node] for node in range(graph.number_of_nodes())]
    )
    shape = (graph.number_of_nodes(), graph.number_of_nodes())
    D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
    return D_inverse

def GraRep(G, n, dim):
    """
    Create the GraRep matrix.
    """
    """
    Arg types:
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **dim** *(int)* - The dimension of the original GraRep model.
    """
    graph = nx.from_numpy_matrix(G[0:n, 0:n])
    D_inv = _create_D_inverse(graph)
    A_hat = D_inv.dot(G[0:n,0:n])
    A_tilde = sparse.coo_matrix(A_hat**dim)
    scores = np.log(A_tilde.data) - math.log(A_tilde.shape[0])
    rows = A_tilde.row[scores < 0]
    cols = A_tilde.col[scores < 0]
    scores = scores[scores < 0]
    target_matrix = sparse.coo_matrix(
        (scores, (rows, cols)), shape=A_tilde.shape, dtype=np.float32
    )
    return (target_matrix).todense()

def AROPE(G, n, weights = [1,0.01,0.0001]):
    """
    Create the AROPE matrix.
    """
    """
    Arg types:
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **dim** *(int)* - The dimension of the original GraRep model.
        * **weights** *(list)* - The parameters for different order of proximity matrices.
    """
    G_temp = sparse.csc_matrix(np.copy(G[0:n,0:n]))
    G_origin= sparse.csc_matrix(np.copy(G[0:n,0:n]))
    G_new = np.zeros([n,n])
    for i in weights:
        G_new += i * G_temp
        G_temp = G_temp.dot(G_origin)
    return G_new
    
def LE(G, n, state = 0):
    """
    Create the LE matrix.
    """
    """
    Arg types:
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **state** *(binary)* - The indicator of whether use the original matrix as the target matrix
    """
    if state == 0:
        graph = nx.from_numpy_matrix(G[0:n,0:n])
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.normalized_laplacian_matrix(graph, nodelist=range(number_of_nodes))
        M = L_tilde.todense()
    else:
        M = G[0:n, 0:n]
    return M


def netmf(A_origin, n, window = 10, b = 1):
    """
    Create the LE matrix.
    """
    """
    Arg types:
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **window** *(int)* - Window size.
        * **b** *(int)* - The number of negative sampling in skip-gram.
    """
    A = np.copy(A_origin[0:n,0:n])
    vol = float(A.sum())
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    try:
        evals, evecs = sparse.linalg.eigsh(X, 256, which='LA')
    except:
        evals, evecs = sparse.linalg.eigsh(X, 128, which='LA', tol=1E-6)

    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
        sum_eval = np.maximum(evals, 0)
        X = sparse.diags(np.sqrt(sum_eval)).dot(D_rt_invU.T).T * math.sqrt(vol/b)#这里np.sqrt()的目的是为了
        m = T.matrix()
        mmT = T.dot(m, m.T)
        f = theano.function([m], T.log(T.maximum(mmT, 1)))
        deepwalk_matrix = f(X.astype(theano.config.floatX))
    return deepwalk_matrix

def condition(A1, G, n, m, s, method):
    """
    Calculate the equation(7) in the article for threshold judgement.
    """
    """
    Arg types:
        * **A1** *(numpy array)* - The original target matrix with size n.
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **m** *(int)* - The new arrived node size.
        * **s** *(numpy array)* - The singular value sequence in the desending order.
        * **method** *(str)* -The function name of constructing target method.
    """
    G_new = eval(method)(G,(n+m))
    E1 = G_new[n:(n+m),0:n]
    E2 = G_new[n:(n+m), n:(n+m)]
    E11 = G_new[0:n, 0:n] - A1
    _, s1, _ = np.linalg.svd(E1)
    _, s2, _ = np.linalg.svd(E2)
    _, s3, _ = np.linalg.svd(E11)
    result = -np.diff(s)[0] - 2*s1[0] - s2[0] - s3[0]
    return result


def search(G, n, m, method):
    """
    Search for the threshold m_0 for retraining.
    """
    """
    Arg types:
        * **G** *(numpy array)* - The adjacent matrix of graph being embedded.
        * **n** *(int)* - The initially arrived node size.
        * **m** *(int)* - The new arrived node size.
        * **method** *(str)* -The function name of constructing target method.
    """
    A1 = eval(method)(G, n)
    _, s, _ = np.linalg.svd(A1)
    step = 100
    r = condition(A1, G, n, m, s, method)
    while True:
        while r>0:
            m += step
            r = condition(A1, G, n, m, s, method)
        m -= step
        r = condition(A1, G, n, m, s, method)
        if (r>0 and step == 1):
            return m
        step /= 10
        step = int(step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, required=True,
            help=".npy or .pckl input graph file path")
    parser.add_argument("--begin", type=int, required=True,
            help="The first begin size of nodes")
    parser.add_argument("--end", type=int, required=True,
            help="The last begin size of nodes")
    parser.add_argument("--method", type=str, required=True,
            help="The method used to construct adjacent matrix")
    parser.add_argument("--output", type=str, required=True,
            help="Threshold output file path")
    args = parser.parse_args()
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    method = args.method
    ###loading graph###
    logger.info("Loading graph from %s...", args.graph)
    try:
        G = np.load(args.graph)
    except:
        f = open(args.graph, 'rb')
        G = pickle.load(f)
        f.close()
    logger.info("Graph Loaded!")

    begin = int(args.begin)
    end = int(args.end)
    num = int(((end - begin) / 100) + 1)

    threshold = np.zeros([num,2])

    for (i,n) in enumerate(np.linspace(begin, end, num)):
        n = int(n)
        threshold[i,0] = n
        A1 = eval(method)(G, n)
        u, s, v = np.linalg.svd((A1))
        if (condition(A1, G, n, 1, s, method)<0):
            print('n = '+ str(n) + ', m_0 = 0')
            threshold[i,1] = 0
        else:
            m_0 = search(G, n, 1, method)
            print('n = '+ str(n) + ', m_0 = '+ str(m_0))
            threshold[i,1] = m_0

    np.save(args.output, threshold)