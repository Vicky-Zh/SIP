import copy
import scipy.sparse as sparse
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T
import math
import scipy.io
import numpy as np
import argparse
import networkx as nx
import scipy.io
import time
import argparse
import logging
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from GraRep import GraRep

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)


####################################################
###Functions to create embedding based on NetMF#####
####################################################
def netmf(A, dim):
    """
    Create elements to construct NetMF embedding.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of the graph.
        * **dim** *(int)* - The dimension of node embedding.
    """
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
    window = 10
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    sum_eval = np.maximum(evals, 0)
    b = 1
    X = sparse.diags(np.sqrt(sum_eval)).dot(D_rt_invU.T).T * math.sqrt(vol/b)#这里np.sqrt()的目的是为了
    m = T.matrix()
    mmT = T.dot(m, m.T) 
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    deepwalk_matrix = f(X.astype(theano.config.floatX))
    u, s, v = sparse.linalg.svds(deepwalk_matrix, dim, return_singular_vectors=True)
    return u,s,v,evecs,evals,sum_eval, d_rt, deepwalk_matrix, X

def SIP_NetMF_embedding(A, A1, dim=32):
    """
    Create embedding for first n nodes using netmf and for new arrived m nodes using SIP-NetMF.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    vol = float(A.sum())
    u1,s1,v1,evecs1,evals1,sum_eval, d_rt, deepwalk_matrix,X = netmf(A1, dim)
    embedding_origin = (np.sqrt(np.diag(s1))).dot(u1.T).T
    embedding = np.copy(embedding_origin) 
    logger.info('Original embeddings are get.')

    n = A.shape[0]
    n1 = A1.shape[0]
    L, d_rt1 = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    u1 = csr_matrix(u1)
    v1 = csr_matrix(v1)
    evecs1 = csr_matrix(evecs1)

    time_begin = time.time()
    b = 1
    evals1[evals1 < 0.01] = 1
    u_simu = X[n1:n,0:n1].dot(evecs1).dot(sparse.diags(evals1 ** (-1)))
    temp = sparse.diags(d_rt1[n1:n]**-1).dot(u_simu)
    X = temp.dot(sparse.diags(sum_eval)).dot(evecs1.T) * (vol/b)
    X = X.dot(sparse.diags(d_rt1[0:n1]**-1))

    X = X.todense()
    m = T.matrix()
    mmT = m
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    deepwalk_matrix = f(X.astype(theano.config.floatX))

    deepwalk_matrix = csr_matrix(deepwalk_matrix)
    project_embed_all = deepwalk_matrix.dot(v1.T)
    s1[s1 < 0.01] = 1
    simu_embed_all = project_embed_all.dot(sparse.diags(s1 ** (-1/2)))
    time_end = time.time()

    simu_embed_all = simu_embed_all.todense()
    embedding_all = np.vstack((embedding, simu_embed_all))
    return embedding_origin, embedding_all, (time_end - time_begin)

def NetMF_embedding(A,A1,dim = 32):
    """
    Create embedding for first n nodes using netmf and new arrived m nodes using retraining.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    u1,s1,_,_,_,_, _, _, _ = netmf(A1, dim)
    embedding = np.diag(np.sqrt(s1)).dot(u1.T).T
    time_begin = time.time()
    u1,s1,_,_,_,_, _, _, _ = netmf(A, dim)
    embedding_all = np.diag(np.sqrt(s1)).dot(u1.T).T
    time_end = time.time()
    return embedding, embedding_all, (time_end - time_begin)

def NetMF_change_embedding(A,A1,dim = 32):
    """
    Create embedding for all (n+m) nodes using netmf.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    n = A1.shape[0]
    time_begin = time.time()
    u1,s1,_,_,_,_, _, _, _ = netmf(A, dim)
    embedding_all = np.diag(np.sqrt(s1)).dot(u1.T).T
    time_end = time.time()
    return embedding_all[0:n,:], embedding_all, (time_end - time_begin)
####################################################
####################################################
####################################################



####################################################
#Functions to create embedding based on Laplace Eigenmaps#
####################################################
def SIP_LE_embedding(A, A1, dim=32):
    """
    Create embedding for first n nodes using LE and for new arrived m nodes using SIP-LE.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    L_tilde, _ = csgraph.laplacian(A1, normed=True, return_diag=True)
    d_rt, embedding_origin = sparse.linalg.eigsh(
        L_tilde,
        k=dim,
        which="SM",
        tol = 1E-6,
        return_eigenvectors=True
    )
    
    d_rt[d_rt < 0.1] = 1
   
    embedding = sparse.csc_matrix(embedding_origin)

    n = A.shape[0]
    n1 = A1.shape[0]

    L_tilde, _ = csgraph.laplacian(A, normed=True, return_diag=True)
    L_tilde = sparse.csc_matrix(L_tilde)

    time_begin = time.time()
    simu_embed_all = L_tilde[n1:n,0:n1].dot(embedding).dot(sparse.diags(d_rt ** -1))
    time_end = time.time()

    embedding_all = np.vstack((embedding_origin, simu_embed_all.todense()))
    return embedding_origin, embedding_all, (time_end - time_begin)

def LE_embedding(A,A1,dim = 32):
    """
    Create embedding for first n nodes using LE and new arrived m nodes using retraining.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    from karateclub import LaplacianEigenmaps
    G1 = nx.from_numpy_matrix(A1)
    G = nx.from_numpy_matrix(A)

    model = LaplacianEigenmaps(dimensions=dim)
    model.fit(G1)
    embedding = model.get_embedding()

    time_begin = time.time()
    model.fit(G)
    embedding_all = model.get_embedding()
    time_end = time.time()

    return embedding, embedding_all, (time_end - time_begin)

def LE_change_embedding(A,A1,dim = 32):
    """
    Create embedding for all (n+m) nodes using LE.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    from karateclub import LaplacianEigenmaps
    n = A1.shape[0]
    G = nx.from_numpy_matrix(A)

    model = LaplacianEigenmaps(dimensions=dim)

    time_begin = time.time()
    model.fit(G)
    embedding_all = model.get_embedding()
    time_end = time.time()

    return embedding_all[0:n,:], embedding_all, (time_end - time_begin)
####################################################
####################################################
####################################################


####################################################
###Functions to create embedding based on GraRep####
####################################################
def create_target_matrix(A_tilde, A_hat):
    """
    Create the target matrix of GraRep for further decomposition.
    """
    """
    Arg types:
        * **A_tilde** *(Numpy array)* - The matrix constructed before.
        * **A_hat** *(Numpy array)* - D^{-1}A, where D is the diagonal degree matrix and A is the adjacent matrix.
    """
    A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
    scores = np.log(A_tilde.data) - math.log(A_tilde.shape[0])
    rows = A_tilde.row[scores < 0]
    cols = A_tilde.col[scores < 0]
    scores = scores[scores < 0]
    target_matrix = sparse.coo_matrix(
        (scores, (rows, cols)), shape=A_tilde.shape, dtype=np.float32
    )
    return A_tilde, sparse.csc_matrix(target_matrix)

def SIP_GraRep_embedding(A, A1,dim = 32, order = 4, dimensions = 8):
    """
    Create embedding for first n nodes using GraRep and for m new arrived nodes using SIP-GraRep.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of the node embedding.
        * **order** *(int)* - The parameter representing the number of matrices to be decomposed in GraRep.
        * **dimensions** *(int)* - The dimension of a single embedding.
    """
    n = A1.shape[0]
    n1 = A.shape[0]
    G1 = nx.from_numpy_matrix(A1)

    model = GraRep(dimensions = dimensions, order = order)
    model.fit(G1)
    embedding = model.get_embedding()
    basis = model._basis

    _, D = csgraph.laplacian(A, normed=False, return_diag=True)
    M_origin = sparse.diags(D ** -1).dot(A)
    M = copy.deepcopy(M_origin)
    simu_embed = sparse.csc_matrix(((n1-n),dim))

    time_begin = time.time()
    for i in range(order):
        M, S = create_target_matrix(M, M_origin)
        simu_embed[:,(i*dimensions):(i+1)*dimensions] = S[n:n1,0:n].dot(basis[i])
        #M = M.dot(M_origin)    
    time_end = time.time()
    print(embedding.shape)
    embedding_all = np.vstack((embedding, simu_embed.todense()))
    return embedding, embedding_all, (time_end - time_begin)

def GraRep_embedding(A,A1,dim = 32):
    """
    Create embedding for first n nodes using GraRep and new arrived m nodes using retraining.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    G1 = nx.from_numpy_matrix(A1)
    G = nx.from_numpy_matrix(A)

    model = GraRep(dimensions=8, order = 4)
    model.fit(G1)
    embedding = model.get_embedding()

    time_begin = time.time()
    model.fit(G)
    embedding_all = model.get_embedding()
    time_end = time.time()

    return embedding, embedding_all, (time_end - time_begin)

def GraRep_change_embedding(A,A1,dim = 32):
    """
    Create embedding for all (n+m) nodes using GraRep.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    n = A1.shape[0]
    G = nx.from_numpy_matrix(A)

    model = GraRep(dimensions=8, order = 4)

    time_begin = time.time()
    model.fit(G)
    embedding_all = model.get_embedding()
    time_end = time.time()

    return embedding_all[0:n,:], embedding_all, (time_end - time_begin)
####################################################
####################################################
####################################################


####################################################
###Functions to create embedding based on AROPE####
####################################################
def Eigen_Reweighting(X,order,coef):
    """
    Create the Reweighted matrix.
    """
    """
    Arg types:
        * **X** *(Numpy array)* - The original adjacent matrix.
        * **order** *(int)* - The number of order of X.
        * **coef** *(list)* - The weights for each ordered matrix.
    """
    if order == -1:     # infinity
        assert len(coef) == 1, 'Eigen_Reweighting wrong.'
        coef = coef[0]
        assert np.max(np.absolute(X)) * coef < 1, 'Decaying constant too large.'
        X_H = np.divide(X, 1 - coef * X)
    else:
        assert len(coef) == order, 'Eigen_Reweighting wrong.'
        X_H = coef[0] * X
        X_temp = X
        for i in range(1,order):
            X_temp = np.multiply(X_temp,X)
            X_H += coef[i] * X_temp
    return X_H

from scipy.sparse.linalg import eigs
def Eigen_TopL(A, d):
    L = d + 10
    lambd = np.array([0])
    while sum(lambd > 0) < d:         # can be improved to reduce redundant calculation if L <= 2d + 10 not hold
        L = L + d
        lambd, X = eigs(A, L)
        lambd, X = lambd.real, X.real
        # only select top-L
    temp_index = np.absolute(lambd).argsort()[::-1]
    lambd = lambd[temp_index]
    temp_max, = np.where(np.cumsum(lambd > 0) >= d)
    lambd, temp_index = lambd[:temp_max[0]+1], temp_index[:temp_max[0]+1]
    X = X[:,temp_index]
    return lambd, X


def Shift_Embedding(lambd, X, order, coef, d):
    # lambd, X: top-L eigen-decomposition 
    # order: a number indicating the order
    # coef: a vector of length order, indicating the weights for each order
    # d: preset embedding dimension
    # return: content/context embedding vectors
    lambd_H = Eigen_Reweighting(lambd,order,coef)             # High-order transform
    temp_index = np.absolute(lambd_H).argsort()[::-1]         # select top-d
    temp_index = temp_index[:d+1]
    lambd_H = lambd_H[temp_index]
    lambd_H_temp = np.sqrt(np.absolute(lambd_H))
    U = np.dot(X[:,temp_index], np.diag(lambd_H_temp))        # Calculate embedding
    V = np.dot(X[:,temp_index], np.diag(np.multiply(lambd_H_temp ** -1, np.sign(lambd_H))))
    return U, V

def AROPE_matrix(X, order = 3, coef= [1, 0.01, 0.0001]):
    X = sparse.csc_matrix(X)
    X_H = coef[0] * X
    for i in range(1,order):
        X_H += coef[i] * (X**(i+1))
    return X_H
    
def AROPE(A, d, order = [3], weights = [[1, 0.01, 0.0001]]):
    # A: adjacency matrix A or its variations, sparse scipy matrix
    # d: dimensionality 
    # r different high-order proximity:
        # order: 1 x r vector, order of the proximity
        # weights: 1 x r list, each containing the weights for one high-order proximity
    # return: 1 x r list, each containing the embedding vectors 
    A = A.asfptype()
    lambd, X = Eigen_TopL(A, d)
    r = len(order)
    U_output, V_output = [], []
    for i in range(r):
        U_temp, V_temp = Shift_Embedding(lambd, X, order[i], weights[i], d)
        U_output.append(U_temp)
        V_output.append(V_temp)
    return U_output[0],V_output[0]

def SIP_AROPE_embedding(A, A1, dim = 32):
    embedding, basis = AROPE(A1, dim)
    M = AROPE_matrix(A)
    #M = A
    n = A1.shape[0]
    n1 = A.shape[0]

    time_begin = time.time()
    embedding_simu = M[n:n1,0:n].dot(basis)
    time_end = time.time()

    embedding_all = np.vstack((embedding, embedding_simu))

    return embedding, embedding_all, (time_end - time_begin)

def AROPE_embedding(A, A1, dim = 32):
    embedding, _ = AROPE(A1, dim)
    
    time_begin = time.time()
    embedding_all, _ = AROPE(A, dim)
    time_end = time.time()
    return embedding, embedding_all, (time_end - time_begin)

def AROPE_change_embedding(A, A1, dim = 32):
    #embedding = AROPE(A1, dim)
    n = A1.shape[0]
    
    time_begin = time.time()
    embedding_all, _ = AROPE(A, dim)
    time_end = time.time()
    return embedding_all[0:n,:], embedding_all, (time_end - time_begin)




def construct_indicator(y_score, y):
    num_label = np.sum(y, axis=1, dtype=np.int64)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int64)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def predict_cv(X, y, newx, newy, C=1.):
    micro, macro = [], []
    micro_new, macro_new = [], []

    X_test_new, y_test_new = newx, newy
    n = X.shape[0]
    m = newx.shape[0]

    X_train, X_test = X[0:(n-m)], X[(n-m):n]
    y_train, y_test = y[0:(n-m)], y[(n-m):n]
    
    clf = OneVsRestClassifier(              
            LogisticRegression(
                C=C,
                solver="liblinear", 
                multi_class="ovr",
                max_iter=500),
            n_jobs=-1)
    clf.fit(X_train, y_train) 
    y_score = clf.predict_proba(X_test)
    y_pred = construct_indicator(y_score, y_test)
    mi = f1_score(y_test, y_pred, average="micro")
    ma = f1_score(y_test, y_pred, average="macro")
    micro.append(mi)
    macro.append(ma)
    y_score = clf.predict_proba(X_test_new)
    y_pred = construct_indicator(y_score, y_test_new)
    mi = f1_score(y_test_new, y_pred, average="micro")
    ma = f1_score(y_test_new, y_pred, average="macro")
    micro_new.append(mi)
    macro_new.append(ma)
    return np.mean(micro), np.mean(macro), np.mean(micro_new), np.mean(macro_new)

def predict_all(A_origin, n_begin, n_end, m_all, method, label, output):
    micro, macro, time_here = [],[],[]
    micro_new, macro_new = [], []
    i = 0
    AOriginD = A_origin.todense()
    for n in range(n_begin, n_end, 100):
        logger.info(n)
        n_new = n
        m = int(m_all[i])
        A_before = np.copy(AOriginD[0:n_new,0:n_new])
        A_before = sparse.lil_matrix(A_before)

        A_origin = np.copy(AOriginD[0:(n_new+m),0:(n_new+m)])
        A_origin = sparse.lil_matrix(A_origin)

        embedding, embedding_all, update_time = eval(method)(A_origin, A_before)
        
        X = embedding[0:n_new,:]
        y = label[0:n_new,:]
        newx = embedding_all[n_new:(n_new+m),:]
        newy = label[n_new:(n_new+m),:]

        mi,ma, mi_new, ma_new = predict_cv(X, y, newx, newy)
 
        print('mi = ', str(mi), ', mi_new = ', str(mi_new),'ma = ', str(ma), ', ma_new = ', str(ma_new))
        micro.append(mi)
        macro.append(ma)
        micro_new.append(mi_new)
        macro_new.append(ma_new)
        time_here.append(update_time)
        i = i+1
    return micro, macro, micro_new, macro_new, time_here



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True,
            help=".npy or .fckl input file path")
    parser.add_argument("--label", type=str, required=True,
            help=".npy input file path")
    parser.add_argument("--threshold", type=str, required=True,
            help=".csv input file path")
    parser.add_argument("--dim", default=32, type=int,
            help="dimension of embedding")
    parser.add_argument("--method", type=str, required=True,
            help ="the type of method to be used (choosing from NetMF, LE, AROPE, GraRep)")
    parser.add_argument("--output", type=str, required=True,
            help="accuracy output file path")
    args = parser.parse_args()
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s') 
    dim = args.dim

    m_all = np.loadtxt(args.threshold, delimiter=',')
    n_begin = int(m_all[0,0])
    n_end = int(m_all[-1,0])
    n_end += 100
    m_different_n = m_all[:,1]
    n_num = int((n_end - n_begin)/100)

    label = np.load(file = args.label)
    label_num = label.shape[1]                     
    logger.info("Label loaded!")
    try:
        A_origin = np.load(args.network,allow_pickle=True)
        A_origin = sparse.lil_matrix(A_origin)
    except:
        import pickle
        F=open(args.network,'rb')
        A_origin=pickle.load(F)
    logger.info("Graph loaded!")

    if args.method == 'LE':
        methods = ['SIP_LE', 'LE', 'LE_change']
    elif args.method == 'GraRep':
        methods = ['SIP_GraRep', 'GraRep', 'GraRep_change']
    elif args.method == 'AROPE':
        methods = ['SIP_AROPE', 'AROPE', 'AROPE_change']
    elif args.method == 'NetMF':
        methods = ['SIP_NetMF', 'NetMF', 'NetMF_change']

    micro = np.zeros([len(methods),n_num])
    macro = np.zeros([len(methods),n_num])
    micro_new = np.zeros([len(methods),n_num])
    macro_new = np.zeros([len(methods),n_num])
    time_all = np.zeros([len(methods),n_num])


    for (i, method) in enumerate(methods):
        print(method)
        func_name = method + '_embedding'
        mi, ma, mi_new, ma_new, up_time = predict_all(A_origin, n_begin, n_end, m_different_n, func_name, label, args.output)
        micro[i,:] = np.array(mi)
        macro[i,:] = np.array(ma)
        micro_new[i,:] = np.array(mi_new)
        macro_new[i,:] = np.array(ma_new)
        time_all[i,:] = np.array(up_time)
    np.savetxt(args.output + str(n_begin) + 'to'+ str(n_end) + '_microF1_grarep.csv', micro_new, delimiter=',')
    np.savetxt(args.output + str(n_begin) + 'to'+ str(n_end) + '_macroF1_grarep.csv', macro_new, delimiter=',')
    np.savetxt(args.output + str(n_begin) + 'to'+ str(n_end) + 'time_grarep.csv', time_all, delimiter = ',')