import math
import pandas as pd
from operator import ilshift
import time
import scipy.io
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import numpy as np
import argparse
import random
import logging
import theano
from theano import tensor as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import logging
from IPython.core.interactiveshell import InteractiveShell
from karateclub import LaplacianEigenmaps
InteractiveShell.ast_node_interactivity = "all"
logger = logging.getLogger(__name__)



name = 'LP-superuser'
file = 'data/LP-superuser/sx-superuser-washed.txt'

CM_origin = np.loadtxt(file, dtype=int, delimiter=' ')
date = CM_origin[:,2].tolist()

def create_adjacency(CM_origin, t0):
    ##
    ## Input: t0, the time (number of day) that the streaming starts.
    ## Output: adjacent matrix of nodes arrived in the first (t0-1) days.
    ##
    CM = np.copy(CM_origin)
    #CM[:,2] = (CM[:,2] - CM[0,2])/60/60/24
    l = CM[:,2].tolist()
    edge_num = l.index(t0)
    node = np.max(CM[0:edge_num, 0:2])
    print('node = '+ str(node))
    A = np.zeros([node, node])
    i = 0
    while(CM[i,2] < t0):
        A[CM[i,0]-1, CM[i,1]-1] = 1
        A[CM[i,1]-1, CM[i,0]-1] = 1
        i = i + 1
    A = sparse.lil_matrix(A)
    return A

def SIP_LE_embedding(A, A1, embedding_origin, d_rt, dim=32):
    """
    Create embedding for first n nodes using LE and for new arrived m nodes using SIP-LE.
    """
    """
    Arg types:
        * **A** *(Numpy array)* - The adjacent matrix of all (n+m) nodes.
        * **A1** *(Numpy array)* - The adjacent matrix of first n nodes.
        * **dim** *(int)* - The dimension of node embedding.
    """
    # L_tilde, _ = csgraph.laplacian(A1, normed=True, return_diag=True)
    # d_rt, embedding_origin = sparse.linalg.eigsh(
    #     L_tilde,
    #     k=dim,
    #     which="SM",
    #     tol = 1E-6,
    #     return_eigenvectors=True
    # )
    
    # d_rt[d_rt < 0.1] = 1
   
    embedding = sparse.csc_matrix(embedding_origin)

    n = A.shape[0]
    n1 = A1.shape[0]

    L_tilde, _ = csgraph.laplacian(A, normed=True, return_diag=True)
    L_tilde = sparse.csc_matrix(L_tilde)

    time_begin = time.time()
    simu_embed_all = L_tilde[n1:n,0:n1].dot(embedding).dot(sparse.diags(d_rt ** -1))
    time_end = time.time()

    embedding_all = np.vstack((embedding_origin, simu_embed_all.todense()))
    return embedding_all, (time_end - time_begin)

def calculate_update_embedding(A, A1, embedding_all, dim=32):
    #用这个函数来实现同时投影出embedding & 更新embedding！！
    # A: day = t + t_gap的邻接矩阵
    # A1: day = t的邻接矩阵

    embedding = np.copy(embedding_all) 

    n = A.shape[0]
    n1 = A1.shape[0]
    m = n-n1
    #simu_embed_all = np.zeros([m, dim])

    time_begin = time.time()
    A_D = A.todense()
    for i in range(m):
        connection = A_D[n1+i, 0:(n-m)]
        simu_embed = embedding_all[n1+i,:]

        neighbors = np.where(connection != 0)
        neighbors_num = neighbors[0].shape[0]
        if neighbors_num > 0 :
            #step = 1 - math.sqrt(1 - 1/neighbors_num)
            for j in neighbors[0]:
                neighbors = np.where(A_D[j, 0:(n-m)] != 0)
                neighbors_num = neighbors[0].shape[0]
                step = 1 - math.sqrt(1 - 1/neighbors_num)

                origin_embed = np.copy(embedding_all[j,:])
                update_embed = origin_embed - step * simu_embed
                embedding[j,:] = np.copy(update_embed)
    time_end = time.time()

    #embedding_all = np.vstack((embedding, simu_embed_all))
    return embedding, (time_end - time_begin)

def sample(CM_origin, t1, delta_t = 1):
    CM = np.copy(CM_origin)
    #CM[:,2] = (CM[:,2] - CM[0,2])/60/60/24
    l = CM[:,2].tolist()
    edge_begin = l.index(t1)
    edge_end = l.index(t1+delta_t)
    node = np.max(CM[0:edge_begin, 0:2])
    edges = np.copy(CM[edge_begin:edge_end,0:2])
    edges_positive = tuple(tuple([y for y in x]) for x in edges)

    edges_negative = []
    number = edge_end - edge_begin
    t = 0
    while (t < number):
        edge_posi = edges_positive[t]
        if (edge_posi[0] > node) or (edge_posi[1] > node):
            edges_positive = edges_positive[:t] + edges_positive[(t+1):]
            number -= 1
            continue
        else:
            edge = tuple(random.sample(range(1,node),2))
            if (edge in edges_positive) or (edge[::-1] in edges_positive) or (edge in edges_negative) or (edge[::-1] in edges_negative):
                continue
            else:
                edges_negative.append(edge)
                t = t+1
    return edges_positive, tuple(edges_negative)

def nodes_to_edge(embedding, edges, mode = 'average'):
    edge_embeds = np.zeros((len(edges), embedding.shape[1]))
    i = 0
    if mode == 'average':
        for edge in edges:
            edge_embeds[i,:] = (embedding[edge[0]-1, :] + embedding[edge[1]-1, :]) / 2
            i += 1  

    elif mode == 'hadamard':
        for edge in edges:
            try:
                edge_embeds[i,:] = embedding[edge[0]-1, :] * embedding[edge[1]-1, :]
                i += 1
            except:
                continue
    elif mode == 'weighted_l1':
        for edge in edges:
            try:
                edge_embeds[i,:] = np.abs(embedding[edge[0]-1, :] - embedding[edge[1]-1, :])
                i += 1
            except:
                continue
    elif mode == 'weighted_l2':
        for edge in edges:
            try:
                edge_embeds[i,:] = np.power(embedding[edge[0]-1, :] - embedding[edge[1]-1, :], 2)
                i += 1
            except:
                continue
    else:
        return ('Wrong Mode!')
    return edge_embeds


def predict_cv(X, y, train_ratio=0.7, n_splits=5, random_state=0, clf= LogisticRegression()):#这是最主要的一个函数
    F1_score, AUC_score = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)#这个函数定义了一种随机分训练集和测试集的机制，并取名为shuffle
    
    for train_index, test_index in shuffle.split(X):#把这个shuffle应用在X上
        try:
            #print(train_index.shape, test_index.shape)#输出某一次的训练集与测试集的个数
            assert len(set(train_index) & set(test_index)) == 0
            assert len(train_index) + len(test_index) == X.shape[0]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train) # 用刚才定义的分类器来训练我们目前的训练集
            y_pred = clf.predict(X_test)
            #y_score = clf.predict_proba(X_test)#输出的是一个数据矩阵，行=样本点个数，列=类别个数
            #y_pred = np.zeros_like(y_score, dtype=np.int64)#利用y_score构造一个它预测出来的类别矩阵，是0/1变量
            #y_pred[np.where(y_score > threshold)] = 1
            #y_pred[np.where(y_score <= threshold)] = 0
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            #logger.info("micro f1 %f macro f1 %f", mi, ma)
            F1_score.append(f1)
            AUC_score.append(auc)
        except:
            continue
    
    return np.mean(F1_score)*100, np.mean(AUC_score)*100, clf
    

def create_embedding_sample(embedding, t, delta_t = 1):
    #A = create_adjacency(CM_origin, t)
    #u,s,evecs,evals,sum_eval, d_rt, deepwalk_matrix, X = netmf(A, None, None)
    #embedding = (np.sqrt(np.diag(s))).dot(u.T).T   
    edges_positive, edges_negative = sample(CM_origin, t, delta_t)
    edges_positive_embedding = nodes_to_edge(embedding, edges_positive)
    edges_negative_embedding = nodes_to_edge(embedding, edges_negative)
    number = len(edges_positive_embedding)
    length = dim
    X = np.zeros([number*2, length])
    X[0:number,:] = edges_positive_embedding
    X[number : number*2,:] = edges_negative_embedding
    y = np.zeros(number * 2)
    y[0:number] = 1
    return X,y

def pred_over_time(t_begin, t_end, t_gap, C = 1.):
    F1_all, AUC_all = [],[]
    future_num = t_gap.shape[0]
    F1_future_all = np.zeros([t_end - t_begin, future_num])
    AUC_future_all = np.zeros([t_end - t_begin, future_num])
    time_all = np.zeros([t_end - t_begin, future_num])

    classifier =  LogisticRegression(C=C,solver="liblinear")

    t_num = 0
    for t in range(t_begin, t_end, 1):
        t = int(t)
        print(t)
        sign = 0
    
        try:
            A1 = create_adjacency(CM_origin, t)
        except:
            if sign==0:
                F1_all.append(0)
                AUC_all.append(0)
            t_num += 1
            continue

        L_tilde, _ = csgraph.laplacian(A1, normed=True, return_diag=True)
        d_rt, embedding_origin = sparse.linalg.eigsh(
            L_tilde,
            k=dim,
            which="LA",
            tol = 1E-6,
            return_eigenvectors=True
        )       
        d_rt[d_rt < 0.1] = 1
    

        #embedding_origin = (np.sqrt(np.diag(s1))).dot(u1.T).T
        X,y = create_embedding_sample(embedding_origin, t)
    
        F1, AUC, classifier = predict_cv(X, y, train_ratio=0.9, n_splits=10, random_state=0, clf = classifier)
        F1_all.append(F1)
        AUC_all.append(AUC)
        sign = 1
        for tG in range(future_num):
            A = create_adjacency(CM_origin, t + t_gap[tG])
            embedding_all, time_embed = SIP_LE_embedding(A, A1, embedding_origin, d_rt, dim=32)
            embedding_all, _ = calculate_update_embedding(A, A1, embedding_all, dim=32)

            X_future, y_future = create_embedding_sample(embedding_all, t + t_gap[tG])

            y_future_pred = classifier.predict(X_future)
            f1_future = f1_score(y_future, y_future_pred)
            auc_future = roc_auc_score(y_future, y_future_pred)

            F1_future_all[t_num, tG] = f1_future * 100
            AUC_future_all[t_num,tG] = auc_future * 100
            time_all[t_num, tG] = time_embed    
        
        t_num = t_num + 1

    return F1_all, AUC_all, F1_future_all, AUC_future_all, time_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tBegin", type=str, required=True,######################
            help="Begin time of the flow")
    parser.add_argument("--tEnd", type=str, required=True,######################
            help="End time of the flow")
    parser.add_argument("--tGapBegin", type=str, required=True,######################
            help="Gap time of prediction")
    parser.add_argument("--tGapEnd", type=str, required=True,######################
            help="Gap time of prediction")
    parser.add_argument("--tGapNum", type=str, required=True,######################
            help="Gap time of prediction")
    parser.add_argument("--dim", type=str, required=True,######################
            help="Dimension")
    args = parser.parse_args()
    t_begin = int(args.tBegin)
    t_end = int(args.tEnd)
    t_gap = np.linspace(int(args.tGapBegin), int(args.tGapEnd), int(args.tGapNum))
    #t_gap = int(args.tGap)
    dim = int(args.dim)

    F1_all, AUC_all, F1_future_all, AUC_future_all, time_all = pred_over_time(t_begin, t_end, t_gap)
    #print(F1_all)
    #print(AUC_all)
    #print(F1_future_all)

    t_num = 0
    for t in t_gap:
        data = pd.DataFrame({'F1':F1_all, 'AUC': AUC_all, 'F1_future': F1_future_all[:,t_num], 'AUC_future': AUC_future_all[:,t_num], 'time': time_all[:,t_num]})
        data.to_csv('data/'+str(name)+'/tGap_SIP-LE/accuarcy_tGap='+str(int(t))+'.csv')
        t_num = t_num + 1