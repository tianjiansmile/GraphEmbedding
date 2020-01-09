
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings,comm):
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    # 获得标签数据
    X, Y = read_node_label('../data/my/'+comm+'_label.txt')

    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    # 选逻辑回归分类器
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,comm,page):
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    X, Y = read_node_label('../data/'+page+'/'+comm+'_label.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    # 降维
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    comm = '2334530'
    page = 'user_network'
    G = nx.read_edgelist('../data/'+page+'/'+comm+'edgelist.txt',
                         create_using=nx.Graph(), nodetype=None,
                         data=[('type', str),('call_times', str),('call_len', str),('type_real', str)])

    # G = nx.karate_club_graph()

    # walk_length 游走步长， num_walks模拟轮数，workers 线程数
    model = DeepWalk(G, walk_length=20, num_walks=40, workers=1)
    model.train(window_size=5, iter=3)
    # 返回词向量字典，节点和嵌入向量
    embeddings = model.get_embeddings()

    # 评估嵌入向量
    evaluate_embeddings(embeddings,comm)
    plot_embeddings(embeddings,comm,page)
