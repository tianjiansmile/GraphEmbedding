import numpy as np



from ge.classify import read_node_label,Classifier

from ge import Struc2Vec

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE



def evaluate_embeddings(embeddings):

    # X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)

    X, Y = read_node_label('../data/my/45456803_label.txt')

    tr_frac = 0.8

    print("Training classifier using {:.2f}% nodes...".format(

        tr_frac * 100))

    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    clf.split_train_evaluate(X, Y, tr_frac)





def plot_embeddings(embeddings,comm,page):

    # X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)

    X, Y = read_node_label('../data/'+page+'/'+comm+'_label.txt')

    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)



    color_idx = {}

    for i in range(len(X)):

        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)



    for c, idx in color_idx.items():

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()

    plt.show()

# 通过word2vec生成节点特征
def word_to_vec(model,emd):
    model.wv.save_word2vec_format(emd+'word_vec.txt', binary=False)

if __name__ == "__main__":
    # G = nx.read_edgelist('../data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
    #                      data=[('weight', int)])

    comm = '2334530'
    page = 'user_network'
    G = nx.read_edgelist('../data/'+page+'/'+comm+'edgelist.txt',
                         create_using=nx.Graph(), nodetype=None,
                         data=[('type', str),('call_times', str),('call_len', str),('type_real', str)])

    model = Struc2Vec(G, 10, 80, workers=1, verbose=40, )
    model.train()
    embeddings = model.get_embeddings()
    emb = 'data/'+comm
    # 保存嵌入向量
    word_to_vec(model.w2v_model, emb)

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings,comm,page)