"""
使用infomap 进行无监督文本聚类
"""
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import infomap

from vectorizer import convert2vecs


min_sim = 0.8  # 最小相似度


def build_links(vecs, min_sim=min_sim):
    vecs /= (vecs**2).sum(axis=1, keepdims=True)**0.5  # normalization
    links = {}
    for i in tqdm(range(vecs.shape[0])):
        sims = np.dot(vecs, vecs[i])
        idxs = sims.argsort()[::-1][1:]
        for j in idxs[:200]:
            if sims[j]>min_sim:
                links[(i,j)] = float(sims[j])
            else:
                break
    return links

def cluster_links(links):
    infomapWrapper = infomap.Infomap('--two-level --directed')
    for (i, j), sim in links.items():
        _ = infomapWrapper.addLink(i, j, sim)
        
    infomapWrapper.run()
    # 收集结果
    cluster2docIds = defaultdict(list)
    for node in infomapWrapper.nodes:
        cluster2docIds[node.module_id].append(node.node_id)
    return cluster2docIds


def infomap_cluster(data, keep_outlier=True):
    """
    data: 
        文本列表，e.g: [str1, str2, ...]
    keep_outlier: 
        是否保留孤立点，由于存在某些文本与任意其他文本之间的相似度都小于阈值，所以会存在孤立点，设为True 时，每个孤立点作为单独一类添加在末尾
    """
    vecs = convert2vecs(data)
    links = build_links(vecs)
    cluster2docIds = cluster_links(links)
    docIds2cluster = {}
    for cluster_id, docIds in cluster2docIds.items():
        for docId in docIds:
            docIds2cluster[docId] = cluster_id
    
    if keep_outlier:
        max_cluster_id = max(list(cluster2docIds.keys()))
        for i in range(len(data)):
            if i not in docIds2cluster:
                max_cluster_id += 1
                docIds2cluster[i] = max_cluster_id

    # 恢复顺序
    docIds2cluster = sorted(docIds2cluster.items(), key=lambda x: x[0])
    doc2cluster = [[data[d_id], c_id] for d_id, c_id in docIds2cluster]
    return doc2cluster 


if __name__ == '__main__':
    data = ['早上好', '你好啊', 'hello', '奥运健儿加油', '冬奥会真好看', 'good morning']
    print(infomap_cluster(data, True))