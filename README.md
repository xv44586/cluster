# 中文无监督文本聚类
- 不需要人为参与的真·无监督文本聚类

## 说明
主要思路为文本向量化 + 聚类，通过文本向量化，获取两两文本之间相似度，然后根据相似度进行聚类。

## 文本向量化
文本向量化使用的是目前中文句向量效果最好的方案：<a href='https://github.com/ZhuiyiTechnology/roformer-sim'>chinese-roformer-sim</a>

### 下载地址
- [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip)


## 聚类
聚类使用InfoMap 进行无监督聚类，参考：<a href='https://kexue.fm/archives/7006'>最小熵原理（五）：“层层递进”之社区发现与聚类</a>

## 使用方法
- 下载对应的模型文件后修改`vectorizer.py`  中的配置目录；
-  函数`informap_cluster` 方法中的`threshold` 是只保留两个边的最小阈值，主要作用是减少图中的边，减少计算量
```python
from cluster import infomap_cluster


data = ['早上好', '你好啊', 'hello', '奥运健儿加油', '冬奥会真好看', 'good morning']
print(infomap_cluster(data, keep_outlier=True, threshold=0.9))
"""
[['早上好', 2], ['你好啊', 1], ['hello', 1], ['奥运健儿加油', 3], ['冬奥会真好看', 4], ['good morning', 2]]
"""
```


