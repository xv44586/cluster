# 中文无监督文本聚类
- 不需要人为参与的真·无监督文本聚类

## 说明
主要思路为文本向量化 + 聚类，通过文本向量化，获取两两文本之间相似度，然后根据相似度进行聚类。

## 文本向量化
文本向量化使用的是目前中文句向量效果最好的方案：<a herf='https://github.com/ZhuiyiTechnology/roformer-sim'>chinese-roformer-sim</a>

## 聚类
聚类使用InfoMap 进行无监督聚类，参考：<a href='https://kexue.fm/archives/7006'>最小熵原理（五）：“层层递进”之社区发现与聚类</a>

## 使用方法
```python
from cluster import infomap_cluster


data = ['早上好', '你好啊', 'hello', '奥运健儿加油', '冬奥会真好看']
print(infomap_cluster(data, True))
```
