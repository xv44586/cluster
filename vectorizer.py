"""
使用chinese_roformer_sim 模型获取文本的向量表示
下载地址：https://github.com/ZhuiyiTechnology/roformer-sim
"""
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from tqdm import tqdm

# 模型配置
config_path = '/data2/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data2/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data2/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'
maxlen = 128  # 文本最大长度

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True) 

# 建立加载模型
encoder = build_transformer_model(
    config_path, checkpoint_path, model='roformer', with_pool='linear'
)

def convert2vecs(data):
    '''句子向量化'''
    vecs = []
    token_ids, segment_ids = [],[]
    for doc in tqdm(data):
        t_ids, s_ids = tokenizer.encode(doc, maxlen=maxlen)
        token_ids.append(t_ids)
        segment_ids.append(s_ids)
    
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    vecs = encoder.predict([token_ids, segment_ids], verbose=True)

    return vecs
    
