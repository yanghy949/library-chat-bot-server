import numpy as np
from sentence_transformers import SentenceTransformer
import jieba
import faiss

print('############构建索引#############')
model = SentenceTransformer('m3e-base')
with open ('text','r',encoding='utf-8') as f:
    txt = f.read()
txt = jieba.lcut(txt)
#print(txt)
np.save('splited.npy',txt)
encode=model.encode(txt)
index =faiss.IndexFlatL2(encode.shape[1])
index.add(encode)

np.savetxt('knowledge',encode)
faiss.write_index(index, 'index')