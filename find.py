import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

current_file_path = os.path.abspath(__file__)
path = os.path.dirname(current_file_path)

model= SentenceTransformer(f'{path}/m3e-base')

# base= np.loadtxt('knowledge')
txt = np.load(f'{path}/splited.npy')
# with open('text','r',encoding='utf-8') as f:
#     txts=f.read()
#print("\n".join(txt))
index = faiss.read_index(f'{path}/index')
def find(query):
    em_query=model.encode([query])
    D,I=index.search(em_query,k=3)
    #print(D,I)
    res= []
    ans=[]
    # for i in I[0]:
    #     ans.append(txt[i])
    #     #print(ans)
    # for i in txt:
    #     for j in ans:
    #         if j in i:
    #             res.append(i)
    n=0
    for i in I[0]:
        if D[0][n] <180:
            res.append(txt[i])
        n+=1        
    # arr=list(set(res))
    #print(res)
    return res

