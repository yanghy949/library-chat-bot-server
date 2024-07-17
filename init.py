import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openpyxl

current_file_path = os.path.abspath(__file__)
path = os.path.dirname(current_file_path)
print("#####执行重构#####")
model = SentenceTransformer(f'{path}/m3e-base')

workbook = openpyxl.load_workbook(f'{path}/file.xlsx')
sheet = workbook["Sheet1"]
txt = ''
for i in sheet.iter_rows(values_only=True):
    if i[0] and i[1]:
        txt += "##\t" + str(i[0]) + "\n\t" + str(i[1]) + "\n\n"
text = txt.split("\n\n")
for i in text:
    if i == "":
        text.remove(i)
np.save('splited.npy',text)
encode=model.encode(text)
index =faiss.IndexFlatL2(encode.shape[1])
index.add(encode)

with open('text','w',encoding='utf-8')as f:
    f.write(txt)
np.savetxt('knowledge',encode)
faiss.write_index(index, 'index')
