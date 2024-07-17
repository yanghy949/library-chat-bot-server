import faiss
import openpyxl
# import numpy
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('moka-ai/m3e-base')
# 打开工作簿
workbook = openpyxl.load_workbook('file.xlsx')

# 选择工作簿中的工作表
sheet = workbook['Sheet1']
text = ''
# 读取并打印工作表中的所有单元格数据
for row in sheet.iter_rows(min_row=2, min_col=2, values_only=True):
    if row[0] and row[1]:
        text += "##\t" + str(row[0]) + "\n\t" + str(row[1]) + "\n\n"
print(text)
a = text.split('\n\n')
print(a)
af = model.encode(a)
index = faiss.IndexFlatL2(af.shape[1])
index.add(af)
qu = "图书馆"
em = model.encode([qu])
D, I = index.search(em, k=10)
rs = []
for i in I[0]:
    rs.append(a[i])
print(rs)
