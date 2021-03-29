# 导入库  
import numpy as np  
import faiss  
import time
# 向量维度  
vec_dim = 256

vectors = []
vector_ids = []
id_query = {}
with open("embed.txt","r")as f:
    for line in f:
        line_lst = line.strip().split("\t")
        vectors.append(list(map(float,(line_lst[2].split()))))
        vector_ids.append(int(line_lst[1]))
        id_query[int(line_lst[1])] = line_lst[0]

vectors = np.array(vectors)
vectors = vectors.astype("float32")
# 创建索引  

quantizer = faiss.IndexFlatL2(vec_dim)  # 使用欧式距离作为度量  
nlist = 16384
faiss_index = faiss.IndexIVFFlat(quantizer, vec_dim, nlist, faiss.METRIC_L2)
faiss_index.nprobe = 16

#ssert not index.is_trained
faiss_index.train(vectors)
faiss_index.add(vectors) 

faiss.write_index(faiss_index,"large.index")
# 查询向量 假设有5个
query_vectors = vectors[:20]
# 搜索结果
# 分别是 每条记录对应topk的距离和索引
# ndarray类型 。shape：len(query_vectors)*topk

res_distance, res_index = faiss_index.search(query_vectors, 5)
t = time.time()
res_distance, res_index = faiss_index.search(query_vectors, 20)
for i,index in enumerate(res_index):
    print(id_query[i])
    print("===")
    for recall in index:
        print(id_query[recall])  
    print("---------------------")
print(time.time()-t)

faiss_index = faiss.read_index("large.index")
res_distance, res_index = faiss_index.search(query_vectors, 5)
t = time.time()
res_distance, res_index = faiss_index.search(query_vectors, 20)
for i,index in enumerate(res_index):
    print(id_query[i])
    print("===")
    for recall in index:
        print(id_query[recall])
    print("---------------------")
print(time.time()-t)
