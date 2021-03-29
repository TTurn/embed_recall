from milvus import Milvus, IndexType, MetricType, Status
#连续服务器
milvus = Milvus(host='localhost',port='19530')
import time
import random
import numpy as np
#vectors = [[random.random() for _ in range(256)] for _ in range(3)]
#print(np.shape(np.array(vectors)))
#vector_ids = [1,2,3]
vectors = []
vector_ids = []
id_query = {}
with open("embed.txt","r")as f:
    for line in f:
        line_lst = line.strip().split("\t")
        vectors.append(list(map(float,(line_lst[2].split()))))
        vector_ids.append(int(line_lst[1]))
        id_query[int(line_lst[1])] = line_lst[0]

ivf_param = {'nlist': 16384}
milvus.create_index('test01', IndexType.IVF_FLAT, ivf_param)

search_param = {'nprobe': 16}
q_records = vectors[:1]
t = time.time()
result = milvus.search(collection_name='test01', query_records=q_records, top_k=50, params=search_param)[1]
for i,item in enumerate(result):
    print(id_query[i])
    for recall in item:
        print(id_query[recall.id]) 
    print("---------------------")  
#print(result)
print(time.time()-t)
