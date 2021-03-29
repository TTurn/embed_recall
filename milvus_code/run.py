from milvus import Milvus, IndexType, MetricType, Status
#连续服务器
milvus = Milvus(host='localhost',port='19530')
#创建集合
param = {'collection_name':'test01', 'dimension':256, 'index_file_size':1024, 'metric_type':MetricType.L2}
#milvus.create_collection(param)
#删除集合
milvus.drop_collection(collection_name="test01")
print("collection:",milvus.list_collections())
milvus.create_collection(param)
#print(milvus.list_partitions("test01"))
#创建分区
milvus.create_partition('test01', 'tag01')
#print(milvus.list_partitions("test01"))
#删除分区
#milvus.drop_partition('test01','tag01')
print("partition:",milvus.list_partitions("test01"))

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

milvus.insert(collection_name='test01', records=vectors, ids=vector_ids)


ivf_param = {'nlist': 16384}
milvus.create_index('test01', IndexType.IVF_FLAT, ivf_param)

search_param = {'nprobe': 16}
q_records = vectors[:1]
t = time.time()
result = milvus.search(collection_name='test01', query_records=q_records, top_k=10, params=search_param)[1]
print(result)
print(time.time()-t)

t = time.time()
result = milvus.search(collection_name='test01', query_records=q_records, top_k=10, params=search_param)[1]
print(result)
print(time.time()-t)
