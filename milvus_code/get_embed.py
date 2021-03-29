# -*- coding: utf-8 -*-
import json
import requests
import time
import numpy as np
# 本地测试
FLASK_HOST = 'localhost'
FLASK_PORT = 18182
api_name = 'sentence_server'
requrl = f'http://{FLASK_HOST}:{FLASK_PORT}/{api_name}'
#requrl = f'http://nlu.kuaimai.com/sentence_server'


'''
线上环境：
    http://nlu.kuaimai.com/sentence_server

测试环境：
    http://nlu-test.kuaimai.com/sentence_server
'''

def test_api_single(sentence_pair):
    request_data = {
            'api_type': 'single',
            'sentences': sentence_pair,
        }
    r = requests.post(requrl, json=request_data)

    result_str = r.text
#    print(f'Result of test_api_single of {requrl} is: ', result_str)
    return json.loads(result_str)["sentence_vectors"]

if __name__ == '__main__':
    idd = 0
    with open("dev.csv","r")as f,open("embed.txt","w")as fw:
        for line in f:
            line_lst = line.strip().split(",")
            if len(line_lst) != 3:continue
            query1 = line_lst[0]
            query2 = line_lst[1]
            embeds = test_api_single([query1,query2])
            for i,embed in enumerate(embeds):
                fw.write(line_lst[i]+"\t"+str(idd+i)+"\t"+" ".join(map(str,embed))+"\n")         
            idd+=2
