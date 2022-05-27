# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:34:22 2021

@author: 
"""
import os

def merge_stopwords(stopwords_path, stopwords_list=None): 
    # 合并不同来源的停用词 
    
    stopwords_set = set() 
    if not stopwords_list: 
        # 如果没有指定停用词列表txt，就直接从文件夹里合并停用词表
        stopwords_list = [f for f in os.listdir(stopwords_path) if f[-3:]=="txt"]
        
    for stopwords in stopwords_list:
        with open(os.path.join(stopwords_path, stopwords), "r", encoding='UTF-8') as f:
            for line in f.readlines():
                stopwords_set.add(line.strip()) #  删除空格
                
    with open("stopwords/stopwords.txt", "w") as f: 
        for word in stopwords_set:
            f.write(word+"\n") 
    return 



merge_stopwords("stopwords/our_stopwords")