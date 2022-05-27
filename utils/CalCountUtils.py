# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:51:21 2021

@author: 
"""
from collections import Counter 
from tqdm import tqdm 
import pandas as pd 
import os 

# 本函数的目的是计算词频,从而更新停用词
# 核心函数是filter_min_max,看想要采用怎样的手段获得新的停用词?
# CalCountUtilis最开始调用get_word_count(text_type, min_cnt, max_freq)

    # 输入数据
        # results/content的根目录下
    # 输出数据
        # results/content/word_count里的word_count.csv是词频数统计
        # results/content/word_count里的content_max_frequence.txt content_min_count.txt
    # 调用merge_word_count(word_count, cur_count),将每日的各条新闻的单词频数统计合并
    # 调用get_word_count_df(word_count) ,将词频进行排序,得到一个dataframe
    # 调用add_new_stopwords_by_word_frequence(text_type, words_path, min_cnt, max_freq),更新停用词表
        # 调用load_word_count读入词频统计结构形成一个词典word_count
        # 调用filter_min_max,对频率过高或者频数过小的展开过滤
            # 调用get_word_freq(word_count),得到词频
            # 数据写出:存储到content_min_count和content_max_freq.txt
        


path = os.path.dirname(__file__) 
path = os.path.dirname(path) # python-主程序

# =============================================================================
def merge_word_count(word_count, cur_count):
    ''' 用来合并各条新闻的Counter,有助于把所有新闻的Word Counter加到一起'''
    
    for k,v in cur_count.items():
        word_count[k] = word_count.get(k,0) + v 
    return word_count 

# =============================================================================
def get_word_count_df(word_count):
    ''' 对word_count进行排序，并且生成一个dataframe'''
    word_cnt_lst = sorted([(k,v) for k,v in word_count.items()], key=lambda x: (-x[1], x[0])) 
    df = pd.DataFrame(data=word_cnt_lst, columns=["WORD", "COUNT"]) 
    return df 

# =============================================================================
def get_word_count(text_type, min_cnt, max_freq):
    '''
    统计所有新闻的词频 get_word_count(text_type, min_cnt, max_freq)
    min_cnt代表代表最小频数
    max_freq代表最大频率
    '''

    words_path = os.path.join(path, "results/{}/".format(text_type)) 

    word_count = {} 
    for file in tqdm(os.listdir(words_path)): 
        if file.split(".")[-1] != "csv": 
            continue 

        file_path = os.path.join(words_path, file) 
        df = pd.read_csv(file_path) 

        for obj_id, words in df[["OBJECT_ID", "WORDS"]].values:          
            if not words or words != words:
                continue 
            word_list = words.strip().split(" ")
            if not word_list:
                continue 
            # 调用merge_word_count统计所有天所有新闻总词频
            # word_count是词典,Counter是一条新闻初步统计结果
            word_count = merge_word_count(word_count, Counter(word_list)) 

    # print("Saving word count in all docs...")
    
    # 对word_count进行排序，并且生成一个dataframe
    word_df = get_word_count_df(word_count) 
    
    # 存储这个word_df到csv
    if not os.path.exists(os.path.join(words_path, "word_count")):
        os.mkdir(os.path.join(words_path, "word_count"))      
    
    save_path = os.path.join(words_path, "word_count", "word_count.csv")
    word_df.to_csv(save_path, index=False, header=True, encoding='utf_8_sig') 

    # with open(os.path.join(words_path, "word_list.txt"), "w") as f:
    #     for word in word_df["WORD"].values:
    #         f.write(word+"\n")
    # print("Saved.")
    add_new_stopwords_by_word_frequence(text_type, words_path, min_cnt, max_freq)

# =============================================================================

def load_word_count(df_path): 
    '''
    读入词频统计的dataframe,并返回一个词典
    '''
    df = pd.read_csv(df_path) 
    print("get word count...")
    word_count = {} 
    for word, cnt in tqdm(df.values, desc="word count"):
        word_count[word] = cnt 
    return word_count 

# =============================================================================

def get_word_freq(word_count): 
    '''
    根据输入的词典(词的频数统计),得到词的频率统计word_freq
    '''
    print("get word freq...")
    tot = 0 
    #先得到总共有多少次
    for v in word_count.values():
        tot += v 

    word_freq = {} 
    for word, cnt in tqdm(word_count.items(), desc="word freq"):
        word_freq[word] = cnt / tot 
    return word_freq 

# =============================================================================

def filter_min_max(word_count, min_cnt, max_freq, words_path): 
    
    ''' 
    根据word_count筛选出词频高(高于max_freq)或出现频次过小(低于min_cnt)的词 
    存储到content_min_count和content_max_freq.txt
    '''
    
    #得到词的频率统计
    word_freq = get_word_freq(word_count)
    
    min_words = [] 
    for word, cnt in word_count.items(): 
        if cnt < min_cnt: 
            min_words.append((word, cnt)) 
    
    max_words = [] 
    for word, freq in word_freq.items(): 
        if freq > max_freq: 
            max_words.append(word) 

        
    # 存储过滤的停用词表(存储之前做一个简单的排序)
    min_words = sorted(min_words, key=lambda x: (-x[1], x[0])) 
    with open(os.path.join(words_path, "word_count", "content_min_count.txt"), "w") as f: 
        for word, _ in min_words:
            f.write(word+"\n") 
    with open(os.path.join(words_path, "word_count", "content_max_frequence.txt"), "w") as f: 
        for word in max_words: 
            f.write(word+"\n") 
    print("Saved.") 

# =============================================================================

# if __name__ == '__main__':
def add_new_stopwords_by_word_frequence(text_type, words_path, min_cnt, max_freq):    
    
    ''' 
    更新停用词表
    其中words_path是存放分词结果的地方
    '''
    #读入词频统计的dataframe,并返回一个词典
    word_count = load_word_count(os.path.join(path, "results/{}/word_count/word_count.csv".format(text_type)))
    filter_min_max(word_count, min_cnt, max_freq, words_path)







