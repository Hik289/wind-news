 # -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:11:13 2021

@author: 
"""
from pyhanlp import * #是HanLP的python封装，方便从Python中调用
import string 
import jieba
import re 
import os    
from tqdm import tqdm 
import pandas as pd 

path = os.path.dirname(__file__) 
path = os.path.dirname(path) # python-主程序


# SplitUtils函数，可以对新闻进行处理，包括新闻的初步处理（正则化、去除特殊符号、分词等，删除停用词，加入公司和行业代码等）
# split_words 进行分词
    # 分词: 调用 clean_and_split 包含正则化、去除特殊符号（链接等），用jieba分词等
        # 调用remove_special_marks 去除特殊符号（链接等）
    # 删去停用词: 调用 delete stopwords 
        # 调用filter stopwords
        # 调用get_wcode_and_icode获取行业代码和公司代码
        # 调用join_wcode_and_icode加入公司和行业代码
    #读入数据来源
        #data/ready文件夹下的数据
        #stopwords下的停用词
    #写出数据位置
        #results/content(title)下的分词结果（最终包含了行业代码）
        #data/codes下的codes暂存

# =============================================================================

def update_stopwords(text_type):
    stopwords_set = set() 
    #打开new_stopwords.txt
    try:
        with open(os.path.join(path, "data/stopwords/new_stopwords.txt"), "r",encoding = 'gbk') as f:
                for line in f.readlines():
                    stopwords_set.add(line.strip()) #  删除空格
    except:
        with open(os.path.join(path, "data/stopwords/new_stopwords.txt"), "r",encoding = 'utf-8') as f:
                for line in f.readlines():
                    stopwords_set.add(line.strip()) #  删除空格
                
    # 存放分词结果文件夹
    csv_path = os.path.join(path, "results/{}/".format(text_type))
    file_lst = sorted(os.listdir(csv_path))
    save_paths = os.path.join(path, "results/{}/".format(text_type))
    # 对一个一个文件进行处理
    for file in tqdm(file_lst, desc="刷新停用词"):
        if file.split(".")[-1] != "csv":
            continue
        # 读入相关文件
        file_path = os.path.join(csv_path, file) 
        df = pd.read_csv(file_path)
        # 过滤停用词
        for i in df.index:
            words = df.loc[i,'WORDS']
            ch_lst = filter_stopwords(words, stopwords_set=stopwords_set)
            ch_words = " ".join(ch_lst)
            if not ch_words:
                continue 
            df.loc[i,'WORDS'] = ch_words

        # 存储
        word_file = file
        df.to_csv(os.path.join(save_paths, word_file), header=True, index=False, encoding='utf_8_sig') 
        
# =============================================================================

def is_nan(text):
    '''判断文本是否为空'''
    return text != text or text == "nan" 

# =============================================================================

def remove_special_marks(text):
    '''
    remove special marks用来去除特殊符号，包含链接、百分比数字、价格数字、
    数字、html和javascript代码片段、其他噪音(参考以下来源)、标点符号等      
    '''
    
    # 去掉链接
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+\.com', '', text)

    # 去掉百分比的数字
    text = re.sub(r'(\d+(\.\d+)?)%', ' ', text)
    # 去掉价格数字
    text = re.sub(r'[$￥€£](\d+(\.\d+)?)', ' ', text) 
    # 去掉数字
    # text = re.sub(r'\b\d+(\.\d+)?\b', ' ', text) 
    text = re.sub(r'\d+(\.\d+)?', ' ', text) 
    # 去掉 html & javascript 代码片段
    text = re.sub(r'&lt;.+&gt;', " ", text) 
    
    # 清理其他噪音 
    text = re.sub(r"本文参考以下来源.+", " ", text) 
    text = re.sub(r'<br>', ' ', text) 
    text = re.sub(r'\u3000', ' ', text) 
    text = re.sub(r'&amp;', ' ', text)
    
    # 去掉英文里的标点符号
    text = "".join([ch if ch not in set(string.punctuation) else " " for ch in text ])
    
    # 去掉中文里的标点符号 
    text = re.sub(r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：∶·；《）《》“”()»〔〕-]+', " ", text)
    
    return text 

# =============================================================================

def clean_and_split(text):
    '''
    clean and split函数用来进行正则化、去除特殊字符、分词
      用dictionary.other.CharTable进行正则化
      调用remove special marks去除特殊符号
      用jieba库分词 
    '''
    
    # 利用pyhanlp库将文本字符正规化
    CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    # 包含多种分词器，如果hanlp后面跟tokenizer，可能包含IndexTokenizer/NLPTokenizer/BasicTokenizer
    # 也可以用dictiionary将字符正则化
    if is_nan(text):
        text = ""
    else:
        text = CharTable.convert(text)  
    
    # 去除文本中的特殊标点符号
    text = remove_special_marks(text) 
    
    
    # 用jieba库分词
    res = [] 
    for word in list(jieba.cut(text)):  
        if len(word) == 1:   # including " "
            continue 
        res.append(word)
        
    return res 

# =============================================================================

def split_words(text_type):
    '''
    主函数，进行分词
    输入:text_type, 可选的有content或title, 表示对内容或标题展开分析
    '''
    # 最开始的分词函数
    
    # 原始数据中列名称为大写英文
    col_name = text_type.upper()  

    # 原始数据文件夹
    csv_path = os.path.join(path, "data/ready")  
    
    # 存放分词结果文件夹
    save_paths = os.path.join(path, "results/{}/".format(text_type),"tempo") #修改之处  

    # 在result文件夹中，看哪些文件已经分过词
    processed_set = set() 
    for file in os.listdir(save_paths): 
        if file.split(".")[-1] == 'csv': 
            # 已经存好的有前缀"fenci_"
            processed_set.add(file[6:])  # processed_set是已经初步分词完成的文件
    
    # 在data/ready文件夹中，看有哪些文件待分词
    file_lst = sorted(os.listdir(csv_path))  #原始数据的list
    
    
    # 对一个一个文件进行处理
    for file in tqdm(file_lst, desc="初步分词"): 
        
        # 选择未分过词的部分,分过词的部分直接跳过
        if file.split(".")[-1] != "csv" or file in processed_set: 
            continue 
        
        # 没分过词的文件,获取文件路径并读入
        file_path = os.path.join(csv_path, file) 
        df = pd.read_csv(file_path) 
        
        # 截断部分太长的新闻，选取前50000
        df[col_name] = df[col_name].apply(lambda x: x[:50000] if type(x) is str else " ")   
        df[col_name] = df[col_name].apply(lambda x: x.lower() if type(x) is str else " ") 

        # 清洗文本并分词
        words_list = [] 
        for i, (idx, txt,w_code,i_code) in tqdm(enumerate(df[["OBJECT_ID", col_name,"WINDCODES","INDUSTRYCODES"]].values)): #修改之处 
            words = " ".join(clean_and_split(txt))  
            if not words:
                words = " "
            
            # words是已经分好的词,分好的词用空格再连接起来形成字符串
            words_list.append((i, idx, words,w_code,i_code)) 
        
        word_file = "fenci_"+file 
        new_df = pd.DataFrame(data=words_list, columns=["INDEX", "OBJECT_ID", "WORDS","WINDCODES","INDUSTRYCODES"])  #修改之处
        # 修改之处,把它写到一个tempo的位置
        new_df.to_csv(os.path.join(save_paths, word_file), header=True, index=False, encoding='utf_8_sig')  
        
    # delete_stopwords(text_type, file_lst)

# =============================================================================

def filter_stopwords(words, stopwords_set=set()):
    '''
    filter_stopwords函数用来过滤停用词
      输入的words是字符串，各个单词用" "分隔,stopwords set是一个set
      输出的是一个list
    '''
    # 删去停用词
    
    # 检查是否是空集
    if not words or words != words:
        return [] 
    
    # 检查是否是纯英文，删去纯英文新闻
    if not has_chinese(words): 
        return [] 
    
    return [word for word in words.split(" ") if word and not is_nan(word) and word not in stopwords_set]

# =============================================================================

def has_chinese(text):
    ''' 文本中是否包含中文（用于去除纯英文新闻）'''
    for ch in text[:200]:       # 为了效率，只看前 200 个字符 
        if ch >= u'\u4e00' and ch <= u'\u9fa5':
            return True 
    return False 

# =============================================================================

def delete_stopwords(text_type): 
    '''
    delete_stopwords函数用来删去停用词
      #通过data/stopwords/stopwords.txt读取停用词列表 
    '''
    
    file_lst = sorted(os.listdir(os.path.join(path, "data/ready"))) 
    # 读取停用词列表
    stopwords_set = set() 
    with open(os.path.join(path, "data/stopwords/stopwords.txt"), "r", encoding='utf-8') as f:
            for line in f.readlines():
                stopwords_set.add(line.strip()) #  删除空格
    # 设置读入路径
    read_paths = os.path.join(path, "results/{}/".format(text_type),"tempo")
    # 设置存储路径
    save_paths = os.path.join(path, "results/{}/".format(text_type))

    # 为防止后来更改初始文件列表，放置备份
    original_file_lst = file_lst.copy() #初始文件列表
    
    # 统一要读取的文件名格式
    for i in range(len(file_lst)):
        file_lst[i] = "fenci_" + file_lst[i] #修改之处
    
    # file_list是tempo
    for file in tqdm(file_lst, desc = "删去停用词" + text_type):
        if file.split(".")[-1] != "csv":
            continue 
        
        # 读取文件
        file_path = os.path.join(read_paths, file) #此处修改
        df = pd.read_csv(file_path) 
        
        # 过滤停用词
        words_list = [] 
        
        for i,idx,words,w_code,i_code in df[["INDEX", "OBJECT_ID", "WORDS","WINDCODES","INDUSTRYCODES"]].values: #修改之处
            
            ch_lst = filter_stopwords(words, stopwords_set=stopwords_set)
            ch_words = " ".join(ch_lst)
            if not ch_words:
                continue 
            words_list.append((i, idx, ch_words,w_code,i_code)) 

        # 存储
        word_file = file
        new_df = pd.DataFrame(data=words_list, columns=["INDEX", "OBJECT_ID", "WORDS","WINDCODES","INDUSTRYCODES"]) 
        new_df.to_csv(os.path.join(save_paths, word_file), header=True, index=False, encoding='utf_8_sig') 
    # 加入公司和行业代码
    # get_wcode_and_icode(text_type, original_file_lst)
    # join_wcode_and_icode(text_type, original_file_lst)

# =============================================================================

def get_wcode_and_icode(text_type): #修改之处
    '''
    函数get_wcode_and_icode获取公司和行业的代码
      读取数据是通过data/ready中的原始新闻获得
      获取公司代码的方式是通过原始新闻
      写出数据是到data/codes文件夹中的“code_日期”文件
    '''
    # 获得行业和公司代码
    
    # 原始数据文件夹
    csv_path = os.path.join(path, "data/ready")
    file_lst = os.listdir(csv_path)
    
    # 设置存储路径，表示哪些已经加入了公司代码
    save_file_path = os.path.join(path, "data/codes")
    
 
    processed_set = set() 
    for file in os.listdir(save_file_path): 
        if file.split(".")[-1] == 'csv': 
            processed_set.add(file[6:]) 
            # 已经存好的会有前缀"codes_"

    for file in tqdm(file_lst, desc="获得公司和行业代码" + text_type):
     
        if file.split(".")[-1] != "csv" or file in processed_set:
            continue 
        
        # 读取文件
        file_path = os.path.join(csv_path, file) 
        df = pd.read_csv(file_path)
        
        # 读取行业和公司代码
        words_list = []        
        for (idx, txt1, txt2) in tqdm(df[["OBJECT_ID", "WINDCODES", "INDUSTRYCODES"]].values): 
            words_list.append((idx, txt1, txt2)) #  words是已经分好的词
        
        # 存储
        save_paths = os.path.join(path, "data/codes/codes_{}".format(file)) 
        new_df = pd.DataFrame(data=words_list, columns=["OBJECT_ID", "WINDCODES", "INDUSTRYCODES"]) 
        new_df.to_csv(save_paths, header=True, index=False, encoding='utf_8_sig') 


# =============================================================================

def join_wcode_and_icode(text_type): #修改之处
    '''
    函数join_wcode_and_icode获取公司和行业的代码
      读入的文件是存放在result文件下后的分词后文件
      读入的文件还有存放在data/codes下的codes文件
    '''
    
    
    # 在处理好的分词结果中加入公司和行业代码，以便后续归总行业热度
    
    
    file_lst = os.listdir(os.path.join(path, "data/ready")) #修改之处
    # 设置存储路径(修改之处)
    save_paths = os.path.join(path, "results/{}/".format(text_type),'tempo') 
    
    # 统一文件名格式
    for i in range(len(file_lst)):
        file_lst[i] = "fenci_" + file_lst[i]
        
    for file in tqdm(file_lst, desc="加入公司和行业代码" + text_type):
        if file.split(".")[-1] != "csv": 
            continue 

        # 读取文件(原来的分词后文件)
        file_path = os.path.join(save_paths, file) 
        df = pd.read_csv(file_path)
        
        # 构建新闻编号与公司代码、行业代码的关联字典
        code_dict = {} 
        code_df = pd.read_csv("data/codes/codes_{}".format(file.split("_", 1)[-1])) #  以_为分隔符，分隔成两个，取后面那个
        for oid, wcode, icode in code_df.values:
            code_dict[oid] = (wcode, icode) 

        # 整合所有数据（将公司代码和行业代码加入到处理好的分词结果中）
        new_df_data = [] 
        for i, idx, words in df[["INDEX", "OBJECT_ID", "WORDS"]].values:
            (wcode, icode) = code_dict[idx]
            new_df_data.append((i, idx, words, wcode, icode)) 

        # 存储
        new_df = pd.DataFrame(data=new_df_data, columns=["INDEX", "OBJECT_ID", "WORDS", "WINDCODES", "INDUSTRYCODES"]) 
        new_df.to_csv(os.path.join(save_paths, file), header=True, index=False, encoding='utf_8_sig')         