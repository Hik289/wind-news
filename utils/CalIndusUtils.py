from tqdm import tqdm 
import pandas as pd 
import os 
import numpy as np 
from datetime import datetime 
from collections import Counter 


path = os.path.dirname(__file__) 
path = os.path.dirname(path) # python-主程序

# calculate_idf，计算单词的idf(这里指的是所有新闻)
        #其实tf针对的是一段时间的新闻或者一个行业的新闻
    # 调用calculate_tf_idf, 计算每个新闻每个单词的tf-idf
        #调用load_idf,把idf数据读入
        #调用get_tf_vec,计算词频向量
        #调用increment_tf_vec_by_time_period,计算每个月总共的词频向量
        #调用cal_tf_idf_vec,就是把tf_vec和idf_vec相乘
        #调用get_topN_word，得到每个月词频最高的N个单词
        #调用calculate_indus_tf，获得行业高频词，各行业不同的高频词
            #调用remove_dup,去除各行业相同的高频词
            #调用acquire_indus_heat，计算月度/周度某行业的热度


def is_nan(text):
    ''' 判断文本是否为空'''
    return text != text or text == "nan" 

def remove_time_dup(text_type,time_period, threshold = 0.1):
    '''
    210715 fxc创建
    去除各时段共同的热词
    threshold表示各州之间最大频率和最小频率的差异性，例如最小频率达到最大频率的0.1，则认为这个词一直出现，不能成为一段时间的热词
    将去重后的结果写在new_tf_idf和time_dup文件夹里
    '''
    root_path = os.path.join(path,'results',text_type)
    npy_path = os.path.join(root_path,'tf_idf',time_period,'vec')
    file_lst = os.listdir(npy_path)
    first = 1
    for file in file_lst:
        input_np = np.load(os.path.join(npy_path,file))
        input_np = input_np/np.sum(input_np)
        input_np = input_np.reshape(len(input_np),1)
        if first == 1:
            first = 0
            collect_np = input_np
        else:
            collect_np = np.hstack((collect_np,input_np))
    id2word, word2id, idf_vec = load_idf(os.path.join(root_path, "word_idf", "word_idf.csv"))
    common_index = []
    common_words = []
    for i in range(0,len(collect_np)):
        line = collect_np[i]
        if min(line) > threshold * max(line):
            collect_np[i] = 0
            common_index.append(i) #保存它的序号
            common_words.append(id2word[i])  #保存它的停用词
    for j in range(0,collect_np.shape[1]):
        top_N = get_topN_word(collect_np[:,j], id2word, N=200)
        txt_file = file_lst[j][:-4] + '.txt'
        if not os.path.exists(os.path.join(root_path, "new_tf_idf")):
            os.mkdir(os.path.join(root_path, "new_tf_idf"))
        if not os.path.exists(os.path.join(root_path, "new_tf_idf",time_period)):
            os.mkdir(os.path.join(root_path, "new_tf_idf",time_period))
        if not os.path.exists(os.path.join(root_path, "new_tf_idf",time_period,'vec')):
            os.mkdir(os.path.join(root_path, "new_tf_idf",time_period,'vec'))
            
        with open(os.path.join(root_path, "new_tf_idf",time_period, txt_file), "w") as f:
            for word in top_N: 
                f.write(word+"\n")
        np.save(os.path.join(root_path, "new_tf_idf",time_period,'vec',file_lst[j]), collect_np[:,j]) 
    
    # 写出停用词    
    if not os.path.exists(os.path.join(root_path,'time_dup')):
        os.mkdir(os.path.join(root_path,'time_dup'))
    save_path = os.path.join(root_path,'time_dup',time_period)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
                          
    with open(os.path.join(save_path, "time_dulicate.txt"), "w") as f: 
        for word in common_words:
            f.write(word+"\n")
        

# =============================================================================

def calculate_idf(text_type, time_period, certain_start, certain_end, topN):
    '''
    函数calculate_idf用来计算整个新闻集合上词的idf
        在计算单词频率的idf的定义是逆向词频，用总文件数除以该词出现的文件数，再求log10得到
        计算单词idf时，需要对同一新闻内的单词去重，方法是list(set(word_list))
        读入:分词后的结果
        写出:是按顺序排好的idf,存在results/content/word_idf里
    参数:
        text_type：可选有content/title，即分析标题或内容
        time_period: 表示按月统计热度
        certain_start/certain_end： 表示新闻统计的头/尾
        topN是用来计算排名前N的热词
        计算idf时用的分词以后的结果，自然也有删除停用词
    '''
    words_path = os.path.join(path, "results/{}/".format(text_type)) 
    
    # 每个词在多少份新闻中出现过
    word_docs = {} 
    
    # 总新闻数 
    cnt_docs = 0 
    
    #计算idf,在所有的新闻上
    for file in tqdm(os.listdir(words_path)): 
        if file.split(".")[-1] != "csv": 
            continue 
        file_path = os.path.join(words_path, file) 
        df = pd.read_csv(file_path) 
        #用来统计一个单词在多少个文件里出现过
        for obj_id, words in df[["OBJECT_ID", "WORDS"]].values: 
            # word_list = filter_stopwords(words, stopwords_set=stopwords_set) 
            if not words or words != words:
                continue 
            word_list = words.strip().split(" ") #删掉头尾指定的空格并分词
            if not word_list: 
                continue 
            cnt_docs += 1 #  统计总新闻数
            uniq_word_list = list(set(word_list)) #  创建没有重复词的单词表
            for word in uniq_word_list: 
                # 一条新闻中出现的每个词的 idf + 1
                word_docs[word] = word_docs.get(word, 0) + 1  
                #dict.get()表示拿值，如果不存在就返回0
    word_idf = {} #什么意思?
    word_large_idf = {} #什么意思?
    # tf是词频，表示某一个词在某一个文件中的出现频率
    # idf是逆向词频，用总文件数除以改词出现的文件数得到
    #  idf应该越大越好
    for w, c in word_docs.items(): 
        if is_nan(word) or not word:
            continue 
        word_large_idf[w] = float(cnt_docs / c)         # 尝试扩大 idf （不加 log）
        word_idf[w] = float(np.log2(cnt_docs / c))     # 原始的 idf （ log(出现该词的新闻数 / 总新闻数)) #修改之处,把idf变大,变成log2       
    print("Sort word idf...")   
    word_large_idf_list = sorted([(w,idf) for w,idf in word_large_idf.items()], key=lambda x: (-x[1], x[0]))
    word_idf_list = sorted([(w,idf) for w,idf in word_idf.items()], key=lambda x: (-x[1], x[0])) 
    # 存下两个版本的 idf 
    idf_df = pd.DataFrame(data=word_idf_list, columns=["WORD", "IDF"])
    idf_df = idf_df.drop_duplicates(['WORD'])
    if not os.path.exists(os.path.join(words_path, "word_idf")):
        os.mkdir(os.path.join(words_path, "word_idf"))      
    idf_df.to_csv(os.path.join(words_path, "word_idf", "word_idf.csv"), index=False, header=True, encoding='utf_8_sig') 
    large_idf_df = pd.DataFrame(data=word_large_idf_list, columns=["WORD", "IDF"])
    large_idf_df = large_idf_df.drop_duplicates(['WORD'])
    large_idf_df.to_csv(os.path.join(words_path, "word_idf", "word_large_idf.csv"), index=False, header=True, encoding='utf_8_sig') 
    # # 按照idf从大到小给单词排序并存储
    # lst = [w for w,_ in word_idf_list] 
    # with open(os.path.join(words_path, "word_idf", "word_list.txt"), "w") as f:
    #     for w in lst: 
    #         f.write(w+"\n") 
    calculate_tf_idf(text_type, time_period, certain_start, certain_end, topN)
   
    
   
# =============================================================================

def load_idf(idf_path):
    '''
    导入提前计算好的 idf 数据
      load_idf函数,用来调用idf
      输出有:
      idf_df是pandas结构的存储，从大到小
      idf_vec是IDF值的numpy形式，从大到小
      id2word是对应的单词，用array of object(ndarray object of numpy module)来存储
      word2id建立了单词和index之间的映射，是一个字典  
    '''
    idf_df = pd.read_csv(idf_path) 
    idf_df = idf_df.drop_duplicates(['WORD'])
    idf_vec = np.array(idf_df["IDF"].values)
    id2word = idf_df["WORD"].values
    word2id = {} 
    for i, word in enumerate(id2word):
        word2id[word] = i 
    return id2word, word2id, idf_vec


def norm(vec):
    sum_ = np.sum(vec)
    if sum_ > 0:
        return vec / sum_
    return vec 

# =============================================================================

def get_tf_vec(word_list, word2id):
    '''
    对于一条新闻，计算词频向量,把它映射到编号上
    调用get_tf_vec,计算一条新闻的词频向量
      输入的是word_list,是一个list
      输出的是一个np.array
    '''
    tf_vec = np.zeros(len(word2id)) 
    for word in word_list: 
        try:
            tf_vec[word2id[word]] += 1.0 
        except:
            pass 
    return norm(tf_vec) 

# =============================================================================
def cal_tf_idf_vec(tf_vec, idf_vec):
    '''
    计算 tf-idf
    调用cal_tf_idf_vec: 根据tf向量和某个时间段的idf向量计算tf_idf
    '''
    tf_idf_vec = tf_vec * idf_vec 
    return tf_idf_vec 


# =============================================================================
def increment_tf_vec_by_time_period(tp_tf_idf_vec, new_tf_idf_vec): 
    ''' 
    累积月度/周度词频向量 （这样可以尽量减少计算 tf-idf 中乘法的次数，提高效率）
    每条新闻的权重一样 
    调用increment_tf_vec_by_time_period: 对tf向量进行累加
    '''
    return tp_tf_idf_vec + new_tf_idf_vec 

# =============================================================================

def get_topN_word(tf_idf_vec, id2word, N=5): 
    '''
    调用get_topN_word,根据进入的tf_idf_vec和id2word(根据索引找单词),得到topN的单词
    '''
    desc_topN_id = np.argsort(tf_idf_vec)[::-1][:N] #sort后，得到各元素在原向量中的位置
    return ["{} {}".format(id2word[id_].upper(), tf_idf_vec[id_]) for id_ in desc_topN_id]

# =============================================================================

def str2week(s): 
    '''将一个字符串日期找到它对饮的周数'''
    dt = datetime.strptime(s, "%Y%m%d") 
    return int(dt.timestamp() - datetime(2014,12,28,0,0,0).timestamp()) // (86400*7)  #一周的秒数

# =============================================================================

def week_name(week):
    '''根据周的序号得到周名'''
    start = datetime.fromtimestamp(week*86400*7 + datetime(2014,12,28,0,0,0).timestamp())
    return "%04d%02d%02d" % (start.year, start.month, start.day) 

# =============================================================================

def calculate_tf_idf(text_type, time_period, certain_start, certain_end, topN):

    """ 
    calculate_tf_idf函数,用来计算新闻里的tf_idf
      调用load_idf,将全部新闻上得到的idf读入
      把一条新闻读进来以后，需要调用get_tf_vec计算词频向量
      把所有新闻的tf_vec,累加到month_tf_vec上,运用increment_tf_vec_by_time_period
      在一个月结束后,运用cal_tf_idf_vec,就是把tf_vec和idf_vec相乘
      调用get_topN_word，得到每个月词频最高的N个单词
      
      certain_start,certain_end在time_period为month时没有实际作用
      在统计时,所有的文件都会统计
      
      写出文件主要由两个
          results/tf_idf/month(week)根目录下的201701存放了topN的热词
          results/tf_idf/month(week)/vec下存放了所有的热词向量
      调用calculate_indus_tf，获得行业高频词，各行业不同的高频词
          调用remove_dup,去除各行业相同的高频词
          调用acquire_indus_heat，计算月度/周度某行业的热度
    """
    # text_type = "content"  # 新闻标题 title，或新闻内容 content
    # time_period = "month" #  "month" or  "week"
    # certain_start = "20150101"
    # certain_end = "20150103
    # topN是什么?
    # 这个主要是计算tf的
    
    
    print("calculating tf-idf for time {} - {}".format(certain_start, certain_end)) 
        
    # topN = 200

    words_path = os.path.join(path, "results/{}/".format(text_type)) 

    # load idf  id2word是索引编号(是单词的编号)，从0开始
    # id2word,word2id, idf_vec到底是啥?
        #id2word里存的是单词的名称, id2word = idf_df["WORD"].values
        #word2id里面存的是每个单词对应的编号
        #idf_vec里面是每个词的idf
    id2word, word2id, idf_vec = load_idf(os.path.join(words_path, "word_idf", "word_idf.csv")) 

    if time_period == "month": 
        # 以月为单位计算tf-idf
        print("Calculating tf-idf by month")
        month_tf_vec = np.zeros(len(idf_vec)) 
        cur_month = None 
        
        #对分词结束后的文件一个个处理
        file_list = sorted(os.listdir(words_path))[::-1] 
        for file in tqdm(file_list): 
            if file.split(".")[-1] != "csv": 
                continue 

            file_path = os.path.join(words_path, file) 
            df = pd.read_csv(file_path) 
            
            #关键是要按月计算tf_idf，所以要判断是否还在本月
            if not cur_month:
                cur_month = file[20:26] 
            
            elif cur_month != file[20:26]: 
                # 如果来了一个新的month,怎么办? 这里主要处理这一点,是上个月的tf-idf要存起来吗？
                
                # 计算td-idf(根据某个月的tf_vec和全部单词的idf_vec)
                month_tf_idf_vec = cal_tf_idf_vec(month_tf_vec, idf_vec)
                
                # 选取月度前topN单词
                month_topN = get_topN_word(month_tf_idf_vec, id2word, N=topN) 
                
                # 保存tf-idf结果
                save_file = "{}.txt".format(cur_month)
                with open(os.path.join(words_path, "tf_idf", "month", save_file), "w") as f:
                    for word in month_topN: 
                        f.write(word+"\n") 
                np.save(os.path.join(words_path, "tf_idf", "month", "vec", "{}.npy".format(cur_month)), month_tf_idf_vec) 

                # 更新
                cur_month = file[20:26] 
                month_tf_vec = np.zeros(len(idf_vec)) 
            
            # 计算词频,doc_tf_vec存放的是一个新闻的词频,month_tf_vec存放的是一个月内的累计词频
            # 如果一个月结束了，会在上面一个if判定里清除
            
            # =============================================================================
            # df_copy = df.copy(deep=True).drop(columns = ['WORDS'])
            # df_copy['WORD_COUNT'] = ''
            # df_copy['WORD_COUNT'] = df_copy['WORD_COUNT'].apply(lambda x:np.zeros(len(word2id)))
            # word_count_matrix = np.zeros((len(df),len(word2id)))
            # =============================================================================
            
            
            # for idx,obj_id, words in zip(df.index,df[["OBJECT_ID", "WORDS"]].values):
            for idx in df.index:
                obj_id = df.loc[idx,'OBJECT_ID']
                words = df.loc[idx,'WORDS']
                if not words or words != words:
                    continue 
                word_list = words.strip().split(" ")
                if not word_list:
                    continue 
                # =============================================================================
                #以下几行代码代替了原来的get_tf_vec
                
                # 以下代码计算新闻的词频
                # 直接创造出一个单词：出现次数的字典,用Collection的Counter类
                # 之前在计算tf的时候是一次次累加的(get_tf_vec函数)
                word_cnt = Counter(word_list)           
                # 数所有单词总共出现个数(单词出现总个数)
                tot = sum([v for v in word_cnt.values()]) 
                # 创造一个  单词：词频的字典 tf
                tf = {k:v*1.0/tot for k,v in word_cnt.items()}
               
                doc_tf_vec = np.zeros(len(word2id)) 
                tempindex = np.array(list(map(lambda x: word2id[x] if x in word2id.keys() else -1,tf.keys())))
                doc_tf_vec[tempindex[tempindex !=-1]] = np.array(list(tf.values()))[tempindex!=-1]
                # doc_tf_vec[list(map(lambda x: word2id[x],tf.keys()))] = list(tf.values())
                
                # word_count_matrix[idx,:] = doc_tf_vec
                # =============================================================================
                # df_copy修改
                # df_copy.at[idx,'WORD_COUNT'] = doc_tf_vec
                # =============================================================================
                # doc_tf_vec = get_tf_vec(word_list, word2id)               
                month_tf_vec = increment_tf_vec_by_time_period(month_tf_vec, doc_tf_vec) 
            # =============================================================================
            # 以下几行代码将doc_tf_vec/word_count_matrix进行输出
            # 如果没有文件夹,建立文件夹
            # if not os.path.exists(os.path.join(words_path,'count_by_news')):
            #     os.mkdir(os.path.join(words_path,'count_by_news'))
            # file_name = 'count_'+file[6:-4]
            # 发现用pandas存储到csv不合适,因为一个单元格里没办法存储向量
            # 发现用numpy存储到npy不合适,因为太大了
            # np.save(os.path.join(words_path,'count_by_news',file_name+'.npy'),word_count_matrix) 
            # df_copy.to_csv((os.path.join(words_path,'count_by_news',file_name)),\
            #                index=False, header=True, encoding='utf_8_sig') 

            
                
                
            # =============================================================================

        # 计算tf-idf,这相当于计算月度热词
        month_tf_idf_vec = cal_tf_idf_vec(month_tf_vec, idf_vec)
        
        # 取月度前topN单词
        month_topN = get_topN_word(month_tf_idf_vec, id2word, N=topN) 
        
        # 保存
        save_file = "{}.txt".format(cur_month)
        if not os.path.exists(os.path.join(os.path.join(words_path, "tf_idf"))):
            os.mkdir(os.path.join(words_path, "tf_idf"))
        if not os.path.exists(os.path.join(os.path.join(words_path, "tf_idf", "month"))):
            os.mkdir(os.path.join(words_path, "tf_idf", "month"))
                    
            
        with open(os.path.join(words_path, "tf_idf", "month", save_file), "w") as f:
            for word in month_topN: 
                f.write(word+"\n") 
        
        if not os.path.exists(os.path.join(words_path, "tf_idf", "month", "vec")):
            os.mkdir(os.path.join(words_path, "tf_idf", "month", "vec"))
        np.save(os.path.join(words_path, "tf_idf", "month", "vec", "{}.npy".format(cur_month)), month_tf_idf_vec) 

    elif time_period == "week":
        # 以周为单位计算tf-idf
        # 与月的唯二不同是week_name()和str2week()两个函数的使用
        
        print("Calculating tf-idf by week")
        week_tf_vec = np.zeros(len(idf_vec)) 
        cur_week = None 
        
        # 读取文件
        file_list = sorted(os.listdir(words_path)) 
        if certain_start and certain_end: 
            file_list = [file for file in file_list if file[20:28]>=certain_start and file[20:28]<=certain_end]
        if not file_list:
            return
        
        print("{} - {}".format(file_list[0], file_list[-1]))
        
        
        for file in tqdm(file_list): 
            if file.split(".")[-1] != "csv": 
                continue 

            file_path = os.path.join(words_path, file) 
            df = pd.read_csv(file_path) 

            if not cur_week: 
                #如果没有
                cur_week = str2week(file[20:28]) 

            elif cur_week != str2week(file[20:28]): 
                # 计算最新的tf-idf并保存结果
                week_tf_idf_vec = cal_tf_idf_vec(week_tf_vec, idf_vec) 
                week_topN = get_topN_word(week_tf_idf_vec, id2word, N=topN) 
                save_file = "{}.txt".format(week_name(cur_week)) 
                with open(os.path.join(words_path, "tf_idf", "week", save_file), "w") as f: 
                    for word in week_topN: 
                        f.write(word+"\n") 
                np.save(os.path.join(words_path, "tf_idf", "week", "vec", "{}.npy".format(week_name(cur_week))), week_tf_idf_vec) 

                # 更新
                cur_week = str2week(file[20:28]) 
                week_tf_vec = np.zeros(len(idf_vec)) 

            for obj_id, words in df[["OBJECT_ID", "WORDS"]].values:
                if not words or words != words:
                    continue 
                word_list = words.strip().split(" ")
                if not word_list:
                    continue 


                word_cnt = Counter(word_list)           
                # 数所有单词总共出现个数(单词出现总个数)
                tot = sum([v for v in word_cnt.values()]) 
                # 创造一个  单词：词频的字典 tf
                tf = {k:v*1.0/tot for k,v in word_cnt.items()}
                
                doc_tf_vec = np.zeros(len(word2id)) 
                tempindex = np.array(list(map(lambda x: word2id[x] if x in word2id.keys() else -1,tf.keys())))
                doc_tf_vec[tempindex[tempindex !=-1]] = np.array(list(tf.values()))[tempindex!=-1]
                
            
                # doc_tf_vec = get_tf_vec(word_list, word2id) 
                week_tf_vec = increment_tf_vec_by_time_period(week_tf_vec, doc_tf_vec) 

        week_tf_idf_vec = cal_tf_idf_vec(week_tf_vec, idf_vec)
        week_topN = get_topN_word(week_tf_idf_vec, id2word, N=topN) 
        save_file = "{}.txt".format(week_name(cur_week))
        if not os.path.exists(os.path.join(os.path.join(words_path, "tf_idf"))):
            os.mkdir(os.path.join(words_path, "tf_idf"))
        if not os.path.exists(os.path.join(os.path.join(words_path, "tf_idf", "week"))):
            os.mkdir(os.path.join(words_path, "tf_idf", "week"))
        
        
        with open(os.path.join(words_path, "tf_idf", "week", save_file), "w") as f:
            for word in week_topN: 
                f.write(word+"\n") 
        if not os.path.exists(os.path.join(words_path, "tf_idf", "week", "vec")):
            os.mkdir(os.path.join(words_path, "tf_idf", "week", "vec"))
        np.save(os.path.join(words_path, "tf_idf", "week", "vec", "{}.npy".format(week_name(cur_week))), week_tf_idf_vec) 
    
    calculate_indus_tf(text_type, time_period)

# =============================================================================

def remove_dup(dict_,text_type): 
    '''
     统计每个词的词频在每个行业中的排名情况，只保留最高排名的行业，认为这个词是该行业的专属词之一
     目的：尽可能消除行业之间的共同信息，保留当前行业特有的信息 
     输入的dict_是二元词典
     返回的new_dlist也是个二元词典
    '''
    word_rank = {} 
    dlist = {} 
    rankdict = {}
    # get word rank
    # 这是一个词典嵌套,其中ind对应的是行业名称,ind_dict对应的是该行业的热词词典
    for ind, ind_dict in tqdm(dict_.items(), desc="word rank"): 
        #dlist是经过排序后的词典
        dlist[ind] = sorted([(k,v) for k,v in ind_dict.items()], key=lambda x: (-x[1], x[0])) #  行业内部排名
        #得到每个word在各个行业里的最高排名
        
        for i, (word, _) in enumerate(dlist[ind]): #i对应某词在某个行业里的排名
            if word not in word_rank:
                word_rank[word] = i 
                
                # 修改的地方
                rankdict[word] = []
                rankdict[word].append(i)
            else:
                word_rank[word] = min(i, word_rank[word]) 
                # 修改的地方
                rankdict[word].append(i)
    # word_rank对应一个词在所有行业中的最高排名
    new_dlist = {} 
    for ind, ind_lst in tqdm(dlist.items(), desc="filter dup"):
        new_dlist[ind] = [] 
        for i, (word, v) in enumerate(dlist[ind]):
            if i > word_rank[word]:
                continue 
            new_dlist[ind].append((word, v)) #  添加行业专属词（因为这个词在这个行业中相较于其他行业排名最高）
    # 修改的部分
    common_words = []
    for word,ranks in rankdict.items():
        if len(ranks) == len(dlist):
            if max(ranks) - min(ranks) <= 0.05 * len(word_rank):
                common_words.append(word)
    save_path = os.path.join(path,'results',text_type,'indus_dup')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path, "indus_dulicate.txt"), "w") as f: 
        for word in common_words:
            f.write(word+"\n")
        
    return new_dlist 



# =============================================================================
def calculate_indus_tf(folder, time_period):
    '''
    目的是得到分行业的词频（是在所有的数据集上）
    Parameters
    ----------
    folder : 可以是content/title
    time_period : 可以是month/week
    -------
    .
    读入:
        从data根目录下读入indus1(2)_belong_2
        从result根目录下读入所有的分词结果
    写出:
        results/content(title)/industry1(2)里的每个行业的热词.csv
    调用函数:remove_dup
    
    调用函数:acquire_indus_heat
    '''
    
    #读入行业归属,是一个二维矩阵,行是date,列是股票代码,值是对应的行业代码
    ind1_df = pd.read_pickle(os.path.join(path, "data", "indus1_belong_2.pickle")) #一级行业
    ind2_df = pd.read_pickle(os.path.join(path, "data", "indus2_belong_2.pickle")) #二级行业

    df_path = os.path.join(path, "results/{}/".format(folder))

    industry1_to_tf = {}    # 一级行业的词频词典,二维的,一个维度是行业,一个维度是词
    industry2_to_tf = {}    # 二级行业 


    file_lst = sorted(os.listdir(df_path))
    for file in tqdm(file_lst, desc=folder):
        # 穷举每个文件
        if file.split(".")[-1] != "csv":
            continue 
        origin_file = file[6:]
        try:
            df_origin = pd.read_csv(os.path.join(path,"data","ready",origin_file))
        except:
            continue

        date = file[20:28] # 比如 “20150101“   原格式为“fenci_FinancialNews_20150101.csv”
        ind1_dict = {} #行业词典归属
        ind2_dict = {} 
        
        #找到在date这一日，各股票对应的一级行业
        ind1 = ind1_df.loc[date] 
        #c对应的是股票(公司代码),i1对应的是一级行业
        for c, i1 in zip(ind1_df.columns, ind1): 
            # c 为公司代码，i1 为对应的一级行业 
            if i1 and i1==i1:
                ind1_dict[c] = i1  
                                
        ind2 = ind2_df.loc[date]
        for c, i2 in zip(ind2_df.columns, ind2):
            # c 为公司代码，i2 为对应的二级行业 
            if i2 and i2==i2:
                ind2_dict[c] = i2       

        if not ind1_dict and not ind2_dict:
            continue 
        
        #读入分词后的结果
        file_path = os.path.join(df_path, file) 
        df = pd.read_csv(file_path) 
        
        
        # for i, idx, words, wcode, _ in df.values: 
        for idx, obj_id,words in df[["INDEX", "OBJECT_ID", "WORDS"]].values: 
            #穷举每条新闻
            temp_index = df_origin[df_origin['OBJECT_ID'] == obj_id].index.tolist()
            if not temp_index:
                continue
            wcode = df_origin.loc[temp_index[0],'WINDCODES']
            
            
            if wcode and wcode == wcode: 
                # 把相关公司的代码都整理出来
                # 为什么是这样的？因为wcode的格式是 000001.SH:上证综指|ON0209:沪深|399001.SZ:深证成指|ON02:公司

                wcode_lst = [w.split(":")[0] for w in wcode.split("|")]   
            else: 
                continue 
            
            
            # 以下代码计算新闻的词频
            # 直接创造出一个单词：出现次数的字典,用Collection的Counter类
            # 之前在计算tf的时候是一次次累加的(get_tf_vec函数)
            word_cnt = Counter(words.split(" "))           
            # 数所有单词总共出现个数(单词出现总个数)
            tot = sum([v for v in word_cnt.values()]) 
            # 创造一个  单词：词频的字典 tf
            tf = {k:v*1.0/tot for k,v in word_cnt.items()} 

            # 目的是得到分行业的热词            
            # 对于wcode_lst之内每个公司wc进行循环
            for wc in wcode_lst:        
                # wind code  新闻涉及的相关公司代码列表
                
                if wc in ind1_dict:
                    # 得到一级行业
                    i1 = ind1_dict[wc]
                    
                    if i1 not in industry1_to_tf:
                        industry1_to_tf[i1] = {} 

                    for w,f in tf.items(): 
                        # 一级行业 i1 相关的新闻中词语 w 出现的词频增加 f 
                        industry1_to_tf[i1][w] = industry1_to_tf[i1].get(w, 0.0) + f  
                        
                if wc in ind2_dict: 
                    # 得到二级行业
                    i2 = ind2_dict[wc] 
                    if i2 not in industry2_to_tf: 
                        industry2_to_tf[i2] = {} 
                    for w,f in tf.items():
                        # 二级行业 i2 相关的新闻中词语 w 出现的词频增加 f 
                        industry2_to_tf[i2][w] = industry2_to_tf[i2].get(w, 0.0) + f 

    # 创建存储路径
    folder1_path = os.path.join(df_path,"industry1")
    if not os.path.isdir(folder1_path):
        os.makedirs(folder1_path)  
    folder2_path = os.path.join(df_path,"industry2")
    if not os.path.isdir(folder2_path):
        os.makedirs(folder2_path)

    # 筛选标准是这个词在这个行业中相较于其他行业排名最高
    # 去掉不同行业之间的共同词，保留当前行业特有的词 
    industry1_to_tf_uniq = remove_dup(industry1_to_tf,folder)
    for i1, wdict in tqdm(industry1_to_tf_uniq.items(), desc="saving ind1"):
        pd.DataFrame(data=wdict, columns=["WORD", "IND_TF"]).to_csv(
			os.path.join(folder1_path, "{}.csv".format(i1)), index=False, header=True, encoding='utf_8_sig') 

    industry2_to_tf_uniq = remove_dup(industry2_to_tf,folder)
    for i2, wdict in tqdm(industry2_to_tf_uniq.items(), desc="saving ind2"):
        pd.DataFrame(data=wdict, columns=["WORD", "IND_TF"]).to_csv(
			os.path.join(folder2_path, "{}.csv".format(i2)), index=False, header=True, encoding='utf_8_sig')
    
    acquire_indus_heat(folder, time_period)


# =============================================================================

def indus_vec(df, word2id):
    '''
     获得代表每个行业专属词的词频向量(需要进行正则化),数据结构是np.array
    （利用在 industry_related_word_freq.py 中获得的每个行业的专属词及频率来计算）
    '''
    

    tf_vec = np.zeros(len(word2id)) 
    scores = np.array(df["IND_TF"].values) / np.sum(df["IND_TF"].values) 
    for i in range(len(scores)):
        w = df["WORD"].values[i] #w表示单词
        s = scores[i]            #s表示得到的词频
        if w in word2id.keys():
            tf_vec[word2id[w]] = s  #将单词转换为单词序号
    return tf_vec #返回的tf_vec是个np.array

# =============================================================================

def indus_vec_dict(path, word2id):
    '''
    生成行业词频向量的字典，即通过行业名称来获取该行业专属词的词频向量 
    最后得到的是np.array
    path表示行业热词存储的路径
    '''
    ind_vec_dict = {} 
    for file in tqdm(os.listdir(path), desc="indus vec dict"):
        if file.split(".")[-1] != "csv":
            continue 
        #ind表示行业
        ind = file.split(".")[0] 
        df = pd.read_csv(os.path.join(path, file)) 
        #ind_vec_dict得到的是行业热词的词典
        ind_vec_dict[ind] = indus_vec(df, word2id) # 对于该行业 ind，计算其行业专属词词频向量 
    # print(len(ind_vec_dict))
    return ind_vec_dict

# =============================================================================

def acquire_indus_heat(text_type, time_period):
    '''
    # 计算 月度/周度 行业热度
    # 月度 or 周度的 tf-idf * 行业专属词的词频向量
    '''
    words_path = os.path.join(path, "results/{}/".format(text_type))
    id2word, word2id, idf_vec = load_idf(os.path.join(words_path, "word_idf", "word_idf.csv"))    

    save_path = words_path
    # 读入的是各行业的热词,得到的是一个一维词典,key是行业名称,value是一个np.array
    ind1_vec_dict = indus_vec_dict(os.path.join(save_path, "industry1"), word2id) 
    ind2_vec_dict = indus_vec_dict(os.path.join(save_path, "industry2"), word2id)

    # 导入事先计算好的月度/周度 tf-idf，通过calculate_tf_idf计算得到
    tf_idf_path = os.path.join(words_path, "tf_idf", time_period, "vec") 

    for file in tqdm(os.listdir(tf_idf_path), desc="by {}".format(time_period)):
        tf_idf = np.load(os.path.join(tf_idf_path, file)) 

        # tf-idf(某段时间) * 一级行业词频向量，获得行业热度得分 
        # 行业热度得分是某段时间内的tf_idf和行业词频的点积
        # 如果这个点积越高，说明该时间段内的热词会存在于该行业之内，所以行业热度越高
        indus1_scores = [(k,tf_idf.dot(v)) for k,v in ind1_vec_dict.items()]   
        indus1_sort_scores = sorted(indus1_scores, key=lambda x:(-x[1], x[0])) 

        # tf-idf * 二级行业词频向量，获得行业热度得分 
        indus2_scores = [(k,tf_idf.dot(v)) for k,v in ind2_vec_dict.items()]   
        indus2_sort_scores = sorted(indus2_scores, key=lambda x:(-x[1], x[0]))

        if not os.path.exists(os.path.join(save_path, "industry1")):
            os.mkdir(os.path.join(save_path, "industry1"))
        if not os.path.exists(os.path.join(save_path, "industry1", "indus_heat")):
            os.mkdir(os.path.join(save_path, "industry1", "indus_heat"))
        if not os.path.exists(os.path.join(save_path, "industry1", "indus_heat", time_period)):
            os.mkdir(os.path.join(save_path, "industry1", "indus_heat", time_period))
        if not os.path.exists(os.path.join(save_path, "industry2")):
            os.mkdir(os.path.join(save_path, "industry2"))
        if not os.path.exists(os.path.join(save_path, "industry2", "indus_heat")):
            os.mkdir(os.path.join(save_path, "industry2", "indus_heat"))
        if not os.path.exists(os.path.join(save_path, "industry2", "indus_heat", time_period)):
            os.mkdir(os.path.join(save_path, "industry2", "indus_heat", time_period))
        
            
        pd.DataFrame(data=indus1_sort_scores, columns=["INDUSTRY", "HEAT"]).to_csv(
                os.path.join(save_path, "industry1", "indus_heat", time_period, file.split(".")[0]+".csv"), index=False, header=True, encoding='utf_8_sig') 
        pd.DataFrame(data=indus2_sort_scores, columns=["INDUSTRY", "HEAT"]).to_csv(
                os.path.join(save_path, "industry2", "indus_heat", time_period, file.split(".")[0]+".csv"), index=False, header=True, encoding='utf_8_sig') 
        
        
# =============================================================================
def gen_news_factor(text_type,time_period, num_ind = 5):
    
    csv_path = os.path.join(path,'results',text_type,'industry1','indus_heat',time_period)
    file_lst = os.listdir(csv_path)
    df_indus_collect = pd.DataFrame(columns = ['石油石化', '煤炭', '有色金属', '电力及公用事业', '钢铁', '基础化工', '建筑', '建材', '轻工制造', '机械',\
       '电力设备及新能源', '国防军工', '汽车', '商贸零售', '消费者服务', '家电', '纺织服装', '医药', '农林牧渔',\
       '银行', '房地产', '交通运输', '电子', '通信', '计算机', '传媒', '综合金融', '酒类', '饮料', '食品',\
       '证券Ⅱ', '保险Ⅱ', '多元金融'])
    for file in file_lst:
        if file[-4:] != '.csv':
            continue
        indus_heat = pd.read_csv(os.path.join(csv_path,file))
        indus_heat = indus_heat.T
        if time_period == 'month':
            heat_time = file[:6]
            year = int(heat_time[:4]) 
            month = int(heat_time[4:6])
            
            if month<9:
                month = month+1
                heat_time = str(year) + '0' + str(month) + '01'
            elif month<12:
                month = month+1
                heat_time = str(year) + str(month) + '01'
            else:
                year = year+1
                heat_time = str(year) + '0101'
        elif time_period == 'week':
            heat_time = week_name(str2week(file[:8])+1)
        else:
            return
        
        indus_heat = indus_heat.rename(columns = indus_heat.loc['INDUSTRY']).\
            drop(index = 'INDUSTRY').rename(index = {'HEAT':heat_time})
        df_indus_collect = pd.concat([df_indus_collect,indus_heat])
    
    news_factor = df_indus_collect.reindex(columns = ['石油石化', '煤炭', '有色金属', '电力及公用事业', '钢铁', '基础化工', '建筑', '建材', '轻工制造', '机械',\
       '电力设备及新能源', '国防军工', '汽车', '商贸零售', '消费者服务', '家电', '纺织服装', '医药', '农林牧渔',\
       '银行', '房地产', '交通运输', '电子', '通信', '计算机', '传媒', '综合金融', '酒类', '饮料', '食品',\
       '证券Ⅱ', '保险Ⅱ', '多元金融'])
    news_factor_indnorm = news_factor.apply(lambda x: x/np.sum(x),axis = 1) # 按全部新闻热度归一化
    news_factor_timenorm = news_factor_indnorm.copy(deep = True)
    for ind in news_factor_indnorm.columns:
        column = news_factor_indnorm[ind]
        for i in range(0,len(column)):
             news_factor_timenorm[ind][i] = column[i]/np.mean(column[0:i+1])
    if not os.path.exists(os.path.join(path,'results',text_type, 'factor')):
        os.mkdir(os.path.join(path,'results',text_type, 'factor'))
    save_path = os.path.join(path,'results',text_type, 'factor', time_period)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    news_factor_timenorm.to_csv(os.path.join(save_path,'norm_facotor.csv'), index=True, header=True, encoding='utf_8_sig')
    gen_factor = news_factor_timenorm.copy(deep=True)
    for i in news_factor_timenorm.index:
        gen_factor.loc[i] = 0
        line = np.array(news_factor_timenorm.loc[i])
        minthreshold = np.sort(line[~np.isnan(line)])[::1][num_ind]
        maxthreshold = np.sort(line[~np.isnan(line)])[::-1][num_ind]
        gen_factor.loc[i, line > maxthreshold] = -1
        gen_factor.loc[i, line < minthreshold]  = 1
    gen_factor.to_csv(os.path.join(save_path,'gen_facotor.csv'), index=True, header=True, encoding='utf_8_sig')
    gen_factor.index = gen_factor.index.map(pd.to_datetime)
    gen_factor.to_pickle(os.path.join(save_path,'news_factor'))
        # sortindex = news_factor.loc[i].argsort().values
        # news_factor.loc[i,sortindex == 1]
        # news_factor.loc[i,sortindex[-3:]] = 1
        # news_factor.loc[i,sortindex[4:-3]] = 0
        
        
        
        
   
        