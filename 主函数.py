# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:29:54 2021

@author: 
"""
#%%
import utils.SplitUtils as SplitUtils
import utils.CalIndusUtils as CalIndusUtils
import utils.CalCountUtils as CalCountUtils
#%%
if __name__ == "__main__":
        
    # 初步分词
    # SplitUtils.split_words("content")  # 根据content or title展开分析
    # # 删去停用词
    # SplitUtils.delete_stopwords("content")
    
    # 更新停用词
    # SplitUtils.update_stopwords("content") #更新停用词(筛选一些新的停用词)
    
    # 计算行业热度
    # CalIndusUtils.calculate_idf("content", "week", "20170101", "20170101", 200)
    CalIndusUtils.gen_news_factor("title", "week")
    # # 更新停用词
    # CalCountUtils.get_word_count("title", 6, 0.1)
    # CalIndusUtils.remove_time_dup("content","month")
