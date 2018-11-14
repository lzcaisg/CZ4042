
# coding: utf-8

# In[3]:


import math
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pylab as plt
import pickle
import os
import csv
import sys
import time
from datetime import timedelta
import gensim
import sklearn.decomposition


# In[4]:


LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 128
MAX_DOC_LEN=100
CHAR_NUM=256
WORD_WIDTH = 20
W2V_WIDTH = 300

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


# In[5]:


DATA_DIR = "../data"
SAVE_DIR = "../data/PartB_Result"
TRAIN_FILENAME = "train_medium.csv"
TEST_FILENAME = "test_medium.csv"


# In[1]:


def text2words(text):
    words = [tmp.split(' ')[1:] for tmp in text]
    return words


# In[13]:


def word2flatten_feature(X_words):
    # words = X_train_words
    feature = []
    hash_feature = []
    for words in X_words:
        tmp_hash = []
        for tmp in words:
            try:
                feature.append(model.get_vector(tmp))
                tmp_hash.append(1)
            except KeyError as err:
#                 print(err)
                tmp_hash.append(0)
                pass
        hash_feature.append(tmp_hash)
    print(len(feature))
    return [feature, hash_feature]


# In[37]:


def generate_structured_feat(feature, hash_feature):
    count = 0
    feature_list = []
    for i in range(len(hash_feature)):
        tmp_feat_list = []
        for j in range(len(hash_feature[i])):
            if hash_feature[i][j]:

                tmp_feat_list.append(feature[count])
                count += 1
            else:
                tmp_feat_list.append(np.zeros((WORD_WIDTH,), dtype = np.float32))
        feature_list.append(tmp_feat_list)
    
    print(count)
    return feature_list


# In[32]:


def pad_structure(feat_structured):
    for i, element in enumerate(feat_structured):
        if(len(element)>=MAX_DOC_LEN):
            feat_structured[i]=element[:MAX_DOC_LEN]
        else:
            remain_num = MAX_DOC_LEN-len(element)
            for j in range(remain_num):
                feat_structured[i].append(np.zeros((WORD_WIDTH,), dtype=np.float32))
    return feat_structured


# In[6]:


model = gensim.models.KeyedVectors.load_word2vec_format('/Volumes/Transcend/Google Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)


# In[9]:


with open(os.path.join(DATA_DIR, TEST_FILENAME),'r', encoding='ISO-8859-1', newline='') as f:
    reader = csv.reader(f)
    test_list = list(reader)
    print(test_list[0])


# In[10]:


with open(os.path.join(DATA_DIR, TRAIN_FILENAME),'r', encoding='ISO-8859-1', newline='') as f:
    reader = csv.reader(f)
    train_list = list(reader)
    print(train_list[0])


# In[44]:


def process_data(train_list, test_list):
    y_test =[int(tmp[0]) for tmp in test_list]
    y_train =[int(tmp[0]) for tmp in train_list]
    X_train_text = [tmp[2] for tmp in train_list]
    X_test_text = [tmp[2] for tmp in test_list]
    print("1/7 :Get Text Finished!")
    
    X_train_words = text2words(X_train_text)
    X_test_words = text2words(X_test_text)
    del X_train_text, X_test_text
    print("2/7 :Get Words Finished!")
    
    X_train_flatten_feat, X_train_hash = word2flatten_feature(X_train_words)
    X_test_flatten_feat, X_test_hash = word2flatten_feature(X_test_words)
    del X_train_words, X_test_words
    print("3/7 :Get Flattened Feature Finished!")
    
    pca = sklearn.decomposition.PCA(n_components=20)
    pca.fit(X_train_flatten_feat+X_test_flatten_feat)
    print("4/7 :PCA Set-up Finished!")
    
    X_train_flatten_feat_20 =  pca.transform(X_train_flatten_feat)
    X_test_flatten_feat_20 = pca.transform(X_test_flatten_feat)
    del X_train_flatten_feat, X_test_flatten_feat, pca
    print("5/7 :Feature Projection Finished!")
    
    X_train_structured_feat = generate_structured_feat(X_train_flatten_feat_20, X_train_hash)
    X_test_structured_feat = generate_structured_feat(X_test_flatten_feat_20, X_test_hash)
    del X_train_flatten_feat_20, X_train_hash, X_test_flatten_feat_20, X_test_hash
    print("6/7 :Structure Feature Finished!")
    
    X_train_structured_feat_padded = pad_structure(X_train_structured_feat)
    X_test_structured_feat_padded = pad_structure(X_test_structured_feat)
    del X_train_structured_feat, X_test_structured_feat
    print("7/7 :Convert to Fixed-length Finished!")
    
    return [np.array(X_train_structured_feat_padded), np.array(X_test_structured_feat_padded),
           np.array(y_train), np.array(y_test)]
    


# In[45]:


X_train, X_test, y_train, y_test = process_data(train_list, test_list)


# In[46]:


X_train.shape


# In[47]:


X_test.shape


# In[48]:


X_train[0]


# In[49]:


with open(os.path.join(SAVE_DIR, "Train_words_20.out"), 'wb') as fp:
    pickle.dump([X_train, y_train], fp)
with open(os.path.join(SAVE_DIR, "Test_words_20.out"), 'wb') as fp:
    pickle.dump([X_test, y_test], fp)

