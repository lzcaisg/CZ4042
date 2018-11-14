
# coding: utf-8

# In[18]:


import math
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pylab as plt
import pickle
import os
import csv
import sys


# In[240]:


LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 128
MAX_DOC_LEN=100
CHAR_NUM=256

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


# In[63]:


DATA_DIR = "../data"
SAVE_DIR = "../data/PartB_Result"
TRAIN_FILENAME = "train_medium.csv"
TEST_FILENAME = "test_medium.csv"


# In[29]:


with open(os.path.join(DATA_DIR, TEST_FILENAME),'r', encoding='ISO-8859-1', newline='') as f:
    reader = csv.reader(f)
    test_list = list(reader)
    print(test_list[0])


# In[30]:


with open(os.path.join(DATA_DIR, TRAIN_FILENAME),'r', encoding='ISO-8859-1', newline='') as f:
    reader = csv.reader(f)
    train_list = list(reader)
    print(train_list[0])


# In[31]:


y_test =[int(tmp[0]) for tmp in test_list]
print(y_test[:10])


# In[33]:


y_train =[int(tmp[0]) for tmp in train_list]
print(y_train[:10])


# In[43]:


X_train_text = [tmp[2] for tmp in train_list]
print(X_train_text[0])


# In[44]:


X_test_text = [tmp[2] for tmp in test_list]
print(X_test_text[0])


# ### Produce feature_word

# In[21]:


vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOC_LEN)


# In[72]:


X_train_array=np.array(list(vocab_processor.fit_transform(X_train_text)))


# In[73]:


X_train_array.shape


# In[74]:


X_test_array = np.array(list(vocab_processor.transform(X_test_text)))


# In[75]:


X_test_array.shape


# In[76]:


X_test_array[0]


# In[64]:


if not os.path.exists(SAVE_DIR):
    print("Not Exist")
    os.makedirs(SAVE_DIR) # CANNOT USE for nor absolute path


# In[71]:


with open(os.path.join(SAVE_DIR, "Train_word_raw.out"), 'wb') as fp:
    pickle.dump([X_train,y_train], fp)


# In[70]:


with open(os.path.join(SAVE_DIR, "Test_word_raw.out"), 'wb') as fp:
    pickle.dump([X_test,y_test], fp)


# In[80]:


X_train = np.array([tmp.reshape(-1,20) for tmp in X_train_array])
X_test = np.array([tmp.reshape(-1,20) for tmp in X_test_array])


# In[82]:


# For WORDS ONLY!!!!
X_train.shape


# In[84]:


with open(os.path.join(SAVE_DIR, "Train_word.out"), 'wb') as fp:
    pickle.dump([X_train,y_train], fp)
with open(os.path.join(SAVE_DIR, "Test_word.out"), 'wb') as fp:
    pickle.dump([X_test,y_test], fp)


# ### Produce feature_char

# In[86]:


X_test_text[0]


# In[237]:


idxs = [np.fromstring(tmp,dtype=np.uint8) for tmp in X_train_text]
X_train_charindex = np.zeros([len(idxs),MAX_DOC_LEN]).astype(int)
for i,j in enumerate(idxs):
    X_train_charindex[i][0:min(len(j), MAX_DOC_LEN)] = j[:min(len(j), MAX_DOC_LEN)]


# In[238]:


idxs = [np.fromstring(tmp,dtype=np.uint8) for tmp in X_test_text]
X_test_charindex = np.zeros([len(idxs),MAX_DOC_LEN]).astype(int)
for i,j in enumerate(idxs):
    X_test_charindex[i][0:min(len(j), MAX_DOC_LEN)] = j[:min(len(j), MAX_DOC_LEN)]


# In[243]:


with open(os.path.join(SAVE_DIR, "Train_char.out"), 'wb') as fp:
    pickle.dump([X_train_charindex,y_train], fp)
with open(os.path.join(SAVE_DIR, "Test_char.out"), 'wb') as fp:
    pickle.dump([X_test_charindex,y_test], fp)


# In[244]:


X_train_charonehot=tf.one_hot(X_train_charindex, CHAR_NUM)
X_test_charonehot=tf.one_hot(X_test_charindex, CHAR_NUM)


# In[245]:


X_train_charonehot

