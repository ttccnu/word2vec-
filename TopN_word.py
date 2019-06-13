import pickle
import numpy as np
import heapq

with open("D:\python project\multimodal_emotion/text_rnn\emb_mat_dict.pkl", 'rb') as f:
    embed_mat = pickle.load(f)


ori_word = "飞机"
word = embed_mat[ori_word]
print("原始单词：", ori_word)

index = []
s = []
for key in embed_mat:
    score = np.dot(word, embed_mat[key])
    s.append(score)
    index.append(key)
    pass

result = heapq.nlargest(5, s)
for r in result:
    print("与该单词语义相近的单词:", index[s.index(r)])
