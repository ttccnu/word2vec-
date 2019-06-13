import jieba
import codecs
import re
import numpy as np
import heapq
import pickle

re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # 正则项

filename = "D:\python project\multimodal_emotion/text_rnn\datasets\cnews.test.txt"

text = []
label = []

with codecs.open(filename, 'r', encoding='utf-8') as f:
    for _, line in enumerate(f):
        line = line.strip()
        line = line.split('\t')
        assert len(line) == 2
        blocks = re_han.split(line[1])
        word = []
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        text.append(word)
        label.append(line[0])

embed_mat = {}

with open("D:\python project\multimodal_emotion/text_rnn\data/vector_word.txt", "r", encoding='UTF-8') as f:
    for l in f:
        l = l.strip("\n")
        key = l.split(" ")[0]
        value_s = l.split(" ")[1:]
        value = np.asarray([float(i) for i in value_s])
        embed_mat.update({key: value})
        pass
embed_mat.pop("370695")

# with open("emb_mat_dict", 'wb') as f:
#     pickle.dump(embed_mat, f, pickle.HIGHEST_PROTOCOL)


