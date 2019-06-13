import pickle
import numpy as np

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

# with open("emb_mat_dict.pkl", 'wb') as f:    # 保存该字典以pkl格式
#     pickle.dump(embed_mat, f, pickle.HIGHEST_PROTOCOL)
#     f.close()

