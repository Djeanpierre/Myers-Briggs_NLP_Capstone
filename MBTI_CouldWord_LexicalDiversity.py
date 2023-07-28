import ast
import os


# os.chdir('/Users/tina/PycharmProjects/BAN675/675_ASMT')
import pandas as pd
import nltk
import re
import string
import gensim
import numpy as np
import matplotlib.pyplot as plt
import pickle
# plotting
import seaborn as sns
import matplotlib.pyplot as plt


# df = pd.read_csv('clean_v2.csv', index_col=False)
# df['New'] = df['New'].apply(lambda s : ast.literal_eval(s))
# # show the distribution of types
# types = df['type'].value_counts()
# plt.figure(figsize=(12,4))
# sns.barplot(types.index, types.values, alpha=0.8)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Types', fontsize=12)
# plt.show()

# # group the type individually
# def get_types(post):
#     type = post['type']
#
#     I = 0;
#     N = 0
#     T = 0;
#     J = 0
#
#     if type[0] == 'I':
#         I = 1
#     elif type[0] == 'E':
#         I = 0
#     else:
#         print('I-E incorrect')
#
#     if type[1] == 'N':
#         N = 1
#     elif type[1] == 'S':
#         N = 0
#     else:
#         print('N-S incorrect')
#
#     if type[2] == 'T':
#         T = 1
#     elif type[2] == 'F':
#         T = 0
#     else:
#         print('T-F incorrect')
#
#     if type[3] == 'J':
#         J = 1
#     elif type[3] == 'P':
#         J = 0
#     else:
#         print('J-P incorrect')
#     return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})
#
#
# myerstype = df.join(df.apply(lambda post: get_types(post), axis=1))
# print(myerstype.columns) # 'type', 'Clean', 'New', 'IE', 'NS', 'TF', 'JP'

# # extract the adj for each type
# def clean(post):
#     words = [word for word in post if len(word) > 3 and word.isalpha()]
#     tagged = nltk.pos_tag(words)
#     adjs = [x for x, y in tagged if y.startswith('JJ')]
#
#     return adjs
# #
# #
# myerstype['ADJs'] = myerstype['New'].apply(lambda x: clean(x))
# newdf = myerstype[['ADJs', 'IE', 'NS', 'TF', 'JP']]
# I = newdf.groupby('IE').get_group(1)  # Introvert
# E = newdf.groupby('IE').get_group(0)  # total 1999 out of 8675 Extrovert
# N = newdf.groupby('NS').get_group(1)
# S = newdf.groupby('NS').get_group(0)
# T = newdf.groupby('TF').get_group(1)
# F = newdf.groupby('TF').get_group(0)
# J = newdf.groupby('JP').get_group(1)
# P = newdf.groupby('JP').get_group(0)
#print(E.ADJs)

# cloud word Finished---------------------------------------

# from wordcloud import WordCloud
#
# # E_adj = [adj for post in Extro.ADJs for adj in post]
# # I_adj = [adj for post in Intro.ADJs for adj in post]
# S_adj = [adj for post in S.ADJs for adj in post]
# str_adjs = ' '.join([str(elem) for elem in S_adj])  # convert list back to string for word cloud
#
#
# wc = WordCloud(max_words=50, max_font_size=60, background_color='white', colormap="tab20b").generate(str_adjs) # font size is important
# plt.imshow(wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# -------------------------------------------------------------------------------------------------------------------

# def clean(post):
#     words = [word for word in post if len(word) > 3 and word.isalpha()]
#     return words
#
# myerstype['New'] = myerstype['New'].apply(lambda x: clean(x))
# mydata = myerstype[['New', 'IE', 'NS', 'TF', 'JP']]

# # lexical diversity (new add)
def lexical_diversity(text): # text is a word list
    return len(set(text)) / len(text)

# # lexical diversity
# for index in mydata.NS.unique():   # did IE, NS, TF
#     data = mydata[mydata['NS'] == index]
#     CorpusInType = [word for post in data['New'] for word in post]
#     offset = np.arange(0, len(CorpusInType), 5000)
#     ld_list = []
#     for i in offset:
#         chunk = CorpusInType[i: i + 5000]
#         ld = lexical_diversity(chunk)
#         ld_list.append(ld)
#     plt.plot(offset, ld_list, label = index)
# plt.legend()
# plt.show()





