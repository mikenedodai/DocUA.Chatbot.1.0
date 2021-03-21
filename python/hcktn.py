#%%
from node2vec import Node2Vec
from scipy.spatial import distance

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import csv
import re
import pandas as pd
import os
from bs4 import BeautifulSoup
import re

from tqdm.autonotebook import tqdm

import numpy as np
pd.set_option('display.max_columns', 500)
print(os.listdir())
# %%
body_symptom = pd.read_csv("body_symptom.csv", sep=',', error_bad_lines=False)
#df = pd.read_csv("disease.csv", sep=',',  error_bad_lines=False)
#%%
#pd.read_csv("specialty.csv", error_bad_lines=False,escapechar='\\')


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

col_names = ["id", "group_id", "name", "alias", "gender",
             "deleted", "description", "about", "popularity", "active"]
col_names1 = ['id', 'name', 'alias', 'seo_title', 'keywords', 'description',
              'content', 'date_create', 'date_update', 'deleted', 'case_parental',
              'case_parental_plural', 'case_plural', 'top', 'popular', 'child',
              'to_home', 'child_doctor', 'to_home_doctor', 'name_local']
with open('disease.csv', 'r', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, escapechar='\\')
    ss = list(spamreader)

diasese = pd.DataFrame(ss, columns=col_names)
diasese = diasese.iloc[1:]

with open('specialty.csv', 'r', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, escapechar='\\')
    ss = list(spamreader)

specialty = pd.DataFrame(ss, columns=col_names1)
specialty = diasese.iloc[1:]

#%%   
#diasese = pd.read_csv("clean_d.csv", sep=',',  error_bad_lines=False, names=col_names)
disease_symptom = pd.read_csv("disease_symptom.csv", sep=',', error_bad_lines=False)
doctor = pd.read_csv("doctor.csv", sep=',', error_bad_lines=False)
doctor_diseases = pd.read_csv("doctor_diseases.csv", sep=',', error_bad_lines=False)
doc_spec = pd.read_csv("doc_spec.csv", sep=',', error_bad_lines=False)
hackathon_order = pd.read_csv(
    "hackathon_order.csv", sep=',', error_bad_lines=False)
#specialty = pd.read_csv("specialty.csv", sep=',', error_bad_lines=False)
symptom = pd.read_csv("symptom.csv", sep=',', error_bad_lines=False)

diasese['description'] = diasese['description'].apply(lambda x: cleanhtml(x))
diasese['about'] = diasese['about'].apply(lambda x: cleanhtml(x))

specialty['description'] = specialty['description'].apply(lambda x: cleanhtml(x))
specialty['about'] = specialty['about'].apply(lambda x: cleanhtml(x))
# %%
body_symptom.head(10)
# %%
diasese.head(50)
# %%
disease_symptom.head(10)
# %%
doctor.head(10)
# %%
doctor_diseases.head(10)
# %%
doc_spec.head(10)
# %%
hackathon_order.head(10)
# %%
specialty.head(10)
# %%
symptom.head(10)
# %%
'''embedding = TransformerDocumentEmbeddings("bert-base-multilingual-cased")
sentence = Sentence(hackathon_order['comment'].iloc[1])
embedding_vector = embedding.embed(sentence)
v1 = sentence.embedding.detach().cpu().numpy()


sentence = Sentence(hackathon_order['comment'].iloc[2])
embedding_vector = embedding.embed(sentence)
v2 = sentence.embedding.detach().cpu().numpy()'''

# %%
hackathon_order['specialty_id'].value_counts()
# %%
'''sympt_vec = []
embedding = TransformerDocumentEmbeddings("bert-base-multilingual-cased")

for i in tqdm(symptom['name'].values):
    sentence = Sentence(i)
    embedding_vector = embedding.embed(sentence)
    v = sentence.embedding.detach().cpu().numpy()
    sympt_vec.append(v)'''


# %%
symptom_list = symptom['name'].values
symptom_list = [i.lower() for i in symptom_list]
# %%
comments = hackathon_order['comment'].values
#%%
symptom_list[:20]
# %%
'''ner_data = []
with open("train.txt", "w", encoding="utf-8") as f:
    for c in tqdm(comments):
        if len(ner_data) <= 5000:
            try:
                for i in symptom_list:
                    if (c.lower().find(i) != -1):
                        target_split = i.split()
                        target_len = len(target_split)
                        counter = 0
                        for j in c.split():
                            if j in target_split:
                                if counter == 0:
                                    f.write(f"{j} B-SYM")
                                    f.write("\n")
                                    counter += 1
                                else:
                                    f.write(f"{j} I-SYM")
                                    f.write("\n")
                            
                            else:
                                f.write(f"{j} O")
                                f.write("\n")
                        f.write("\n")
                        ner_data.append(c)
            except Exception as e:
                print(e)

    for i in symptom_list:
        counter = 0
        for j in i.split():
            if counter == 0:
                f.write(f"{j} B-SYM")
                f.write("\n")
                counter += 1
            else:
                f.write(f"{j} I-SYM")
                f.write("\n")
        f.write("\n")

ner_data = []
with open("valid.txt", "w", encoding="utf-8") as f:
    for idx, c in tqdm(enumerate(comments)):
        if idx >= 20_000:
            if len(ner_data) <= 1000:
                try:
                    for i in symptom_list:
                        if (c.lower().find(i) != -1):
                            target_split = i.split()
                            target_len = len(target_split)
                            counter = 0
                            for j in c.split():
                                if j in target_split:
                                    if counter == 0:
                                        f.write(f"{j} B-SYM")
                                        f.write("\n")
                                        counter += 1
                                    else:
                                        f.write(f"{j} I-SYM")
                                        f.write("\n")

                                else:
                                    f.write(f"{j} O")
                                    f.write("\n")
                            f.write("\n")
                            ner_data.append(c)

                except Exception as e:
                    print(e)'''
         
# %%
'''def load_data(filename: str):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = [line[:-1].split() for line in file]
    samples, start = [], 0
    for end, parts in enumerate(lines):
        if not parts:
            sample = [(token, tag.split('-')[-1]) 
                        for token, tag in lines[start:end]]
            samples.append(sample)
            start = end + 1
    if start < end:
        samples.append(lines[start:end])
    return samples

train_samples = load_data('train.txt')
val_samples = load_data('valid.txt')
samples = train_samples + val_samples
schema = ['_'] + sorted({tag for sentence in samples 
                             for _, tag in sentence})'''
# %%
'''symptom_vec = []
embedding = TransformerDocumentEmbeddings("bert-base-multilingual-cased")

for i in tqdm(symptom['name'].values):
    sentence = Sentence(i)
    embedding_vector = embedding.embed(sentence)   
    symptom_vec.append(sentence.embedding.detach().cpu().numpy())'''



# %%
import pandas as pd
import re
import numpy as np
sympt= pd.read_csv("symptom.csv")
body = pd.read_csv("body.csv")
sympt.head(20)
#%%
body.head(10)
sympt.fillna(999, inplace=True)
body.fillna(999, inplace=True)
#%%
body.head(10)
# %%
body['id'].value_counts()
# %%
body_dict = dict(zip(body['id'], body['name']))
#%%
sympt["body_name"] = sympt["body_id"].map(body_dict)
# %%
f_sympt = sympt.loc[sympt['gender'] != 'male']
m_sympt = sympt.loc[sympt['gender'] != 'female']

# %%
sympt.tail()
# %%
from tqdm.autonotebook import tqdm

sympt_list = sympt['name'].values
sympt_list = [i.lower() for i in sympt_list]

s = []
for c in tqdm(diasese['description'].values):
    try:
        for i in sympt_list:
            c = re.sub('[A-Za-z0-9]+', '', c)
            f = re.findall(i, c.lower())
            if len(f) >=1:
                tmp=[]
                for ii in f:
                    tmp.append(ii)
                s.append(tmp)
    except Exception as e:
        print(e)
# %%
print(len(s))
s[:30]
# %%
sympt_list[0]
# %%
disease_symptom.head()
#%%
symptom.head()
# %%
symptom_dict = dict(zip(symptom['id'], symptom['name']))
d_dict = dict(zip(diasese['id'], diasese['name']))
disease_symptom["symp_name"] = disease_symptom["symptom_id"].map(symptom_dict)
disease_symptom["d_name"] = disease_symptom["disease_id"].map(d_dict)

# %%
disease_symptom.head()
# %%
dias = disease_symptom['disease_id'].unique()
# %%
disease_symptom.loc[disease_symptom['disease_id'] == 5]

 # %%
from collections import Counter
import networkx as nx
# %%
G = nx.Graph()
for idx, row in disease_symptom.iterrows():
    G.add_node(row['symp_name'])

for idx, row in disease_symptom.iterrows():
    G.add_edge(row['disease_id'], row['symp_name'])
# %%
plt.figure(figsize=[50, 20])

params = {
    'edge_color': '#FFDEA2',
    'width': 1,
    'font_weight': 'regular'
}
for component in list(nx.connected_components(G)):
    if len(component) < 8:
        for node in component:
            G.remove_node(node)
nx.draw_networkx(G, **params)  
# %%

n2v = Node2Vec(G, dimensions=3, num_walks=100, workers=4)
node_model = n2v.fit(size=3, window=2, seed=42, iter=1, sg=1)

# %%
node_model.most_similar(['Онемение', 'Отечность пальца'], topn=15)
# %%
print(diasese['name'].loc[diasese['id'] == str(559)].values)
disease_symptom.loc[disease_symptom['disease_id'] == 559].head(50)
# %%
diasese['name'].loc[diasese['id'] == str(52)].values
# %%
