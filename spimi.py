import nltk
import csv
import os
import re
import json
import math
import collections
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopwords_list = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

class LoadData:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(self.filename, delimiter=',', on_bad_lines='skip')

    def get_data(self):
        return self.data

class SPIMI:
    def __init__(self, data):
        self.block_limit = 2000
        self.data = data
        self.data = self.data[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName']]

    def preprocess(self, row):
        object = {}
        id = row[0]
        doc = ' '.join(str(item) for item in row[1:])
        tokens = nltk.word_tokenize(doc)
        texto_filtrado = [word for word in tokens if not word in stopwords_list and re.match("^[a-zA-Z]+$", word)]
        texto_filtrado = [stemmer.stem(w) for w in texto_filtrado]
        object["id"] = id
        object["terms"] = texto_filtrado
        #object[terms] = [{}]
        return object
    
    def binary_merge(self, n_blocks):
        total_blocks = n_blocks
        merge_size = 1
        step = 0
        while merge_size < total_blocks:
            curr_block = 0
            while curr_block < total_blocks:
                next_block = curr_block + merge_size
                if next_block >= total_blocks:
                    next_block = curr_block
                    curr_block -= merge_size

                folder_name = 'local_indexes' if step == 0 else f'Pasada_{step}'
                with open(f'{folder_name}/block_{curr_block}.json', 'r') as file1:
                    index1 = json.load(file1)
                with open(f'{folder_name}/block_{next_block}.json', 'r') as file2:
                    index2 = json.load(file2)

                for key, value in index2.items():
                    if key in index1:
                        index1[key].extend(value)
                    else:
                        index1[key] = value

                merged_dict = index1

                os.makedirs(f'Pasada_{step+1}', exist_ok=True)
                with open(f'Pasada_{step+1}/block_{curr_block}.json', 'w') as file:
                    json.dump(merged_dict, file)

                curr_block += merge_size * 2

            if step > 0:
                shutil.rmtree(f'Pasada_{step}')

            merge_size *= 2
            step += 1

        if step > 0:
            with open(f'Pasada_{step}/block_0.json', 'r') as file:
                global_index = json.load(file)
            keys = sorted(global_index.keys())
            block_size = len(keys) // n_blocks
            for i in range(n_blocks):
                if i == n_blocks - 1:
                    block_keys = keys[i*block_size:]
                else:
                    block_keys = keys[i*block_size:(i+1)*block_size]
                block_dict = {key: global_index[key] for key in block_keys}
                os.makedirs('global_index', exist_ok=True)
                with open(f'global_index/block_{i}.json', 'w') as file:
                    json.dump(block_dict, file)

            shutil.rmtree(f'Pasada_{step}')
    
    def merge(self,folder_path):
        merged_dict = {}
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        for json_file in json_files:
            with open(os.path.join(folder_path, json_file), 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    if key in merged_dict:
                        merged_dict[key].extend(value)
                    else:
                        merged_dict[key] = value
        with open('merged.json', 'w') as file:
            json.dump(merged_dict, file)

    def spimi_invert(self):
        block_number = 0
        local_index = {}
        for index, row in self.data.iterrows():
            doc = self.preprocess(row)
            for term in doc["terms"]:
                if term not in local_index:
                    local_index[term] = [{"id": doc["id"], "tf": 1}]
                else:
                    # Comprueba si el id del documento ya existe para este término
                    doc_found = False
                    for doc_i in local_index[term]:
                        if doc_i["id"] == doc["id"]:
                            doc_i["tf"] += 1
                            doc_found = True
                            break
                    if not doc_found:
                        # Si el id del documento no se encuentra después de la iteración completa, agrega un nuevo documento
                        local_index[term].append({"id": doc["id"], "tf": 1})
                if(len(local_index) > self.block_limit):
                    sorted_dictionary = collections.OrderedDict(sorted(local_index.items()))
                    os.makedirs('local_indexes', exist_ok=True)
                    with open(f'local_indexes/block_{block_number}.json', 'w') as file:
                        json.dump(sorted_dictionary, file)
                    local_index.clear()
                    block_number += 1
        self.merge('local_indexes')


class TextRetrival:
    def __init__(self):
        with open('merged.json', 'r') as file:
            self.inverted_index = json.load(file)

    def process_query(self,query):
        tokens = nltk.word_tokenize(query)
        filtered_text = [stemmer.stem(w) for w in tokens if not w in stopwords_list and re.match("^[a-zA-Z]+$", w)]
        return filtered_text
    
    def cosine_score(self, query, k):
        processed_query = self.process_query(query)
        query_tf = {term: processed_query.count(term) for term in processed_query}
        norm_q = math.sqrt(sum(tf**2 for tf in query_tf.values()))

        document_scores = defaultdict(float)
        for term, tf_q in query_tf.items():
            if term in self.inverted_index:
                df_t = len(self.inverted_index[term])
                tfidf_t_q = math.log1p(tf_q) * math.log(len(self.inverted_index) / df_t)
                for doc in self.inverted_index[term]:
                    tfidf_t_d = math.log1p(doc["tf"]) * math.log(len(self.inverted_index) / df_t)
                    document_scores[doc["id"]] += tfidf_t_d * tfidf_t_q

        for doc_id in document_scores:
            document_scores[doc_id] /= norm_q

        sorted_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id":doc_id, "score":score} for doc_id, score in sorted_documents[:k]]
    
    def show_results(self, query, k, dataset):
        relevant_doc_ids = [doc['id'] for doc in self.cosine_score(query, k)]
        relevant_docs = dataset[dataset['id'].isin(relevant_doc_ids)]
        print(relevant_docs)
        
            

if __name__ == "__main__":
    data = LoadData('styles.csv').get_data()
    spimi = SPIMI(data)
    #spimi.spimi_invert()

    text_retrival = TextRetrival()
    query = "rns"
    k = 10
    
    text_retrival.show_results(query, k, data)