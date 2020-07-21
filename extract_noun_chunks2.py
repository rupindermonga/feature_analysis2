import spacy
import sys
import glob
import json
import os
import csv
import time, json
import pandas as pd
from itertools import islice

nlp = spacy.load('en_core_web_sm', disable = ['ner'])

#inputs
#path - path to the category
# threshold_num_review   - process only 1000 reviews overall
# threshold_num_review_per_product - process only 100 reviews per product
# threshold_review_words - process only reviews with at least 100 words
# min_freq - minimum overall freq for noun_chunk
#no_sentences - number of sentences for the important noun_chunks


# in_slug = sys.argv[1]

def extractNounChunk(path, threshold_num_review, threshold_num_review_per_product, threshold_review_words, min_freq, no_sentences):
    

    start_time = time.time()
    path_to_json = path
    texts = []
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    final_dict = {}
    count_review = 0
    sentence_dict = {}
    sentence_out = csv.writer(open(os.path.join(path_to_json, "sentence.csv"),'w'))
    overAllFreq = {}
    for filename in json_files:        
        if count_review >= threshold_num_review:
            break
        with open(os.path.join(path_to_json, filename)) as f:
            data = json.load(f)
            count_review_product = 0
            for r in data['reviews']:
                if count_review_product >= threshold_num_review_per_product:
                    break
                if count_review >= threshold_num_review:
                    break
                text = r.get('reviewText','').strip()
                if len(text.split()) < threshold_review_words:
                    continue
                texts.append(text)
                count_review_product += 1
                count_review += 1
        docs = nlp.pipe(texts)
        freq = {}
        
    
        
        for doc in docs:
            lemma_set = set()    

            for chunk in doc.noun_chunks:
                tokens = [t.lemma_ for t in chunk if not t.is_stop and t.lemma_ not in ['-PRON-']]
                lemma = ' '.join(tokens).strip()
                if lemma not in ['-PRON-','']:
                    freq[lemma] = freq.get(lemma,0) + 1
                    overAllFreq[lemma] = overAllFreq.get(lemma,0) + 1
                    lemma_set.add(lemma)
                    
            for eachLemma in lemma_set:
                if eachLemma not in sentence_dict.keys():
                    sentence_dict[eachLemma] = [doc]
                    sentence_out.writerow([eachLemma, doc])
                else:
                    if len(sentence_dict[eachLemma])< no_sentences:
                        sentence_dict[eachLemma].append(doc)
                        sentence_out.writerow([eachLemma,doc])

        final_dict[filename] = freq
        

    out = csv.writer(open(os.path.join(path_to_json, "Final.csv"),'w'))
    out.writerow(['noun_chunk', 'product', 'freq', "overall_freq"])

    for k, v in final_dict.items():
        sort = sorted(v, key=lambda k1:v[k1], reverse=True)
        for s in sort:
            if overAllFreq[s] > min_freq:
                out.writerow([s,k,v[s],overAllFreq[s]])
    
    
    # updated_data = pd.read_csv(os.path.join(path_to_json, "intermediate.csv"))
    sentences_data = pd.read_csv(os.path.join(path_to_json, "sentence.csv"))
    sentences_data.columns = ['noun_chunk', 'sentences']

    # updated_data.columns = ['noun_chunk', 'product', 'freq']

    # updated_data['overall_freq'] = updated_data.groupby('noun_chunk').freq.transform('sum')

    # updated_data["Rank"] = updated_data["overall_freq"].rank( ascending = False, method = "dense")

    # updated_data.sort_values("Rank", inplace = True)

    # final_data = updated_data.drop(updated_data[updated_data.overall_freq<=min_freq].index)
    
    # final_data.to_csv(os.path.join(path_to_json, "final.csv"))
    # os.remove(os.path.join(path_to_json, "intermediate.csv"))
    
    # df = sentences_data[sentences_data.noun_chunk.isin(final_data.noun_chunk)]
    # df.to_csv(os.path.join(path_to_json, "final_sentence.csv"))
    # os.remove(os.path.join(path_to_json, "sentence.csv"))
    return time.time()-start_time

f = extractNounChunk('/media/rupinder/C49A5A1B9A5A0A76/Users/Rupinder/Desktop/BVR/New/feature_analysis-master/bvrblackbox_workspace/humidifier',
                     500, 100,50, 100, 10)

print(f)

#inputs
#path - path to the category
# threshold_num_review   - process only 1000 reviews overall
# threshold_num_review_per_product - process only 100 reviews per product
# threshold_review_words - process only reviews with at least 100 words
# min_freq - minimum overall freq for noun_chunk
#no_sentences - number of sentences for the important noun_chunks