import spacy
import sys
import glob
import json
import os
import csv
import time, json
import pandas as pd

nlp = spacy.load('en_core_web_sm')

# in_slug = sys.argv[1]

def extractNounChunk(path, no_of_products, length_of_review, min_freq):
    start_time = time.time()
    path_to_json = path
    texts = []
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    final_dict = {}
    new_count = 0
    count = 0
    for filename in json_files:        
        with open(os.path.join(path_to_json, filename)) as f:
            data = json.load(f)
            try:
                if len(data['reviews']) < length_of_review:
                    break
                else:
                    count += 1
                    if count <= no_of_products:
                        for r in data['reviews']:
                            text = r.get('reviewText','').strip()
                            if text != '':
                                texts.append(text)
                    else:
                        break
            except:
                pass
        docs = nlp.pipe(texts)
        freq = {}

        for doc in docs:
            for chunk in doc.noun_chunks:
                tokens = [t.lemma_ for t in chunk if not t.is_stop and t.lemma_ not in ['-PRON-']]
                lemma = ' '.join(tokens).strip()
                if lemma not in ['-PRON-','']:
                    freq[lemma] = freq.get(lemma,0) + 1
        final_dict[filename] = freq
        new_count += 1

    out = csv.writer(open(os.path.join(path_to_json, "intermediate.csv"),'w'))

    for k, v in final_dict.items():
        sort = sorted(v, key=lambda k1:v[k1], reverse=True)
        for s in sort:
            out.writerow([s,k,v[s]])

    updated_data = pd.read_csv(os.path.join(path_to_json, "intermediate.csv"))

    updated_data.columns = ['noun_chunk', 'product', 'freq']

    updated_data['overall_freq'] = updated_data.groupby('noun_chunk').freq.transform('sum')

    updated_data["Rank"] = updated_data["overall_freq"].rank( ascending = False, method = "dense")

    updated_data.sort_values("Rank", inplace = True)

    final_data = updated_data.drop(updated_data[updated_data.overall_freq<=min_freq].index)
    final_data = final_data.dropna(subset = ["overall_freq"], inplace = True)
    final_data.to_csv(os.path.join(path_to_json, "final.csv"))
    os.remove(os.path.join(path_to_json, "intermediate.csv"))
    
    return time.time()-start_time

f = extractNounChunk('/media/rupinder/C49A5A1B9A5A0A76/Users/Rupinder/Desktop/BVR/New/feature_analysis-master/bvrblackbox_workspace/coffee-grinder', 5, 50,100)

print(f)
