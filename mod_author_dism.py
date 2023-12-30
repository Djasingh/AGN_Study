#!/usr/bin/env python
# coding: utf-8

import re
import textdistance
import jellyfish
import itertools
import editdistance
import fuzzy
import pickle
import argparse
import pandas as pd
from collections import Counter
from cdifflib import CSequenceMatcher
from fuzzywuzzy import fuzz
from tqdm import tqdm
import traceback
chars=[',',";","\."]

def rm_splChar(name):
    name = str(name)
    name1 = re.sub(" +","",name)
    regex = "|".join(chars)
    name1 = re.sub(regex,"", name1)
    name2 = re.sub(regex,"", name)
    #val = re.sub('[^A-Za-z]+', '', val)
    return name1, name2

def char_dist(name1, name2):
    name1=rm_splChar(name1)
    name2=rm_splChar(name2)
    return int(Counter(name1)==Counter(name2))

def diff_lib(name1, name2):
    name1=name1.lower()
    name2=name2.lower()
    ratio=CSequenceMatcher(lambda x: x == ' ', name1, name2).ratio()
    return ratio

def jaro_winkler_score(name1, name2):
    jw_score=textdistance.jaro_winkler.normalized_similarity(name1,name2)
    return jw_score

def levenshtein_score(name1, name2):
    leven_score = textdistance.levenshtein.normalized_similarity(name1,name2)
    return leven_score

def fuzzy_nysiis(name1, name2):
    ny1=fuzzy.nysiis(name1)
    ny2=fuzzy.nysiis(name2)
    if (ny1 or ny2):
        nysiis_score = editdistance.eval(ny1, ny2)/max(len(ny1),len(ny2))
    else:
        nysiis_score = 0
    return nysiis_score

def fuzzy_DMetaphone(name1, name2):
    d1=jellyfish.metaphone(name1)
    d2=jellyfish.metaphone(name2)
    if (d1 or d2):
        meta_score = editdistance.eval(d1, d2)/max(len(d1),len(d2))
    else:
        meta_score = 0
    return meta_score

#Soundex is a phonetic algorithm 
def jellyfish_soundex(name1, name2):
    s1=jellyfish.soundex(name1)
    s2=jellyfish.soundex(name2)
    sound_score = editdistance.eval(s1,s2)/max(len(s1),len(s2))
    return sound_score

def fuzzy_wuzzy(name1, name2):
    fuzz_score=fuzz.token_set_ratio(name1, name2)/100
    return fuzz_score


def hamming_similarity(name1, name2):
    h_score=textdistance.hamming.normalized_similarity(name1,name2)
    return h_score

def jaccard_similarity(name1, name2):
    j_score=textdistance.jaccard.normalized_similarity(name1,name2)
    return j_score

def cosine_similarity(name1, name2):
    c_score=textdistance.jaccard.normalized_similarity(name1,name2)
    return c_score


def damerau_levenshtein_similarity(name1, name2):
    dl_score=textdistance.damerau_levenshtein.normalized_similarity(name1,name2)
    return dl_score


def sorensen_dice_similarity(name1, name2):
    sd_score=textdistance.sorensen_dice.normalized_similarity(name1,name2)
    return sd_score


def calculate_feats(name1, name2):
    sim_score=[]
    #sim_score.append(char_dist(name1, name2))
    sim_score.append(diff_lib(name1,name2))
    sim_score.append(jaro_winkler_score(name1, name2))
    sim_score.append(levenshtein_score(name1, name2))
    sim_score.append(1-fuzzy_nysiis(name1, name2))#distance
    sim_score.append(1-fuzzy_DMetaphone(name1, name2))#distance
    sim_score.append(1-jellyfish_soundex(name1,name2))#distance
    sim_score.append(fuzzy_wuzzy(name1, name2))
    sim_score.append(hamming_similarity(name1, name2))
    sim_score.append(jaccard_similarity(name1, name2))
    sim_score.append(cosine_similarity(name1, name2))
    sim_score.append(damerau_levenshtein_similarity(name1, name2))
    sim_score.append(sorensen_dice_similarity(name1, name2))
    return sim_score


#problem 1 : different names reffering to same person--Name Linking (Problem 2 : Name Resolution Problem)


def lcs(name1, name2):
    match = CSequenceMatcher(None, name1, name2).find_longest_match(0, len(name1), 0, len(name2))
    common_subs=name1[match.a: match.a + match.size]
    name1=re.sub(re.escape(common_subs),"",name1)
    name2=re.sub(re.escape(common_subs),"",name2)
    return name1,name2


def find_similar_names(org_df, names_df, index=0, size=50000) : 
    similar_names = {}
    keep_cnt = 0 # org_df.shape[0]*index
    count = 0
    try:
        #list_of_pairs=((x, y) for i, x in enumerate(names_df['r_names']) for j, y in enumerate(names_df['r_names'][index:index+size]) if (i > j+index))
        #for name1, name2 in tqdm(list_of_pairs, total = names_df['r_names'].shape[0]*size):
        for name1, name2 in tqdm(itertools.combinations(names_df['r_names'], 2), total=(names_df.shape[0]*(names_df.shape[0]-1))/2):
            keep_cnt += 1
            n1, n10 = rm_splChar(name1)
            n2, n20 = rm_splChar(name2)
            check=len(set(n1).intersection(n2))/max(len(n1),len(n2))
            if check > 0.70:
                n11, n21 = lcs(n1, n2)
                n101, n201 = lcs(n10,n20)
                if (n11 or n21): 
                    vec1 = calculate_feats(n11, n21)
                    vec2 = calculate_feats(n11.lower(), n21.lower())
                else:
                    vec1 = calculate_feats(n1, n2)
                    vec2 = calculate_feats(n1.lower(), n2.lower())
                if (n101 or n201): 
                    vec3=calculate_feats(n101, n201)
                    vec4=calculate_feats(n101.lower(), n201.lower())
                else:
                    vec3=calculate_feats(n10, n20)
                    vec4=calculate_feats(n10.lower(), n20.lower())
                if (sum(vec1) > 10) or (sum(vec2) > 10) or (sum(vec3) > 10) or (sum(vec4) > 10) :
                    inst1 = org_df[(org_df['dc.contributor.advisor[]'] == name1) | (org_df['dc.creator.researcher[]'] == name1)]["dc.publisher.institution[]"]
                    inst2 = org_df[(org_df['dc.contributor.advisor[]'] == name2) | (org_df['dc.creator.researcher[]'] == name2)]["dc.publisher.institution[]"]
                    dept1 = dataset[(dataset['dc.contributor.advisor[]'] == name1) | (dataset['dc.creator.researcher[]'] == name1)]["dc.publisher.department[]"]
                    dept2 = dataset[(dataset['dc.contributor.advisor[]'] == name2) | (dataset['dc.creator.researcher[]'] == name2)]["dc.publisher.department[]"]
                    common_inst=set(inst1).intersection(inst2)
                    common_dept=set(dept1).intersection(dept2)
                    if common_inst and common_dept:
                        similar_names[keep_cnt]=(name1,name2)
                    count += 1

    except KeyboardInterrupt:
        save_obj(similar_names, str(keep_cnt))
        print("Pairs having similarity value greater than 10: "+str(count))
        return similar_names

    except ZeroDivisionError:
        print('divided by zero error')
        save_obj(similar_names, str(keep_cnt))
        print("Pairs having similarity value greater than 10: "+str(count))
        return similar_names
    except Exception as e:
        save_obj(similar_names, str(keep_cnt))
        print("Pairs having similarity value greater than 10: "+str(count))
        print(traceback.print_exc())
        return similar_names



    print("Pairs having similarity value greater than 10: "+str(count))
    save_obj(similar_names, str(index)+"_"+str(size))
    #print(similar_names)
    print("*"*30+'Similar name completed'+"*"*30)
    return similar_names


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

#sim_names=find_similar_names(dataset, name_index, 0, 10, 0)


def gen_newIndex(org_dataset, names, index, size) : 
    similar_n = find_similar_names(org_dataset, names, index, size) 
    for key in similar_n:
        index1 = names[names['r_names'].isin(similar_n[key])]['rid'].values
        names.loc[names['r_names'].isin(similar_n[key]),'rid'] = min(index1)
    #org_dataset["advId"] = org_dataset['dc.contributor.advisor[]'].map(names.set_index('r_names')['rid'])
    #org_dataset["studId"] = org_dataset['dc.creator.researcher[]'].map(names.set_index('r_names')['rid'])
    names.to_csv('mod_researcher_index'+str(index)+'.csv', sep=",", index=False)
    #org_dataset.to_csv('mod_sodhganaga_dataset.csv', sep = ",", index = False)
    print('Done')


if __name__ == "__main__":
    dataset=pd.read_csv("./outputFile_updated.csv",sep=",")
    parser = argparse.ArgumentParser(description='Author name disambiguation')
    parser.add_argument('--index', default = 0, type=int, help="Enter the value to start from") 
    parser.add_argument('--size', default = dataset.shape[0], type=int, help="Batch size")  
    args = parser.parse_args()
    
    res_names = pd.unique(dataset[['dc.contributor.advisor[]', 'dc.creator.researcher[]']].values.ravel('K'))
    
    # Intial indexing names considering exact match
    name_index = pd.DataFrame({'r_names':res_names})
    name_index['rid'] = name_index.index
    #dataset["advisorId"] = dataset['dc.contributor.advisor[]'].map(name_index.set_index('r_names')['rid'])
    #dataset["researcherId"] = dataset['dc.creator.researcher[]'].map(name_index.set_index('r_names')['rid'])
    
    #Creating new index
    gen_newIndex(dataset, name_index, args.index, args.size)



