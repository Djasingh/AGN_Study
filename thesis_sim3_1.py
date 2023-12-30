#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import traceback
import pickle
import itertools
import spacy
import textdistance
from collections import Counter
from cdifflib import CSequenceMatcher
from tqdm import tqdm


chars=[',',";","\.",":","-","_"]

def rm_splChar(name):
    name = str(name)
    name1 = re.sub(" +","",name)
    regex = "|".join(chars)
    name1 = re.sub(regex,"", name1)
    name2 = re.sub(regex,"", name)
    #val = re.sub('[^A-Za-z]+', '', val)
    #name1=name1.lower()
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

def hamming_similarity(name1, name2):
    h_score=textdistance.hamming.normalized_similarity(name1,name2)
    return h_score

def jaccard_similarity(name1, name2):
    j_score=textdistance.jaccard.normalized_similarity(name1,name2)
    return j_score

def damerau_levenshtein_similarity(name1, name2):
    dl_score=textdistance.damerau_levenshtein.normalized_similarity(name1,name2)
    return dl_score

def sorensen_dice_similarity(name1, name2):
    sd_score=textdistance.sorensen_dice.normalized_similarity(name1,name2)
    return sd_score

def cosine_similarity(name1, name2):
    c_score=textdistance.jaccard.normalized_similarity(name1,name2)
    return c_score

def calculate_feats(name1, name2):
    sim_score=[]
    #sim_score.append(char_dist(name1, name2))
    sim_score.append(diff_lib(name1,name2))
    #sim_score.append(jaro_winkler_score(name1, name2))
    sim_score.append(levenshtein_score(name1, name2))
    #sim_score.append(hamming_similarity(name1, name2))
    sim_score.append(jaccard_similarity(name1, name2))
    sim_score.append(cosine_similarity(name1, name2))
    sim_score.append(damerau_levenshtein_similarity(name1, name2))
    sim_score.append(sorensen_dice_similarity(name1, name2))
    return sim_score

def lcs(name1, name2):
    match = CSequenceMatcher(None, name1, name2).find_longest_match(0, len(name1), 0, len(name2))
    common_subs=name1[match.a: match.a + match.size]
    name1=re.sub(re.escape(common_subs),"",name1)
    name2=re.sub(re.escape(common_subs),"",name2)
    return name1,name2

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def thesis_similarity(thesis_df, thresold=5):
    unique_thesis = pd.unique(thesis_df['title']).copy()
    thesis_dict={}
    count=0
    try:
        for thesis1, thesis2 in tqdm(itertools.combinations(unique_thesis, 2), total=(unique_thesis.shape[0]*(unique_thesis.shape[0]-1))/2):
            thesis_lst1 = set(thesis1.split())
            thesis_lst2 = set(thesis2.split()) 
            score = len(thesis_lst1.intersection(thesis_lst2))/max(len(thesis_lst1),len(thesis_lst2))

            #print(thesis1+"___"+thesis2+"\n")
            #print(score)
            if score > 0.50:
                n1,n10= rm_splChar(thesis1)
                n2,n20 = rm_splChar(thesis2)
                n11, n21 = lcs(n1, n2)
                vec1=[0]
                vec2=[0]
                vec3=[0]
                vec4=[0]
                vec5=0
                #n101, n201 = lcs(n10,n20)
                if n1  and n2: 
                    vec1 = calculate_feats(n1, n2)
                    vec2 = calculate_feats(n1.lower(), n2.lower())
                if n10  and n20 :
                    vec3=calculate_feats(n10, n20)
                    vec4=calculate_feats(n10.lower(), n20.lower())
                if (n11.strip()=="" and n21.strip()==""):
                    vec5=6
                if (sum(vec1) > thresold) or (sum(vec2) > thresold) or (sum(vec3) > thresold) or (sum(vec4) > thresold) or (vec5 > thresold):
                    tid1 = thesis_df[thesis_df['title']==thesis1].copy()  #['thesisId'])                
                    tid2 = thesis_df[thesis_df['title']==thesis2].copy()  #['thesisId'])
                    dept1 = tid1['DepartmentId'].tolist()
                    dept2 = tid2['DepartmentId'].tolist()
                    inst1 = tid1['instituteId'].tolist()
                    inst2 = tid2['instituteId'].tolist()
                    comm_inst = set(inst1).intersection(inst2)
                    comm_dept = set(dept1).intersection(dept2)
                    if comm_inst and comm_dept :
                        tid11 = pd.unique(tid1['thesisId'])
                        tid21 = pd.unique(tid2['thesisId'])
                        thesis_dict[(tuple(tid11), tuple(tid21),sum(vec1),sum(vec2),sum(vec3), sum(vec4),vec5)]=(thesis1, thesis2)
                        count+=1
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        print('No.of similar thesis :'+str(count))
        save_obj(thesis_dict, folder+"similar_thesis_"+str(count))
    return

if __name__ == "__main__":
    folder="dataset_v5/v5_2/"
    tqdm.pandas()
    print("Thesis Disambiguation started:"+"\n")
    ment =  pd.read_csv(folder+"processed_sodhganga_mentorship_dept_rev_with_initial_ids.csv")
    thesis_similarity(ment)


