#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import traceback
import spacy
from tqdm import tqdm
import pickle
import itertools
import re


nlp=spacy.load('en_core_web_lg')
tqdm.pandas()


def thesis_pre_processing(name):
    name = str(name)
    name = re.sub(" +","",name)
    #name = re.sub(regex,"", name)
    #val = re.sub('[^A-Za-z]+', '', val)
    return name

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def thesis_similarity(thesis_df, thresold=0.85):
    thesis_dict={}
    count=0
    try:
        for thesis1, thesis2 in tqdm(itertools.combinations(thesis_df['dc.title[]'], 2), total=(thesis_df.shape[0]*(thesis_df.shape[0]-1))/2):
            score = nlp(thesis1).similarity(nlp(thesis2))
            if score > thresold:
                tid1=pd.unique(thesis_df[thesis_df['dc.title[]']==thesis1]['thesisId'])
                tid2=pd.unique(thesis_df[thesis_df['dc.title[]']==thesis2]['thesisId'])
                thesis_dict[(tuple(tid1),tuple(tid2),score)]=(thesis1, thesis2)
                count+=1
    except Exception as e:
        print(e)
        traceback.print_exception()
    finally:
        print('No.of similar thesis :'+str(count))
        save_obj(thesis_dict, "./similar_thesis/similar_thesis_"+str(count))
        #print(thesis_dict)
    return



ment =  pd.read_csv("./index_files4/final_mod_ment_w_baseline_gen4.csv")
#ment2=ment[ment['researcherId']==186818].copy()
thesis_similarity(ment, 0.85)




