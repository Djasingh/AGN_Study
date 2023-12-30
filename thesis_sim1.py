#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import traceback
import spacy
from tqdm import tqdm
import pickle
import itertools
import re
import argparse


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

def thesis_similarity(thesis_df, thresold=0.85, index=0, size=50000):
    thesis_dict={}
    count=0
    try:
        list_of_pairs=((x, y) for i, x in enumerate(thesis_df['dc.title[]']) for j, y in enumerate(thesis_df['dc.title[]'][index:index+size]) if (i > j+index))
        for thesis1, thesis2 in tqdm(list_of_pairs, total = thesis_df['dc.title[]'].shape[0]*size):
        #for thesis1, thesis2 in tqdm(itertools.combinations(thesis_df['dc.title[]'], 2), total=(thesis_df.shape[0]*(thesis_df.shape[0]-1))/2):
            thesis11 = str(thesis1).split(" ")
            thesis21 = str(thesis2).split(" ")
            common_len = len(set(thesis11).intersection(thesis21))/ max(len(thesis11),len(thesis21))
            if common_len > 0.50: 
                #print(common_len)
                score = nlp(thesis1).similarity(nlp(thesis2))
                if score > thresold:
                    tid1=pd.unique(thesis_df[thesis_df['dc.title[]']==thesis1]['thesisId'])
                    tid2=pd.unique(thesis_df[thesis_df['dc.title[]']==thesis2]['thesisId'])
                    thesis_dict[(tuple(tid1),tuple(tid2),score)]=(thesis1, thesis2)
                    count+=1
            else:
                continue
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        print('No.of similar thesis :'+str(count))
        save_obj(thesis_dict, "./similar_thesis/similar_thesis_"+str(index)+"_"+str(index+size))
        #print(thesis_dict)
    return


if __name__ == "__main__":
    ment =  pd.read_csv("index_files4/final_mod_ment_w_baseline_gen4.csv")
    #ment = ment[ment['researcherId']==186818].copy()
    ment['dc.title[]'].fillna("Not_Appl",inplace=True)
    parser = argparse.ArgumentParser(description='Thesis Similarity')
    parser.add_argument('--index', default = 0, type=int, help="Enter the value to start from")
    parser.add_argument('--size', default = ment.shape[0], type=int, help="Batch size")
    args = parser.parse_args()
    thesis_similarity(ment, 0.85, args.index, args.size)




