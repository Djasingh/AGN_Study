#!/usr/bin/env python
# coding: utf-8
import re
import traceback
import pandas as pd
import numpy as np
from collections import Counter
from cdifflib import CSequenceMatcher
import random
from tqdm import tqdm
import numpy as np
import warnings
import spacy
import sys

nlp=spacy.load('en_core_web_lg')
warnings.filterwarnings("ignore")
tqdm.pandas()

def lcs(name1, name2):
    match = CSequenceMatcher(None, name1, name2).find_longest_match(0, len(name1), 0, len(name2))
    common_subs=name1[match.a: match.a + match.size]
    name1=re.sub(common_subs," ",name1)
    name2=re.sub(common_subs," ",name2)
    return name1,name2

def spacy_similarity(name1, name2):
    n1, n2 = lcs(name1, name2)
    score=(nlp(n1).similarity(nlp(n2)))
    if score == 0:
        score=(nlp(name1).similarity(nlp(name2)))
    return score > 0.85

def spacy_similarity1(name1, name2):
    n1, n2 = lcs(name1, name2)
    score=(nlp(n1).similarity(nlp(n2)))
    if score == 0:
        score=(nlp(name1).similarity(nlp(name2)))
    return score 


def index_advisor(row, df):
    thresold = 0.95
    try:
        #min_date=df[df['advisor_name']==row['advisor']
        match_stud = df[(df['researcherId']==row['advisorId'])]
        if len(match_stud) == 1:
            return np.nan
        elif len(match_stud) > 1 :
            return np.nan
        else:
            match = df[(df['researcher_name']==row['advisor_name'])]
            if len(match) == 0:
                return np.nan
            else:
                tmp1 = match[match['DepartmentId']==row['DepartmentId']]
                if len(tmp1)==0:
                    tmp1 = match[match['publisher_dept'].apply(spacy_similarity,args=(row['publisher_dept'],))]
                if len(tmp1)==1:
                    return tmp1['researcherId'].values[0] if tmp1['date_submitted'].values[0] < min(row['date_submitted']) else np.nan   
                elif len(tmp1)==0:
                    tmp4 = match[match['instituteId']==row['instituteId']]                 
                    if len(tmp4) == 1:
                        score = spacy_similarity1(tmp4['publisher_dept'].values[0] , row["publisher_dept"])
                        return tmp4['researcherId'].values[0] if (tmp4['date_submitted'].values[0] < min(row['date_submitted'])) and score > 0.55  else np.nan
                    elif len(tmp4) == 0:
                        tmp5=match[match['dc.subject.ddc'].str.contains(random.choice(row['dc.subject.ddc']),na=False)]
                        if len(tmp5) == 1:
                            score = spacy_similarity1(tmp5['publisher_dept'].values[0] , row["publisher_dept"])
                            return tmp5['researcherId'].values[0] if (tmp5['date_submitted'].values[0] < min(row['date_submitted'])) and score > 0.70 else np.nan
                        elif len(tmp5)==0:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted']) 
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in match.iterrows()  ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0] if max_val[0][1] > thresold  else np.nan
                        else:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp5.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0] if max_val[0][1] > thresold  else np.nan 
                    else:
                        tmp7 = tmp4[tmp4['dc.subject.ddc'].str.contains(random.choice(row['dc.subject.ddc']),na=False)]
                        if len(tmp7)==1:
                            return tmp7['researcherId'].values[0] if tmp7['date_submitted'].values[0]  < min(row['date_submitted']) else np.nan
                        elif len(tmp7)==0:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp4.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0] if max_val[0][1] > thresold  else np.nan
                        else:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp7.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0] if max_val[0][1] > thresold  else np.nan

                else:
                    tmp2=tmp1[tmp1['instituteId']==row['instituteId']] 
                    if len(tmp2)==1:
                        return tmp2['researcherId'].values[0] if tmp2['date_submitted'].values[0] < min(row['date_submitted']) else np.nan
                    elif len(tmp2)==0:
                        tmp6 = tmp1[tmp1['dc.subject.ddc'].str.contains(random.choice(row['dc.subject.ddc']),na=False)]
                        if len(tmp6)==1:
                            return tmp6['researcherId'].values[0] if tmp6['date_submitted'].values[0]  < min(row['date_submitted']) else np.nan
                        elif len(tmp6)==0:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp1.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0] if max_val[0][1] > thresold  else np.nan
                        else:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp6.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            id_lst=[val[0] for val in max_val if val[1] > thresold]
                            if len(id_lst)==1:
                                return id_lst[0]
                            elif len(id_lst)==0:
                                return np.nan
                            else:
                                values1=[(row1['researcherId'],nlp(random.choice(row['title'])).similarity(nlp(row1['title'])),row1['date_submitted'])
                                    if row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp6.iterrows()]
                                max_val1=sorted(values1,key=lambda x: x[1],reverse=True)
                                return max_val1[0][0]

                    
                    else:
                        tmp3 = tmp2[tmp2['dc.subject.ddc'].str.contains(random.choice(row['dc.subject.ddc']), na=False)]
                        if len(tmp3)==1:
                            return tmp3['researcherId'].values[0] if tmp3['date_submitted'].values[0]  < min(row['date_submitted']) else np.nan
                        elif len(tmp3)==0:
                            values=[(row1['researcherId'],nlp(random.choice(row['dc.subject.ddc'])).similarity(nlp(row1['dc.subject.ddc'])),row1['date_submitted'])  
                                    if row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp2.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            id_lst = [val[0] for val in max_val if val[1] > thresold]
                            if len(id_lst) == 1 :
                                return id_lst[0]
                            else:
                                values1=[(row1['researcherId'],nlp(random.choice(row['title'])).similarity(nlp(row1['title'])),row1['date_submitted'])
                                    if row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp2.iterrows() ]
                                max_val1=sorted(values1,key=lambda x: x[1],reverse=True)
                                return max_val1[0][0] # dept same inst same 
                            
                        else:
                            values=[(row1['researcherId'],nlp(random.choice(row['title'])).similarity(nlp(row1['title'])),row1['date_submitted'])  
                                    if  row1['date_submitted'] < min(row['date_submitted']) else (np.nan,0,0)  for index, row1  in tmp3.iterrows() ]
                            max_val=sorted(values,key=lambda x: x[1],reverse=True)
                            return max_val[0][0]  #dept same , inst same , ddc subject same
    except Exception as e:
        print(e)
        print(row.values)
        traceback.print_exc()
        sys.exit("error occured")
    return np.nan

if __name__ == "__main__":

    ment = pd.read_csv('./base_data/sodhganga_mentorship_dept_rev.csv', sep = ",")
    #ment.drop(columns=['advId_1','studId_1'], inplace=True)
    #ment=ment.iloc[1:1000,:].copy()
    ment['instituteId'].fillna("I00000",inplace=True)
    ment['date_submitted'].fillna(value=ment['dc.date.awarded'], inplace=True)
    ment['DepartmentId'].fillna("D00000",inplace=True)
    ment['dc.subject.ddc']= ment['dc.subject.ddc'].fillna(value=ment['publisher_dept'])
    #ment['publisher_dept']= ment['publisher_dept'].fillna(value=ment['publisher_institution'])
    ment['dc.subject.ddc']= ment['dc.subject.ddc'].replace(r"\|?\d+::",",", regex=True).str.strip(",")
    ment['date_submitted'] = pd.to_datetime(ment['date_submitted'],errors = 'coerce')
    ment['advisor_inst_dept']=ment['advisor_name']+"@"+ment['instituteId']+"@"+ment['DepartmentId']
    ment['stud_inst_dept']=ment['researcher_name']+"@"+ment['instituteId']+"@"+ment['DepartmentId']
    index1=pd.unique(ment[['advisor_inst_dept', 'stud_inst_dept']].values.ravel('K'))
    name_index=pd.DataFrame({'names_inst_dept':index1})
    name_index['rid']=name_index.index
    ment["advisorId"]=ment['advisor_inst_dept'].map(name_index.set_index('names_inst_dept')['rid'])
    ment["researcherId"]=ment['stud_inst_dept'].map(name_index.set_index('names_inst_dept')['rid'])


    advisor_detail=ment.groupby(['advisorId','advisor_name','instituteId','DepartmentId'], as_index=False)[['publisher_dept','publisher_institution','date_submitted','title','dc.subject.ddc']].agg(lambda x: list(x))
    advisor_detail['publisher_institution']=advisor_detail['publisher_institution'].apply(lambda x: x[0])
    advisor_detail['publisher_dept']=advisor_detail['publisher_dept'].apply(lambda x:x[0])
    advisor_detail["new_advId_1"] = advisor_detail.progress_apply(index_advisor, args=(ment,),axis=1)
    ment.to_csv("dataset_v4/mod_ment_w_baseline_gen4.csv", index=False)
    advisor_detail.to_csv("dataset_v4/advisorid_cor_studentid_gen4.csv", index=False)

