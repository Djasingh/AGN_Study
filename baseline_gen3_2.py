#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
from tqdm import tqdm
import numpy as np
import warnings
import spacy

def index_advisor(row, df):
    try:
        match = df[df['researcher_name']==row['advisor_name']]
        if len(match) == 0:
            return np.nan
        else:
            tmp1 = match[match['DepartmentId']==row['DepartmentId']]
            if len(tmp1)==1:
                return tmp1['studId_1'].values[0]
            elif len(tmp1)==0:
                tmp4 =match[match['instituteId']==row['instituteId']]                 
                if len(tmp4) == 1:
                    return tmp4['studId_1'].values[0]
                elif len(tmp4)== 0:
                    tmp5=match[match['dc.subject.ddc'].str.contains(row['dc.subject.ddc'],na=False)]
                    if len(tmp5) == 1:
                         return tmp5['studId_1'].values[0]
                    elif len(tmp5)==0:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc']))) 
                                for index, row1 in match.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                    else:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp5.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                else:
                    tmp7 = tmp4[tmp4['dc.subject.ddc'].str.contains(row['dc.subject.ddc'],na=False)]
                    if len(tmp7)==1:
                        return tmp7['studId_1'].values[0]    
                    elif len(tmp7)==0:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp4.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                    else:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp7.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan

            else:
                tmp2=tmp1[tmp1['instituteId']==row['instituteId']] 
                if len(tmp2)==1:
                    return tmp2['studId_1'].values[0]
                elif len(tmp2)==0:
                    tmp6 = tmp1[tmp1['dc.subject.ddc'].str.contains(row['dc.subject.ddc'],na=False)]
                    if len(tmp6)==1:
                        return tmp6['studId_1'].values[0]    
                    elif len(tmp6)==0:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp1.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                    else:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp6.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                
                else:
                    tmp3 = tmp2[tmp2['dc.subject.ddc'].str.contains(row['dc.subject.ddc'], na=False)]
                    if len(tmp3)==1:
                        return tmp3['studId_1'].values[0]
                    elif len(tmp3)==0:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp2.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
                    else:
                        values=[(row1['studId_1'],nlp(row['dc.subject.ddc']).similarity(nlp(row1['dc.subject.ddc'])))  
                                for index, row1 in tmp3.iterrows()]
                        max_val=sorted(values,key=lambda x: x[1],reverse=True)
                        return max_val[0][0] if max_val[0][1] > 0.95 else np.nan
    except Exception:
        print(row)
        sys.exit("error occured")
    return np.nan

if __name__ == "__main__":
    nlp=spacy.load('en_core_web_lg')
    warnings.filterwarnings("ignore")
    tqdm.pandas()

    ment = pd.read_csv('ment/Shodhganga_mentorship.csv', sep = ",")
    #ment=ment.iloc[1:1000,:].copy()
    ment['dc.subject.ddc']= ment['dc.subject.ddc'].fillna("Not_Applicable")
    ment['dc.subject.ddc']= ment['dc.subject.ddc'].replace(r"\|?\d+::",",", regex=True).str.strip(",")

    ment['stud_inst_dept']=ment['researcher_name']+"@"+ment['instituteId']+"@"+ment['DepartmentId']
    uniq_stud=pd.unique(ment['stud_inst_dept'])
    stud_index=pd.DataFrame({'stud_inst_dept':uniq_stud})
    stud_index['rid']=stud_index.index
    ment["studId_1"]=ment['stud_inst_dept'].map(stud_index.set_index('stud_inst_dept')['rid'])
    ment_cp=ment.copy()
    ment["advId_1"]=ment.progress_apply(index_advisor, args=(ment_cp,),axis=1)
    ment.to_csv("base_index_files/mod_ment_w_baseline_gen3_1.csv", index=False)

