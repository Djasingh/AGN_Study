#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
import itertools
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import sys

def get_names_inst_dept(ment, top_5, f_col, cols=['advisor_name','publisher_institution','publisher_dept']):
    cols=[f_col]+cols
    filter_df=ment[ment[f_col].isin(top_5)][cols].copy()
    group_df=filter_df.groupby([f_col],as_index=False)[cols[1:]].agg(lambda x: "; ".join(set(x)))
    group_df[cols[1]]=group_df[cols[1]].apply(lambda x : x.split(';')[0])
    group_df.set_index(f_col,inplace=True)
    df=group_df.reindex(top_5)
    return df

def fertility_calculation(g):
    fertility={}
    for node in g.nodes:
        count=1
        successor = g.successors(node)
        #print(list(successor))
        child=g.out_degree(successor)
        count=[k for k, v in child if v > 0]
        fertility[node]=len(count)
    return fertility

def h_index_expert(citations):
    citations = np.array(citations)
    n         = citations.shape[0]
    array     = np.arange(1, n+1)
    citations = np.sort(citations)[::-1]
    h_idx = np.max(np.minimum(citations, array))
    return h_idx

def child_deg(g):
    hindex={}
    for node in g.nodes:
        childs = list(g.successors(node))
        if len(childs) > 0:
            #g_child=[len(list(g.successors(c))) for c in child ]
            childs_deg = [deg for n ,deg in g.out_degree(childs)]
            #print(child_deg)
            h_index = h_index_expert(childs_deg)
            #print(h_index)
        else:
            h_index = 0
        hindex[node] = h_index
    return hindex

def gm_index(graph):
    gm_index={}
    for node in graph.nodes:
        childs = list(graph.successors(node))
        if len(childs) > 0:
            childs_deg = sum([deg for n ,deg in graph.out_degree(childs)])
            #gm_inx=np.floor(np.sqrt(childs))
            for i in range(len(childs),-1,-1):
                if childs_deg >= i*i:
                    gm_index[node]=i
                    break
                else:
                    continue
        else:
            gm_index[node]=0
    return gm_index

def gm_index_mod(graph):
    gm_index={}
    for node in graph.nodes:
        childs = list(graph.successors(node))
        if len(childs) > 0 :
            childs_deg = [deg for n ,deg in graph.out_degree(childs)]
            childs_deg.sort(reverse = True)
            if sum(childs_deg)>0:
                for i in range(len(childs)):
                    if sum(childs_deg[:len(childs)-i]) >= (len(childs)-i)**2:
                        gm_index[node]=len(childs)-i
                        break
                    else:
                        continue
            else:
                gm_index[node]=0
        else:
            gm_index[node]=0
    return gm_index

def Generation(g):
    gen_dict={}
    try:
        for i,node in enumerate(g.nodes):
            gen=0
            childs=list(g.successors(node))
            #print(child)
            while len(childs) > 0:
                gen+=1
                childs=[list(g.successors(c)) for c in childs if len(list(g.successors(c)))>0]
                childs=[coc for child in childs for coc in child]
                #print(child)
            gen_dict[node]=gen
            if i%100000==0:
                print(f"completed :{i} nodes")
    except:
        return gen_dict
    return gen_dict

def cousins(g):
    cousins={}
    for node in g.nodes:
        cousin=[]
        pred=list(g.predecessors(node))
        if len(pred)==0:
            cousin=[]
        else:
            pred1=[list(g.predecessors(p)) for p in pred if len(list(g.predecessors(p)))>0]
            pred1=[n for p in pred1 for n in p]
            #print(pred1)
            if len(pred1)==0:
                cousin=[]
            else:
                succ=[list(g.successors(p)) for p in pred1 if len(list(g.successors(p)))>0]
                #print(succ)
                succ=[n for p in succ for n in p]
                #print(succ)
                #print(pred)
                uncle=[p for p in succ if p not in pred]
                #print(uncle)
                if len(uncle) > 0:
                    cousin=[list(g.successors(u)) for u in uncle if len(list(g.successors(u)))>0]
                    cousin=[n for c in cousin for n in c]
                    #print(cousin)
        cousins[node]=len(cousin)
    return cousins

def desc_calculation(g):
    desc_dist={}
    for node in g.nodes:
        desc = len(nx.descendants(g,node))
        desc_dist[node]=desc
    return desc_dist

def fertility_calculation(g):
    fertility={}
    for node in g.nodes:
        successor = g.successors(node)
        #print(list(successor))
        child=g.out_degree(successor)
        count=[k for k, v in child if v > 0]
        fertility[node]=len(count)
    return fertility

#!pip install import-ipynb
def draw_dist_graph(metric_dict,title='Institute wise fecundity distribution', ylabel='Institute Count',xlabel='Inst. Fecundity value',bin_size=np.arange(0,600,100)):
    #counts, bins = np.histogram(list(metric_dict.values()))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(list(metric_dict.values()), bins=10, log=True)
    plt.title(title)
    plt.ylabel(ylabel,fontweight='bold',fontsize=20)
    plt.xlabel(xlabel,fontweight='bold',fontsize=20)
    plt.show()
    return

def load_obj(folder):
    with open(folder, 'r') as fp:
        data = json.load(fp)
    return data

def save_obj(name, data):
    save_fol="../save_data/"
    with open(save_fol+name+'.json', 'w') as fp:
        json.dump(data, fp)
    print(f"file save successfully in folder : {save_fol}")

def save_obj_inst(name, data):
    save_fol="dataset_v5/v5_2/top_10/institute_level/"
    with open(save_fol+name+'.json', 'w') as fp:
        json.dump(data, fp)
    print(f"file save successfully in folder : {save_fol}")

def find_unique_values(df,col):
    uniq_val=df[col].unique()
    return uniq_val

def inst_wise_val(df, col, dist_dict):
    metric={}
    uniq_col_val = find_unique_values(df,col)
    for val in uniq_col_val:
        filter_df=df[df[col]==val]
        uniq_id = pd.unique(filter_df[['resId','advId']].values.ravel('K'))
        metric_sum=sum([int(dist_dict[idd]) for idd in uniq_id])
        metric[val]=metric_sum
    return metric

def top_metric_val(metric_dict,top=10):
    metric_sort=dict(sorted(metric_dict.items(), key = lambda item: item[1], reverse = True))
    top_val=list(metric_sort.keys())[:top]
    top_key_val={key:metric_sort[key] for i, key in enumerate(metric_sort) if i < top}
    return metric_sort, top_key_val #change return type from top_val to metric_sort

def filter_df_based_on_date(df, st_date='2000-01-01', end_date='2020-01-01'):
    df['new_date_awarded']=pd.to_datetime(df['new_date_awarded'],errors='coerce')
    df = df.set_index('new_date_awarded')
    df_filter=df.loc[st_date:end_date].copy()
    return df_filter

def ntw_graph(filter_data):
    graph1 = nx.convert_matrix.from_pandas_edgelist(filter_data, 'advId','resId','publisher_institution', create_using=nx.DiGraph())
    graph1.remove_edges_from(nx.selfloop_edges(graph1))
    cycle=list(nx.simple_cycles(graph1))
    graph1.remove_edges_from(cycle)
    return graph1

def ntw_graph_with_instId(filter_data):
    graph1 = nx.convert_matrix.from_pandas_edgelist(filter_data, 'advId','resId','instituteId', create_using=nx.DiGraph())
    graph1.remove_edges_from(nx.selfloop_edges(graph1))
    cycle=list(nx.simple_cycles(graph1))
    graph1.remove_edges_from(cycle)
    return graph1

#np.floor(np.sqrt(15))

def ntw_graph_ddc(filter_data):
    graph = nx.convert_matrix.from_pandas_edgelist(filter_data, 'advId','resId','upper_ddc_code', create_using=nx.DiGraph())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    cycle=list(nx.simple_cycles(graph))
    graph.remove_edges_from(cycle)
    return graph

def rank_grid(rank_change, labels, xlabels=None, title="Change in Institute Rank Over Time"):
    alabels = labels
    xlabels = xlabels
    ranklabels=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
               '11th','12th','13th','14th','15th','16th','17th','18th','19th','20th']
    nsize=rank_change.shape[0]
    ylabels=ranklabels[:nsize]
 
    mycolors = colors.ListedColormap(['#de425b','#f7f7f7','#67a9cf'])
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(rank_change, cmap=mycolors)
 
    # Show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
 
    # Create white grid.
    ax.set_xticks(np.arange(rank_change.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(rank_change.shape[0]+0.5)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.grid(which="major", visible=False)
 
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[1,0,-1], shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_yticklabels(['Increased','No Change','Decreased'])
 
    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            if rank_change[i,j] < 0:
                text = ax.text(j, i, alabels[i, j],
                           ha="center", va="center", color="w",fontsize=5,wrap=True,weight='bold')
            else:
                text = ax.text(j, i, alabels[i, j],
                           ha="center", va="center", color="k",fontsize=5,wrap=True,weight='bold')
 
    ax.set_title(title)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    fig.tight_layout()
    plt.show()
    return ax

def rank_grid_not_colorbar(rank_change, labels, xlabels=None, title="Change in Institute Rank Over Time"):
    alabels = labels
    xlabels = xlabels
    ranklabels=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
               '11th','12th','13th','14th','15th','16th','17th','18th','19th','20th']
    nsize=rank_change.shape[0]
    ylabels=ranklabels[:nsize]
 
    mycolors = colors.ListedColormap(['#de425b','#f7f7f7','#67a9cf'])
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(rank_change, cmap=mycolors)
 
    # Show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
 
    # Create white grid.
    ax.set_xticks(np.arange(rank_change.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(rank_change.shape[0]+0.5)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.grid(which="major", visible=False)
 
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[1,0,-1], shrink=0.3)
    cbar.remove()
#     cbar.ax.tick_params(labelsize=5)
#     cbar.ax.set_yticklabels(['Increased','No Change','Decreased'])
    
 
    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            if rank_change[i,j] < 0:
                text = ax.text(j, i, alabels[i, j],
                           ha="center", va="center", color="w",fontsize=5,wrap=True,weight='bold')
            else:
                text = ax.text(j, i, alabels[i, j],
                           ha="center", va="center", color="k",fontsize=5,wrap=True,weight='bold')
 
    ax.set_title(title)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    fig.tight_layout()
    plt.show()
    return ax

def rank_change(geoarray):
    rowcount=geoarray.shape[0]
    colcount=geoarray.shape[1]
 
    # Create a number of blank lists
    changelist = np.zeros((rowcount, colcount))
 
    for i in range(colcount):
        if i==0:
            # Rank change for 1st year is 0, as there is no previous year
            for j in range(rowcount):
                changelist[j,i]=0
        else:
            col=geoarray[:,i] #Get all values in this col
            prevcol=geoarray[:,i-1] #Get all values in previous col
            for v in col:
                array_pos=np.where(col == v) #returns array
                current_pos=int(array_pos[0]) #get first array value
                array_pos2=np.where(prevcol == v) #returns array
                if len(array_pos2[0])==0: #if array is empty, because place was not in previous year
                    previous_pos=current_pos+1
                else:
                    previous_pos=int(array_pos2[0]) #get first array value
                if current_pos==previous_pos:
                    changelist[current_pos,i]=0
                    #No change in rank
                elif current_pos > previous_pos: #Larger value = smaller rank
                    changelist[current_pos,i]=-1
                    #Rank has decreased
                else:
                    changelist[current_pos,i]=1
    #rankchange=np.array(changelist)
    return changelist

def get_labels(array,inx_to_inst):
    labels=[]
    for i in range(array.shape[0]):
        tmp=[]
        for j in range(array.shape[1]):
            tmp.append(inx_to_inst[array[i,j]])
        labels.append(tmp)
    return np.array(labels)

def get_rank_array(dict1,dict2,dict3,dict4):
    rank_list=[]
    rank_list.append(list(top10_dict_0.keys()))
    rank_list.append(list(top10_dict.keys()))
    rank_list.append(list(top10_dict_1.keys()))
    rank_list.append(list(top10_dict_2.keys()))
    arr=np.array(rank_list).T
    return arr

def get_rank_array1(dict_list):
    rank_list=[]
    for tmp in dict_list:
        rank_list.append(list(tmp.keys()))
    arr=np.array(rank_list).T
    return arr

def filter_dataframe(df, till_date='2000-01-01'):
    df['new_date_awarded']=pd.to_datetime(df['new_date_awarded'],errors='coerce')
    df = df.set_index('new_date_awarded')
    df_filter=df.loc[:till_date].copy()
    return df_filter

def thesis_advised_cumlative(df=None, metric=0, till_date='1980-01-01', func1=Counter, attr="instituteId", func3=ntw_graph_with_instId):
    if metric > 0:
        print("Change func2 parameter")
        sys.exit()
    filter_data = filter_dataframe(df, till_date=till_date)
    graph = func3(filter_data) 
    inst_stud_tuple = set([(value[attr],v) for ((u, v), value) in graph.edges.items()])
    attr_count = [inst for inst, stud in inst_stud_tuple]
    inst_wise_stud_dist = func1(attr_count)
    top10, top10_dict = top_metric_val(inst_wise_stud_dist)
    return inst_wise_stud_dist, top10_dict

def other_metrics1(df=None, metric=1, till_date='1980-01-01', func1=inst_wise_val, attr="instituteId", func3=ntw_graph_with_instId):
    filter_data=filter_dataframe(df, till_date=till_date)
    graph = func3(filter_data)
    if metric==1:
        fecundity=graph.out_degree()
        metric_dict=dict(fecundity)
    elif metric==2:
        fertiltiy=fertility_calculation(graph)
        metric_dict=dict(fertiltiy)
    elif metric==3:
        hinx=child_deg(graph)
        metric_dict=dict(hinx)
    elif metric==4:
        ginx=gm_index_mod(graph)
        metric_dict=dict(ginx)
    else:
        #print('')
        sys.exit('unknown metric index')
    inst_fecund_metric=func1(filter_data, attr, metric_dict)
    _, top10_dict=top_metric_val(inst_fecund_metric)
    return inst_fecund_metric, top10_dict

def draw_rank_heatmap1(df=None, inx_to_inst=None, from_y=1940, to_y=2020, step_size=20, func1=Counter, func2=thesis_advised_cumlative, metric=0, attr="instituteId",func3=ntw_graph_with_instId,title=None):
    top_dict_list=[]
    to_y = to_y+step_size
    for year in range(from_y, to_y, step_size):
        _, top_dict = func2(df, metric=metric, till_date=f'{year}-01-01', func1=func1, attr=attr, func3=func3)
        top_dict_list.append(top_dict)

    #_ , top_dict_last = func2(flag='complete', metric=metric)
    #top_dict_list = top_dict_list+[top_dict_last]
    
    arr = get_rank_array1(top_dict_list)
    labels = get_labels(arr, inx_to_inst)
    rank = rank_change(arr)
    xlabels = [a for a in range(from_y,to_y,step_size)]
    fig1 = rank_grid(rank, labels, xlabels = xlabels, title=title)
    return fig1

def draw_rank_heatmap2(df=None, inx_to_inst=None, from_y=1940, to_y=2020, step_size=20, func1=Counter, func2=thesis_advised_cumlative, metric=0, attr="instituteId",func3=ntw_graph_with_instId,title=None):
    top_dict_list=[]
    to_y = to_y+step_size
    for year in range(from_y, to_y, step_size):
        _, top_dict = func2(df, metric=metric, till_date=f'{year}-01-01', func1=func1, attr=attr, func3=func3)
        top_dict_list.append(top_dict)

    #_ , top_dict_last = func2(flag='complete', metric=metric)
    #top_dict_list = top_dict_list+[top_dict_last]
    
    arr = get_rank_array1(top_dict_list)
    labels = get_labels(arr, inx_to_inst)
    rank = rank_change(arr)
    xlabels = [a for a in range(from_y,to_y,step_size)]
    fig1 = rank_grid_not_colorbar(rank, labels, xlabels = xlabels, title=title)
    return fig1

def student_produced_over_time(df=None, st_year=1900, end_year=2020, interval_size=10, func3=ntw_graph_with_instId):
    over_time=[]
    labels=[]
    end_year=end_year+interval_size
    for year in range(st_year, end_year, interval_size):
        #filter_data = filter_df_based_on_date(df, st_date=f'{year}-01-01',end_date=f'{year+interval_size}-01-01')
        filter_data = filter_dataframe(df, till_date=f'{year}-01-01')
        graph = func3(filter_data)
        total = len(set([v for ((u, v), value) in graph.edges.items()]))
        over_time.append(total)
        #labels.append(str(year)+"-"+str(year+interval_size)[2:4])
        labels.append(year)
    print(over_time)
    print(labels)
    fig, ax =  plt.subplots(figsize=(12,8))
    plt.plot(labels, over_time, color="b",marker="*")
    plt.title("", fontsize=12)#No. of Researcher Graduated over time
    plt.ylabel("Frequency", fontsize=20,fontweight='bold')
    plt.xlabel("Year", fontsize=20,fontweight='bold')
    plt.savefig('../new_graphs/thesis_distribution_over_time.svg',bbox_inches='tight')
    plt.show()
    return over_time

def inst_increase_over_time(df=None, st_year=1900, end_year=2020, interval_size=10, func3=ntw_graph_with_instId):
    over_time=[]
    labels=[]
    end_year=end_year+interval_size
    for year in range(st_year, end_year, interval_size):
        filter_data = filter_dataframe(df, till_date=f'{year}-01-01')
        graph = func3(filter_data)
        total = len(set([value['instituteId'] for ((u, v), value) in graph.edges.items()]))
        over_time.append(total)
        labels.append(year)
    print(over_time)
    print(labels)
    fig, ax =  plt.subplots(figsize=(12,8))
    ax.plot(labels, over_time, color="b",marker="*")
    ax.set_title("",fontsize=12)#No. of University/Institute increase over time
    ax.set_ylabel("Frequency",fontsize=20,fontweight='bold')
    ax.set_xlabel("Year",fontsize=20,fontweight='bold')
    plt.savefig('../new_graphs/inst_distribution_over_time.svg',bbox_inches='tight')
    plt.show()
    return over_time

# def mod_jaccard_dist(dict1, dict2, inx_to_inst):
#     order_rank1= [inx_to_inst[i] for i in dict1.keys()]
#     order_rank2= [inx_to_inst[i] for i in dict2.keys()]
#     top1_mod={inx:len(dict1)-i for i, inx in enumerate(list(dict1.keys()))}
#     top2_mod={inx:len(dict2)-i for i, inx in enumerate(list(dict2.keys()))}
#     comm_dict1={}
#     common_keys=set(dict1.keys()).intersection(dict2.keys())
#     for k in common_keys:
#         comm_dict1[k] = top1_mod.get(k,0) if top2_mod.get(k,0) >= top1_mod.get(k,0) else top2_mod.get(k,0)
#     dist=sum(comm_dict1.values())/(sum(top1_mod.values())+sum(top2_mod.values()))
#     return order_rank1, order_rank2, dist

#extended jaccard
def mod_jaccard_dist(dict1, dict2, inx_to_inst):
    order_rank1= [inx_to_inst[i] for i in dict1.keys()]
    order_rank2= [inx_to_inst[i] for i in dict2.keys()]
    top1_mod={inx:len(dict1)-i for i, inx in enumerate(list(dict1.keys()))}
    top2_mod={inx:len(dict2)-i for i, inx in enumerate(list(dict2.keys()))}
    comm_dict1={}
    common_keys=set(dict1.keys()).intersection(dict2.keys())
    union_keys=set(dict1.keys()).union(dict2.keys())
    union_dict1={}
    for k in common_keys:
        comm_dict1[k] = top1_mod.get(k,0) if top2_mod.get(k,0) >= top1_mod.get(k,0) else top2_mod.get(k,0)
    for k in union_keys:
        union_dict1[k] = top2_mod.get(k,0) if top2_mod.get(k,0) >= top1_mod.get(k,0) else top1_mod.get(k,0)
    dist=1-sum(comm_dict1.values())/sum(union_dict1.values())
    return  order_rank1, order_rank2, dist

def draw_rank_dist(df=None, inx_to_inst=None, from_y=1940, to_y=2020, step_size=10, func1=Counter, func2=thesis_advised_cumlative, metric=0, title="",attr="instituteId",func3=ntw_graph_with_instId):
    top_dict_list=[]
    dist_list=[]
    rank_list=[]
    #to_y=to_y+step_size
    for year in range(from_y, to_y+step_size, step_size):
        _, top_dict = func2(df, metric=metric, till_date=f'{year}-01-01', func1=func1, attr=attr, func3=func3)
        top_dict_list.append(top_dict)
    print(len(top_dict_list))
  
    xlabels = [a+step_size for a in range(from_y, to_y, step_size)]

    #xlabels = [a for a in range(from_y+2*step_size, to_y, 2*step_size)]
    for i,j in zip(top_dict_list, top_dict_list[1:]):
#         dict1=top_dict_list[i]
#         dict2=top_dict_list[i+1]
        l1,l2,dist=mod_jaccard_dist(i,j,inx_to_inst)
        dist_list.append(dist)
        rank_list.append((l1,l2))
            
    print(dist_list)
    print(xlabels)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(xlabels, dist_list, color="b", marker='o')
    ax.set_title(title)
    ax.set_ylabel("Distance",fontweight='bold',fontsize=20)
    ax.set_xlabel("Year",fontweight='bold',fontsize=20)
    #ax.set_xticks([d for d in deg])
    #ax.set_xticklabels(deg)
    plt.show()
    return rank_list, ax

def mod_jaccard_dist_exp(dict1, dict2):
    top1_mod={inx:len(dict1)-i for i, inx in enumerate(list(dict1.keys()))}
    top2_mod={inx:len(dict2)-i for i, inx in enumerate(list(dict2.keys()))}
    comm_dict1={}
    common_keys=set(dict1.keys()).intersection(dict2.keys())
    union_keys=set(dict1.keys()).union(dict2.keys())
    union_dict1={}
    for k in common_keys:
        comm_dict1[k] = top1_mod.get(k,0) if top2_mod.get(k,0) >= top1_mod.get(k,0) else top2_mod.get(k,0)
    for k in union_keys:
        union_dict1[k] = top2_mod.get(k,0) if top2_mod.get(k,0) >= top1_mod.get(k,0) else top1_mod.get(k,0)
    dist=1-sum(comm_dict1.values())/sum(union_dict1.values())
    return dist, comm_dict1, union_dict1

def draw_area_plot(df=None, inx_to_inst=None, from_y=1940, to_y=2020, step_size=10, func1=Counter, func2=thesis_advised_cumlative, metric=0, title="", attr="instituteId", func3=ntw_graph_with_instId):
    top_dict_list=[]
    comp_dist_list=[]
    xlabels=[]
    for year in range(from_y, to_y+step_size, step_size):
        comp_dist, top_dict = func2(df, metric=metric, till_date=f'{year}-01-01', func1=func1, attr=attr, func3=func3)
        top_dict_list.append(top_dict)
        comp_dist_list.append(comp_dist)
        xlabels.append(year)
    uniq_inst=[]
    inst_year={}
    for dict1 in top_dict_list:
        uniq_inst+=list(dict1.keys())
    for inst in set(uniq_inst):
        for dict1 in comp_dist_list:
            if inst in inst_year:
                inst_year[inst].append(dict1.get(inst,0))
            else:
                inst_year[inst]=[]
                inst_year[inst].append(dict1.get(inst,0))
    #xlabels = [a+step_size for a in range(from_y, to_y, step_size)]
    print(xlabels)
    inst_year = {inx_to_inst[key]: inst_year[key] for key in inst_year}
    print(list(inst_year.values()))
    print(list(inst_year.keys()))
#     NUM_COLORS = len(inst_year)
    #graph plot
#     cm = plt.get_cmap('gist_rainbow')
    fig, ax = plt.subplots(figsize=(12,10))
#     colors = sns.color_palette("hls", NUM_COLORS)
#     ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.stackplot(xlabels,list(inst_year.values()), labels=list(inst_year.keys()))
    plt.yticks(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    print(labels)
#     sort both labels and handles by labels
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles[::-1], labels[::-1],loc='upper left',fancybox=True, shadow=True,prop={"weight":'bold',"size":16}) #size=12
    ax.set_xlabel('Year',fontweight='bold',fontsize=20)
    ax.set_ylabel('Thesis Advised',fontweight='bold',fontsize=20)
    ax.set_title(title)
    plt.show()
    return inst_year, ax

def mod_draw_rank_dist(df=None, inx_to_inst=None, from_y=1940, to_y=2020, step_size=10, func1=Counter, func2=thesis_advised_cumlative, metric=0,attr="instituteId",func3=ntw_graph_with_instId):
    top_dict_list=[]
    dist_list=[]
    rank_list=[]
    #to_y=to_y+step_size
    for year in range(from_y, to_y+step_size, step_size):
        _, top_dict = func2(df, metric=metric, till_date=f'{year}-01-01', func1=func1, attr=attr, func3=func3)
        top_dict_list.append(top_dict)
    print(len(top_dict_list))
  
    xlabels = [a+step_size for a in range(from_y, to_y, step_size)]

    #xlabels = [a for a in range(from_y+2*step_size, to_y, 2*step_size)]
    for i,j in zip(top_dict_list, top_dict_list[1:]):
#         dict1=top_dict_list[i]
#         dict2=top_dict_list[i+1]
        l1,l2,dist=mod_jaccard_dist(i,j,inx_to_inst)
        dist_list.append(dist)
        rank_list.append((l1,l2))
            
#     print(dist_list)
#     print(xlabels)
    return xlabels,dist_list,rank_list

def draw_bar_graph(metric_value_list,title='Fecundity Distribution',ylabel='Researcher Count',xlabel='Fecundity Value',bins=10, loc="../new_graphs/hist", text=False, print_bin=False):
    #counts, bins = np.histogram(metric_value_list)
    plt.figure(figsize=(12,8))
    ax = plt.axes()
    n,bins,container=plt.hist(metric_value_list, bins=bins,color='#0504aa',density=False,log=True,rwidth=0.95)#  alpha=0.7
    #plt.hist(deg, color="b", bins=bin_size)
    plt.yticks(fontsize=16,fontweight='bold')
    plt.xticks(fontsize=16,fontweight='bold')
    plt.title(title)
    plt.ylabel(ylabel,fontweight='bold', fontsize=20) #fontsize=18
    plt.xlabel(xlabel,fontweight='bold', fontsize=20)
    if text:
        for i,v in zip(bins[0:-1], n):
            if v > 0:
                plt.text(i, v, str(int(v)), color='green', fontweight='bold',fontsize=16)
    if print_bin:
        ax.set_xticks([int(d) for d in bins])
        ax.set_xticklabels([int(b) for b in bins])
    #plt.xticks(bins[0::5])
    plt.savefig(loc,bbox_inches='tight')
    plt.show()
    print(n)
    print(bins)
    return

def draw_line_graph(metric_counter,title="",xlabel="",ylabel="",loc='../new_graphs/line'):
    components1_dist_sort=sorted(metric_counter.items())
    deg, cnt = zip(*components1_dist_sort)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(deg, cnt, color="b")
    plt.title(title)
    plt.ylabel(ylabel,fontweight='bold', fontsize=20)
    plt.xlabel(xlabel,fontweight='bold', fontsize=20)
    plt.yscale('log')
    #ax.set_xticks([d for d in deg])
    #ax.set_xticklabels(deg)
    plt.savefig(loc,bbox_inches='tight')
    plt.show()
    print(deg)
    print(cnt)
    return

if __name__=='__main__':
    print("main")
#     a = [2,2,3,3,3,4,4,5,5,6,6,6,6,6,7,7,77,7,7,8]
#     draw_bar_graph(a, text=True,print_bin=True)
#     folder="../dataset_v5/v5_2/"

#     mod_ment2 = pd.read_csv(folder+'v5_2_2/'+'final_shodhganga_dataset_v5_2_6.csv', sep =",")

#     inx_to_inst = dict(zip(mod_ment2.instituteId,mod_ment2.publisher_institution))

#     year_wise_inst_dist, ax6 = draw_area_plot(mod_ment2, inx_to_inst=inx_to_inst)
#     fig6=ax6.get_figure()
#     fig6.savefig('../new_graphs/inst_stackplot_with_thesis.pdf',bbox_inches='tight')