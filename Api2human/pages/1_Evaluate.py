# import nltk
# from nltk.tokenize import sent_tokenize
# import numpy as np
import re
import json
import streamlit as st
from stanfordcorenlp import StanfordCoreNLP
from collections import defaultdict, Counter

if "dscores" not in st.session_state:
    st.session_state["dscores"] = 0
if "mscores" not in st.session_state:    
    st.session_state["mscores"] = 0 
if "rscores" not in st.session_state:    
    st.session_state["rscores"] = 0

def div_name(name):
    # 常量命名/驼峰命名/帕斯卡命名 ->下划线命名
    _name = re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', '_\g<0>', name).lower()
    # 下划线命名
    return _name.strip("$").replace("-","_").split("_")

def cal_MPDs(names, labels):
    MPDs = []
    score, consist_score = 0, 0
    mean_scores, direct_scores, read_scores = 0, 0, 0
    for i,label in enumerate(labels):
        divname = div_name(names[i])
        MPDs_pre = re.split('[:.;!?]', label)
        mean_score, direct_score, read_score = 0, 0, 0
        for idx, MPD in enumerate(MPDs_pre):
            MPD = MPD.lower()
            # 剔除重复的话 
            if (sorted(Counter(MPD.split(" ")).items(),key=lambda x:x[1],reverse=True)[0][1] > 5):
                continue                
            # 根据名称提及率提取MPD
            cnt_mention = 0
            for word in divname:
                if word in MPD.replace(" ",""):
                    cnt_mention += 1
            tmp = cnt_mention/len(divname)
            # 等于的情况不往后
            if (tmp > mean_score):
                # 计算直接性，有意义性，可读性
                mean_score = cnt_mention/len(divname)
                direct_score = idx
                # direct_score = (idx+1)/len(MPD_pre)
                read_score = cnt_mention/len(MPD.split(" "))
                # break
        MPDs.append(MPDs_pre[direct_score])
        # 有意义性为0的情况直接性往后，MPD不往后
        if (mean_score==0):
            direct_score = len(MPDs_pre)
        # 根据重复词汇修正可读性，可能出现负数
        # read_score -= max(Counter(MPD.split(" ")).items())/len(MPD.split(" "))
        mean_scores +=mean_score
        direct_scores+=direct_score
        read_scores+=read_score
    st.session_state["dscores"] = 1-direct_scores/len(labels)
    st.session_state["mscores"] = mean_scores/len(labels)
    st.session_state["rscores"] = 2*read_scores/len(labels)
    return MPDs

# def classifer(sentence= 'enddatetime'):
#     # print(nlp.word_tokenize(sentence))
#     # print(nlp.pos_tag(sentence))
#     # print(nlp.ner(sentence))
#     if "nlp" not in st.session_state:
#         nlp = StanfordCoreNLP(r'/wangcan/Api2human/Api2human/stanford-nlp/')
#     else:
#         nlp = st.session_state['nlp']
#     if (sorted(Counter(sentence.split(" ")).items(),key=lambda x:x[1],reverse=True)[0][1] > 5):
#         sentence = " ".join(sentence.split(" ")[:40])  
#     tree = nlp.parse(sentence.replace('"',"").replace("'","").replace('/',"").replace("`","").replace(")","").replace("(","").replace("*",""))
#     # nlp.dependency_parse(sentence)
#     # 剔除重复的话 
#     stack = []
#     layers = defaultdict(list)
#     layer = 0
#     beginlayer = 2e5
#     while tree:
#         tree = tree.strip(' \r\n')
#         if tree[0]=='(':
#             dividx = tree.strip(' (').find('(') +1
#             if (dividx==0) or (')' in tree[:dividx]):
#                 dividx = tree.strip('(').find(')') +1
#             stack.append((layer,tree[:dividx].strip("\n\r '")))
#             layer+=1
#             tree = tree[dividx:]
#         elif tree[0]==')':
#             item = stack.pop()
#             # （层数，词性，【单词】）
#             l = item[0]
#             p_and_w = item[1].strip('(').split(" ")
#             if len(p_and_w)>1:
#                 beginlayer = min(beginlayer, l)
#             layers[l].append(p_and_w[0].strip(" "))
#             layer-=1
#             tree = tree[1:]
#     # 句式解析
#     res = None
#     for idx in range(beginlayer,0,-1):
#         # print(layers[idx])
#         # if layers[idx] == ['NP', 'PP'] or layers[idx] == ['NP', 'PP', 'VP']:
#         if layers[idx] == ['NP', 'PP'] or layers[idx] == ['NP', 'S'] or layers[idx] == ['NP','NP','VP'] or layers[idx]==['NP-TMP','NP','VP']:
#             res = 'NP+PP'
#             break
#             # return res
#         elif layers[idx] == ['NP', 'VP']:
#             res = 'NP+VP'
#             break
#             # return res
#         elif layers[idx]==['NP'] or set(layers[idx]) == set(['NP','NP']):
#             res = 'NP'
#             break
#             # return res
#         elif layers[idx]==['VP']:
#             res = 'VP'
#             break
#             # return res
#         # elif layers[idx] == ['NP','S']:
#             # res = 'NP+S'
#             # return res
#         # elif layers[idx]==['FRAG']:
#             # res = 'FRAG'
#             # return res
#         elif 'SBAR' in layers[idx] or 'SBARQ' in layers[idx]:
#             res = 'SBAR'
#             break
#             # return res 
#     # print(sentence+'\n', layers)
#     otherres = layers[2]
#     while ',' in otherres:
#         otherres.remove(',')
#     return res if res else str(otherres)

def upload():
    try:
        # To read file as string:
        string_data = uploaded_file.read().decode("utf-8")
        # rawdata = string_data.split("\n")
        names, labels = [], []
        testdata = json.loads(string_data)
        for data in testdata:
            if data:
                names.append(data['name'])
                labels.append(data['desc'])
        MPDs = cal_MPDs(names, labels)
        # classes = defaultdict(int)
        # for MPD in MPDs:
        #     classes[classifer(MPD)]+=1
        # format_score = 0
        # for c, c_num in classes.items():
        #     prob = c_num / len(MPDs)
        #     format_score += (-prob)*math.log(prob,2)  
        # print(format_score)
        st.success("Evaluate successfully!")
    except:
        st.error("Error happened. Please check the document!")
        
st.markdown('### Please Select a json file in the following format:\n')
# st.markdown('### 请选择一个json文件，格式如下:\n')
st.code('[\n\t{"name": param_name1, "desc": param_desc1},\n\t{"name": param_name2, "desc": param_desc2}\n\t...\n]', language ="python")
# st.code('[\n\t{"name": 参数名称1, "desc": 参数描述1},\n\t{"name": 参数名称2, "desc": 参数描述2}\n\t...\n]', language ="python")
uploaded_file = st.file_uploader(' ',label_visibility ="collapsed")
st.button("Evaluate", on_click = upload,type="primary",use_container_width=True)
# st.button("进行评估", on_click = upload,type="primary",use_container_width=True)

st.empty()
st.markdown("---")
st.markdown('### The evaluate result:\n')

col1, col2, col3 = st.columns([4,4,4])
dscores, mscores, rscores = st.session_state["dscores"],st.session_state["mscores"],st.session_state["rscores"]
with col1:
    empty, col = st.columns([1,4])
    col.markdown("#### Direct score")
    col.metric(label="Direct score",label_visibility ="collapsed", value ='%.2f' %dscores, delta = '%.2f%%' % ((dscores-0.5)*100/0.5))
with col2:
    empty, col = st.columns([1,4])
    col.markdown("#### Mean score")
    col.metric(label ="Mean Score",label_visibility ="collapsed", value ='%.2f' %mscores, delta ='%.2f%%' % ((mscores-0.5)*100/0.5))
with col3:
    empty, col = st.columns([1,4])
    col.markdown("#### Read score")
    col.metric(label ="Read Scores",label_visibility ="collapsed", value ='%.2f' %rscores, delta = '%.2f%%' % ((rscores-0.5)*100/0.5))
