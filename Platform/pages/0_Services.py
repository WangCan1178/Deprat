import json
import numpy as np
import pandas as pd
import streamlit as st

from utils import database
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 设置页面，加载session
def init_page():
    st.set_page_config(
    page_title="My_services",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    )
    st.markdown("""
        <style>
            .stButton .edgvbvh10 {
                background-color: #e8d8ff;
                border: 0px solid transparent;
                margin:0px 0px 0px 0px;
            }

            .stButton .edgvbvh10:hover {
                background-color:#f2e6ff;
            }
            
            .stDownloadButton .edgvbvh10 {
                background-color: #A682FF;
                border: 1px solid #A682FF;
                margin:0px 0px 0px 0px;
            }
            .stDownloadButton .edgvbvh10 .e16nr0p34{
                color: #FFFFFF;
            }
            .stDownloadButton .edgvbvh10:hover {
                background-color: #9166FF;
                border: 1px solid #9166FF;
                margin:0px 0px 0px 0px;
            }
            
        </style>
        """, unsafe_allow_html=True)
    
    if "model" not in st.session_state:
        st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained('/data/wangcan/T5/model-t5-base/checkpoint-75000')
    if "tokenizer" not in st.session_state:
        st.session_state['tokenizer'] = AutoTokenizer.from_pretrained('/data/wangcan/T5/model-t5-base/checkpoint-75000')
    if "now_id" not in st.session_state:
        st.session_state["now_id"] = 0
 
init_page()
typelst = ["GET","PUSH","DELETE"]
# 顺序存储apiiid
idlst = []
# 顺序存储box
boxlst = []
# 查数据库，获取所有服务
# db = database.DB()
# services = db.findall() 
with open('./utils/data.json') as f:
    services = json.load(f)

# 通过apiid变化决定页面显示
def change_id(apiid):
    st.session_state["now_id"] = apiid
    del st.session_state["req_df"]
    del st.session_state["rep_df"]
 
# 生成功能
def gen(abstract):
    temperature = 0.9
    num_beams = 2
    max_gen_length = 512
    inputs = st.session_state['tokenizer']([abstract], max_length=512, return_tensors='pt')
    title_ids = st.session_state['model'].generate(
        inputs['input_ids'], 
        num_beams=num_beams, 
        temperature=temperature, 
        max_length=max_gen_length, 
        early_stopping=True
    )
    title = st.session_state['tokenizer'].decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return title

# 删除api
def delete(lst):
    for i,flag in enumerate(lst):
        if flag:
            for idx,ser in enumerate(services):
                if idx== idlst[i]:
                    services.pop(idx)
            with open('./utils/data.json','w') as f:
                result = json.dump(services, f)
            # db.delete(idlst[i])
    st.session_state["now_id"] = -1

# 侧边栏
def view_sidebar():
    st.sidebar.header("View, edit, and manage your services!")
    with st.sidebar:
        for service in services:
            empty, ckcbox, serlst = st.columns([1,1,15])
            with ckcbox:
                boxlst.append(st.checkbox(label=" ",label_visibility="collapsed", key="box"+str(service['apiid'])))
            with serlst:
                idlst.append(service['apiid'])
                st.button("{:<5s}&nbsp;&nbsp;&nbsp;{:>15s}".format(typelst[service['apitype']],service['apipath']), 
                        key= service['apiid'],
                        on_click = change_id,
                        args = [int(service['apiid'])],
                        use_container_width=True)
        st.markdown("\n")
        st.button("Create a New Service +", on_click = change_id,
                    args = [-1], use_container_width=True, type="primary")
        st.button("⚠️ Delete selected Services - ⚠️", on_click = delete,
                    args = [boxlst], use_container_width=True, type="primary")
 
#  主界面
def edit(apiid):
    if apiid!=-1:
        tmp = services[idlst.index(apiid)]
        apitype, apipath, apidesc, request, response = tmp['apitype'], tmp['apipath'], tmp['apidesc'], tmp['request'], tmp['response']
        st.write("## "+ typelst[apitype] +"&nbsp;&nbsp;&nbsp;"+apipath)
    else:
        apiid = len(services)
        apitype, apipath, apidesc, request, response = 0, "", "", [], []
        st.write("## "+ "Unnamed New API")
    
    l_co, input_col, emtpy_col = st.columns([2,8,5])
    with l_co:
        st.write('<font size="4.5">api-id:</font>', unsafe_allow_html =True)
        st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
        st.write('<font size="4.5">api-path:</font>', unsafe_allow_html =True)
        st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
        st.write('<font size="4.5">api-type:</font>', unsafe_allow_html =True)
        st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
        st.write('<font size="4.5">api-desc:</font>', unsafe_allow_html =True)
    with input_col:
        st.write(str(apiid))
        st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
        _apipath = st.text_input(label=" ", value=apipath, placeholder = "please enter your api path here.",label_visibility ='collapsed')
        _apitype = st.selectbox(label=" ",index=apitype, options=["GET","PUSH","DELETE"], label_visibility ='collapsed')
        _apidesc = st.text_input(label=" ", value=apidesc, placeholder = "please enter your api descriptions here.",label_visibility ='collapsed')

    
    if "req_df" not in st.session_state:
        st.session_state["req_df"] = pd.DataFrame(request)
    if "rep_df" not in st.session_state:
        st.session_state["rep_df"] = pd.DataFrame(response)
        
    st.write('<font size="0.5">           </font>', unsafe_allow_html =True)
    st.write("### Request-param")
    if len(st.session_state["req_df"])==0:
        request = [{"name": None, "type": None, "desc":None, "format":None}]
        reqed_df = st.experimental_data_editor(pd.DataFrame(request), num_rows="dynamic",key = "emptyreq",use_container_width=True)
    else:
        reqed_df = st.experimental_data_editor(st.session_state["req_df"], num_rows="dynamic",use_container_width=True)

    st.write('<font size="0.5">           </font>', unsafe_allow_html =True)
    st.write("### Response-param")
    if len(st.session_state["rep_df"])==0:
        response = [{"name": None, "type": None, "desc":None, "format":None}]
        reped_df = st.experimental_data_editor(pd.DataFrame(response), num_rows="dynamic",key = "emptyrep", use_container_width=True)
    else:
        reped_df = st.experimental_data_editor(st.session_state["rep_df"],key="dataeditor"+str(apiid), num_rows="dynamic",use_container_width=True)
    
    gencol, savecol, download = st.columns(3)
    _request = reqed_df.to_json(orient="records",force_ascii=False)
    _response = reped_df.to_json(orient="records",force_ascii=False)
    # 产生注释
    with gencol:
        w = st.button('Generate parameter descriptions', use_container_width=True, type="primary")
        if w:
            # st.write(len(reqed_df))
            # abstracts = []
            for i in range(len(reqed_df)):
                if ((reqed_df.loc[i,"desc"]==None) or len(reqed_df.loc[i,"desc"])==0) and (reqed_df.loc[i,'name']!=None):
                    abstract = "desc:"+apidesc+"#name:"+str(reqed_df.loc[i,'name'])+"#type"+str(reqed_df.loc[i,'type'])
                    reqed_df.loc[i,"desc"] = gen(abstract)
                    # abstracts.append((i, "desc:"+apidesc+"#name:"+str(reqed_df.loc[i,'name'])+"#type"+str(reqed_df.loc[i,'type'])))
            st.session_state["req_df"] = reqed_df.copy()
            # abstracts = []
            for i in range(len(reped_df)):
                if ((reped_df.loc[i,"desc"]==None) or len(reped_df.loc[i,"desc"])==0)and (reped_df.loc[i,'name']!=None):
                    abstract = "desc:"+apidesc+"#name:"+str(reped_df.loc[i,'name'])+"#type"+str(reped_df.loc[i,'type'])
                    reped_df.loc[i,"desc"] = gen(abstract)
            st.session_state["rep_df"] = reped_df.copy()
            st.experimental_rerun()    
    # 更新数据库
    with savecol:
        if st.button('Save API information', use_container_width=True, type="primary"):
            with open('./utils/data.json', 'w') as f:
                flag = False
                for idx,ser in enumerate(services):
                    if ser['apiid']== apiid:
                        flag = True
                        services[idx] = {"apiid":apiid, "apipath":_apipath, "apitype":typelst.index(_apitype), "apidesc":_apidesc, "request":json.loads(_request), "response":json.loads(_response)} 
                        break
                if not flag:
                    services.append({"apiid":apiid, "apipath":_apipath, "apitype":typelst.index(_apitype), "apidesc":_apidesc, "request":json.loads(_request), "response":json.loads(_response)})
                result = json.dump(services, f)
            # result = db.save(apiid, _apipath, , _apidesc, json.loads(_request), json.loads(_response))
            if result:
                st.success("Save successfully!")
            else:
                st.error("Save error!")
            st.experimental_rerun()            
    # 下载
    with download:
        jsondata = []
        for param in json.loads(_request):
            if param["name"]!=None and param["desc"]!=None:
                tmp = {"name":param["name"], "desc":param["desc"]}
                jsondata.append(tmp)
        for param in json.loads(_response):
            if param["name"]!=None and param["desc"]!=None:
                tmp = {"name":param["name"], "desc":param["desc"]}
                jsondata.append(tmp)
        st.download_button(
            label="Download params(json)",
            file_name="data.json",
            mime="application/json",
            data=json.dumps(jsondata),
            use_container_width = True
        )   

view_sidebar()
edit(st.session_state["now_id"])



# import json
# import numpy as np
# import pandas as pd
# import streamlit as st

# from utils import database
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # 设置页面，加载session
# def init_page():
#     st.set_page_config(
#     page_title="My_services",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     )
#     st.markdown("""
#         <style>
#             .stButton .edgvbvh10 {
#                 background-color: #e8d8ff;
#                 border: 0px solid transparent;
#                 margin:0px 0px 0px 0px;
#             }

#             .stButton .edgvbvh10:hover {
#                 background-color:#f2e6ff;
#             }
            
#             .stDownloadButton .edgvbvh10 {
#                 background-color: #A682FF;
#                 border: 1px solid #A682FF;
#                 margin:0px 0px 0px 0px;
#             }
#             .stDownloadButton .edgvbvh10 .e16nr0p34{
#                 color: #FFFFFF;
#             }
#             .stDownloadButton .edgvbvh10:hover {
#                 background-color: #9166FF;
#                 border: 1px solid #9166FF;
#                 margin:0px 0px 0px 0px;
#             }
            
#         </style>
#         """, unsafe_allow_html=True)
    
#     if "model" not in st.session_state:
#         st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained('/wangcan/T5/model-t5-base/checkpoint-74500/')
#     if "tokenizer" not in st.session_state:
#         st.session_state['tokenizer'] = AutoTokenizer.from_pretrained('/wangcan/T5/model-t5-base/checkpoint-74500/')
#     if "now_id" not in st.session_state:
#         st.session_state["now_id"] = 0
 
# init_page()
# typelst = ["GET","PUSH","DELETE"]
# # 顺序存储apiiid
# idlst = []
# # 顺序存储box
# boxlst = []
# # 查数据库，获取所有服务
# db = database.DB()
# services = db.findall() 
    
# # 通过apiid变化决定页面显示
# def change_id(apiid):
#     st.session_state["now_id"] = apiid
#     del st.session_state["req_df"]
#     del st.session_state["rep_df"]
 
# # 生成功能
# def gen(abstract):
#     temperature = 0.9
#     num_beams = 2
#     max_gen_length = 512
#     inputs = st.session_state['tokenizer']([abstract], max_length=512, return_tensors='pt')
#     title_ids = st.session_state['model'].generate(
#         inputs['input_ids'], 
#         num_beams=num_beams, 
#         temperature=temperature, 
#         max_length=max_gen_length, 
#         early_stopping=True
#     )
#     title = st.session_state['tokenizer'].decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     return title

# # 删除api
# def delete(lst):
#     for i,flag in enumerate(lst):
#         if flag:
#             db.delete(idlst[i])
#     st.session_state["now_id"] = -1

# # 侧边栏
# def view_sidebar():
#     st.sidebar.header("View, edit, and manage your services!")
#     with st.sidebar:
#         for service in services:
#             empty, ckcbox, serlst = st.columns([1,1,15])
#             with ckcbox:
#                 boxlst.append(st.checkbox(label=" ",label_visibility="collapsed", key="box"+str(service['apiid'])))
#             with serlst:
#                 idlst.append(service['apiid'])
#                 st.button("{:<5s}&nbsp;&nbsp;&nbsp;{:>15s}".format(typelst[service['apitype']],service['apipath']), 
#                         key= service['apiid'],
#                         on_click = change_id,
#                         args = [int(service['apiid'])],
#                         use_container_width=True)
#         st.markdown("\n")
#         st.button("Create a New Service +", on_click = change_id,
#                     args = [-1], use_container_width=True, type="primary")
#         st.button("⚠️ Delete selected Services - ⚠️", on_click = delete,
#                     args = [boxlst], use_container_width=True, type="primary")
 
# #  主界面
# def edit(apiid):
#     if apiid!=-1:
#         tmp = services[idlst.index(apiid)]
#         apitype, apipath, apidesc, request, response = tmp['apitype'], tmp['apipath'], tmp['apidesc'], tmp['request'], tmp['response']
#         st.write("## "+ typelst[apitype] +"&nbsp;&nbsp;&nbsp;"+apipath)
#     else:
#         apiid = len(services)
#         apitype, apipath, apidesc, request, response = 0, "", "", [], []
#         st.write("## "+ "Unnamed New API")
    
#     l_co, input_col, emtpy_col = st.columns([2,8,5])
#     with l_co:
#         st.write('<font size="4.5">api-id:</font>', unsafe_allow_html =True)
#         st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
#         st.write('<font size="4.5">api-path:</font>', unsafe_allow_html =True)
#         st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
#         st.write('<font size="4.5">api-type:</font>', unsafe_allow_html =True)
#         st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
#         st.write('<font size="4.5">api-desc:</font>', unsafe_allow_html =True)
#     with input_col:
#         st.write(str(apiid))
#         st.write('<font size="0.5">         </font>', unsafe_allow_html =True)
#         _apipath = st.text_input(label=" ", value=apipath, placeholder = "please enter your api path here.",label_visibility ='collapsed')
#         _apitype = st.selectbox(label=" ",index=apitype, options=["GET","PUSH","DELETE"], label_visibility ='collapsed')
#         _apidesc = st.text_input(label=" ", value=apidesc, placeholder = "please enter your api descriptions here.",label_visibility ='collapsed')

    
#     if "req_df" not in st.session_state:
#         st.session_state["req_df"] = pd.DataFrame(request)
#     if "rep_df" not in st.session_state:
#         st.session_state["rep_df"] = pd.DataFrame(response)
        
#     st.write('<font size="0.5">           </font>', unsafe_allow_html =True)
#     st.write("### Request-param")
#     if len(st.session_state["req_df"])==0:
#         request = [{"name": None, "type": None, "desc":None, "format":None}]
#         reqed_df = st.experimental_data_editor(pd.DataFrame(request), num_rows="dynamic",key = "emptyreq",use_container_width=True)
#     else:
#         reqed_df = st.experimental_data_editor(st.session_state["req_df"], num_rows="dynamic",use_container_width=True)

#     st.write('<font size="0.5">           </font>', unsafe_allow_html =True)
#     st.write("### Response-param")
#     if len(st.session_state["rep_df"])==0:
#         response = [{"name": None, "type": None, "desc":None, "format":None}]
#         reped_df = st.experimental_data_editor(pd.DataFrame(response), num_rows="dynamic",key = "emptyrep", use_container_width=True)
#     else:
#         reped_df = st.experimental_data_editor(st.session_state["rep_df"],key="dataeditor"+str(apiid), num_rows="dynamic",use_container_width=True)
    
#     gencol, savecol, download = st.columns(3)
#     _request = reqed_df.to_json(orient="records",force_ascii=False)
#     _response = reped_df.to_json(orient="records",force_ascii=False)
#     # 产生注释
#     with gencol:
#         w = st.button('Generate parameter descriptions', use_container_width=True, type="primary")
#         if w:
#             st.write(len(reqed_df))
#             # abstracts = []
#             for i in range(len(reqed_df)):
#                 if ((reqed_df.loc[i,"desc"]==None) or len(reqed_df.loc[i,"desc"])==0) and (reqed_df.loc[i,'name']!=None):
#                     abstract = "desc:"+apidesc+"#name:"+str(reqed_df.loc[i,'name'])+"#type"+str(reqed_df.loc[i,'type'])
#                     reqed_df.loc[i,"desc"] = gen(abstract)
#                     # abstracts.append((i, "desc:"+apidesc+"#name:"+str(reqed_df.loc[i,'name'])+"#type"+str(reqed_df.loc[i,'type'])))
#             st.session_state["req_df"] = reqed_df.copy()
#             # abstracts = []
#             for i in range(len(reped_df)):
#                 if ((reped_df.loc[i,"desc"]==None) or len(reped_df.loc[i,"desc"])==0)and (reped_df.loc[i,'name']!=None):
#                     abstract = "desc:"+apidesc+"#name:"+str(reped_df.loc[i,'name'])+"#type"+str(reped_df.loc[i,'type'])
#                     reped_df.loc[i,"desc"] = gen(abstract)
#             st.session_state["rep_df"] = reped_df.copy()
#             st.experimental_rerun()    
#     # 更新数据库
#     with savecol:
#         if st.button('Save API information', use_container_width=True, type="primary"):
#             result = db.save(apiid, _apipath, typelst.index(_apitype), _apidesc, json.loads(_request), json.loads(_response))
#             if result:
#                 st.success("Save successfully!")
#             else:
#                 st.error("Save error!")
#             st.experimental_rerun()            
#     # 下载
#     with download:
#         jsondata = []
#         for param in json.loads(_request):
#             if param["name"]!=None and param["desc"]!=None:
#                 tmp = {"name":param["name"], "desc":param["desc"]}
#                 jsondata.append(tmp)
#         for param in json.loads(_response):
#             if param["name"]!=None and param["desc"]!=None:
#                 tmp = {"name":param["name"], "desc":param["desc"]}
#                 jsondata.append(tmp)
#         st.download_button(
#             label="Download params(json)",
#             file_name="data.json",
#             mime="application/json",
#             data=json.dumps(jsondata),
#             use_container_width = True
#         )   

# view_sidebar()
# edit(st.session_state["now_id"])


