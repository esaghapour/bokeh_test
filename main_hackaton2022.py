# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 00:20:15 2022

@author: ehsan
"""

# from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model, predict_model
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode
st.set_page_config(
    layout="wide")


def plt_boundry_drug(thr,out1,sample_name,style):
      lists=[]
      for i in range(1,8):
          lists.append([i-.3,i+.3,None])
      x = pd.concat([pd.Series(x) for x in lists], axis=0)

      lists=[]
      lists1=[]
      for minn,maxx in zip(thr['Min'] ,thr['Max']):
          lists.append([minn, minn, None])
          lists1.append([maxx, maxx, None])
      y = pd.concat([pd.Series(x) for x in lists], axis=0)
      y1 = pd.concat([pd.Series(x) for x in lists1], axis=0)

      
      fig = go.Figure()
      fig.add_trace(go.Scatter(
          x=np.arange(1,8), 
          y=out1["Label"],
          name=str(sample_name),
          mode='markers'
      ))
      if style==1:
          fig.add_trace(go.Scatter(
                x=x,
                y=y1[0:21],
                name='Max threshold',
                mode='lines'
            ))
          fig.add_trace(go.Scatter(
              x=x,
              y=y[0:21],
              name='Min threshold',
              mode='lines',
              
    
          ))
          layout = dict(
              xaxis=dict(
                  
                  tickvals=np.arange(1,8).astype(int),
                  ticktext=out1["DRUG_NAME"].to_list(),

              )
              ,width=1200, height=500,
              yaxis_title="IC50 Value",
              xaxis_title= "DRUG_NAME",
              legend_title="Sample_ID",
              title='Drug Screenign Result with Min-Max on Threshold Based on Literature Survey'
          )
      if style==0:
           fig.add_trace(go.Scatter(
                 x=x,
                 y=y1[21:],
                 name='Max threshold',
                 mode='lines'
             ))
           fig.add_trace(go.Scatter(
               x=x,
               y=y[21:],
               name='Min threshold',
               mode='lines'
     
           ))
           layout = dict(
               xaxis=dict(
                   
                   tickvals=np.arange(1,8).astype(int),
                   ticktext=out1["DRUG_NAME"].to_list(),

               )
               ,width=1200, height=500,
               yaxis_title="IC50 Value",
               xaxis_title= "DRUG_NAME",
               legend_title="Sample_ID",
               title='Drug Screenign Result with Min-Max on Threshold Based on GDSC Data'
           )
      
      
      fig.update_layout(layout)
      fig.update_yaxes(range = [-7,7.5])
      
      st.plotly_chart(fig)


st.title('AI-based Personalized Drug Screening for Cancer Treatment Web App (GBM Cancer case study)')
image = Image.open('logo1.png')
st.sidebar.image(image)

st.subheader('Team Name : MedvAIsor')

with st.expander('Description'):
     st.markdown('The MedvAIsor acronym is a fusion of the word Medicine + (Advisor + AI). This tool showcases a GBM case '
                 +'where we can perform personalized virtual drug screening for seven FDA-approved GBM drugs. The tool takes the '
                 +'GBM cancer cell line of 657 genes as input and predicts the efficacy of the seven drugs. '
                 +'The tool screens individual cancer cell lines and shows the effectiveness of each drug along with its threshold values. '
                 )
with st.expander('Project Significant'):
     st.markdown('**Abstract**')
     st.markdown('The MedvAIsor acronym is a fusion of the word Medicine + (Advisor + AI).' 
                +'Personalized drug screening is one of the vital drug development phases for precision oncology.'+ 
                'In this phase, each patient is screened for the efficacy of a given drug on a particular type of cancer. The in-vivo and in-vitro models for screening has limitation such as cost and time. In contrast, the in-silico methods can reduce both. In this project, we take the problem of personalized drug screening and try to predict the efficacy based on the omics profile of cancer patients. Drug effect on tumors is the key to solving this problem; therefore, we intend drug perturbation data on cancer cell lines to build the predictive models.  In this project, we also aim to develop a web-based prototype of a recommender system that uses these in-silico drug screening models and provides monotherapy treatment suggestions ( based on patients’ omics profiles) to the doctors.')
     st.markdown('**Method**')
     st.markdown('The project has two components: first, the ML modeling, which contributes to' 
             +'the scientific community where we intend to develop novel in-silico methods for '
             +'personalized drug screening; Second, a tool where a doctor provides the omics profile of the patients'
             +'tumor and the software provides a list of drugs ranked by their efficacy. '
             +'Besides treatment benefits, This project also has an economic impact because it '
             +'has the potential for commercialization and generation of employment.'
             +'Figure below shows the proposed method.')
     image = Image.open('Proposed_method.png')
     st.image(image)
     st.markdown('**Database**')
     st.markdown('Many databases provide drug perturbation information on cancer tumors following are a few, and we chose Genomics of Drug Sensitivity of Cancer (GDSC) for this study. Please see this [link](https://www.cancerrxgene.org/)')

with st.expander('Steps to use the tool:'):
     st.markdown('1. Load the CSV file containing the gene expression of patients. ' 
                 +'The example CSV can be downloaded from [link](https://github.com/u-brite/MedvAIsor/blob/main/App/AllGE_GBM_657.csv. )'
                 +'The first row in the CSV file is the header, and the rest of the rows are gene expression values. '
                 +'The First column is the ID of the sample (Integer value), and the rest columns are 657 genes as described '
                 +'(see the supplementary section of Hackathon Project reports at '
                 +'[link](https://github.com/u-brite/MedvAIsor/blob/main/Project_Report.docx))'
                 )
     st.markdown('2. Select a single sample to view the screening results, i.e., LN(IC50), of seven drugs based ' 
                 +'on the patient’s gene expression. This result also shows the minimum and maximum '
                 +'threshold ranges for each drug. If the predicted LN(IC50) values fall within the range, then '
                 +'the drug can be termed as effective. '+'**OR** '
                 +'Select multiple samples to see the efficacy of the drugs on multiple samples together.'
                 )
     st.markdown('3. After selecting a “Predict” button will appear below the sample table. Press the button to view the plots.'
                 )
     st.markdown('**For more details, please see related github ([link 1](https://github.com/u-brite/MedvAIsor)) and poject report ([link 2](https://github.com/u-brite/MedvAIsor/blob/main/Project_Report.docx)).**')
        
        
uploaded_files = st.sidebar.file_uploader("Choose a CSV file")
# if uploaded_files:
with st.sidebar.expander('Team member:'):
    st.markdown('1. Arsalan Ahmad | aahmad@asfaschool.org | Data Analysis work')

    st.markdown('2. DaVonte Curtis | davontecurtis4@gmail.com | Product Developer')
    
    st.markdown('3. Kermit Glenn Booker Jr | kermit.booker@bulldogs.aamu.edu | Product Developer')
    
    st.markdown('4. Rizwan Ahmad | rizwanahmad95@outlook.com | Data Analysis work')
    
    st.markdown('5. Mary Doamekpor | mdoam5@uab.edu | Biologist')
    
    st.markdown('6. Radomir Slominski | rslom@uab.edu | Biologist')
    
    st.markdown('7. Dr. Ehsan Saghapour | esaghapo@uab.edu | Machine Learning and Data Science')
    
    st.markdown('8. Dr. Rahul Sharma | rsharma3@uab.edu | Team Lead')
    
    st.markdown('9. Dr. Jake Y. Chen | jakechen@uab.edu | Mentor')
    
if uploaded_files:
    thr=pd.read_csv('Threshold_drug.csv')
    features=pd.read_csv('feature_exp11.csv')
    fetatures = features['Feature'].to_list()
    Disease_feature=fetatures[:31]
    drug_features = fetatures[31:31+1968]
    gene_features = fetatures[31+1968:]
    drugs = pd.read_csv('gbm_drugs_FE.csv')
    drug_list = ['Temozolomide', 'Carmustine' ,'Topotecan' ,'Crizotinib' ,'Gefitinib' ,'Bortezomib' ,'Teniposide']
    # shows =pd.read_csv('AllGE_GBM_657.csv',sep=',',index_col=False)
    shows =pd.read_csv(uploaded_files,sep=',',index_col=False)
    
    st.text('')
    st.text('')

    st.markdown('**Choose Sample/s ID:**')
    gb = GridOptionsBuilder.from_dataframe(shows)
    # enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
    gridOptions = gb.build()
    
    response = AgGrid(
        shows,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )
    ge = pd.DataFrame(response["selected_rows"])
    # st.table(ge.iloc[:,1:])
    if len(ge)>0:
        ge =ge.iloc[:,1:]
        ge['TCGA_DESC']='GBM'
        for cn in Disease_feature:
            if cn =='GBM':
                ge[cn]=1
            else:
                ge[cn]=0
        ge = ge.assign(key=1).merge(drugs.assign(key=1),how='outer',on='key')
        ge.drop(columns=['key'],inplace=True)
        ge =ge[Disease_feature+drug_features+gene_features+['TCGA_DESC','SAMPLE_ID','DRUG_NAME']]
        
        
        
        # ge = pd.read_csv('AllGE_GBM_657.csv')
        # st.table(df)
        
        # ge['TCGA_DESC']='GBM'
        model=load_model('Exp1_hackathon')
        # col1,col2=st.columns([1, 2])
        # with col1:
        
        if st.button('Predict'):
            unseen_predict1 =predict_model(model, data=ge)
            
            out=unseen_predict1[unseen_predict1.columns[-4:]]
            # st.table(out)
            
            if len(np.unique(out['SAMPLE_ID']))==1:
                sample_name=out['SAMPLE_ID'][0]
                idx=np.where(out['SAMPLE_ID']==sample_name)[0]
                out1=out.iloc[idx,:]
                plt_boundry_drug(thr,out1,sample_name,1) 
                plt_boundry_drug(thr,out1,sample_name,0) 
            else:
                import plotly.express as px
                fig = px.scatter(out, x="DRUG_NAME", y="Label", symbol=out['TCGA_DESC'].astype(str), color=out['SAMPLE_ID'].astype(str))
                fig.update_layout(
                   
                    yaxis_title="IC50 Value",
                    legend_title="Sample_ID, Cancer Type",
                    width=1000, height=500)
                fig.update_traces(marker=dict(size=10,
                                              line=dict(width=2,
                                                        color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
                st.plotly_chart(fig)
                
            

