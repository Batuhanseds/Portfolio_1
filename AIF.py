import pandas as pd
import numpy as np
import pandas_datareader as pdr
#Time
from datetime import datetime, timedelta
from datetime import date
import time
#Graf
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
import cpi
import ta
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#Save Load Data & Model
import pickle
import random
import sys
import multiprocessing as mp

class Aif:
    "This class makes data transformations_"       
    def calculate_Inflate(self, df:pd.DataFrame,norm_colum:list=[],date_col:str='Date',add_subs:str='Real_' ):
        "This function calculate real value for inflate" 
        data=df.copy()
        if len(norm_colum)==0 : norm_colum=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        for c in tqdm(norm_colum):
            for i in (range(0,data.shape[0])):    
                if data[date_col].iloc[i].year < date.today().year:                    
                    data.at[i,add_subs+c] = cpi.inflate(data[c].iloc[i],data[date_col].iloc[i])
                else :
                    data.at[i,add_subs+c] = data[c].iloc[i]                  
        #data=data[[date_col]].merge(data.loc[:,(data.columns.str.contains(add_subs))],how='left',left_index=True, right_index=True)
        data = pd.concat([data[[date_col]], data.loc[:,(data.columns.str.contains(add_subs))]], axis=1)
        return data
    def movingavg_avg_real(self, df:pd.DataFrame,avg_columns:list=[],period:list=[15,30],date_col:str='Date',type_avg:str='SMA'):
        "This function calculate Moving Avg for SMA and EMA values" 
        data = df
        if len(avg_columns)==0 : avg_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        data_ma = pd.DataFrame(data[date_col],columns=[date_col])
        data_ma[date_col]=pd.to_datetime(data_ma[date_col],format='%Y-%b-%d')        
        for c in tqdm(avg_columns):
            #print(c,' - ',data[c].iloc[0])
            for i in period:
                if type_avg =='SMA':
                    data_ma[c+'_'+str(i)+type_avg] = data[c].rolling(window=i).mean()
                elif type_avg =='EMA':
                    data_ma[c+'_'+str(i)+type_avg] = data[c].ewm(span=i,adjust=True,ignore_na=True).mean()
                else :
                    raise Exception("type_avg parameter must be SMA or EMA")
        return data_ma.dropna(axis=0)
    def movingavg_avg(self, df:pd.DataFrame,avg_columns:list=[],period:list=[15,30],date_col:str='Date',type_avg:str='SMA'):
        "This function calculate Moving Avg for SMA and EMA values" 
        data = df
        if len(avg_columns)==0 : avg_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        data_ma = pd.DataFrame(data[date_col],columns=[date_col])
        data_ma[date_col]=pd.to_datetime(data_ma[date_col],format='%Y-%b-%d')        
        for c in tqdm(avg_columns):
            #print(c,' - ',data[c].iloc[0])
            for i in period:
                if type_avg =='SMA':
                    data_ma[c+'_'+str(i)+type_avg] = (data[c]/(data[c].rolling(window=i).mean()))-1
                elif type_avg =='EMA':
                    data_ma[c+'_'+str(i)+type_avg] = (data[c]/(data[c].ewm(span=i,adjust=True,ignore_na=True).mean()))-1
                else :
                    raise Exception("type_avg parameter must be SMA or EMA")
        return data_ma.dropna(axis=0)
    def kama(self, df:pd.DataFrame,value_columns:list =[],window:int=21,date_col:str='Date', 
             pow1:int=2, pow2:int=30, fillna: bool=False,add_subs:str='Kama'):
        ''' kama indicator '''    
        ''' accepts pandas dataframe of prices '''
        ''' value_columns gets columns for kama'''
        ''' This function returns not null values only'''
        data = df.drop(columns=[date_col])
        if len(value_columns)==0 : value_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        data_ma = pd.DataFrame()        
        for c in tqdm(value_columns):
            data_ma[c+'_'+add_subs] = ta.momentum.kama(data[c],window=window, pow1=pow1, pow2=pow2, fillna=fillna) 
        data_ma.dropna(axis=1,how='all',inplace=True)
        data_ma.dropna(axis=0,how='all',inplace=True)        
        data_ma[date_col]=pd.to_datetime(df[date_col],format='%Y-%b-%d')        
        return data_ma
    def rsi(self, df:pd.DataFrame,value_columns:list=[],window:int=14,fillna: bool=False,add_subs='RSI',date_col:str='Date'):        
        ''' RSI indicator '''    
        ''' accepts pandas dataframe of prices '''
        data = df
        if len(value_columns)==0 : value_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        data_ma = pd.DataFrame(data[date_col],columns=[date_col])
        data_ma[date_col]=pd.to_datetime(data_ma[date_col],format='%Y-%b-%d')        
        for c in tqdm(value_columns):
            data_ma[c+'_'+add_subs] = ta.momentum.RSIIndicator(data[c], window=window, fillna=fillna).rsi()            
        return data_ma.dropna(axis=0)
    def intersection(self, df:pd.DataFrame,value_columns:list=[],add_subs='_I'):        
        ''' Intersection '''    
        ''' Accepts pandas dataframe of prices '''
        ''' Give intersection of columns'''
        data = df 
        data_ma  = pd.DataFrame(index=df.index)     
        if len(value_columns)==0 : value_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        for i,c in enumerate((value_columns)):    
            for l,cc in enumerate(value_columns[i+1:]):                
                Signal_col             = c+'_'+cc+'_'+'S'+add_subs                
                Position_col           = c+'_'+cc+'_'+'P'+add_subs                
                Signal_count_col       = c+'_'+cc+'_'+'C'+add_subs                
                data_ma[Signal_col]    = np.where(data[c] > data[cc], 1.0, -1.0)
                data_ma[Position_col]  = data_ma[Signal_col].diff()/2
                index_list=data.index.values.tolist()
                deger=0
                for i,ind in enumerate((index_list)):
                    if i !=0 :                         
                        if data_ma.loc[ind,Signal_col]==data_ma.loc[index_list[i-1],Signal_col]:            
                            deger +=1
                            data_ma.loc[ind,Signal_count_col]=deger
                        else:
                            deger=0
                            data_ma.loc[ind,Signal_count_col]=0
                data_ma[Signal_count_col] = data_ma[Signal_count_col] * data_ma[Signal_col]                                
                data_ma=data_ma.drop([Signal_col], axis=1)        
                #data_ma=data_ma.drop([Position_col], axis=1)        
        return data_ma
    def intersectionN(self, df:pd.DataFrame,value_columns:list=[],add_subs='_I'):        
        ''' Intersection '''    
        ''' Accepts pandas dataframe of prices '''
        ''' Give intersection of columns'''
        data = df 
        data_ma  = pd.DataFrame(index=df.index)     
        if len(value_columns)==0 : value_columns=df.select_dtypes(exclude=['datetime64[ns]']).columns.to_list() 
        for i,c in enumerate((value_columns)):    
            for l,cc in enumerate(value_columns[i+1:]):                
                Signal_col                 = c+'_'+cc+'_'+add_subs                                
                data_ma[Signal_col]  = (data[c] - data[cc])/data[c]                
        return data_ma
def show_buy_sell(data,
            col_val:str   ='XAGUSD_Adj Close',
            col_date:str  ='Date',
            kalici:pd.DataFrame=pd.DataFrame(),                
            **param_dict_other            
           ):
    plt.figure(figsize = (30,15))
    temp=data[[col_date,col_val]]
    plt.plot(temp[col_date],temp[col_val],color = 'b', lw = 1, label = 'Close Price')
    for i in range(0,kalici.shape[0]):
        date_bitis,date_bas = kalici[[col_date,col_date+'_Shift']].iloc[i]        
        if i <1 :
            plt.plot(temp[(temp[col_date] > date_bas) & (temp[col_date] < date_bitis)][col_date],temp[(temp[col_date] > date_bas) & (temp[col_date] < date_bitis)][col_val],color = 'r', lw = 3,label='Hold')
        else:            
            plt.plot(temp[(temp[col_date] > date_bas) & (temp[col_date] < date_bitis)][col_date],temp[(temp[col_date] > date_bas) & (temp[col_date] < date_bitis)][col_val],color = 'r', lw = 3)

        

    plt.ylabel('Price in Dolar', fontsize = 15 )
    plt.xlabel('Date', fontsize = 15 )
    plt.title(col_val+' Model Predict', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()

def trade(data,oran=0.982311, # Gümüşte 0.9823 altında 0.987
            col_val:str   ='XAGUSD_Close',
            col_date:str  ='Date',
            detay:bool=False,
            agents_p_buy_up:pd.Series=pd.Series(),
            agents_p_buy_down:pd.Series=pd.Series(),
            agents_p_sell_up:pd.Series=pd.Series(),
            agents_p_sell_down:pd.Series=pd.Series(),
            agents_w_buy_up:pd.Series=pd.Series(),
            agents_w_buy_down:pd.Series=pd.Series(),
            agents_w_sell_up:pd.Series=pd.Series(),
            agents_w_sell_down:pd.Series=pd.Series(),
            agents_buy_t:pd.Series=pd.Series(),
            agents_sell_t:pd.Series=pd.Series(),                
            **param_dict_other            
           ):        
    agent={
    'agents_p_buy_up'    :agents_p_buy_up,
    'agents_p_buy_down'  :agents_p_buy_down,
    'agents_p_sell_up'   :agents_p_sell_up,
    'agents_p_sell_down' :agents_p_sell_down,
    'agents_w_buy_up'    :agents_w_buy_up,
    'agents_w_buy_down'  :agents_w_buy_down,
    'agents_w_sell_up'   :agents_w_sell_up,
    'agents_w_sell_down' :agents_w_sell_down,
    'agents_buy_t'       :agents_buy_t,
    'agents_sell_t'      :agents_sell_t
        }
    #-------------------------------------
    data_compair=pd.DataFrame()
    temp1=pd.DataFrame()
    iter_type_p ='agents_p_sell_up'
    iter_type_w ='agents_w_sell_up'
    iter_type_0=iter_type_p+"0"
    for key in agent[iter_type_p].keys():
        temp1[key]=np.where(data[key]>agent[iter_type_p][key],agent[iter_type_w][key],0)        
    #-------------------------------------
    temp2=pd.DataFrame()
    iter_type_p ='agents_p_sell_down'
    iter_type_w ='agents_w_sell_down'
    iter_type_0=iter_type_p+"0"
    for key in agent[iter_type_p].keys():
        temp2[key]=np.where(data[key]<agent[iter_type_p][key],agent[iter_type_w][key],0)

    data_compair['sell']=np.where(temp1.sum(axis=1)+temp2.sum(axis=1)>agent['agents_sell_t']['threshold'],1,0)
    #-------------------------------------
    temp1=pd.DataFrame()
    iter_type_p ='agents_p_buy_up'
    iter_type_w ='agents_w_buy_up'
    iter_type_0=iter_type_p+"0"
    for key in agent[iter_type_p].keys():
        temp1[key]=np.where(data[key]>agent[iter_type_p][key],agent[iter_type_w][key],0)        
    #-------------------------------------
    temp2=pd.DataFrame()
    iter_type_p ='agents_p_buy_down'
    iter_type_w ='agents_w_buy_down'
    iter_type_0=iter_type_p+"0"
    for key in agent[iter_type_p].keys():
        temp2[key]=np.where(data[key]<agent[iter_type_p][key],agent[iter_type_w][key],0)
    data_compair['buy']=np.where(temp1.sum(axis=1)+temp2.sum(axis=1)>agent['agents_buy_t']['threshold'],1,0)        
    #-------------------------------------        
    kalici=pd.concat([data_compair[['sell','buy']], data[['Date',col_val]].reset_index(drop=True)], axis=1)  
    #Elde mal kaldıysa en son satıp kar hesaplaması yapılabilsin, en son mal alınmış mı satılmış mı gözükebilsin
    kalici['sell'].iloc[-1]=1

    #sadece işlemi olanlar 
    kalici=kalici[((kalici['buy']==1) | (kalici['sell']==1))]
    #Kullanılacak olan değerlerin bir öncekileri shift ediliyor
    kalici=pd.concat([kalici,
                      kalici[[col_val,'Date','sell','buy']].shift(1).rename(columns={'Date':'Date_Shift',col_val:col_val+'_Shift','sell': 'sell_shift','buy':'buy_shift'})], 
                     axis=1
                    )
    #--------------------------------
    #Alımdan sonra satış gelmeden tekrar alım isteklerini siliyoruz, shift kullandığımız için ilk alışa ulaşmalıyız
    kalici=kalici[~(
                    (kalici['buy']==1) & 
                    (kalici['buy_shift']==1) &  
                    (kalici['sell']!=1)
                  )
                 ]
    #Tekrarlıyan sel işlemlerini kaldırıyoruz, alış olmadan tekrarlıyan satışlara gerek yok
    kalici=kalici[~(
                    (kalici['sell']==1) & 
                    (kalici['sell_shift']==1) &  
                    (kalici['buy']!=1)
                  )
                 ]
    #Alış yapmış sonra satış yapmışları belirliyoruz
    kalici=kalici[(
            ((kalici['buy_shift']==1) & (kalici['sell']==1)) 
                  )
                ]
    #kalici['kar']=(1+((kalici['XAGUSD_Adj Close']- kalici['XAGUSD_Adj Close_Shift'])/kalici['XAGUSD_Adj Close_Shift']))*oran
    kalici['kar']=((kalici[col_val]/kalici[col_val+'_Shift'])*oran)
    return_val=np.prod(kalici[['kar']].T.values)

    if (detay==True):            
        return return_val,  kalici
    else:
        return return_val