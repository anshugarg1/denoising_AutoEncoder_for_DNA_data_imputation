import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random

# %%
###########################
#Load data
###########################
def load_data(path):
    a = pd.read_json(path, lines=True,typ='frame')  #load data fromn notepad to dataframe
    return a

# %%
###########################
# Retrun unique DNAs
###########################

def uniSnpDnaDf(a):
    uArr = pd.unique(a[['snpMother', 'snpSelf', 'snpFather']].values.ravel('K'))   ### (i) make single dim array in row major form and (ii) keep only unique values
    uDF = pd.DataFrame(uArr, columns=['snpDNA'])  
    uDF.dropna(axis = 0, inplace=True)  #drop None values
    return uDF

# %%
def main_f(path):
#     path = '../../data/DeepIntegrate_Data_HOH_2019.txt'
    a = load_data(path)
    print("Columns of Dataframe - ", a.columns)
    print("Number of records initially - ", len(a))
    
    uDF = uniSnpDnaDf(a)
    return uDF

###########################
# Retrun the one-hot-encoder
###########################

def noiseIntro(l, noisePercentage):
    rn = random.sample(range(0, 13714), noisePercentage)
    for i in np.arange(noisePercentage):
        l[rn[i]] = '-'
    return l


def ohe_enc(udf, ls, flag, noisePercentage):
    ohe = OneHotEncoder(drop='first')   

    for i in np.arange(len(udf)):
        one_recd = list(udf['snpDNA'][0])
        if(flag==1):
            noiseIntro(one_recd, noisePercentage)
        a = np.array(one_recd).reshape(-1, 1) ### list() will make the list from string, reshape to n*1 array, 
        eohe = ohe.fit_transform(a).todense()   ## convert to ohe in matrix format
        ls[i] = eohe.T

    return ls

# %%
def UniqueNoisyDNASequences(path):
    uDF = main_f(path)
    
    totalDNASequence = len(uDF['snpDNA']) #1314
    DNALength = len(uDF['snpDNA'][0]) #13714
    encodingLen = 8 
    noisePercentage = len(uDF['snpDNA'][0])*.2  #2742
    
    ls = np.ones((totalDNASequence, encodingLen, DNALength))
    noisyX = ohe_enc(uDF, ls, 1, noisePercentage)
    orgY = ohe_enc(uDF, ls, 0, noisePercentage)
    
    DNAListX = np.array(noisyX)
    DNAListY = np.array(orgY)
    return DNAListX, DNAListY



