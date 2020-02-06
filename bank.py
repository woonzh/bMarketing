import pandas as pd
from utils import processor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import time

start=time.time()
timeInt=start
timeStore={}

def plotMatplot():
    fig=plt.figure()
    
    ax=Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    vals=[0,1]
    colours=['r','b']
    
    for val, col in zip(vals, colours):
        subDf=pcaResults[pcaResults[target]==val]
        ax.scatter(subDf['pc1'],subDf['pc2'],c=col)
    
    ax.legend=vals
    ax.grid()
    
def takeTime(naming, complete=False):
    global timeInt
    
    if complete==False:
        timeStore[naming]=str(round((time.time()-timeInt)/60,2))
        timeInt=time.time()
    else:
        timeStore[naming]=str(round((timeInt-start)/60,2))
        for i in timeStore:
            print('%s: %s'%(i,timeStore[i]))
    

df=pd.read_csv('bank-additional-full.csv', sep=';')

target='y'
name='main'
runpca=False

pc=processor()
cleanDf, trainDf, testDf=pc.insertNewDf(df, name, target, runPCA=runpca)
takeTime('cleaning')

if runpca==False:
    pcaResults=pc.runpca(name)
    takeTime('PCA')

#svmResults=pc.runSVM(name)
#takeTime('SVM')
#
#logResults=pc.runLogs(name)
#takeTime('Logs')
#    
#xgBoost=pc.runXGBoost(name)
#takeTime('xgBoost')
    
rndForrest=pc.runRandomForest(name)
takeTime('RandomForrest')

takeTime('total', True)