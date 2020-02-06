import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import xgboost as xgb

class processor:
    def __init__(self):
        self.raw={}
        self.labels={}
        self.clean={}
        self.train={}
        self.test={}
        self.targetCol={}
        self.corr={}
        self.le=LabelEncoder()
        self.models={}
        self.testSplit=0.2
        self.corrCutOff=0.1
        self.PCADimensions=3
    
    def insertNewDf(self,df, name, tarCol, runPCA=False):
        self.targetCol[name]=tarCol
        self.raw[name]=df
        self.convertCategorical(name)
        self.removeLowCorr(name)
        if runPCA:
            cmp= self.runpca(name, parx=self.clean[name].drop(columns=tarCol), pary=self.clean[name][[tarCol]])
            self.clean[name]=cmp
        self.splitData(name)
        
        return self.clean[name], self.train[name],self.test[name]
    
    def convertCategorical(self, name):
        df=self.raw[name]
        
        dtypes=df.dtypes
        catCols=list(dtypes[dtypes=='object'].index)
        
        self.labels[name]={}
        
        for col in catCols:
            self.le.fit(df[col])
            df[col]=self.le.transform(df[col])
            self.labels[name][col]=list(self.le.classes_)
        
        self.clean[name]=df
        
    def splitData(self, name):
        train, test = train_test_split(self.clean[name], test_size=self.testSplit)
        train.reset_index(inplace=True, drop=True)
        test.reset_index(inplace=True, drop=True)
        self.train[name]={}
        self.train[name]['y']=train[self.targetCol[name]]
        self.train[name]['x']=train.drop(columns=self.targetCol[name])
        self.test[name]={}
        self.test[name]['y']=test[self.targetCol[name]]
        self.test[name]['x']=test.drop(columns=self.targetCol[name])
    
    def removeLowCorr(self, name):
        tarCol=self.targetCol[name]
        df=self.clean[name]
        corr=df.corr()[tarCol].to_frame()
        self.corr[name]=corr
        corr[tarCol]=[abs(x) for x in corr[tarCol]]
        
        keptLabels=list(corr[corr[tarCol]>=self.corrCutOff].index)
        self.clean[name]=self.clean[name][keptLabels]
    
    def runpca(self, name, parx=None, pary=None):
        if parx is None or pary is None:
            x=self.train[name]['x']
            y=self.train[name]['y']
        else:
            x=parx
            y=pary
            
        df=StandardScaler().fit_transform(x)
        components=self.PCADimensions
        pca=PCA(n_components=components)
        pc = pca.fit_transform(df)
        cmp=pd.DataFrame(data=pc, columns=['pc'+str(x+1) for x in range(components)])
        
        var=np.var(pc,axis=0)
        varRatio=var/np.sum(var)
        print('PAC variance ratio: %s' %str(varRatio))
        
        cmp[self.targetCol[name]]=y
        
        return cmp
    
    def cmpScore(self, prediction, ans, test, name):
        results=[1 if x==y else 0 for x,y in zip(prediction, ans)]
        accuracy=str(sum(results)/len(results))
        compilation=pd.DataFrame()
        compilation['prediction']=prediction
        compilation['ans']=ans
        compilation['result']=results
        
        print('%s-%s score: %s'%(name, test, accuracy))
        
        return compilation
        
    
    def runSVM(self, name):
        train_x=self.train[name]['x']
        train_y=self.train[name]['y']
            
#        params={
#            'kernel':['linear'],
#            'decision_function_shape':['ovo'],
#            'gamma':[1e-1, 1, 1e1],
#            'C':[1e-2, 1, 1e2]
#                }
#        clf=svm.SVC(probability=True)
#        grid=GridSearchCV(clf, param_grid=params, cv=2)
#        grid.fit(train_x,train_y)
#        print(grid.best_params_)
#        return grid.best_params_
    
        clf=svm.SVC(probability=True, kernel='linear', decision_function_shape='ovo', gamma=1e-1, C=0.01)
        
        clf.fit(train_x,train_y)
        
        try:
            self.models[name]['svm']=clf
        except:
            self.models[name]={}
            self.models[name]['svm']=clf
        
        test_x=self.test[name]['x']
        prediction=clf.predict(test_x)
        
        test_y=self.test[name]['y']
        
        compilation=self.cmpScore(prediction,test_y,'SVM',name)
        
        return compilation
        
    def runLogs(self,name):
        clf=LogisticRegression()
        clf.fit(self.train[name]['x'],self.train[name]['y'])
        
        test_x=self.test[name]['x']
        prediction=clf.predict(test_x)
        test_y=self.test[name]['y']
        
        compilation=self.cmpScore(prediction,test_y,'Logs',name)
        
        return compilation
    
    def runXGBoost(self, name):
        train=xgb.DMatrix(self.train[name]['x'].to_numpy(),self.train[name]['y'].to_numpy())
        test=xgb.DMatrix(self.test[name]['x'].to_numpy(), self.test[name]['y'].to_numpy())
        
        params={
            'eta':0.3,
            'max_depth':6,
            'objective':'binary:logistic'
#            'num_class':2
                }
        steps=20
        
        model = xgb.train(params, train, steps)
        probability=model.predict(test)
        prediction=[1 if x>=0.5 else 0 for x in probability]
#        print(accuracy_score(prediction,self.test[name]['y']))
        compilation=self.cmpScore(prediction,self.test[name]['y'],'XGBoost',name)
        return compilation
    
    def runRandomForest(self, name):
        params={
            'n_estimators':[100],
            'max_depth':[10, 50],
            'min_samples_split':[2,5,10]
                }
        clf=RandomForestClassifier()
        grid=GridSearchCV(clf, param_grid=params, cv=2)
        grid.fit(self.train[name]['x'],self.train[name]['y'])
        
        print(grid.best_params_)
        
        clf=RandomForestClassifier(**grid.best_params_)
        clf.fit(self.train[name]['x'],self.train[name]['y'])
        pred=clf.predict(self.test[name]['x'])
        
        compilation=self.cmpScore(pred,self.test[name]['y'],'randomForrest', name)
        
        return compilation