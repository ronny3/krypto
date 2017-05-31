#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciKit-learn models
Basic stuff:
model.fit(X,y)
model.predict(x)
"""

import inspect
class LinearRegressionModels():
    """ http://scikit-learn.org/stable/modules/linear_model.html
    """
    def __init__(self):
        self.__models = ['Linear', 'Ridge', 'Lasso', 'MutliTaskLasso', 'ElasticNet'
                         'MultiTaskElasticNet', 'Lars', 'LassoLars', 'OrthogonalMatchingPursuit'
                         ,'BayesianRidge', 'ARDRegression', 'SGDRegressor',
                         'PassiveAggressiveRegressor', 'RANSACRegressor', 'TheilSenRegressor',
                         'HuberRegressor', 'PolynomialtoLinear']
    
    def ModelNames(self):
        return self.__models
    def Linear(self, random_=False, **kwargs):
        from sklearn.linear_model import LinearRegression as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
        return model(**kwargs)
        
    def Ridge(self, random_=False, **kwargs):
        from sklearn.linear_model import Ridge as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
    
    def KernelRidge(self, random_=False, **kwargs):
        from sklearn.kernel_ridge import KernelRidge as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def Lasso(self, random_=False, **kwargs):
        from sklearn.linear_model import Lasso as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def MutliTaskLasso(self, random_=False, **kwargs):
        from sklearn.linear_model import MultiTaskLasso as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def ElasticNet(self, random_=False, **kwargs):
        from sklearn.linear_model import ElasticNet as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def MultiTaskElasticNet(self, random_=False, **kwargs):
        from sklearn.linear_model import MultiTaskElasticNet as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def Lars(self, random_=False, **kwargs):
        from sklearn.ensemble import Lars as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def LassoLars(self, random_=False, **kwargs):
        from sklearn.ensemble import LassoLars as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def OrthogonalMatchingPursuit(self, random_=False, **kwargs):
        from sklearn.ensemble import OrthogonalMatchingPursuit as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def BayesianRidge(self, random_=False, **kwargs):
        from sklearn.ensemble import BayesianRidge as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def ARDRegression(self, random_=False, **kwargs):
        from sklearn.ensemble import ARDRegression as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def SGDRegressor(self, random_=False, **kwargs):
        from sklearn.ensemble import SGDRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def PassiveAggressiveRegressor(self, random_=False, **kwargs):
        from sklearn.ensemble import PassiveAggressiveRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def RANSACRegressor(self, random_=False, **kwargs):
        from sklearn.ensemble import RANSACRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def TheilSenRegressor(self, random_=False, **kwargs):
        from sklearn.ensemble import TheilSenRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def HuberRegressor(self, random_=False, **kwargs):
        from sklearn.ensemble import HuberRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def PolynomialtoLinear(self, random_=False, **kwargs):
        from sklearn.ensemble import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        if random_:
            argspec = inspect.getfullargspec(LinearRegression)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
        model = Pipeline([('poly', PolynomialFeatures()),
                          ('linear', LinearRegression(**kwargs))])

        return model
        
    def RandomModel(self, random_=False, **kwargs):
        import random
        choice = random.choice(self.__models)
        return getattr(self, choice)(random_=True,**kwargs)
    
        
class SVMs():
    """http://scikit-learn.org/stable/modules/lda_qda.html
    """
    
    def __init__(self):
        self.__models = ['SVM_SVR', 'SVM_NuSVR', 'SVM_LinearSVR']
    
    def ModelNames(self):
        return self.__models
        
    def RandomModel(self, random_=False, **kwargs):
        import random
        choice = random.choice(self.__models)
        return getattr(self, choice)(random_=True,**kwargs)
        
        
    def SVM_SVR(self, random_=False, **kwargs):
        from sklearn.svm import SVR as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def SVM_NuSVR(self, random_=False, **kwargs):
        from sklearn.svm import NuSVR as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def SVM_LinearSVR(self, random_=False, **kwargs):
        from sklearn.svm import LinearSVR as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    
class NearestNeighbors():
    def __init__(self):
        self.__models = ['KNeighborsRegressor']
    
    def ModelNames(self):
        return self.__models
        
    def RandomModel(self, random_=False, **kwargs):
        import random
        choice = random.choice(self.__models)
        return getattr(self, choice)(random_=True,**kwargs)
        
    def KNeighborsRegressor(self, random_=False, **kwargs):
        from sklearn.neighbors import KNeighborsRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
class GaussianProcesses():
    def __init__(self):
        1
    
    def ModelNames(self):
        return self.__models
        
    def RandomModel(self, random_=False, **kwargs):
        import random
        choice = random.choice(self.__models)
        return getattr(self, choice)(random_=True,**kwargs)
        
    TODO::
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class RegressionEnsembles():
    """ http://scikit-learn.org/stable/modules/ensemble.html
    """
    
    def __init__(self):
        self.__models = ['Bagging', 'RandomForest', 'ExtraTrees', 'AdaBoost', 
                         'GradientBoosting']
    def ModelNames(self):
        return self.__models
        
    def Bagging(self, random_=False, **kwargs):
        from sklearn.ensemble import BaggingRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def RandomForest(self, random_=False, **kwargs):
        from sklearn.ensemble import RandomForestRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def ExtraTrees(self, random_=False, **kwargs):
        from sklearn.ensemble import ExtraTreesRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
                
        
    def AdaBoost(self, random_=False, **kwargs):
        from sklearn.ensemble import AdaBoostRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def GradientBoosting(self, random_=False, **kwargs):
        from sklearn.ensemble import GradientBoostingRegressor as model
        if random_:
            argspec = inspect.getfullargspec(model)
            for key in list(kwargs):
                if key not in argspec.args:
                    del kwargs[key]
                    
        return model(**kwargs)
        
    def RandomModel(self, random_=False, **kwargs):
        import random
        choice = random.choice(self.__models)
        return getattr(self, choice)(random_=True,**kwargs)
        
        
k = RegressionEnsembles()
kwargs = {'n_jobs':2, 'kakka':5}
model = k.RandomModel(**kwargs)
model.get_params()
x=np.zeros((122,2))
y=np.zeros((122))
model = model.fit(x,y)
x1=np.zeros((1,2))
l = model.predict(x1)        
    
