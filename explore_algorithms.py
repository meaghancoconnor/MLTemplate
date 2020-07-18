# -*- coding: utf-8 -*-

import pandas as pd

data_file = input("Enter file path for csv data: ")
print(data_file)

test_data = pd.read_csv(data_file, index_col=0)

def explore_algorithms(data: pd.DataFrame, supervised: bool = True, 
                       y: str = None, pred_type: str = 'class',
                       text_data: bool = False) -> pd.DataFrame:
    '''
    Based on the pred_type, runs prediction on a set of algorithms and returns
    loss for each
    
    Parameters
    -------
    data = pandas Dataframe of shape [n_observations, n_features+1]
    supervised: bool, default = True
        If true, y value is the labeled data
    y : str, default = None
        The column to be predicted
    pred_type: str, default = 'class'
        Type of prediction either class for classification or reg for
        regression
    text_data: bool, default = 'False'
        If True, then the dataset is comprised of text for analysis
    
    Returns
    -------
    summary: pandas Dataframe of shape [n_algorithms, 2]
        training loss function and dev loss function for each algorithm
    '''
    # Supervised Learning
    assert(pred_type in ['class','reg']), 'use "class" to predict a category\
        or "reg" to predict a numeric value'
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score    
    
    Xtrain,Xdev,ytrain,ydev = train_test_split(data.drop(y,axis=1),data[y],
                                               test_size = .3, 
                                               random_state = 117) 
    def fit_model(model):
        model.fit(Xtrain,ytrain)
        return(accuracy_score(ytrain,model.predict(Xtrain)),
               accuracy_score(ydev,model.predict(Xdev)))
    
    if supervised:
        assert y != None, 'need to specify which column contains labels'
        assert(y in data.columns), 'y column not in the dataset'
        
        models = []
        
        if pred_type == 'class': # Classification Problems
            summary = pd.DataFrame(columns = ['TrainingAccuracy',
                                             'DevAccuracy'])
            if data.shape[0] < 100000: # Less than 100,000 samples
                from sklearn.svm import LinearSVC
                models.append(('LinearSVC',LinearSVC()))
                from sklearn.tree import DecisionTreeClassifier
                models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
                if text_data:
                    from sklearn.naive_bayes import GaussianNB
                    models.append(('GaussianNB', GaussianNB()))
                else:
                    from sklearn.neighbors import KNeighborsClassifier
                    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
                    from sklearn.svm import SVC
                    models.append(('SVC',SVC()))
            else: # More than 100,000 samples
                algorithms = ['SGD Classifier']
                from sklearn.linear_model import SGDClassifier
                models.append('SGDClassifier', SGDClassifier())
                            
            
            for name, model in models:
                summary.loc[name, ['TrainingAccuracy',
                                   'DevAccuracy']] = fit_model(model)
            return summary
        
        else: # Regression Problems
            # Algorithms are ['Lasso','ElasticNet','SVR(kernal="rbf"'
            #                 'Ridge Regression','SVR(kernal="linear"']                    
            print('Coming Soon')
    # Unsupervised Learning
    else:
        print('Coming Soon')
    
print(explore_algorithms(test_data, y = 'Survived'))
    

                