from copy import deepcopy
import numpy as np



# _________________________________________________________________ResponseModel
class ResponseModel():
    # Supporting Features: 'group'
    # Radcliffe2011: "The Unfortunately Named ‘Response’ Model"

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Response'
        self.target = 'converted'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        g0_mask = (X['group'] == 0.0)
        features = [f for f in X.columns if f not in ['group', 'converted']]
        self.estimator.fit(
            X.ix[g0_mask, features],
            y[g0_mask]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]
        return self.estimator.predict_proba(X.loc[:, features])


# __________________________________________________________ResponseRevenueModel
class ResponseRevenueModel():
    # Supporting Features: 'group'
    # Radcliffe2011: "The Unfortunately Named ‘Response’ Model"

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'ResponseRevenue'
        self.target = 'revenue'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        g0_mask = (X['group'] == 0.0)
        features = [
            f for f in X.columns
            if f not in ['group', 'label', 'converted', 'timestamp']
        ]
        self.estimator.fit(
            X.ix[g0_mask, features],
            y[g0_mask]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [
            f for f in X.columns
            if f not in ['group', 'label', 'converted', 'timestamp']
        ]
        return self.estimator.predict(X.loc[:, features])



# ______________________________________________NaiveVariableTransformationModel
class NaiveTransformationModel():
    # Supporting Features: None

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'NaiveTransformation'
        self.target = 'label'
        self.extra_features = None
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.estimator.fit(X, y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)



# ___________________________________________________VariableTransformationModel
class JaskowskiTransformationModel():
    # Supporting Features: None
    # Jaśkowski2012:
    # PT(Y=1|X1,...,Xm) - PC(Y=1|X1,...,Xm) = 2*P(Z=1|X1,...,Xm) - 1

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Jaskowski'
        self.target = 'label'
        self.extra_features = None
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.estimator.fit(X, y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        return 2.0*self.estimator.predict_proba(X) - 1.0



# _____________________________________________________SzymonTransformationModel
class SzymonTransformationModel():
    # Supporting Features: None
    # Szymon: continuous response transformation 

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Szymon'
        self.target = 'revenue_neu'
        self.extra_features = None
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.estimator.fit(X, y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        return self.estimator.predict(X)



# _____________________________________________DiscreteSzymonTransformationModel
class DiscreteSzymonTransformationModel():
    # Supporting Features: None
    # Szymon: response discretization 

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'DiscreteSzymon'
        self.target = 'revenue_discrete'
        self.extra_features = None
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.estimator.fit(X, y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)



# _____________________________________________________________________LaisModel
class LaisTransformationModel():
    # Supporting Features: None
    # Lai2004 / Shaar2016:
    # PT(Y=1|X1,...,Xm) - PC(Y=1|X1,...,Xm) =
    #   P(Z=1|X1,...,Xm)*P(Z=1) - P(Z=0|X1,...,Xm)*P(Z=0)

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Lai'
        self.target = 'label'
        self.extra_features = None
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        length = len(y)
        # group==0.0 is the treatment group, group==1.0 is control
        # positive is: (group==0.0,converted==1.0)&(group==1.0,converted==0.0)
        self.pos_ratio = y.sum()/length
        self.neg_ratio = (length-y.sum())/length
        self.estimator.fit(X, y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)

        proba[:,1] = proba[:,1]*self.pos_ratio - proba[:,0]*self.neg_ratio
        proba[:,0] = 1.0 - proba[:,1]

        return proba



# ______________________________________________________________________TwoModel
class NaiveTwoModel():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator_1 = estimator
        self.estimator_2 = deepcopy(self.estimator_1)

        # set mandatory variables used by SLIB
        self.name = 'NaiveTwoModel'
        self.target = 'converted'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        # treatment group
        g0_mask = (X['group']==0.0)
        # control group
        g1_mask = (X['group']==1.0)

        features = [f for f in X.columns if f not in ['group', 'converted']]

        # train the classifiers
        self.estimator_1.fit(
            X.ix[g0_mask, features],
            y[g0_mask]
        )
        self.estimator_2.fit(
            X.ix[g1_mask, features],
            y[g1_mask]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        r1 = self.estimator_1.predict_proba(X.ix[:, features])
        r2 = self.estimator_2.predict_proba(X.ix[:, features])

        return r1-r2


# __________________________________________________________NaiveTwoModelRevenue
class NaiveTwoModelRevenue():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator_1 = estimator
        self.estimator_2 = deepcopy(self.estimator_1)

        # set mandatory variables used by SLIB
        self.name = 'TwoModelRevenue'
        self.target = 'revenue'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        # treatment group
        g0_mask = (X['group']==0.0)
        # control group
        g1_mask = (X['group']==1.0)

        features = [f for f in X.columns if f not in ['group', 'converted']]

        # train the classifiers
        self.estimator_1.fit(
            X.ix[g0_mask, features],
            y[g0_mask]
        )
        self.estimator_2.fit(
            X.ix[g1_mask, features],
            y[g1_mask]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        r1 = self.estimator_1.predict(X.ix[:, features])
        r2 = self.estimator_2.predict(X.ix[:, features])

        return r1-r2



# _______________________________________________________________________LoModel
class LoModel():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Lo'
        self.target = 'converted'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        df = self.generate_interactions(X)

        # train the classifiers
        self.estimator.fit(
            df.ix[:, self.features],
            y[:]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        df = self.generate_interactions(X, 1)
        r1 = self.estimator.predict_proba(df.ix[:, self.features])

        df = self.generate_interactions(X, 0)
        r2 = self.estimator.predict_proba(df.ix[:, self.features])

        return r1-r2

    # __________________________________________________________________________
    def generate_interactions(self, X, group=None):
        df = deepcopy(X)
        self.features = [f for f in df.columns if f not in ['converted']]

        # Must invert since the method needs T=1, C=0, we have T=0, C=1!
        df['group'] = ~df['group']

        interaction_features = []

        for f in self.features:
            iFeature = 'I-'+f
            if group==None:
                df[iFeature] = df[f]*df['group']
            else:
                df[iFeature] = df[f]*group
            interaction_features.append(iFeature)

        if group!=None:
            df['group']=group

        self.features += interaction_features

        assert('group' in self.features)

        return df


# ________________________________________________________________LoRevenueModel
class LoRevenueModel():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'LoRevenue'
        self.target = 'revenue'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        df = self.generate_interactions(X)

        # train the classifiers
        self.estimator.fit(
            df.ix[:, self.features],
            y[:]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        df = self.generate_interactions(X, 1)
        r1 = self.estimator.predict(df.ix[:, self.features])

        df = self.generate_interactions(X, 0)
        r2 = self.estimator.predict(df.ix[:, self.features])

        return r1-r2

    # __________________________________________________________________________
    def generate_interactions(self, X, group=None):
        df = deepcopy(X)
        self.features = [f for f in df.columns if f not in ['converted']]

        interaction_features = []

        for f in self.features:
            iFeature = 'I-'+f
            if group == None:
                df[iFeature] = df[f]*df['group']
            else:
                df[iFeature] = df[f]*group
            interaction_features.append(iFeature)

        if group != None:
            df['group'] = group

        self.features += interaction_features

        return df


# _____________________________________________________________________TianModel
class TianModel():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Tian'
        self.target = 'converted'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        df = self.generate_interactions(X)

        # train the classifiers
        self.estimator.fit(
            df.ix[:, self.features],
            y[:]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        df = self.generate_interactions(X, 1)
        r1 = self.estimator.predict_proba(df.ix[:, self.features])

        df = self.generate_interactions(X, 0)
        r2 = self.estimator.predict_proba(df.ix[:, self.features])

        return r1-r2

    # __________________________________________________________________________
    def generate_interactions(self, X, group=None):
        df = deepcopy(X)
        self.features = [f for f in df.columns if f not in ['converted']]

        interaction_features = []

        for f in self.features:
            iFeature = 'I-'+f
            if group == None:
                df[iFeature] = (df[f]-df[f].astype(np.float64).mean())*(2*df['group']-1)*0.5
            else:
                df[iFeature] = (df[f]-df[f].astype(np.float64).mean())*(2*group-1)*0.5
            interaction_features.append(iFeature)

            num_nan = df[iFeature].isnull().sum()
            if num_nan > 0:
                print('ERROR: Got NaNs. Feature is {0}.'.format(df[f].dtype))

        if group != None:
            df['group'] = group

        self.features += interaction_features

        return df



# ______________________________________________________________TianRevenueModel
class TianRevenueModel():
    # Supporting Features: 'group'

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'TianRevenue'
        self.target = 'revenue'
        self.extra_features = ['group']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        df = self.generate_interactions(X)

        # train the classifiers
        self.estimator.fit(
            df.ix[:, self.features],
            y[:]
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        df = self.generate_interactions(X, 1)
        r1 = self.estimator.predict(df.ix[:, self.features])

        df = self.generate_interactions(X, 0)
        r2 = self.estimator.predict(df.ix[:, self.features])

        return r1-r2

    # __________________________________________________________________________
    def generate_interactions(self, X, group=None):
        df = deepcopy(X)
        self.features = [f for f in df.columns if f not in ['converted']]

        interaction_features = []

        for f in self.features:
            iFeature = 'I-'+f
            if group == None:
                df[iFeature] = (df[f]-df[f].astype(np.float64).mean())*(2*df['group']-1)*0.5
            else:
                df[iFeature] = (df[f]-df[f].astype(np.float64).mean())*(2*group-1)*0.5
            interaction_features.append(iFeature)

            num_nan = df[iFeature].isnull().sum()
            if num_nan > 0:
                print('ERROR: Got NaNs. Feature is {0}.'.format(df[f].dtype))

        if group != None:
            df['group'] = group

        self.features += interaction_features

        return df



# ______________________________________________________________________TwoModel
class ReflectiveUplift():
    # Supporting Features: 'group', 'converted'
    # Shaar2016

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator_1 = estimator
        self.estimator_2 = deepcopy(estimator)

        # set mandatory variables used by SLIB
        self.name = 'Reflective'
        self.target = 'label'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        # not converted group
        c0_mask = (X['converted'] == 0.0)
        # converted group
        c1_mask = (X['converted'] == 1.0)
        # treatment group
        g0_mask = (X['group'] == 0.0)
        # control group
        g1_mask = (X['group'] == 1.0)

        features = [f for f in X.columns if f not in ['group', 'converted']]

        inv_length = 1.0/len(X)
        self.TR = np.sum(g0_mask & c1_mask)*inv_length
        self.TNR = np.sum(g0_mask & c0_mask)*inv_length
        self.CR = np.sum(g1_mask & c1_mask)*inv_length
        self.CNR = np.sum(g1_mask & c0_mask)*inv_length

        y_R = np.zeros(len(X[c1_mask]))
        y_R[np.array(X.loc[c1_mask, 'group'] == 0.0, dtype=np.bool)] = 1.0

        y_NR = np.zeros(len(X[c0_mask]))
        y_NR[np.array(X.loc[c0_mask, 'group'] == 0.0, dtype=np.bool)] = 1.0

        # train the classifiers
        self.estimator_1.fit(
            X.ix[c1_mask,features],
            y_R
        )
        self.estimator_2.fit(
            X.ix[c0_mask,features],
            y_NR
        )
        
    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        r1 = self.estimator_1.predict_proba(X.ix[:,features])[:,1]
        r2 = self.estimator_2.predict_proba(X.ix[:,features])[:,1]

        p = np.zeros((len(X),2))

        up_pos = r1*self.TR + (1.0-r2)*self.CNR
        up_neg = r2*self.TNR + (1.0-r1)*self.CR

        p[:,1] = up_pos - up_neg
        p[:,0] = 1.0-p[:,1]

        return p

# _____________________________________________________________ReflectiveUplift4
class ReflectiveUplift4():
    """Shaar2016, but with one multiclass model instead of two binary classifier.
    Supporting Features: 'group', 'converted'

    AB==0: TNR
    AB==1: TR
    AB==2: CNR
    AB==3: CR
    """

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Reflective4'
        self.target = 'AB_Class'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.features = [f for f in X.columns if f not in ['group', 'converted']]

        c0_mask = (X['converted']==0.0) # not converted group
        c1_mask = (X['converted']==1.0) # converted group
        g0_mask = (X['group']==0.0)     # treatment group
        g1_mask = (X['group']==1.0)     # control group

        inv_length = 1.0/len(X)
        self.TR  = np.sum(g0_mask & c1_mask)*inv_length
        self.TNR = np.sum(g0_mask & c0_mask)*inv_length
        self.CR  = np.sum(g1_mask & c1_mask)*inv_length
        self.CNR = np.sum(g1_mask & c0_mask)*inv_length

        # train the classifiers
        self.estimator.fit(X.ix[:,self.features], y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        r = self.estimator.predict_proba(X.ix[:,self.features])

        p = np.zeros((len(X),2))

        div_pos = r[:,0]+r[:,1]
        div_neg = r[:,2]+r[:,3]

        # avoid division by zero by setting to a small non-zero value
        eps = 0.001
        div_pos[div_pos==0.0] = eps
        div_neg[div_neg==0.0] = eps

        up_pos = r[:,1]/div_pos
        up_neg = r[:,3]/div_neg

        p[:,1] = up_pos - up_neg
        p[:,0] = 1.0-p[:,1]

        return p

# _____________________________________________________________PessimisticUplift
class PessimisticUplift():

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.reflective = ReflectiveUplift(estimator)
        self.lais = LaisTransformationModel(deepcopy(estimator))

        # set mandatory variables used by SLIB
        self.name = 'Pessimistic'
        self.target = 'label'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        # Reflective Uplift Model:
        self.reflective.fit(X, y)

        # Lai's Model:
        self.lais.fit(X.loc[:,features], y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        p_ref = self.reflective.predict_proba(X.loc[:,features])
        p_lai = self.lais.predict_proba(X.loc[:,features])

        return 0.5*(p_ref + p_lai)


# _________________________________________________________________RealistUplift
class RealistUplift():
    """Modificatin of Shaar2016's Reflective Uplift Model
    Assumption that all customers in CNR count towards the positive uplift
    is not realistc. Weight by overall uplift effect.

    Supporting Features: 'group', 'converted'
    """

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator_1 = estimator
        self.estimator_2 = deepcopy(estimator)

        # set mandatory variables used by SLIB
        self.name = 'Realist'
        self.target = 'label'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        # not converted group
        c0_mask = (X['converted']==0.0)
        # converted group
        c1_mask = (X['converted']==1.0)
        # treatment group
        g0_mask = (X['group']==0.0)
        # control group
        g1_mask = (X['group']==1.0)

        features = [f for f in X.columns if f not in ['group', 'converted']]

        inv_length = 1.0/len(X)
        self.TR  = np.sum(g0_mask & c1_mask)*inv_length
        self.TNR = np.sum(g0_mask & c0_mask)*inv_length
        self.CR  = np.sum(g1_mask & c1_mask)*inv_length
        self.CNR = np.sum(g1_mask & c0_mask)*inv_length
        self.uplift = uplift_score(X)

        y_R = np.zeros(len(X[c1_mask]))
        y_R[np.array(X.loc[c1_mask,'group']==0.0, dtype=np.bool)] = 1.0

        y_NR = np.zeros(len(X[c0_mask]))
        y_NR[np.array(X.loc[c0_mask,'group']==0.0, dtype=np.bool)] = 1.0

        # train the classifiers
        self.estimator_1.fit(
            X.ix[c1_mask,features],
            y_R
        )
        self.estimator_2.fit(
            X.ix[c0_mask,features],
            y_NR
        )

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        r1 = self.estimator_1.predict_proba(X.ix[:,features])[:,1]
        r2 = self.estimator_2.predict_proba(X.ix[:,features])[:,1]

        p = np.zeros((len(X),2))

        up_pos = r1*self.TR + self.uplift*(1.0-r2)*self.CNR
        up_neg = r2*self.TNR + (1.0-r1)*self.CR

        p[:,1] = up_pos - up_neg
        p[:,0] = 1.0-p[:,1]

        return p


# _________________________________________________________________________Kane4
class Kane4():
    """Kane, et al., 2014
    
    AB_Class==0: TN
    AB_Class==1: TR
    AB_Class==2: CN
    AB_Class==3: CR
    """

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Kane'
        self.target = 'AB_Class'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.features = [f for f in X.columns if f not in self.extra_features]

        c0_mask = (X['converted'] == 0.0) # not converted group
        c1_mask = (X['converted'] == 1.0) # converted group
        g0_mask = (X['group'] == 0.0)     # treatment group
        g1_mask = (X['group'] == 1.0)     # control group

        length = np.float64(len(X))
        self.TR  = np.float64(np.sum(g0_mask & c1_mask))/length
        self.TNR = np.float64(np.sum(g0_mask & c0_mask))/length
        self.CR  = np.float64(np.sum(g1_mask & c1_mask))/length
        self.CNR = np.float64(np.sum(g1_mask & c0_mask))/length
        self.T = self.TR + self.TNR
        self.C = self.CR + self.CNR

        # train the classifiers
        self.estimator.fit(X.ix[:, self.features], y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        r = self.estimator.predict_proba(X.ix[:, self.features])

        p = np.zeros((len(X), 2))

        up_pos = r[:, 1]/self.T + r[:, 2]/self.C
        up_neg = r[:, 0]/self.T + r[:, 3]/self.C

        p[:,1] = up_pos - up_neg
        p[:,0] = 1.0-p[:,1]

        return p

# ________________________________________________________________RealistUplift4
class RealistUplift4():
    # Supporting Features: 'group', 'converted'
    # modificatin of Shaar2016: 
    # Assumption that all customers in CNR count towards the positive uplift
    # is not realistc. Weight by overall uplift effect.

    # AB==0: TNR
    # AB==1: TR
    # AB==2: CNR
    # AB==3: CR

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.estimator = estimator

        # set mandatory variables used by SLIB
        self.name = 'Realist4'
        self.target = 'AB_Class'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        self.features = [f for f in X.columns if f not in ['group', 'converted']]

        c0_mask = (X['converted']==0.0) # not converted group
        c1_mask = (X['converted']==1.0) # converted group
        g0_mask = (X['group']==0.0)     # treatment group
        g1_mask = (X['group']==1.0)     # control group

        length = np.float64(len(X))
        self.TR  = np.sum(g0_mask & c1_mask)/length
        self.TNR = np.sum(g0_mask & c0_mask)/length
        self.CR  = np.sum(g1_mask & c1_mask)/length
        self.CNR = np.sum(g1_mask & c0_mask)/length
        self.uplift = uplift_score(X)

        # train the classifiers
        self.estimator.fit(X.ix[:,self.features], y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        r = self.estimator.predict_proba(X.ix[:,self.features])

        p = np.zeros((len(X),2))

        up_pos = self.TR*r[:,1] + self.CNR*self.uplift*r[:,2]
        up_neg = self.TNR*r[:,0] + self.CR*r[:,3]

        p[:,1] = up_pos - up_neg
        p[:,0] = 1.0-p[:,1]

        return p


# _______________________________________________________________DepressedUplift
class DepressedUplift():

    # __________________________________________________________________________
    def __init__(self, estimator):
        self.reflective = RealistUplift(estimator)
        self.lais = LaisTransformationModel(deepcopy(estimator))

        # set mandatory variables used by SLIB
        self.name = 'Depressed'
        self.target = 'label'
        self.extra_features = ['group', 'converted']
        self.params = vars(estimator)
        self.params['classifier_name'] = estimator.__class__.__name__

    # __________________________________________________________________________
    def fit(self, X, y):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        # Reflective Uplift Model:
        self.reflective.fit(X, y)

        # Lai's Model:
        self.lais.fit(X.loc[:,features], y)

    # __________________________________________________________________________
    def predict_proba(self, X):
        features = [f for f in X.columns if f not in ['group', 'converted']]

        p_ref = self.reflective.predict_proba(X.loc[:,features])
        p_lai = self.lais.predict_proba(X.loc[:,features])

        return 0.5*(p_ref + p_lai)