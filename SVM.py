import numpy as np
import cvxpy as cp
from collections import namedtuple
from sklearn.metrics import log_loss

def get_marginals(a, y):
    """Calculate marginal probabilities of the given data"""
    N = y.shape[0]
    P_11 = np.sum(
        [1 / N if a[i] == 1 and y[i] == 1 else 0 for i
         in range(N)])
    P_01 = np.sum(
        [1 / N if a[i] == 0 and y[i] == 1 else 0 for i
         in range(N)])
    P_10 = np.sum(
        [1 / N if a[i] == 1 and y[i] == -1 else 0 for i
         in range(N)])
    P_00 = np.sum(
        [1 / N if a[i] == 0 and y[i] == -1 else 0 for i
         in range(N)])
    if np.abs(P_01 + P_10 + P_11 + P_00 - 1) > 1e-10:
        print(np.abs(P_01 + P_10 + P_11 + P_00 - 1))
        print(P_01 , P_10 , P_11 , P_00)
        print('SVM Marginals are WRONG!')
    return P_11, P_01, P_10, P_00

def get_IndexSet(a,y):
    N=y.shape[0]
    I_11=[]
    I_10=[]
    I_01=[]
    I_00=[]
    
    for i in range(N):
        if a[i]==1 and y[i]==1:
            I_11.append(i)
        if a[i]==1 and y[i]==-1:
            I_10.append(i)
        if a[i]==0 and y[i]==1:
            I_01.append(i)
        if a[i]==0 and y[i]==-1:
            I_00.append(i)
#     print(len(I_11),len(I_10))
    return I_11, I_10, I_01, I_00

def get_LabelSet(y):
    N=y.shape[0]
    I_1=[]
    I_0=[]
    
    for i in range(N):
        if  y[i]==1:
            I_1.append(i)
        else:
            I_0.append(i)
    
    return I_1, I_0


class SVM():
    def __init__(self, fit_intercept=True, reg=0, verbose=False):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.reg = reg
        
        
    def fit(self, X, y):
        """
                Fit the model according to the given training data.
                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.
                y : array-like of shape (n_samples,)
                    Target vector relative to X.
        """
        for i in range(y.shape[0]):
            if y[i]==0:
                y[i]=-1
        
        dim = X.shape[1]
        beta = cp.Variable(dim)  # coeefficients
        b = cp.Variable(1)  # intercept
        t=cp.Variable(X.shape[0]) #epigrahpical variable
        
#         print(cp.sum(np.ones(10)))
        
#         print('test',np.sum(X@np.zeros(2)+0.5))
#         print('test2',cp.pos([-1,1,-1]))
#         print(cp.multiply(y, X@beta - b))
#         print(cp.sum(cp.pos(1 - cp.multiply(y, X@np.zeros(2) + 1))))
        constraints=[]
        for i in range(X.shape[0]):
#             print(X[i])
            constraints.append(1-y[i]*(np.transpose(X[i])@beta+b)<=t[i])
        constraints.append(t>=0)
#         print(constraints)
        
        
#         loss = np.sum(cp.pos(1 - cp.multiply(y, X@beta - b)))
        loss=cp.sum(t)/X.shape[0]
        problem = cp.Problem(cp.Minimize(loss),constraints)
        problem.solve(solver=cp.GUROBI)
        self.coef_ = beta.value
        self.intercept_ = b.value
#         print('status:', problem.status)
#         print('optimal value',problem.value)
        
        
        
        
    def predict(self, X):
        """
            Predict class of covariates.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            thr : scale
                threshold to predict the class of given data x such that if predict_proba(x)>= thr the predicted
                class is 1 and otherwise the predicted class is 0.
            Returns
            -------
            T : vector of shape (n_classes, )
                Returns the predicted classes of the samples
        """
        X_class = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            X_class[i]=np.sign(self.coef_.T@X[i]+self.intercept_)
        return X_class
    
    def score(self, X, y):
        # calculate the accuracy of the given test data set
        predictions = self.predict(X)
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        TN = np.sum([1 if predictions[i] == -1 and y[i] == -1 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == -1 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == -1 and y[i] == 1 else 0 for i in range(N)])
#         print('FN',FN)
        mean_acc = (TP + TN) / (TP + TN + FP + FN)
        return mean_acc
    
    def Fscore(self, X, y):
        # calculate the accuracy of the given test data set
        predictions = self.predict(X)
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        TN = np.sum([1 if predictions[i] == -1 and y[i] == -1 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == -1 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == -1 and y[i] == 1 else 0 for i in range(N)])
#         Precision=TP/(TP+FP)
#         Recall=TP/(TP+FN)
#         print('FN',FN)
#         print('TP',TP)
#         print('TN',TN)
#         print('FP',FP)
#         print('FN',FN)
#         print('precision',Precision)
#         print('recall',Recall)
#         print(2*Precision*Recall/(Precision+Recall))
        return (2*TP) / (2*TP + FP + FN)
    
    def unfairness(self, X, a, y, thr=0.5):
        # calculate the unfairness of the given test data set
        N = X.shape[0]
        P_11, P_01, _, _ = get_marginals(a, y)
        E_h11 = []
        E_h01 = []
        P_111 = 0
        P_001 = 0
        predictions = self.predict(X)
        for i in range(N):
            if a[i] == 1 and y[i] == 1:
                if predictions[i] == 1:
                    P_111 += 1 / P_11 / N
            if a[i] == 0 and y[i] == 1:
                if predictions[i] == -1:
                    P_001 += 1 / P_01 / N

        
        det_unfairness = np.abs(P_111 + P_001 - 1)  # Deterministic Unfairness
        eo_unfairness = P_111 + P_001 - 1
        # det_unfairness = P_111 + P_001 - 1

        UnfairnessMeasures = namedtuple('UnfairnessMeasures', 'det_unfairness, eo_unfairness')(det_unfairness, eo_unfairness)
        return UnfairnessMeasures
    
    def unfairness2(self, X, a, y, thr=0.5):
         # calculate the unfairness of the given test data set
        N = X.shape[0]
        P_11, P_01, P_10, P_00 = get_marginals(a, y)
        I_11, I_10, I_01, I_00 = get_IndexSet(a,y)
        I_1, I_0 = get_LabelSet(y)
        predictions = self.predict(X)
        
        N_111, N_011, N_101, N_001 = 0, 0, 0, 0
        N_1, N_0 = 0, 0
        
        for i in range(N):
            if a[i] == 1 and y[i] == 1:
                if predictions[i] == 1:
                    N_111 += 1
            if a[i] == 0 and y[i] == 1:
                if predictions[i] == 1:
                    N_011 += 1
            if a[i] ==1 and y[i] == -1:
                if predictions[i] == 1:
                    N_101 += 1
            if a[i] ==0 and y[i] == -1:
                if predictions[i] == 1:
                    N_001 += 1
                    
        for i in range(N):
            if a[i] == 1 and predictions[i] == 1:
                N_1 += 1
            if a[i] == 0 and predictions[i] == 1:
                N_0 += 1
                    
        P_111 = N_111/len(I_11)
        P_011 = N_011/len(I_01)
        P_101 = N_101/len(I_10)
        P_001 = N_001/len(I_00)
        
        DP_1 = N_1/(len(I_11) + len(I_10))
        DP_0 = N_0/(len(I_01) + len(I_00))
        
        det_unfairness = np.abs(P_111 - P_011)  # Deterministic Unfairness
        eodds_unfairness = max(np.abs(P_111 - P_011), np.abs(P_101 - P_001))
        dp_unfairness = np.abs(DP_1 - DP_0)
        
        UnfairnessMeasures = namedtuple('UnfairnessMeasures', 'det_unfairness, eodds_unfairness, dp_unfairness')(det_unfairness, eodds_unfairness, dp_unfairness)
        return UnfairnessMeasures
    

def load_adult(DIR_DATA=path+"/datasets/", smaller=False, scaler=True):
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    data = pd.read_csv(
        DIR_DATA + "adult/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        DIR_DATA + "adult/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :-1], target[:len_train // 20])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])
    else:
        print('The dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train, :-1], target[:len_train])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])
    return data, data_test

