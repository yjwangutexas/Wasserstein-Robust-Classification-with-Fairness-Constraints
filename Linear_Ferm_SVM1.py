import numpy as np
from collections import namedtuple
import sys


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
        print(P_01 , P_10 , P_11 , P_00)
        print(np.abs(P_01 + P_10 + P_11 + P_00 - 1))
        print('Marginals are WRONG!')
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


class Linear_FERM:
    # The linear FERM algorithm
    def __init__(self, data, target, model, sensible_feature):
        self.data = data.copy()
        self.target = target.copy()
        self.values_of_sensible_feature = list(set(sensible_feature))
        self.list_of_sensible_feature_train = sensible_feature
        self.val0 = np.min(self.values_of_sensible_feature)
        self.val1 = np.max(self.values_of_sensible_feature)
        self.model = model
        self.u = None
        self.max_i = None
        if min(self.target) != -1:
            self.target[self.target == min(self.target)] = -1

    def new_representation(self, examples):
        if self.u is None:
            sys.exit('Model not trained yet!')
            return 0

        new_examples = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in examples])
        new_examples = np.delete(new_examples, self.max_i, 1)
        return new_examples

    def predict(self, examples):
        new_examples = self.new_representation(examples)
        prediction = self.model.predict(new_examples)
        return prediction

    def fit(self):
        # Evaluation of the empirical averages among the groups
        tmp = [ex for idx, ex in enumerate(self.data)
               if self.target[idx] == 1 and self.list_of_sensible_feature_train[idx] == self.val1]
        average_A_1 = np.mean(tmp, 0)
        tmp = [ex for idx, ex in enumerate(self.data)
               if self.target[idx] == 1 and self.list_of_sensible_feature_train[idx] == self.val0]
        average_not_A_1 = np.mean(tmp, 0)

        # Evaluation of the vector u (difference among the two averages)
        self.u = -(average_A_1 - average_not_A_1)
        self.max_i = np.argmax(self.u)

        # Application of the new representation
        newdata = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in self.data])
        newdata = np.delete(newdata, self.max_i, 1)
        self.data = newdata

        # Fitting the linear model by using the new data
        if self.model:
            self.model.fit(self.data, self.target)

    def score(self, X, y):
        # calculate accuracy of the given test data set
        predictions = self.predict(X)
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        TN = np.sum([1 if predictions[i] == -1 and y[i] == -1 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == -1 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == -1 and y[i] == 1 else 0 for i in range(N)])
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
        Precision=TP/(TP+FP)
        Recall=TP/(TP+FN)
#         print('FN',FN)
        return (2*TP) / (2*TP + FP + FN)

    def unfairness(self, X, a, y, thr=0.5):
        N = X.shape[0]
        P_11, P_01, _, _ = get_marginals(a, y)
        P_111 = 0
        P_001 = 0
        P_101 = 0
        predictions = self.predict(X)
        for i in range(N):
            if a[i] == 1 and y[i] == 1:
                if predictions[i] == 1:
                    P_111 += 1 / P_11 / N
            if a[i] == 0 and y[i] == 1:
                if predictions[i] == -1:
                    P_001 += 1 / P_01 / N
            if a[i] == 0 and y[i] == 1:
                if predictions[i] == 1:
                    P_101 += 1 / P_01 / N
                    
#         print('P_111,P_001, P_101',P_111,P_001, P_101)
#         print('real',np.abs(P_111-P_101))
        det_unfairness = np.abs(P_111 + P_001 - 1)  # Deterministic Unfairness
#         print('log',det_unfairness)
#         det_unfairness = np.abs(P_111 - P_001 )

        UnfairnessMeasures = namedtuple('UnfairnessMeasures', 'det_unfairness')(det_unfairness)
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
    