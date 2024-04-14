"""
Fair Logistic Regression
"""
import numpy as np
import cvxpy as cp
from collections import namedtuple
from sklearn.metrics import log_loss


def get_marginals(a, y, mode = 'DRFPROB'):
    """Calculate marginal probabilities of the given data"""
    N = y.shape[0]
    if mode == 'DRFLR':
        y[y == 0] = -1
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
        print('FSVM Marginals are WRONG!')
        print('margins:',P_11, P_01, P_10, P_00)
    # change back
    if mode =='DRFLR':
        y[y == -1] = 0
    return P_11, P_01, P_10, P_00

def get_IndexSet(a, y, mode = 'DRFPROB'):
    N=y.shape[0]
    I_11=[]
    I_10=[]
    I_01=[]
    I_00=[]
    
    if mode == 'DRFLR':
        y[y == 0] = -1
    for i in range(N):
        if a[i]==1 and y[i]==1:
            I_11.append(i)
        if a[i]==1 and y[i]==-1:
            I_10.append(i)
        if a[i]==0 and y[i]==1:
            I_01.append(i)
        if a[i]==0 and y[i]==-1:
            I_00.append(i)
    if mode =='DRFLR':
        y[y == -1] = 0
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


class DROFairModels():
    def __init__(self, reg=0, radius=0, epsilon = 0.01,
                 fit_intercept=True, kap_A=0.1, kap_Y=0.1, verbose=False, mode = 'DRFSVM', eps=0.01,solver=cp.MOSEK, side='one', sreg=False, 
                 balanced_accuracy = False):
        
        
        self.radius = radius
        self.kap_A = kap_A
        self.kap_Y = kap_Y
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.verbose = verbose
        self.mode = mode
        self.eps = epsilon
        self.solver = solver
        self.side = side
        self.sreg = sreg #small reg term to make FPROB unique
        self.balanced_accuracy = balanced_accuracy
        assert self.mode in ['DRFSVM', 'DRFPROB', 'DRFLR']
        
    def fit(self, X, a, y):
        if self.mode == 'DRFLR':
            # ensure labels are correct
            self.fit_DRFLR(X, a, y)
        elif self.mode == 'DRFPROB':
            self.fit_DRFPROB(X, a, y)
        elif self.mode == 'DRFSVM':
            self.fit_DRFSVM(X, a, y)
            

    def fit_DRFLR(self, X, a, y):
        """
                Fit the model according to the given training data.
                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.
                a : array-like of shape (n_samples,)
                    Sensitive attributes vector relative to X.
                y : array-like of shape (n_samples,)
                    Target vector relative to X.
        """
        self.prepare_model_DRFLR(X, a, y)
        self.eta.value = self.reg
        self.rho.value = self.radius
        self.kappa_A.value = self.kap_A
        self.kappa_Y.value = self.kap_Y
        self.prob.solve(solver=self.solver, verbose=self.verbose)
        self.coef_ = self.beta.value
        self.intercept_ = self.b.value


    def prepare_model_DRFLR(self, X, a, y):
        N = X.shape[0]
        
        
        
        P_11, P_01, P_10, P_00 = get_marginals(a, y, mode = self.mode)
        if self.reg > min(P_01, P_11):
            raise ValueError('Regularization constant should be less than '
                             '$\min{1/P(A=1, Y=1), 1/P(A=0, Y=1)}$')
            
        # y[y == -1] = 0
        dim = X.shape[1]
        r = [1 / P_01, 1 / P_11]

        self.kappa_A = cp.Parameter(nonneg=True)
        self.kappa_Y = cp.Parameter(nonneg=True)
        self.beta = cp.Variable(dim)
        t = cp.Variable(1)
        self.rho = cp.Parameter(nonneg=True)
        self.eta = cp.Parameter(nonpos=False)
        lambda_ = cp.Variable(2, nonneg=True)
        self.b = cp.Variable(1)
        mu_0 = cp.Variable((2, 2))
        mu_1 = cp.Variable((2, 2))
        nu = cp.Variable((2, N))

        u_1 = cp.Variable((2, N), nonneg=True)  # a = 1, ,which cosntraint
        v_1 = cp.Variable((2, N), nonneg=True)

        u_2 = cp.Variable((2, N), nonneg=True)  # a = 1, ,which cosntraint
        v_2 = cp.Variable((2, N), nonneg=True)

        u_3 = cp.Variable((2, N), nonneg=True)  # a = 1, ,which cosntraint
        v_3 = cp.Variable((2, N), nonneg=True)

        u_4 = cp.Variable((2, N), nonneg=True)  # a = 1, ,which cosntraint
        v_4 = cp.Variable((2, N), nonneg=True)

        cons = []

        cons.append(self.rho * lambda_[1] + P_11 * mu_1[1, 1] + P_01 * mu_1[0, 1] + P_10 * mu_1[1, 0] +
                    P_00 * mu_1[0, 0] + 1 / N * cp.sum(nu[1, :]) <= t)
        cons.append(self.rho * lambda_[0] + P_11 * mu_0[1, 1] + P_01 * mu_0[0, 1] + P_10 * mu_0[1, 0] +
                    P_00 * mu_0[0, 0] + 1 / N * cp.sum(nu[0, :]) <= t)



        for i in range(N):
            # 1st constraint
            # a = 1, a' =0, 1st constraint with ell_beta(x, 0)
            rhs = self.kappa_A * cp.abs(1 - a[i]) * lambda_[1] + self.kappa_Y * cp.abs(y[i]) * lambda_[1] + \
                  mu_1[
                      1, 0] + nu[1, i]
            cons.append(u_1[1, i] + v_1[1, i] <= 1)
            # print((self.beta @ X[i, :] + self.b - rhs).shape)
            cons.append(
                cp.constraints.exponential.ExpCone((self.beta @ X[i, :] + self.b - rhs)[0], 1, u_1[1, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_1[1, i]))
            # a= 0, a' =1, 1st constraint with ell_beta(x, 0)
            rhs = self.kappa_A * cp.abs(0 - a[i]) * lambda_[0] + self.kappa_Y * cp.abs(y[i]) * lambda_[0] + \
                  mu_0[
                      0, 0] + nu[0, i]
            cons.append(u_1[0, i] + v_1[0, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((self.beta @ X[i, :] + self.b - rhs)[0], 1, u_1[0, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_1[0, i]))

            # 2nd constraint
            # a=1, a'=0
            rhs = self.kappa_A * cp.abs(0 - a[i]) * lambda_[1] + self.kappa_Y * cp.abs(y[i]) * lambda_[1] + \
                  mu_1[0, 0] + nu[1, i]
            cons.append(u_2[1, i] + v_2[1, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((self.beta @ X[i, :] + self.b - rhs)[0], 1, u_2[1, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_2[1, i]))
            # a= 0, a'=1
            rhs = self.kappa_A * cp.abs(1 - a[i]) * lambda_[0] + self.kappa_Y * cp.abs(y[i]) * lambda_[0] + \
                  mu_0[1, 0] + nu[0, i]
            cons.append(u_2[0, i] + v_2[0, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((self.beta @ X[i, :] + self.b - rhs)[0], 1, u_2[0, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_2[0, i]))

            # 3rd constraint
            # a=1, a'=0
            rhs = self.kappa_A * cp.abs(1 - a[i]) * lambda_[1] + self.kappa_Y * cp.abs(1 - y[i]) * lambda_[
                1] + \
                  mu_1[1, 1] + \
                  nu[1, i]
            rhs = rhs / (1 - self.eta * r[1])
            cons.append(u_3[1, i] + v_3[1, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((- self.beta @ X[i, :] - self.b - rhs)[0], 1, u_3[1, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_3[1, i]))
            # a=0, a'=1
            rhs = self.kappa_A * cp.abs(0 - a[i]) * lambda_[0] + self.kappa_Y * cp.abs(1 - y[i]) * lambda_[
                0] + \
                  mu_0[0, 1] + nu[0, i]
            rhs = rhs / (1 - self.eta * r[0])
            cons.append(u_3[0, i] + v_3[0, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((- self.beta @ X[i, :] - self.b - rhs)[0], 1, u_3[0, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_3[0, i]))

            # 4th constraint

            # a=1, a'=0
            rhs = self.kappa_A * cp.abs(0 - a[i]) * lambda_[1] + self.kappa_Y * cp.abs(1 - y[i]) * lambda_[
                1] + \
                  mu_1[0, 1] + nu[1, i]
            rhs = rhs / (1 + self.eta * r[0])
            cons.append(u_4[1, i] + v_4[1, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((- self.beta @ X[i, :] - self.b - rhs)[0], 1, u_4[1, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_4[1, i]))
            # a=0, a'=1
            rhs = self.kappa_A * cp.abs(1 - a[i]) * lambda_[0] + self.kappa_Y * cp.abs(1 - y[i]) * lambda_[
                0] + \
                  mu_0[1, 1] + nu[0, i]
            rhs = rhs / (1 + self.eta * r[1])
            cons.append(u_4[0, i] + v_4[0, i] <= 1)
            cons.append(
                cp.constraints.exponential.ExpCone((- self.beta @ X[i, :] - self.b - rhs)[0], 1, u_4[0, i]))
            cons.append(cp.constraints.exponential.ExpCone(- rhs, 1, v_4[0, i]))

        cons.append(cp.SOC(lambda_[1] / (1 + self.eta * r[0]), self.beta))
        cons.append(cp.SOC(lambda_[0] / (1 + self.eta * r[1]), self.beta))

        cons.append(cp.SOC(lambda_[1], self.beta))
        cons.append(cp.SOC(lambda_[0], self.beta))
        cons.append(cp.SOC(lambda_[1] / (1 - self.eta * r[1]), self.beta))
        cons.append(cp.SOC(lambda_[0] / (1 - self.eta * r[0]), self.beta))

        if self.fit_intercept == 0:
            cons.append(self.b == 0)

        obj = cp.Minimize(t)
        self.prob = cp.Problem(obj, cons)


    def fit_DRFPROB(self, X, a, y):
        """
                Fit the model according to the given training data.
                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.
                a : array-like of shape (n_samples,)
                    Sensitive attributes vector relative to X.
                y : array-like of shape (n_samples,)
                    Target vector relative to X.
        """
        self.prepare_model_DRFPROB(X, a, y)
        self.eta.value = self.reg
        self.rho.value = self.radius
        self.epsilon.value=self.eps
        self.prob.solve(solver=cp.GUROBI, verbose=self.verbose, TimeLimit=100, FeasibilityTol = 1e-7, IntFeasTol=1e-7)
        # self.prob.solve(solver=cp.MOSEK, verbose=self.verbose)
        self.coef_ = self.w.value
        self.intercept_ = self.b.value
        self.eps_=self.epsilon.value
        self.tvalue=self.t.value
    
    def prepare_model_DRFPROB(self, X, a, y):
        N = X.shape[0]
        dim=X.shape[1]
        P_11, P_01, P_10, P_00 = get_marginals(a, y)
        I_11, I_10, I_01, I_00 = get_IndexSet(a,y)
        
        #define variables
        self.w=cp.Variable(dim)
        self.b=cp.Variable(1)
        self.wb=cp.Variable(dim+1)
        self.t=cp.Variable(N,boolean=True)
        lambda_0=cp.Variable(N,boolean=True)
        lambda_1=cp.Variable(N,boolean=True)
        tau=cp.Variable(1)
        
        #define parameters
        self.rho=cp.Parameter(nonneg=True)
        self.eta=cp.Parameter(nonpos=False)
        self.epsilon=cp.Parameter(nonneg=True)
        M=1000
     
# one norm
        p, p_star=1, 'inf'
        # p_star, p=1, 'inf'
#  inf norm
#         p="inf"
        
        
        cons=[]
        cons.append(self.w<=1)
        cons.append(self.w>=-1)
        cons.append(self.b<=1)
        cons.append(self.b>=-1)
        # cons.append(self.wb[:dim]==self.w)
        # cons.append(self.wb[dim]==self.b)
        # cons.append(cp.norm(self.wb,p_star)<= 1)
 
        for i in range(N):
#             cons.append(-self.w.T@X[i]*y[i]-self.b*y[i]+self.rho*cp.norm(ax,p)+1<=self.t[i]*M)
#             cons.append(-y[i]*(self.w.T@X[i]+self.b)+self.rho*cp.norm(self.w,p)<=self.t[i]*M-self.epsilon)
            cons.append(-y[i]*(self.w.T@X[i]+self.b)+self.rho*cp.norm(self.w,p)<=self.t[i]*M-self.epsilon)
        
        if self.side == 'two':
        #1st constraints
        ##a=0, a'=1
        
            term1=0
            for i in range(len(I_01)):
                term1=term1+lambda_0[I_01[i]]
            term2=0
            for i in range(len(I_11)):
                term2=term2+lambda_0[I_11[i]]
            cons.append(term1/P_01+term2/P_11-len(I_11)/P_11 <= N*self.eta)
            
            
            for i in range(len(I_01)):
                cons.append(self.w.T@X[I_01[i]]+self.rho*cp.norm(self.w,p)+self.b+self.epsilon<=M*lambda_0[I_01[i]])
            for i in range(len(I_11)):
                cons.append(-self.w.T@X[I_11[i]]+self.rho*cp.norm(self.w,p)-self.b<=M*lambda_0[I_11[i]])
            
        
        #2nd constraints
        ##a=1, a'=0        
                        
                        
        term1=0
        for i in range(len(I_11)):
            term1=term1+lambda_1[I_11[i]]
        term2=0
        for i in range(len(I_01)):
            term2=term2+lambda_1[I_01[i]]
        cons.append(term1/P_11+term2/P_01-len(I_01)/P_01 <= N*self.eta)
        
        for i in range(len(I_11)):
            cons.append(self.w.T@X[I_11[i]]+self.rho*cp.norm(self.w,p)+self.b+self.epsilon<=M*lambda_1[I_11[i]])
        for i in range(len(I_01)):
            cons.append(-self.w.T@X[I_01[i]]+self.rho*cp.norm(self.w,p)-self.b<=M*lambda_1[I_01[i]])
            
        # build model
        if self.sreg == False:
            obj = cp.Minimize(1/N*cp.sum(self.t))
        else:
            obj = cp.Minimize(1/N*cp.sum(self.t) + 0.00001*cp.norm(self.w,1)) #add this small regularization can make solution unique
            
            
        if self.balanced_accuracy == True:
            I_1, I_0 = get_LabelSet(y)
            N1,N2=len(I_1),len(I_0)
            obj1=sum(self.t[I_1[i]] for i in range(N1))
            obj2=sum(self.t[I_0[i]] for i in range(N2))
            obj = cp.Minimize(1/N1*obj1+1/N2*obj2)
            if self.sreg == True:
                obj = cp.Minimize(1/N1*obj1+1/N2*obj2 + 0.00001*cp.norm(self.w,1))
        self.prob = cp.Problem(obj, cons)
        
    
    def fit_DRFSVM(self, X, a, y):
        """
                Fit the model according to the given training data.
                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.
                a : array-like of shape (n_samples,)
                    Sensitive attributes vector relative to X.
                y : array-like of shape (n_samples,)
                    Target vector relative to X.
        """
        self.prepare_model_DRFSVM(X, a, y)
        self.eta.value = self.reg
        self.rho.value = self.radius
        self.epsilon.value=self.eps
        self.prob.solve(solver=cp.GUROBI, verbose=self.verbose)
        # self.prob.solve(solver=cp.MOSEK, verbose=self.verbose)
        self.time=self.prob.solver_stats.solve_time
        self.coef_ = self.w.value
        self.intercept_ = self.b.value

    
    def prepare_model_DRFSVM(self, X, a, y):
        N = X.shape[0]
        dim=X.shape[1]
        P_11, P_01, P_10, P_00 = get_marginals(a, y)
        I_11, I_10, I_01, I_00 = get_IndexSet(a,y)
        
        #define variables
        self.w=cp.Variable(dim)
        self.b=cp.Variable(1)
        self.t=cp.Variable(N,nonneg=True)
        self.z=cp.Variable(1,nonneg=True)
        self.lambda_0=cp.Variable(N,nonneg=True)
        self.lambda_1=cp.Variable(N,nonneg=True)
        tau=cp.Variable(1)
        
        #define parameters
        self.rho=cp.Parameter(nonneg=True)
        self.eta=cp.Parameter(nonpos=False)
        self.epsilon=cp.Parameter(nonneg=True)
     
        # one norm
        p=1
        #  inf norm
#         p="inf"
        
        
        cons=[]
        cons.append(self.z==1)
        for i in range(N):
            cons.append(-y[i]*(self.w.T@X[i]+self.b)+self.rho*cp.norm(self.w,p)<=self.t[i]-1)
        
        if self.side == 'two':
        #1st constraints
        ##a=0, a'=1
        
            term1=0
            for i in range(len(I_01)):
                term1=term1+self.lambda_0[I_01[i]]
            term2=0
            for i in range(len(I_11)):
                term2=term2+self.lambda_0[I_11[i]]
            cons.append(term1/P_01+term2/P_11 <= self.z*(N*self.eta+len(I_11)/P_11))
            
            
            for i in range(len(I_01)):
                cons.append(self.z+self.w.T@X[I_01[i]]+self.rho*cp.norm(self.w,p)+self.b<=self.lambda_0[I_01[i]])
            for i in range(len(I_11)):
                cons.append(self.z-self.w.T@X[I_11[i]]+self.rho*cp.norm(self.w,p)-self.b<=self.lambda_0[I_11[i]])
        
        
        #1st constraints
        #a=1, a'=0        
                        
                        
        term1=0
        for i in range(len(I_11)):
            term1=term1+self.lambda_1[I_11[i]]
        term2=0
        for i in range(len(I_01)):
            term2=term2+self.lambda_1[I_01[i]]
        cons.append(term1/P_11+term2/P_01 <= self.z*(N*self.eta+len(I_01)/P_01))
        
        for i in range(len(I_11)):
            cons.append(self.z+self.w.T@X[I_11[i]]+self.rho*cp.norm(self.w,p)+self.b<=self.lambda_1[I_11[i]])
        for i in range(len(I_01)):
            cons.append(self.z-self.w.T@X[I_01[i]]+self.rho*cp.norm(self.w,p)-self.b<=self.lambda_1[I_01[i]])
            
        #build model
        obj = cp.Minimize(1/N*cp.sum(self.t))
        
        if self.balanced_accuracy == True:
                    #balanced accuracy
            I_1, I_0 = get_LabelSet(y)
            N1,N2=len(I_1),len(I_0)
            obj1=sum(self.t[I_1[i]] for i in range(N1))
            obj2=sum(self.t[I_0[i]] for i in range(N2))
            obj = cp.Minimize(1/N1*obj1+1/N2*obj2)
        self.prob = cp.Problem(obj, cons)
      
        

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
        if self.mode == 'DRFLR':
            predictions = self.predict_prob(X)
            predictions[predictions == 0] = -1
        else:
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
        if self.mode == 'DRFLR':
            predictions = self.predict_prob(X)
            predictions[predictions == 0] = -1
        else:
            predictions = self.predict(X)
            
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == -1 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == -1 and y[i] == 1 else 0 for i in range(N)])
        return (2*TP) / (2*TP + FP + FN)
    

    def predict_proba(self, X):
        """
            Probability estimates.
            The returned estimates for the class are ordered by the
            label of classes.
            For a multi_class problem, if multi_class is set to be "multinomial"
            the softmax function is used to find the predicted probability of
            each class.
            Else use a one-vs-rest approach, i.e calculate the probability
            of each class assuming it to be positive using the logistic function.
            and normalize these values across all the classes.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            Returns
            -------
            T : array-like of shape (n_samples, 2)
                Returns the probability of the sample for each class in the model,
                where classes are ordered as they are in ``self.classes_``.
        """
        proba = np.zeros((X.shape[0], 2))
        h = 1 / (1 + np.exp(- self.coef_ @ X.T - self.intercept_))
        proba[:, 1] = h
        proba[:, 0] = 1 - h
        return proba

    def predict_log_proba(self, X):
        """
                Predict logarithm of probability estimates.
                The returned estimates for all classes are ordered by the
                label of classes.
                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    Vector to be scored, where `n_samples` is the number of samples and
                    `n_features` is the number of features.
                Returns
                -------
                T : array-like of shape (n_samples, n_classes)
                    Returns the log-probability of the sample for each class in the
                    model, where classes are ordered as they are in ``self.classes_``.
        """
        prob_clip = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return np.log(prob_clip)

    def predict_prob(self, X, thr=0.5):
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
        probs = self.predict_proba(X)[:, 1]
        probs[probs >= thr] = 1
        probs[probs < thr] = 0
        probs_bin = probs
        return probs_bin

    def logloss(self, X, y):
        return log_loss(y_true=y, y_pred=self.predict_proba(X)[:, 1])

    
    def unfairness(self, X, a, y):
         # calculate the unfairness of the given test data set
        N = X.shape[0]
        I_11, I_10, I_01, I_00 = get_IndexSet(a, y, self.mode)
        if self.mode == 'DRFLR':
            predictions = self.predict_prob(X)
            predictions[predictions == 0] = -1
            y[y == 0] = -1
        else:
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
        eo_unfairness = P_111 - P_011
        
        UnfairnessMeasures = namedtuple('UnfairnessMeasures', 'det_unfairness, eodds_unfairness, dp_unfairness, eo_unfairness')(det_unfairness,
                                                                                                eodds_unfairness, dp_unfairness, eo_unfairness)
        # if self.mode =='DRFLR':
        #     y[y == -1] = 0
        return UnfairnessMeasures


