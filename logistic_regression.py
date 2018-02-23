import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import warnings
####################################################################################
def sigmoid(z):
    warnings.warn = ig_warn
    return 1.0 / ( 1.0 + math.e**(-1*z) )

def cost_J(x,y, theta,lamda,m):
    
    temp=list(theta)
    temp[0]=0
    temp=np.array(temp)
    h = sigmoid(np.dot(x, theta))
    J=-1.0/m*(np.sum(y*(np.log(h)) + (1.0-y)*(np.log(1.0-h)))
               +(float(lamda)/(2*m)*np.sum(temp**2)))
    warnings.warn = ig_warn
    return J

def grad_desc(x,y, itr,theta, lamda, m, alpha):
    s=(x[0]).size
    
    cost=[]
    for i in range(itr):
        cost.append(cost_J(x,y, theta,lamda,m))
        if len(cost)>1:
           if abs(cost[i])>abs(cost[i-1]) or abs(cost[i])==abs(cost[i-1]) :break
        zrd=list(theta)
        zrd[0]=0
        zrd=np.array(zrd)
        h = sigmoid(np.dot(x, theta))
        grad = ((1.0/m ) * (np.dot(x.T,( h - y ))) - ((float(lamda)/m) * zrd))
        #Update thetas based on "gradientz"
        for k in range (s):
            theta = theta - alpha * grad
    return theta

def predict_accu(X, Y, theta):
        Z = np.dot(X, theta)
        Z[ Z >= 0] = 1
        Z[ Z < 0 ] = 0 
        true_p=0
        false_p=0
        false_n=0
        for i in range(Z.size):
            if (Y[i] == 1.0 and Z[i] == 1.0):
                true_p += 1
            elif Y[i] == 0.0 and Z[i] == 1.0:
                false_p += 1
            elif Y[i] == 1.0 and Z[i] == 0.0:
                false_n += 1
        pre = true_p/(true_p + false_p)
        rec = true_p/(true_p + false_n)
        f_meas = (2*pre*rec)/(pre + rec)
        return f_meas
    
def ig_warn(*args, **kwargs):
    pass

def standardize(X):
    for i in range(len(X[1])):
        mx = max(X[:,i])
        mn = min(X[:,i])
        for j in range(len(X)):
            X[j,i] = (X[j,i]-mn)/(mx-mn)
    return X
######################################################################################
# Cleaning the Dataset

data_set = pd.read_csv('chronic_kidney_disease_full_orig.csv')
data_set = data_set.replace({'\t':''}, regex=True)
data_set = data_set.replace(r'\?+', np.nan, regex=True)
data_set[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo'
          , 'pcv', 'wc', 'rc']] = data_set[['age', 'bp', 'bgr',
         'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv'
         , 'wc', 'rc']].apply(pd.to_numeric)
data_set.replace('yes', 1, inplace = True)
data_set.replace('no', 0, inplace = True)
data_set.replace('notpresent', 1, inplace = True)
data_set.replace('present', 0, inplace = True)
data_set.replace('abnormal', 1, inplace = True)
data_set.replace('normal', 0, inplace = True)
data_set.replace('poor', 1, inplace = True)
data_set.replace('good', 0, inplace = True)
data_set.replace('ckd', 1, inplace = True)
data_set.replace('notckd', 0, inplace = True)
column = data_set.columns
data_set= data_set.astype(float)

data_set = data_set.sample(frac=1)

# Tuning Data
avg = []
for i in range(0,25):
    stk = data_set[data_set.columns[i]].mean()
    avg.append(stk)
    data_set[column[i]].replace(np.nan, stk, inplace=True)
    
data_set.to_csv('chronic_kidney_disease_full_tuned.csv', sep='\t')

data_set = pd.read_table('chronic_kidney_disease_full_tuned.csv')
data_set = data_set.astype(float)

data_set =data_set.drop(['Unnamed: 0'], axis=1)

data_set1 = data_set.iloc[:,24]
data_set2 =data_set.drop(['class'], axis=1)


y = np.array(data_set1.values)
x = np.array(data_set2.values)

#Normalization
for i in range(400):
    x[i,:] = (x[i,:]-np.mean(x[i,:],axis=0))/(np.amax(x[i,:],axis=0).T-np.amin(x[i,:],axis=0).T)
x =x.reshape(-1,24)
y.reshape(-1,1)

train_set=int(x.shape[0]*80/100)

# Training Set
X_train = x[:train_set,:]
Y_train = y[:train_set]
# Testing Set
X_test = x[train_set:x.shape[0]+1]
Y_test = y[train_set:y.shape[0]+1]

####################################################################################  
# Parameter Initialization  
theta=np.zeros(len(X_train[1]))
lmda=list(np.linspace(-2,4,36))
m = int(len(X_train))
itr = 10000
learning_rate=0.1
fmeas_train=[]
fmeas_test=[]
#####################################################################################
# Logistic Regression
for i in lmda:
    warnings.warn = ig_warn
    tht = grad_desc(X_train,Y_train,itr,theta,i, m,learning_rate)
    fmeas_tr=predict_accu(X_test,Y_test, tht)
    fmeas_train.append(fmeas_tr)
    fmeas_ts=predict_accu(X_train,Y_train,tht)
    fmeas_test.append(fmeas_ts)
    
print('F-measure for Logistic Regression on training set:')
plt.figure()
plt.plot(lmda,fmeas_test,'b-')
plt.xlabel('Regularization Parameter- Lambda')
plt.ylabel('F-measure')
plt.title('F-measure for training set vs Lambda')
plt.show()
for i in range(len(lmda)):
    print('F-measure when lambda is %f = %f '%(lmda[i], fmeas_test[i]))
    
print('\nF-measure for Logistic Regression on test set:')
plt.figure()
plt.plot(lmda,fmeas_train,'y-')
plt.xlabel('Regularization Parameter- Lambda')
plt.ylabel('F-measure')
plt.title('F-measure for test set vs Lambda')
plt.show()
for i in range(len(lmda)):
    print('F-measure when lambda is %f = %f '%(lmda[i], fmeas_train[i]))
#########################################################################################
# Standardization
X_train = standardize(X_train)
X_test = standardize(X_test)
fmeas_train_std = []
fmeas_test_std = []
# Logistic Regression
for i in lmda:
    warnings.warn = ig_warn
    tht = grad_desc(X_train,Y_train,itr,theta,i, m,learning_rate)
    fmeas_tr=predict_accu(X_test,Y_test, tht)
    fmeas_train_std.append(fmeas_tr)
    fmeas_ts=predict_accu(X_train,Y_train,tht)
    fmeas_test_std.append(fmeas_ts)
    
print('F-measure for Logistic Regression on training set:')
plt.figure()
plt.plot(lmda,fmeas_test_std,'b-')
plt.xlabel('Regularization Parameter- Lambda')
plt.ylabel('F-measure')
plt.title('F-measure for training set vs Lambda')
plt.show()
for i in range(len(lmda)):
    print('F-measure when lambda is %f = %f '%(lmda[i], fmeas_test_std[i]))
    
print('\nF-measure for Logistic Regression on test set:')
plt.figure()
plt.plot(lmda,fmeas_train_std,'y-')
plt.xlabel('Regularization Parameter- Lambda')
plt.ylabel('F-measure')
plt.title('F-measure for test set vs Lambda')
plt.show()
for i in range(len(lmda)):
    print('F-measure when lambda is %f = %f '%(lmda[i], fmeas_train_std[i]))
#######################################################################################
