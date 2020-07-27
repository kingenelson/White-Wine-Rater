import os.path
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

## Data loading utility functions
def get_test_train(fname,seed,datatype):
    '''
    Returns a test/train split of the data in fname shuffled with
    the given seed

    Args:
        fname:      A str/file object that points to the CSV file to load, passed to
                    np.genfromtxt()
        seed:       The seed passed to np.random.seed(). Typically an int or long
        datatype:   The datatype to pass to genfromtxt(), usually int, float, or str


    Returns:
        train_X:    A NxD np array of training data (row-vectors), 80% of all data
        train_Y:    A Nx1 np array of class labels for the training data
        test_X:     A MxD np array of testing data, same format as train_X, 20% of all data
        test_Y:     A Mx1 np array of class labels for the testing data
    '''
    data = np.genfromtxt(fname,delimiter=';',dtype=datatype)[1:,:] # Removing the header
    np.random.seed(seed)
    shuffled_idx = np.random.permutation(data.shape[0])
    cutoff = int(data.shape[0]*0.8)
    train_data = data[shuffled_idx[:cutoff]]
    test_data = data[shuffled_idx[cutoff:]]
    train_X = train_data[:,:-1].astype(float)
    train_Y = train_data[:,-1].reshape(-1,1)
    test_X = test_data[:,:-1].astype(float)
    test_Y = test_data[:,-1].reshape(-1,1)
    return train_X, train_Y, test_X, test_Y

def load_data(path='data'):
    return get_test_train(os.path.join(path,'winequality-white.csv'),seed=1567708904,datatype=float)

def output_results(cv_results):
    res = {}
    idx = np.argsort(cv_results['rank_test_score'])
    res['params'] = np.array(cv_results['params'])[idx]
    res['mean_test_score'] = np.array(cv_results['mean_test_score'])[idx]
    res['std_test_score'] = np.array(cv_results['std_test_score'])[idx]
    return res


def round1():
    data = load_data()

    res = []

    # RandomForestClassifier
    np.random.seed(1)
    param = {'n_estimators':[10,100,1000],'min_samples_split':[2,3,4]}
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, param,cv=3)
    clf.fit(data[0],data[1].flatten())
    score = clf.best_estimator_.score(data[2],data[3].flatten())
    print(f'RandomForestClassifier w/ CV:{score}')
    res.append({'RandomForestClassifier':output_results(clf.cv_results_)})

    # AdaBoostClassifier
    np.random.seed(1)
    param = {'n_estimators':[10,100,1000],'base_estimator':[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2)]}
    abc = AdaBoostClassifier()
    clf = GridSearchCV(abc, param,cv=3)
    clf.fit(data[0],data[1].flatten())
    score = clf.best_estimator_.score(data[2],data[3].flatten())
    print(f'AdaBoostClassifier w/ CV:{score}')
    res.append({'AdaBoostClassifier':output_results(clf.cv_results_)})

    # Linear SVM
    param = {'C': [.5, 1, 10],'kernel':['linear'],'max_iter':[1000]}
    np.random.seed(1)
    svc = svm.SVC()
    clf = GridSearchCV(svc, param,cv=3)
    clf.fit(data[0],data[1].flatten())
    score = clf.best_estimator_.score(data[2],data[3].flatten())
    res.append({'SVM linear':output_results(clf.cv_results_)})
    print(f'SVM Linear:{score}')

    # Non-Linear SVM
    np.random.seed(1)
    param = [
        {'C': [.5, 1, 10],'gamma':10.0**-np.arange(1,4),'max_iter':[1000]},
        {'C': [.5, 1, 10],'kernel':['poly'],
        'degree':2+np.arange(3),'gamma':10.0**-np.arange(1,4),'max_iter':[1000]}
    ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param,cv=3)
    clf.fit(data[0],data[1].flatten())
    score = clf.best_estimator_.score(data[2],data[3].flatten())
    res.append({'SVM Non-linear':output_results(clf.cv_results_)})
    print(f'SVM Non-Linear:{score}')

    # MLPClassifier
    np.random.seed(1)
    scalar = StandardScaler()
    scalar.fit(data[0])
    pdata = scalar.transform(data[0]), scalar.transform(data[2])

    np.random.seed(1)
    h_layers = (11,11,11)
    param = {'hidden_layer_sizes':h_layers,'alpha':10.0 ** -np.arange(1, 5),'max_iter':[1000]} # ,'max_iter':[100,200,500,1000]
    mlp = MLPClassifier()
    clf = GridSearchCV(mlp,param,cv=3)
    clf.fit(pdata[0],data[1].flatten())
    score = clf.best_estimator_.score(pdata[1],data[3].flatten())
    print(f'MLPClassifier:{score}')
    res.append({'MLPClassifier':output_results(clf.cv_results_)})

    print(res)

def round2():
    # Round two of opt
    d1,d2,d3,d4 = load_data()

    # r controls the number of permutations of the data
    r = 5
    x,y = np.concatenate((d1,d3),axis=0),np.concatenate((d2,d4),axis=0)
    idx = np.array([np.random.permutation(x.shape[0]) for i in range(r)])
    cutoff = int(x.shape[0]*0.8)
    train_X, test_X = np.array([x[idx[i][:cutoff]] for i in range(r)]), np.array([x[idx[i][cutoff:]] for i in range(r)])
    train_Y, test_Y = np.array([y[idx[i][:cutoff]] for i in range(r)]), np.array([y[idx[i][cutoff:]] for i in range(r)])

    # RandomForestClassifier
    np.random.seed(1)
    clf = RandomForestClassifier(n_estimators=1000)
    out = []
    for i in range(r):
        clf.fit(train_X[i],train_Y[i].flatten())
        out.append(clf.score(test_X[i],test_Y[i].flatten()))
    print(f'{out} {np.mean(out)} {np.std(out)}')

    # AdaBoostClassifier
    np.random.seed(1)
    clf = AdaBoostClassifier(n_estimators=10,base_estimator=DecisionTreeClassifier(max_depth=2))
    out = []
    for i in range(r):
        clf.fit(train_X[i],train_Y[i].flatten())
        out.append(clf.score(test_X[i],test_Y[i].flatten()))
    print(f'{out} {np.mean(out)} {np.std(out)}')

    # Linear SVM
    np.random.seed(1)
    clf = svm.SVC(C=.5,kernel='linear',max_iter=1000)
    out = []
    for i in range(r):
        clf.fit(train_X[i],train_Y[i].flatten())
        out.append(clf.score(test_X[i],test_Y[i].flatten()))
    print(f'{out} {np.mean(out)} {np.std(out)}')

    # Non-Linear SVM
    np.random.seed(1)
    clf = svm.SVC(C=10,gamma=.1,max_iter=1000)
    out = []
    for i in range(r):
        clf.fit(train_X[i],train_Y[i].flatten())
        out.append(clf.score(test_X[i],test_Y[i].flatten()))
    print(f'{out} {np.mean(out)} {np.std(out)}')

    # MLPClassifier
    np.random.seed(1)
    scalar = StandardScaler()

    np.random.seed(1)
    clf = MLPClassifier(hidden_layer_sizes=(11,11,11),alpha=.001,max_iter=1000)
    out = []
    for i in range(r):
        scalar.fit(train_X[i])
        pdata = scalar.transform(train_X[i]), scalar.transform(test_X[i])
        clf.fit(pdata[0],train_Y[i].flatten())
        out.append(clf.score(pdata[1],test_Y[i].flatten()))
    print(f'{out} {np.mean(out)} {np.std(out)}')

def main():
    ###CODE###
    import warnings
    warnings.filterwarnings("ignore")

    round1()
    round2()


if __name__ == '__main__':
    main()
