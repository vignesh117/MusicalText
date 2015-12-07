__author__ = 'vignesh'
import pickle
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import ConfigParser as CP
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
import numpy as np
#from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
#import plot_learning_curve
from sklearn.cross_validation import cross_val_score



class LearnAndClassify(object):


    features = None
    featurestest = None
    sentences = None
    sentencestest = None
    model = None
    modelnb = None
    modelrf = None
    modelgb = None
    traindata = []
    testdata = []

    def __init__(self):
        self.sentences = pickle.load(open('Sentences.pickle'))
        self.sentencestest = pickle.load(open('SentencesTest.pickle'))
        self.features = pickle.load(open('features.pickle'))
        self.featurestest = pickle.load(open('featuresTest.pickle'))
        self.gen_training_data()
        #self.svm_cv_model()
        self.create_svm_model()
        # self.create_nb_mode()
        self.create_rf_model()
        # self.create_gb_model()
        self.construct_test_data()
        self.fit_model_on_test()


    def gen_training_data(self):
        """
        Generates training
        :return:
        """

        traindata = {}

        if self.features == None:
            raise 'Features are empty'

        f = self.features.values()

        # flatten features

        X = sum(f, []) # flattening lists

        # Get the y values

        labels = [s.labels for s in self.sentences]
        # flatten labels
        Y = [val for sublist in labels for val in sublist] # flatten the labels

        traindata['X'] = X
        traindata['Y'] = Y
        self.traindata = traindata

    def construct_test_data(self):
        """
        Constructs the test data set for feeding to
        the SVC classifier
        :return:
        """

        testdata = {}

        if self.featurestest == None:
            raise 'The test set is empyt'

        f = self.featurestest.values()
        X = sum(f, [])

        # Get the y values

        labels = [s.labels for s in self.sentencestest]
        Y = [val for sublist in labels for val in sublist] # flatten the labels

        testdata['X'] = X
        testdata['Y'] = Y
        self.testdata = testdata

    def create_svm_model(self):

        """
        Creates an svm model
        :return:
        """

        X = self.traindata['X']
        Y = self.traindata['Y']
        # #clf = OneVsRestClassifier(svm.LinearSVC(penalty="l1"))
        # #clf = svm.SVC(kernel = 'linear', )
        # clf = Pipeline([
        #      ('feature_selection', SelectFromModel(RandomForestClassifier())),
        #      ('classification', svm.SVC(C = 1, kernel='rbf',probability=False, class_weight='balanced'))
        #         ])
        #
        # #clf = svm.SVC(C = 1, kernel='linear',probability=True, class_weight='balanced')
        #clf.fit(X, Y)

        clf = Pipeline([('chi2', SelectKBest(chi2, k=30)),
                ('svm', svm.LinearSVC(class_weight='balanced'))])

        multi_clf = OneVsRestClassifier(clf)
        multi_clf.fit(X,Y)
        self.model = multi_clf

        print multi_clf
    		
    def create_nb_mode(self):

        X = self.traindata['X']
        Y = self.traindata['Y']
        gnb = GaussianNB()
        model = gnb.fit(X,Y)
        self.modelnb = model

    def create_rf_model(self):
        X = self.traindata['X']
        Y = self.traindata['Y']

        clf = Pipeline([
             ('feature_selection', SelectFromModel(svm.SVC(C = 1, kernel='linear',probability=False, \
                                                           class_weight='balanced'))),
             ('classification', RandomForestClassifier(n_estimators=1000, class_weight='balanced' ))
                ])
        #clf = RandomForestClassifier(n_estimators=10000, class_weight='balanced')
        clf = clf.fit(X,Y)
        self.modelrf = clf

    def create_gb_model(self):
        X = self.traindata['X']
        Y = self.traindata['Y']
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,
                                         max_depth=1, random_state=0, warm_start=True)
        clf = clf.fit(X,Y)
        self.modelgb = clf
        
    def svm_cv_model(self):

        X = self.traindata['X']
        y = self.traindata['Y']
        
        X = np.array(X)
        y = np.array(y)
        class_weight = {'NM' : 1, '//' : 14, '/' : 33}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        estimator = svm.SVC(class_weight= 'balanced', kernel = 'rbf')
        cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        class_weight =  [{'NM':  0.5 - i, '/': i + 8., '//' : i + 10} for i in range(10)]
        param_grid = dict(gamma = gamma_range, C=C_range)
        #param_grid = dict(C=C_range)
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid)
        classifier.fit(X_train, y_train)
        
        #title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
        #estimator = svm.SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
        #learning_curve.plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
        #plt.show()
        #print cross_val_score(classifier, X, y)
        #classifier.fit(X, y)
        print classifier.score(X_test, y_test)

        self.model = classifier





    def fit_model_on_test(self):

        if self.featurestest == None:
            raise 'The test feature set is empty'

        #print self.model.score(self.testdata['X'], self.testdata['Y'])
        #
        #print self.model.predict(self.testdata['X'])
        # print self.modelnb.predict(self.testdata['X'])
        ypred_svm = self.model.predict(self.testdata['X'])

        print 'Printing accuracy score'

        print 'SVM with class balance score'
        print accuracy_score(self.testdata['Y'], ypred_svm)

         # Classification report on all the classifiers
        target_names = ['/','//','NM']

        print classification_report(self.testdata['Y'], ypred_svm, target_names=target_names)

        ypred_rf = self.modelrf.predict(self.testdata['X'])
        # ypred_gb = self.modelgb.predict(self.testdata['X'])

        #print ypred_gb
        #print ypred_rf
        #print ypred_svm



        print 'Random forest score'
        print accuracy_score(ypred_rf, self.testdata['Y'])



        print 'Classification report for random forest'
        print classification_report(self.testdata['Y'], ypred_rf, target_names=target_names)

        # print 'Classification report for Gradient boosting score'
        # print classification_report(self.testdata['Y'], ypred_gb, target_names=target_names)

        # Verifying test data consistency
      
     #



if __name__ == '__main__':
    lc = LearnAndClassify()

