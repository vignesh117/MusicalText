__author__ = 'vignesh'

#===================
# I M P O R T S
#==================

import pickle
import ConfigParser as cp
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_predict
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from ClassLabelEnum import ClassLabelEnum
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelPropagation
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from tabulate import tabulate
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from functools import partial
import re
import random
import numpy as np

# For getting balanced accuracy
from numpy import *
from pandas import *
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

class LearnAndClassify(object):

    # Parsing configuration files
    config = cp.RawConfigParser()
    config.read('config.py')

    features = None
    featurestest = None
    featuresunlabelled = None
    sentences = None
    sentencestest = None
    sentencesunlabelled = None
    model = None
    modelnb = None
    modelrf = None
    modelgb = None
    modellr = None
    modelsgd = None
    modelrandom = None
    modellu = None
    traindata = []
    testdata = []
    cv = None
    lu = None

    # feature group indices
    """
    We have identified 3 feature groups
    1. Linguistic features
    2. Word level features
    3. Grammatical features
    below are index positions for each of the feature groups
    """
    #lingindices = [0, 1, 2,3,6,11,12,13,14,15,16,17]
    #wlindices = [4,5,10]
    #gramindices = [7,8,9]

    wlindices = range(0,22)
    gramindices = [22, 23, 24, 25, 26, 27]
    lingindices = range(28,298)

    def __init__(self):
        self.sentences = pickle.load(open('Sentences.pickle'))
        self.sentencestest = pickle.load(open('SentencesTest.pickle'))
        self.sentencesunlabelled = pickle.load(open('SentencesUnlabelled.pickle'))
        self.features = pickle.load(open('features.pickle'))
        self.featurestest = pickle.load(open('featuresTest.pickle'))
        self.featuresunlabelled = pickle.load(open('featuresUnlabelled.pickle'))
        self.gen_training_data()
        self.construct_test_data()
        #self.svm_cv_model()

        # correlation coefficient or feature significance
        #self.get_featuregroup_significance()
        #classification results
        self.cv = self.config.get('CV','cv')
        self.lu = self.config.get('LU','lu')
        perclass = self.config.get('CV','perclass')
        featgroupres = self.config.get('CV','featgroupres')

        if self.cv.strip(' ') == '0':
            #
            if self.lu.strip(' \r\n') == '1':
                self.gen_semisup_data()
                self.create_LU_model()
                self.get_lu_results_on_test()
                return

            if perclass ==1:
                if featgroupres == 1:
                    self.get_perclass_featgroup_report()
                    return

            self.create_svm_model()
            self.create_nb_mode()
            self.create_rf_model()
            self.create_lr_model()
            self.create_sgd_model()
            self.create_adaboost_model()
            self.create_gb_model()
            self.create_random_model()
            self.create_dummy_model()
            self.fit_model_on_test()
        else:

            # LU learning with Cross validation
            if self.lu == '1':
                self.gen_semisup_data()
                self.create_LU_model()
                self.get_lu_results_on_test(cv = 1)
                return

            type = 'train'
            if perclass.strip(' ') == '1':

                if featgroupres.strip(' ') == '1':
                    print 'Generating per class CV classification report for feature groups'
                    self.get_cv_perclass_ftgroups_report()
                else:
                    #print 'Generating per class CV classification report'
                    self.get_cv_perclass_report(type = type)
            else:
                print 'Generating CV results for all classes'
                self.get_cv_results()
        self.disp_rand_test_res()

    def gen_training_data(self):
        """
        Generates training
        :return:
        """
        traindata = {}
        if self.features == None:
            raise 'Features are empty'

        # Generate training data
        X = []
        fkeys = self.features.keys()
        for k in fkeys:
            X += self.features[k]

        Y = []
        for i in range(len(self.sentences)):
            Y += self.sentences[i].labels

        #
        # f = self.features.values()
        #
        # # flatten features
        #
        # X = sum(f, []) # flattening lists
        #
        # # Get the y values
        #
        # labels = [s.labels for s in self.sentences]
        # # flatten labels
        # Y = [val for sublist in labels for val in sublist] # flatten the labels
        traindata['X'] = X
        traindata['Y'] = Y
        self.traindata = traindata

    def gen_semisup_data(self):
        """
        Generates semisupervised data by combining
        training and unlabelled data
        :return:
        """

        print 'Generating semi supervised data...'
        semisupdata = {}
        if self.featuresunlabelled == None:
            raise 'Features for unlabelled data are empty'

        # Generate unlabelled data
        X = []
        fkeys = self.featuresunlabelled.keys()
        for k in fkeys:
            X += self.featuresunlabelled[k]

        Y = []
        for i in range(len(self.sentencesunlabelled)):
            Y += self.sentencesunlabelled[i].labels
        #
        X = X[:8000]
        Y = Y[:8000]

        # Now combine training data and unlabelled data
        trainx = self.traindata['X']
        trainy = self.traindata['Y']

        X = X + trainx
        Y = Y + trainy

        """
        Semi supervised models take only numeric class labels
        Converting the labels to numeric
        '/'  : 0
        '//' : 1
        'NM' : 2
        """

        Y = [ClassLabelEnum(x) for x in Y]
        Y = [x.get_label() for x in Y]

        semisupdata['X'] = X
        semisupdata['Y'] = Y
        self.semisupdata = semisupdata

    def create_LU_model(self):
        """
        Creates a semisupervised learning model
        by performing label propogation
        :return:
        """

        print 'Creating LU model ...'

        if self.semisupdata == None:
            raise ' Semi supervised data is not generated'
            return

        X = self.semisupdata['X']
        Y = self.semisupdata['Y']
        #lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
        lp_model = LabelPropagation()
        lp_model.fit(X,Y)
        self.modellu = lp_model # labelled unlabelled learning model

    def construct_test_data(self):
        """
        Constructs the test data set for feeding to
        the SVC classifier
        :return:
        """

        testdata = {}

        if self.featurestest == None:
            raise 'The test set is empyt'


             # Generate training data
        X = []

        fkeys = self.features.keys()
        for k in fkeys:
            X += self.features[k]

        Y = []
        for i in range(len(self.sentences)):
            Y += self.sentences[i].labels


        # f = self.featurestest.values()
        # X = sum(f, [])
        #
        # # Get the y values
        #
        # labels = [s.labels for s in self.sentencestest]
        # Y = [val for sublist in labels for val in sublist] # flatten the labels

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

        clf = Pipeline([('chi2', SelectKBest(chi2)),
                ('svm', svm.LinearSVC(class_weight='balanced'))])

        multi_clf = OneVsRestClassifier(clf)
        multi_clf.fit(X,Y)
        self.model = multi_clf

        print multi_clf

    def create_nb_mode(self):

        X = self.traindata['X']
        Y = self.traindata['Y']
        mnb = MultinomialNB()
        model = mnb.fit(X,Y)
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

        """
        Gradient boosting classifier
        :return:
        """
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

    def create_lr_model(self):

        """
        Logistic regression model
        :return:
        """

        X = self.traindata['X']
        Y = self.traindata['Y']
        #clf = LogisticRegression(class_weight='balanced')
        clf =  LogisticRegressionCV(
                  Cs=50,
                  cv=4,
                  penalty='l2',
                  fit_intercept=True,
                  scoring='f1'
        )
        clf.fit(X,Y)
        self.modellr = clf

    def create_sgd_model(self):

        X = self.traindata['X']
        Y = self.traindata['Y']
        clf = SGDClassifier(class_weight='balanced')
        clf.fit(X,Y)
        self.modelsgd = clf

    def create_adaboost_model(self):
        X = self.traindata['X']
        Y = self.traindata['Y']

        baseestimator = DecisionTreeClassifier(class_weight='balanced')
        clf = AdaBoostClassifier(base_estimator = baseestimator, n_estimators=100)
        clf.fit(X,Y)
        self.modelada = clf

    def create_random_model(self):
        """

        Randomly assigns 3 classes to the input data
        :return:
        """

        class_labels = list(set(self.testdata['Y']))
        randomlabels = []
        for i in range(len(self.testdata['Y'])):
            randomlabels.append(random.choice(class_labels))
        return randomlabels

    def create_dummy_model(self):

        """
        Dummy classifier for baseline performance
        :return:
        """

        # Getting prior distributions

        X = self.traindata['X']
        Y = self.traindata['Y']


        clf = DummyClassifier(strategy='stratified')
        clf.fit(X,Y)
        self.modelrandom = clf

    def perf_measure(self,y_actual, y_hat):

        truth      = y_actual
        prediction = y_hat

        confusion_matrix = Counter()

        #say class 1, 3 are true; all other classes are false
        positives = ['/', '//']

        binary_truth = [x in positives for x in truth]
        binary_prediction = [x in positives for x in prediction]
        # print binary_truth
        # print binary_prediction

        for t, p in zip(binary_truth, binary_prediction):
            confusion_matrix[t,p] += 1

        # print "TP: {} TN: {} FP: {} FN: {}".format(confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False])
        (TP, TN, FP, FN) = (confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False])
        # TP = 0
        # FP = 0
        # TN = 0
        # FN = 0
        #
        # for i in range(len(y_hat)):
        #     if y_actual[i]==y_hat[i]:
        #        TP += 1
        # for i in range(len(y_hat)):
        #     if y_actual[i]==1 and y_actual!=y_hat[i]:
        #        FP += 1
        # for i in range(len(y_hat)):
        #     if y_actual[i]==y_hat[i]==0:
        #        TN += 1
        # for i in range(len(y_hat)):
        #     if y_actual[i]==0 and y_actual!=y_hat[i]:
        #        FN += 1
        #
        return(TP, FP, TN, FN)

    def fit_model_on_test(self):

        if self.featurestest == None:
            raise 'The test feature set is empty'



        # Getting accuracy scores for crossvalidated
        # and train test paradigms based on the setting

        ypred_svm = self.model.predict(self.testdata['X'])
        ypred_rf = self.modelrf.predict(self.testdata['X'])
        ypred_lr = self.modellr.predict(self.testdata['X'])
        ypred_gb = self.modelgb.predict(self.testdata['X'])
        ypred_ada = self.modelada.predict(self.testdata['X'])
        ypred_sgd = self.modelsgd.predict(self.testdata['X'])
        ypred_nb = self.modelnb.predict(self.testdata['X'])
        ypred_random = self.modelrandom.predict(self.testdata['X'])


        # print ' Accuracy scores'
        # print 'SVM with class balance score'
        # ac_svm =  accuracy_score(self.testdata['Y'], ypred_svm)
        #
        # print 'Random forest score'
        # ac_rf =  accuracy_score(ypred_rf, self.testdata['Y'])
        #
        # print 'Logistic regression score'
        # ac_lr =  accuracy_score(self.testdata['Y'], ypred_lr)
        #
        # print 'Gradient Boosting score'
        # ac_gb =  accuracy_score(self.testdata['Y'], ypred_gb)
        #
        # print 'Ada Boosting score'
        # ac_ada =  accuracy_score(self.testdata['Y'], ypred_ada)
        #
        # print 'SGD model score'
        # ac_sgd =  accuracy_score(self.testdata['Y'], ypred_sgd)
        #
        # print 'Multinomial NB Accuracy score '
        # ac_nb =  accuracy_score(self.testdata['Y'], ypred_nb)
        #
        # print 'Random model Accuracy score'
        # ac_rand = accuracy_score(self.testdata['Y'], ypred_random)

        #accuracies = [ac_svm, ac_rf, ac_lr, ac_gb, ac_ada, ac_sgd, ac_nb]

         # Classification report on all the classifiers


        #target_names = ['/','//','NM']

        # print 'Classification report for SVM'
        # print classification_report(self.testdata['Y'], ypred_svm, target_names=target_names)
        #
        # print 'Classification report for random forest'
        # print classification_report(self.testdata['Y'], ypred_rf, target_names=target_names)
        #
        # print 'Classification report for Logistic regression model'
        # print classification_report(self.testdata['Y'], ypred_lr)
        #
        # print 'Classification report for Gradient Boosting model'
        # print classification_report(self.testdata['Y'], ypred_gb)
        #
        # print 'Classification report for Ada Boosting model'
        # print classification_report(self.testdata['Y'], ypred_ada)

        # print 'Classification report for Stochastic gradient model'
        # print classification_report(self.testdata['Y'], ypred_sgd)
        #
        # report = classification_report(self.testdata['Y'], ypred_sgd)

        classifiers = [ypred_svm, ypred_sgd,ypred_rf, ypred_lr, ypred_gb, ypred_ada]
        classifnames = ['SVM', 'SGD','RandomForest','LogisticRegression','GB','ADA']
        for i in range(len(classifiers)):

            ypred = classifiers[i]
            classifiername = classifnames[i]
            print 'Train test Classification report for : '+ classifiername
            report = classification_report(self.testdata['Y'], ypred)
            results = self.get_results_from_classreport(report)
            classlabels = ['/','//','NM']
            for l in classlabels:

                # get the results
                res = results.get(l) # Class specific results
                f1 = res.get('f1')
                print 'F1 for '+l+': ' + str(f1)


            print '===================================\n'
            # get the balanced accuracy
            """
            We can get balanced accuracy by assuming the
            class of interest as positive and the other classes as negative
            """
            balaccuracies = {}
            df = {'ypred' : ypred, 'y' : self.testdata['Y']}
            df = pd.DataFrame(df)
            rdf = com.convert_to_r_dataframe(df)
            base = importr('base')
            caret = importr('caret')
            ypredfact = base.factor(rdf[0])
            ytruefact = base.factor(rdf[1])
            mat = caret.confusionMatrix(ytruefact, ypredfact)
            s = com.convert_robj(mat[3])
            cc = s['Balanced Accuracy']
            cc = dict(cc)
            print cc
            print '===================================\n'


        # Classification report for random model
        print 'Classification report for Random  model'
        print classification_report(self.testdata['Y'], ypred_random)

        report = classification_report(self.testdata['Y'], ypred_random)
        results = self.get_results_from_classreport(report)
        classlabels = ['/','//','NM']
        for l in classlabels:

            # get the results
            res = results.get(l) # Class specific results
            f1 = res.get('f1')
            print 'F1 for '+l+': ' + str(f1)


        print '===================================\n'
        # get the balanced accuracy
        """
        We can get balanced accuracy by assuming the
        class of interest as positive and the other classes as negative
        """
        balaccuracies = {}
        df = {'ypred' : ypred_random, 'y' : self.testdata['Y']}
        df = pd.DataFrame(df)
        rdf = com.convert_to_r_dataframe(df)
        base = importr('base')
        caret = importr('caret')
        ypredfact = base.factor(rdf[0])
        ytruefact = base.factor(rdf[1])
        mat = caret.confusionMatrix(ytruefact, ypredfact)
        s = com.convert_robj(mat[3])
        cc = s['Balanced Accuracy']
        cc = dict(cc)
        print cc


        # print 'Classification report for Multinomial naive bayes'
        # print classification_report(self.testdata['Y'], ypred_nb)
        #
        # print 'Classification report for Random model'
        # print classification_report(self.testdata['Y'], ypred_random)


        #=========================

        # Max vote classifier

        #=======================

        # allvotes = [ypred_gb, ypred_lr, ypred_sgd,ypred_nb]
        # combinevotes = zip(*allvotes)
        #
        # mode = lambda list : max(set(list), key=list.count)
        # ypred_mvote = [mode(x) for x in combinevotes]
        #
        # # find the maximum vote
        # print 'Classification report for maximum vote classifier'
        # print classification_report(self.testdata['Y'], ypred_mvote)


        # print 'Classification report for Gradient boosting score'
        # print classification_report(self.testdata['Y'], ypred_gb, target_names=target_names)

        # Verifying test data consistency

      
        # # Final classification report
        # print ypred_svm
        #
        # print 'FINAL REPORT'
        #
        # ytrue = self.testdata['Y']
        # classifiers = ['SVM', 'RF','LR','GB', 'SGD', 'NB', 'ADA','Vot', 'Random']
        # predictions = [ypred_svm, ypred_rf, ypred_lr, ypred_gb, ypred_sgd, ypred_nb, \
        #                ypred_ada, ypred_mvote, ypred_random]
        # resultssvm = precision_recall_fscore_support(ytrue, ypred_svm, average=None, labels = target_names)
        # resultsrf = precision_recall_fscore_support(ytrue, ypred_rf, average=None, labels = target_names)
        # resultslr = precision_recall_fscore_support(ytrue, ypred_lr, average=None, labels = target_names)
        # resultsgb = precision_recall_fscore_support(ytrue, ypred_gb, average=None, labels = target_names)
        # resultssgd = precision_recall_fscore_support(ytrue, ypred_sgd, average=None, labels = target_names)
        # resultsnb = precision_recall_fscore_support(ytrue, ypred_nb, average=None, labels = target_names)

        # prdict = {}
        # for i in range(len(classifiers)):
        #     classifier = classifiers[i]
        #     (TP, FP, TN, FN) = self.perf_measure(ytrue, predictions[i])
        #     TP = float(TP)
        #     FP = float(FP)
        #     TN = float(TN)
        #     FN = float(FN)
        #
        #     # results
        #     try:
        #         accuracy = (TP + TN ) / (TP + FP + TN + FN)
        #     except ZeroDivisionError:
        #         accuracy = 0
        #     try:
        #         recall = (TP) / (TP + FP)
        #     except ZeroDivisionError:
        #         recall = 0
        #     try:
        #         prec = (TP) / (TP + FN)
        #     except ZeroDivisionError:
        #         prec = 0
        #     try:
        #         bal_acc = ( (0.5 * TP) / (TP + FN) ) + ( (0.5 * TN) / (TN + FP))
        #         inf = ( 2 * bal_acc - 1) # 'Informedness'
        #     except ZeroDivisionError:
        #         bal_acc =0
        #         inf = 0
        #     resdict = {}
        #     resdict['accuracy'] = accuracy
        #     resdict['recall'] = recall
        #     resdict['precision'] = prec
        #     resdict['balaccuracy'] = bal_acc
        #     resdict['inf'] = inf
        #     prdict[classifier] = resdict
        #
        # print tabulate(prdict.items())

    def average_cv_results(self, cvresults):

        """
        Averages the cross validation results
        across k folds for all the classifiers
        :param cvresults: Cross validation results
        :return: Average crossvalidation
        """

        folds = cvresults.keys()
        nfolds = len(folds)
        classifiers = cvresults.get(folds[0]).keys() #
        average_result = OrderedDict()

        results = cvresults.values()

        # For each classifier get the classifier results
        for c in classifiers:

            # Get all the metrics

            get_acc = lambda x:x.get(c)['accuracy']
            get_rec = lambda x:x.get(c)['recall']
            get_prec = lambda x:x.get(c)['precision']
            get_balacc = lambda x:x.get(c)['balaccuracy']
            get_inf = lambda x:x.get(c)['inf']


            temp = map(get_acc, results)
            accuracy = sum(map(get_acc, results)) / nfolds
            recall = sum(map(get_rec, results)) / nfolds
            prec = sum(map(get_prec, results)) / nfolds
            balacc = sum(map(get_balacc, results)) / nfolds
            inf = sum(map(get_inf, results)) / nfolds


            # Publish the results
            prdict = {}
            prdict['accuracy'] = accuracy
            prdict['recall'] = recall
            prdict['precision'] = prec
            prdict['balaccuracy'] = balacc
            prdict['inf'] = inf

            average_result[c] = prdict

        return average_result

    def get_cv_results(self):
        """
        Get crossvalidated results for all the models
        :return:
        """

        # Get the 10 fold cv splits
        n = len(self.traindata['X'])
        nfolds = 10
        kf = KFold(n, n_folds=nfolds)

        # Get the split and run train and test for all of them
        kfoldresults = OrderedDict()
        fold = 1
        for train, test in kf:

            # train and test are the indices given by kfold
            traindata_x = [self.traindata['X'][i] for i in range(len(self.traindata['X'])) if i in train]
            testdata_x = [self.traindata['X'][i] for i in range(len(self.traindata['X'])) if i in test]
            traindata_y = [self.traindata['Y'][i] for i in range(len(self.traindata['X'])) if i in train]
            testdata_y = [self.traindata['Y'][i] for i in range(len(self.traindata['X'])) if i in test]


            # Train all the models on the current split

            X = traindata_x
            Y = traindata_y


            # Logistic regression model

            lr_model=  LogisticRegressionCV(
              Cs=50,
              cv=4,
              penalty='l2',
              fit_intercept=True,
              scoring='f1'
            )
            lr_model.fit(X,Y)
            ypred_lr = lr_model.predict(testdata_x)

            #SVM model

            clf = Pipeline([('chi2', SelectKBest(chi2, k=30)),
                ('svm', svm.LinearSVC(class_weight='balanced'))])

            svm_model = OneVsRestClassifier(clf)
            svm_model.fit(X,Y)
            ypred_svm = svm_model.predict(testdata_x)

            # Random forest model
            rf_model = Pipeline([
                ('feature_selection', SelectFromModel(svm.SVC(C = 1, kernel='linear',probability=False, \
                                                           class_weight='balanced'))),
                ('classification', RandomForestClassifier(n_estimators=1000, class_weight='balanced' ))
                ])
            rf_model.fit(X,Y)
            ypred_rf = rf_model.predict(testdata_x)

            # Grading Boosted
            gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,
                                         max_depth=1, random_state=0, warm_start=True)
            gb_model.fit(X,Y)
            ypred_gb = gb_model.predict(testdata_x)

            # SGD model
            sgd_model = SGDClassifier(class_weight='balanced')
            sgd_model.fit(X,Y)
            ypred_sgd = sgd_model.predict(testdata_x)

            # Naive Bayes model

            mnb = MultinomialNB()
            nb_model = mnb.fit(X,Y)
            ypred_nb = nb_model.predict(testdata_x)

            # Ada boost classifier
            baseestimator = DecisionTreeClassifier(class_weight='balanced')
            ada_model = AdaBoostClassifier(base_estimator = baseestimator, n_estimators=100)
            ada_model.fit(X,Y)
            ypred_ada = ada_model.predict(testdata_x)

            # Max vote classifier

            allvotes = [ypred_gb, ypred_lr, ypred_sgd,ypred_nb]
            combinevotes = zip(*allvotes)

            mode = lambda list : max(set(list), key=list.count)
            ypred_mvote = [mode(x) for x in combinevotes]

            # Random classifier
            random_model = DummyClassifier(strategy='stratified')
            random_model.fit(X,Y)
            ypred_random = random_model.predict(testdata_x)

            # Now test with the test data
            ytrue = testdata_y
            #classifiers = ['SVM', 'RF','LR','GB', 'SGD', 'NB', 'ADA','Vot']
            classifiers = ['LR','SVM', 'RF', 'GB', 'SGD', 'NB', 'ADA', 'Vot', 'Random']

            predictions = [ypred_lr, ypred_svm, ypred_rf, ypred_gb, ypred_sgd, ypred_nb,\
                           ypred_ada, ypred_mvote, ypred_random]
            #predictions = [ypred_svm, ypred_rf, ypred_lr, ypred_gb, ypred_sgd, ypred_nb, ypred_ada, ypred_mvote]

            # Get all the results for each of the classifiers
            prdict = {}
            for i in range(len(classifiers)):
                classifier = classifiers[i]
                (TP, FP, TN, FN) = self.perf_measure(ytrue, predictions[i])
                TP = float(TP)
                FP = float(FP)
                TN = float(TN)
                FN = float(FN)

                # results
                try:
                    accuracy = (TP + TN ) / (TP + FP + TN + FN)
                except ZeroDivisionError:
                    accuracy = 0
                try:
                    recall = (TP) / (TP + FP)
                except ZeroDivisionError:
                    recall = 0
                try:
                    prec = (TP) / (TP + FN)
                except ZeroDivisionError:
                    prec = 0
                try:
                    bal_acc = ( (0.5 * TP) / (TP + FN) ) + ( (0.5 * TN) / (TN + FP))
                    inf = ( 2 * bal_acc - 1) # 'Informedness'
                except ZeroDivisionError:
                    bal_acc =0
                    inf = 0
                resdict = {}
                resdict['accuracy'] = accuracy
                resdict['recall'] = recall
                resdict['precision'] = prec
                resdict['balaccuracy'] = bal_acc
                resdict['inf'] = inf
                prdict[classifier] = resdict
            kfoldresults[ fold] = prdict
            fold += 1


        average_results = self.average_cv_results(kfoldresults)
        for k in average_results.keys():
            print k
            print '=============='
            print tabulate(average_results.get(k).items())

    def get_results_from_classreport(self,report):

        """
        This functions extracts the precision,
        recall, fscores from the classification report
        :param report: Classification report
        :return:
        """

        lines = report.split('\n') # Splitting based on line break
        lines = lines[1:] # skipping the header line
        lines = [x for x in lines if x != ''] # removing empty
        lines = lines[:-1] #Last line contains the total result - omitting that

        resultdict = {}
        for line in lines:
            dict = {}
            temp = re.split('\s+',line)
            temp = [x for x in temp if x != '']
            f1 = float(temp[-2])
            recall = float(temp[-3])
            precision = float(temp[-4])
            classlabel = temp[0] # First element of the line is class label
            dict['precision'] = precision
            dict['recall'] = recall
            dict['f1'] = f1

            # add dict to the resultdict
            resultdict[classlabel] = dict

        return resultdict

    def get_cv_perclass_report(self, type = 'train'):

         # Logistic regression model
        lr_model=  LogisticRegressionCV(
          Cs=50,
          cv=4,
          penalty='l2',
          fit_intercept=True,
          scoring='f1'
        )

        #SVM model

        clf = Pipeline([('chi2', SelectKBest(chi2, k=30)),
            ('svm', svm.LinearSVC(class_weight='balanced'))])

        svm_model = OneVsRestClassifier(clf)
        #svm_model = svm.SVC(kernel = 'rbf')

        # Random forest model
        rf_model = Pipeline([
            ('feature_selection', SelectFromModel(svm.SVC(C = 1, kernel='linear',probability=False, \
                                                       class_weight='balanced'))),
            ('classification', RandomForestClassifier(n_estimators=1000, class_weight='balanced' ))
            ])

        # Grading Boosted
        gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,
                                     max_depth=1, random_state=0, warm_start=True)
        # SGD model
        sgd_model = SGDClassifier(class_weight='balanced')

        # Naive Bayes model
        mnb = MultinomialNB()

        # Ada boost classifier
        baseestimator = DecisionTreeClassifier(class_weight='balanced')
        ada_model = AdaBoostClassifier(base_estimator = baseestimator, n_estimators=100)


        # Random classifier
        random_model = DummyClassifier(strategy='stratified')

        #classifnames = ['LR','SVM', 'RF', 'SGD', 'NB', 'ADA', 'Random']
        #classifiers = [lr_model, svm_model, rf_model, sgd_model, mnb, ada_model, random_model]

        classifnames = ['GB']
        classifiers = [gb_model]

        # Get train and test data

        if type.lower().strip(' ') == 'train':
            print 'Generating CV results for labelled data'
            X = self.traindata['X']
            y = self.traindata['Y']
        elif type.lower().strip(' ') == 'lu':
            print 'Generating CV results for Semisupervised data'
            X = self.semisupdata['X']
            y = self.semisupdata['Y']
                 # Convert the class labels to categorical again
            labelmap = {}
            labelmap[0] = '/'
            labelmap[1] = '//'
            labelmap[2] = 'NM'
            y = [labelmap[x] for x in y]

        allpredictions = []
        for i in range(len(classifiers)):
            estimator = classifiers[i]
            classifiername = classifnames[i]
            y_pred = cross_val_predict(estimator, X, y, cv=10)
            print 'Generating CV classification report for '+classifiername
            print '-----------------------------------'
            report = classification_report(y, y_pred)
            results = self.get_results_from_classreport(report)

            classlabels = ['/','//','NM']
            for l in classlabels:

                # get the results
                res = results.get(l) # Class specific results
                f1 = res.get('f1')
                print 'F1 for '+l+': ' + str(f1)


            print '===================================\n'
            # get the balanced accuracy
            """
            We can get balanced accuracy by assuming the
            class of interest as positive and the other classes as negative
            """
            balaccuracies = {}
            df = {'ypred' : y_pred, 'y' : y}
            df = pd.DataFrame(df)
            rdf = com.convert_to_r_dataframe(df)
            base = importr('base')
            caret = importr('caret')
            ypredfact = base.factor(rdf[0])
            ytruefact = base.factor(rdf[1])
            mat = caret.confusionMatrix(ytruefact, ypredfact)
            s = com.convert_robj(mat[3])
            cc = s['Balanced Accuracy']
            cc = dict(cc)
            print cc
            # Add the predictions to the overall predictions for
            # generating results for the maximum vote classifier
            #allpredictions.append(y_pred)

        # # Generate results for multivote classification
        # combinevotes = zip(*allpredictions)
        # mode = lambda list : max(set(list), key=list.count)
        # ypred_mvote = [mode(x) for x in combinevotes]

        # print classification report for the max vote classifier
        # print classification_report(y, ypred_mvote)

    def get_cv_perclass_ftgroups_report(self):
          # Logistic regression model
        # lr_model=  LogisticRegressionCV(
        #   Cs=50,
        #   cv=4,
        #   penalty='l2',
        #   fit_intercept=True,
        #   scoring='f1'
        # )
        #
        # #SVM model
        # clf = Pipeline([('chi2', SelectKBest(chi2, k=10)),
        #      ('svm', svm.LinearSVC(class_weight='balanced'))])
        # svm_model = OneVsRestClassifier(clf)
        # #svm_model = svm.SVC(kernel = 'rbf')

        # Random forest model
        # rf_model = Pipeline([
        #     ('feature_selection', SelectFromModel(svm.SVC(C = 1, kernel='linear',probability=False, \
        #                                                class_weight='balanced'))),
        #     ('classification', RandomForestClassifier(n_estimators=1000, class_weight='balanced' ))
        #     ])

        # # Grading Boosted
        # gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,
        #                              max_depth=1, random_state=0, warm_start=True)
        # SGD model
        sgd_model = SGDClassifier(class_weight='balanced')

        # # Naive Bayes model
        # mnb = MultinomialNB()
        #
        # # Ada boost classifier
        # baseestimator = DecisionTreeClassifier(class_weight='balanced')
        # ada_model = AdaBoostClassifier(base_estimator = baseestimator, n_estimators=100)
        #
        # Random classifier
        random_model = DummyClassifier(strategy='stratified')

        # #classifnames = ['LR','SVM', 'RF', 'GB', 'SGD', 'NB', 'ADA', 'Random']
        # #classifiers = [lr_model, svm_model, rf_model, gb_model, sgd_model, mnb, ada_model, random_model]

        # # reduced classifier list
        # classifnames = ['LR', 'SVM', 'RF', 'MNB','GB','Random']
        # classifiers = [lr_model, svm_model, rf_model, mnb, gb_model, random_model]

        classifnames = ['SGD']
        classifiers = [sgd_model]
        # Get train and test data
        X = self.traindata['X']
        y = self.traindata['Y']

        # Code for feature group
        groups = ['Linguistic','Word','Grammatical', 'LingandWord', 'LingandGram', 'WordandGram', 'All']
        Xlist = []

        def getfeatdata(trainx, indices):
            temp = []
            for i in range(len(trainx)):
                example = trainx[i]
                temp2 = []
                for j in range(len(indices)):
                    temp2.append(example[j])
                #example = [example[j] for j in range(len(example)) if j in indices]
                temp.append(temp2)
            return temp


        getdataforfeat = lambda x,y : [x[i] for i in y] # x is the training example and y is the indices

        # Linguistic features
        #mapfunc = partial(getdataforfeat, y = self.lingindices)
        #Xling = map(mapfunc, X)
        Xling = getfeatdata(X, self.lingindices)
        Xlist.append(Xling)

        #Word level features
        #mapfunc = partial(getdataforfeat, y = self.wlindices)
        #Xword = map(mapfunc, X)
        Xword = getfeatdata(X, self.wlindices)
        Xlist.append(Xword)

        #Grammatical features
        #mapfunc = partial(getdataforfeat, y = self.gramindices)
        #Xgram = map(mapfunc, X)
        Xgram = getfeatdata(X, self.gramindices)
        Xlist.append(Xgram)

        # Linguistic and Word level combination
        #mapfunc = partial(getdataforfeat, y = self.lingindices + self.wlindices)
        #Xlingword = map(mapfunc, X)
        Xlingword = getfeatdata(X, self.lingindices + self.wlindices)
        Xlist.append(Xlingword)

        # Linguistic and Grammaatical combination
        #mapfunc = partial(getdataforfeat, y = self.lingindices + self.gramindices)
        #Xlinggram = map(mapfunc, X)
        Xlinggram = getfeatdata(X, self.lingindices + self.gramindices)
        Xlist.append(Xlinggram)

        # Word level and grammatical combination
        #mapfunc = partial(getdataforfeat, y = self.gramindices + self.wlindices)
        #Xwordgram = map(mapfunc, X)
        Xwordgram = getfeatdata(X, self.gramindices + self.wlindices)
        Xlist.append(Xwordgram)

        # All features

        #mapfunc = partial(getdataforfeat, y = self.gramindices + self.wlindices + self.lingindices)
        #Xall = map(mapfunc, X)
        Xall = getfeatdata(X, self.gramindices + self.wlindices + self.lingindices)
        Xlist.append(Xall)

        # Generate crossvalidated results here
        for i in range(len(groups)):

            X = Xlist[i] # redefining X here for getting
            print len(X)
            allpredictions = []
            print 'Getting results for the ' + groups[i] +' group'
            for j in range(len(classifiers)):
                estimator = classifiers[j]
                classifiername = classifnames[j]
                y_pred = cross_val_predict(estimator, X, y, cv=10)

                print 'Generating CV classification report for '+classifiername
                print '-----------------------------------'
                print classification_report(y, y_pred)
                print '===================================\n'

                # Add the predictions to the overall predictions for
                # generating results for the maximum vote classifier
                allpredictions.append(y_pred)
                print '===================================\n'
                # get the balanced accuracy
                """
                We can get balanced accuracy by assuming the
                class of interest as positive and the other classes as negative
                """
                balaccuracies = {}
                df = {'ypred' : y_pred, 'y' : y}
                df = pd.DataFrame(df)
                rdf = com.convert_to_r_dataframe(df)
                base = importr('base')
                caret = importr('caret')
                ypredfact = base.factor(rdf[0])
                ytruefact = base.factor(rdf[1])
                mat = caret.confusionMatrix(ytruefact, ypredfact)
                s = com.convert_robj(mat[3])
                cc = s['Balanced Accuracy']
                cc = dict(cc)
                print cc


    def get_perclass_featgroup_report(self):

        print 'Generating results for traintest for feature groups'
        sgd_model = SGDClassifier(class_weight='balanced')

        # # Naive Bayes model
        # mnb = MultinomialNB()
        #
        # # Ada boost classifier
        # baseestimator = DecisionTreeClassifier(class_weight='balanced')
        # ada_model = AdaBoostClassifier(base_estimator = baseestimator, n_estimators=100)
        #
        # Random classifier
        random_model = DummyClassifier(strategy='stratified')

        # #classifnames = ['LR','SVM', 'RF', 'GB', 'SGD', 'NB', 'ADA', 'Random']
        # #classifiers = [lr_model, svm_model, rf_model, gb_model, sgd_model, mnb, ada_model, random_model]

        # # reduced classifier list
        # classifnames = ['LR', 'SVM', 'RF', 'MNB','GB','Random']
        # classifiers = [lr_model, svm_model, rf_model, mnb, gb_model, random_model]

        classifnames = ['SGD']
        classifiers = [sgd_model]
        # Get train and test data
        X = self.traindata['X']
        y = self.traindata['Y']

        # Code for feature group
        groups = ['Linguistic','Word','Grammatical', 'LingandWord', 'LingandGram', 'WordandGram', 'All']
        Xlist = []

        def getfeatdata(trainx, indices):
            temp = []
            for i in range(len(trainx)):
                example = trainx[i]
                temp2 = []
                for j in range(len(indices)):
                    temp2.append(example[j])
                #example = [example[j] for j in range(len(example)) if j in indices]
                temp.append(temp2)
            return temp



        # Linguistic features
        Xling = getfeatdata(X, self.lingindices)
        Xlist.append(Xling)

        #Word level features
        Xword = getfeatdata(X, self.wlindices)
        Xlist.append(Xword)

        #Grammatical features
        Xgram = getfeatdata(X, self.gramindices)
        Xlist.append(Xgram)

        # Linguistic and Word level combination
        Xlingword = getfeatdata(X, self.lingindices + self.wlindices)
        Xlist.append(Xlingword)

        # Linguistic and Grammaatical combination
        Xlinggram = getfeatdata(X, self.lingindices + self.gramindices)
        Xlist.append(Xlinggram)

        # Word level and grammatical combination
        Xwordgram = getfeatdata(X, self.gramindices + self.wlindices)
        Xlist.append(Xwordgram)

        # All features

        Xall = getfeatdata(X, self.gramindices + self.wlindices + self.lingindices)
        Xlist.append(Xall)

        # Generate traintest results here
        for i in range(len(groups)):

            X = Xlist[i] # redefining X here for getting
            print len(X)
            allpredictions = []
            print 'Getting results for the ' + groups[i] +' group'
            for j in range(len(classifiers)):
                estimator = classifiers[j]
                classifiername = classifnames[j]

                # fit the classifier
                model = estimator.fit(X,y)
                testx = self.testdata['X']
                testy = self.testdata['Y']
                y_pred = model.predict(testx)

                print 'Generating CV classification report for '+classifiername
                print '-----------------------------------'
                print classification_report(testy, y_pred)
                print '===================================\n'

                # Add the predictions to the overall predictions for
                # generating results for the maximum vote classifier
                allpredictions.append(y_pred)
                print '===================================\n'
                # get the balanced accuracy
                """
                We can get balanced accuracy by assuming the
                class of interest as positive and the other classes as negative
                """
                balaccuracies = {}
                df = {'ypred' : y_pred, 'y' : y}
                df = pd.DataFrame(df)
                rdf = com.convert_to_r_dataframe(df)
                base = importr('base')
                caret = importr('caret')
                ypredfact = base.factor(rdf[0])
                ytruefact = base.factor(rdf[1])
                mat = caret.confusionMatrix(ytruefact, ypredfact)
                s = com.convert_robj(mat[3])
                cc = s['Balanced Accuracy']
                cc = dict(cc)
                print cc


    # Deprecated
    def disp_rand_test_res(self):

        randomlabels = self.create_random_model()
        ytrue = self.testdata['X']

        print 'Classification report for Random Classifier'
        print classification_report(self.testdata['Y'], randomlabels)

        prdict = {}
        (TP, FP, TN, FN) = self.perf_measure(ytrue, randomlabels)
        TP = float(TP)
        FP = float(FP)
        TN = float(TN)
        FN = float(FN)
        print (TP, FP, TN, FN)

        # results
        try:
            accuracy = (TP + TN ) / (TP + FP + TN + FN)
        except ZeroDivisionError:
            accuracy = 0
        try:
            recall = (TP) / (TP + FP)
        except ZeroDivisionError:
            recall = 0
        try:
            prec = (TP) / (TP + FN)
        except ZeroDivisionError:
            prec = 0
        try:
            bal_acc = ( (0.5 * TP) / (TP + FN) ) + ( (0.5 * TN) / (TN + FP))
            inf = ( 2 * bal_acc - 1) # 'Informedness'
        except ZeroDivisionError:
            bal_acc =0
            inf = 0
        resdict = {}
        resdict['accuracy'] = accuracy
        resdict['recall'] = recall
        resdict['precision'] = prec
        resdict['balaccuracy'] = bal_acc
        resdict['inf'] = inf
        prdict['Random'] = resdict
        print tabulate(prdict.items())

    def get_lu_results_on_test(self, cv = 0):

        """
        Get the results of running the LU model
        on test data
        :return:
        """
        print 'Testing LU model on test data and generating results....'
        testx = self.testdata['X']
        testy = self.testdata['Y']

        model = self.modellu

        if cv == 0:
            ypred = model.predict(testx)

            # Convert the class labels to categorical again
            labelmap = {}
            labelmap[0] = '/'
            labelmap[1] = '//'
            labelmap[2] = 'NM'
            ypred = [labelmap[x] for x in ypred]

            print 'Classification report on the LU model with test data'
            print  classification_report(testy, ypred)

            report = classification_report(testy, ypred)
            results = self.get_results_from_classreport(report)
            classlabels = ['/','//','NM']
            for l in classlabels:

                # get the results
                res = results.get(l) # Class specific results
                f1 = res.get('f1')
                print 'F1 for '+l+': ' + str(f1)


            print '===================================\n'
            # get the balanced accuracy
            """
            We can get balanced accuracy by assuming the
            class of interest as positive and the other classes as negative
            """
            balaccuracies = {}
            df = {'ypred' : ypred, 'y' : testy}
            df = pd.DataFrame(df)
            rdf = com.convert_to_r_dataframe(df)
            base = importr('base')
            caret = importr('caret')
            ypredfact = base.factor(rdf[0])
            ytruefact = base.factor(rdf[1])
            mat = caret.confusionMatrix(ytruefact, ypredfact)
            s = com.convert_robj(mat[3])
            cc = s['Balanced Accuracy']
            cc = dict(cc)
            print cc



        else:
            estimator = model
            classifiername = 'LU'
            y_pred = cross_val_predict(estimator, self.semisupdata['X'], self.semisupdata['Y'], cv=10)
            print 'Generating CV classification report for '+classifiername
            print '-----------------------------------'
            y = self.semisupdata['Y']
            report = classification_report(y, y_pred)
            results = self.get_results_from_classreport(report)

            print list(set(y))
            #classlabels = ['/','//','NM']
            classlabels = [0,1,2]
            for l in classlabels:

                # get the results
                res = results.get(l) # Class specific results
                f1 = res.get('f1')
                print 'F1 for '+l+': ' + str(f1)


            print '===================================\n'
            # get the balanced accuracy
            """
            We can get balanced accuracy by assuming the
            class of interest as positive and the other classes as negative
            """
            balaccuracies = {}
            df = {'ypred' : y_pred, 'y' : testy}
            df = pd.DataFrame(df)
            rdf = com.convert_to_r_dataframe(df)
            base = importr('base')
            caret = importr('caret')
            ypredfact = base.factor(rdf[0])
            ytruefact = base.factor(rdf[1])
            mat = caret.confusionMatrix(ytruefact, ypredfact)
            s = com.convert_robj(mat[3])
            cc = s['Balanced Accuracy']
            cc = dict(cc)
            print cc


    def get_featuregroup_significance(self):

        """
        Gives the significance of each feature group
        with respect to each class label
        :return:
        """

        feature_groups = ['l', 'w','g'] # l : linguistic, g : grammatical, w : word
        classes = ['/','//','NM']

        X = self.traindata['X']
        Y = self.traindata['Y']

        print Counter(Y)
        # for each data point,
        # get the contributions of each feature group
        contributions = defaultdict(list)
        for i in range(len(X)):
            example = X[i]
            l = any([example[j] for j in self.lingindices])
            w = any([example[j] for j in self.wlindices])
            g = any([example[j] for j in self.gramindices])
            contributions['l'].append(l)
            contributions['w'].append(w)
            contributions['g'].append(g)

            # Add the class indicator
            for c in classes:
                contributions[c].append(c == Y[i])

        # Now find the correlations
        for c in classes:
            for f in feature_groups:
                print 'Correlations between the class : ' + c +' and feature group ' + f
                print '----------------------------------------'
                print matthews_corrcoef(contributions.get(c), contributions.get(f))
                print '========================================'



if __name__ == '__main__':
    lc = LearnAndClassify()
