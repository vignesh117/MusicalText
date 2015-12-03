__author__ = 'vignesh'
import pickle
from sklearn import svm

class LearnAndClassify(object):


    features = None
    sentences = None
    model = None

    def __init__(self):
        self.sentences = pickle.load(open('Sentences.pickle'))
        self.features = pickle.load(open('features.pickle'))
        self.gen_training_data()
        self.create_svm_model()

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

    def create_svm_model(self):

        """
        Creates an svm model
        :return:
        """

        X = self.traindata['X']
        Y = self.traindata['Y']
        clf = svm.SVC()
        clf.fit(X, Y)
        self.model = clf



if __name__ == '__main__':
    lc = LearnAndClassify()
    print lc.model
