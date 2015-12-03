__author__ = 'vignesh'
import ConfigParser as cp
from TrainDocument import TrainDocument
from TestDocument import TestDocument
import pickle


if __name__ == '__main__':

    # Parse the configuration
    config = cp.RawConfigParser()
    config.read('config.cfg')
    trainfile = config.get('init', 'trainfile')
    content = open(trainfile).read()

    doc = TrainDocument(content)
    sent = doc.get_sentences()[0]
    pickle.dump(doc,open('TrainDoc.pickle','w'))
    pickle.dump(doc.sentences,open('Sentences.pickle','w'))

    # Open test documents as wel
    testfile = config.get('init', 'testfile')
    testcontent = open(testfile).read()
    testdoc = TestDocument(testcontent)
    pickle.dump(doc.sentences, open('SentencesTest.pickle','w'))
