__author__ = 'vignesh'
import ConfigParser as cp
# Setting for handling the unicode issue
import sys
from TrainDocument import TrainDocument
from TestDocument import TestDocument
from UnlabelledDoc import UnlabelledDoc
import pickle


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    # Parse the configuration
    config = cp.RawConfigParser()
    config.read('config.py')
    trainfile = config.get('init', 'trainfile')
    content = open(trainfile).read()

    doc = TrainDocument(content)
    sent = doc.get_sentences()[0]
    print sent.pos_tags
    pickle.dump(doc,open('TrainDoc.pickle','w'))
    pickle.dump(doc.sentences,open('Sentences.pickle','w'))

    # # Open test documents as wel
    testfile = config.get('init', 'testfile')
    testcontent = open(testfile).read()
    testdoc = TestDocument(testcontent)
    pickle.dump(testdoc.sentences, open('SentencesTest.pickle','w'))

    # IF lu learning is enabled, then read the unlabelled corpus also

    lu = int(config.get('LU','lu'))
    print 'LU learning configuration : ' + str(lu)
    if lu == 1 :
        #read the unlabelled corpus
        unlabelledfile = config.get('init','unlabelled')
        unlabelledcontent = open(unlabelledfile).read()
        unlabelleddoc = UnlabelledDoc(unlabelledcontent)
        pickle.dump(unlabelleddoc.sentences, open('SentencesUnlabelled.pickle','w'))

