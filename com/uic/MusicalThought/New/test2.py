__author__ = 'vignesh'
from TrainDocument import TrainDocument
from collections import Counter
import sys

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')


trainfile = '/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/LabelledFirstSet/allfiles/corpus1025fixed.txt'
content = open(trainfile).read()
doc = TrainDocument(content)
labels = []
for s in doc.sentences:
    l = s.labels
    labels += l
print len(labels)
print Counter(labels)


