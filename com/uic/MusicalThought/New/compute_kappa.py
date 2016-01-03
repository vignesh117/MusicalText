__author__ = 'vignesh'

import TrainDocument
from collections import Counter
import sys

f1 = '/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/kappa/vigneshkappafull.txt'


# reading
c1 = open(f1).read()

# Creating train corpus
doc1 = TrainDocument.TrainDocument(c1)
sent1 = doc1.get_sentences()

# sentences

labels1 = []
for i in range(len(sent1)):
    l1 = sent1[i].labels
    labels1 += l1

print len(labels1)
# write them

outfile = open('labels1forkappa.txt','w')
for i in range(len(labels1)):
    outfile.write(labels1[i])
    outfile.write('\n')
outfile.close()