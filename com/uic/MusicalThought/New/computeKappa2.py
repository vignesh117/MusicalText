__author__ = 'vignesh'

from TrainDocument import TrainDocument
f2 = '/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/kappa/sahisnukappafull.txt'
c2 = open(f2).read()

doc2 = TrainDocument(c2)
sent2 = doc2.get_sentences()
labels2 = []

for i in range(len(sent2)):

    l2 = sent2[i].labels
    labels2 += l2

print len(labels2)
# write them

outfile = open('labels2forkappa.txt','w')
for i in range(len(labels2)):
    outfile.write(labels2[i])
    outfile.write('\n')
outfile.close()
