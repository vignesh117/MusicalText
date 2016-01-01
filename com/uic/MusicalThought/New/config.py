[init]

# final full train corpus. Combines both the first and second set
trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/labelledfullcorpus.txt
#trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/Annotated/corpus1025.txt
#trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/sappa.txt
#trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/FullCorpusAnnotated.txt
#trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/allcorpuses.txt
#trainfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/labelledsecondset/labelledsecondsetfullfixed.txt

#testfile= /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/Annotated/corpus1025.txt
#testfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/sappa.txt
testfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/processed/WhyImakerobotsthesizeofagrainofsize.txt

#unlabelled corpus
unlabelled = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/unlabelledcorpusfullfixed.txt

reportdir = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/toReport
transwordsfile=/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/transitionwords.txt
conjfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/conjunctions.txt
templistfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/temporalConnectives.txt
resultfile = /Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/corpus/results/testresult.txt

[NER]
classifier=/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/StanfordNERTagger/stanford-ner-2015-04-20/classifiers/english.muc.7class.distsim.crf.ser.gz
tagger=/Users/vignesh/Documents/Phd/Courses/fall15/CS521SNLP/ClassProject/StanfordNERTagger/stanford-ner-2015-04-20/stanford-ner.jar

[CV]
cv = 1
perclass = 1
featgroupres = 0

# For LU learning
[LU]
lu = 0