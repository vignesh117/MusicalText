__author__ = 'vignesh'

import ConfigParser as cp


def getWordFeatures(word):

    wordFeatures = []
    if word[ 0].isupper():
        wordFeatures.append('Capitalized')
    if word[-1] == '.':
        wordFeatures.append('End')
    if ',' in word:
        wordFeatures.append('Comma')
    if word[-1] == '\n':
        wordFeatures.append('Last')
    return ' '.join(wordFeatures)

# Read the configuration from the configuration file

config = cp.RawConfigParser()
config.read('config.cfg')
trainfile = config.get('init','trainfile')

traindata = open(trainfile).read()
theWords = traindata.split(' ')
theWords = [x for x in theWords if x != '']
theFeatures = {}
theClasses  = {}

for i in range(len(theWords)):

    curr_word = theWords[i]

    # Case where the word is either // or /
    # the class of the previous word is // or / correspondingly
    if curr_word == '/' or curr_word == '//':
        theClasses[ i - 1] = curr_word
        wordFeatures = getWordFeatures(curr_word)
        curr_word = curr_word.lower()
        curr_word = curr_word.replace('.','').replace(',','').replace('\n','')
        wordFeatures = wordFeatures + ' ' + curr_word
        theFeatures[ i] = wordFeatures
        theClasses[ i] = 'None'

    elif curr_word != '/' and '/' in curr_word:

        # Replace / with ''
        curr_word = curr_word.replace('/', '')

        wordFeatures = getWordFeatures(curr_word)
        curr_word = curr_word.lower()
        curr_word = curr_word.replace('.','').replace(',','').replace('\n','')
        wordFeatures = wordFeatures + ' ' + curr_word
        theFeatures[ i] = wordFeatures
        theClasses[ i] = '/' #Marking the class for the current word as /

    elif curr_word != '//' and '//' in curr_word:

        # Replace // with ''
        curr_word = curr_word.replace('/', '')

        wordFeatures = getWordFeatures(curr_word)
        curr_word = curr_word.lower()
        curr_word = curr_word.replace('.','').replace(',','').replace('\n','')
        # if wordFeatures == []:
        #     wordFeatures = [curr_word]
        # else:
        #     wordFeatures.append(curr_word)
        wordFeatures = wordFeatures + ' ' + curr_word

        theFeatures[ i] = wordFeatures
        theClasses[ i] = '//'

    else:

        wordFeatures = getWordFeatures(curr_word)
        curr_word = curr_word.lower()
        curr_word = curr_word.replace('.','').replace(',','').replace('\n','')
        wordFeatures = wordFeatures + ' ' + curr_word

        theFeatures[ i] = wordFeatures

        theClasses[ i] = 'Other'


#Display

classes = theClasses.values()
features = theFeatures.values()
for i in range(len(features)):
    if classes[ i] != 'None':
        print classes[ i] + "\t" + features[ i]


