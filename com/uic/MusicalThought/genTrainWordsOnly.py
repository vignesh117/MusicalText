__author__ = 'vignesh'

import ConfigParser as cp

# Read the configuration from the configuration file

config = cp.RawConfigParser()
config.read('config.cfg')
trainfile = config.get('init','trainfile')

# Tokenize
traindata = open(trainfile).read()
theWords = traindata.split(' ')

# TODO : convert to lower case
theWords = [word.replace('.','').replace(',','').replace('\n','') for word in theWords] #removing leading and trailing spaces
theWords = [x for x in theWords if x != '']
theClasses = {}

for i in range(len(theWords)):

    curr_word = theWords[i]

    # Case where the word is either // or /
    # the class of the previous word is // or / correspondingly
    if curr_word == '/' or curr_word == '//':
        theClasses[ i - 1] = curr_word
        theClasses[ i] = 'None'

    elif '/' in curr_word:

        # Replace / with ''
        theWords[ i] = curr_word.replace('/', '')
        theClasses[ i] = '/' #Marking the class for the current word as /

    elif '//' in curr_word:

        # Replace // with ''
        theWords[ i] = curr_word.replace('//','')
        theClasses[ i] = '//'

    else:
        theClasses[ i] = 'Other'


# Display
classes = theClasses.values()

for i in range(len(theWords)):
    if classes[ i] != 'None':
        print classes[ i] + "\t" + theWords[ i]

