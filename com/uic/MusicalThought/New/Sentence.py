__author__ = 'vignesh'

# Imports
#-----------------------------------------
from nltk.tokenize import word_tokenize
import nltk
import jsonrpc
from jsonrpc import *
from json import loads
import re
from nltk.tag.stanford import StanfordNERTagger
import ConfigParser as CP

# Setting for handling the unicode issue
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#-----------------------------------------

class Sentence(object):
    """

    """
    ann_sent = None # Annotated sentence
    sent_pos = None
    orig_sent = None
    pos_tags = None
    deps = None # Dependency parsing dependencies
    nerdict = None # Dictionary of Named entities for a sentence
    #word_pos = None
    labels = None # Class labels
    candidatepos = [] # Candidate positions in the sentence
    type = 'train' # Tells if the sentence is a training or testing sentence or unlabelled
    config = None

    # set config file
    config = CP.RawConfigParser()
    config = config
    config.read('config.py')

    # Dependency parsing server
    server = None

    # Named entity tagger
    st = None



    def __init__(self, sent, pos, server, st, type = 'train'):

        self.sent_pos = pos

        # Curate annotated sentence
        sent = sent.strip(" \n\r")
        self.type = type
        self.ann_sent = sent
        self.ann_sent = sent
        self.server = server
        self.st = st
        self.set_orig_sent()
        self.set_pos_tags()
        #self.set_dep()
        #self.set_ner()
        #self.set_candidate_pos()
        self.set_class_labels()

    def set_orig_sent(self):

        """
        Removes all class labels from the annotated sentence
        :return:
        """

        temp = self.ann_sent.replace('/','')
        temp = temp.replace('//','')
        temp = temp.replace('\n','')
        temp = temp.strip(' \r')
        self.orig_sent = temp

    def set_pos_tags(self):

        print 'Making pos tags...\n'

        if self.orig_sent == None:
            raise 'Error! Original sentence empty'
		
		 # remove punctuations
        punctuations = [',','//','/','?','.','!','"','\'']
        sent = self.orig_sent
        for p in punctuations:
            sent = sent.replace(p,'')
        tokens = word_tokenize(sent)
        tagsandwords = nltk.pos_tag(tokens)
        #onlytags = [x[1] for x in tagsandwords]
        self.pos_tags = dict(tagsandwords)

    def set_dep(self):

        """
        Get dependency parsing tags
        :return:
        """
        print 'Generating dependency tree ...'

        #TODO: Check if we need to create a word object and store dependencies
        server = self.server
        s = self.orig_sent # Getting
        print s
        try:
            result = loads(server.parse(s)) # This generates the dependencies
        except RPCInternalError:
            print 'Corenlp server is not available'
            return
        dependencies = []
        try:
            sentences = result['sentences']
            for i in range(len(sentences)):
                dep = result['sentences'][i]['dependencies'] # Get the dependencies from the json
                dependencies += dep

            self.deps = dependencies
        except KeyError: # happens when the sentence is sometimes too long
            self.deps = []

    def set_ner(self):


        nertags = self.st.tag(self.orig_sent.split())

        # Make a dictionary of the tags and store it
        nerdict = dict(nertags)
        self.nerdict = nerdict

    def set_candidate_pos(self):

        candidatepos = []
        if self.orig_sent == None:
            raise 'The sentence is empty'

        # Remove punctuations

        sent = self.orig_sent
        sent = sent.replace(',','')

        # find the candidate positions in the sentence
        candidatepos.append(0) # First position is a candidate position
        otherpositions = [m.start() for m in re.finditer('\s+', sent)] # find all occurances of space in sent
        candidatepos += otherpositions
        candidatepos.append((len(sent)) - 1)
        #print sent
        #print candidatepos

        self.candidatepos = candidatepos
        # count the number of blank spaces

    def set_class_labels(self):
        """
        Sets the class labels for this sentence
        :return:
        """


        # if self.type == 'test': # We just have to set the candidate positions
        #
        #     self.candidatepostest = [m.start() for m in re.finditer(" ", self.orig_sent)]


        # If the sentence is of type unlabelled, then we just have to set the class labels as -1

        # U N L A B E L L E D
        if self.type.lower() == 'unlabelled':
            sent = self.ann_sent
            positions = [m.start() for m in re.finditer(' ', sent)]
            positions = sorted(positions)
            labels = [-1 for i in range(len(positions))]
            self.labels = labels
            self.candidatepos = positions
            return

        else:

            # L A B E L L E D
            labels = []

            # find positions of each marker
            markerpositions = {}
            reversedict = {}
            exclusionlist = []
            sent = self.ann_sent

            markers0 = [' // ','// ']

            for ma in markers0:
                numpos = 4
                positions = [m.start() for m in re.finditer(ma, sent)]
                expositions = [range(x, x+numpos) for x in positions] # Positions that need to be excluded
                expositions = sum(expositions, [])

                # Filter all postions beyond sen len
                expositions = [x for x in expositions if x <= len(sent)]
                exclusionlist += expositions
                #markerpositions[ma] = [m.start() for m in re.finditer(ma, sent)]
                markerpositions['//'] = positions


            markers1 = [' / ']
            numpos = 3
            for ma in markers1:

                positions = [m.start() for m in re.finditer(ma, sent)]
                positions = [x for x in positions if x not in exclusionlist]
                expositions = [range(x, x+numpos) for x in positions] # Positions that need to be excluded
                expositions = sum(expositions, [])

                # Filter all postions beyond sen len
                expositions = [x for x in expositions if x <= len(sent)]
                exclusionlist += expositions
                #markerpositions[ma] = [m.start() for m in re.finditer(ma, sent)]
                markerpositions['/'] = positions


            markers2 = [' ']
            for ma in markers2:
                positions = [m.start() for m in re.finditer(ma, sent)]
                positions = [x for x in positions if x not in exclusionlist]
                markerpositions[ma] = positions

            # Create an inverse dict
            for k in markerpositions.keys():
                ind = markerpositions[k]
                ind  = sorted(ind)
                for i in ind:

                    reversedict[i] = k

                    # # handling conflicts between / and //
                    # if k == '//' and reversedict.get(i) == '/':
                    #     reversedict[i] = k
                    # elif k == '/' and reversedict.get(i) == '//':
                    #     continue
                    # elif k == '/' and reversedict.get(i-1) == '/':
                    #     reversedict[i-1] = '//'
                    #     continue
                    #
                    # elif k == '/' and reversedict.get(i+1) == '/':
                    #     reversedict[i] = '//'
                    #     continue
                    #
                    # else:
                    #     reversedict[i] = k


            invKeysSorted = sorted(reversedict.keys())
            # Convert marker positions to class lables
            for i in range(len(invKeysSorted)):

                l = reversedict[invKeysSorted[i]]
                if l == ' ':
                    labels.append('NM')
                else:
                    labels.append(l)

            if 0 not in invKeysSorted:
                invKeysSorted.append(0)
                labels = ['NM'] + labels

            if len(sent) not in invKeysSorted:
                invKeysSorted.append(len(sent))
                labels = labels + ['NM']

            invKeysSorted = sorted(invKeysSorted)
            self.labels = labels # labels for candidate positions
            self.candidatepos = invKeysSorted # Candidate key positions
            #
            # for zipped in zip(invKeysSorted, labels):
                # print zipped


            # tokens = word_tokenize(self.ann_sent)
            # tokens = [x.strip(' ') for x in tokens]
            #
            #  # First token could be contain a  marker or be a marker
            # if '/' not in tokens[0] or '//' not in tokens[0]:
            #     labels.append('NM')
            #
            #
            # for t in tokens:
            #     if '/' in t or '//' in t:
            #         labels.append(t)
            #
            # # Last token could be contain marker
            # if '/' not in tokens[-1] or '//' not in tokens[-1] :
            #     labels.append('NM')
            #
            # self.labels = labels

