__author__ = 'vignesh'

# Imports
#-----------------------------------------
from nltk.tokenize import word_tokenize
import nltk
import jsonrpc
from jsonrpc import *
from json import loads
import re
#-----------------------------------------

class Sentence(object):
    """

    """
    ann_sent = None # Annotated sentence
    sent_pos = None
    orig_sent = None
    pos_tags = None
    deps = None # Dependency parsing dependencies
    #word_pos = None
    labels = None # Class labels
    candidatepos = [] # Candidate positions in the sentence
    type = 'train' # Tells if the sentence is a training or testing sentence


    def __init__(self, sent, pos, type = 'train'):
        self.sent_pos = pos

        # Curate annotated sentence
        sent = sent.strip(" \n\r")
        self.type = type
        self.ann_sent = sent
        self.ann_sent = sent
        self.set_orig_sent()
        self.set_pos_tags()
        self.set_dep()
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

        tokens = word_tokenize(self.orig_sent)
        print tokens
        tagsandwords = nltk.pos_tag(tokens)
        onlytags = [x[1] for x in tagsandwords]
        self.pos_tags = onlytags

    def set_dep(self):

        """
        Get dependency parsing tags
        :return:
        """
        print 'Generating dependency tree ...'

        #TODO: Check if we need to create a word object and store dependencies

        server = ServerProxy(JsonRpc20(),TransportTcpIp(addr=("127.0.0.1", 8080)))
        s = self.orig_sent # Getting
        try:
            result = loads(server.parse(s)) # This generates the dependencies
        except RPCInternalError:
            raise 'Corenlp server is not available'
            return
        dependencies = []
        sentences = result['sentences']
        for i in range(len(sentences)):
            dep = result['sentences'][i]['dependencies'] # Get the dependencies from the json
            dependencies += dep

        self.deps = dependencies

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


        if self.type == 'test': # We just have to set the candidate positions

            self.candidatepos = [m.start() for m in re.finditer(" ", self.orig_sent)]
        labels = []
        markers = ['/ ','//',' ']
        sent = self.ann_sent

        # find positions of each marker
        markerpositions = {}
        reversedict = {}
        for ma in markers:
            markerpositions[ma] = [m.start() for m in re.finditer(ma, sent)]

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

        print reversedict

     # Convert marker positions to classlables
        invKeysSorted = sorted(reversedict.keys())
        print invKeysSorted
        for i in range(len(invKeysSorted)):

            l = reversedict[invKeysSorted[i]]
            if l == ' ':
                labels.append('NM')
            else:
                labels.append(l)

        self.labels = labels # labels for candidate positions
        self.candidatepos = invKeysSorted # Candidate key positions


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





