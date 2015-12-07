__author__ = 'vignesh'

from nltk.tokenize import sent_tokenize
from Sentence import Sentence
from nltk.tag.stanford import StanfordNERTagger
from jsonrpc import *
import ConfigParser as CP

"""
This class contains the entire corpora.

"""

class TestDocument(object):
    """
    Each document contains a bunch of sentences
    """


    document = None # it is the raw text
    sentences = []

    def __init__(self, doc):
        self.document = doc
        self.make_sentences()

    def get_sentences(self):
        return self.sentences

    def set_sentences(self,sentences):
        self.sentences = sentences

    def make_sentences(self):

        """
        Makes sentences from raw documents.
        Each sentence is wrapped up in a sentence class
        :return: None
        """

        if self.document == None:
            return

        sent = sent_tokenize(self.document) # contains raw sentences


        # Create parameters for NER and Dependency Parsing a
        # and pass it to the sentence objcet

        # set config file
        config = CP.RawConfigParser()
        config = config
        config.read('config.cfg')

        # Server for dependency parsing

        server = ServerProxy(JsonRpc20(),TransportTcpIp(addr=("127.0.0.1", 8080)))

        # Parameters for Named entitye recognition

        # get the classifier and tagger location from config file
        tagger = config.get('NER','tagger') # gets the path of the stanford tagger
        classifier = config.get('NER','classifier') # gets the path of the stanford classifier
        st = StanfordNERTagger(classifier,tagger)
        for i in range(len(sent)):
            s = Sentence(sent[i],i,server, st, 'test')
            self.sentences.append(s)




