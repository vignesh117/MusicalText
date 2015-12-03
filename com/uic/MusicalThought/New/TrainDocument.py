__author__ = 'vignesh'

from nltk.tokenize import sent_tokenize
from Sentence import Sentence

"""
This class contains the entire corpora.

"""

class TrainDocument(object):
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
        for i in range(len(sent)):
            s = Sentence(sent[i],i)
            self.sentences.append(s)




