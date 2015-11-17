__author__ = 'vignesh'

import ConfigParser as cp
import os
from nltk.tokenize import sent_tokenize


def countSentAndSpaces(s):
    sentences = sent_tokenize(s) # Tokenize into sentences
    spaces = s.count(' ')
    newline = s.count('\n')
    tabs = s.count('\t')
    posslabels = spaces + newline + tabs
    return len(sentences), posslabels


config = cp.RawConfigParser()
config.read('config.cfg')
reportdir = config.get('init', 'reportdir')

reportfiles = os.listdir(reportdir)

# Count all sentences and spaces
totalSentences = 0
totalSpaces = 0

for r in reportfiles:
    content = open(reportdir + '/' + r).read()
    (st, spaces) = countSentAndSpaces(content)
    print r, st, spaces
    totalSentences += st
    totalSpaces += spaces

print totalSentences, totalSpaces

