# coding=utf-8
__author__ = 'vignesh'
import ConfigParser as cp
from Sentence import Sentence
import pickle
import nltk
from nltk.tokenize import word_tokenize
from collections import OrderedDict


class ExtractFeatures(object):
    """
    Generates featurs for training and
    testing dataset
    """
    # Parsing configuration files
    config = cp.RawConfigParser()
    config.read('config.py')

    # Other declarations
    sentences = None
    sentencestest = None
    sentencesunlabelled = None
    features = None  # Dictionary of features for all sentences
    featurestest = None
    featuresunlabelled = None

    # variables for translatoin
    transwordsuni = []
    transwordbi = []
    transwordtri = []

    # variables for conjunction
    conjwordsuni = []
    conjwordsbi = []
    conjwordstri = []
    traindata = None  # Dictionary of traindata Containing X and Y

    # variables for temporal words
    tempwordsuni = []
    tempwordsbi = []
    tempwordstri = []


    # Example words
    exwords = ["like", "viz", "ie", "eg", 'for example', 'for instance', 'an example']

    def __init__(self):
        self.sentences = pickle.load(open('Sentences.pickle'))
        self.sentencestest = pickle.load(open('SentencesTest.pickle'))
        self.sentencesunlabelled = pickle.load(open('SentencesUnlabelled.pickle'))
        self.get_transition_words()
        self.get_conjunctions()
        self.get_temporal_words()
        self.compile_feature_vector()

        # Serialize feature
        pickle.dump(self.features, open('features.pickle', 'w'))
        pickle.dump(self.featurestest, open('featuresTest.pickle', 'w'))
        pickle.dump(self.featuresunlabelled, open('featuresUnlabelled.pickle', 'w'))

    def get_transition_words(self):

        """
        Gets the list of transition words from the
        transition words dictionary file
        :return:
        """
        transfile = self.config.get('init', 'transwordsfile')
        alltransitions = open(transfile).readlines()
        transwordsuni = []
        transwordsbi = []
        transwordstri = []

        # Unigram transition words
        for t in alltransitions:
            t = t.strip(' \r\n')
            splitt = t.split(' ')

            if len(splitt) == 1:
                transwordsuni.append(t)
            elif len(splitt) == 2:
                transwordsbi.append(t)
            elif len(splitt) == 3:
                transwordstri.append(t)

        # Assign it back to the class
        self.transwordsuni = [x.lower() for x in transwordsuni]  # converting to lower case
        self.transwordbi = [x.lower() for x in transwordsbi]  # converting to lower case
        self.transwordtri = [x.lower() for x in transwordstri]  # converting to lower case

    def get_conjunctions(self):
        conjfile = self.config.get('init', 'conjfile')
        allconjunctions = open(conjfile).readlines()
        conjwordsuni = []
        conjwordsbi = []
        conjwordstri = []

        # Unigram transition words
        for t in allconjunctions:
            t = t.strip(' \r\n')
            splitt = t.split(' ')

            if len(splitt) == 1:
                conjwordsuni.append(t)
            elif len(splitt) == 2:
                conjwordsbi.append(t)
            elif len(splitt) == 3:
                conjwordstri.append(t)

        # Assign it back to the class
        self.conjwordsuni = [x.lower() for x in conjwordsuni]  # converting to lower case
        self.conjwordsbi = [x.lower() for x in conjwordsbi]  # converting to lower case
        self.conjwordstri = [x.lower() for x in conjwordstri]  # converting to lower case
        self.conjwordsuni.append('and')

    def get_temporal_words(self):
        """
        Gets the list of temporal words
        :return:
        """

        tempwordlistfile = self.config.get('init', 'templistfile')
        allwords = open(tempwordlistfile).readlines()
        tempwordsuni = []
        tempwordsbi = []
        tempwordstri = []

        # Unigram transition words
        for t in allwords:
            t = t.strip(' \r\n')
            splitt = t.split(' ')

            if len(splitt) == 1:
                tempwordsuni.append(t)
            elif len(splitt) == 2:
                tempwordsbi.append(t)
            elif len(splitt) == 3:
                tempwordstri.append(t)

        # Assign it back to the class
        self.tempwordsuni = [x.lower() for x in tempwordsuni]  # converting to lower case
        self.tempwordsbi = [x.lower() for x in tempwordsbi]  # converting to lower case
        self.tempwordstri = [x.lower() for x in tempwordstri]  # converting to lower case

    def ft_foll_by_conj(self, sent, pos):
        if sent == '':
            return [0]

        unifeat = 0  # Unigram feature
        bifeat = 0  # bigram feature
        trifeat = 0  # trigram feature

        s = sent[pos:]  # sentence before the candidate position

        # remove punctuations
        punctuations = [',','//','/','?','.','!']

        for p in punctuations:
            s = s.replace(p,'')
        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']

        try:
            precuni = unigrams[0]  # unigram before the candidate position
        except IndexError:
            return [0]

        if precuni.lower() in self.conjwordsuni:
            unifeat = 1

        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        if bigrams != []:
            precbi = bigrams[0]
            if precbi in self.conjwordsbi:
                bifeat = 1

        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]

        if trigrams != []:
            prectri = trigrams[0]

            if prectri in self.conjwordstri:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_prec_by_transword(self, sent, pos):
        """
        Gives the binary features for weather the candidate position was
        preceeded by a transition word / phrase. This gives 3 features
        one for unigram, bigram, trigram
        :param sent: The un annotate sentence
        :param pos: Candidate position
        :return:
        """

        if sent == '':
            return [0]

        unifeat = 0  # Unigram feature
        bifeat = 0  # bigram feature
        trifeat = 0  # trigram feature

        #sent = sent.split(' ')
        s = sent[:pos]  # sentence before the candidate position
        #s = ' '.join(s)

        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            s = s.replace(p,'')

        print s
        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[-1]  # unigram before the candidate position
        except IndexError:
            return [0]

        if precuni.lower() in self.transwordsuni:
            unifeat = 1


        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        bigrams = [x for x in bigrams if x != '']
        if bigrams != []:
            precbi = bigrams[-1]
            if precbi.lower() in self.transwordbi:
                bifeat = 1


        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
        trigrams = [x for x in trigrams if x != '']

        if trigrams != []:
            prectri = trigrams[-1]

            if prectri.lower() in self.transwordtri:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_foll_by_transword(self, sent, pos):
        """
        Gives the binary features for weather the candidate position was
        preceeded by a transition word / phrase. This gives 3 features
        one for unigram, bigram, trigram
        :param sent: The un annotate sentence
        :param pos: Candidate position
        :return:
        """

        if sent == '':
            return [0]

        unifeat = 0  # Unigram feature
        bifeat = 0  # bigram feature
        trifeat = 0  # trigram feature

        #sent = sent.split(' ')
        s = sent[pos:]  # sentence before the candidate position
        #s = ' '.join(s)


        punctuations = [',','//','/','?','.','!']

        for p in punctuations:
            s = s.replace(p,'')

        # remove punctuations
        s = s.replace(',', '')
        s = s.replace('//','')
        s = s.replace('/','')


        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[0]  # unigram before the candidate position
        except IndexError:
            return [0]

        if precuni.lower() in self.transwordsuni:
            unifeat = 1


        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        bigrams = [x for x in bigrams if x != '']
        if bigrams != []:
            precbi = bigrams[0]
            if precbi.lower() in self.transwordbi:
                bifeat = 1


        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
        trigrams = [x for x in trigrams if x != '']

        if trigrams != []:
            prectri = trigrams[0]

            if prectri.lower() in self.transwordtri:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_set_of_all_trans(self, sent, pos):

        """

        we construct a feature vector of size k
        corresponding to the set of all transition
        words. If a transition word is present in
        the preceeding uni, bi, trigram then that
        value gets 1
        :param sent:
        :param pos:
        :return:
        """

        """
        Gives the binary features for weather the candidate position was
        preceeded by a transition word / phrase. This gives 3 features
        one for unigram, bigram, trigram
        :param sent: The un annotate sentence
        :param pos: Candidate position
        :return:
        """

        if sent == '':
            return [0]
        alltranswords = self.transwordsuni + self.transwordbi + self.transwordtri
        precuni = []
        precbi = []
        prectri = []
        #sent = sent.split(' ')
        s = sent[:pos]  # sentence before the candidate position
        #s = ' '.join(s)

        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            s = s.replace(p,'')

        print s
        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[-1]  # unigram before the candidate position
        except IndexError:
            return [0 for i in range(len(alltranswords))]

        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        bigrams = [x for x in bigrams if x != '']
        if bigrams != []:
            precbi = bigrams[-1]


        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
        trigrams = [x for x in trigrams if x != '']

        if trigrams != []:
            prectri = trigrams[-1]

        # ngrams

        ngrams = [precuni,precbi, prectri]

        alltransword_feat = []
        for i in range(len(alltranswords)):
            if alltranswords[i] in ngrams:
                alltransword_feat.append(1)
            else:
                alltransword_feat.append(0)

        return alltransword_feat

    def ft_wh_word(self, sent, pos):
        """
        Gets features for preceeded or succeded  by wh word
        :param sent:
        :param pos:
        :return:
        """


        # Get the sentences before and after the position
        #sent = sent.split(' ')
        sbfore = sent[:pos]  # sentence before the candidate position
        #sbfore = ' '.join(sbfore)
        safter = sent[pos:]
        #safter = ' '.join(safter)

        # remove pun ctuations
        #sent = sent.replace(',', '')


        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            sbfore = sbfore.replace(p,'')
            safter = safter.replace(p,'')


        # punctuations = []
        # sbfore = sbfore.replace(',','')
        # sbfore = sbfore.replace('/','')
        # sbfore = sbfore.replace('//','')
        #
        # safter = safter.replace(',','')
        # safter = safter.replace('/','')
        # safter = safter.replace('//','')

        # unigrams before candidate position
        unigramsbf = sbfore.split(' ')
        unigramsbf = [x for x in unigramsbf if x != '']

        # unigrams after candidate position
        unigramsaf = safter.split(' ')
        unigramsaf = [x for x in unigramsaf if x != '']

        # wh word before and after
        try:
            whbefore = int('wh' in unigramsbf[-1].lower())
            whafter = int('wh' in unigramsaf[0].lower())
        except IndexError:
            return [0, 0]

        return [whbefore, whafter]

    def ft_rep_words(self, sent, pos):
        feat = 0

        if sent == '':
            return [0]

        # Get the sentences before and after the position
        #sent = sent.split(' ')
        sbfore = sent[:pos]  # sentence before the candidate position
        #sbfore = ' '.join(sbfore)
        safter = sent[pos:]
        #safter = ' '.join(safter)

        # remove pun ctuations
        #sent = sent.replace(',', '')

        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            sbfore = sbfore.replace(p,'')
            safter = safter.replace(p,'')

        # unigrams before candidate position
        unigramsbf = sbfore.split(' ')
        unigramsbf = [x for x in unigramsbf if x != '']

        # unigrams after candidate position
        unigramsaf = safter.split(' ')
        unigramsaf = [x for x in unigramsaf if x != '']

        # Check preceeding and succeeding words

        try:
            if unigramsbf[-1].lower() == unigramsaf[0].lower():  # words before and after cand position are the same
                feat = 1
        except IndexError:
            return [0]

        return [feat]

    def ft_end_of_sent(self, sent, pos):
        """
        Feature that marks end of sentence
        :param sent:
        :param pos:
        :return:
        """
        

        if sent == '':
            return [0]


        #sent = sent.split(' ')
        s = sent[:pos]  # sentence before the candidate position
        feat = 0
        #s = ' '.join(s)

        punctuations = [',','//','/','"','\'']

        for p in punctuations:
            s = s.replace(p,'')

        print s
        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[-1]  # unigram before the candidate position
        except IndexError:
            return [0]

        if '.' in precuni.lower()[-1] or '?' in precuni.lower()[-1] or '!' in precuni.lower()[-1]:
            feat = 1

        else:
            feat = 0

        return [feat]

    def ft_beg_of_sent(self, sent, pos):
        """
        Feature that marks end of sentence
        :param sent:
        :param pos:
        :return:
        """
        feat = 0
        if pos == 0:
            feat =  1

        else:
            feat = 0

        return [feat]

    def ft_prec_by_ne(self, sentob, pos):


        unifeat = 0  # Unigram feature
        sent = sentob.ann_sent
        nerdict = sentob.nerdict

        if sent == '':
            return [0]

        #sent = sent.split(' ')
        s = sent[:pos]  # sentence before the candidate position
        #s = ' '.join(s)


        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            s = s.replace(p,'')

        # # remove punctuations
        # s = s.replace(',', '')
        # s = s.replace('//','')
        # s = s.replace('/','')
        # s = s.replace(',', '')

        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[-1]  # unigram before the candidate position
        except IndexError:
            return [0]

        # Get the named entities
        try:

            if nerdict[precuni] != 'o':
                return [1]
            else:
                return [0]
        except KeyError:
            print precuni
            return [0]

    def ft_foll_by_ne(self, sentob, pos):

        sent = sentob.ann_sent
        nerdict = sentob.nerdict

        if sent == '':
            return [0]

        #sent = sent.split(' ')
        s = sent[pos:]  # sentence before the candidate position
        #s = ' '.join(s)


        # remove punctuations
        punctuations = [',','//','/','?','.','!','"','\'']
        for p in punctuations:
            s.replace(p,'')
        #
        # s = s.replace(',', '')
        # s = s.replace('//','')
        # s = s.replace('/','')
        # s = s.replace(',', '')

        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            succuni = unigrams[0]  # unigram before the candidate position

        except IndexError:
            return [0]

        # Get the named entities
        try:


            if nerdict[succuni] != 'o':
                return [1]
            else:
                return [0]
        except KeyError:
            return [0]

    def ft_nndepthat(self, sentob, pos):


        sent = sentob.ann_sent
        sentpos = sentob.pos_tags
        feat = 0

        if sent == '':
            return [0]


        #sent = sent.split(' ')
        sbefore = sent[pos:]  # sentence before the candidate position
        safter = sent[:pos]
        #s = ' '.join(s)


        # remove punctuations
        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            sbefore = sbefore.replace(p,'')
            safter = safter.replace(p,'')

        # unigrams before candidate position
        unigramsbf = sbefore.split(' ')
        unigramsbf = [x for x in unigramsbf if x != '']

        # unigrams after candidate position
        unigramsaf = safter.split(' ')
        unigramsaf = [x for x in unigramsaf if x != '']

        # Check for nount that pattern

        try:
            sbeforepos = sentpos[unigramsbf[-1]]

            if 'N' in sbeforepos and unigramsaf[0].lower()  == 'that':
                feat = 1

            else:
                feat = 0
        except KeyError:
            print 'The unigram before the position of interest :' + unigramsbf[-1]
            return [0]

        except TypeError:
            print 'The unigram before the position of interest :' + unigramsbf[-1]
            return [0]

        except IndexError:
            return [0]

        return [feat]

    def ft_vbdepthat(self, sentob, pos):

        sent = sentob.ann_sent
        sentpos = sentob.pos_tags

        if sent == '':
            return [0]

        #sent = sent.split(' ')
        sbefore = sent[:pos]  # sentence before the candidate position

        # remove punctuations
        punctuations = [',','//','/','?','.','!','"','\'']

        for p in punctuations:
            sbefore = sbefore.replace(p,'')

        # unigrams before candidate position
        unigramsbf = sbefore.split(' ')
        unigramsbf = [x for x in unigramsbf if x != '']

        # Check for nount that pattern

        try:
            sbeforebfpos = sentpos[unigramsbf[-2]] # last but 2 before pos should be vn
            sbefore= unigramsbf[-1] # The word before should be 'that'

            if 'V' in sbeforebfpos and sbefore.lower() == 'that':
                return [1]

            else:
                return [0]
        except KeyError:
            print 'The unigram before the position of interest :' + unigramsbf[-1]
            return [0]

        except TypeError:
            print 'The unigram before the position of interest :' + unigramsbf[-1]
            return [0]

        except IndexError: # Could happen if the pos is in the first two positions
            #print 'The unigram before the position of interest :' + unigramsbf[-1]
            return [0]

    def ft_nounandnoun(self, sentob, pos):

        sent = sentob.ann_sent
        sentpos = sentob.pos_tags
        feat = 0

        if sent == '':
            return [0]


        #sent = sent.split(' ')
        sbefore = sent[pos:]  # sentence before the candidate position
        safter = sent[:pos]
        #s = ' '.join(s)


        # remove punctuations
        #punctuations = [',','//','/','?','.','!','"', '(', ')','\'', '[', ']', ':']
        punctuations = [',','//','/','?','.','!','"','\'']


        for p in punctuations:
            sbefore = sbefore.replace(p,'')
            safter = safter.replace(p,'')

        # unigrams before candidate position
       # unigramsbf = sbefore.split(' ')
        unigramsbf = word_tokenize(sbefore)
        unigramsbf = [x for x in unigramsbf if x != '']

        # unigrams after candidate position
        #unigramsaf = safter.split(' ')
        unigramsaf = word_tokenize(safter)
        unigramsaf = [x for x in unigramsaf if x != '']

        # checking for Noun and Noun form

        try:
            print unigramsbf
            print sentpos
            posbf = sentpos[unigramsbf[-1]]
            uniaf = unigramsaf[0]
            posaf = unigramsaf[1]

            # check

            if 'N' in posbf and 'N' in posaf and uniaf.lower() == 'and':
                feat = 1

            else:
                feat = 0

        except IndexError:
            return [0]

        except TypeError: # The unigram has for some reason more than one pos tag
            print len(unigramsbf[-1])
            print sentpos[unigramsbf[-1]]
            posbf = sentpos[unigramsbf[-1]]
            uniaf = unigramsaf[0]
            posaf = unigramsaf[1]

            # check

            if 'N' in posbf and 'N' in posaf and uniaf.lower() == 'and':
                return [1]

            else:
                return [0]
        return [feat]

    def ft_prec_by_punc(self, sent, pos):
		        # Get the sentences before and after the position
        #sent = sent.split(' ')
        sbfore = sent[:pos]  # sentence before the candidate position
        feat = 0

        # remove class labels

        punctoremove = ['/','//']
        for p in punctoremove:
            sbfore = sbfore.replace(p,'')
        punctuations = [',','?','!', ':','\'','\"','-','(',')',';']
        # unigrams before candidate position
        unigramsbf = sbfore.split(' ')
        unigramsbf = [x for x in unigramsbf if x != '']
        featlist = []

        try:
            for p in punctuations:
                if p in unigramsbf[-1]:
                    featlist.append(1)
                    feat = 1
                else:
                    featlist.append(0)
                    continue
        except IndexError:
            return [0 for i in range(len(punctuations))]

        return featlist

    def ft_foll_by_punc(self, sent, pos):
		        # Get the sentences before and after the position
        #sent = sent.split(' ')
        safter = sent[pos:]  # sentence before the candidate position
        feat = 0

        # remove class labels

        punctoremove = ['/','//']
        for p in punctoremove:
            sbfore = safter.replace(p,'')
        punctuations = [',','?','!', ':','\'','\"','-','(',')',';']
        # unigrams before candidate position
        unigramsaf = safter.split(' ')
        unigramsaf = [x for x in unigramsaf if x != '']

        featlist = []

        try:
            for p in punctuations:
                if p in unigramsaf[0]:
                    featlist.append(1)
                    feat = 1
                else:
                    featlist.append(0)
                    continue
        except IndexError:
            return [0 for i in range(len(punctuations))]

        return featlist


    def ft_foll_by_exwords(self, sent, pos):

        if sent == '':
            return [0]

        unifeat = 0  # Unigram feature
        bifeat = 0  # bigram feature
        trifeat = 0  # trigram feature

        #sent = sent.split(' ')
        s = sent[pos:]  # sentence before the candidate position
        #s = ' '.join(s)


        punctuations = [',','//','/','?','.','!']

        for p in punctuations:
            s = s.replace(p,'')

        # remove punctuations
        s = s.replace(',', '')
        s = s.replace('//','')
        s = s.replace('/','')


        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[0]  # unigram before the candidate position
        except IndexError:
            return [0]

        if precuni.lower() in self.exwords:
            unifeat = 1


        # bigrams after candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        bigrams = [x for x in bigrams if x != '']
        if bigrams != []:
            precbi = bigrams[0]
            if precbi.lower() in self.exwords:
                bifeat = 1


        # trigrams after candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
        trigrams = [x for x in trigrams if x != '']

        if trigrams != []:
            prectri = trigrams[0]

            if prectri.lower() in self.exwords:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_prec_by_exwords(self, sent, pos):

        if sent == '':
            return [0]

        unifeat = 0  # Unigram feature
        bifeat = 0  # bigram feature
        trifeat = 0  # trigram feature

        #sent = sent.split(' ')
        s = sent[:pos]  # sentence before the candidate position
        #s = ' '.join(s)


        punctuations = [',','//','/','?','.','!']

        for p in punctuations:
            s = s.replace(p,'')

        # remove punctuations
        s = s.replace(',', '')
        s = s.replace('//','')
        s = s.replace('/','')


        try:
            # unigrams before candidate position
            unigrams = s.split(' ')
            unigrams = [x for x in unigrams if x != '']
            try:
                precuni = unigrams[-1]  # unigram before the candidate position
            except IndexError:
                return [0]

            if precuni.lower() in self.exwords:
                unifeat = 1


            # bigrams after candidate position
            tokens = nltk.word_tokenize(s)
            bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
            bigrams = [x for x in bigrams if x != '']
            if bigrams != []:
                precbi = bigrams[-1]
                if precbi.lower() in self.exwords:
                    bifeat = 1


            # trigrams after candidate position
            trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
            trigrams = [x for x in trigrams if x != '']

            if trigrams != []:
                prectri = trigrams[-1]

                if prectri.lower() in self.exwords:
                    trifeat = 1

            return [int(any([unifeat, bifeat, trifeat]))]
        except IndexError:
            return [0]

    def ft_temp_connec_in_sent(self,sent,pos):
        """
        There is likely a pause if there is a temporal
        connective in the sentence after the pause
        :param sent: Entire sentence under consideration
        :param pos: current marker position
        :return:
        """

        # Initialization
        unifeat = 0
        bifeat = 0
        trifeat = 0

        if sent == '':
            return [0]

        s = sent[pos:]  # sentence before the candidate position
        punctuations = [',','//','/','?','.','!']

        for p in punctuations: # remove all punctuations
            s = s.replace(p,'')

        # Go through the list of temporal connectives and check if any of the connectives
        # is present in the words that follow
        # unigrams before candidate position
        unigrams = s.split(' ')
        unigrams = [x for x in unigrams if x != '']
        try:
            precuni = unigrams[0]  # unigram before the candidate position
        except IndexError:
            return [0]

        if precuni.lower() in self.tempwordsuni:
            unifeat = 1


        # bigrams after candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [" ".join(pair) for pair in nltk.bigrams(tokens)]
        bigrams = [x for x in bigrams if x != '']
        if bigrams != []:
            precbi = bigrams[0]
            if precbi.lower() in self.tempwordsbi:
                bifeat = 1


        # trigrams after candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]
        trigrams = [x for x in trigrams if x != '']

        if trigrams != []:
            prectri = trigrams[0]

            if prectri.lower() in self.tempwordstri:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_junction_of_dep_facts(self, ss, pos):

        """
        when there is two facts in the sentence dependent on
        each other……generally such sentence starts with WH-word
        and composed of two clauses

        We look for the following
        1. 'When' as the start of the sentence
        2. look for the nsub dependency immediately after the position
        3. nsub dependency before the position

        :param ss: sentence object
        :param pos:
        :return:
        """

        sent = ss.orig_sent
        feat = 0
        sbfore = sent[:pos]  # sentence before the candidate position
        safter = sent[pos:]

        # remove all the punctuations
        punctuations = [',', '//', '/', '?', '.', '!', '"', '\'']
        for p in punctuations:
            sbfore = sbfore.replace(p, '')
            safter = safter.replace(p, '')

        sbfore = nltk.word_tokenize(sbfore)
        safter = nltk.word_tokenize(safter)

        # Check if the starting of the sentence is 'When'

        try:

            if sbfore[0].strip(' ').lower() != 'when':
                return [0]

            # Extract the 'nsubj' dependencies
            dependencies = ss.deps # Dependencies for the given sentence
            if dependencies == []:
                return [0]
            nsubjdep = [ x for x in dependencies if 'nsubj' in x[0]]

            # Check if the word next to the marker is part of the nsubj dependency
            wordaftermarker = safter[0]
            wordbeforemarker = sbfore[-1]
            tempflag = 0
            tempflat2 = 0
            for d in nsubjdep:
                # typcially the word after marker is noun or pronoun and appears in 2 pos in
                # the dependencies
                if d[2] == wordaftermarker:
                    tempflag = 1

                if d[1] == wordbeforemarker:
                    tempflat2 = 1

            feat = tempflag & tempflat2 # only if both the dependencies are present, the flag is set

            return [feat]
        except IndexError:
            return [0]


        # check if the word before the marker is part of the nsubj dependency

    def ft_foll_by_sent(self, ss, pos):

        """
        Check if the string after the marker position is a
        sentence in itself
        :param ss: Sentence object
        :param pos: potential marker position
        :return:
        """

        feat = 0
        sent = ss.orig_sent
        safter = sent[pos:]

        # remove all the punctuations
        punctuations = [',', '//', '/', '?', '.', '!', '"', '\'']
        for p in punctuations:
            safter = safter.replace(p, '')

        safter = nltk.word_tokenize(safter)

        # Check if the starting of the sentence is 'When'

        try:
            # Extract the 'nsubj' dependencies
            dependencies = ss.deps  # Dependencies for the given sentence
            if dependencies == []:
                return [0]
            if dependencies == None:
                return [0]
            nsubjdep = [x for x in dependencies if 'nsubj' in x[0]]

            # Check if the word next to the marker is part of the nsubj dependency
            wordaftermarker = safter[0]
            for d in nsubjdep:
                # typcially the word after marker is noun or pronoun and appears in 2 pos in
                # the dependencies
                if d[2] == wordaftermarker:
                    feat = 1

            return [feat]
        except IndexError:
            return [0]

    def ft_junction_of_inf_stmts(self, ss, pos):

        """
        We need to see an if word in the string before the marker
        and the string after the marker should be a sentence in its own
        :param ss: sentence object
        :param pos: marker position
        :return:
        """

        sent = ss.orig_sent
        feat = 0
        sbfore = sent[:pos]  # sentence before the candidate position
        safter = sent[pos:]

        # remove all the punctuations
        punctuations = [',', '//', '/', '?', '.', '!', '"', '\'']
        for p in punctuations:
            sbfore = sbfore.replace(p, '')
            safter = safter.replace(p, '')

        sbfore = nltk.word_tokenize(sbfore)
        safter = nltk.word_tokenize(safter)

        # Check if the starting of the sentence is 'When'
        try:

            if 'if' not in sbfore:
                return [0]

            # Extract the 'nsubj' dependencies
            dependencies = ss.deps  # Dependencies for the given sentence
            if dependencies == []:
                return [0]

            if dependencies == None:
                return [0]
            nsubjdep = [x for x in dependencies if 'nsubj' in x[0]]

            # Check if the word next to the marker is part of the nsubj dependency
            wordaftermarker = safter[0]
            for d in nsubjdep:
                # typcially the word after marker is noun or pronoun and appears in 2 pos in
                # the dependencies
                if d[2] == wordaftermarker:
                    feat = 1

            return [feat]
        except IndexError:
            return [0]

    def compile_feature_vector(self):

        # Run through all the sentences
        all_features = OrderedDict()
        all_featurestest = OrderedDict()
        all_featuresunlabelled = OrderedDict()

        #Compiling features for the training sentences
        #for ss in self.sentences:
        for k in range(len(self.sentences)):
            ss = self.sentences[k]

            s = ss.ann_sent
            candidatepos = ss.candidatepos
            # remove punctuations - handle it inside the function
            # s = s.replace(',', '')

            # for every position run grab the features
            features_for_sent = []

            # Find the features for every candidate position
            for p in range(len(candidatepos)):
                i = candidatepos[p]
                features_for_pos = []

                # Run it through all the functions that
                # compute featurs before the candidate position

                # Word level features 0 - 21 indices
                features_for_pos += self.ft_beg_of_sent(s,i)
                features_for_pos += self.ft_end_of_sent(s,i)
                features_for_pos += self.ft_prec_by_punc(s,i)
                features_for_pos += self.ft_foll_by_punc(s,i)

                # Grammatical features 22 - 27
                features_for_pos += self.ft_nndepthat(ss,i)
                features_for_pos += self.ft_vbdepthat(ss,i)
                features_for_pos += self.ft_nounandnoun(ss,i)
                features_for_pos += self.ft_junction_of_dep_facts(ss, i)
                features_for_pos += self.ft_foll_by_sent(ss, i)
                features_for_pos += self.ft_junction_of_inf_stmts(ss, i)

                # Linguistic features 27 - 297

                features_for_pos += self.ft_prec_by_transword(s, i)
                features_for_pos += self.ft_wh_word(s, i)
                features_for_pos += self.ft_foll_by_transword(s, i)
                features_for_pos += self.ft_rep_words(s, i)
                features_for_pos += self.ft_foll_by_conj(s,i)
                features_for_pos += self.ft_foll_by_exwords(s,i)
                features_for_pos += self.ft_prec_by_exwords(s,i)
                features_for_pos += self.ft_set_of_all_trans(s,i)
                features_for_pos += self.ft_temp_connec_in_sent(s,i)

                #features_for_pos += self.ft_prec_by_ne(ss,i)
                #features_for_pos += self.ft_foll_by_ne(ss,i)
                features_for_sent.append(features_for_pos)
            all_features[ss] = features_for_sent

        # Compiling features for the testing sequences sentences
        #for ss in self.sentencestest:
        for k in range(len(self.sentencestest)):
            ss = self.sentencestest[k]

            s = ss.orig_sent
            candidatepos = ss.candidatepos
            print 'Candidate positions for test'
            print candidatepos

            # remove punctuations
            s = s.replace(',', '')

            # split and merge to get candidate positions
            words = s.split(' ')

            # for every position run grab the features

            features_for_sent = []


            # Find the features for every candidate position


            for p in range(len(candidatepos)):
                i = candidatepos[p]
                features_for_pos = []

                # Run it through all the functions that
                # compute featurs before the candidate position
                # Word level features 0 - 21 indices
                features_for_pos += self.ft_beg_of_sent(s,i)
                features_for_pos += self.ft_end_of_sent(s,i)
                features_for_pos += self.ft_prec_by_punc(s,i)
                features_for_pos += self.ft_foll_by_punc(s,i)

                # Grammatical features 22 - 27
                features_for_pos += self.ft_nndepthat(ss,i)
                features_for_pos += self.ft_vbdepthat(ss,i)
                features_for_pos += self.ft_nounandnoun(ss,i)
                features_for_pos += self.ft_junction_of_dep_facts(ss, i)
                features_for_pos += self.ft_foll_by_sent(ss, i)
                features_for_pos += self.ft_junction_of_inf_stmts(ss, i)

                # Linguistic features 27 - 297

                features_for_pos += self.ft_prec_by_transword(s, i)
                features_for_pos += self.ft_wh_word(s, i)
                features_for_pos += self.ft_foll_by_transword(s, i)
                features_for_pos += self.ft_rep_words(s, i)
                features_for_pos += self.ft_foll_by_conj(s,i)
                features_for_pos += self.ft_foll_by_exwords(s,i)
                features_for_pos += self.ft_prec_by_exwords(s,i)
                features_for_pos += self.ft_set_of_all_trans(s,i)
                features_for_pos += self.ft_temp_connec_in_sent(s,i)

                #features_for_pos += self.ft_prec_by_ne(ss,i)
                #features_for_pos += self.ft_foll_by_ne(ss,i)
                features_for_sent.append(features_for_pos)
            all_featurestest[ss] = features_for_sent

        # Compiling features for the unlabelled sequences sentences
        #for ss in self.sentencestest:

        if self.config.get('LU','lu').strip(' ') == '1':
            for k in range(len(self.sentencesunlabelled)):
                ss = self.sentencesunlabelled[k]

                s = ss.orig_sent
                candidatepos = ss.candidatepos
                print 'Candidate positions for unlabelled'
                print candidatepos

                # remove punctuations
                s = s.replace(',', '')

                # split and merge to get candidate positions
                words = s.split(' ')

                # for every position run grab the features

                features_for_sent = []


                # Find the features for every candidate position


                for p in range(len(candidatepos)):
                    i = candidatepos[p]
                    features_for_pos = []

                    # Run it through all the functions that
                    # compute featurs before the candidate position

                    # Word level features 0 - 21 indices
                    features_for_pos += self.ft_beg_of_sent(s,i)
                    features_for_pos += self.ft_end_of_sent(s,i)
                    features_for_pos += self.ft_prec_by_punc(s,i)
                    features_for_pos += self.ft_foll_by_punc(s,i)

                    # Grammatical features 22 - 27
                    features_for_pos += self.ft_nndepthat(ss,i)
                    features_for_pos += self.ft_vbdepthat(ss,i)
                    features_for_pos += self.ft_nounandnoun(ss,i)
                    features_for_pos += self.ft_junction_of_dep_facts(ss, i)
                    features_for_pos += self.ft_foll_by_sent(ss, i)
                    features_for_pos += self.ft_junction_of_inf_stmts(ss, i)

                    # Linguistic features 27 - 297

                    features_for_pos += self.ft_prec_by_transword(s, i)
                    features_for_pos += self.ft_wh_word(s, i)
                    features_for_pos += self.ft_foll_by_transword(s, i)
                    features_for_pos += self.ft_rep_words(s, i)
                    features_for_pos += self.ft_foll_by_conj(s,i)
                    features_for_pos += self.ft_foll_by_exwords(s,i)
                    features_for_pos += self.ft_prec_by_exwords(s,i)
                    features_for_pos += self.ft_set_of_all_trans(s,i)
                    features_for_pos += self.ft_temp_connec_in_sent(s,i)

                    #features_for_pos += self.ft_prec_by_ne(ss,i)
                    #features_for_pos += self.ft_foll_by_ne(ss,i)
                    features_for_sent.append(features_for_pos)
                all_featuresunlabelled[ss] = features_for_sent
                # for i in range(len(words)):
                #     features_for_pos = []
                #
                #     # Run it through all the functions that
                #     # compute featurs before the candidate position
                #
                #     features_for_pos += self.ft_prec_by_transword(s, i)
                #     features_for_pos += self.ft_wh_word(s, i)
                #     features_for_pos += self.ft_foll_by_transword(s,i)
                #     features_for_pos += self.ft_rep_words(s, i)
                #
                #     features_for_sent.append(features_for_pos)
                # all_features[ss] = features_for_sent

        # TODO length of features for every pos is diff, fix that

        self.features = all_features
        self.featurestest = all_featurestest
        self.featuresunlabelled = all_featuresunlabelled

        # print 'FEatures for training set'
        # print all_features
        #
        # print 'All features for test'
        # print all_featurestest


if __name__ == '__main__':
    ef = ExtractFeatures()
    ef.compile_feature_vector()

    feat_train = ef.features
    feat_test = ef.featurestest
    feat_unlabelled = ef.featuresunlabelled
    print ' /n /n '
    print '======================'
    print 'Printing feature stats fot train'
    keys = feat_train.keys()
    for i in range(len(keys)):
        print keys[i].orig_sent
        print len(feat_train[keys[i]])
    print '======================'
    print 'Printing feature stats fot test'
    keys = feat_test.keys()
    for i in range(len(keys)):
        print keys[i].orig_sent
        print feat_test[keys[i]]
    print '======================'
    print 'Printing feature stats fot unlabelled'
    keys = feat_unlabelled.keys()
    for i in range(len(keys)):
        print keys[i].orig_sent
        print feat_unlabelled[keys[i]]