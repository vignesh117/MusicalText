__author__ = 'vignesh'
import ConfigParser as cp

import pickle
import nltk
class ExtractFeatures(object):
    # Parsing configuration files
    config = cp.RawConfigParser()
    config.read('config.cfg')

    # Other declarations
    sentences = None
    features = None # Dictionary of features for all sentences
    transwordsuni = []
    transwordbi = []
    transwordtri = []
    traindata = None # Dictionary of traindata Containing X and Y
    senttype = 'train'


    def __init__(self):
        self.sentences = pickle.load(open('Sentences.pickle'))
        self.sentencestest = pickle.load(open('SentencesTest.pickle'))
        self.get_transition_words()
        self.compile_feature_vector()

        # Serialize feature
        pickle.dump(self.features, open('features.pickle','w'))
        pickle.dump(self.featurestest, open('featuresTest.pickle','w'))

    def get_transition_words(self):
        transfile = self.config.get('init','transwordsfile')
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
        self.transwordsuni = [x.lower() for x in transwordsuni] # converting to lower case
        self.transwordbi = [x.lower() for x in transwordsbi] # converting to lower case
        self.transwordtri = [x.lower() for x in transwordstri] # converting to lower case

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


        unifeat = 0 # Unigram feature
        bifeat = 0 # bigram feature
        trifeat = 0 # trigram feature

        sent = sent.split(' ')
        s = sent[:pos] # sentence before the candidate position
        s = ' '.join(s)

        # remove punctuations
        s = s.replace(',','')

         # unigrams before candidate position
        unigrams = s.split(' ')
        precuni = unigrams[-1] # unigram before the candidate position

        if precuni.lower() in self.transwordsuni:
            unifeat = 1


        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [ " ".join(pair) for pair in nltk.bigrams(tokens)]
        if bigrams != []:
            precbi = bigrams[-1]
            if precbi in self.transwordbi :
                bifeat = 1


        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]

        if trigrams != []:
            prectri = trigrams[-1]

            if prectri in self.transwordtri:
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

        unifeat = 0 # Unigram feature
        bifeat = 0 # bigram feature
        trifeat = 0 # trigram feature

        sent = sent.split(' ')
        s = sent[pos:] # sentence before the candidate position
        s = ' '.join(s)

        # remove punctuations
        s = s.replace(',','')

         # unigrams before candidate position
        unigrams = s.split(' ')
        precuni = unigrams[0] # unigram before the candidate position

        if precuni.lower() in self.transwordsuni:
            unifeat = 1


        # bigrams before candidate position
        tokens = nltk.word_tokenize(s)
        bigrams = [ " ".join(pair) for pair in nltk.bigrams(tokens)]
        if bigrams != []:
            precbi = bigrams[0]
            if precbi in self.transwordbi :
                bifeat = 1


        # trigrams before candidate position
        trigrams = [" ".join(tri) for tri in nltk.trigrams(tokens)]

        if trigrams != []:
            prectri = trigrams[0]

            if prectri in self.transwordtri:
                trifeat = 1

        return [int(any([unifeat, bifeat, trifeat]))]

    def ft_wh_word(self, sent, pos):
        """
        Gets features for preceeded or succeded  by wh word
        :param sent:
        :param pos:
        :return:
        """

        # remove pun ctuations
        sent = sent.replace(',','')

        # Get the sentences before and after the position
        sent = sent.split(' ')
        sbfore = sent[:pos] # sentence before the candidate position
        sbfore = ' '.join(sbfore)
        safter = sent[pos:]
        safter= ' '.join(safter)

         # unigrams before candidate position
        unigramsbf = sbfore.split(' ')

        # unigrams after candidate position
        unigramsaf = safter.split(' ')

        # wh word before and after
        whbefore = int('wh' in unigramsbf[-1].lower())
        whafter = int('wh' in unigramsaf[0].lower())

        return [ whbefore, whafter]

    def ft_rep_words(self, sent, pos):
        feat = 0

        if sent == '':
            return [0]
         # remove pun ctuations
        sent = sent.replace(',','')

        # Get the sentences before and after the position
        sent = sent.split(' ')
        sbfore = sent[:pos] # sentence before the candidate position
        sbfore = ' '.join(sbfore)
        safter = sent[pos:]
        safter= ' '.join(safter)

         # unigrams before candidate position
        unigramsbf = sbfore.split(' ')

        # unigrams after candidate position
        unigramsaf = safter.split(' ')

        # Check preceeding and succeeding words

        if unigramsbf[-1] == unigramsaf[0]: # words before and after cand position are the same
            feat = 1

        return [feat]

    def compile_feature_vector(self):

        # Run through all the sentences

        all_features = {}

        # Compiling features for the training sentences
        for ss in self.sentences:

            s = ss.orig_sent
            candidatepos = ss.candidatepos
            print candidatepos

            # remove punctuations
            s = s.replace(',','')

            # split and merge to get candidate positions
            words = s.split(' ')

            # for every position run grab the features

            features_for_sent = []


            # Find the features for every candidate position

            for i in candidatepos:
                features_for_pos = []

                # Run it through all the functions that
                # compute featurs before the candidate position

                features_for_pos += self.ft_prec_by_transword(s, i)
                features_for_pos += self.ft_wh_word(s, i)
                features_for_pos += self.ft_foll_by_transword(s,i)
                features_for_pos += self.ft_rep_words(s, i)

                features_for_sent.append(features_for_pos)
            all_features[ss] = features_for_sent


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
        print all_features



if __name__ == '__main__':

    ef = ExtractFeatures()
    #ef.compile__feature_vector()

    # # testing prec by transition word features
    # sent = 'Bill earned an A on his essay, but Susan got a B.'
    # sent2 = 'James is not feeling well. Therefore, he will not be here today.'
    # senttri = 'I forgot that the cake was in the oven. As a consequence, it burned.'
    # print ef.ft_prec_by_transword(sent,8)
    # print ef.ft_prec_by_transword(sent2, 6)
    # print ef.ft_prec_by_transword(senttri,12)
    #
    # # testing followed by trans features
    #
    # print ef.ft_foll_by_transword(sent,7)
    # print ef.ft_foll_by_transword(sent2, 5)
    #
    # # testing wh features
    #
    # s = 'He was really rich while he was in the Alaska'
    # print ef.ft_wh_word(s,5)



