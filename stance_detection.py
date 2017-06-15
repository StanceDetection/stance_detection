# -*- coding: utf-8 -*-

# import training data
# build features
# train naive bayes' classifer on features

# features: Jaccard similarity.
#   for sentence in body:
#       compute and store Jaccard similarity between body and sentence
#   compute average and maximum Jaccard similarity for the body
# The features passed to the classifier will be:
# ● the number of bigram and trigram
# repetitions between the article and
# headline, normalized by the length of
# the article or headline (whichever is
# longest).
# ● The average and maximum Jaccard
# similarities between the headline and
# each sentence in the article body (two
# numbers).

# TODO: add stemming, lowercase everything?, replace bad characters

import pdb
import string
from csv import DictReader
import nltk
nltk.download('punkt')
from sklearn.metrics import jaccard_similarity_score


class StanceDetectionClassifier:
    REMOVE_PUNC_MAP = dict((ord(char), None) for char in string.punctuation)

    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []

    def gen_training_features(self, bodies_fpath, stances_fpath):
        self._train_bodies, self._train_stances = self._read(bodies_fpath, stances_fpath, True)

        self._train_unigrams = self._train_ngrams(1, self._train_bodies, self._train_stances)

        self.train_avg_sims, self.train_max_sims = self._gen_jaccard_sims(self._train_bodies, self._train_stances)

        for i in range(len(self._train_stances)):
            labeled_feature = ({
                'unigrams':self._train_unigrams[i],
                'avg_sims':self.train_avg_sims[i],
                'max_sims':self.train_max_sims[i]},
                self._train_stances[i]['Stance'])
            self._labeled_feature_set.append(labeled_feature)

    def gen_testing_features(self, bodies_fpath, stances_fpath):
        self._test_bodies, self._test_stances = self._read(bodies_fpath, stances_fpath, False)

        self._test_unigrams = self._train_ngrams(1, self._test_bodies, self._test_stances)

        self.test_avg_sims, self.test_max_sims = self._gen_jaccard_sims(self._test_bodies, self._test_stances)

        for i in range(len(self._test_stances)):
            feature = ({
                'unigrams':self._test_unigrams[i],
                'avg_sims':self.test_avg_sims[i],
                'max_sims':self.test_max_sims[i]})
            self._test_feature_set.append(feature)

    def _gen_jaccard_sims(self, bodies, stances):
        # currently assumes both body and headline are longer than 0.
        punc_rem_tokenizer = nltk.RegexpTokenizer(r'\w+')

        avg_sims = []
        max_sims = []

        for st in stances:
            body = bodies[st['Body ID']]
            headline = st['Headline']
            headline = headline.translate(self.REMOVE_PUNC_MAP)
            headline = nltk.word_tokenize(headline)
            sents = nltk.sent_tokenize(body)
            sents = self._remove_punctuation(sents)
            sents = self._word_tokenize(sents)
            num_sents = len(sents)
            jacc_sims = []
            for sent in sents:
                if len(sent) < 1:
                    continue
                # extend shorter word list so that both are the same length
                len_diff = len(headline) - len(sent)
                headline_cpy = headline
                sent_cpy = sent

                if len_diff < 0: # sent longer than headline
                    headline_cpy = headline_cpy + ([headline_cpy[-1]] * abs(len_diff))
                elif len_diff > 0: # headline longer than sent
                    sent_cpy = sent_cpy + ([sent_cpy[-1]] * abs(len_diff))

                jacc_sims.append(jaccard_similarity_score(headline_cpy, sent_cpy))
            avg_sim = self._threshold_parser((sum(jacc_sims) / len(jacc_sims)), [0.2])
            max_sim = self._threshold_parser(max(jacc_sims), [0.2])
            avg_sims.append(avg_sim)
            max_sims.append(max_sim)
        return avg_sims, max_sims

    def _threshold_parser(self, val, threshold_ranges):
        threshold_ranges.sort()
        numbuckets = len(threshold_ranges)
        counter = 0
        while counter < numbuckets:
            if(val < threshold_ranges[counter]):
                return counter
            counter = counter + 1
        return numbuckets

    def _word_tokenize(self, str_list):
        return map(lambda s: nltk.word_tokenize(s), str_list)

    def _remove_punctuation(self, str_list):
        return map(lambda s: s.translate(self.REMOVE_PUNC_MAP), str_list)

    def _read(self, bodies_fpath, stances_fpath, is_training):
        with open(bodies_fpath, 'r') as f:
            r = DictReader(f)
            bodies = {}
            for line in r:
                body = line['articleBody'].decode('utf-8')
                bodies[int(line['Body ID'])] = body

        with open(stances_fpath, 'r') as f:
            r = DictReader(f)
            stances = []
            for line in r:
                headline = line['Headline'].decode('utf-8')
                body_id = int(line['Body ID'])
                if(is_training):
                    stance = line['Stance'].decode('utf-8')
                    stances.append({
                            'Headline': headline,
                            'Body ID': body_id,
                            'Stance': stance})
                else:
                    stances.append({
                            'Headline': headline,
                            'Body ID': body_id})

        return bodies, stances

    def _get_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        tokens = [ token.lower() for token in tokens if len(token) > 1 ]
        return nltk.ngrams(tokens, n)

    def _train_ngrams(self, n, bodies, stances):
        stance_similarities = []
        body_ngrams = {}

        for bodyId in bodies:
            body_ngrams[bodyId] = self._get_ngrams(bodies[bodyId], n)

        for stance in stances:
            stance_ngrams = self._get_ngrams(stance['Headline'], n)
            num_ngrams_common = 0
            for ngram in stance_ngrams:
                if ngram in body_ngrams[stance['Body ID']]:
                    num_ngrams_common += 1
            stance_similarities.append(num_ngrams_common)

        # normalize the counts based on length of the article
        for i in range(len(stance_similarities)):
            body_id = stances[i]['Body ID']
            stance_similarities[i] = self._threshold_parser((float(stance_similarities[i])/len(bodies[body_id])), [0.2])

        return stance_similarities

    def train(self):
        self._nbc = nltk.classify.NaiveBayesClassifier.train(self._labeled_feature_set)
        return

    def predict(self):
        result = self._nbc.classify_many(self._test_feature_set)
        print(result)
        return

cls = StanceDetectionClassifier()
cls.gen_training_features('training_data/train_bodies.csv',
        'training_data/train_stances.csv')
cls.train()

cls.gen_testing_features('testing_data/test_bodies.csv',
        'testing_data/test_stances_unlabeled.csv')
cls.predict()
