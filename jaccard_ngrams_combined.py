# -*- coding: utf-8 -*-

# TODO: add stemming, lowercase everything?, replace bad characters,
#       remove stop words

from csv import DictReader
import pdb
import string
import sys
import time

import nltk
from nltk.classify import NaiveBayesClassifier

from libs.dataset import DataSet
from libs.gen_ngrams import NgramsGenerator
from libs.generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from libs.score import score_submission


class JaccardClassify:
    REMOVE_PUNC_MAP = dict((ord(char), None) for char in string.punctuation)

    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []
        self.dataset = DataSet()
        self._ngram_len = 2

    def do_validation(self):
        # each fold is a list of body ids.
        folds, hold_out = kfold_split(self.dataset, n_folds=10)
        #  fold_stances is a dict. keys are fold number (e.g. 0-9). hold_out_stances is list
        fold_stances, hold_out_stances = get_stances_for_folds(self.dataset, folds, hold_out)

        labeled_feat_dict = {}

        print "Generating features for each fold"
        for fold_id in fold_stances:
            print "Generating features for fold ", fold_id
            bodies = folds[fold_id]
            stances = fold_stances[fold_id]

            fold_avg_sims, fold_max_sims = self._gen_jaccard_sims(
                    bodies, stances)
            common_ngrams = NgramsGenerator().gen_common_ngrams(
                    self.dataset, bodies, stances, self._ngram_len)

            labeled_feature_set = []
            for i in range(len(stances)):
                labeled_feature = ({
                    'avg_sims':fold_avg_sims[i],
                    'max_sims':fold_max_sims[i],
                    'common_ngrams':common_ngrams[i]},
                    self._process_stance(stances[i]['Stance']))
                labeled_feature_set.append(labeled_feature)

            labeled_feat_dict[fold_id] = labeled_feature_set

        print "Generating features for hold out fold"
        holdout_avg_sims, holdout_max_sims = self._gen_jaccard_sims(
                hold_out, hold_out_stances)
        holdout_common_ngrams = NgramsGenerator().gen_common_ngrams(
                self.dataset, hold_out, hold_out_stances, self._ngram_len)

        h_unlabeled_features = []
        h_labels = []
        for i in range(len(hold_out_stances)):
            unlabeled_feature = {
                'avg_sims': holdout_avg_sims[i],
                'max_sims': holdout_max_sims[i],
                'common_ngrams':holdout_common_ngrams[i]}
            label = self._process_stance(hold_out_stances[i]['Stance'])

            h_unlabeled_features.append(unlabeled_feature)
            h_labels.append(label)

        fold_accuracy = {}
        best_fold_accuracy = 0.0
        classifiers = []

        print "Validating using each fold as testing set"
        for fold_id in fold_stances:
            fold_ids = list(range(len(folds)))
            del fold_ids[fold_id] # deleted fold is test set for this run

            training_set = [feat for fid in fold_ids for feat in labeled_feat_dict[fid]]

            testing_set = []
            testing_labels = []

            for feat, label in labeled_feat_dict[fold_id]:
                testing_set.append(feat)
                testing_labels.append(label)

            classifier = NaiveBayesClassifier.train(labeled_feature_set)
            classifiers.append(classifier)
            pred = classifier.classify_many(testing_set)

            accuracy = self._score(pred, testing_labels)
            print "Fold ", fold_id, "accuracy: ", accuracy
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                best_fold_cls = classifier

        h_res = best_fold_cls.classify_many(h_unlabeled_features)
        print 'holdout score:', self._score(h_res, h_labels)


    def _score(self, predicted, actual):
        num_correct = 0
        for idx in range(len(predicted)):
            if predicted[idx] == actual[idx]:
                num_correct += 1
        accuracy = num_correct / float(len(predicted))
        return accuracy


    def _process_stance(self, stance):
        return 'unrelated' if stance == 'unrelated' else 'related'


    def _get_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        tokens = [ token.lower() for token in tokens if len(token) > 1 ]
        ngram_list = list(nltk.ngrams(tokens, n))
        return ngram_list


    def _gen_common_ngrams(self, body_ids, stances, n):
        common_ngrams = []
        body_ngrams_dict = {}

        for body_id in body_ids:
            body_ngrams_dict[body_id] = self._get_ngrams(self.dataset.articles[body_id], n)

        for stance in stances:
            stance_ngrams = self._get_ngrams(stance['Headline'], n)

            num_ngrams_common = 0
            for ngram in stance_ngrams:
                if ngram in body_ngrams_dict[stance['Body ID']]:
                    num_ngrams_common += 1
            common_ngrams.append(num_ngrams_common)

        return common_ngrams


    def _gen_jaccard_sims(self, body_ids, stances):
        # currently assumes both body and headline are longer than 0.
        punc_rem_tokenizer = nltk.RegexpTokenizer(r'\w+')

        avg_sims = []
        max_sims = []

        parsed_bodies_dict = {}
        # for body_id, body in self.dataset.articles.iteritems():
        for body_id in body_ids:
            body = self.dataset.articles[body_id].lower()
            sents = nltk.sent_tokenize(body)
            sents = self._remove_punctuation(sents)
            sents = self._word_tokenize(sents)
            parsed_bodies_dict[body_id] = sents # cache parsed body

        for st in stances:
            headline = st['Headline'].lower()
            headline = headline.translate(self.REMOVE_PUNC_MAP)
            headline = nltk.word_tokenize(headline)
            body_id = st['Body ID']
            sents = parsed_bodies_dict[body_id]

            jacc_sims = []
            for sent in sents:
                if len(sent) < 1:
                    continue
                hs = set(headline)
                ss = set(sent)
                jacc_sim = len(hs.intersection(ss)) / float(len(hs.union(ss)))
                jacc_sims.append(jacc_sim)

            max_sim = max(jacc_sims)
            avg_sim = sum(jacc_sims) / float(len(jacc_sims))

            max_sims.append(max_sim)
            avg_sims.append(avg_sim)

        return avg_sims, max_sims


    def _word_tokenize(self, str_list):
        return map(lambda s: nltk.word_tokenize(s), str_list)


    def _remove_punctuation(self, str_list):
        return map(lambda s: s.translate(self.REMOVE_PUNC_MAP), str_list)

if __name__ == "__main__":
    JaccardClassify().do_validation()

