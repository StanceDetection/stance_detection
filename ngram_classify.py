# -*- coding: utf-8 -*-

# import training data

# TODO: add stemming, replace bad characters, remove punction
# Idea: what if we take 0 as being unknown

from csv import DictReader
import pdb
import string
import sys

import nltk
from nltk.classify import NaiveBayesClassifier

from libs.dataset import DataSet
from libs.gen_ngrams import NgramsGenerator
from libs.generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from libs.score import score_submission

class NgramClassify:

    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []
        self.dataset = DataSet()
        self._ngram_len = 3

    def do_validation(self):
        folds, hold_out = kfold_split(self.dataset, n_folds=10)
        fold_stances, hold_out_stances = get_stances_for_folds(self.dataset, folds, hold_out)

        labeled_feat_dict = {}

        print "Generating features for each fold"
        for fold_id in fold_stances:
            print "Generating features for fold ", fold_id
            bodies = folds[fold_id]
            stances = fold_stances[fold_id]

            # split into 50/50 unrelated-related
            related = []
            unrelated = []
            for stance in stances:
                if stance['Stance'] in ['discuss', 'agrees', 'disagrees']:
                    related.append(stance);
                else:
                    unrelated.append(stance);

            unrelated = unrelated[:len(related)]
            stances = related + unrelated;

            # need to write this function, I dont think i'll be returning both
            # of these, since common ngrams is only 1 value, theres not really
            # a max/avg
            common_ngrams = NgramsGenerator().gen_common_ngrams(
                    self.dataset, bodies, stances, self._ngram_len)
            labeled_feature_set = []
            for i in range(len(stances)):
                labeled_feature = ({
                    'common_ngrams':common_ngrams[i]},
                    self._process_stance(stances[i]['Stance']))
                labeled_feature_set.append(labeled_feature)

            labeled_feat_dict[fold_id] = labeled_feature_set

        print "Generating features for hold out fold"
        holdout_common_ngrams = NgramsGenerator().gen_common_ngrams(
                self.dataset, hold_out, hold_out_stances, self._ngram_len)

        h_unlabeled_features = []
        h_labels = []
        for i in range(len(hold_out_stances)):
            unlabeled_feature = {
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
            del fold_ids[fold_id]

            training_set = [feat for fid in fold_ids for feat in labeled_feat_dict[fid]]
            testing_set = []
            testing_labels = []

            for feat, label in labeled_feat_dict[fold_id]:
                testing_set.append(feat)
                testing_labels.append(label)

            classifier = NaiveBayesClassifier.train(training_set)
            classifiers.append(classifier)
            pred = classifier.classify_many(testing_set)

            accuracy = self._score(pred, testing_labels)
            print "Fold ", fold_id, ", accuracy: ", accuracy
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                best_fold_cls = classifier

        h_res = best_fold_cls.classify_many(h_unlabeled_features)
        print 'holdout score: ', self._score(h_res, h_labels)

    def _score(self, predicted, actual):
        num_correct = 0
        for idx in range(len(predicted)):
            if predicted[idx] == actual[idx]:
                num_correct += 1
        accuracy = num_correct / float(len(predicted))
        return accuracy

    def _process_stance(self, stance):
        return 'unrelated' if stance == 'unrelated' else 'related'

if __name__ == "__main__":
    NgramClassify().do_validation()
