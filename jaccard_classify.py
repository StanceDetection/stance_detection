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

# TODO: add stemming, lowercase everything?, replace bad characters,
#       remove stop words

from csv import DictReader
import pdb
import string
import sys
import time

import nltk
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import jaccard_similarity_score

from dataset import DataSet
from generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from score import score_submission


class JaccardClassify:
    REMOVE_PUNC_MAP = dict((ord(char), None) for char in string.punctuation)

    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []
        self.dataset = DataSet()


    def do_validation(self, max_thresh=0.0, avg_thresh=0.0):
        print 'Validating with max_thresh ', max_thresh, '. avg_thresh: ', avg_thresh
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

            fold_avg_sims, fold_max_sims = self._gen_jaccard_sims(bodies, stances, max_thresh, avg_thresh)

            labeled_feature_set = []
            for i in range(len(stances)):
                labeled_feature = ({
                    'avg_sims':fold_avg_sims[i],
                    'max_sims':fold_max_sims[i]},
                    self._process_stance(stances[i]['Stance']))
                labeled_feature_set.append(labeled_feature)

            labeled_feat_dict[fold_id] = labeled_feature_set

        print "Generating features for hold out fold"
        holdout_avg_sims, holdout_max_sims = self._gen_jaccard_sims(
                hold_out, hold_out_stances, max_thresh, avg_thresh)

        h_unlabeled_features = []
        h_labels = []
        for i in range(len(hold_out_stances)):
            unlabeled_feature = {
                'avg_sims': holdout_avg_sims[i],
                'max_sims': holdout_max_sims[i]}
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


    def _gen_jaccard_sims(self, body_ids, stances, max_thresh, avg_thresh):
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
                # extend shorter word list so that both are the same length
                # len_diff = len(headline) - len(sent)
                # headline_cpy = headline
                # sent_cpy = sent

                # if len_diff < 0: # sent longer than headline
                #     headline_cpy = headline_cpy + ([headline_cpy[-1]] * abs(len_diff))
                # elif len_diff > 0: # headline longer than sent
                #     sent_cpy = sent_cpy + ([sent_cpy[-1]] * abs(len_diff))

                hs = set(headline)
                ss = set(sent)
                jacc_sim = len(hs.intersection(ss)) / float(len(hs.union(ss)))
                jacc_sims.append(jacc_sim)
                # jacc_sims.append(jaccard_similarity_score(headline_cpy, sent_cpy))

            # max_sim = self._threshold_parser(max(jacc_sims), [max_thresh])
            # avg_sim = self._threshold_parser((sum(jacc_sims) / len(jacc_sims)), [avg_thresh])
            max_sim = max(jacc_sims)
            avg_sim = sum(jacc_sims) / float(len(jacc_sims))
            max_sims.append(max_sim)
            avg_sims.append(avg_sim)

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
    # stances: [{'Headline': headline, 'Body ID': body_id, 'Stance': stance}, ..]
        with open(bodies_fpath, 'r') as f:
            r = DictReader(f)
            bodies_dict = {}
            for line in r:
                body = line['articleBody'].decode('utf-8')
                bodies_dict[int(line['Body ID'])] = body

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

        return bodies_dict, stances

if __name__ == "__main__":
    print sys.argv
    JaccardClassify().do_validation(max_thresh=sys.argv[1], avg_thresh=sys.argv[2])

