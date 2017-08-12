import nltk
import numpy

from libs.dataset import DataSet
from libs.generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from libs.score import score_submission

from gensim.models import word2vec

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.layers.embedding import Embedding # TODO: needed?

class StanceClassifier:
    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []
        self.dataset = DataSet()
        self._ngram_len = 2

    def do_validation(self):

        folds, hold_out = kfold_split(self.dataset, n_folds=10)
        fold_stances, hold_out_stances = get_stances_for_folds(self.dataset, folds, hold_out)

        print "Generating features for each fold"
        for fold_id in fold_stances:
            print "Generating features for fold ", fold_id
            bodies = folds[fold_id]
            stances = fold_stances[fold_id]

            fold_avg_sims, fold_max_sims = JaccardGenerator().gen_jaccard_sims(
                    self.dataset, bodies, stances)
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
        holdout_avg_sims, holdout_max_sims = JaccardGenerator().gen_jaccard_sims(
                self.dataset, hold_out, hold_out_stances)
        holdout_common_ngrams = NgramsGenerator().gen_common_ngrams(
                self.dataset, hold_out, hold_out_stances, self._ngram_len)

        # TODO: Needed?
        h_unlabeled_features = []
        h_labels = []
        for i in range(len(hold_out_stances)):
            unlabeled_feature = {
                'avg_sims': holdout_avg_sims[i],
                'max_sims': holdout_max_sims[i],
                'common_ngrams': holdout_common_ngrams[i]}
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
            x_train = []
            y_train = []

            # Pull out x_train and y_train
            for tup in training_set:
                print(tup[0])
                print(tup[1])

            testing_set = []
            testing_labels = []

            for feat, label in labeled_feat_dict[fold_id]:
                testing_set.append(feat)
                testing_labels.append(label)


            # Init Classifier
            model = Sequential()

            # Stack Layers
            model.add(Dense(units=64, input_dim=100))
            model.add(Activation('relu'))
            model.add(Dense(units=10))
            model.add(Activation('softmax'))

            # Configure Learning Process
            model.compile(loss='categorical_crossentropy',
                        optimizer='sgd',
                        metrics=['accuracy'])

            # NOTE: Manually feed model using model.train_on_batch(x_batch, y_batch)
            model.fit(x_train, y_train, epochs=5, batch_size=32)

            classes = model.predict(x_test, batch_size=128)




            classifier = NaiveBayesClassifier.train(training_set)
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


if __name__ == "__main__":
    StanceClassifier().do_validation()
