import nltk
import numpy

from libs.dataset import DataSet
from libs.generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from libs.score import score_submission

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout

class StanceClassifier:
    def __init__(self):
        self._labeled_feature_set = []
        self._test_feature_set = []
        self.dataset = DataSet()

    def do_validation(self):
        folds, hold_out = kfold_split(self.dataset, n_folds=1)
        fold_stances, hold_out_stances = get_stances_for_folds(self.dataset, folds, hold_out)

        fold_accuracy = {}
        best_fold_accuracy = 0.0
        classifiers = []

        stance_options = {'discuss':0,
                        'unrelated':1,
                        'agree':2,
                        'disagree':3}

        print "Validating using each fold as testing set"
        for fold_id in fold_stances:
            fold_ids = list(range(len(folds)))
            del fold_ids[fold_id] # deleted fold is test set for this run

            stances = fold_stances[fold_id]

            x_train = [] # List of lists of indexes
            y_train = [] # Stances number 0-3
            for stance in stances:
                body = self.dataset.getArticle(stance['Body ID'])
                x_train.append(body) # TODO: incorporate head
                y_train.append(stance_options[stance['Stance']])

            x_test = []
            y_test = []
            for stance in hold_out_stances:
                body = self.dataset.getArticle(stance['Body ID'])
                x_test.append(body)
                y_test.append(stance_options[stance['Stance']])

            # Keras preprocessing
            num_words = 5000 # Only consider most common words in dataset
            seq_len = 100

            tokenizer = Tokenizer(num_words = num_words)

            tokenizer.fit_on_texts(x_train + x_test)

            # Convert from text to sequence (list of indices)
            x_train = tokenizer.texts_to_sequences(x_train)
            x_test = tokenizer.texts_to_sequences(x_test)

            # Truncate or pad to sequence len
            x_train = pad_sequences(x_train, maxlen=seq_len)
            x_test = pad_sequences(x_test, maxlen=seq_len)

            # Create LSTM model
            model = Sequential()

            # Stack Layers
            model.add(Embedding(num_words,output_dim=256))
            model.add(LSTM(128))
            #model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=1, batch_size=32)
            scores = model.evaluate(x_test, y_test, batch_size=32)

            print('Score:')
            print(scores)

if __name__ == "__main__":
    StanceClassifier().do_validation()
