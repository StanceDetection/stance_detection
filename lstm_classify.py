import nltk
import numpy

from libs.dataset import DataSet
from libs.generate_test_splits import generate_hold_out_split, kfold_split, get_stances_for_folds
from libs.score import score_submission

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout
from keras.layers.wrappers import Bidirectional

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

        stance_options = {'discuss':[1,0,0,0],
                        'unrelated':[0,1,0,0],
                        'agree':[0,0,1,0],
                        'disagree':[0,0,0,1]}

        print "Validating using each fold as testing set"
        for fold_id in fold_stances:
            fold_ids = list(range(len(folds)))
            del fold_ids[fold_id] # deleted fold is test set for this run

            stances = fold_stances[fold_id]

            x_train_bodies = [] # List of lists of indexes
            x_train_heads = []
            y_train = [] # Stances numbered 0-3
            for stance in stances:
                body = self.dataset.getArticle(stance['Body ID'])
                x_train_bodies.append(body) # TODO: incorporate head
                x_train_heads.append(stance['Headline'].encode('utf-8'))
                y_train.append(stance_options[stance['Stance']])

            x_test_bodies = []
            x_test_heads = []
            y_test = []
            for stance in hold_out_stances:
                body = self.dataset.getArticle(stance['Body ID'])
                x_test_bodies.append(body)
                x_test_heads.append(stance['Headline'].encode('utf-8'))
                y_test.append(stance_options[stance['Stance']])

            # Keras preprocessing
            num_words = 5000 # Only consider most common words in dataset
            seq_len = 100 #seq_len is doubled when bodies and heads are combined

            tokenizer = Tokenizer(num_words = num_words)

            tokenizer.fit_on_texts(x_train_bodies +
                                x_train_heads +
                                x_test_bodies +
                                x_test_heads)

            # Convert from text to sequence (list of indices)
            x_train_bodies = tokenizer.texts_to_sequences(x_train_bodies)
            x_train_heads = tokenizer.texts_to_sequences(x_train_heads)
            x_test_bodies = tokenizer.texts_to_sequences(x_test_bodies)
            x_test_heads = tokenizer.texts_to_sequences(x_test_heads)

            # Truncate or pad to sequence len
            x_train_bodies = pad_sequences(x_train_bodies, maxlen=seq_len)
            x_train_heads = pad_sequences(x_train_heads, maxlen=seq_len)
            x_test_bodies = pad_sequences(x_test_bodies, maxlen=seq_len)
            x_test_heads = pad_sequences(x_test_heads, maxlen=seq_len)

            # TODO: Doesn't work
            # Concatenate each body with head
            x_train = [list(a) + list(b) for a, b in zip(x_train_bodies, x_train_heads)]
            x_test = [list(a) + list(b) for a, b in zip(x_test_bodies, x_test_heads)]

            # Create LSTM model
            model = Sequential()

            # Stack Layers
            model.add(Embedding(num_words,output_dim=256))
            model.add(LSTM(128) )
            model.add(Dropout(0.5))
            model.add(Dense(4, activation='softmax'))

            model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=1, batch_size=32)
            scores = model.evaluate(x_test, y_test, batch_size=32)

            print('Score:')
            print(scores)

if __name__ == "__main__":
    StanceClassifier().do_validation()
