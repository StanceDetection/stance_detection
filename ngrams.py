# -*- coding: utf-8 -*-

# import training data

# TODO: add stemming, replace bad characters, remove punction
# Idea: what if we take 0 as being unknown

import pdb
import string
from csv import DictReader
import csv
import nltk
from sklearn.metrics import jaccard_similarity_score

class StanceDetectionClassifier:
    REMOVE_PUNC_MAP = dict((ord(char), None) for char in string.punctuation)
    _ngram_len = 1
    ngrams_dict = {"related": [], "unrelated": []}
    _bodies = {}
    _stances = []
    _test_bodies = {}
    _test_stances = []
    _test_results = []

    avg_ngrams_unrelated = 0
    avg_ngrams_related = 0

    def __init__(self):
        self._features = []

    def train(self, bodies_fpath, stances_fpath):
        self._read(bodies_fpath, stances_fpath)

        print('generating ngrams of length ' + str(self._ngram_len))
        self._train_ngrams(self._ngram_len)

        unrelated = self.ngrams_dict["unrelated"]
        related = self.ngrams_dict["related"]

        self.avg_ngrams_unrelated = float(reduce(lambda x, y: x + y, unrelated)) / len(unrelated)
        self.avg_ngrams_related = float(reduce(lambda x, y: x + y, related)) / len(related)

        print('avg ngrams for unrelated headlines - bodies: ' + str(self.avg_ngrams_unrelated))
        print('avg ngrams for related headlines - bodies: ' + str(self.avg_ngrams_related))

    def predict(self, bodies_fpath, stances_fpath):
        self._read_tests(bodies_fpath, stances_fpath)
        self._predict_relevance(self._ngram_len)
        pdb.set_trace()

    def _remove_punctuation(self, str_list):
        return map(lambda s: s.translate(self.REMOVE_PUNC_MAP), str_list)

    def _read(self, bodies_fpath, stances_fpath):
        with open(bodies_fpath, 'r') as f:
            r = DictReader(f)
            for line in r:
                body = line['articleBody'].decode('utf-8')
                self._bodies[int(line['Body ID'])] = body

        with open(stances_fpath, 'r') as f:
            r = DictReader(f)
            for line in r:
                headline = line['Headline'].decode('utf-8')
                stance = line['Stance'].decode('utf-8')
                body_id = int(line['Body ID'])
                self._stances.append({
                        'Headline': headline,
                        'Body ID': body_id,
                        'Stance': stance})

    def _read_tests(self, bodies_fpath, stances_fpath):
        with open(bodies_fpath, 'r') as f:
            r = DictReader(f)
            for line in r:
                body = line['articleBody'].decode('utf-8')
                self._test_bodies[int(line['Body ID'])] = body

        with open(stances_fpath, 'r') as f:
            r = DictReader(f)
            for line in r:
                headline = line['Headline'].decode('utf-8')
                body_id = int(line['Body ID'])
                self._test_stances.append({
                        'Headline': headline,
                        'Body ID': body_id, })

    def _get_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        tokens = [ token.lower() for token in tokens if len(token) > 1 ]
        ngram_list = list(nltk.ngrams(tokens, n))
        return ngram_list

    def _train_ngrams(self, n):
        body_ngrams = {}

        for bodyId in self._bodies:
            body_ngrams[bodyId] = self._get_ngrams(self._bodies[bodyId], n)

        for stance in self._stances:
            stance_ngrams = self._get_ngrams(stance['Headline'], n)

            num_ngrams_common = 0
            for ngram in stance_ngrams:
                if ngram in body_ngrams[stance['Body ID']]:
                    num_ngrams_common += 1
            if stance["Stance"] == "unrelated":
                self.ngrams_dict["unrelated"].append(num_ngrams_common)
            else:
                self.ngrams_dict["related"].append(num_ngrams_common)

    def _predict_relevance(self, n):
        with open('results.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow( ('Headline','Body ID','Stance') )

            body_ngrams = {}
            mid = (self.avg_ngrams_unrelated + self.avg_ngrams_related)/2
            mid = 4.2

            for bodyId in self._test_bodies:
                body_ngrams[bodyId] = self._get_ngrams(self._test_bodies[bodyId], n)

            for headline in self._test_stances:
                headline_ngrams = self._get_ngrams(headline['Headline'], n)
                num_ngrams_common = 0
                for ngram in headline_ngrams:
                    if ngram in body_ngrams[headline['Body ID']]:
                        num_ngrams_common += 1
                prediction = ""
                if num_ngrams_common < mid:
                    prediction = "unrelated"
                else:
                    prediction = "discuss"

                txt = headline['Headline'].encode('utf-8')
                csvwriter.writerow( (txt, str(headline['Body ID']), prediction) )

if __name__ == "__main__":
    cls = StanceDetectionClassifier()
    cls.train('training_data/train_bodies.csv', 'training_data/train_stances.csv')
    cls.predict('testing_data/test_bodies.csv', 'testing_data/test_stances_unlabeled.csv')
