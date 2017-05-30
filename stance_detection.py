# import training data
# build features
# train naive bayes' classifer on features

# features: Jaccard similarity.
#   for sentence in body:
#       compute and store Jaccard similarity between body and sentence
#   compute average and maximum Jaccard similarity for the body

import pdb
from csv import DictReader

class StanceDetectionClassifier:
    def gen_training_features(self, bodies_fpath, stances_fpath):
        # load data and parse features. store in instance state
        self._read(bodies_fpath, stances_fpath)

    def _read(self, bodies_fpath, stances_fpath):
        with open(bodies_fpath, 'r') as f:
            r = DictReader(f)
            self._bodies = {}
            for line in r:
                body = line['articleBody'].decode('utf-8')
                self._bodies[int(line['Body ID'])] = body

        with open(stances_fpath, 'r') as f:
            r = DictReader(f)
            self._stances = []
            for line in r:
                headline = line['Headline'].decode('utf-8')
                stance = line['Stance'].decode('utf-8')
                body_id = int(line['Body ID'])
                self._stances.append({'Headline': headline,
                    'Body ID': body_id, 'Stance': stance})

    def train(self):
        pass

    def predict(self, bodies_fpath, stances_fpath):
        pass

cls = StanceDetectionClassifier()
cls.gen_training_features('training_data/train_bodies.csv', 'training_data/train_stances.csv')
