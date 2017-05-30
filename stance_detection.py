# import training data
# build features
# train naive bayes' classifer on features

# features: Jaccard similarity.
#   for sentence in body:
#       compute and store Jaccard similarity between body and sentence
#   compute average and maximum Jaccard similarity for the body

import pdb
from csv import DictReader
import nltk

def get_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    tokens = [ token.lower() for token in tokens if len(token) > 1 ]
    return nltk.ngrams(tokens, n)

class StanceDetectionClassifier:
    def gen_training_features(self, bodies_fpath, stances_fpath):
        # load data and parse features. store in instance state
        self._read(bodies_fpath, stances_fpath)
        unigrams = self._train_ngrams(1)
        # bigrams = self._train_ngrams(2)

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

    def _train_ngrams(self, n):
        stance_similarities = []
        body_bigrams = {}

        for bodyId in self._bodies:
            body_bigrams[bodyId] = get_ngrams(self._bodies[bodyId], n)

        for stance in self._stances:
            stance_bigrams = get_ngrams(stance['Headline'], n)
            num_bigrams_common = 0
            for bigram in stance_bigrams:
                if bigram in body_bigrams[stance['Body ID']]:
                    num_bigrams_common += 1
            stance_similarities.append(num_bigrams_common)

        # normalize the counts based on length of the article
        for i in range(len(stance_similarities)):
            body_id = self._stances[i]['Body ID']
            stance_similarities[i] = float(stance_similarities[i])/len(self._bodies[body_id])

        return stance_similarities

    def train(self):
        pass

    def predict(self, bodies_fpath, stances_fpath):
        pass

cls = StanceDetectionClassifier()
cls.gen_training_features('training_data/train_bodies.csv', 'training_data/train_stances.csv')
