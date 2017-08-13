from __future__ import division
from scipy import spatial
from csv import DictReader
import string

class WordVector:

    def gen_wordvectors(self, dataset, body_ids, stances, model):
        wordvectordists = []

        exclude = set(string.punctuation)

        for stance in stances:
            headingvector = None
            sentencevector = None
            newSentence = True
            newHeading = True

            for word in dataset.articles[stance['Body ID']].split('\n', 1)[0].split(' '):
                word = ''.join(ch for ch in word if ch not in exclude).lower()
                words += 1
                if word in model.wv.vocab:
                    foundwords += 1
                    if newSentence:
                        sentencevector = model.wv[word]
                        newSentence = False
                    else:
                        sentencevector = sentencevector + model.wv[word]
            for word in stance['Headline'].split(' '):
                if word in model.wv.vocab:
                    if newHeading:
                        headingvector = model.wv[word]
                        newHeading = False
                    else:
                        headingvector = headingvector + model.wv[word]
            if(headingvector is not None and sentencevector is not None):
                wordvectordists.append(1 - spatial.distance.cosine(headingvector, sentencevector))
            else:
                wordvectordists.append(0)

        return wordvectordists
