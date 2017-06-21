from gensim.models import word2vec
from scipy import spatial
from csv import DictReader

class WordVector:

    def gen_wordvectors(self, dataset, body_ids, stances):
        # https://cs.fit.edu/~mmahoney/compression/textdata.html
        sentences = word2vec.Text8Corpus('text8')
        model = word2vec.Word2Vec(sentences, size=200)
        wordvectordists = []

        for stance in stances:
            headingvector = None
            sentencevector = None
            newSentence = True
            newHeading = True
            for word in dataset.articles[stance['Body ID']].split('\n', 1)[0].split(' '):
                if word in model.wv.vocab:
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
                print(stance['Headline'])
                wordvectordists.append(0)

        return wordvectordists
