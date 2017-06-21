from gensim.models import word2vec
from scipy import spatial
from csv import DictReader

class WordVector:

    # def train(self, bodies_fpath, stances_fpath):
    #     print 'Generating training features'
    #     self._train_bodies, self._train_stances = self._read(bodies_fpath, stances_fpath, True)

    # stances: [{'Headline': headline, 'Body ID': body_id, 'Stance': stance}, ..]
    # def _read(self, bodies_fpath, stances_fpath, is_training):
    #     with open(bodies_fpath, 'r') as f:
    #         r = DictReader(f)
    #         bodies_dict = {}
    #         for line in r:
    #             body = line['articleBody'].decode('utf-8')
    #             bodies_dict[int(line['Body ID'])] = body
    #
    #     with open(stances_fpath, 'r') as f:
    #         r = DictReader(f)
    #         stances = []
    #         for line in r:
    #             headline = line['Headline'].decode('utf-8')
    #             body_id = int(line['Body ID'])
    #             if(is_training):
    #                 stance = line['Stance'].decode('utf-8')
    #                 stances.append({
    #                         'Headline': headline,
    #                         'Body ID': body_id,
    #                         'Stance': stance})
    #             else:
    #                 stances.append({
    #                         'Headline': headline,
    #                         'Body ID': body_id})
    #
    #     return bodies_dict, stances

    def gen_wordvectors(self, dataset, body_ids, stances):
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
