import string
import nltk

class JaccardGenerator:
    REMOVE_PUNC_MAP = dict((ord(char), None) for char in string.punctuation)

    def gen_jaccard_sims(self, dataset, body_ids, stances):
        # currently assumes both body and headline are longer than 0.
        punc_rem_tokenizer = nltk.RegexpTokenizer(r'\w+')

        avg_sims = []
        max_sims = []

        parsed_bodies_dict = {}
        # for body_id, body in self.dataset.articles.iteritems():
        for body_id in body_ids:
            body = dataset.articles[body_id].lower()
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
                hs = set(headline)
                ss = set(sent)
                jacc_sim = len(hs.intersection(ss)) / float(len(hs.union(ss)))
                jacc_sims.append(jacc_sim)

            max_sim = max(jacc_sims)
            avg_sim = sum(jacc_sims) / float(len(jacc_sims))

            max_sims.append(max_sim)
            avg_sims.append(avg_sim)

        return avg_sims, max_sims


    def _word_tokenize(self, str_list):
        return map(lambda s: nltk.word_tokenize(s), str_list)


    def _remove_punctuation(self, str_list):
        return map(lambda s: s.translate(self.REMOVE_PUNC_MAP), str_list)


