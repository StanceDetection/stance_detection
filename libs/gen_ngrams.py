# -*- coding: utf-8 -*-

# TODO: add stemming, lowercase everything?, replace bad characters,
#       remove stop words

import nltk

class NgramsGenerator:
    def _get_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        tokens = [ token.lower() for token in tokens if len(token) > 1 ]
        ngram_list = list(nltk.ngrams(tokens, n))
        return ngram_list


    def gen_common_ngrams(self, dataset, body_ids, stances, n):
        common_ngrams = []
        body_ngrams_dict = {}

        for body_id in body_ids:
            body_ngrams_dict[body_id] = self._get_ngrams(dataset.articles[body_id], n)

        for stance in stances:
            stance_ngrams = self._get_ngrams(stance['Headline'], n)

            num_ngrams_common = 0
            for ngram in stance_ngrams:
                if ngram in body_ngrams_dict[stance['Body ID']]:
                    num_ngrams_common += 1
            common_ngrams.append(num_ngrams_common)

        return common_ngrams
