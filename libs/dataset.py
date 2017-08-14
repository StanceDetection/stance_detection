from csv import DictReader
from nltk.corpus import stopwords
import pdb


class DataSet():
    def __init__(self):
        print("Reading dataset")
        bodies = "training_data/train_bodies.csv"
        stances = "training_data/train_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])
            s['Headline'] = s['Headline'].decode('utf-8')

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody'].decode('utf-8')

        #remove_stop_words
        self.remove_stop_words()

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    def remove_stop_words(self):
        stopWords = stopwords.words("english")
        for stance in self.stances:
            filteredStance = ' '.join([word for word in stance['Headline'].split(" ") if word not in stopWords])
            stance['Headline'] = filteredStance
        for article in self.articles:
            filteredArticle = ' '.join([word for word in self.articles[article].split(" ") if word not in stopWords])
            self.articles[article] = filteredArticle

    def read(self,filename):
        rows = []
        with open(filename, "r") as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
