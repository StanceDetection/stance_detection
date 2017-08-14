from csv import DictReader
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

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))


    def read(self,filename):
        rows = []
        with open(filename, "r") as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    def getArticle(self, article_id):
        body = self.articles[article_id].encode('utf-8')
        return body
