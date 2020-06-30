import numpy as np

import WordStatistic

class NBayesClassifier(object) :

    def __init__(self) :

        self.dicth = {}

        self.dictl = {}

        self.dictSize = [0, 0]

        self.ratio = [0.5, 0, 0]

        return

    def fit(self, resulth, resultl) :

        self.dicth, counth = WordStatistic.createDict(resulth)

        self.dictl, countl = WordStatistic.createDict(resultl)

        self.dictSize = [counth, countl]

        self.ratio = [len(resulth) / (len(resulth) + len(resultl)), len(resultl), len(resulth)]

    def predict(self, text, lam = 1.0) :

        text, tfidf = WordStatistic.text_format(text)

        lowScore = 0.0

        highScore = 0.0

        for word in text:

            t = 1

            for i in tfidf :

                if(i[0] == word) :
                    t = i[1]

                    break

            if (word in self.dictl):

                lowScore += np.log(t * (self.dictl[word] + lam) / (self.dictSize[1] + lam * len(self.dictl)))

            else :

                lowScore += np.log(t * lam / (self.dictSize[1] + lam * len(self.dictl)))

        lowScore += np.log(1 - self.ratio[0])

        for word in text:

            t = 1

            for i in tfidf:

                if (i[0] == word):
                    t = i[1]

                    break

            if (word in self.dicth):

                highScore += np.log(t * (self.dicth[word] + lam) / (self.dictSize[0] + lam * len(self.dicth)))

            else :

                highScore += np.log(t * lam / (self.dictSize[0] + lam * len(self.dicth)))

        highScore += np.log(self.ratio[0])

        return (lowScore, highScore)
