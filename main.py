import NBayes
import WordStatistic

if __name__ == "__main__" :

    nb = NBayes.NBayesClassifier()

    """
        初始化朴素贝叶斯分类器
    """

    cL = WordStatistic.lawCutting(WordStatistic.loadLaw("criminalLaw.TXT"))

    """
        刑法训练样本、正类
    """

    dL = WordStatistic.lawCutting(WordStatistic.loadLaw("citizenLaw.TXT"))

    """
        民法训练样本，负类
    """

    nb.fit(cL, dL)

    """
        进行拟合
    """

    print("字典规模{0}".format(nb.dictSize))

    reCall_cL = 0

    reCall_dL = 0

    for law in cL :

        lS, hS = nb.predict(law)

        if(lS < hS) : reCall_cL += 1
        else : print("Wrong[criminal] " + law)



    for law in dL :

        lS, hS = nb.predict(law)

        if(lS > hS) : reCall_dL += 1
        else : print("Wrong[citizen] " + law)


    print("拟合-刑法召回率：{0}".format(reCall_cL / len(cL)))

    print("拟合-民法召回率：{0}".format(reCall_dL / len(dL)))

    """
        召回检验
    """



    criminalLaw = WordStatistic.lawCutting(WordStatistic.loadLaw("criminalLawTest.TXT"))

    citizenLaw = WordStatistic.lawCutting(WordStatistic.loadLaw("citizenLawTest.TXT"))

    reCall_cL = 0

    reCall_dL = 0

    for law in criminalLaw :

        lS, hS = nb.predict(law)

        if(lS < hS) : reCall_cL += 1
        else : print("Wrong[criminal] " + law)

    for law in citizenLaw :

        lS, hS = nb.predict(law)

        if(lS > hS) : reCall_dL += 1
        else : print("Wrong[citizen] " + law)

    print("泛化-刑法召回率：{0}".format(reCall_cL / len(criminalLaw)))

    print("泛化-民法召回率：{0}".format(reCall_dL / len(citizenLaw)))