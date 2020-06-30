import re
import jieba
import jieba.analyse


def stopwordslist(path):

    """
        生成停用词（stop_word)字典
    """

    stopwords = [line.strip() for line in open(path, 'r', encoding='utf-8').readlines()]

    return stopwords

stopword = stopwordslist('stop_words.txt')

def loadLaw(path) :

    """
        读取法条
    """

    res = ""

    for line in open(path, 'r', encoding='utf-8').readlines() :

        res += " " + line.strip()

    return res


def lawCutting(text):

    """
        法条剪切函数
    """


    """
        根据第XX编/章/条进行划分
    """

    text = re.compile('第.*?[条|编|章]').split(text)

    res = []

    for law in text:

        """
            去除掉过短的无意义样本
        """

        if(len(law) > 4) : res.append(law)

    return res

def text_format(text):

    """
        分词函数
    """

    res = []

    temp = []

    """
        去除文本中的非汉字
    """

    TFIDF = jieba.analyse.extract_tags(text, topK = 100, withWeight = True)

    text = re.compile('[^\u4e00-\u9fa5]').split(text)

    text = re.compile('\W+|\d+|[a-z]+|[A-Z]+').split(' '.join(text))


    """
        jieba进行语素划分
    """

    for word in text:
        temp.extend(jieba.cut(word))

    """
        去除停用词(stop-word)
    """

    for word in temp:
        if(word not in stopword): res.append(word)

    return (res, TFIDF)

def createDict(dataSet, thres = 2):

    temp = {}

    res = {}

    count = 0

    """
        语素统计
    """

    for i in range(0, len(dataSet)):

        text = dataSet[i]

        text, tfidf = text_format(text)

        for word in text:

            if (word not in temp):

                temp[word] = 1

            else:

                temp[word] += 1


    """
        过滤阈值以下的语素
    """

    for i in temp:

        if (temp[i] <= thres): continue

        res[i] = temp[i]

        count += res[i]

    return (res, count)