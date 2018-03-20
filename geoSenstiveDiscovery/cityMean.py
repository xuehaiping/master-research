from sparkCity.cityNamer import Cities
from gensim.models import *
from util import cosineSim
import matplotlib.pyplot as plt
import numpy as np

THRESHOLD = 0.3

def findCityCosSimMean():
    """
    Find average cosine similarity for each cities with global
    """
    # dictionary for cosine similarity mean with global model of each city
    cityCosineMean = {}

    glbPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/bigram/GLB/GLB_word_model'
    glbModel = Word2Vec.load(glbPath)

    # this is for geo sensitive word
    for city in Cities:
        loadPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model = Word2Vec.load(loadPath)

        # sims = []
        summation = 0
        geoSensitive = []
        # compute mean for each city
        for word in model.wv.vocab:
            sim = cosineSim(model.wv[word], glbModel[word])
            if sim <= THRESHOLD:
                geoSensitive.append([word, sim])
            # sims.append(sim)
            summation += sim

        cityMean = (1.0 * summation) / len(model.wv.vocab)
        cityCosineMean[city[0]] = cityMean

        # plot the data
        plotHistogram(mean=cityMean, sims=sims, city=city[0])



        # sort the geo sensitive word with similarity
        geoSensitive = sorted(geoSensitive, key=lambda x: x[1], reverse=True)
        # write geo sensitive words to file
        with open('commonGeoSensitiveWord/%s_words' % city[0], 'w+') as file:
            for geoWord in geoSensitive:
                file.write('%s %.8f\n' % (geoWord[0], geoWord[1]))
        file.close()

    # # write to file
    # with open("cityMeanResult", "w+") as f:
    #     for city in Cities:
    #         f.write("%s %.8f\n" % (city[0], cityCosineMean[city[0]]))

    # f.close()


def plotHistogram(mean, sims, city):
    mu = mean  # mean of distribution
    sigma = np.std(np.array(sims))  # standard deviation of distribution
    x = sims

    num_bins = 20

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins)
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('Frequency')
    ax.set_title(r'Histogram of IQ: $\mu= %.8f$, $\sigma=%.8f$' % (mu, sigma))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.savefig('histogramSim/ %s_histogram.png' % city)


def findCommonGeoSensitiveWord():

    glbPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/bigram/GLB/GLB_word_model'
    glbModel = Word2Vec.load(glbPath)

    # find common words all models have
    commonWords = set(glbModel.wv.vocab)
    for city in Cities:
        loadPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model = Word2Vec.load(loadPath)
        commonWords.intersection(model.wv.vocab)

    citiesWord = {}
    # this is for geo sensitive word
    for city in Cities:
        loadPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model = Word2Vec.load(loadPath)

        geoSensitive = []
        # compute mean for each city
        for word in model.wv.vocab:
            sim = cosineSim(model.wv[word], glbModel[word])
            if sim <= THRESHOLD and word in commonWords:
                geoSensitive.append([word, sim])

        citiesWord[city[0]] = geoSensitive

    # find common words all model have
    allInCommon = set(glbModel.wv.vocab)
    for key in citiesWord:
        words = [x[0] for x in citiesWord[key]]
        allInCommon = allInCommon.intersection(words)

    for key in citiesWord:
        geoSensitive = citiesWord[key]
        # write geo sensitive words to file
        with open('commonGeoSensitiveWord/%s_words' % key, 'w+') as fi:
            for geoWord in geoSensitive:
                if geoWord[0] not in allInCommon:
                    fi.write('%s %.8f\n' % (geoWord[0], geoWord[1]))
        fi.close()


def notCommonGeoSensitiveWord():

    glbPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/bigram/GLB/GLB_word_model'
    glbModel = Word2Vec.load(glbPath)

    # find common words all models have
    commonWords = set(glbModel.wv.vocab)
    for city in Cities:
        loadPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model = Word2Vec.load(loadPath)
        commonWords = commonWords.intersection(model.wv.vocab)


    # this is for geo sensitive word
    for city in Cities:
        loadPath = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model = Word2Vec.load(loadPath)

        geoSensitive = []
        # compute mean for each city
        for word in model.wv.vocab:
            sim = cosineSim(model.wv[word], glbModel[word])
            if sim <= THRESHOLD and (word not in commonWords):
                geoSensitive.append([word, sim])

        # sort the geo sensitive word with similarity
        geoSensitive = sorted(geoSensitive, key=lambda x: x[1], reverse=True)

        # write geo sensitive words to file
        with open('notCommonGeoSensitiveWord/%s_words' % city[0], 'w+') as file:
            for geoWord in geoSensitive:
                file.write('%s %.8f\n' % (geoWord[0], geoWord[1]))
        file.close()


findCommonGeoSensitiveWord()
