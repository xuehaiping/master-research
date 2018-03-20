from gensim.models import Phrases
from Sentences import MySentences
import glob, logging
from cities import Cities


# create phrase detector for bigram
bigram = Phrases(min_count=10)


print "====================================== Start bigram training ======================================"

# train bigram
for city in Cities:

    print "====================================== City %s bigram training ======================================" % city

    filepath = "CityTextCorpus/%s/*/part-00000" % city

    cityFileList = glob.glob(filepath)

    cityTweets = MySentences(cityFileList)

    bigram.add_vocab(cityTweets)


print "====================================== Start trigram training ======================================"

# create phrase detector for trigram
trigram = Phrases(min_count=5)

# train trigram
for city in Cities:

    print "====================================== City %s trigram training ======================================" % city

    filepath = "CityTextCorpus/%s/*/part-00000" % city

    cityFileList = glob.glob(filepath)

    cityTweets = MySentences(cityFileList)

    trigram.add_vocab(bigram[cityTweets])


print "====================================== Start writing sentences to file ======================================"

# write trained sentences to new file for each city
for city in Cities:

    filepath = "CityTextCorpus/%s/*/part-00000" % city

    newFilePath = "CityPhraseTextCorpus/%s_phrase" % city

    f = open(newFilePath, "w+")

    cityFileList = glob.glob(filepath)

    cityTweets = MySentences(cityFileList)

    for lines in cityTweets:
        newLine = trigram[bigram[lines]]
        f.write(" ".join(newLine) + '\n')

    f.close()
