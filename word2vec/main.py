import glob, logging, gensim
from Sentences import MyPhraseSentences
from Sentences import MySentences
from cities import Cities




# train with words
def word_training():
    # log training processes
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    for city in Cities:

        filepath = "CityTextCorpus/%s/*/part-00000" % city

        cityFileList = glob.glob(filepath)

        cityTweets = MySentences(cityFileList)

        model = gensim.models.Word2Vec(cityTweets, min_count = 5, size = 100, workers = 8)

        save_path = 'Word2vec_model/words/%s/%s_word_model' % (city, city)

        model.save(save_path)


# train with phrases
def phraseTraining():
    # log training processes
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    for city in Cities:

        filepath = "CityPhraseTextCorpus/%s_phrase" % city

        cityFileList = glob.glob(filepath)

        cityTweets = MyPhraseSentences(cityFileList)

        model = gensim.models.Word2Vec(cityTweets, min_count = 5, size = 100, workers = 8)

        save_path = 'Word2vec_model/bigram/%s/%s_word_model' % (city, city)

        model.save(save_path)


def globPhraseTraining():
    filepath = "/Users/xuehaiping/Documents/research/master-research/word2vec/CityPhraseTextCorpus/*"

    cityFileList = glob.glob(filepath)

    cityTweets = MySentences(cityFileList)

    model = gensim.models.Word2Vec(cityTweets, min_count = 5, size = 100, workers = 8 )

    save_path = 'Word2vec_model/bigram/GLB/GLB_word_model'

    model.save(save_path)


# phraseTraining()

# filepath = "CityTextCorpus/*/*/part-00000"
#
# cityFileList = glob.glob(filepath)
#
# cityTweets = MySentences(cityFileList)
#
# model = gensim.models.Word2Vec(cityTweets, min_count = 5, size = 100, workers = 8 )
#
# save_path = 'Word2vec_model/words/GLB/GLB_word_model'
#
# model.save(save_path)

globPhraseTraining()