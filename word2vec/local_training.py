# import modules & set up logging
import gensim, logging, os, gzip, nltk, re
import simplejson as json
from nltk import word_tokenize
from pprint import pprint
from gensim.models import Phrases
from sklearn.cluster import KMeans
import numpy as np


def point_in_poly(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


counter_for_words = 0
##location rectans
SF = [[-122.524804,37.811153], [-122.358636,37.848032], [-122.355203,37.706925], [-122.509698,37.706925]]
NY = [[-73.91857,40.920544], [-74.080618,40.655405], [-73.969381,40.529224], [-73.724936,40.592865], [-73.698843,40.762628], [-73.783987,40.883177]]
Chicago = [[-87.647698,42.031929], [-87.897637,42.018667], [-87.710869,41.643132], [-87.521355,41.642106], [-87.517235,41.747725]]
Houston = [[-95.64917,29.9065], [-95.642304,29.70273], [-95.492615,29.584568], [-95.171265,29.586956], [-95.189118,29.887451]]
Seattle = [[-122.436499,47.73829], [-122.433753,47.498537], [-122.225013,47.496682], [-122.255225,47.736443]]
Denver = [[-105.113633,39.81921], [-105.117753,39.613748], [-104.842409,39.610574], [-104.843095,39.761172], [-104.731172,39.763283], [-104.734605,39.815518]]


logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s', level=logging.INFO)


#memory friendly global text trainning generator
class MySentences_global(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for yname in os.listdir(self.dirname):
            ydir=os.path.join(self.dirname, yname)
            for mname in os.listdir(ydir):
                mdir= os.path.join(ydir, mname)
                for dname in os.listdir(mdir):
                    ddir = os.path.join(mdir, dname)
                    for dname in os.listdir(ddir):
                        tweets =uncompress(ddir, dname)
                        text_corpus = parse(tweets)
                        for sentence in text_corpus:
                            yield sentence


# memory friendly global text trainning generator
class MySentences_city(object):
    def __init__(self, dirname, city):
        self.dirname = dirname
        self.city = city

    def __iter__(self):
        for yname in os.listdir(self.dirname):
            ydir = os.path.join(self.dirname, yname)
            for mname in os.listdir(ydir):
                mdir = os.path.join(ydir, mname)
                for dname in os.listdir(mdir):
                    ddir = os.path.join(mdir, dname)
                    for dayfile in os.listdir(ddir):
                        cityfile = self.city + '_' + mname + '_' + dname + '_' + yname + '.txt.gz'
                        if dayfile == cityfile:
                            print 'Opening: ' + cityfile
                            tweets = uncompress(ddir, dayfile)
                            text_corpus = parse(tweets)
                            for sentence in text_corpus:
                                yield sentence
                            break




#ungzip the file and put tweets in an array
def uncompress(path, fname):
    file_location = os.path.join(path,fname)
    print 'Open file in location:'+ file_location
    with gzip.open(file_location, 'rb') as f:
        tweets = f.readlines()
    return tweets




#generate the data for training
def parse(twlist):
    twi_sentence = []
    #url format
    term = re.compile("[\w']+", re.I)
    url = re.compile(r"https?\://\S+", re.DOTALL)
    #extract text data
    for tweet in twlist:
        js_data = json.loads(tweet)
        if 'text' in js_data:
            text_data = re.sub(r'[^\w]', ' ', re.sub(url,'', js_data['text']).lower())
            if type(text_data) is str:
                text_data = unicode(text_data, 'utf-8')
                twi_sentence.append(term.findall(text_data))
    return twi_sentence


#check the availability of geo information in the 'place' tag
def examing_location(dir_location):

    ##counter
    tweet_count = 0
    place_count = 0
    ##check all files in the directory
    for fname in os.listdir(dir_location):
        twlist=uncompress(dir_location, fname)
        ##parse the tweets and count tweets have name or full name
        for tweet in twlist:
            data = json.loads(tweet)
            tweet_count= tweet_count+1
            ##check text-based place information
            if 'place' in data:
                if data['place']!= None:
                    if data['place']['name'] != None:
                        place_count=place_count+1
                    elif data['place']['full_name']!=None:
                        place_count=place_count+1

    print "Total tweets is " + str(tweet_count)
    print "Tweets with text-based location " + str(place_count)

#tranning datas
def training_data():
    city_dir = ['Chicago', 'Seattle', 'San_francisico', 'Houston', 'Denver', 'New_york']

    for city in city_dir:
        sentences_city = MySentences_city('/home/xuehaipng/Downloads/twi_data/', city)
        model = gensim.models.Word2Vec(sentences_city, min_count = 5, size = 100, workers = 4 )
        save_path = '/home/xuehaipng/Documents/Word2vec_model/' + city +'/words/' + city + '_words'
        model.save(save_path)


    city_bi = ['Chicago', 'Seattle', 'San_francisico', 'Houston', 'Denver', 'New_york']

    for city in city_bi:
        sentences_city = MySentences_city('/home/xuehaipng/Downloads/twi_data/', city)
        bigram = Phrases(sentences_city, min_count=10)
        model = gensim.models.Word2Vec(bigram[sentences_city], min_count = 5, size = 100, workers = 4 )
        save_path = '/home/xuehaipng/Documents/Word2vec_model/' + city +'/bigram/' + city + '_bigram'
        model.save(save_path)


#print model.vocab
# for i in range(0,100):
#     print model.vocab

# for words in model.vocab:
#     print 'word is: ' + words + '    count is: ' + str(model.vocab[words].count)

#get count key
def get_count(item):
    return item[1]


def KMean(w2v_model, output_file):
    word_vectors = w2v_model.syn0
    num_clusters = word_vectors.shape[0] / 20
    # training kmeans
    print "training the KMean..........................."
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=4)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map=dict(zip(w2v_model.index2word, idx))
    # write to the output file
    print "Open output file: " + output_file + " ......................"
    file = open(output_file, 'w+')
    for cluster in range(num_clusters):
        print cluster
        #file.write("cluster " + str(cluster) + '\n')
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
            #file.write(word_centroid_map.keys()[i] + "  ")
             print word_centroid_map.keys()[i]
        #file.write('\n')
    file.close()


# count the words being trained in each model
def word_count():
    city_dir = ['Chicago', 'Seattle', 'San_francisico', 'Houston', 'Denver', 'New_york']
    output_path = "/home/xuehaipng/Documents/general_statistic/word_count_statistic"

    output_file = open(output_path, 'w+')

    for city in city_dir:
        path = '/home/xuehaipng/Documents/Word2vec_model/' + city + '/words/' + city + '_' + 'words'

        model = gensim.models.Word2Vec.load(path)
        sum_of_word = 0
        for word in model.index2word:
            sum_of_word = sum_of_word + model.vocab[word].count

        output_file.write(city + '\'s word count is ' + str(sum_of_word) +'\n')
        output_file.write(city + '\'s vocabulary size is ' + str(len(model.vocab)) + '\n\n')

    output_file.close()






#model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/Houston/words/Houston_words')

#save_path = '/home/xuehaipng/Documents/Word2vec_model/' + city + '/words/' + city + '_words'
   # model.save(save_path)
#
# my_sentence = MySentences_global('/home/xuehaipng/Downloads/twi_data')
# # model = gensim.models.Word2Vec(my_sentence, min_count = 5, size = 100, workers = 4 )
# # save_path = '/home/xuehaipng/Documents/Word2vec_model/' + 'global' +'/words/' + 'global' + '_words'
# # model.save(save_path)
#
# bigram = Phrases(my_sentence, min_count=10)
# model1 = gensim.models.Word2Vec(bigram[my_sentence], min_count = 5, size = 100, workers = 4 )
# save_path1 = '/home/xuehaipng/Documents/Word2vec_model/' + 'global' +'/bigram/' + 'global' + '_bigram'
# model1.save(save_path1)

model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/Houston/words/Houston_words')
