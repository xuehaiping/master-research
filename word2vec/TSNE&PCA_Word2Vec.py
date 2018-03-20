# import modules & set up logging
import gensim, logging, os, gzip, nltk, re
from gensim.models import Phrases
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA



city_dir = ['Chicago', 'Seattle', 'San_francisico', 'Houston', 'Denver', 'New_york']


#load tranined model in the city_list, model type is 'words' or 'bigram'
def load_list(city_list, model_type):
    #model list for return
    model_list = []
    #open all trained model in the list
    for city in city_list:
        model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/' + city +'/' + model_type + '/' + city + '_' + model_type)
        model_list.append(model)

    return model_list


#tf-idf find out document unique words and terms
def tf_idf(tf, doc_num, doc_occour):
    return math.log10(1 + tf)* math.log10(1.0*doc_num/doc_occour)


#sort assistance function
def get_weight(item):
    return item[1]


#find local senstive word, model type is 'words' or 'bigram'
def vocabulary_classifier(city, model_type):
    #load the target model
    target_model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/' + city +'/' + model_type + '/' + city + '_' + model_type)
    #load model list
    model_list = load_list(city_dir, model_type)
    #result list   tuple (word, weight)
    word_weight=[]

    print "computing td-idf for all terms ............................................."

    #compute tf_idf for each word
    for word in target_model.index2word:
        doc_count = 0
        for model in model_list:
            if word in model.index2word:
                doc_count = doc_count + 1
        #add the result to the word_weight
        word_weight.append( (word, tf_idf(target_model.vocab[word].count, len(model_list), doc_count)) )

    #sort the word weight in descending order
        word_weight_sorted = sorted(word_weight, key = get_weight, reverse = True)

    #write the result to the file
    out_fn = '/home/xuehaipng/Documents/td_idf_result/'+ city + '_' + model_type + '_td_idf_result'
    file = open(out_fn, 'w+')

    print "writing the result to: " + out_fn

    file.write("format is word and td-idf" + '\n')

    for result in word_weight_sorted:
        file.write(result[0] + '  ' + str(result[1]) + '\n' )

    file.close()



def get_rid_of_stop_word(city, model_type, vec_dic):
    #read the tf-idf into memory
    fname = '/home/xuehaipng/Documents/td_idf_result/'+ city + '_' + model_type + '_td_idf_result'
    file = open(fname,'r')
    lines = file.readlines()
    lines = lines[1:]
    #label list and vecter list for return
    label_list = []
    reduced_vec = []
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
        tokens = lines[i].split()
        if float(tokens[1]) <= 1:
            continue
        label_list.append(tokens[0])
        reduced_vec.append(vec_dic[tokens[0]])
    return (label_list, np.array(reduced_vec))




#auto computing td-idf for all cities, model type is 'words' or 'bigram'
def auto_tf_idf(model_type):
    for city in city_dir:
        vocabulary_classifier(city, model_type)

# PCA transform
def pca_transform(model):
    # vector list
    feature_vec = []
    # word label
    label_list = []
    # make a corresponeding list for vectors and words
    for word in model.index2word:
        label_list.append(word)
        feature_vec.append(model[word])
        pca_model = PCA(n_components=2)

    reduced_vecs = pca_model.fit_transform(np.array(feature_vec, dtype='float'))

    print "Model's explained variance vector "
    print pca_model.explained_variance_

    return dict(zip(label_list, reduced_vecs))


#plot the reduced pca vectors to a image
def plot(plot_pack):
    #set up the coodinates for plot
    xs = plot_pack[1][:, 0]
    ys = plot_pack[1][:, 1]

    #draw
    f=plt.figure(figsize=(70,70))
    plt.scatter(xs, ys)

    for i in range(len(plot_pack[0])):
        plt.annotate(
            plot_pack[0][i],
            xy = (xs[i],ys[i]), xytext = (3,3),
            textcoords = 'offset points', ha = 'left', va = 'top'
        )
    plt.savefig("test.svg")



#plot a pca model with filtered input, model type is 'words' or 'bigram'
def word2vec_pca_plot(city, model_type):
    model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/' + city + '/' + model_type + '/' + city + '_' + model_type)
    plot(get_rid_of_stop_word(city, model_type, pca_transform(model)))


#find top 20 similar words in a model
def find_similar_words(word_list,model):
    similar_words = []
    for word in word_list:
        if word in model.index2word:
            raw_sim = model.most_similar(positive = word, topn=20)
            for data in raw_sim:
                similar_words.append(data[0])
    return similar_words



#select select pca transfered word coordinates from the model, the order of coordinates is slected word for similarity and the words from the function input
def select_word_pca_plot(word_list, model_type):
    #trained word2vec model
    model_list = []
    #load trained models into memory and pass labels and coordinates to ploting function
    for city in city_dir:
        model = gensim.models.Word2Vec.load('/home/xuehaipng/Documents/Word2vec_model/' + city + '/' + model_type + '/' + city + '_' + model_type)
        word_dic = pca_transform(model)
        #get all the word coordinates
        select_word = find_similar_words(word_list,model)
        #for similar words
        cord_list = []
        #for centers
        center_list = []
        center_cords = []
        for word in select_word:
            cord_list.append(word_dic[word])

        for word in word_list:
            if word in model.index2word:
                center_list.append(word)
                center_cords.append(word_dic[word])

        plot_words( (center_list, select_word, np.array(cord_list), np.array(center_cords), city, model_type) )




#plot the reduced pca vectors to a image
def plot_words(plot_pack):
    #set up the coodinates for plot
    xs = plot_pack[2][:, 0]
    ys = plot_pack[2][:, 1]

    #draw similar words
    fig = plt.figure(figsize=(70,70))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c='b')

    for i in range(len(plot_pack[1])):
        ax1.annotate(
            plot_pack[1][i],
            xy = (xs[i],ys[i]), xytext = (3,3),
            textcoords = 'offset points', ha = 'left', va = 'top'
        )

    #draw centers
    xs_c = plot_pack[3][:, 0]
    ys_c = plot_pack[3][:, 1]
    ax1.scatter(xs_c, ys_c, c= 'r')

    for i in range(len(plot_pack[0])):
        ax1.annotate(
            plot_pack[0][i],
            xy = (xs_c[i],ys_c[i]), xytext = (3,3),
            textcoords = 'offset points', ha = 'left', va = 'top'
        )


    plt.savefig(plot_pack[4] +'_' + plot_pack[0][0] + '_' + plot_pack[5] + ".svg")



#prepare the vector for TNSE
def getWrodVecs(model):
    #vector list
    feature_vec = []
    #word label
    label_list = []
    #make a corresponeding list for vectors and words
    for word in model.index2word:
        label_list.append(word)
        feature_vec.append(model[word])
        ts = TSNE(2)

    reduced_vecs = ts.fit_transform(np.array(feature_vec, dtype = 'float'))
    #return (np.array(feature_vec, dtype = 'float'), label_list)


#reduce the dimension of the vector
def TSNE_training(feature_vec):
    print feature_vec
    ts = TSNE(2)
    reduced_vecs = ts.fit_transform(feature_vec)
    return reduced_vecs







word_list = ['airport', 'baseball','city','college','football','hall','highway','landmark',
             'location','mayor','park','restaurant','football','baseball', 'hall', 'stadium', 'highway']

#select_word_pca_plot(['landmark'], 'words')

# for word in word_list:
#     select_word_pca_plot([word], 'words')

#
# for word in word_list:
#     select_word_pca_plot([word], 'bigram')
#










