import alginEmbed
from gensim.models import *
from sparkCity.cityNamer import Cities





def alignTwoCity(city1, city2):
    """
    align two cities's word embedding, save it to directory
    """
    load_path1 = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/bigram/%s/%s_word_model' % (city1, city1)
    load_path2 = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/bigram/%s/%s_word_model' % (city2, city2)
    model1 = Word2Vec.load(load_path1)
    model2 = Word2Vec.load(load_path2)
    return alginEmbed.smart_procrustes_align_gensim(model1, model2)


def alginCity():
    """
    Align all words with global embedding
    """
    for city in Cities:
        print "Aligning city " + city[0]
        model = alignTwoCity("GLB", city[0])
        load_path1 = '/Users/xuehaiping/Documents/research/master-research/word2vec/Word2vec_model/aligned/%s/%s_aligned_GLB_model' % (city[0], city[0])
        model.save(load_path1)
