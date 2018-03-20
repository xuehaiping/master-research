import numpy as np
import gensim

# def ensure_dir(file_path):
#     if not os.path.isdir(file_path):
#         os.makedirs(file_path)
#
#
# def alignCity(city1, city2):
#     """
#     align two cities's word embedding, save it to directory
#     """
#     load_path1 = '%s/%s_word_model' % (city1, city1)
#     load_path2 = '%s/%s_word_model' % (city2, city2)
#     model1 = Word2Vec.load(load_path1)
#     model2 = Word2Vec.load(load_path2)
#
#     # find common words in two models
#     commonWords = findCommonWord(model1, model2)
#     # build matrix for two cities based on common words
#     m1 = buildMatrix(model1, commonWords)
#     m2 = buildMatrix(model2, commonWords)
#
#     # alignment
#     m = m2.T.dot(m1)
#     u, _, v = np.linalg.svd(m)
#     ortho = u.dot(v)
#
#     # save model and vocabulary to file
#     directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), city1 + '-' + city2)
#     ensure_dir(directory)
#     np.save(os.path.join(directory, city1 + '-' + city2 + "_model"), m2.dot(ortho))
#     with open(os.path.join(directory, city1 + '-' + city2 + ".vocab"), "w+") as f:
#         for word in commonWords:
#             f.write("%s\n" % word)
#     f.close()
#
#     return commonWords, m2.dot(ortho)
#
#
# def findCommonWord(model1, model2):
#     """
#     find common words in two word2vec models
#     """
#     return list(set(model1.wv.vocab).intersection(model2.wv.vocab))
#
#
# def buildMatrix(model, commonWords):
#     """
#     create a new matrix only contains common words
#     """
#     resMat = []
#     for word in commonWords:
#         resMat.append(model.wv[word])
#
#     return np.array(resMat)
#
# alignCity("LAX", "NYC")

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)
    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)
    # Then for each model...
    l = []
    l.append(m1)
    l.append(m2)
    for m in l:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        m.wv.init_sims()
        old_arr = m.wv.syn0norm
        lu = [old_arr[index] for index in indices]
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return (m1, m2)