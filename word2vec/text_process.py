import re, os, gzip
from nltk.tokenize import TweetTokenizer
import nltk.tokenize
from nltk.corpus import stopwords
import simplejson as json


stop_word_list = ["a", "secondly", "all", "consider", "whoever", "everybody", "four", "go", "mill", "evermore", "causes", "seemed", "whose", "certainly", "to", "does",
 "th", "under", "sorry", "sent", "very", "every", "yourselves", "did", "forth", "list", "fewer", "try", "p", "round", "someday", "says", "ten", "till",
 "d", "past", "likely", "notwithstanding", "further", "hopefully", "even", "what", "appear", "brief", "goes", "sup", "new", "mustn", "rd", "ever", "thin", "hasn",
 "full", "respectively", "never", "here", "let", "others", "alone", "along", "fifteen", "ahead", "k", "allows", "amount", "howbeit", "usually",
 "que", "changes", "thats", "hither", "via", "followed", "merely", "put", "ninety", "viz", "yourself", "use", "from", "would", "contains", "two", "next",
 "few", "call", "therefore", "taken", "themselves", "thru", "until", "more", "knows", "becomes", "hereby", "herein", "everywhere", "particular", "known",
 "must", "me", "none", "f", "this", "oh", "anywhere", "nine", "can", "mr", "following", "my", "example", "indicated", "give", "neverf", "near", "indicates",
 "something", "want", "needs", "end", "thing", "rather", "six", "how", "low", "instead", "needn", "okay", "tried", "haven", "may", "after", "eighty", "different",
 "hereupon", "such", "third", "whenever", "amid", "appreciate", "q", "ones", "so", "specifying", "allow", "keeps", "thirty", "help", "undoing", "indeed", "over", "move",
 "mainly", "soon", "whilst", "through", "looks", "fify", "still", "its", "before", "thank", "thence", "somewhere", "inward", "ll", "actually", "better", "thanx", "ours",
 "might", "versus", "then", "them", "someone", "somebody", "thereby", "underneath", "course", "they", "half", "not", "now", "nor", "gets", "name", "always", "reasonably",
 "didn", "whither", "l", "each", "found", "went", "side", "mean", "everyone", "directly", "doing", "eg", "weren", "ex", "our", "beyond", "out", "furthermore", "since",
 "forty", "looking", "re", "seriously", "got", "cause", "thereupon", "given", "quite", "whereupon", "besides", "ask", "anyhow", "inasmuch", "backwards", "couldn", "g", "could",
 "tries", "keep", "caption", "w", "ltd", "hence", "onto", "think", "first", "already", "seeming", "thereafter", "one", "done", "another", "thick", "miss", "awfully", "little",
 "their", "twenty", "top", "system", "least", "anyone", "indicate", "too", "hundred", "gives", "mostly", "that", "exactly", "took", "immediate", "regards", "somewhat", "kept", "believe",
 "herself", "than", "specify", "daren", "b", "unfortunately", "gotten", "zero", "i", "r", "were", "toward", "minus", "anyways", "and", "alongside", "beforehand", "mine",
 "say", "unlikely", "have", "need", "seen", "seem", "saw", "clearly", "relatively", "abroad", "thoroughly", "latter", "able", "aside", "thorough", "also", "take",
 "which", "begin", "towards", "unless", "though", "any", "who", "most", "eight", "but", "nothing", "why", "sub", "forever", "don", "especially", "nobody", "noone", "sometimes",
 "m", "amoungst", "mrs", "definitely", "neverless", "normally", "came", "saying", "particularly", "show", "anyway", "ending", "find", "fifth", "hadn", "outside",
 "should", "only", "going", "do", "his", "above", "get", "de", "overall", "truly", "cannot", "nearly", "despite", "during", "him", "is", "regarding", "qv", "h", "cry",
 "twice", "she", "contain", "x", "where", "thanks", "ignored", "theirs", "see", "computer", "are", "best", "said", "away", "currently", "please", "behind", "various",
 "between", "probably", "neither", "across", "available", "we", "recently", "however", "nd", "come", "both", "c", "last", "many", "taking", "whereafter", "according",
 "against", "selves", "s", "became", "com", "comes", "otherwise", "among", "liked", "co", "afterwards", "seems", "whatever", "hers", "non", "moreover", "throughout",
 "considering", "meantime", "described", "second", "three", "been", "whom", "much", "interest", "likewise", "hardly", "empty", "wants", "corresponding", "latterly",
 "concerning", "else", "former", "those", "myself", "novel", "look", "unlike", "these", "bill", "value", "n", "will", "while", "ain", "shall", "theres", "seven", "almost",
 "wherever", "sincere", "thus", "it", "cant", "vs", "in", "ie", "if", "containing", "inc", "etc", "perhaps", "insofar", "make", "same", "wherein", "beside", "several", "shan",
 "fairly", "used", "upon", "uses", "recent", "lower", "off", "whereby", "nevertheless", "whole", "nonetheless", "well", "anybody", "obviously", "without", "y", "the", "con",
 "yours", "know", "lest", "things", "just", "less", "being", "downwards", "presumably", "front", "greetings", "useful", "yes", "yet", "unto", "farther", "had", "except",
 "has", "adj", "ought", "around", "possible", "whichever", "five", "makes", "using", "part", "dare", "hereafter", "maybe", "necessary", "like", "follows", "either",
 "become", "therein", "twelve", "because", "old", "often", "namely", "twitter", "some", "back", "self", "sure", "specified", "ourselves", "happens", "provided", "for",
 "bottom", "opposite", "per", "everything", "asking", "provides", "tends", "t", "be", "sensible", "nowhere", "although", "sixty", "by", "on", "about", "ok", "anything",
 "getting", "of", "v", "o", "whence", "plus", "consequently", "or", "seeing", "own", "formerly", "into", "within", "due", "down", "appropriate", "mightn", "couldnt",
 "your", "her", "eleven", "aren", "there", "amidst", "accordingly", "inner", "way", "forward", "was", "himself", "elsewhere", "enough", "becoming", "amongst",
 "somehow", "hi", "trying", "with", "he", "made", "whether", "inside", "up", "tell", "placed", "below", "un", "z", "gone", "later", "associated", "certain",
 "describe", "am", "doesn", "an", "meanwhile", "as", "sometime", "right", "at", "et", "fill", "again", "hasnt", "entirely", "no", "whereas", "when", "detail",
 "lately", "other", "you", "really", "regardless", "welcome", "upwards", "ago", "e", "together", "hello", "itself", "u", "apart", "far", "serious", "backward", "having", "once"]


stop_list = stopwords.words('english')

##location rectans
SF = [[-122.524804,37.811153], [-122.358636,37.848032], [-122.355203,37.706925], [-122.509698,37.706925]]
NY = [[-73.91857,40.920544], [-74.080618,40.655405], [-73.969381,40.529224], [-73.724936,40.592865], [-73.698843,40.762628], [-73.783987,40.883177]]
Chicago = [[-87.647698,42.031929], [-87.897637,42.018667], [-87.710869,41.643132], [-87.521355,41.642106], [-87.517235,41.747725]]
Houston = [[-95.64917,29.9065], [-95.642304,29.70273], [-95.492615,29.584568], [-95.171265,29.586956], [-95.189118,29.887451]]
Seattle = [[-122.436499,47.73829], [-122.433753,47.498537], [-122.225013,47.496682], [-122.255225,47.736443]]
Denver = [[-105.113633,39.81921], [-105.117753,39.613748], [-104.842409,39.610574], [-104.843095,39.761172], [-104.731172,39.763283], [-104.734605,39.815518]]

def uncompress(path, fname):
    file_location = os.path.join(path,fname)
    print 'Open file in location:'+ file_location
    with gzip.open(file_location, 'rb') as f:
        tweets = f.readlines()
    return tweets


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


#filter out unuseful words
def text_filter(sentence, stop_words):

    #only english characters
    term = re.compile(r"[^A-Za-z0-9]+|@\S+|http+?\://\S+", re.S)
    new_sentence = re.sub(term, ' ', sentence)
    tknzr = TweetTokenizer()

    #tokenize the sentence
    tokenized = tknzr.tokenize(new_sentence.lower())

    #remove the stop words
    processed_sentence = [word for word in tokenized if word not in stop_words]

    return processed_sentence


#create text corpus for each city
def city_text_corpus(city, dirname):
    print "writing to output file..............................................."
    #outputfile =
    for yname in os.listdir(dirname):
        ydir = os.path.join(dirname, yname)
        for mname in os.listdir(ydir):
            mdir = os.path.join(ydir, mname)
            for dname in os.listdir(mdir):
                ddir = os.path.join(mdir, dname)
                for dayfile in os.listdir(ddir):
                    cityfile = city + '_' + mname + '_' + dname + '_' + yname + '.txt.gz'
                    #find out the right city
                    if dayfile == cityfile:
                        print 'Opening: ' + cityfile
                        twlist = uncompress(ddir, dayfile)
                        for tweet in twlist:
                            js_data = json.loads(tweet)
                            if 'text' in js_data:
                                text_corpus_list = text_filter(js_data['text'])


