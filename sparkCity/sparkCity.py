## Imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf

import math
import re

stop_words = ["a", "secondly", "all", "consider", "whoever", "everybody", "four", "go", "mill", "evermore", "causes", "seemed", "whose", "certainly", "to", "does", "th", "under", "sorry", "sent", "very", "every", "yourselves", "did", "forth", "list", "fewer", "try", "p", "round", "someday", "says", "ten", "till", "d", "past", "likely", "notwithstanding", "further", "hopefully", "even", "what", "appear", "brief", "goes", "sup", "new", "mustn", "rd", "ever", "thin", "hasn", "full", "respectively", "never", "here", "let", "others", "alone", "along", "fifteen", "ahead", "k", "allows", "amount", "howbeit", "usually", "que", "changes", "thats", "hither", "via", "followed", "merely", "put", "ninety", "viz", "yourself", "use", "from", "would", "contains", "two", "next", "few", "call", "therefore", "taken", "themselves", "thru", "until", "more", "knows", "becomes", "hereby", "herein", "everywhere", "particular", "known", "must", "me", "none", "f", "this", "oh", "anywhere", "nine", "can", "mr", "following", "my", "example", "indicated", "give", "neverf", "near", "indicates", "something", "want", "needs", "end", "thing", "rather", "six", "how", "low", "instead", "needn", "okay", "tried", "haven", "may", "after", "eighty", "different", "hereupon", "such", "third", "whenever", "amid", "appreciate", "q", "ones", "so", "specifying", "allow", "keeps", "thirty", "help", "undoing", "indeed", "over", "move", "mainly", "soon", "whilst", "through", "looks", "fify", "still", "its", "before", "thank", "thence", "somewhere", "inward", "ll", "actually", "better", "thanx", "ours", "might", "versus", "then", "them", "someone", "somebody", "thereby", "underneath", "course", "they", "half", "not", "now", "nor", "gets", "name", "always", "reasonably", "didn", "whither", "l", "each", "found", "went", "side", "mean", "everyone", "directly", "doing", "eg", "weren", "ex", "our", "beyond", "out", "furthermore", "since", "forty", "looking", "re", "seriously", "got", "cause", "thereupon", "given", "quite", "whereupon", "besides", "ask", "anyhow", "inasmuch", "backwards", "couldn", "g", "could", "tries", "keep", "caption", "w", "ltd", "hence", "onto", "think", "first", "already", "seeming", "thereafter", "one", "done", "another", "thick", "miss", "awfully", "little", "their", "twenty", "top", "system", "least", "anyone", "indicate", "too", "hundred", "gives", "mostly", "that", "exactly", "took", "immediate", "regards", "somewhat", "kept", "believe", "herself", "than", "specify", "daren", "b", "unfortunately", "gotten", "zero", "i", "r", "were", "toward", "minus", "anyways", "and", "alongside", "beforehand", "mine", "say", "unlikely", "have", "need", "seen", "seem", "saw", "clearly", "relatively", "abroad", "thoroughly", "latter", "able", "aside", "thorough", "also", "take", "which", "begin", "towards", "unless", "though", "any", "who", "most", "eight", "but", "nothing", "why", "sub", "forever", "don", "especially", "nobody", "noone", "sometimes", "m", "amoungst", "mrs", "definitely", "neverless", "normally", "came", "saying", "particularly", "show", "anyway", "ending", "find", "fifth", "hadn", "outside", "should", "only", "going", "do", "his", "above", "get", "de", "overall", "truly", "cannot", "nearly", "despite", "during", "him", "is", "regarding", "qv", "h", "cry", "twice", "she", "contain", "x", "where", "thanks", "ignored", "theirs", "see", "computer", "are", "best", "said", "away", "currently", "please", "behind", "various", "between", "probably", "neither", "across", "available", "we", "recently", "however", "nd", "come", "both", "c", "last", "many", "taking", "whereafter", "according", "against", "selves", "s", "became", "com", "comes", "otherwise", "among", "liked", "co", "afterwards", "seems", "whatever", "hers", "non", "moreover", "throughout", "considering", "meantime", "described", "second", "three", "been", "whom", "much", "interest", "likewise", "hardly", "empty", "wants", "corresponding", "latterly", "concerning", "else", "former", "those", "myself", "novel", "look", "unlike", "these", "bill", "value", "n", "will", "while", "ain", "shall", "theres", "seven", "almost", "wherever", "sincere", "thus", "it", "cant", "vs", "in", "ie", "if", "containing", "inc", "etc", "perhaps", "insofar", "make", "same", "wherein", "beside", "several", "shan", "fairly", "used", "upon", "uses", "recent", "lower", "off", "whereby", "nevertheless", "whole", "nonetheless", "well", "anybody", "obviously", "without", "y", "the", "con", "yours", "know", "lest", "things", "just", "less", "being", "downwards", "presumably", "front", "greetings", "useful", "yes", "yet", "unto", "farther", "had", "except", "has", "adj", "ought", "around", "possible", "whichever", "five", "makes", "using", "part", "dare", "hereafter", "maybe", "necessary", "like", "follows", "either", "become", "therein", "twelve", "because", "old", "often", "namely", "twitter", "some", "back", "self", "sure", "specified", "ourselves", "happens", "provided", "for", "bottom", "opposite", "per", "everything", "asking", "provides", "tends", "t", "be", "sensible", "nowhere", "although", "sixty", "by", "on", "about", "ok", "anything", "getting", "of", "v", "o", "whence", "plus", "consequently", "or", "seeing", "own", "formerly", "into", "within", "due", "down", "appropriate", "mightn", "couldnt", "your", "her", "eleven", "aren", "there", "amidst", "accordingly", "inner", "way", "forward", "was", "himself", "elsewhere", "enough", "becoming", "amongst", "somehow", "hi", "trying", "with", "he", "made", "whether", "inside", "up", "tell", "placed", "below", "un", "z", "gone", "later", "associated", "certain", "describe", "am", "doesn", "an", "meanwhile", "as", "sometime", "right", "at", "et", "fill", "again", "hasnt", "entirely", "no", "whereas", "when", "detail", "lately", "other", "you", "really", "regardless", "welcome", "upwards", "ago", "e", "together", "hello", "itself", "u", "apart", "far", "serious", "backward", "having", "once"]

##city parameters
radius = 0.87544195
##city list
Cities = [
#Los Angeles
["LAX",[-118.248191,34.045006]],
#New York  -73.061719, 40.821452
["NYC",[-74.003105, 40.716205]],
#miami
["MIA",[-80.194734, 25.749890]],
#Orlando
["ORL",[-81.369054, 28.537684]],
#atlanta
["ATL",[-84.387956, 33.740127]],
#Charlotte
["CHA",[-80.841435, 35.229785]],
#Nashville
["NAV",[-86.766753, 36.151462]],
#Louisville
["LOU",[-85.748268, 38.249390]],
#Cincinnati
["CIN",[-84.500767, 39.099486]],
#Indianapolis
["IND",[-86.157326, 39.778972]],
#Columbus
["COL",[-82.998694, 39.961160]],
#Detroit
["DET",[-83.044275, 42.330866]],
#Chicago
["CHI",[-87.627793, 41.874563]],
#Milwaukee
["MIL",[-87.919372, 43.040745]],
#Minneapolis
["MIN",[-93.285971, 44.982799]],
#Philadelphia
["PHI",[-75.159460, 39.945103]],
#Rhode Island
["RHI",[-71.403432, 41.823369]],
#Washington
["WAS",[-76.604185, 39.282868]],
#Pittsburgh
["PIT",[-79.988188, 40.430130]],
#Kansas City
["KAN",[-94.588492, 39.089903]],
#Denver
["DEN",[-104.940650, 39.727695]],
#Dallas
["DAL",[-96.831669, 32.776106]],
#Houston
["HOU",[-95.366488, 29.761600]],
#Phoenix
["PHX",[-112.060061, 33.466748]],
#Las Vegas
["LAV",[-115.144610, 36.168834]],
#San Francisco
["SAF",[-122.419689, 37.775005]],
#Portland
["POR",[-122.676377, 45.523670]],
#Seattle
["SEA",[-122.339670, 47.573777]]
]

year = [2015, 2014]

# month_fifteen = ["06", "07", "08", "09", "10", "11"]

month_fifteen = ["10", "11"]

month_forteen = ["01", "02", "03", "04"]

#compute the distance between two points
def distance(p1, old_p2):
    p2 = [old_p2[1], old_p2[0]]
    return math.sqrt(math.pow(float(p1[0])-float(p2[0]), 2) + math.pow(float(p1[1])-float(p2[1]), 2))

#check if a point is in an circle
def in_circle(center,radius, point):
    if point is not None and distance(center, point) <= radius:
        return True
    else:
        return False

#check if the point is inside one of the cities
def city_code(cities, rad, point):
    for city in cities:
        if in_circle(city[1],rad, point):
            return city[0]
    return "None"

def udf_city_code(point):
    return city_code(Cities, radius, point)

# text preprocessing
def preprocessing(sentence):
    tokens = []
    if sentence is not None:
        url = re.compile("http[^ \t\n\r\f\v'\"]+", re.I)
        term = re.compile("[\w']+", re.I)
        #reomve all un necessary information
        text_data = re.sub(r'[^\w]', ' ', re.sub(url,'', sentence.lower()))
        # split words
        tokens = term.findall(text_data.encode('ascii', 'ignore'))
        # remove stop word
        tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def city_tweets_partion(path, fname):

    twi = sqlContext.read.parquet(path)

    udfCity = udf(udf_city_code, StringType())

    twi = twi.withColumn("city", udfCity('geo.coordinates'))

    udfPreprocess = udf(preprocessing, StringType())
    #twi.where(twi.city != "None").groupBy(twi.city).count().orderBy("count",ascending = False).show()

    for idx, city in enumerate(Cities):
        cityTwi = twi.where(twi.city == Cities[idx][0]) \
            .select("text") \
            .withColumn("tokens", udfPreprocess('text')) \
            .drop('text')

        cityTwi.rdd \
        .map(lambda row: [str(c) for c in row]) \
        .repartition(1) \
        .saveAsTextFile("/user/claymore/CityTextCorpus/" + Cities[idx][0] + "/" + Cities[idx][0] + "_" + fname + ".txt")

##create sc for spark
sc = SparkContext()

#creating data frame
sqlContext = SQLContext(sc)

for m in month_fifteen:
    path = "/user/claymore/GeoTweets/2015-%s_geo_tweet.parquet" % m
    city_tweets_partion(path, "2015-%s_geo_tweet.parquet" % m)


for m in month_forteen:
    path = "/user/claymore/GeoTweets/2014-%s_geo_tweet.parquet" % m
    city_tweets_partion(path, "2015-%s_geo_tweet.parquet" % m)
