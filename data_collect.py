## Imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json, os, gzip
import math
import decimal
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf


## Module Constants
APP_NAME = "HaipingXue Spark Twitter Collector"


conf = (SparkConf()
	.setAppName(APP_NAME)
#	.set("spark.network.timeout", "1200s")
#	.set("spark.executor.memory", "5g")
)
# 	.set("spark.shuffle.io.preferDirectBufs", "false"))



sc = SparkContext(conf = conf)
#creating data frame
sqlContext = SQLContext(sc)

#schema for data collection
#schema for data collection
geo_twitter_schema = StructType([StructField('created_at',StringType(),True),
                                 StructField('lang',StringType(),True),
                                 StructField("user",  StructType([StructField('id',LongType(),True)])),
                                 StructField('text',StringType(),True),
                                StructField('geo',StructType([StructField("coordinates", ArrayType(DecimalType()), True)]),True),
                                StructField('place',StructType([StructField("place_type", StringType(), True),
                                                                 StructField("name", StringType(), True),
                                                                 StructField("full_name", StringType(), True),
                                                                 StructField("country", StringType(), True)]),True)])


year = [2015, 2014]

month_fifteen = ["06", "07", "08", "09", "10", "11"]

month_forteen = ["01", "02", "03", "04"]

for m in month_fifteen:

    path =("/user/infobot/GeoTweets/2015/*/*/geo.2015-%s*" % m)

    twi = sqlContext.read.json(path, schema = geo_twitter_schema)

    twi.where(twi.lang == 'en').drop('lang').write.parquet("/user/claymore/GeoTweets/2015-%s_geo_tweet.parquet" % m)

for m in month_forteen:

    path ="/user/infobot/GeoTweets/2014/*/*/geo.2014-%s*" % m

    twi = sqlContext.read.json(path, schema = geo_twitter_schema)

    twi.where(twi.lang == 'en').drop('lang').write.parquet("/user/claymore/GeoTweets/2014-%s_geo_tweet.parquet" % m)
