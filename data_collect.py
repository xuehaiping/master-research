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
geo_twitter_schema = StructType([StructField('created_at',StringType(),True),
                                 StructField("user",  StructType([StructField('id',LongType(),True)])),
                                 StructField('text',StringType(),True),
                                StructField('geo',StructType([StructField("coordinates", ArrayType(DecimalType()), True)]),True),                                 
                                StructField('place',StructType([StructField("place_type", StringType(), True), 
                                                                 StructField("name", StringType(), True),
                                                                 StructField("full_name", StringType(), True), 
                                                                 StructField("country", StringType(), True)]),True)])



#"geo.2014-04-03_21-16.txt"
path ="/user/data/GeoTweets/geo.2015-11*"
 
twi = sqlContext.read.json(path, schema = geo_twitter_schema)
#coalesce
#twi.show()
#udfScoreToCategory=udf(centerdis, FloatType())


twi.write.parquet("/user/claymore/tmp_geo/2015-11_geo_tweet.parquet")

#twi = twi.withColumn("City Code", udfScoreToCategory('geo.coordinates'))
