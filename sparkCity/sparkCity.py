## Imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from cityNamer import *




##create sc for spark
sc = SparkContext()

#creating data frame
sqlContext = SQLContext(sc)

#"geo.2014-04-03_21-16.txt"
path = "/user/data/GeoTweets-parquet/2015-11_geo_tweet.parquet"

twi = sqlContext.read.parquet(path)

udfCity = udf(udf_city_code, StringType())

twi = twi.withColumn("city", udfCity('geo.coordinates'))

twi.where(twi.city != "None").groupBy(twi.city).count().orderBy("count",ascending = False)

#twi = twi.where(twi.city != "None")
#
#write_path = "/home/xuehaipng/Documents/research/cityTest.parquet"
#
# twi.write.parquet(write_path)
