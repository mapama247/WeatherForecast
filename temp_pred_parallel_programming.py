# coding=utf-8

from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext

path_to_temp = "./data/temperature-readings.csv"
path_to_stations = "./data/stations.csv"

#Specify the smoothing factors here:
h_place = 30
h_date = 3
h_time = 1

#Specify date, time and location here:
date = "2014-05-17"
location = [58.4108,15.6214] #[latitude , longitude]

times = ["04:00:00","06:00:00","08:00:00","10:00:00","12:00:00","14:00:00","16:00:00","18:00:00","20:00:00","22:00:00","00:00:00"]

def PlaceDiff(lon1,lat1,lon2,lat2):
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a))
	km = 6367 * c
	return km

def DayDiff(day1,day2):
	return (datetime.strptime(day1,'%Y-%m-%d') - datetime.strptime(day2,'%Y-%m-%d')).days

def TimeDiff(time1,time2):
	interval = datetime.strptime(time1,'%H:%M:%S') - datetime.strptime(time2,'%H:%M:%S')
	seconds_diff = interval.total_seconds()
	hours_diff = seconds_diff / 3600
	return hours_diff

def gaussian(d,h):
	return exp( float(-(d**2)) / float(2*(h**2)) )

sc = SparkContext(appName="tempPred")

# ( STATION , ( DATE , TIME , TEMPERATURE ) )
rdd_temp = sc.textFile(path_to_temp) \
			 .map(lambda line: line.split(";")) \
			 .map(lambda x: ( x[0] , (str(x[1]),str(x[2]),float(x[3])) )) \
			 .filter(lambda x: datetime.strptime(x[1][0],'%Y-%m-%d') < datetime.strptime(date,'%Y-%m-%d'))

# ( STATION , ( LATITUDE , LONGITUDE ) )
rdd_stations = sc.textFile(path_to_stations) \
				 .map(lambda line: line.split(";")) \
				 .map(lambda x: ( x[0] , (float(x[3]),float(x[4])) ))

# {u'122430': (62.72909, 12.7264), u'65090': (56.1498, 15.5921), u'112510': (61.8526, 12.2689), ... }
map_stations = rdd_stations.collectAsMap()
bc_stations = sc.broadcast(map_stations)

# ( STATION , ( DATE , TIME , TEMPERATURE , LATITUDE , LONGITUDE ) )
training_data = rdd_temp.map(lambda x: (x[0] , (x[1][0],x[1][1],x[1][2],bc_stations.value[x[0]][0],bc_stations.value[x[0]][1]) )).cache()

# ( TEMP , (DIST1,DIST2,DIST3) ) -> ( TEMP , K1+K2+K3 ) -> ( 0 , (K,temp*K) ) -> ( 0 , sum(temp*K)/sum(K) )
predictions = []
for i in times:
	pred = training_data.map(lambda x: (x[1][2] , (TimeDiff(i,x[1][1]) , DayDiff(date,x[1][0]) , PlaceDiff(location[1],location[0],x[1][4],x[1][3])))) \
						.map(lambda x: (x[0] , gaussian(x[1][0],h_time) * gaussian(x[1][1],h_date) * gaussian(x[1][2],h_place) ) ) \
						.map(lambda x: (0 , (x[1],x[0]*x[1]))) \
						.reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1])) \
						.map(lambda x: (float(x[1][1])/float(x[1][0]))) \
						.collect()
	predictions.append(pred)

print([(times[i],predictions[i]) for i in range(0,len(times))])