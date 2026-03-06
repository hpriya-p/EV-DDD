PROCESSED_STATIONS_DF = "data/processed/stations_data.csv"
DIST_MATRIX_JSON = "data/processed/dist_matrix.json"

# bounding box
MAX_LAT = 45
MIN_LAT = 32
MAX_LNG = -67
MIN_LNG = -120

# Truck properties
RANGE = 305.7 #km. in miles its 190 miles

"""
~~~ ORIGINAL (for a 100 kwh station) ~~~
[1, 10]: 35 seconds each percentage point
[11, 40]: 25
[41, 60]: 30
[61, 80]: 45
[81, 90]: 80
[91, 95]: 130
[96, 100]: 400
~~~

time to charge to 80 (in seconds): 10 * 35 + 29 * 25 + 19 * 30 + 19 * 45 = 2500s
time to charge to 80 at a 100 kwh station should be 60min * 25/10 = 9000s

~~~ Scaled (for a 100 kwh station) ~~~
[1, 10]: 35 * 3.6 seconds each percentage point
[11, 40]: 25 * 3.6
[41, 60]: 30 * 3.6
[61, 80]: 45 * 3.6
[81, 90]: 80 * 3.6
[91, 95]: 130 * 3.6
[96, 100]: 400 * 3.6
~~~

"""

# for a 100kwh station,
speed_curve ={0: {'speed': 3600 * 1/(35*3.6), 'minbat': 0, 'maxbat': 11}, \
                   1: {'speed': 3600 * 1/(25*3.6), 'minbat': 11, 'maxbat': 41}, \
                   2: {'speed': 3600 * 1/(30*3.6), 'minbat': 41, 'maxbat': 61}, \
                   3: {'speed': 3600 * 1/(45*3.6), 'minbat': 61, 'maxbat': 81}, \
                   4: {'speed': 3600 * 1/(80*3.6), 'minbat': 81, 'maxbat': 91}, \
                   5: {'speed': 3600 * 1/(130*3.6), 'minbat': 91, 'maxbat': 96}, \
                   6: {'speed': 3600 * 1/(400 * 3.6), 'minbat': 96, 'maxbat': 100}} # speed = amt per hour


# Other parameters
BAT_FROM_TIME = 3600 * 1000 * .75 # battery consumption = (3600 * 1000 * .8) * (time in hrs)

