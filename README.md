# WeatherForecast
Program that returns the predicted temperatures from 4AM to 12AM for a specific date and place in Sweden.

A combination of 3 Gaussian kernels is used:
* The first to account for the distance from a station to the point of interest.
* The second to account for the distance between the day a temperature measurement was made and the day of interest.
* The third to account for the distance between the hour of the day a temperature measurement was made and the hour of interest.


Data provided by the Swedish Meteorological and Hydrological Institute (SMHI).
