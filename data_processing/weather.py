import requests
import urllib.parse
import meteostat as me
from datetime import datetime
import pandas as pd

def get_coordinates(query, debug=False):
    
    
    url = 'https://nominatim.openstreetmap.org/search?q=' + urllib.parse.quote(query) +'&format=jsonv2'

    response = requests.get(url).json()
    
    if debug:
        print(response[0]["lat"])
        print(response[0]["lon"])

    return float(response[0]["lat"]), float(response[0]["lon"])

def get_week(date):

    return date.week

def get_month(date):

    return date.month

def calc_normalization(temp,targ,ind,label='t'): # here, targ is the df containing all avg and std info, while ind is either the week or the month of the line being currently evaluated

    return (temp - targ.loc[ind][label+'mean'])/targ.loc[ind][label+'std']

def generate_aggregated_weather_data(location, radius, start, end, tzone='EST', savefile=''):
    ###### PARAMETERS
    # location : searchable name of a place (i.e. New York City)
    # radius (in meters) : radius within which weather stations will be considered (from coord extracted from location search)
    # start (datetime) : start date
    # end (datetime) : end date 
    # tzone : timezone
    ######## END PARAMETERS

    loclat,loclong = get_coordinates(location)
    stats = me.Stations()
    stats = stats.nearby(loclat,loclong,radius)
    avail_stats = stats.inventory('hourly',(start,end))
    hourly_dat = me.Hourly(avail_stats.fetch(),start,end,tzone).normalize().interpolate(6).aggregate('1H',True).fetch().reset_index(level='time')
    hourly_dat['dtime'] = pd.to_datetime(hourly_dat['time'])
    hourly_dat['week'] = hourly_dat['dtime'].apply(get_week) #gets week number inside dframe for anomaly calculations
    hourly_dat['month'] = hourly_dat['dtime'].apply(get_month)

    ## Gives table from which one can compute weekly weather anomalies

    anom = hourly_dat.groupby('week').mean()
    anom['tmin'] = hourly_dat.groupby('week').min()['temp']
    anom['tmax'] = hourly_dat.groupby('week').max()['temp']
    anom['tmean'] = anom['temp']
    anom['tstd'] = hourly_dat.groupby('week').std()['temp']
    anom['precmean'] = anom['prcp']
    anom['precstd'] = hourly_dat.groupby('week').std()['prcp']
    anom['wspdmean'] = anom['wspd']
    anom['wspdstd'] = hourly_dat.groupby('week').std()['wspd']
    anom['pressmean'] = anom['pres']
    anom['pressstd'] = hourly_dat.groupby('week').std()['wspd']
    anom = anom.drop(['temp','wpgt','tsun','snow','time','month','dtime','prcp','wspd','pres'], axis=1)

    ## Gives table from which one can compute monthly weather anomalies (we do not add this to normals table since normals is over 29 year period, while our stat markers will be over much shorter periods)

    mon_anom = hourly_dat.groupby('month').mean()
    mon_anom['tmin'] = hourly_dat.groupby('month').min()['temp']
    mon_anom['tmax'] = hourly_dat.groupby('month').max()['temp']
    mon_anom['tmean'] = mon_anom['temp']
    mon_anom['tstd'] = hourly_dat.groupby('month').std()['temp']
    mon_anom['precmean'] = mon_anom['prcp']
    mon_anom['precstd'] = hourly_dat.groupby('month').std()['prcp']
    mon_anom['wspdmean'] = mon_anom['wspd']
    mon_anom['wspdstd'] = hourly_dat.groupby('month').std()['wspd']
    mon_anom['pressmean'] = mon_anom['pres']
    mon_anom['pressstd'] = hourly_dat.groupby('month').std()['wspd']
    mon_anom = mon_anom.drop(['temp','wpgt','tsun','snow','time','week','dtime','prcp','wspd','pres'], axis=1)

    ## Calc anomalies within hourly_dat dataframe

    hourly_dat['weekly_T_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['temp'],anom,x['week']),axis=1)
    hourly_dat['monthly_T_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['temp'],mon_anom,x['month']),axis=1)
    hourly_dat['weekly_Prec_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['prcp'],anom,x['week'],label='prec'),axis=1)
    hourly_dat['monthly_Prec_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['prcp'],mon_anom,x['month'],label='prec'),axis=1)
    hourly_dat['weekly_Wind_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['wspd'],anom,x['week'],label='wspd'),axis=1)
    hourly_dat['monthly_Wind_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['wspd'],mon_anom,x['month'],label='wspd'),axis=1)
    hourly_dat['weekly_Pressure_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['pres'],anom,x['week'],label='press'),axis=1)
    hourly_dat['monthly_Pressure_anom'] = hourly_dat.apply(lambda x: calc_normalization(x['pres'],mon_anom,x['month'],label='press'),axis=1)

    ### Qualitative Columns

    hourly_dat['snowing'] = 0
    hourly_dat['raining'] = 0
    hourly_dat['hail'] = 0
    hourly_dat['cloudy'] = 0

    hourly_dat.loc[((hourly_dat['coco'] < 17) & (hourly_dat['coco'] > 13)) | ((hourly_dat['coco'] <= 22) & (hourly_dat['coco'] >= 21)), 'snowing'] = 1
    hourly_dat.loc[((hourly_dat['coco'] < 10) & (hourly_dat['coco'] > 6)) | ((hourly_dat['coco'] < 19) & (hourly_dat['coco'] >= 17)), 'raining'] = 1
    hourly_dat.loc[((hourly_dat['coco'] < 14) & (hourly_dat['coco'] > 9)) | ((hourly_dat['coco'] < 21) & (hourly_dat['coco'] >= 19)), 'hail'] = 1
    hourly_dat.loc[((hourly_dat['coco'] < 7) & (hourly_dat['coco'] > 2)), 'cloudy'] = 1

    ##### Dropping unneeded columns

    hourly_dat = hourly_dat.drop(['snow','tsun','time','wpgt'],axis=1)

    ##### Save (if savefile parameter not an empty string, then save)

    if savefile:
        hourly_dat.to_csv('%s.csv' % savefile)
    
    return hourly_dat

