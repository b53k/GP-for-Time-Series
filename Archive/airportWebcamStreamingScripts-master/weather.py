import avwx
import time
import os, sys
from datetime import datetime, timedelta
import lxml.etree as etree

##Bryn note:
#METAR = Meteorological Aerodrone Reports, standard format - has current surface conditions
#        at an airport which is updated every hour, used as standard throughout ICAO
#Data source is: NOAA (National Oceanic & Atmospheric Adminstration - US govt)

##Understanding METAR data:
#Q1013: Pressure
#Can find list of codes here:
#http://www.moratech.com/aviation/metaf-abbrev.html

#icao = "EDDL" ##Dusseldorf Airport
#icao = "EGLL" ##Heathrow Airport
icao = "EPGD" ##Gdansk Airport

metars = avwx.Metar( icao )
metars.update()
#print( metars.summary )
#print( dir(metars) )
print( metars.data )
#print( metars.data.flight_rules )
#print( metars.raw )
#print("Updated time: ", metars.data.time.dt)

outFile='output/weather.xml'

#Initialise root
root = etree.Element('data')
root.append(etree.Comment("source: NOAA"))
airport = etree.SubElement(root, 'aiport')

##Store airport information
#-------------------------
airport.set( 'icao', metars.station.icao )
airport.set( 'iata', metars.station.iata )
airport.set( 'elevation_m', str(metars.station.elevation_m) )
airport.set( 'elevation_ft', str(metars.station.elevation_ft) )
airport.set( 'name', metars.station.name )
airport.set( 'city', metars.station.city )
airport.set( 'country', metars.station.country )
airport.set( 'lat', str(metars.station.latitude) )
airport.set( 'lon', str(metars.station.longitude) )
airport.set( 'reporting', str(metars.station.reporting) )
airport.set( 'state', metars.station.state )
airport.set( 'type', metars.station.type )

for i, rways in enumerate(metars.station.runways, 1):
    elem = etree.SubElement(airport, 'runway_' + str(i) )
    for keys, values in rways.__dict__.items():
        elem.set( keys, str(values) )


##Store weather information
#--------------------------

timezone = metars.data.time.dt.tzinfo
prev_time = datetime.now(timezone) - timedelta(days=3*365)

try:
    while True:
        metars.update()
        #Store only if new data is available
        if metars.data.time.dt > prev_time:
            metar = etree.SubElement(airport, 'METAR')
            metar.set( "raw", metars.raw )
            metar.set( "summary", metars.summary )
            metar.set( "utc_time", metars.data.time.dt.strftime("%d/%m/%Y, %H:%M:%S") )
            metar.set( "temperature", str(metars.data.temperature.value) )
            metar.set( "dew_point", str(metars.data.dewpoint.value) )
            metar.set( "pressure", str(metars.data.altimeter.value) )

            wind = etree.SubElement( metar, 'wind' )
            wind.set( "speed", str(metars.data.wind_speed.value) )
            wind.set( "direction", str(metars.data.wind_direction.value) )

            visib = etree.SubElement( metar, 'visibility' )
            visib.set( "range", str(metars.data.visibility.value) )
            visib.set( "spoken", metars.data.visibility.spoken )

            #If cloud data is available
            #print( metars.data.clouds[0] )
            if len( metars.data.clouds ) > 0:
                #print( metars.data.clouds[0].__dict__ )
                #quit()
                for i, cloud in enumerate(metars.data.clouds, 1):
                    elem = etree.SubElement( metar, 'cloud_' + str(i) )
                    elem.set( "type", cloud.type )
                    elem.set( "altitude", str(cloud.base*100) )	#Height of clouds Above Ground Level (AGL) in feet

            metar.set( "remarks", metars.data.remarks )

            prev_time = metars.data.time.dt
            print( prev_time, metars.summary, '\n' )
        time.sleep( 60*10 )	#check every half an hour

    string = etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
    with open(outFile, 'wb') as f:
        f.write(string)
except KeyboardInterrupt:
    string = etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
    with open(outFile, 'wb') as f:
        f.write(string)
    sys.exit(0)
