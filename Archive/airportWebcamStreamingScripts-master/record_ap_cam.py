#!/usr/bin/env python3
import re
import time
import subprocess
import signal
import os, sys
from pathlib import Path
from datetime import datetime
import pytz

"""
Dependencies:
	ffmpeg
	youtube-dl: "pip install youtube-dl"

Supported arguments:
	-all	: Record all airport streams simultaneously
		  Are you sure you want to do this...?!
	-o	: Add airport and record time overlay to top left of output
		  Higher processing load - best not to use for 4K
"""


plsFile = "inputs/good-airport-streams-playlist.pls"
segment = "01:00:00"
fmt = "mkv"

#Get system arguments
args = sys.argv[1:]

def get_info( plsFile ):
    with open(plsFile) as f:
        data = f.read()
        #Split block text into list of lines
        lines = data.split( '\n' )
        #Remove lines with Length parameter
        lines = [x for x in lines if "Length" not in x]
        #Remove commented lines
        lines = [x for x in lines if not x.startswith("#")]
        #Remove empty elements
        lines = [x for x in lines if x]
        
        files = [x.split('=', maxsplit=1)[1] for x in lines if "File" in x]
        titles = [x.split('=', maxsplit=1)[1] for x in lines if "Title" in x]

        #Replace youtube links with youtube-dl input as hls instead
        for i, f in enumerate(files):
            if "youtube" in f:
                files[i] = "\"$(youtube-dl -f best -g \"" + f + "\")\""

        return files, titles

def get_output_details(files, titles):
    #Print user-selection list if not for all airports
    if "-all" not in args:
        print( "Airports:" )
        for i, ap in enumerate(titles, 1):
            print( "\t" + str(i) + ": " + ap )

        sel = int( input( "Select airport stream to record: " ) )

        dirname = [ titles[sel - 1].replace(",", "").split(" ")[0] ]
        dirpath = [ "output/" + dirname[0] ]
        url = [ files[sel - 1] ]

        #Create output directory
        Path( dirpath[0] ).mkdir(parents=True, exist_ok=True)

        airport = [ dirname[0] + " Airport" ]
        fname = get_valid_filename(titles[sel - 1])
        outFile = [ dirpath[0] + '/' + fname + "_" ]
        #outFile = [ dirpath[0] + '/' + "test" + '.' + fmt ]
        #outFile = [ "test" + '.' + fmt ]
    else:
        dirname = [ x.replace(",", "").split(" ")[0] for x in titles ]
        dirpath = [ "output/" + x for x in dirname ]
        url = files

        #Create output directory
        for p in dirpath:
            Path( p ).mkdir(parents=True, exist_ok=True)

        airport = [ x + " Airport" for x in dirname ]
        fname = [ get_valid_filename(x) for x in titles ]
        outFile = []
        for p, f in zip( dirpath, fname ):
            outFile.append( p + '/' + f + "_"  )

    return dirname, dirpath, url, airport, outFile

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

#Get stream info from playlist file
files, titles = get_info( plsFile )

#Create output paths and directories
dirname, dirpath, url, airport, outFile = get_output_details( files, titles )

count = 1
while True:
    cmds = []
    for stream, aport, ofile in zip( url, airport, outFile ):
        #Get start date and time
        now = datetime.now()
        utc_now = pytz.utc.localize(datetime.utcnow())
        #now = utc_now.astimezone(pytz.timezone("Europe/London"))
        #now = utc_now.astimezone(pytz.timezone("Europe/Berlin"))
        dtime = now.strftime("%d-%m-%Y_%Hh%Mm%Ss")

        ofile = ofile + dtime + '.' + fmt

        inputArgs = "-nostats -loglevel info -threads 1 "
        if "http" in stream:
            inputArgs += "-headers \"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0\""
        elif "rtsp" in stream:
            inputArgs += "-rtsp_transport tcp"	#Fix for video smearing issue with udp
        #Overlay Airport and conversion time-stamp directly into output
        if "-o" not in args:
            cmds.append( "ffmpeg " + inputArgs + " -y " + " -i " + stream + " -movflags +faststart -map_metadata 0 -t " + segment + " -c:v copy -c:a copy " + ofile )
        else:
            #Reference for time formats:
            #https://man7.org/linux/man-pages/man3/strftime.3.html
            tz = pytz.timezone("Europe/London")
            dt = datetime.utcnow()
            offset = tz.utcoffset( dt ).seconds
            dtime = str( now.timestamp() + offset )	#Time in epoch format

            textOverlay = "-vf drawtext=\"fontfile=inputs/Raleway/static/Raleway-Light.ttf: fontsize=h*0.05: box=1: boxcolor=black@0.6: boxborderw=5: fontcolor=white: x=10: y=10: text=\'" + aport + " %{pts\:gmtime\:" + dtime + "\:" + r"%H\\\\\:%M\\\\\:%S" + " GMT}'\" "
            cmds.append( "ffmpeg " + inputArgs + " -fflags +genpts -y " + " -i " + stream + ' ' + textOverlay + " -movflags +faststart -map_metadata 0 -c:v libx264 -t " + segment + ' ' + ofile )
            #cmd = "ffmpeg -nostats -loglevel 0 " + framefix + " -i " + stream + ' ' + textOverlay + " -vcodec libx264 -t " + segment + " -movflags +faststart " + framefix + ' ' + ofile

    #[ print(i) for i in cmds ]
    #subprocess.Popen( ["echo", "\"hello\""] )
    #quit()
    procs = [ subprocess.Popen(i, shell=True) for i in cmds ]
    try:
        for p in procs:
           p.wait()
    except KeyboardInterrupt:
        end = [ os.kill(proc.pid, signal.SIG_DFL ) for proc in procs ]
        sys.exit(0)

    ##cmd = " | ".join( cmds )
    ##os.system( cmd )
    #print( cmd )
    #break
    count += 1
    #if count > 1:
    #    break
    #time.sleep(1)
