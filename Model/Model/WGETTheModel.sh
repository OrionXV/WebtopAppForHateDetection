#!/bin/sh

# alvinalexander.com
# a shell script used to download a specific url.
# this is executed from a crontab entry every day.

DIR=Model\Model

# wget output file
FILE=dailyinfo.`date +"%Y%m%d"`

# wget log file
LOGFILE=wget.log

# wget download url
#URL=http://foo.com/myurl.html

cd $DIR
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LbCOiT5lssUDatt44YxElmwbyXZoTjU2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LbCOiT5lssUDatt44YxElmwbyXZoTjU2" -O Model/Model/TextModel.pt && rm -rf /tmp/cookies.txt