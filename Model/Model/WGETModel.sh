#!/bin/bash

#One-time

DIR = Model\Model

FILE = dailyinfo.'date + "%Y%m%d"'

LOGFILE = wget.log

cd $DIR
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LbCOiT5lssUDatt44YxElmwbyXZoTjU2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LbCOiT5lssUDatt44YxElmwbyXZoTjU2" -O Model/Model/TextModel.pt && rm -rf /tmp/cookies.txt