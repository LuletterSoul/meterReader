#!/usr/bin/env bash
ls
cd meterReader
python ./server/manage.py makemigrations makemigrations ins --setting=server.settings.prod
python ./server/manage.py migrate ins --setting=server.settings.prod
python ./server/manage.py --setting=server.settings.prod 0.0.0.8080