#!/usr/bin/env bash
ls
#cd meterReader
python ./server/manage.py makemigrations ins --setting=server.settings.prod
python ./server/manage.py migrate ins --setting=server.settings.prod
python ./server/manage.py runserver 0.0.0.0:8080 --setting=server.settings.prod