#!/usr/bin/env bash
python ./server/manage.py makemigrations ins
python ./server/manage.py migrate ins
python ./server/manage.py --setting=server.settings.prod 0.0.0.8080