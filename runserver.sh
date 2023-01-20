#!/bin/sh
python manage.py runserver 0.0.0.0:8000 &
celery -A livis.celeryy worker --loglevel=info &
python /critical_data -m http.server 3306 &
