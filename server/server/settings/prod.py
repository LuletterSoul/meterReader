from .base import *

DEBUG = True
ALLOWED_HOSTS = ['*']
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'ins',
        'USER': 'root',
        'PASSWORD': '15298382837LIUxia_',
        'HOST': 'mysql',
        'PORT': '3306',
    }
}
