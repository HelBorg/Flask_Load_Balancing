import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'some-key-2'
    FLASK_APP = 'appfc/__init__.py'
    FLASK_ENV = 'development'
    FLASK_DEBUG = 0
