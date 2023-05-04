import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'some-key'
    FLASK_APP = 'app/__init__.py'
    FLASK_ENV = 'development'
    FLASK_DEBUG = 0
