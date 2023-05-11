import logging

from flask import Flask
from flask_bootstrap import Bootstrap
from flaskwebgui import FlaskUI
from mpi4py import MPI

from config import Config

logging.basicConfig(filename=f'cache/loggs/main_loggs.log', filemode='w', level=logging.INFO)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config.from_object(Config)
bootstrap = Bootstrap(app)
ui = FlaskUI(app, browser_path="C:\Program Files\Google\Chrome\Application\chrome.exe", port=5000, width=500, height=500)


from appfc import routes, forms, services, errors


if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    # app.run()
    ui.run()
