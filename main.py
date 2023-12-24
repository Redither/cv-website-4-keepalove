import os
from flask import Flask, request, render_template
# from flask_cors import CORS

application = Flask(__name__)
# CORS(application) 

@application.route("/")
@application.route("/home")
def hello():
    return render_template('index.html')

# Обработчик для эндпоинта
@application.route('/api/detect', methods=['POST'])
def submit_form():

    # return response
    return 

# Запускаем сервер
if __name__ == '__main__':
    application.run(host="0.0.0.0", port="5000", debug=True)