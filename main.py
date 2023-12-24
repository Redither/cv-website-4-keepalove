import os

import detection

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


application = Flask(__name__)  # Создаем приложение фласк
application.config['SECRET_KEY'] = 'somesecretkey' # Создаем секретный ключ для работы апи
application.config['UPLOAD_FOLDER'] = 'Resources/uploaded' # Указываем папку для сохранения файлов из формы

# Создаем класс формы для загрузки изображений с клиента
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


# Создаем обработчик для корневого эндпоинта (+ домашняя страница)
@application.route("/", methods=['GET', 'POST'])
@application.route("/home", methods=['GET', 'POST'])
def hello():
    form = UploadFileForm() # Создаем образец формы
    if form.validate_on_submit():
        file = form.file.data # Получаем файл впервые
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), application.config['UPLOAD_FOLDER'], secure_filename(file.filename)))  # Сохраняем файл
        print('file has been uploaded to: ' + application.config['UPLOAD_FOLDER'] + '/' + secure_filename(file.filename))
        detection.start_image_object_detection(application.config['UPLOAD_FOLDER'] + '/' + secure_filename(file.filename))
        return 'finish'

    return render_template('index.html', form = form) # Рендерим страницу из шаблона, передаем в нее форму

# Запускаем сервер
if __name__ == '__main__':
    application.run(host="0.0.0.0", port="5000", debug=True)