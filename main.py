import os

import detection

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


# Приложение Flask
application = Flask(__name__)

# Секретный ключ для работы апи
application.config["SECRET_KEY"] = "somesecretkey"

# Папка со всеми изображениями и шаблонами
application.config["STATIC_FOLDER"] = "static"

# Папка для сохранения файлов из формы
application.config["UPLOADED_FOLDER"] = os.path.join(application.config["STATIC_FOLDER"], "uploaded")

# Папка для сохранения распознанных изображений
application.config["RESULT_FOLDER"] = os.path.join(application.config["STATIC_FOLDER"], "result")


# Создаем класс формы для загрузки изображений с клиента
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


# Создаем обработчик для корневого эндпоинта (+ домашняя страница)
@application.route("/", methods=["GET", "POST"])
@application.route("/home", methods=["GET", "POST"])
def home() -> None:
    form = UploadFileForm()  # Создаем образец формы
    page_img = os.path.join(application.config["STATIC_FOLDER"], "camera.png")
    if form.validate_on_submit():
        # Получаем файл
        file = form.file.data

        # Создаём папки, если не существуют
        if not os.path.exists(application.config["UPLOADED_FOLDER"]):
            os.makedirs(application.config["UPLOADED_FOLDER"])
        if not os.path.exists(application.config["RESULT_FOLDER"]):
            os.makedirs(application.config["RESULT_FOLDER"])

        # Сохраняем файл
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                application.config["UPLOADED_FOLDER"],
                file.filename,
            )
        )
        print(f"Uploaded to: {os.path.join(application.config['UPLOADED_FOLDER'], secure_filename(file.filename))}")

        # Запускаем процесс распознавания
        page_img = os.path.join(application.config["RESULT_FOLDER"], secure_filename(file.filename))
        detection.start_image_object_detection(
            os.path.join(application.config["UPLOADED_FOLDER"], secure_filename(file.filename)),
            page_img,
        )

        # Рендерим страницу из шаблона, передаем в нее форму
        return render_template("index.html", form=form, image=page_img)

    # Рендерим страницу из шаблона, передаем в нее форму
    return render_template("index.html", form=form, image=page_img)


# Запускаем сервер
if __name__ == "__main__":
    application.run(host="0.0.0.0", port="5000", debug=True)
