# Всем привет в этом чате

## Представляю вашему вниманию лухари-сервис, способный распознавать объекты на фото

## Что он умеет?

### Распознавать до 80 объектов! А именно

```text
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
sofa
pottedplant
bed
diningtable
toilet
tvmonitor
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

### Реализовано оно при помощи

1. OpenCV YOLO для распознавания образов
2. Flask фреймворк для веб-клиента

## А ТЕПЕРЬ

### Прежде, чем запустить нужно

1. Создать виртуальное окружение
```python -m venv venv```  
2. Запустить виртуальное окружение
   Windows:

   ```shell
   venv\Scripts\Activate.bat
   ```

   Linux:

   ```shell
   source venv/bin/activate
   ```

3. Установить зависимости

```shell
pip install -r requirements.txt
```

### Теперь, при условии, что все установилось корректно, мы можем запустить проект

1. Запускаем основной исполняемый файл
```python main.py```
2. Открываем наш сайт кликом по ссылке в терминале или переходим по ссылке:
```http://127.0.0.1:5000/```
3. Нажимаем на кнопку `Choose file`, выбираем файл для распознавания

### WEEEE ARE THE CHAMPIONS, MY FRIEND

⠀⠀⠀⠀⠀⠀⢀⣴⡿⢿⣓⡲⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⣿⣿⣧⢿⣿⣷⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣄⢹⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠘⠿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣰⣿⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⠟⠃⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣷⣀⣣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣍⠉⢳⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⠿⣟⡉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣠⡾⢛⣿⣿⣿⠦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿⠿⣿⣃⣠⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⣿⣟⣴⣾⣿⣿⣇⣼⠷⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⠗⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣭⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⣿⣿⣿⣿⣷⣶⣷⢠⡄⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⠟⠛⠉⢀⣤⣶⣆⠈⠻⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⠟⢿⣟⣶⣾⣿⣿⣼⣿⣧⣲⣧⢠⣻⣶⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⢟⣴⣿⣿⣿⠷⣶⣿⣿⣿⣿⣤⣿⣿⣿⠿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣷⣷⣾⣿⠟⣿⣼⣿⣯⣭⡷⣄⡙⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⡟⠁⠀⠈⠻⣿⣿⣿⣷⣽⣟⣻⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠘⣿⣿⣿⣷⡞⣽⣿⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡍⠛⠛⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣟⡓⣾⣗⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡴⣂⠀⠀⠈⠉⢿⣹⠏⠈⠹⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⢀⣀⣀⣾⣟⣁⣼⠋⠀⠀⠀⠀⠀⢻⣦⠀⢠⠿⢻⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡴⠾⢁⣾⠶⣶⣿⠉⠈⠙⢛⠉⠀⠀⠀⠀⠀⠀⠀⠘⠿⠀⠀⢠⠾⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣟⣥⣾⣿⣿⣄⠉⠹⡄⠀⠀⠘⠷⣄⡀⠀⠀⡀⠀⠀⠀⠀⢀⡶⠉⠀⠀⠈⠳⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⣶⣟⡋⠀⠸⠛⣿⡿⢿⣷⣄⣇⠀⢠⣔⣠⣤⣽⣶⠴⠛⠀⠀⠀⣠⠟⠁⠀⠀⠀⢀⠀⠀⠘⠳⣄⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⡴⢻⣾⣿⣿⣿⡁⢀⠙⢧⣴⣉⣨⣿⣶⣿⣿⣟⠛⠋⠀⠀⠀⠀⢀⣰⠏⠀⠀⢀⡤⠚⠃⠀⠀⠀⠀⠙⠳⣄⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⡷⣿⣿⣿⣿⣿⡅⠀⣀⣤⣿⣿⣿⣿⣿⣿⡌⠉⠀⠀⠀⠀⠀⣰⠿⣇⣰⡶⠞⠋⠀⢀⣀⠀⠀⠀⠀⠀⠀⠹⡆⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⢲⣄⣀⣀⣠⠾⢿⠏⠙⠿⠳⢾⣽⡒⠒⣃⣀⣀⠀⠀⠀⠀⠀⢹⡀⠀⠀⠀
⠀⠀⠀⠀⠻⣿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢀⡾⠉⠉⠉⠁⣴⠏⠀⠀⠀⠀⣠⣬⠶⣿⠉⠉⠉⠉⠁⠀⠀⠀⠘⣷⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⣼⡁⠀⠀⠀⣰⠃⠀⠀⠀⢠⠚⠁⠀⠀⠈⢧⡀⠀⠀⠀⠀⠀⠀⠀⠈⡇⠀⠀
⠀⠀⠀⠀⢀⣉⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⡼⢻⠁⢀⡠⠊⢹⠀⠰⡇⠀⠀⠀⠀⠀⠀⠀⠈⠳⡀⠀⠀⠀⠀⣾⡿⠿⢿⠀⠀
⠀⠀⠀⣠⢾⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢐⣧⡾⠞⠋⢀⡴⢿⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀⢀⡀⠙⢧⡀⠀⠀⣷⠤⠀⠀⠀⠀
⠀⠀⠀⠈⠸⠏⠉⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⣼⠟⠀⣠⠶⠂⢀⣸⡄⠈⣿⠀⢠⡀⠀⠀⠀⠀⠀⠑⢄⡈⠻⣄⠀⠸⡇⠀⢱⣦⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⣸⠏⣶⠚⠁⢀⣠⠟⠁⢻⡄⠸⣆⠀⠻⣦⣄⡀⠀⠀⠀⠀⠙⢦⡙⣆⠀⢱⡀⠈⣿⡆
⠀⠀⠀⠀⠀⠀⢀⣀⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⢹⣷⠀⠐⣿⢀⣴⠞⠁⠀⠀⢀⡿⣦⣿⣄⠀⠀⠉⠛⠳⣦⣀⠀⠀⠀⢳⡜⣦⠈⣧⣀⡿⠁
⠀⠀⠀⠀⠀⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢸⠛⢧⣄⡼⠋⠁⠀⠀⣠⡶⠀⠁⠘⣿⣿⣦⣀⠀⢄⠀⠀⠁⠀⠀⠀⠀⢻⠸⡇⢹⡏⠁⠀
⠀⠀⠀⠀⠀⠉⠹⠿⠟⠻⣿⣿⣿⣿⣿⣿⣿⠃⢰⡀⠀⠁⠀⠀⢀⡴⠊⠁⠀⠀⢀⣰⠛⣿⠈⠻⣦⡀⠱⡄⠀⠀⠀⠀⠀⠀⣧⠀⠈⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⡿⠀⠀⠙⠛⢻⣷⡞⠉⠀⠀⠀⣠⠾⠋⠀⠀⠸⡆⠀⠈⠙⢦⡙⢦⡀⠀⠀⠀⠀⠸⡆⠀⢻⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣠⣿⣿⣿⣿⣿⠇⢀⣀⣀⣀⠸⠙⡇⠀⢀⠴⠋⠀⠀⠀⠀⠀⣀⣷⠀⠀⠀⠀⠀⠀⠻⣆⠀⠀⠀⢀⠁⠀⣼⠀⠀
⠀⠀⠀⠀⠀⡀⢠⣲⣶⣿⣿⣿⣿⣿⣿⣿⡀⠀⠈⠉⠉⡛⢧⣽⠞⠃⠀⠀⠀⠀⠀⠀⠈⠉⢿⡇⠀⠀⠀⠀⠀⠀⠘⢧⣀⡀⠙⡇⠀⢹⠀⠀
⠀⠀⡀⣴⣻⣶⣿⣿⣿⣿⣿⣿⠿⠛⠩⡿⢳⣄⡀⠐⠛⠛⠛⠃⠀⠀⠀⣀⠴⠆⠀⠀⠀⠀⣸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠀⣅⠀⢿⠀⠀
⠀⢰⣿⡿⣿⣯⣿⣿⡿⠛⠉⠀⠀⠀⠀⡇⠀⠛⢿⣦⡀⠀⠀⣠⡤⣶⡈⠀⠀⠀⠀⢀⡤⠞⠹⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡞⠀⠈⠀⠀
⢠⣾⣿⣷⣿⣟⠋⠁⠈⠀⠀⠀⠀⠀⢰⣷⣤⣄⠀⠈⠓⢤⣀⢻⡀⠈⠻⣄⣀⡤⠞⠋⠀⠀⢠⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣗⠀⠀⠀⠀
⠀⠈⠁⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠘⡆⠀⢙⡛⠶⢤⣄⣈⠁⠻⣆⣴⠟⠁⠀⠀⠀⠀⣀⠴⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⠘⢷⣍⠐⠶⠍⣉⢉⣒⣛⢣⡀⠀⣀⣀⠰⠊⠁⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠈⠉⢙⣿⡷⠶⠾⠛⠛⢿⣹⡟⠹⣦⠀⠀⣀⣼⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠘⠿⣶⣶⠾⠋⠉⠀⠀⠀⠀⠀⠺⣿⣳⣄⣸⣷⡞⠁⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣽⠋⠀⠰⣾⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
