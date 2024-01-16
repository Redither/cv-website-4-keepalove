import os
import cv2
import numpy as np

# Подгружаем YOLO scales из файлом И подготавливаем сеть
net = cv2.dnn.readNetFromDarknet(
    os.path.join("Resources", "yolov4-tiny.cfg"), os.path.join("Resources", "yolov4-tiny.weights")
)
layer_names = net.getLayerNames()
out_layers_indexes = net.getUnconnectedOutLayers()
out_layers = [layer_names[index - 1] for index in out_layers_indexes]

# Грузим из файла объектов classes которые YOLO Может обнаружить
with open(os.path.join("Resources", "coco.names.txt"), "r", encoding="utf-8") as file:
    classes = file.read().split("\n")

# Названия классов
classes_to_look_for = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def apply_yolo_object_detection(image_to_process):
    """Распознаёт и определяет координаты объектов на изображении

    Args:
        image_to_process (_type_): исходное изображение

    Returns:
        _type_: изображение с отмеченными объектами и подписями к ним
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Запуск поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Отбор
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # Для отладки рисуем объекты, входящие в нужные классы
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process, class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box):
    """Рисует границы объекта с надписыми

    Args:
        image_to_process (_type_): исходное изображение
        index (_type_): индекс класса объекта, определенного с помощью YOLO
        box (_type_): координаты области вокруг объекта

    Returns:
        _type_: изображение с отмеченными объектами
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count):
    """подписывает количество найденных объектов на изображении

    Args:
        image_to_process (_type_): исходное изображение
        objects_count (_type_): количество объектов нужного класса

    Returns:
        _type_: изображение с указанием количества найденных объектов
    """

    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    # Вывод текста штрихом
    # (чтобы было видно при разном освещении снимка)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(
        image_to_process, text, start, font, font_size, black_outline_color, width * 3, cv2.LINE_AA
    )
    final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width, cv2.LINE_AA)

    return final_image


def start_image_object_detection(img_src: str, img_dst: str):
    """Запускает процесс анализа изображений

    Args:
        img_src (str): путь к исходному изображению
        img_dst (str): путь, куда сохранить распознанное изображение
    """
    try:
        # Применение методов распознавания объектов на изображении от YOLO
        image = cv2.imread(img_src)
        image = apply_yolo_object_detection(image)
        print(f"Recognized image: {img_dst}")
        cv2.imwrite(img_dst, image)

        # Вывод обработанного изображения на экран
        # cv2.imshow("Image", image)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
