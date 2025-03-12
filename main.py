import numpy as np
import cv2
from ultralytics import YOLO
from short import Sort
import time

# Определение координат зоны пешеходного перехода
crosswalk_zone = [(242, 285), (420, 316), (462, 450), (239, 431)] #
# crosswalk_zone = [(168, 272), (420, 316), (462, 450), (140, 416)] # + зона за светофором
group_zone = [(10, 530), (8, 126), (414, 266), (884, 534)]
VCL_zone = [(8, 400), (4, 276), (516, 328), (944, 488)]  # Зона автомобилей

# Координаты светофора
traffic_light_green = (366, 215)  # Зеленый свет
traffic_light_red = (365, 205)  # Красный свет

# Счетчики людей, перешедших на разный свет
crossing_counter_green = 0
crossing_counter_red = 0
crossed_ids_green = set()  # ID людей, перешедших на зеленый
crossed_ids_red = set()  # ID людей, перешедших на красный

# Классы для детекции (YOLO)
PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # Автомобиль, мотоцикл, автобус, грузовик

# Функция для определения цвета светофора по пикселю
# Работает, анализируя цвет в заданной точке
# Предполагаем, что красный цвет имеет большее значение в красном канале, а зеленый - в зеленом

# def get_traffic_light_color(frame):
#     red_pixel = frame[traffic_light_red[1], traffic_light_red[0]]
#     green_pixel = frame[traffic_light_green[1], traffic_light_green[0]]
#
#     # Если в зеленой области больше зелёного канала, значит горит зелёный
#     if green_pixel[1] > green_pixel[2] and green_pixel[1] > green_pixel[0]:
#     # if green_pixel[1] > max(green_pixel[0], green_pixel[2]):
#         return "green"
#
#     # Если в красной области больше красного канала, значит горит красный
#     if red_pixel[2] > red_pixel[1] and red_pixel[2] > red_pixel[0]:
#     # if red_pixel[2] > max(red_pixel[0], red_pixel[1]):
#         return "red"
#
#     return "unknown"  # Если не удалось определить


def get_traffic_light_color(frame):
    """ Определение цвета светофора по спектру пикселей. """
    red_pixel = frame[traffic_light_red[1], traffic_light_red[0]]
    green_pixel = frame[traffic_light_green[1], traffic_light_green[0]]

    if red_pixel[2] > 240: # 0-2: B,G,R
        return "red"
    if green_pixel[1] > 150:
        return "green"
    return "unknown"

# def count_people_in_zone(tracks, zone): # старая реализация, фиксирующая переход по средней нижней точке
#     count = 0
#     for xmin, ymin, xmax, ymax, _ in tracks:
#         center_x = (xmin + xmax) // 2
#         center_y = ymax # человек в зоне, если нижняя средняя точка его box'а в зоне
#         if cv2.pointPolygonTest(np.array(zone, np.int32), (float(center_x), float(center_y)), False) >= 0:
#             count += 1
#     return count

def count_people_in_zone(tracks, zone):
    """ Подсчет количества людей в группе. """
    return sum(
        1 for xmin, ymin, xmax, ymax, _ in tracks
        if cv2.pointPolygonTest(np.array(zone, np.int32), ((xmin + xmax) / 2, ymax), False) >= 0
    )


def count_vehicles_in_vcl_zone(vehicles):
    """ Подсчет количества автомобилей в зоне VCL. """
    return sum(
        1 for xmin, ymin, xmax, ymax, _ in vehicles
        if any(cv2.pointPolygonTest(np.array(VCL_zone, np.int32), (x, y), False) >= 0
               for x in [xmin, xmax] for y in [ymin, ymax])
    )

if __name__ == '__main__':
    cap = cv2.VideoCapture("videosource/sochi1.mp4")
    model = YOLO("yolov8n.pt") # ("yolov8s.pt") - точнее, но fps в 2 раза меньше
    tracker = Sort()
    vehicle_tracker = Sort()
    prev_frame_time = 0  # Для вычисления FPS

    with open("crossings_log12.txt", "w") as log_file:
        log_file.write("Time | ID | Traffic Light | Group | Vehicles\n")

        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break

            # Определяем текущий цвет светофора
            traffic_light_status = get_traffic_light_color(frame)

            # Получаем результаты детекции
            results = model(frame, stream=True)

            person_boxes, vehicle_boxes = [], []
            for res in results:
                filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
                boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
                class_ids = res.boxes.cls.cpu().numpy()[filtered_indices].astype(int)  # Получаем ID классов

                # Фильтруем только людей (класс "person" с ID=0)
                # boxes = [box for i, box in enumerate(boxes) if class_ids[i] == 0]  # Только люди (для машин - or class_ids[i] == 2) # старая реализация

                for i, box in enumerate(boxes):
                    if class_ids[i] == PERSON_CLASS:
                        person_boxes.append(box)
                    elif class_ids[i] in VEHICLE_CLASSES:
                        vehicle_boxes.append(box)


                # Проверяем, есть ли хоть один человек на кадре
                # if len(boxes) > 0:
                #     tracks = tracker.update(np.array(boxes))
                #     tracks = tracks.astype(int)
                # else:
                #     tracks = np.array([])  # Если людей нет, создаем пустой массив

            person_tracks = tracker.update(np.array(person_boxes)) if person_boxes else np.array([])
            vehicle_tracks = vehicle_tracker.update(np.array(vehicle_boxes)) if vehicle_boxes else np.array([])

            vehicle_count = count_vehicles_in_vcl_zone(vehicle_tracks)

            for xmin, ymin, xmax, ymax, track_id in person_tracks.astype(int):
                center_x, center_y = (xmin + xmax) // 2, ymax

                if track_id not in crossed_ids_green | crossed_ids_red and \
                        cv2.pointPolygonTest(np.array(crosswalk_zone, np.int32), (float(center_x), float(center_y)),
                                             False) >= 0:

                    group_param = count_people_in_zone(person_tracks, group_zone)

                    timestamp = time.strftime('%H:%M:%S', time.localtime())

                    if traffic_light_status == "green":
                        crossing_counter_green += 1
                        crossed_ids_green.add(track_id)
                        print(
                            f"{timestamp} | ID: {track_id} | Светофор: Зеленый | Группа: {group_param} | Авто: {vehicle_count}")
                        log_file.write(f"{timestamp} | {track_id} | Зеленый | {group_param} | {vehicle_count}\n")
                    elif traffic_light_status == "red":
                        crossing_counter_red += 1
                        crossed_ids_red.add(track_id)
                        print(
                            f"{timestamp} | ID: {track_id} | Светофор: Красный | Группа: {group_param} | Авто: {vehicle_count}")
                        log_file.write(f"{timestamp} | {track_id} | Красный | {group_param} | {vehicle_count}\n")

                # Отображение ID человека
                # cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin - 10),
                #             fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                # Определяем координаты для квадрата с ID (внешний верхний правый угол рамки)
                # id_box_x1, id_box_y1 = xmax, ymin - 20
                # id_box_x2, id_box_y2 = xmax - 20, ymin

                # Отрисовка квадрата
                # cv2.rectangle(frame, (id_box_x1, id_box_y1), (id_box_x2, id_box_y2), (149,179,215), -5)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Отображение ID внутри квадрата
                # cv2.putText(frame, str(track_id), (id_box_x1 - 18, id_box_y2 - 5),
                #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(track_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                # Отображение рамки вокруг человека
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

                # Отображение средней точки человека
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # # Отображение зоны пешеходного перехода
            # cv2.polylines(frame, [np.array(crosswalk_zone, np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
            # # Отображение зоны пешеходного группы (GRP)
            # cv2.polylines(frame, [np.array(group_zone, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

            for xmin, ymin, xmax, ymax, _ in vehicle_tracks.astype(int):
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

            cv2.polylines(frame, [np.array(crosswalk_zone, np.int32)], True, (255, 255, 0), 2) # B,G,R
            cv2.polylines(frame, [np.array(group_zone, np.int32)], True, (0, 128, 255), 2) # B,G,R
            cv2.polylines(frame, [np.array(VCL_zone, np.int32)], True, (255, 0, 255), 2) # B,G,R

            # Вычисление FPS
            curr_frame_time = time.time()
            fps = 1 / (curr_frame_time - prev_frame_time)
            prev_frame_time = curr_frame_time

            # Отображение FPS на экране
            # cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Отображение текущего статуса светофора
            cv2.putText(frame, f"Traffic Light: {traffic_light_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if traffic_light_status == "green" else (0, 0, 255), 2)

            # Показываем результат
            cv2.imshow("YOLO Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
