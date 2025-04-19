import cv2

# Координаты светофора
traffic_light_green = (364, 215)  # Зеленый свет
traffic_light_red = (365, 206)  # Красный свет

def get_traffic_light_color(frame, last_status):
    """ Определение цвета светофора по спектру пикселей. """
    red_pixel = frame[traffic_light_red[1], traffic_light_red[0]]
    green_pixel = frame[traffic_light_green[1], traffic_light_green[0]]

    new_status = last_status  # По умолчанию используем прошлый статус

    if red_pixel[2] > 150: # 0-2: B,G,R
        return "red"
    if green_pixel[1] > 120:
        return "green"

    return new_status

if __name__ == '__main__':
    cap = cv2.VideoCapture("videosource/sochi7(5am).mp4")
    traffic_light_status = "green"  # Начальное состояние

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        # Определяем текущий цвет светофора
        traffic_light_status = get_traffic_light_color(frame, traffic_light_status)

        # Отображение текущего статуса светофора
        cv2.putText(frame, f"Traffic Light: {traffic_light_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if traffic_light_status == "green" else (0, 0, 255), 2)

        # Показываем результат
        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()