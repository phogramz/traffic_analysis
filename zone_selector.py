import cv2

# Список для хранения координат точек
points = []


def draw_points(event, x, y, flags, param):
    """Обработчик кликов мыши для добавления точек на видео."""
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Добавлена точка: ({x}, {y})")


if __name__ == "__main__":
    cap = cv2.VideoCapture("videosource/sochi7(5am).mp4")
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", draw_points)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Отрисовка точек
        for i, point in enumerate(points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
