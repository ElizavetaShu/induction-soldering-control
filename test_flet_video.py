import flet as ft
import cv2
import base64


def main(page: ft.Page):
    page.title = "Test Video"
    page.window.width = 800
    page.window.height = 600

    # Открываем видео
    cap = cv2.VideoCapture("D:/VM/1.mp4")

    img = ft.Image(
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        width=640,
        height=480
    )

    page.add(img)

    frame_count = 0

    def update_frame(e=None):
        nonlocal frame_count
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        # Обновляем
        img.src = f"data:image/jpeg;base64,{img_b64}"
        page.update()

        frame_count += 1
        if frame_count < 100:
            page.run_task(update_frame, 0.033)  # 30 FPS

    page.run_task(update_frame)


ft.run(main)