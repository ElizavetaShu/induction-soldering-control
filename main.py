import os
import time
import cv2
import base64
import threading
import flet as ft
from ml_integration import import_model, inference_model, draw_gird, import_seg_model, inference_segmentation, \
    draw_timing_info
import torch

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "1"


class VideoCaptureHandler:
    def __init__(self, page):
        self.frame_skip = 2  # Обрабатывать каждый 2-й кадр (1 пропускаем)
        self.skip_counter = 0

        self.ui_update_counter = 0
        self.ui_update_rate = 3

        self.page = page
        self.cap = None
        self.running = False
        self.initialized = False
        self.mirror = False
        self.inference = False
        self.gride = False
        self.segmentation = False
        self.seg_model = None
        self.use_camera = True
        self.video_path = None
        self.camera_resolution = (640, 480)  # Меньше для стабильности

        # Простые кнопки
        self.start_btn = ft.ElevatedButton("▶ Start", on_click=self.start_capture, disabled=True)
        self.stop_btn = ft.ElevatedButton("⏹ Stop", on_click=self.stop_capture, disabled=True)
        self.select_video_btn = ft.ElevatedButton("📁 Видео", on_click=self._select_video_file)
        self.status = ft.Text("Загрузка...", color=ft.Colors.BLUE)

        # Простое изображение
        self.img = ft.Image(
            src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", # ← 1x1 прозрачный пиксель
            width=640,
            height=480,
            fit="contain"
        )

        # Запуск в фоне
        threading.Thread(target=self._init_models, daemon=True).start()

    def _init_models(self):
        try:
            self.model = import_model()
            self.classNames = self.model.names
            self.seg_model = import_seg_model("yolo11n-seg.pt")
            self.status.value = "✅ Готово. Нажмите Start"
            self.status.color = ft.Colors.GREEN
            self.start_btn.disabled = False
            self.page.update()
        except Exception as e:
            self.status.value = f"❌ Ошибка: {e}"
            self.page.update()

    def _select_video_file(self, e):
        def close_dlg(e):
            if path_input.value:
                # 1. Убираем лишние пробелы
                raw_path = path_input.value.strip()
                # 2. Заменяем обратные слэши Windows на прямые (безопасно для OpenCV)
                clean_path = raw_path.replace('\\', '/')

                # 3. Проверяем, существует ли файл физически
                if not os.path.exists(clean_path):
                    self.status.value = f"❌ Файл не найден!\nПроверьте путь: {clean_path}"
                    self.status.color = ft.Colors.RED
                else:
                    self.video_path = clean_path
                    self.use_camera = False
                    self.status.value = f"✅ Готово: {os.path.basename(clean_path)}"
                    self.status.color = ft.Colors.GREEN
                    self.initialized = True
                    self.start_btn.disabled = False

            dialog.open = False
            self.page.update()

        path_input = ft.TextField(
            label="Путь к видеофайлу",
            hint_text="Пример: C:/Users/mishz/Videos/test.mp4",
            expand=True
        )
        dialog = ft.AlertDialog(
            title=ft.Text("📂 Выбор видео"),
            content=ft.Column([path_input], tight=True),
            actions=[
                ft.Button("OK", on_click=close_dlg),
                ft.Button("Отмена", on_click=lambda e: setattr(dialog, 'open', False))
            ]
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def start_capture(self, e):
        print(" [DEBUG] Нажата кнопка Start")
        self.running = True
        self.stop_btn.disabled = False
        self.start_btn.disabled = True

        if self.use_camera:
            print("📷 [DEBUG] Открываем камеру...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ [DEBUG] Камера не открылась!")
                self._stop_with_error("❌ Камера не найдена или занята")
                return
            print("✅ [DEBUG] Камера открыта успешно")
        else:
            print(f"🎬 [DEBUG] Открываем видео: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("❌ [DEBUG] Видео не открылось!")
                self._stop_with_error("❌ Не удалось открыть видео")
                return
            print("✅ [DEBUG] Видео открыто успешно")

        self.status.value = "🎥 Идёт обработка..."
        self.status.color = ft.Colors.GREEN
        self.page.update()
        print("🚀 [DEBUG] Запускаем поток обработки")
        threading.Thread(target=self._process_frames, daemon=True).start()

    def _stop_with_error(self, msg):
        """Вспомогательный метод для безопасной остановки с ошибкой"""
        self.status.value = msg
        self.status.color = ft.Colors.RED
        self.running = False
        self.stop_btn.disabled = True
        self.start_btn.disabled = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.page.update()

    def _process_frames(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Устройство: {device}")

        frame_count = 0

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                if not self.use_camera:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1

            try:
                det_time_sec = 0.0
                seg_time_sec = 0.0

                if self.inference and hasattr(self, 'model'):
                    res = inference_model(self.model, self.classNames, frame, device=device)
                    if isinstance(res, (tuple, list)) and len(res) == 2:
                        frame, det_time_ms = res
                        det_time_sec = det_time_ms / 1000.0
                    else:
                        frame = res

                if self.segmentation and self.seg_model:
                    res = inference_segmentation(self.seg_model, self.classNames, frame, device=device)
                    if isinstance(res, (tuple, list)) and len(res) == 2:
                        frame, seg_time_ms = res
                        seg_time_sec = seg_time_ms / 1000.0
                    else:
                        frame = res

                frame = draw_timing_info(frame, det_time_sec, seg_time_sec)

                if self.gride:
                    frame = draw_gird(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 📦 Кодируем
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_b64 = base64.b64encode(buffer).decode('utf-8')

                # 🔥 ОБНОВЛЕНИЕ UI (РАДИКАЛЬНОЕ)
                if frame_count % 2 == 0:  # Каждые 2 кадра
                    try:
                        # ✅ Формируем полный Data URI
                        data_uri = f"data:image/jpeg;base64,{img_b64}"

                        # ✅ Обновляем существующий объект self.img
                        self.img.src = data_uri
                        self.img.update()

                    except Exception as ui_err:
                        print(f"⚠️ UI ошибка: {ui_err}")
                        import traceback
                        traceback.print_exc()

                if frame_count % 30 == 0:
                    h, w = frame.shape[:2]
                    print(f"✅ Кадр #{frame_count} | {w}x{h} | Det: {det_time_sec:.3f}s | Seg: {seg_time_sec:.3f}s")
                    try:
                        self.status.value = f"🎥 {device.upper()} | Det: {det_time_sec:.3f}s | Seg: {seg_time_sec:.3f}s"
                        self.status.color = ft.Colors.GREEN
                        self.page.update()
                    except:
                        pass

            except Exception as e:
                print(f"❌ Ошибка кадра: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(0.01)

        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        try:
            self.stop_btn.disabled = True
            self.start_btn.disabled = False
            self.status.value = "⏹ Остановлено"
            self.page.update()
        except:
            pass

    def stop_capture(self, e):
        self.running = False
        if self.cap:
            self.cap.release()
        self.stop_btn.disabled = True
        self.start_btn.disabled = False
        self.status.value = "⏹ Остановлено"
        self.page.update()

    def create_ui(self):
        return ft.Column([
            self.status,
            self.img,
            ft.Row([self.start_btn, self.stop_btn, self.select_video_btn], spacing=10),
            ft.Text("💡 Включите детекцию/сегментацию в настройках (⚙️)")
        ],
        alignment="center",                  # ✅ Строка вместо enum
        horizontal_alignment="center",       # ✅ БЫЛО: ft.CrossAxisAlignment (без .CENTER)
        spacing=15
        )

def main(page: ft.Page):
    page.title = "AI Camera"
    page.window.width = 800
    page.window.height = 700
    page.padding = 20

    handler = VideoCaptureHandler(page)

    # Простые настройки
    inference_sw = ft.Switch(label="Детекция", value=False)
    segment_sw = ft.Switch(label="Сегментация", value=False)

    def apply_settings(e):
        handler.inference = inference_sw.value
        handler.segmentation = segment_sw.value
        dlg.open = False
        page.update()

    dlg = ft.AlertDialog(
        title=ft.Text("Настройки"),
        content=ft.Column([inference_sw, segment_sw]),
        actions=[ft.ElevatedButton("OK", on_click=apply_settings)]
    )

    page.overlay.append(dlg)
    page.appbar = ft.AppBar(
        title=ft.Text("📹 AI Camera"),
        actions=[ft.IconButton(ft.Icons.SETTINGS, on_click=lambda e: setattr(dlg, 'open', True))]
    )

    page.add(handler.create_ui())


ft.run(main)