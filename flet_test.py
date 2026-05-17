import flet as ft


def main(page: ft.Page):
    page.title = "Test"
    page.window.width = 800
    page.window.height = 600

    # Просто добавляем элементы БЕЗ Container/Column
    page.add(
        ft.Text("🟢 ИНТЕРФЕЙС РАБОТАЕТ!", size=30, color=ft.Colors.RED, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        ft.Text("Если вы видите этот текст — Flet работает"),
        ft.ElevatedButton("Нажми меня", on_click=lambda e: print("Клик!")),
        ft.Container(
            content=ft.Text("Белый блок"),
            width=400,
            height=300,
            bgcolor=ft.Colors.WHITE,
            border=ft.Border.all(3, ft.Colors.BLUE)
        )
    )


ft.run(main)