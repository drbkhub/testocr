Кратко:

- YoloDocLayout - для распознавания макета страницы (картинки, таблицы, параграфы, формулы и прочее)

- EasyOCR используется для распознавания текста 

- Получаем изображение страницы как png с помощью Pdfium

- сопоставляем данные макета с распознанным текстом

- отсортировал сверху вниз


Больше заморачиваться не стал так как требует больше времени (на это 2.5 часа)


# Запуск
`python run.py --file путь_к_файлу_pdf --index индекс_страницы --langs en ru`


# Результат

result.json - данные

result.png - визуализация

