# Chemistry

## Подготовка данных
Сырые данные для обучения приходят в виде больший csv-шек с кучей столбцов. Далее нам нужен gnuplot чтобы из этих csv-шек получить фазовые диограммы (подробности у физиков. Я на виртуалку с ubuntu устанавливал). В папке "gnuplot" лежат немного модифицированные файлы постоения диограмм. Помимо графика, они ещё сохраняют 4 файла с контурами каждой фазы. Эти 2 файла кидаем в папку с данными и запускаем really_all_region.sh. Скрипт filter.py отфильтрует только файлы контуров в нужное вам место и переименут. В нём нужно будет прописать пити до этих папок. Это будет обучающим/тестовым набором, путь до которого нужно будет указать в fit_models.ipynb.

В папке ZipData лежат архивы с сырыми данными на которых у нас происходило обучение и тестирование.

## Обучение
Обучение производится в соответствии с комментариями в файле fit_models.ipynb.

## Draw phases
Файл draw_phases.py запускает программу для рисования фазовой диограммы.

Для рисования нужно выбрать интсрумент pen(карандаш) или eraser(ластик) и выбрать его размер.

Можно выбрать одну из 4-х фаз для рисования. При переключении на другую фазу, неактивные фазы затеняются.

В поле file name можно ввести имя файла для сохранения маски фаз(кнопка save file) или загрузки ранее сохзранённого файла(кнопка load file).

Кнопка clear полностью очистит холст.

## Предсказание
draw_phases.py сохраняет свои фазы в виде матриц (4, 64, 64) в csv файле. Эти файлы можно использовать для предсказания параметров с помощью файла predict_params.py. Ему можно передать либо путь к одному файлу, либо к папке с файлами. Результат будет записан в файл.

## Requirements
Pillow, Numpy, Pytorch
