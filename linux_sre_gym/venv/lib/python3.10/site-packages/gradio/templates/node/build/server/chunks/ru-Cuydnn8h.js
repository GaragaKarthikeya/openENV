const _name = "Русский";
const annotated_image = { "annotated_image": "Изображение с аннотациями" };
const audio = { "allow_recording_access": "Пожалуйста, разрешите доступ к микрофону для записи.", "audio": "Аудио", "record_from_microphone": "Запись с микрофона", "stop_recording": "Остановить запись", "no_device_support": "Невозможно получить доступ к медиа-устройствам. Убедитесь, что вы работаете с безопасного источника (https) или localhost (или передали действительный SSL-сертификат в ssl_verify), и что вы разрешили браузеру доступ к вашему устройству.", "stop": "Стоп", "resume": "Продолжить", "record": "Запись", "no_microphone": "Микрофон не найден", "pause": "Пауза", "play": "Воспроизвести", "waiting": "Ожидание", "drop_to_upload": "Перетащите аудиофайл сюда для загрузки" };
const blocks = { "connection_can_break": "На мобильном устройстве соединение может прерваться, если эта вкладка потеряет фокус или устройство перейдет в спящий режим, что приведет к потере вашего места в очереди.", "long_requests_queue": "Существует длинная очередь ожидающих запросов. Создайте дубликат этого Space, чтобы пропустить очередь.", "lost_connection": "Соединение потеряно из-за ухода со страницы. Возвращение в очередь...", "waiting_for_inputs": "Ожидание завершения загрузки файла(ов), пожалуйста, повторите попытку." };
const checkbox = { "checkbox": "Флажок", "checkbox_group": "Группа флажков" };
const code = { "code": "Код" };
const color_picker = { "color_picker": "Выбор цвета" };
const common = { "built_with": "Создано с помощью", "built_with_gradio": "Создано с помощью Gradio", "clear": "Очистить", "download": "Скачать", "edit": "Редактировать", "empty": "Пусто", "error": "Ошибка", "hosted_on": "Размещено на", "loading": "Загрузка", "logo": "Логотип", "or": "или", "remove": "Удалить", "settings": "Настройки", "share": "Поделиться", "submit": "Отправить", "undo": "Отменить", "no_devices": "Устройства не найдены", "language": "Язык", "display_theme": "Тема оформления", "pwa": "Прогрессивное веб-приложение" };
const dataframe = { "incorrect_format": "Неверный формат, поддерживаются только файлы CSV и TSV", "new_column": "Добавить столбец", "new_row": "Новая строка", "add_row_above": "Добавить строку выше", "add_row_below": "Добавить строку ниже", "add_column_left": "Добавить столбец слева", "add_column_right": "Добавить столбец справа", "delete_row": "Удалить строку", "delete_column": "Удалить столбец", "sort_column": "Сортировать столбец", "sort_ascending": "Сортировать по возрастанию", "sort_descending": "Сортировать по убыванию", "drop_to_upload": "Перетащите файлы CSV или TSV сюда для импорта данных в таблицу данных", "clear_sort": "Очистить сортировку" };
const dropdown = { "dropdown": "Выпадающий список" };
const errors = { "build_error": "Произошла ошибка сборки", "config_error": "Произошла ошибка конфигурации", "contact_page_author": "Пожалуйста, свяжитесь с автором страницы, чтобы сообщить об этом.", "no_app_file": "Отсутствует файл приложения", "runtime_error": "Произошла ошибка выполнения", "space_not_working": '"Space не работает, потому что" {0}', "space_paused": "Space приостановлен", "use_via_api": "Использовать через API", "use_via_api_or_mcp": "Использовать через API или MCP" };
const file = { "uploading": "Загрузка..." };
const highlighted_text = { "highlighted_text": "Выделенный текст" };
const image = { "allow_webcam_access": "Пожалуйста, разрешите доступ к веб-камере для записи.", "brush_color": "Цвет кисти", "brush_radius": "Размер кисти", "image": "Изображение", "remove_image": "Удалить изображение", "select_brush_color": "Выбрать цвет кисти", "start_drawing": "Начать рисование", "use_brush": "Использовать кисть", "drop_to_upload": "Перетащите изображение сюда для загрузки" };
const label = { "label": "Метка" };
const login = { "enable_cookies": "Если вы посещаете Space HuggingFace в режиме инкогнито, вам необходимо включить сторонние куки.", "incorrect_credentials": "Неверные учетные данные", "username": "Имя пользователя", "password": "Пароль", "login": "Войти" };
const number = { "number": "Число" };
const plot = { "plot": "График" };
const radio = { "radio": "Переключатель" };
const slider = { "slider": "Ползунок" };
const upload_text = { "click_to_upload": "Нажмите для загрузки", "drop_audio": "Перетащите аудио сюда", "drop_csv": "Перетащите CSV сюда", "drop_file": "Перетащите файл сюда", "drop_image": "Перетащите изображение сюда", "drop_video": "Перетащите видео сюда", "drop_gallery": "Перетащите медиа сюда", "paste_clipboard": "Вставить из буфера обмена" };
const video = { "drop_to_upload": "Перетащите видеофайл сюда для загрузки" };
const chatbot = { "edit": "Редактировать", "retry": "Повторить", "undo": "Отменить", "submit": "Отправить", "cancel": "Отмена", "like": "Нравится", "dislike": "Не нравится", "clear": "Очистить чат", "copy_message": "Копировать сообщение", "copied_message": "Сообщение скопировано" };
const chat_interface = { "new_chat": "Новый чат", "message_placeholder": "Введите сообщение...", "additional_inputs": "Дополнительные поля", "chatbot": "Чат-бот", "conversation": "Разговор" };
const ru = {
  _name,
  "3D_model": { "3d_model": "3D-модель", "drop_to_upload": "Перетащите сюда 3D-модель (.obj, .glb, .stl, .gltf, .splat или .ply) для загрузки" },
  annotated_image,
  audio,
  blocks,
  checkbox,
  code,
  color_picker,
  common,
  dataframe,
  dropdown,
  errors,
  file,
  highlighted_text,
  image,
  label,
  login,
  number,
  plot,
  radio,
  slider,
  upload_text,
  video,
  chatbot,
  chat_interface
};

export { _name, annotated_image, audio, blocks, chat_interface, chatbot, checkbox, code, color_picker, common, dataframe, ru as default, dropdown, errors, file, highlighted_text, image, label, login, number, plot, radio, slider, upload_text, video };
//# sourceMappingURL=ru-Cuydnn8h.js.map
