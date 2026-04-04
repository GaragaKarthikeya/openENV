const _name = "O'zbek";
const common = { "built_with_gradio": "Gradio bilan yaratilgan", "clear": "Tozalash", "submit": "Yuborish", "built_with": "Bilan yaratilgan", "download": "Yuklab olish", "edit": "Tahrirlash", "empty": "Bo'sh", "error": "Xato", "hosted_on": "Joylashtirilgan", "loading": "Yuklanmoqda", "logo": "Logo", "or": "yoki", "remove": "O'chirish", "settings": "Sozlamalar", "share": "Ulashish", "undo": "Bekor qilish", "no_devices": "Hech qanday qurilma topilmadi", "language": "Til", "display_theme": "Ko'rinish mavzusi", "pwa": "Progressive web-ilova" };
const upload_text = { "click_to_upload": "Yuklash uchun bosing", "drop_audio": "Audioni shu yerga tashlang", "drop_csv": "CSV faylni shu yerga tashlang", "drop_file": "Faylni shu yerga tashlang", "drop_image": "Rasmni shu yerga tashlang", "drop_video": "Videoni shu yerga tashlang", "drop_gallery": "Mediafaylni shu yerga tashlang", "paste_clipboard": "Klipboarddan qo'yish" };
const annotated_image = { "annotated_image": "Izohli rasm" };
const audio = { "allow_recording_access": "Iltimos, yozib olish uchun mikrofonga ruxsat bering.", "audio": "Audio", "drop_to_upload": "Audio faylni yuklash uchun shu yerga tashlang", "record_from_microphone": "Mikrofondan yozib olish", "stop_recording": "Yozib olishni to'xtatish", "no_device_support": "Media qurilmalariga kira olmadik. Xavfsiz manbada (https) yoki localhost-da ishlayotganingizni tekshiring (yoki to'g'ri SSL sertifikatini ssl_verify-ga berganingizni) va brauzer qurilmangizga kirishiga ruxsat berganingizni tekshiring.", "stop": "To'xtatish", "resume": "Davom ettirish", "record": "Yozib olish", "no_microphone": "Mikrofon topilmadi", "pause": "Pauza", "play": "Ijro etish", "waiting": "Kutilmoqda" };
const blocks = { "connection_can_break": "Mobil qurilmada, agar bu yorliq fokusdan chiqsa yoki qurilma uyqu rejimiga o'tsa, aloqa uzilib, navbatdagi o'rningizni yo'qotishingiz mumkin.", "long_requests_queue": "So'rovlar navbati juda uzun. Navbatni chetlab o'tish uchun bu Space-ni nusxalang.", "lost_connection": "Sahifadan chiqib ketganligi sababli aloqa uzildi. Navbatga qayta qo'shilmoqda...", "waiting_for_inputs": "Fayl(lar) yuklanib bo'lishini kutish, iltimos, qayta urinib ko'ring." };
const checkbox = { "checkbox": "Belgilash katagi", "checkbox_group": "Belgilash kataklari guruhi" };
const code = { "code": "Kod" };
const color_picker = { "color_picker": "Rang tanlagich" };
const dataframe = { "incorrect_format": "Noto'g'ri format, faqat CSV va TSV fayllari qo'llab-quvvatlanadi", "new_column": "Ustun qo'shish", "new_row": "Yangi qator", "add_row_above": "Yuqoriga qator qo'shish", "add_row_below": "Pastga qator qo'shish", "delete_row": "Qatorni o'chirish", "delete_column": "Ustunni o'chirish", "add_column_left": "Chapga ustun qo'shish", "add_column_right": "O'ngga ustun qo'shish", "sort_column": "Ustunni saralash", "sort_ascending": "O'sish tartibida saralash", "sort_descending": "Kamayish tartibida saralash", "drop_to_upload": "Ma'lumotlarni ma'lumotlar jadvaliga import qilish uchun CSV yoki TSV fayllarini shu yerga tashlang", "clear_sort": "Saralashni tozalash" };
const dropdown = { "dropdown": "Ochiluvchi ro'yxat" };
const errors = { "build_error": "Tuzilish xatosi mavjud", "config_error": "Konfiguratsiya xatosi mavjud", "contact_page_author": "Iltimos, sahifa muallifiga bog'laning va ularga xabar bering.", "no_app_file": "Ilova fayli yo'q", "runtime_error": "Bajarilish vaqti xatosi mavjud", "space_not_working": '"Space ishlamayapti, chunki" {0}', "space_paused": "Space to'xtatilgan", "use_via_api": "API orqali foydalaning", "use_via_api_or_mcp": "API yoki MCP orqali foydalaning" };
const file = { "uploading": "Yuklanmoqda..." };
const highlighted_text = { "highlighted_text": "Ajratilgan matn" };
const image = { "allow_webcam_access": "Iltimos, yozib olish uchun veb-kameraga ruxsat bering.", "brush_color": "Cho'tka rangi", "brush_radius": "Cho'tka o'lchami", "image": "Rasm", "remove_image": "Rasmni o'chirish", "select_brush_color": "Cho'tka rangini tanlang", "start_drawing": "Chizishni boshlang", "use_brush": "Cho'tkadan foydalaning", "drop_to_upload": "Rasm faylini yuklash uchun shu yerga tashlang" };
const label = { "label": "Yorliq" };
const login = { "enable_cookies": "Agar HuggingFace Space-ni Inkognito rejimida ko'rayotgan bo'lsangiz, uchinchi tomon kukilarini yoqishingiz kerak.", "incorrect_credentials": "Noto'g'ri kirish ma'lumotlari", "username": "Foydalanuvchi nomi", "password": "Parol", "login": "Kirish" };
const number = { "number": "Raqam" };
const plot = { "plot": "Grafik" };
const radio = { "radio": "Radio tugma" };
const slider = { "slider": "Slayder" };
const video = { "drop_to_upload": "Video faylni yuklash uchun shu yerga tashlang" };
const chatbot = { "edit": "Tahrirlash", "retry": "Qayta urinish", "undo": "Bekor qilish", "submit": "Yuborish", "cancel": "Bekor qilish", "like": "Yoqdi", "dislike": "Yoqmadi", "clear": "Suhbatni tozalash" };
const uz = {
  _name,
  common,
  upload_text,
  "3D_model": { "3d_model": "3D model", "drop_to_upload": "3D model (.obj, .glb, .stl, .gltf, .splat yoki .ply) faylni yuklash uchun shu yerga tashlang" },
  annotated_image,
  audio,
  blocks,
  checkbox,
  code,
  color_picker,
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
  video,
  chatbot
};

export { _name, annotated_image, audio, blocks, chatbot, checkbox, code, color_picker, common, dataframe, uz as default, dropdown, errors, file, highlighted_text, image, label, login, number, plot, radio, slider, upload_text, video };
//# sourceMappingURL=uz-CKVFBI7d.js.map
