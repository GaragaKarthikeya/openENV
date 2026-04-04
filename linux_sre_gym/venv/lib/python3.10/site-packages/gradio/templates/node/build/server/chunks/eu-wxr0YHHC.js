const _name = "Euskara";
const annotated_image = { "annotated_image": "Irudi Erantsia" };
const audio = { "allow_recording_access": "Mesedez, baimendu mikrofonoaren sarbidea grabatzeko.", "audio": "Audioa", "record_from_microphone": "Mikrofonotik grabatu", "stop_recording": "Grabaketa gelditu", "no_device_support": "Ezin izan da multimedia gailuetara sartu. Ziurtatu jatorri seguru batean (https) edo localhost-en exekutatzen ari zarela (edo SSL ziurtagiri baliozkoa eman diozula ssl_verify-ri), eta nabigatzaileari zure gailura sarbidea eman diozula.", "stop": "Gelditu", "resume": "Jarraitu", "record": "Grabatu", "no_microphone": "Ez da mikrofonorik aurkitu", "pause": "Pausatu", "play": "Erreproduzitu", "waiting": "Itxaroten", "drop_to_upload": "Arrastatu audio fitxategia hemen igotzeko" };
const blocks = { "connection_can_break": "Mugikorretan, konexioa eten daiteke fitxa honek fokua galtzen badu edo gailua lokartu egiten bada, ilaran duzun posizioa galduz.", "long_requests_queue": "Eskaera ilara luzea dago zain. Space hau bikoiztu saltatzeko.", "lost_connection": "Konexioa galdu da orria uzteagatik. Ilarara itzultzen...", "waiting_for_inputs": "Itxaron fitxategiak igo arte, saiatu berriro mesedez" };
const checkbox = { "checkbox": "Kontrol-laukia", "checkbox_group": "Kontrol-laukien taldea" };
const code = { "code": "Kodea" };
const color_picker = { "color_picker": "Kolore hautatzailea" };
const common = { "built_with": "honekin eraikia", "built_with_gradio": "Gradio-rekin eraikia", "clear": "Garbitu", "download": "Deskargatu", "edit": "Editatu", "empty": "Hutsik", "error": "Errorea", "hosted_on": "Ostatatuta", "loading": "Kargatzen", "logo": "logoa", "or": "edo", "remove": "Kendu", "settings": "Ezarpenak", "share": "Partekatu", "submit": "Bidali", "undo": "Desegin", "no_devices": "Ez da gailurik aurkitu", "language": "Hizkuntza", "display_theme": "Bistaratze gaia", "pwa": "Web Aplikazio Aurreratua" };
const dataframe = { "incorrect_format": "Formatu okerra, CSV eta TSV fitxategiak soilik onartzen dira", "new_column": "Zutabea gehitu", "new_row": "Errenkada berria", "add_row_above": "Errenkada gehitu gainean", "add_row_below": "Errenkada gehitu azpian", "add_column_left": "Zutabea gehitu ezkerrean", "add_column_right": "Zutabea gehitu eskuinean", "delete_row": "Ezabatu errenkila", "delete_column": "Ezabatu zutabea", "sort_column": "Ordenatu zutabea", "sort_ascending": "Behetik gorako ordena", "sort_descending": "Goitik beherako ordena", "drop_to_upload": "Arrastatu CSV edo TSV fitxategiak hemen datuak dataframe-an inportatzeko", "clear_sort": "Garbitu ordena" };
const dropdown = { "dropdown": "Goitibeherako menua" };
const errors = { "build_error": "eraikitze errore bat dago", "config_error": "konfigurazio errore bat dago", "contact_page_author": "Mesedez, jarri harremanetan orriaren egilearekin jakinarazteko.", "no_app_file": "ez dago aplikazio fitxategirik", "runtime_error": "exekuzio errore bat dago", "space_not_working": '"Space-a ez dabil honako arrazoi honengatik:" {0}', "space_paused": "space-a pausatuta dago", "use_via_api": "API bidez erabili" };
const file = { "uploading": "Igotzen..." };
const highlighted_text = { "highlighted_text": "Nabarmendutako testua" };
const image = { "allow_webcam_access": "Mesedez, baimendu web-kameraren sarbidea grabatzeko.", "brush_color": "Pintzelaren kolorea", "brush_radius": "Pintzelaren tamaina", "image": "Irudia", "remove_image": "Irudia kendu", "select_brush_color": "Aukeratu pintzelaren kolorea", "start_drawing": "Hasi marrazten", "use_brush": "Pintzela erabili", "drop_to_upload": "Arrastatu irudi fitxategia hemen igotzeko" };
const label = { "label": "Etiketa" };
const login = { "enable_cookies": "HuggingFace Space bat Inkognito moduan bisitatzen ari bazara, hirugarrenen cookieak gaitu behar dituzu.", "incorrect_credentials": "Kredentzial okerrak", "username": "erabiltzaile izena", "password": "pasahitza", "login": "Saioa hasi" };
const number = { "number": "Zenbakia" };
const plot = { "plot": "Grafikoa" };
const radio = { "radio": "Aukera botoia" };
const slider = { "slider": "Graduatzailea" };
const upload_text = { "click_to_upload": "Klikatu igotzeko", "drop_audio": "Utzi audioa hemen", "drop_csv": "Utzi CSV-a hemen", "drop_file": "Utzi fitxategia hemen", "drop_image": "Utzi irudia hemen", "drop_video": "Utzi bideoa hemen", "drop_gallery": "Utzi multimedia hemen", "paste_clipboard": "Itsatsi arbeleko edukia" };
const video = { "drop_to_upload": "Arrastatu bideo fitxategia hemen igotzeko" };
const chatbot = { "edit": "Editatu", "retry": "Saiatu berriro", "undo": "Desegin", "submit": "Bidali", "cancel": "Utzi", "like": "Gustuko dut", "dislike": "Ez dut gustuko", "clear": "Garbitu txata" };
const eu = {
  _name,
  "3D_model": { "3d_model": "3D Modeloa", "drop_to_upload": "Arrastatu 3D modelo bat (.obj, .glb, .stl, .gltf, .splat edo .ply) igotzeko." },
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
  chatbot
};

export { _name, annotated_image, audio, blocks, chatbot, checkbox, code, color_picker, common, dataframe, eu as default, dropdown, errors, file, highlighted_text, image, label, login, number, plot, radio, slider, upload_text, video };
//# sourceMappingURL=eu-wxr0YHHC.js.map
