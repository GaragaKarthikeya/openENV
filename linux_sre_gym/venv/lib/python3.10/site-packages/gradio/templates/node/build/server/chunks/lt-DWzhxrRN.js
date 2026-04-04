const _name = "Lietuvių";
const common = { "built_with_gradio": "Sukurta su Gradio", "clear": "Išvalyti", "or": "arba", "submit": "Pateikti", "settings": "Nustatymai", "built_with": "sukurta su", "download": "Atsisiųsti", "edit": "Redaguoti", "empty": "Tuščia", "error": "Klaida", "hosted_on": "Talpinama", "loading": "Kraunama", "logo": "logotipas", "remove": "Pašalinti", "share": "Dalintis", "undo": "Atšaukti", "no_devices": "Nerasta jokių įrenginių", "language": "Kalba", "display_theme": "Ekrano tema", "pwa": "Progresyvi interneto programėlė" };
const upload_text = { "click_to_upload": "Spustelėkite norėdami įkelti", "drop_audio": "Įkelkite garso įrašą čia", "drop_csv": "Įkelkite CSV čia", "drop_file": "Įkelkite failą čia", "drop_image": "Įkelkite paveikslėlį čia", "drop_video": "Įkelkite vaizdo įrašą čia", "drop_gallery": "Įkelkite mediją čia", "paste_clipboard": "Įklijuoti iš iškarpinės" };
const annotated_image = { "annotated_image": "Pažymėtas paveikslėlis" };
const audio = { "allow_recording_access": "Prašome suteikti prieigą prie mikrofono įrašymui.", "audio": "Garsas", "drop_to_upload": "Įkelkite garso failą čia", "record_from_microphone": "Įrašyti iš mikrofono", "stop_recording": "Stabdyti įrašymą", "no_device_support": "Nepavyko pasiekti medijos įrenginių. Patikrinkite, ar naudojate saugią prieigą (https) arba localhost (arba perduodate galiojantį SSL sertifikatą į ssl_verify), ir leidžiate naršyklei prieiti prie jūsų įrenginio.", "stop": "Stabdyti", "resume": "Tęsti", "record": "Įrašyti", "no_microphone": "Mikrofonas nerastas", "pause": "Pristabdyti", "play": "Groti", "waiting": "Laukiama" };
const blocks = { "connection_can_break": "Mobiliajame įrenginyje ryšys gali nutrūkti, jei šis skirtukas bus neaktyvus arba įrenginys užmigs, todėl galite prarasti savo vietą eilėje.", "long_requests_queue": "Yra ilga laukiančių užklausų eilė. Dubliuokite šią erdvę, kad praleistumėte eilę.", "lost_connection": "Ryšys nutrūko išėjus iš puslapio. Vėl jungiamasi į eilę...", "waiting_for_inputs": "Laukiama, kol bus baigti įkelti failai, prašome bandyti dar kartą." };
const checkbox = { "checkbox": "Žymimasis langelis", "checkbox_group": "Žymimųjų langelių grupė" };
const code = { "code": "Kodas" };
const color_picker = { "color_picker": "Spalvos pasirinkimas" };
const dataframe = { "incorrect_format": "Neteisingas formatas, palaikomi tik CSV ir TSV failai", "new_column": "Pridėti stulpelį", "new_row": "Nauja eilutė", "add_row_above": "Pridėti eilutę viršuje", "add_row_below": "Pridėti eilutę žemiau", "delete_row": "Ištrinti eilutę", "delete_column": "Ištrinti stulpelį", "add_column_left": "Pridėti stulpelį kairėje", "add_column_right": "Pridėti stulpelį dešinėje", "sort_column": "Rikiuoti stulpelį", "sort_ascending": "Rikiuoti didėjančia tvarka", "sort_descending": "Rikiuoti mažėjančia tvarka", "drop_to_upload": "Įkelkite CSV arba TSV failus čia, kad importuotumėte duomenis į duomenų lentelę", "clear_sort": "Išvalyti rikiavimą" };
const dropdown = { "dropdown": "Išskleidžiamasis sąrašas" };
const errors = { "build_error": "yra kompiliavimo klaida", "config_error": "yra konfigūracijos klaida", "contact_page_author": "Prašome susisiekti su puslapio autoriumi.", "no_app_file": "nėra programos failo", "runtime_error": "yra vykdymo klaida", "space_not_working": '"Erdvė neveikia, nes" {0}', "space_paused": "erdvė yra pristabdyta", "use_via_api": "Naudoti per API", "use_via_api_or_mcp": "Naudoti per API arba MCP" };
const file = { "uploading": "Įkeliama..." };
const highlighted_text = { "highlighted_text": "Paryškintas tekstas" };
const image = { "allow_webcam_access": "Prašome suteikti prieigą prie internetinės kameros įrašymui.", "brush_color": "Teptuko spalva", "brush_radius": "Teptuko dydis", "image": "Paveikslėlis", "remove_image": "Pašalinti paveikslėlį", "select_brush_color": "Pasirinkite teptuko spalvą", "start_drawing": "Pradėti piešti", "use_brush": "Naudoti teptuką", "drop_to_upload": "Įkelkite paveikslėlio failą čia" };
const label = { "label": "Etiketė" };
const login = { "enable_cookies": "Jei lankotės HuggingFace erdvėje inkognito režimu, turite leisti trečiųjų šalių slapukus.", "incorrect_credentials": "Neteisingi prisijungimo duomenys", "username": "Vartotojo vardas", "password": "Slaptažodis", "login": "Prisijungti" };
const number = { "number": "Skaičius" };
const plot = { "plot": "Grafikas" };
const radio = { "radio": "Pasirinkimo mygtukas" };
const slider = { "slider": "Slankiklis" };
const video = { "drop_to_upload": "Įkelkite vaizdo įrašo failą čia" };
const chatbot = { "edit": "Redaguoti", "retry": "Bandyti dar kartą", "undo": "Atšaukti", "submit": "Pateikti", "cancel": "Atšaukti", "like": "Patinka", "dislike": "Nepatinka", "clear": "Išvalyti pokalbį" };
const lt = {
  _name,
  common,
  upload_text,
  "3D_model": { "3d_model": "3D modelis", "drop_to_upload": "Įkelkite 3D modelio failą (.obj, .glb, .stl, .gltf, .splat arba .ply) čia" },
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

export { _name, annotated_image, audio, blocks, chatbot, checkbox, code, color_picker, common, dataframe, lt as default, dropdown, errors, file, highlighted_text, image, label, login, number, plot, radio, slider, upload_text, video };
//# sourceMappingURL=lt-DWzhxrRN.js.map
