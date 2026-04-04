const _name = "Suomi";
const annotated_image = { "annotated_image": "Merkitty kuva" };
const audio = { "allow_recording_access": "Anna lupa mikrofonin käyttöön nauhoitusta varten.", "audio": "Ääni", "record_from_microphone": "Nauhoita mikrofonista", "stop_recording": "Lopeta nauhoitus", "no_device_support": "Medialaitteisiin ei saada yhteyttä. Varmista, että käytät suojattua lähdettä (https) tai localhostia (tai olet antanut kelvollisen SSL-varmenteen ssl_verify:lle), ja olet antanut selaimelle luvan käyttää laitettasi.", "stop": "Pysäytä", "resume": "Jatka", "record": "Nauhoita", "no_microphone": "Mikrofonia ei löydy", "pause": "Tauko", "play": "Toista", "waiting": "Odotetaan", "drop_to_upload": "Lataa kuvatiedosto tähän pudottamalla se" };
const blocks = { "connection_can_break": "Mobiililaitteilla yhteys voi katketa, jos tämä välilehti menettää fokuksen tai laite menee lepotilaan, jolloin paikkasi jonossa menetetään.", "long_requests_queue": "Jonossa on paljon odottavia pyyntöjä. Monista tämä Space ohittaaksesi jonon.", "lost_connection": "Yhteys katkesi sivulta poistumisen vuoksi. Palataan jonoon...", "waiting_for_inputs": "Odotetaan tiedostojen lataamisen päättymistä, yritä uudelleen." };
const checkbox = { "checkbox": "Valintaruutu", "checkbox_group": "Valintaruuturyhmä" };
const code = { "code": "Koodi" };
const color_picker = { "color_picker": "Värivalitsin" };
const common = { "built_with": "Tehty käyttäen", "built_with_gradio": "Tehty Gradiolla", "clear": "Tyhjennä", "download": "Lataa", "edit": "Muokkaa", "empty": "Tyhjä", "error": "Virhe", "hosted_on": "Isännöity palvelimella", "loading": "Ladataan", "logo": "logo", "or": "tai", "remove": "Poista", "settings": "Asetukset", "share": "Jaa", "submit": "Lähetä", "undo": "Kumoa", "no_devices": "Laitteita ei löytynyt", "language": "Kieli", "display_theme": "Näyttöteema", "pwa": "Progressiivinen verkkosovellus" };
const dataframe = { "incorrect_format": "Väärä muoto, vain CSV- ja TSV-tiedostot ovat tuettuja", "new_column": "Lisää sarake", "new_row": "Uusi rivi", "add_row_above": "Lisää rivi yläpuolelle", "add_row_below": "Lisää rivi alapuolelle", "add_column_left": "Lisää sarake vasemmalle", "add_column_right": "Lisää sarake oikealle", "delete_row": "Poista rivi", "delete_column": "Poista sarake", "sort_column": "Lajittele sarake", "sort_ascending": "Lajittele nousevaan järjestykseen", "sort_descending": "Lajittele laskevaan järjestykseen", "drop_to_upload": "Lataa CSV- tai TSV-tiedostoja tähän tuodaaksesi datan dataframeen.", "clear_sort": "Tyhjennä järjestys" };
const dropdown = { "dropdown": "Pudotusvalikko" };
const errors = { "build_error": "rakennusvirhe", "config_error": "määritysvirhe", "contact_page_author": "Ota yhteyttä sivun tekijään ilmoittaaksesi.", "no_app_file": "sovellustiedostoa ei ole", "runtime_error": "ajonaikainen virhe", "space_not_working": '"Space ei toimi koska" {0}', "space_paused": "Space on keskeytetty", "use_via_api": "Käytä API:n kautta" };
const file = { "uploading": "Ladataan..." };
const highlighted_text = { "highlighted_text": "Korostettu teksti" };
const image = { "allow_webcam_access": "Anna lupa web-kameran käyttöön nauhoitusta varten.", "brush_color": "Siveltimen väri", "brush_radius": "Siveltimen koko", "image": "Kuva", "remove_image": "Poista kuva", "select_brush_color": "Valitse siveltimen väri", "start_drawing": "Aloita piirtäminen", "use_brush": "Käytä sivellintä", "drop_to_upload": "Lataa kuva tiedosto tähän pudottamalla se" };
const label = { "label": "Tunniste" };
const login = { "enable_cookies": "Jos vierailet HuggingFace Spacessa Incognito-tilassa, sinun täytyy sallia kolmannen osapuolen evästeet.", "incorrect_credentials": "Virheelliset kirjautumistiedot", "username": "käyttäjätunnus", "password": "salasana", "login": "Kirjaudu sisään" };
const number = { "number": "Numero" };
const plot = { "plot": "Kaavio" };
const radio = { "radio": "Valintanappi" };
const slider = { "slider": "Liukusäädin" };
const upload_text = { "click_to_upload": "Napsauta ladataksesi", "drop_audio": "Pudota äänitiedosto tähän", "drop_csv": "Pudota CSV-tiedosto tähän", "drop_file": "Pudota tiedosto tähän", "drop_image": "Pudota kuva tähän", "drop_video": "Pudota video tähän", "drop_gallery": "Pudota media tähän", "paste_clipboard": "Liitä leikepöydältä" };
const video = { "drop_to_upload": "Lataa videotiedosto pudottamalla se tähän." };
const chatbot = { "edit": "Muokkaa", "retry": "Yritä uudelleen", "undo": "Kumoa", "submit": "Lähetä", "cancel": "Peruuta", "like": "Tykkää", "dislike": "En tykkää", "clear": "Tyhjennä keskustelu" };
const fi = {
  _name,
  "3D_model": { "3d_model": "3D-malli", "drop_to_upload": "Lataa 3D-malli (.obj, .glb, .stl, .gltf, .splat tai .ply) tähän pudottamalla tiedosto." },
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

export { _name, annotated_image, audio, blocks, chatbot, checkbox, code, color_picker, common, dataframe, fi as default, dropdown, errors, file, highlighted_text, image, label, login, number, plot, radio, slider, upload_text, video };
//# sourceMappingURL=fi-BGbOKfoP.js.map
