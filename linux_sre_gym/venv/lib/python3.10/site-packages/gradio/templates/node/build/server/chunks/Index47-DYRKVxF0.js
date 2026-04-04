import './async-D55cHugf.js';
import { d as bind_props, f as attr_class, a as attr, e as ensure_array_like, g as attr_style, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio, f as format_time } from './utils.svelte-aYomt3KL.js';
import { o as onDestroy, t as tick } from './index-server-CQz6EZl_.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { F as File } from './File-2S6P7zIO.js';
import { M as Microphone } from './Microphone-CCAKTpuQ.js';
import { M as Music } from './Music-DcoyPX64.js';
import { S as Square } from './Square-CJzdI5ja.js';
import { V as Video } from './Video-_1zl9-Cr.js';
import { S as ScrollFade } from './ScrollFade-BvKMcxYM.js';
import { U as Upload } from './Upload2-OQ1gRY0J.js';
import { I as Image } from './Image-BLyvTcTy.js';
import { M as MinimalAudioPlayer } from './MinimalAudioPlayer-Bn_OZwMw.js';
import './record.esm-DyTKO0GB.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example23-Cdj-Vjy_.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './IconButton-D82v5nCM.js';
import './Video2-De5bYQY6.js';

function ArrowUp($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>`);
}

function Paperclip($$renderer) {
  $$renderer.push(`<svg fill="currentColor" width="100%" height="100%" viewBox="0 0 1920 1920" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M1752.768 221.109C1532.646.986 1174.283.986 954.161 221.109l-838.588 838.588c-154.052 154.165-154.052 404.894 0 558.946 149.534 149.421 409.976 149.308 559.059 0l758.738-758.626c87.982-88.094 87.982-231.417 0-319.51-88.32-88.208-231.642-87.982-319.51 0l-638.796 638.908 79.85 79.849 638.795-638.908c43.934-43.821 115.539-43.934 159.812 0 43.934 44.047 43.934 115.877 0 159.812l-758.739 758.625c-110.23 110.118-289.355 110.005-399.36 0-110.118-110.117-110.005-289.242 0-399.247l838.588-838.588c175.963-175.962 462.382-176.188 638.909 0 176.075 176.188 176.075 462.833 0 638.908l-798.607 798.72 79.849 79.85 798.607-798.72c220.01-220.123 220.01-578.485 0-798.607" fill-rule="evenodd"></path></g></svg>`);
}
function MinimalAudioRecorder($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      label,
      waveform_settings = {},
      recording = void 0,
      upload,
      root,
      max_file_size = null,
      upload_promise = void 0,
      onchange,
      onstoprecording,
      onclear
    } = $$props;
    let seconds = 0;
    let interval;
    onDestroy(() => {
      clearInterval(interval);
    });
    $$renderer2.push(`<div class="minimal-audio-recorder svelte-doy9oc"${attr("aria-label", label || "Audio Recorder")} data-testid="minimal-audio-recorder">`);
    {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="waveform-wrapper svelte-doy9oc"></div> <div class="timestamp svelte-doy9oc">${escape_html(format_time(seconds))}</div> <button class="stop-button svelte-doy9oc" aria-label="Stop recording">`);
      Square($$renderer2, {});
      $$renderer2.push(`<!----></button>`);
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { recording, upload_promise });
  });
}
function MultimodalTextbox($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = void 0,
      value_is_output = false,
      lines = 1,
      i18n: _i18n,
      placeholder = "",
      disabled = false,
      label,
      info = void 0,
      show_label = true,
      max_lines,
      submit_btn = null,
      stop_btn = null,
      rtl = false,
      autofocus = false,
      text_align = void 0,
      autoscroll = true,
      root,
      file_types_string = null,
      max_file_size = null,
      upload,
      stream_handler,
      file_count = "multiple",
      max_plain_text_length = 1e3,
      waveform_settings,
      waveform_options: _waveform_options = { show_recording_waveform: true },
      sources_string = "upload",
      active_source = void 0,
      html_attributes = null,
      upload_promise = void 0,
      dragging = void 0,
      onchange,
      onsubmit,
      onstop,
      onblur,
      onselect,
      oninput,
      onfocus,
      ondrag,
      onupload,
      onclear,
      onload: _onload,
      onerror,
      onstop_recording
    } = $$props;
    let show_fade = false;
    let uploading = false;
    value?.text ?? "";
    let recording = false;
    let mic_audio = null;
    let sources = sources_string.split(",").map((s) => s.trim()).filter((s) => s === "upload" || s === "microphone");
    let file_types = file_types_string ? file_types_string.split(",").map((s) => s.trim()) : null;
    let show_upload = sources && sources.includes("upload") && !(file_count === "single" && value?.files?.length > 0);
    async function handle_change() {
      onchange?.(value);
      if (!value_is_output) {
        oninput?.();
      }
    }
    async function handle_upload(detail) {
      handle_change();
      if (Array.isArray(detail)) {
        for (let file of detail) {
          value.files.push(file);
        }
        value = value;
      } else {
        value.files.push(detail);
        value = value;
      }
      await tick();
      onchange?.(value);
      onupload?.(detail);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div${attr_class("full-container svelte-1qn0337", void 0, { "dragging": dragging })} role="group" aria-label="Multimedia input field">`);
      BlockTitle($$renderer3, {
        show_label,
        info,
        rtl,
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->${escape_html(label)}`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> <div class="input-container svelte-1qn0337">`);
      if (sources && sources.includes("microphone") && active_source === "microphone") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div${attr_class("recording-overlay svelte-1qn0337", void 0, { "has-audio": mic_audio !== null })}>`);
        if (!mic_audio) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="recording-content svelte-1qn0337">`);
          MinimalAudioRecorder($$renderer3, {
            label: label || "Audio",
            waveform_settings,
            recording,
            upload,
            root,
            max_file_size,
            onchange: (audio_value) => {
              mic_audio = audio_value;
            },
            onstoprecording: () => {
              recording = false;
              onstop_recording?.();
            },
            onclear: () => {
              active_source = null;
              recording = false;
              mic_audio = null;
              onclear?.();
            },
            get upload_promise() {
              return upload_promise;
            },
            set upload_promise($$value) {
              upload_promise = $$value;
              $$settled = false;
            }
          });
          $$renderer3.push(`<!----></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`<div class="recording-content svelte-1qn0337">`);
          MinimalAudioPlayer($$renderer3, { value: mic_audio, label: label || "Audio", loop: false });
          $$renderer3.push(`<!----> <div class="action-buttons svelte-1qn0337"><button class="confirm-button svelte-1qn0337" aria-label="Attach audio">`);
          Check($$renderer3);
          $$renderer3.push(`<!----></button> <button class="cancel-button svelte-1qn0337" aria-label="Clear audio">`);
          Clear($$renderer3);
          $$renderer3.push(`<!----></button></div></div>`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (show_upload) {
        $$renderer3.push("<!--[-->");
        Upload($$renderer3, {
          onload: handle_upload,
          file_count,
          filetype: file_types,
          root,
          max_file_size,
          show_progress: false,
          disable_click: true,
          onerror,
          hidden: true,
          upload,
          stream_handler,
          get upload_promise() {
            return upload_promise;
          },
          set upload_promise($$value) {
            upload_promise = $$value;
            $$settled = false;
          },
          get dragging() {
            return dragging;
          },
          set dragging($$value) {
            dragging = $$value;
            $$settled = false;
          },
          get uploading() {
            return uploading;
          },
          set uploading($$value) {
            uploading = $$value;
            $$settled = false;
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div${attr_class("input-wrapper svelte-1qn0337", void 0, { "has-files": (value?.files?.length ?? 0) > 0 || uploading })}>`);
      if ((value?.files?.length ?? 0) > 0 || uploading) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="thumbnails svelte-1qn0337" aria-label="Uploaded files" data-testid="container_el">`);
        if (show_upload) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button data-testid="upload-button" class="upload-button thumbnail-add svelte-1qn0337"${attr("disabled", disabled, true)} aria-label="Upload a file">`);
          Paperclip($$renderer3);
          $$renderer3.push(`<!----></button>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <!--[-->`);
        const each_array = ensure_array_like(value?.files ?? []);
        for (let index = 0, $$length = each_array.length; index < $$length; index++) {
          let file = each_array[index];
          $$renderer3.push(`<span class="thumbnail-wrapper svelte-1qn0337" role="listitem" aria-label="File thumbnail"><div class="thumbnail-item thumbnail-small svelte-1qn0337">`);
          if (file.mime_type && file.mime_type.includes("image")) {
            $$renderer3.push("<!--[-->");
            Image($$renderer3, {
              src: file.url,
              restProps: {
                title: null,
                alt: "",
                loading: "lazy",
                class: "thumbnail-image"
              }
            });
          } else {
            $$renderer3.push("<!--[!-->");
            if (file.mime_type && file.mime_type.includes("audio")) {
              $$renderer3.push("<!--[-->");
              Music($$renderer3);
            } else {
              $$renderer3.push("<!--[!-->");
              if (file.mime_type && file.mime_type.includes("video")) {
                $$renderer3.push("<!--[-->");
                Video($$renderer3);
              } else {
                $$renderer3.push("<!--[!-->");
                File($$renderer3);
              }
              $$renderer3.push(`<!--]-->`);
            }
            $$renderer3.push(`<!--]-->`);
          }
          $$renderer3.push(`<!--]--> <button class="delete-button svelte-1qn0337" aria-label="Remove file">`);
          Clear($$renderer3);
          $$renderer3.push(`<!----></button></div></span>`);
        }
        $$renderer3.push(`<!--]--> `);
        if (uploading) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="loader svelte-1qn0337" role="status" aria-label="Uploading"></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div class="input-row svelte-1qn0337">`);
      if (show_upload && (value?.files?.length ?? 0) === 0 && !uploading) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button data-testid="upload-button" class="upload-button icon-button svelte-1qn0337"${attr("disabled", disabled, true)} aria-label="Upload a file">`);
        Paperclip($$renderer3);
        $$renderer3.push(`<!----></button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div class="textarea-wrapper svelte-1qn0337"><textarea data-testid="textbox"${attr("dir", rtl ? "rtl" : "ltr")}${attr("placeholder", placeholder)}${attr("rows", lines)}${attr("disabled", disabled, true)}${attr_style(text_align ? "text-align: " + text_align : "")}${attr("autocapitalize", html_attributes?.autocapitalize)}${attr("autocorrect", html_attributes?.autocorrect)}${attr("spellcheck", html_attributes?.spellcheck)}${attr("autocomplete", html_attributes?.autocomplete)}${attr("tabindex", html_attributes?.tabindex)}${attr("enterkeyhint", html_attributes?.enterkeyhint)}${attr("lang", html_attributes?.lang)}${attr_class("svelte-1qn0337", void 0, { "no-label": !show_label })}>`);
      const $$body = escape_html(value.text);
      if ($$body) {
        $$renderer3.push(`${$$body}`);
      }
      $$renderer3.push(`</textarea> `);
      ScrollFade($$renderer3, { visible: show_fade, position: "absolute" });
      $$renderer3.push(`<!----></div> `);
      if (sources && sources.includes("microphone")) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button data-testid="microphone-button"${attr_class("microphone-button svelte-1qn0337", void 0, { "recording": recording })}${attr("disabled", disabled, true)} aria-label="Record audio">`);
        Microphone($$renderer3);
        $$renderer3.push(`<!----></button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (submit_btn) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button${attr_class("submit-button svelte-1qn0337", void 0, { "padded-button": submit_btn !== true })} data-testid="submit-button"${attr("disabled", disabled, true)} aria-label="Submit">`);
        if (submit_btn === true) {
          $$renderer3.push("<!--[-->");
          ArrowUp($$renderer3);
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`${escape_html(submit_btn)}`);
        }
        $$renderer3.push(`<!--]--></button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (stop_btn) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button${attr_class("stop-button svelte-1qn0337", void 0, { "padded-button": stop_btn !== true })} aria-label="Stop">`);
        if (stop_btn === true) {
          $$renderer3.push("<!--[-->");
          Square($$renderer3, { fill: "none", stroke_width: 2.5 });
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`${escape_html(stop_btn)}`);
        }
        $$renderer3.push(`<!--]--></button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div></div></div></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, active_source, upload_promise, dragging });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let upload_promise = null;
    class MultimodalTextboxGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new MultimodalTextboxGradio(props);
    gradio.props.value = gradio.props.value ?? { text: "", files: [] };
    let dragging = false;
    let active_source = null;
    const waveform_settings = {
      height: 50,
      barWidth: 2,
      barGap: 3,
      cursorWidth: 2,
      cursorColor: "#ddd5e9",
      autoplay: false,
      barRadius: 10,
      dragToSeek: true,
      normalize: true,
      minPxPerSec: 20,
      waveColor: "",
      progressColor: "",
      mediaControls: false,
      sampleRate: 44100
    };
    ({
      color: gradio.props?.waveform_options?.trim_region_color
    });
    const upload_fn = (...args) => gradio.shared.client.upload(...args);
    const i18n = (s) => gradio.i18n(s);
    const stream_handler_fn = (...args) => gradio.shared.client.stream(...args);
    let sources_string = gradio.props.sources.join(",");
    let file_types_string = (() => (gradio.props.file_types || []).join(",") || null)();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: [...gradio.shared.elem_classes || [], "multimodal-textbox"],
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        padding: false,
        border_mode: dragging ? "focus" : "base",
        rtl: gradio.props.rtl,
        children: ($$renderer4) => {
          if (gradio.shared.loading_status) {
            $$renderer4.push("<!--[-->");
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          MultimodalTextbox($$renderer4, {
            value_is_output: true,
            file_types_string,
            root: gradio.shared.root,
            label: gradio.shared.label || "MultimodalTextbox",
            info: gradio.props.info,
            show_label: gradio.shared.show_label,
            lines: gradio.props.lines,
            rtl: gradio.props.rtl,
            text_align: gradio.props.text_align,
            waveform_settings,
            i18n,
            max_lines: !gradio.props.max_lines ? gradio.props.lines + 1 : gradio.props.max_lines,
            placeholder: gradio.props.placeholder,
            submit_btn: gradio.props.submit_btn,
            stop_btn: gradio.props.stop_btn,
            autofocus: gradio.props.autofocus,
            autoscroll: gradio.shared.autoscroll,
            file_count: gradio.props.file_count,
            sources_string,
            max_file_size: gradio.shared.max_file_size,
            onchange: (e) => {
              gradio.props.value = e;
              gradio.dispatch("change", gradio.props.value);
            },
            oninput: () => gradio.dispatch("input"),
            onsubmit: () => gradio.dispatch("submit"),
            onstop: () => gradio.dispatch("stop"),
            onblur: () => gradio.dispatch("blur"),
            onselect: (e) => gradio.dispatch("select", e),
            onfocus: () => gradio.dispatch("focus"),
            onerror: (detail) => {
              gradio.dispatch("error", detail);
            },
            onstop_recording: () => gradio.dispatch("stop_recording"),
            onupload: (e) => gradio.dispatch("upload", e),
            onclear: () => gradio.dispatch("clear"),
            disabled: !gradio.shared.interactive,
            upload: upload_fn,
            stream_handler: stream_handler_fn,
            max_plain_text_length: gradio.props.max_plain_text_length,
            html_attributes: gradio.props.html_attributes,
            get upload_promise() {
              return upload_promise;
            },
            set upload_promise($$value) {
              upload_promise = $$value;
              $$settled = false;
            },
            get value() {
              return gradio.props.value;
            },
            set value($$value) {
              gradio.props.value = $$value;
              $$settled = false;
            },
            get dragging() {
              return dragging;
            },
            set dragging($$value) {
              dragging = $$value;
              $$settled = false;
            },
            get active_source() {
              return active_source;
            },
            set active_source($$value) {
              active_source = $$value;
              $$settled = false;
            }
          });
          $$renderer4.push(`<!---->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { MultimodalTextbox as BaseMultimodalTextbox, Index as default };
//# sourceMappingURL=Index47-DYRKVxF0.js.map
