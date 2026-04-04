import './async-D55cHugf.js';
import { d as bind_props, f as attr_class, a as attr, i as stringify, c as spread_props, g as attr_style, j as clsx, e as ensure_array_like } from './index-u8mz_F03.js';
import { o as onDestroy, t as tick } from './index-server-CQz6EZl_.js';
import { a as prepare_files } from './2-BAeILJ1g.js';
import { G as Gradio, u as uploadToHuggingFace } from './utils.svelte-aYomt3KL.js';
import { S as Static } from './index3-BSof0MuO.js';
import { A as AudioPlayer, S as StaticAudio } from './StaticAudio-CGENs1q8.js';
import { U as Upload } from './Upload2-OQ1gRY0J.js';
import { M as ModifyUpload } from './ModifyUpload-D_fhEqLO.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { S as ShareButton } from './ShareButton-DnAPtcgb.js';
import { M as Music } from './Music-DcoyPX64.js';
import { S as SelectSource } from './SelectSource-CeUXfvCc.js';
import { C as CustomButton } from './IconButtonWrapper-DvxA4nj6.js';
import { S as StreamingBar } from './StreamingBar-B1BHh5xK.js';
import './MinimalAudioPlayer-Bn_OZwMw.js';
import './record.esm-DyTKO0GB.js';
import { B as Block } from './Block-JRVGROlG.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
export { default as BaseExample } from './Example10-DPWDxkA2.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';
import './DownloadLink-BMNxSkQ8.js';
import './Empty-BHvMDNv5.js';
import './Download-ByiErn53.js';
import './VolumeLevels-C0Xso12f.js';
import './Play-BlZWjD1q.js';
import './Undo-Col2BcUY.js';
import './Edit-W_0aHh0i.js';
import './Upload-CIQ-D6yx.js';
import './Microphone-CCAKTpuQ.js';
import './Video-_1zl9-Cr.js';
import './Webcam-iV1BaoQQ.js';

function Spinner($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 50 50" class="svelte-l5a973"><circle cx="25" cy="25" r="20" fill="none" stroke-width="3.0" stroke-linecap="round" stroke-dasharray="94.2477796076938 94.2477796076938" stroke-dashoffset="0" class="svelte-l5a973"><animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" repeatCount="indefinite" class="svelte-l5a973"></animateTransform></circle></svg>`);
}
function DeviceSelect($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { i18n, micDevices = void 0, onerror } = $$props;
    $$renderer2.push(`<select class="mic-select svelte-ym1wxn" aria-label="Select input device"${attr("disabled", !micDevices || micDevices.length === 0, true)}>`);
    if (!micDevices || micDevices.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.option({ value: "" }, ($$renderer3) => {
        $$renderer3.push(`${escape_html(i18n("audio.no_microphone"))}`);
      });
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(micDevices);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let micDevice = each_array[$$index];
        $$renderer2.option({ value: micDevice.deviceId }, ($$renderer3) => {
          $$renderer3.push(`${escape_html(micDevice.label)}`);
        });
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></select>`);
    bind_props($$props, { micDevices });
  });
}
function AudioRecorder($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      mode = void 0,
      i18n,
      dispatch_blob,
      waveform_settings,
      waveform_options = { show_recording_waveform: true },
      handle_reset_value,
      editable = true,
      recording = false,
      onstartrecording,
      onpauserecording,
      onstoprecording,
      onstop,
      onplay,
      onpause,
      onedit
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="component-wrapper svelte-j9q3sk"><div class="microphone svelte-j9q3sk" data-testid="microphone-waveform"></div> <div data-testid="recording-waveform"></div> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { mode });
  });
}
function StreamAudio($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      recording = false,
      paused_recording = false,
      i18n,
      waveform_options = { show_recording_waveform: true },
      waiting = false
    } = $$props;
    let micDevices = [];
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="mic-wrap svelte-m6ymia">`);
      if (waveform_options.show_recording_waveform) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="svelte-m6ymia"${attr_style("", { display: recording ? "block" : "none" })}></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div class="controls svelte-m6ymia">`);
      if (recording && !waiting) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button${attr_class(clsx(paused_recording ? "stop-button-paused" : "stop-button"), "svelte-m6ymia")}><span class="record-icon svelte-m6ymia"><span class="pinger svelte-m6ymia"></span> <span class="dot svelte-m6ymia"></span></span> ${escape_html(paused_recording ? i18n("audio.pause") : i18n("audio.stop"))}</button>`);
      } else {
        $$renderer3.push("<!--[!-->");
        if (recording && waiting) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button class="spinner-button svelte-m6ymia"><div class="icon svelte-m6ymia">`);
          Spinner($$renderer3);
          $$renderer3.push(`<!----></div> ${escape_html(i18n("audio.waiting"))}</button>`);
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`<button class="record-button svelte-m6ymia"><span class="record-icon svelte-m6ymia"><span class="dot svelte-m6ymia"></span></span> ${escape_html(i18n("audio.record"))}</button>`);
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]--> `);
      DeviceSelect($$renderer3, {
        i18n,
        get micDevices() {
          return micDevices;
        },
        set micDevices($$value) {
          micDevices = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----></div></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
let media_recorder;
async function init_media_recorder() {
  const { MediaRecorder, register } = await import('./module3-BzuIcAYT.js');
  const { connect } = await import('./module-Wh9pZYyV.js');
  register(await connect());
  media_recorder = MediaRecorder;
  return media_recorder;
}
function InteractiveAudio($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = null,
      subtitles = null,
      label,
      root,
      loop,
      show_label = true,
      buttons = ["download", "share"],
      on_custom_button_click = null,
      sources = ["microphone", "upload"],
      pending = false,
      streaming = false,
      i18n,
      waveform_settings,
      trim_region_settings = {},
      waveform_options = {},
      dragging = false,
      active_source = "microphone",
      handle_reset_value = () => {
      },
      editable = true,
      max_file_size = null,
      upload,
      stream_handler,
      stream_every = 0.1,
      uploading = false,
      recording = false,
      class_name = "",
      upload_promise = void 0,
      initial_value = void 0,
      playback_position = void 0,
      time_limit = null,
      stream_state = "closed",
      onchange,
      onstream,
      onedit,
      onplay,
      onpause,
      onstop,
      ondrag,
      onerror,
      onupload,
      onclear,
      onstart_recording,
      onpause_recording,
      onstop_recording,
      onclose_stream,
      children
    } = $$props;
    let mode = "";
    const is_browser = typeof window !== "undefined";
    if (is_browser && streaming) {
      init_media_recorder().then((a) => {
      });
    }
    const to_blob_parts = (parts) => parts.map((part) => {
      if (part instanceof Blob) return part;
      return part.slice();
    });
    const dispatch_blob = async (blobs, event) => {
      let _audio_blob = new File(to_blob_parts(blobs), "audio.wav", { type: "audio/wav" });
      if (_audio_blob.size === 0) {
        return;
      }
      const val = await prepare_files([_audio_blob], event === "stream");
      initial_value = value;
      value = (await upload(val, root, void 0, max_file_size || void 0))?.filter(Boolean)[0];
      if (event === "stream") {
        onstream?.(value);
      } else if (event === "change") {
        onchange?.(value);
      } else if (event === "stop_recording") {
        onstop_recording?.();
      }
    };
    onDestroy(() => {
    });
    function clear() {
      onchange?.(null);
      onclear?.();
      mode = "";
      value = null;
    }
    function handle_load(detail) {
      value = detail;
      onchange?.(detail);
      onupload?.(detail);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, {
        show_label,
        Icon: Music,
        float: active_source === "upload" && value === null,
        label: label || i18n("audio.audio")
      });
      $$renderer3.push(`<!----> <div${attr_class(`audio-container ${stringify(class_name)}`, "svelte-ocxd3m")}${attr("data-testid", label ? "waveform-" + label : "unlabelled-audio")}>`);
      StreamingBar($$renderer3, { time_limit });
      $$renderer3.push(`<!----> `);
      if (value == null || streaming) {
        $$renderer3.push("<!--[-->");
        if (active_source === "microphone") {
          $$renderer3.push("<!--[-->");
          ModifyUpload($$renderer3, { i18n, onclear: clear });
          $$renderer3.push(`<!----> `);
          if (streaming) {
            $$renderer3.push("<!--[-->");
            StreamAudio($$renderer3, {
              recording,
              i18n,
              waveform_options,
              waiting: stream_state === "waiting"
            });
          } else {
            $$renderer3.push("<!--[!-->");
            AudioRecorder($$renderer3, {
              i18n,
              editable,
              recording,
              dispatch_blob,
              waveform_settings,
              waveform_options,
              handle_reset_value,
              onstartrecording: () => onstart_recording?.(),
              onpauserecording: () => onpause_recording?.(),
              onstoprecording: () => onstop_recording?.(),
              get mode() {
                return mode;
              },
              set mode($$value) {
                mode = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer3.push(`<!--]-->`);
        } else {
          $$renderer3.push("<!--[!-->");
          if (active_source === "upload") {
            $$renderer3.push("<!--[-->");
            Upload($$renderer3, {
              filetype: "audio/aac,audio/midi,audio/mpeg,audio/ogg,audio/wav,audio/x-wav,audio/opus,audio/webm,audio/flac,audio/vnd.rn-realaudio,audio/x-ms-wma,audio/x-aiff,audio/amr,audio/*",
              onload: handle_load,
              onerror: (detail) => onerror?.(detail),
              root,
              max_file_size,
              upload,
              stream_handler,
              aria_label: i18n("audio.drop_to_upload"),
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
              },
              children: ($$renderer4) => {
                if (children) {
                  $$renderer4.push("<!--[-->");
                  children($$renderer4);
                  $$renderer4.push(`<!---->`);
                } else {
                  $$renderer4.push("<!--[!-->");
                }
                $$renderer4.push(`<!--]-->`);
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]-->`);
      } else {
        $$renderer3.push("<!--[!-->");
        ModifyUpload($$renderer3, {
          i18n,
          onclear: clear,
          onedit: () => {
            mode = "edit";
            onedit?.();
          },
          download: buttons === null ? value.url : buttons.some((btn) => typeof btn === "string" && btn === "download") ? value.url : null,
          children: ($$renderer4) => {
            if (value !== null && buttons) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<!--[-->`);
              const each_array = ensure_array_like(buttons);
              for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                let btn = each_array[$$index];
                if (typeof btn === "string") {
                  $$renderer4.push("<!--[-->");
                  if (btn === "share") {
                    $$renderer4.push("<!--[-->");
                    ShareButton($$renderer4, {
                      i18n,
                      onerror,
                      onshare: () => {
                      },
                      formatter: async (fileData) => {
                        if (!fileData || !fileData.url) return "";
                        let url = await uploadToHuggingFace(fileData.url);
                        return `<audio controls src="${url}"></audio>`;
                      },
                      value
                    });
                  } else {
                    $$renderer4.push("<!--[!-->");
                  }
                  $$renderer4.push(`<!--]-->`);
                } else {
                  $$renderer4.push("<!--[!-->");
                  CustomButton($$renderer4, {
                    button: btn,
                    on_click: (id) => {
                      if (on_custom_button_click) {
                        on_custom_button_click(id);
                      }
                    }
                  });
                }
                $$renderer4.push(`<!--]-->`);
              }
              $$renderer4.push(`<!--]-->`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
        });
        $$renderer3.push(`<!----> `);
        AudioPlayer($$renderer3, {
          value,
          subtitles: Array.isArray(subtitles) ? subtitles : subtitles?.url,
          label,
          i18n,
          dispatch_blob,
          waveform_settings,
          waveform_options,
          trim_region_settings,
          handle_reset_value,
          editable,
          loop,
          interactive: true,
          onstop,
          onplay,
          onpause,
          onedit,
          get mode() {
            return mode;
          },
          set mode($$value) {
            mode = $$value;
            $$settled = false;
          },
          get playback_position() {
            return playback_position;
          },
          set playback_position($$value) {
            playback_position = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!---->`);
      }
      $$renderer3.push(`<!--]--> `);
      SelectSource($$renderer3, {
        sources,
        handle_clear: clear,
        get active_source() {
          return active_source;
        },
        set active_source($$value) {
          active_source = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      value,
      dragging,
      active_source,
      uploading,
      recording,
      upload_promise,
      initial_value,
      playback_position
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    let upload_promise = void 0;
    props.props.stream_every = 0.1;
    class AudioGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const gradio = new AudioGradio(props);
    let label = gradio.shared.label || gradio.i18n("audio.audio");
    let minimal = props.minimal ?? gradio.props.minimal ?? false;
    let active_source = (() => gradio.props.sources ? gradio.props.sources[0] : null)();
    let initial_value = gradio.props.value;
    const handle_reset_value = () => {
      if (initial_value === null || gradio.props.value === initial_value) {
        return;
      }
      gradio.props.value = initial_value;
    };
    let dragging = false;
    let recording = gradio.props.recording ?? false;
    let color_accent = "darkorange";
    let waveform_settings = {
      height: 50,
      barWidth: 2,
      barGap: 3,
      cursorWidth: 2,
      cursorColor: "#ddd5e9",
      autoplay: gradio.props.autoplay,
      barRadius: 10,
      dragToSeek: true,
      normalize: true,
      minPxPerSec: 20,
      waveColor: gradio.props.waveform_options.waveform_color || "#9ca3af",
      progressColor: gradio.props.waveform_options.waveform_progress_color || color_accent,
      mediaControls: gradio.props.waveform_options.show_controls ?? false,
      sampleRate: gradio.props.waveform_options.sample_rate || 44100
    };
    const trim_region_settings = {
      color: gradio.props.waveform_options.trim_region_color,
      drag: true,
      resize: true
    };
    function handle_error(detail) {
      const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];
      if (gradio.shared.loading_status) {
        gradio.shared.loading_status.status = status;
        gradio.shared.loading_status.message = detail;
      }
      gradio.dispatch(level, detail);
    }
    gradio.watch_for_change();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          variant: "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          allow_overflow: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          visible: gradio.shared.visible,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            StaticAudio($$renderer4, {
              i18n: gradio.i18n,
              show_label: gradio.shared.show_label,
              buttons: gradio.props.buttons ?? ["download", "share"],
              value: gradio.props.value,
              subtitles: gradio.props.subtitles,
              label,
              loop: gradio.props.loop,
              waveform_settings,
              waveform_options: gradio.props.waveform_options,
              editable: gradio.props.editable,
              minimal,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              onshare: (detail) => gradio.dispatch("share", detail),
              onerror: (e) => gradio.dispatch("error", e.detail),
              onplay: () => gradio.dispatch("play"),
              onpause: () => gradio.dispatch("pause"),
              onstop: () => gradio.dispatch("stop"),
              get playback_position() {
                return gradio.props.playback_position;
              },
              set playback_position($$value) {
                gradio.props.playback_position = $$value;
                $$settled = false;
              }
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        Block($$renderer3, {
          variant: gradio.props.value === null && active_source === "upload" ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          allow_overflow: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          visible: gradio.shared.visible,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            InteractiveAudio($$renderer4, {
              label,
              show_label: gradio.shared.show_label,
              buttons: gradio.props.buttons ?? [],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              value: gradio.props.value,
              subtitles: gradio.props.subtitles,
              onchange: (detail) => gradio.props.value = detail,
              onstream: (detail) => {
                gradio.props.value = detail;
                gradio.dispatch("stream", gradio.props.value);
              },
              ondrag: (detail) => dragging = detail,
              root: gradio.shared.root,
              sources: gradio.props.sources,
              active_source: active_source || void 0,
              pending: gradio.shared.loading_status.pending,
              streaming: gradio.props.streaming,
              loop: gradio.props.loop,
              max_file_size: gradio.shared.max_file_size,
              handle_reset_value,
              editable: gradio.props.editable,
              onedit: () => gradio.dispatch("edit"),
              onplay: () => gradio.dispatch("play"),
              onpause: () => gradio.dispatch("pause"),
              onstop: () => gradio.dispatch("stop"),
              onstart_recording: () => gradio.dispatch("start_recording"),
              onpause_recording: () => gradio.dispatch("pause_recording"),
              onstop_recording: () => {
                gradio.dispatch("stop_recording");
                gradio.dispatch("input");
              },
              onupload: () => {
                gradio.dispatch("upload");
                gradio.dispatch("input");
              },
              onclear: () => {
                gradio.dispatch("clear");
                gradio.dispatch("input");
              },
              onerror: handle_error,
              onclose_stream: () => gradio.dispatch("close_stream", "stream"),
              i18n: gradio.i18n,
              waveform_settings,
              waveform_options: gradio.props.waveform_options,
              trim_region_settings,
              stream_every: gradio.props.stream_every,
              stream_state: gradio.shared.loading_status.stream_state,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: (...args) => gradio.shared.client.stream(...args),
              time_limit: gradio.shared.loading_status.time_limit,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get initial_value() {
                return initial_value;
              },
              set initial_value($$value) {
                initial_value = $$value;
                $$settled = false;
              },
              get recording() {
                return recording;
              },
              set recording($$value) {
                recording = $$value;
                $$settled = false;
              },
              get dragging() {
                return dragging;
              },
              set dragging($$value) {
                dragging = $$value;
                $$settled = false;
              },
              get playback_position() {
                return gradio.props.playback_position;
              },
              set playback_position($$value) {
                gradio.props.playback_position = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                UploadText($$renderer5, { i18n: gradio.i18n, type: "audio" });
              },
              $$slots: { default: true }
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { InteractiveAudio as BaseInteractiveAudio, AudioPlayer as BasePlayer, StaticAudio as BaseStaticAudio, Index as default };
//# sourceMappingURL=index53-37ifDFpM.js.map
