import './async-D55cHugf.js';
import { d as bind_props, c as spread_props } from './index-u8mz_F03.js';
import { U as Upload } from './Upload2-OQ1gRY0J.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import { V as Video } from './Video-_1zl9-Cr.js';
import './2-BAeILJ1g.js';
import { S as SelectSource } from './SelectSource-CeUXfvCc.js';
import { W as Webcam } from './Webcam2-Bm2qe3t4.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { P as Player, p as prettyBytes, V as VideoPreview } from './VideoPreview-BUKiZR1J.js';
export { l as loaded, a as playable } from './VideoPreview-BUKiZR1J.js';
export { default as BaseExample } from './Example28-D1-aV4K8.js';
import { B as Block } from './Block-JRVGROlG.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
import { S as Static } from './index3-BSof0MuO.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './Upload-CIQ-D6yx.js';
import './Microphone-CCAKTpuQ.js';
import './Webcam-iV1BaoQQ.js';
import './StreamingBar-B1BHh5xK.js';
import './DownloadLink-BMNxSkQ8.js';
import './IconButton-D82v5nCM.js';
import './Empty-BHvMDNv5.js';
import './ShareButton-DnAPtcgb.js';
import './Download-ByiErn53.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './Maximize-B77VDSzq.js';
import './VolumeLevels-C0Xso12f.js';
import './Play-BlZWjD1q.js';
import './Undo-Col2BcUY.js';
import './Video2-De5bYQY6.js';
import './ModifyUpload-D_fhEqLO.js';
import './Clear-DH-TDCgr.js';
import './Edit-W_0aHh0i.js';
import './clone-Yk88IHKV.js';

function InteractiveVideo($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = null,
      subtitle = null,
      sources = ["webcam", "upload"],
      label = void 0,
      show_download_button = false,
      show_label = true,
      webcam_options,
      include_audio,
      autoplay,
      root,
      i18n,
      active_source: initial_active_source = "webcam",
      handle_reset_value = () => {
      },
      max_file_size = null,
      upload,
      stream_handler,
      loop,
      uploading = void 0,
      upload_promise = void 0,
      playback_position = void 0,
      buttons = null,
      on_custom_button_click = null,
      onchange,
      onclear,
      onplay,
      onpause,
      onend,
      ondrag,
      onerror,
      onupload,
      onstart_recording,
      onstop_recording,
      onstop,
      children
    } = $$props;
    let has_change_history = false;
    let active_source = /* @__PURE__ */ (() => {
      return initial_active_source ?? "webcam";
    })();
    function handle_load(detail) {
      value = detail;
      onchange?.(detail);
      if (detail) {
        onupload?.(detail);
      }
    }
    function handle_clear() {
      value = null;
      onchange?.(null);
      onclear?.();
    }
    function handle_change(video) {
      has_change_history = true;
      onchange?.(video);
    }
    let dragging = false;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: Video, label: label || "Video" });
      $$renderer3.push(`<!----> <div data-testid="video" class="video-container svelte-ey25pz">`);
      if (value === null || value?.url === void 0) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="upload-container svelte-ey25pz">`);
        if (active_source === "upload") {
          $$renderer3.push("<!--[-->");
          Upload($$renderer3, {
            filetype: "video/x-m4v,video/*",
            onload: handle_load,
            max_file_size,
            onerror: (detail) => onerror?.(detail),
            root,
            upload,
            stream_handler,
            aria_label: i18n("video.drop_to_upload"),
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
          if (active_source === "webcam") {
            $$renderer3.push("<!--[-->");
            Webcam($$renderer3, {
              root,
              mirror_webcam: webcam_options.mirror,
              webcam_constraints: webcam_options.constraints,
              include_audio,
              mode: "video",
              i18n,
              upload,
              stream_every: 1
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
        if (value?.url) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<!---->`);
          {
            Player($$renderer3, {
              upload,
              root,
              interactive: true,
              autoplay,
              src: value.url,
              subtitle: subtitle?.url,
              is_stream: false,
              onplay: () => onplay?.(),
              onpause: () => onpause?.(),
              onstop: () => onstop?.(),
              onend: () => onend?.(),
              onerror: (error) => onerror?.(error),
              mirror: webcam_options.mirror && active_source === "webcam",
              label,
              handle_change,
              handle_reset_value,
              loop,
              value,
              i18n,
              show_download_button,
              handle_clear,
              has_change_history,
              get playback_position() {
                return playback_position;
              },
              set playback_position($$value) {
                playback_position = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer3.push(`<!---->`);
        } else {
          $$renderer3.push("<!--[!-->");
          if (value.size) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<div class="file-name svelte-ey25pz">${escape_html(value.orig_name || value.url)}</div> <div class="file-size svelte-ey25pz">${escape_html(prettyBytes(value.size))}</div>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]--> `);
      SelectSource($$renderer3, {
        sources,
        handle_clear,
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
    bind_props($$props, { value, uploading, upload_promise, playback_position });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    let upload_promise = void 0;
    class VideoGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const gradio = new VideoGradio(props);
    gradio.props.value;
    let uploading = false;
    let dragging = false;
    let active_source = (() => gradio.props.sources ? gradio.props.sources[0] : void 0)();
    let initial_value = gradio.props.value;
    const handle_reset_value = () => {
      if (initial_value === null || gradio.props.value === initial_value) {
        return;
      }
      gradio.props.value = initial_value;
    };
    function handle_change(detail) {
      if (detail != null) {
        gradio.props.value = detail;
      } else {
        gradio.props.value = null;
      }
    }
    function handle_error(detail) {
      const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];
      gradio.shared.loading_status.status = status;
      gradio.shared.loading_status.message = detail;
      gradio.dispatch(level, detail);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: gradio.props.value === null && active_source === "upload" ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          allow_overflow: false,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            VideoPreview($$renderer4, {
              value: gradio.props.value,
              subtitle: gradio.props.subtitles,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              autoplay: gradio.props.autoplay,
              loop: gradio.props.loop,
              buttons: gradio.props.buttons ?? ["download", "share"],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              onplay: () => gradio.dispatch("play"),
              onpause: () => gradio.dispatch("pause"),
              onstop: () => gradio.dispatch("stop"),
              onend: () => gradio.dispatch("end"),
              onshare: (detail) => gradio.dispatch("share", detail),
              onerror: (detail) => gradio.dispatch("error", detail),
              i18n: gradio.i18n,
              upload: (...args) => gradio.shared.client.upload(...args),
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
          visible: gradio.shared.visible,
          variant: gradio.props.value === null && active_source === "upload" ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          allow_overflow: false,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            InteractiveVideo($$renderer4, {
              value: gradio.props.value,
              subtitle: gradio.props.subtitles,
              onchange: handle_change,
              ondrag: (detail) => dragging = detail,
              onerror: handle_error,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              buttons: gradio.props.buttons ?? ["download", "share"],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              sources: gradio.props.sources,
              active_source,
              webcam_options: gradio.props.webcam_options,
              include_audio: gradio.props.include_audio,
              autoplay: gradio.props.autoplay,
              root: gradio.shared.root,
              loop: gradio.props.loop,
              handle_reset_value,
              onclear: () => {
                gradio.props.value = null;
                gradio.dispatch("clear");
                gradio.dispatch("input");
              },
              onplay: () => gradio.dispatch("play"),
              onpause: () => gradio.dispatch("pause"),
              onupload: () => {
                gradio.dispatch("upload");
                gradio.dispatch("input");
              },
              onstop: () => gradio.dispatch("stop"),
              onend: () => gradio.dispatch("end"),
              onstart_recording: () => gradio.dispatch("start_recording"),
              onstop_recording: () => gradio.dispatch("stop_recording"),
              i18n: gradio.i18n,
              max_file_size: gradio.shared.max_file_size,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: (...args) => gradio.shared.client.stream(...args),
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get uploading() {
                return uploading;
              },
              set uploading($$value) {
                uploading = $$value;
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
                UploadText($$renderer5, { i18n: gradio.i18n, type: "video" });
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

export { InteractiveVideo as BaseInteractiveVideo, Player as BasePlayer, VideoPreview as BaseStaticVideo, Index as default, prettyBytes };
//# sourceMappingURL=index43-DMLSOqb-.js.map
