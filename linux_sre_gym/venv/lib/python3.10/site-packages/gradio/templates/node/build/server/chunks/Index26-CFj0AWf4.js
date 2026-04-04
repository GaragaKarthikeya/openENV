import { f as fallback } from './async-D55cHugf.js';
import { d as bind_props, f as attr_class, g as attr_style, c as spread_props, s as slot } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import ImagePreview from './ImagePreview-CjXAuu8E.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { I as Image } from './Image2-BLbvQKZw.js';
import { S as SelectSource } from './SelectSource-CeUXfvCc.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FullscreenButton } from './FullscreenButton-DVAVExlR.js';
import { W as Webcam } from './Webcam2-Bm2qe3t4.js';
import { U as Upload, a as UploadProgress } from './Upload2-OQ1gRY0J.js';
import { I as Image$1 } from './Image-BLyvTcTy.js';
import { B as Block } from './Block-JRVGROlG.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example6-CY1BLAOH.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './DownloadLink-BMNxSkQ8.js';
import './ShareButton-DnAPtcgb.js';
import './Download-ByiErn53.js';
import './Maximize-B77VDSzq.js';
import './Upload-CIQ-D6yx.js';
import './Microphone-CCAKTpuQ.js';
import './Video-_1zl9-Cr.js';
import './Webcam-iV1BaoQQ.js';
import './StreamingBar-B1BHh5xK.js';

function ImageUploader($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let active_streaming;
    let value = fallback($$props["value"], null);
    let label = fallback($$props["label"], void 0);
    let show_label = $$props["show_label"];
    let sources = fallback($$props["sources"], () => ["upload", "clipboard", "webcam"], true);
    let streaming = fallback($$props["streaming"], false);
    let pending = fallback($$props["pending"], false);
    let webcam_options = $$props["webcam_options"];
    let selectable = fallback($$props["selectable"], false);
    let root = $$props["root"];
    let i18n = $$props["i18n"];
    let max_file_size = fallback($$props["max_file_size"], null);
    let upload = $$props["upload"];
    let stream_handler = $$props["stream_handler"];
    let stream_every = $$props["stream_every"];
    let time_limit = $$props["time_limit"];
    let show_fullscreen_button = fallback($$props["show_fullscreen_button"], true);
    let stream_state = fallback($$props["stream_state"], "closed");
    let upload_promise = fallback($$props["upload_promise"], null);
    let onerror = fallback($$props["onerror"], void 0);
    let upload_input;
    let uploading = fallback($$props["uploading"], false);
    let active_source = fallback($$props["active_source"], null);
    let fullscreen = fallback($$props["fullscreen"], false);
    let files = [];
    let upload_id;
    async function handle_upload(detail) {
      if (!streaming) {
        if (detail.path?.toLowerCase().endsWith(".svg") && detail.url) {
          const response = await fetch(detail.url);
          const svgContent = await response.text();
          value = {
            ...detail,
            url: `data:image/svg+xml,${encodeURIComponent(svgContent)}`
          };
        } else {
          value = detail;
        }
        await tick();
      }
    }
    function handle_clear() {
      value = null;
    }
    function handle_remove_image_click(event) {
      handle_clear();
      event.stopPropagation();
    }
    let dragging = fallback($$props["dragging"], false);
    async function handle_select_source(source) {
      switch (source) {
        case "clipboard":
          upload_input.paste_clipboard();
          break;
      }
    }
    if (!active_source && sources) {
      active_source = sources[0];
    }
    active_streaming = streaming && active_source === "webcam";
    if (uploading && !active_streaming) value = null;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: Image, label: label || "Image" });
      $$renderer3.push(`<!----> <div data-testid="image" class="image-container svelte-6uxbr3">`);
      IconButtonWrapper($$renderer3, {
        children: ($$renderer4) => {
          if (value?.url && !active_streaming) {
            $$renderer4.push("<!--[-->");
            if (show_fullscreen_button) {
              $$renderer4.push("<!--[-->");
              FullscreenButton($$renderer4, { fullscreen });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            IconButton($$renderer4, {
              Icon: Clear,
              label: "Remove Image",
              onclick: handle_remove_image_click
            });
            $$renderer4.push(`<!---->`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]-->`);
        }
      });
      $$renderer3.push(`<!----> <div${attr_class("upload-container svelte-6uxbr3", void 0, { "reduced-height": sources.length > 1 })}${attr_style("", { width: value ? "auto" : "100%" })}>`);
      Upload($$renderer3, {
        hidden: value !== null || active_source === "webcam",
        filetype: active_source === "clipboard" ? "clipboard" : "image/*",
        onload: handle_upload,
        onerror,
        root,
        max_file_size,
        disable_click: !sources.includes("upload") || value !== null,
        upload,
        stream_handler,
        aria_label: i18n("image.drop_to_upload"),
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
        get dragging() {
          return dragging;
        },
        set dragging($$value) {
          dragging = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          if (value === null) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<!--[-->`);
            slot($$renderer4, $$props, "default", {});
            $$renderer4.push(`<!--]-->`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> `);
      if (active_source === "webcam" && !streaming && pending) {
        $$renderer3.push("<!--[-->");
        UploadProgress($$renderer3, { root, upload_id, stream_handler, files });
      } else {
        $$renderer3.push("<!--[!-->");
        if (active_source === "webcam" && (streaming || !streaming && !value)) {
          $$renderer3.push("<!--[-->");
          Webcam($$renderer3, {
            root,
            value,
            stream_state,
            mirror_webcam: webcam_options.mirror,
            stream_every,
            streaming,
            mode: "image",
            include_audio: false,
            i18n,
            upload,
            time_limit,
            webcam_constraints: webcam_options.constraints
          });
        } else {
          $$renderer3.push("<!--[!-->");
          if (value !== null && !streaming) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<div${attr_class("image-frame svelte-6uxbr3", void 0, { "selectable": selectable })}>`);
            Image$1($$renderer3, { src: value.url, restProps: { alt: value.alt_text } });
            $$renderer3.push(`<!----></div>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]--></div> `);
      if (sources.length > 1 || sources.includes("clipboard")) {
        $$renderer3.push("<!--[-->");
        SelectSource($$renderer3, {
          sources,
          handle_clear,
          handle_select: handle_select_source,
          get active_source() {
            return active_source;
          },
          set active_source($$value) {
            active_source = $$value;
            $$settled = false;
          }
        });
      } else {
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
    bind_props($$props, {
      value,
      label,
      show_label,
      sources,
      streaming,
      pending,
      webcam_options,
      selectable,
      root,
      i18n,
      max_file_size,
      upload,
      stream_handler,
      stream_every,
      time_limit,
      show_fullscreen_button,
      stream_state,
      upload_promise,
      onerror,
      uploading,
      active_source,
      fullscreen,
      dragging
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let stream_data = { value: null };
    let upload_promise = void 0;
    class ImageGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        if (props.props.streaming) {
          data.value = stream_data.value;
        }
        return data;
      }
    }
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new ImageGradio(props);
    let fullscreen = false;
    let dragging = false;
    let active_source = (() => gradio.props.sources ? gradio.props.sources[0] : null)();
    gradio.props.value;
    let status = gradio?.shared?.loading_status.stream_state;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          allow_overflow: false,
          container: gradio.shared.container,
          "scale{gradio.shared.scale}": true,
          min_width: gradio.shared.min_width,
          get fullscreen() {
            return fullscreen;
          },
          set fullscreen($$value) {
            fullscreen = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            ImagePreview($$renderer4, {
              fullscreen,
              value: gradio.props.value,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              selectable: gradio.props._selectable,
              i18n: gradio.i18n,
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
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
          variant: gradio.props.value === null ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          allow_overflow: false,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          get fullscreen() {
            return fullscreen;
          },
          set fullscreen($$value) {
            fullscreen = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            if (gradio.shared.loading_status.type === "output") {
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
            ImageUploader($$renderer4, {
              selectable: gradio.props._selectable,
              root: gradio.shared.root,
              sources: gradio.props.sources,
              fullscreen,
              show_fullscreen_button: gradio.props.buttons === null ? true : gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "fullscreen"),
              onerror: (detail) => {
                gradio.shared.loading_status.status = "error";
                gradio.dispatch("error", detail);
              },
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              pending: gradio.shared.loading_status?.status === "pending" || gradio.shared.loading_status?.status === "streaming",
              streaming: gradio.props.streaming,
              webcam_options: gradio.props.webcam_options,
              stream_every: gradio.props.stream_every,
              time_limit: gradio.shared.loading_status?.time_limit,
              max_file_size: gradio.shared.max_file_size,
              i18n: gradio.i18n,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: gradio.shared.client?.stream,
              stream_state: status,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get active_source() {
                return active_source;
              },
              set active_source($$value) {
                active_source = $$value;
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
              children: ($$renderer5) => {
                if (active_source === "upload" || !active_source) {
                  $$renderer5.push("<!--[-->");
                  UploadText($$renderer5, {
                    i18n: gradio.i18n,
                    type: "image",
                    placeholder: gradio.props.placeholder
                  });
                } else {
                  $$renderer5.push("<!--[!-->");
                  if (active_source === "clipboard") {
                    $$renderer5.push("<!--[-->");
                    UploadText($$renderer5, { i18n: gradio.i18n, type: "clipboard", mode: "short" });
                  } else {
                    $$renderer5.push("<!--[!-->");
                    Empty($$renderer5, {
                      unpadded_box: true,
                      size: "large",
                      children: ($$renderer6) => {
                        Image($$renderer6);
                      },
                      $$slots: { default: true }
                    });
                  }
                  $$renderer5.push(`<!--]-->`);
                }
                $$renderer5.push(`<!--]-->`);
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

export { Image$1 as BaseImage, ImageUploader as BaseImageUploader, ImagePreview as BaseStaticImage, Webcam, Index as default };
//# sourceMappingURL=Index26-CFj0AWf4.js.map
