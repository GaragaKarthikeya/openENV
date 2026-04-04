import './async-D55cHugf.js';
import { d as bind_props, c as spread_props, s as slot } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import Model3D from './Model3D-Byrr9Lao.js';
import { U as Upload } from './Upload2-OQ1gRY0J.js';
import { M as ModifyUpload } from './ModifyUpload-D_fhEqLO.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { F as File } from './File-2S6P7zIO.js';
import { B as Block } from './Block-JRVGROlG.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example22-Bx6IyqEy.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Download-ByiErn53.js';
import './Undo-Col2BcUY.js';
import './DownloadLink-BMNxSkQ8.js';
import './Clear-DH-TDCgr.js';
import './Edit-W_0aHh0i.js';
import './Upload-CIQ-D6yx.js';

function Model3DUpload($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = void 0,
      display_mode = "solid",
      clear_color = [0, 0, 0, 0],
      label = "",
      show_label,
      root,
      i18n,
      zoom_speed = 1,
      pan_speed = 1,
      max_file_size = null,
      uploading = void 0,
      upload_promise = void 0,
      camera_position = [null, null, null],
      upload,
      stream_handler,
      onchange,
      onclear,
      ondrag,
      onload,
      onerror
    } = $$props;
    let dragging = false;
    async function handle_upload(detail) {
      value = detail;
      await tick();
      onchange?.(value);
      onload?.(value);
    }
    async function handle_clear() {
      value = null;
      await tick();
      onclear?.();
      onchange?.(null);
    }
    async function handle_undo() {
    }
    function handle_error(error) {
      onerror?.(error);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: File, label: label || "3D Model" });
      $$renderer3.push(`<!----> `);
      if (value == null) {
        $$renderer3.push("<!--[-->");
        Upload($$renderer3, {
          upload,
          stream_handler,
          onload: handle_upload,
          root,
          max_file_size,
          filetype: [
            ".stl",
            ".obj",
            ".gltf",
            ".glb",
            "model/obj",
            ".splat",
            ".ply"
          ],
          onerror: handle_error,
          aria_label: i18n("model3d.drop_to_upload"),
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
            $$renderer4.push(`<!--[-->`);
            slot($$renderer4, $$props, "default", {});
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<div class="input-model svelte-18wa0f8">`);
        ModifyUpload($$renderer3, {
          undoable: true,
          onclear: handle_clear,
          i18n,
          onundo: handle_undo
        });
        $$renderer3.push(`<!----> `);
        {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`<!---->`);
          $$renderer3.push(`<!---->`);
        }
        $$renderer3.push(`<!--]--></div>`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, uploading, upload_promise });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    class Model3dGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Model3dGradio(props);
    gradio.props.value;
    let uploading = false;
    let dragging = false;
    let has_change_history = false;
    let upload_promise = void 0;
    const is_browser = typeof window !== "undefined";
    function handle_change(detail) {
      gradio.props.value = detail;
      gradio.dispatch("change", detail);
      has_change_history = true;
    }
    function handle_drag(detail) {
      dragging = detail;
    }
    function handle_clear() {
      gradio.props.value = null;
      gradio.dispatch("clear");
    }
    function handle_load(detail) {
      gradio.props.value = detail;
      gradio.dispatch("upload");
    }
    function handle_error(detail) {
      if (gradio.shared.loading_status) gradio.shared.loading_status.status = "error";
      gradio.dispatch("error", detail);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: gradio.props.value === null ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          height: gradio.props.height,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            if (gradio.props.value && is_browser) {
              $$renderer4.push("<!--[-->");
              Model3D($$renderer4, {
                value: gradio.props.value,
                i18n: gradio.i18n,
                display_mode: gradio.props.display_mode,
                clear_color: gradio.props.clear_color,
                label: gradio.shared.label,
                show_label: gradio.shared.show_label,
                camera_position: gradio.props.camera_position,
                zoom_speed: gradio.props.zoom_speed,
                has_change_history
              });
            } else {
              $$renderer4.push("<!--[!-->");
              if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
                $$renderer4.push("<!--[-->");
                IconButtonWrapper($$renderer4, {
                  buttons: gradio.props.buttons,
                  on_custom_button_click: (id) => {
                    gradio.dispatch("custom_button_click", { id });
                  }
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              BlockLabel($$renderer4, {
                show_label: gradio.shared.show_label,
                Icon: File,
                label: gradio.shared.label || "3D Model"
              });
              $$renderer4.push(`<!----> `);
              Empty($$renderer4, {
                unpadded_box: true,
                size: "large",
                children: ($$renderer5) => {
                  File($$renderer5);
                },
                $$slots: { default: true }
              });
              $$renderer4.push(`<!---->`);
            }
            $$renderer4.push(`<!--]-->`);
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
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          height: gradio.props.height,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            Model3DUpload($$renderer4, {
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              root: gradio.shared.root,
              display_mode: gradio.props.display_mode,
              clear_color: gradio.props.clear_color,
              camera_position: gradio.props.camera_position,
              zoom_speed: gradio.props.zoom_speed,
              onchange: handle_change,
              ondrag: handle_drag,
              onclear: handle_clear,
              onload: handle_load,
              onerror: handle_error,
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
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              },
              get uploading() {
                return uploading;
              },
              set uploading($$value) {
                uploading = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                UploadText($$renderer5, { i18n: gradio.i18n, type: "file" });
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

export { Model3D as BaseModel3D, Model3DUpload as BaseModel3DUpload, Index as default };
//# sourceMappingURL=Index35-C85cD6o-.js.map
