import './async-D55cHugf.js';
import { c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { F as File } from './File-2S6P7zIO.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FilePreview, a as FileUpload } from './FileUpload-Bz0iVHB2.js';
import { B as Block } from './Block-JRVGROlG.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example5-DyYnewTx.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Upload2-OQ1gRY0J.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';
import './Upload-CIQ-D6yx.js';
import './DownloadLink-BMNxSkQ8.js';
import './html-CfyvkLET.js';

function File_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      label,
      show_label,
      selectable,
      i18n,
      height,
      buttons = null,
      on_custom_button_click = null,
      on_select,
      on_download
    } = $$props;
    if (show_label && buttons && buttons.length > 0) {
      $$renderer2.push("<!--[-->");
      IconButtonWrapper($$renderer2, { buttons, on_custom_button_click });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    BlockLabel($$renderer2, {
      show_label,
      float: value === null,
      Icon: File,
      label: label || "File"
    });
    $$renderer2.push(`<!----> `);
    if (value && (Array.isArray(value) ? value.length > 0 : true)) {
      $$renderer2.push("<!--[-->");
      FilePreview($$renderer2, { i18n, selectable, value, height });
    } else {
      $$renderer2.push("<!--[!-->");
      Empty($$renderer2, {
        unpadded_box: true,
        size: "large",
        children: ($$renderer3) => {
          File($$renderer3);
        },
        $$slots: { default: true }
      });
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    let upload_promise = null;
    let dragging = false;
    class FileGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const gradio = new FileGradio(props);
    gradio.props.value;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        variant: gradio.props.value ? "solid" : "dashed",
        border_mode: dragging ? "focus" : "base",
        padding: false,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              status: gradio.shared.loading_status?.status || "complete",
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (!gradio.shared.interactive) {
            $$renderer4.push("<!--[-->");
            File_1($$renderer4, {
              on_select: ({ detail }) => gradio.dispatch("select", detail),
              on_download: ({ detail }) => gradio.dispatch("download", detail),
              selectable: gradio.props._selectable,
              value: gradio.props.value,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              height: gradio.props.height,
              i18n: gradio.i18n,
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
            FileUpload($$renderer4, {
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: (...args) => gradio.shared.client.stream(...args),
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              value: gradio.props.value,
              file_count: gradio.props.file_count,
              file_types: gradio.props.file_types,
              selectable: gradio.props._selectable,
              height: gradio.props.height ?? void 0,
              root: gradio.shared.root,
              allow_reordering: gradio.props.allow_reordering,
              max_file_size: gradio.shared.max_file_size,
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              onchange: (detail) => {
                gradio.props.value = detail;
              },
              ondrag: (detail) => dragging = detail,
              onclear: () => gradio.dispatch("clear"),
              onselect: (detail) => gradio.dispatch("select", detail),
              onupload: () => gradio.dispatch("upload"),
              onerror: (error) => {
                gradio.shared.loading_status = gradio.shared.loading_status || {};
                gradio.shared.loading_status.status = "error";
                gradio.dispatch("error", error);
              },
              ondelete: (detail) => {
                gradio.dispatch("delete", detail);
              },
              i18n: gradio.i18n,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                UploadText($$renderer5, { i18n: gradio.i18n, type: "file" });
              },
              $$slots: { default: true }
            });
          }
          $$renderer4.push(`<!--]-->`);
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

export { File_1 as BaseFile, FileUpload as BaseFileUpload, FilePreview, Index as default };
//# sourceMappingURL=Index22-DAGaNe0Y.js.map
