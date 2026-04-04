import './async-D55cHugf.js';
import { a as attr, i as stringify } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Button } from './Button-C8geAOuh.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Image-BLyvTcTy.js';

function UploadButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      elem_id = "",
      elem_classes = [],
      visible = true,
      label,
      value,
      file_count,
      file_types = [],
      root,
      size = "lg",
      icon = null,
      scale = null,
      min_width = void 0,
      variant = "secondary",
      disabled = false,
      max_file_size = null,
      upload,
      onclick,
      onchange,
      onupload,
      onerror,
      children
    } = $$props;
    let hidden_upload;
    let accept_file_types = (() => {
      if (file_types == null) {
        return null;
      }
      const mapped = file_types.map((x) => {
        if (x.startsWith(".")) {
          return x;
        }
        return x + "/*";
      });
      return mapped.join(", ");
    })();
    function open_file_upload() {
      onclick?.();
      hidden_upload.click();
    }
    $$renderer2.push(`<input class="hide svelte-94gmgt"${attr("accept", accept_file_types)} type="file"${attr("multiple", file_count === "multiple" || void 0, true)}${attr("webkitdirectory", file_count === "directory" || void 0, true)}${attr("mozdirectory", file_count === "directory" || void 0)}${attr("data-testid", `${stringify(label)}-upload-button`)}/> `);
    Button($$renderer2, {
      size,
      variant,
      elem_id,
      elem_classes,
      visible,
      onclick: open_file_upload,
      scale,
      min_width,
      disabled,
      children: ($$renderer3) => {
        if (icon) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<img class="button-icon svelte-94gmgt"${attr("src", icon.url)}${attr("alt", `${value} icon`)}/>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (children) {
          $$renderer3.push("<!--[-->");
          children($$renderer3);
          $$renderer3.push(`<!---->`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]-->`);
      }
    });
    $$renderer2.push(`<!---->`);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let value = gradio.props.value;
    async function handle_event(detail, event) {
      gradio.props.value = detail;
      gradio.dispatch(event);
    }
    const disabled = !gradio.shared.interactive;
    UploadButton($$renderer2, {
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      visible: gradio.shared.visible,
      file_count: gradio.props.file_count,
      file_types: gradio.props.file_types,
      size: gradio.props.size,
      scale: gradio.shared.scale,
      icon: gradio.props.icon,
      min_width: gradio.shared.min_width,
      root: gradio.shared.root,
      value,
      disabled,
      variant: gradio.props.variant,
      label: gradio.shared.label,
      max_file_size: gradio.shared.max_file_size,
      onclick: () => gradio.dispatch("click"),
      onchange: (detail) => handle_event(detail, "change"),
      onupload: (detail) => handle_event(detail, "upload"),
      onerror: (detail) => {
        gradio.dispatch("error", detail);
      },
      upload: (...args) => gradio.shared.client.upload(...args),
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(gradio.shared.label ?? "")}`);
      }
    });
  });
}

export { UploadButton as BaseUploadButton, Index as default };
//# sourceMappingURL=Index38-CNGW8M28.js.map
