import './async-D55cHugf.js';
import { a as attr } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Button } from './Button-C8geAOuh.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Image-BLyvTcTy.js';

function DownloadButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      elem_id = "",
      elem_classes = [],
      visible = true,
      variant = "secondary",
      size = "lg",
      value,
      icon,
      disabled = false,
      scale = null,
      min_width = void 0,
      on_click,
      children
    } = $$props;
    function download_file() {
      on_click?.();
      if (!value?.url) {
        return;
      }
      let file_name;
      if (!value.orig_name && value.url) {
        const parts = value.url.split("/");
        file_name = parts[parts.length - 1];
        file_name = file_name.split("?")[0].split("#")[0];
      } else {
        file_name = value.orig_name;
      }
      const a = document.createElement("a");
      a.href = value.url;
      a.download = file_name || "file";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    Button($$renderer2, {
      size,
      variant,
      elem_id,
      elem_classes,
      visible,
      onclick: download_file,
      scale,
      min_width,
      disabled,
      children: ($$renderer3) => {
        if (icon) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<img class="button-icon svelte-4ac0fl"${attr("src", icon.url)}${attr("alt", `${value} icon`)}/>`);
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
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    DownloadButton($$renderer2, {
      value: gradio.props.value,
      variant: gradio.props.variant,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      size: gradio.props.size,
      scale: gradio.shared.scale,
      icon: gradio.props.icon,
      min_width: gradio.shared.min_width,
      visible: gradio.shared.visible,
      disabled: !gradio.shared.interactive,
      on_click: () => gradio.dispatch("click"),
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(gradio.shared.label ?? "")}`);
      }
    });
  });
}

export { DownloadButton as BaseButton, Index as default };
//# sourceMappingURL=Index33-BNqtueh_.js.map
