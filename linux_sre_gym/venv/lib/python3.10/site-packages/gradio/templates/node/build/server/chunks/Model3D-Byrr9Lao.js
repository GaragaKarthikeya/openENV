import './async-D55cHugf.js';
import { a as attr } from './index-u8mz_F03.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { D as Download } from './Download-ByiErn53.js';
import { F as File } from './File-2S6P7zIO.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import './2-BAeILJ1g.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Model3D($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      display_mode = "solid",
      clear_color = [0, 0, 0, 0],
      label = "",
      show_label,
      i18n,
      zoom_speed = 1,
      pan_speed = 1,
      camera_position = [null, null, null],
      has_change_history = false
    } = $$props;
    function handle_undo() {
    }
    BlockLabel($$renderer2, {
      show_label,
      Icon: File,
      label: label || i18n("3D_model.3d_model")
    });
    $$renderer2.push(`<!----> `);
    if (value) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="model3D svelte-pnaihf" data-testid="model3d">`);
      IconButtonWrapper($$renderer2, {
        children: ($$renderer3) => {
          {
            $$renderer3.push("<!--[-->");
            IconButton($$renderer3, {
              Icon: Undo,
              label: "Undo",
              onclick: () => handle_undo(),
              disabled: !has_change_history
            });
          }
          $$renderer3.push(`<!--]--> <a${attr("href", value.url)}${attr("target", window.__is_colab__ ? "_blank" : null)}${attr("download", window.__is_colab__ ? null : value.orig_name || value.path)}>`);
          IconButton($$renderer3, { Icon: Download, label: i18n("common.download") });
          $$renderer3.push(`<!----></a>`);
        }
      });
      $$renderer2.push(`<!----> `);
      {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<!---->`);
        $$renderer2.push(`<!---->`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}

export { Model3D as default };
//# sourceMappingURL=Model3D-Byrr9Lao.js.map
