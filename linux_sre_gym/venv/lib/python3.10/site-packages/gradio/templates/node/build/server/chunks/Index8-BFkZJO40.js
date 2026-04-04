import './async-D55cHugf.js';
import { c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { I as Info } from './Info-BeS-ygt-.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
import { C as Checkbox } from './Checkbox-COert_L_.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './html-CfyvkLET.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
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
          Checkbox($$renderer4, {
            label: gradio.shared.label || gradio.i18n("checkbox.checkbox"),
            interactive: gradio.shared.interactive,
            show_label: gradio.shared.show_label,
            on_change: (val) => gradio.dispatch("change", val),
            on_input: () => gradio.dispatch("input"),
            on_select: (data) => gradio.dispatch("select", data),
            get value() {
              return gradio.props.value;
            },
            set value($$value) {
              gradio.props.value = $$value;
              $$settled = false;
            }
          });
          $$renderer4.push(`<!----> `);
          if (gradio.props.info) {
            $$renderer4.push("<!--[-->");
            Info($$renderer4, { info: gradio.props.info });
          } else {
            $$renderer4.push("<!--[!-->");
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

export { Checkbox as BaseCheckbox, Index as default };
//# sourceMappingURL=Index8-BFkZJO40.js.map
