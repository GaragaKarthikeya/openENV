import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, a as attr } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value ??= 0;
    gradio.props.value;
    const disabled = !gradio.shared.interactive;
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      padding: gradio.shared.container,
      allow_overflow: false,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      children: ($$renderer3) => {
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            show_validation_error: false,
            on_clear_status: () => {
              gradio.dispatch("clear_status", gradio.shared.loading_status);
            }
          }
        ]));
        $$renderer3.push(`<!----> <label${attr_class("block svelte-16ty2ow", void 0, { "container": gradio.shared.container })}>`);
        if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
          $$renderer3.push("<!--[-->");
          IconButtonWrapper($$renderer3, {
            buttons: gradio.props.buttons,
            on_custom_button_click: (id) => {
              gradio.dispatch("custom_button_click", { id });
            }
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        BlockTitle($$renderer3, {
          show_label: gradio.shared.show_label,
          info: gradio.props.info,
          children: ($$renderer4) => {
            $$renderer4.push(`<!---->${escape_html(gradio.shared.label || "Number")} `);
            if (gradio.shared.loading_status?.validation_error) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<div class="validation-error svelte-16ty2ow">${escape_html(gradio.shared.loading_status?.validation_error)}</div>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----> <input${attr("aria-label", gradio.shared.label || "Number")} type="number"${attr("value", gradio.props.value)}${attr("min", gradio.props.minimum)}${attr("max", gradio.props.maximum)}${attr("step", gradio.props.step)}${attr("placeholder", gradio.props.placeholder)}${attr("disabled", disabled, true)}${attr_class("svelte-16ty2ow", void 0, {
          "validation-error": gradio.shared.loading_status?.validation_error
        })}/></label>`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index28-BmqNkw5i.js.map
