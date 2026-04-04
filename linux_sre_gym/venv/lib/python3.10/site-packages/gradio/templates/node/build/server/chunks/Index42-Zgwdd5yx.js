import './async-D55cHugf.js';
import { a as attr, f as attr_class, i as stringify, d as bind_props, c as spread_props, e as ensure_array_like } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example26-4aUdI3US.js';
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

let id = 0;
function Radio($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      selected = void 0,
      display_value,
      internal_value,
      disabled,
      rtl,
      on_input
    } = $$props;
    let is_selected = selected === internal_value;
    $$renderer2.push(`<label${attr("data-testid", `${stringify(display_value)}-radio-label`)}${attr_class("svelte-19qdtil", void 0, { "disabled": disabled, "selected": is_selected, "rtl": rtl })}><input${attr("disabled", disabled, true)} type="radio"${attr("name", `radio-${stringify(++id)}`)}${attr("value", internal_value)}${attr("aria-checked", is_selected)}${attr("checked", selected === internal_value, true)} class="svelte-19qdtil"/> <span class="svelte-19qdtil">${escape_html(display_value)}</span></label>`);
    bind_props($$props, { selected });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let disabled = !gradio.shared.interactive;
    gradio.props.value;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        type: "fieldset",
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        rtl: gradio.props.rtl,
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
              on_custom_button_click: (id2) => {
                gradio.dispatch("custom_button_click", { id: id2 });
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          BlockTitle($$renderer4, {
            show_label: gradio.shared.show_label,
            info: gradio.props.info,
            children: ($$renderer5) => {
              $$renderer5.push(`<!---->${escape_html(gradio.shared.label || gradio.i18n("radio.radio"))}`);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> <div class="wrap svelte-e4x47i"><!--[-->`);
          const each_array = ensure_array_like(gradio.props.choices);
          for (let i = 0, $$length = each_array.length; i < $$length; i++) {
            let [display_value, internal_value] = each_array[i];
            Radio($$renderer4, {
              display_value,
              internal_value,
              disabled,
              rtl: gradio.props.rtl,
              on_input: () => {
                gradio.dispatch("input");
                gradio.dispatch("select", { value: internal_value, index: i });
              },
              get selected() {
                return gradio.props.value;
              },
              set selected($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer4.push(`<!--]--></div>`);
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

export { Radio as BaseRadio, Index as default };
//# sourceMappingURL=Index42-Zgwdd5yx.js.map
