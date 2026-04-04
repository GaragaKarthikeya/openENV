import './async-D55cHugf.js';
import { a as attr, g as attr_style, d as bind_props, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { t as tinycolor } from './tinycolor-DVnMVnIq.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example13-8uwQHwRN.js';
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

function format_color(color, mode) {
  {
    return tinycolor(color).toHexString();
  }
}
function Colorpicker($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = void 0,
      label,
      info,
      disabled,
      show_label,
      on_input = () => {
      },
      on_submit = () => {
      },
      on_blur = () => {
      },
      on_focus = () => {
      }
    } = $$props;
    (() => format_color(value))();
    BlockTitle($$renderer2, {
      show_label,
      info,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(label)}`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <button class="dialog-button svelte-nbn1m9"${attr("disabled", disabled, true)}${attr_style("", { background: value })}></button> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { value });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props, { value: "#000000" });
    gradio.props.value;
    let label = gradio.shared.label || gradio.i18n("color_picker.color_picker");
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          Colorpicker($$renderer4, {
            label,
            info: gradio.props.info,
            show_label: gradio.shared.show_label,
            disabled: !gradio.shared.interactive,
            on_input: () => gradio.dispatch("input"),
            on_submit: () => gradio.dispatch("submit"),
            on_blur: () => gradio.dispatch("blur"),
            on_focus: () => gradio.dispatch("focus"),
            get value() {
              return gradio.props.value;
            },
            set value($$value) {
              gradio.props.value = $$value;
              $$settled = false;
            }
          });
          $$renderer4.push(`<!---->`);
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

export { Colorpicker as BaseColorPicker, Index as default };
//# sourceMappingURL=Index40-yihVSKCF.js.map
