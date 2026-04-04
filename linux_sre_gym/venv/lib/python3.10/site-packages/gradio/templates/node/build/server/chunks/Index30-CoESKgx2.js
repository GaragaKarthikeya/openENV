import './async-D55cHugf.js';
import { c as spread_props, a as attr } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
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

let _id = 0;
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    let gradio = new Gradio(props);
    gradio.props.value;
    gradio.props.value;
    const id = `range_id_${_id++}`;
    let minimum_value = gradio.props.minimum ?? 0;
    (() => {
      const min = gradio.props.minimum;
      const max = gradio.props.maximum;
      const val = gradio.props.value;
      if (val > max) {
        return 100;
      } else if (val < min) {
        return 0;
      }
      return (val - min) / (max - min) * 100;
    })();
    let disabled = !gradio.shared.interactive;
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container: gradio.shared.container,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      children: ($$renderer3) => {
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
          }
        ]));
        $$renderer3.push(`<!----> <div class="wrap svelte-8epfm4"><div class="head svelte-8epfm4"><label${attr("for", id)} class="svelte-8epfm4">`);
        BlockTitle($$renderer3, {
          show_label: gradio.shared.show_label,
          info: gradio.props.info,
          children: ($$renderer4) => {
            $$renderer4.push(`<!---->${escape_html(gradio.shared.label || "Slider")}`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></label> <div class="tab-like-container svelte-8epfm4"><input${attr("aria-label", `number input for ${gradio.shared.label}`)} data-testid="number-input" type="number"${attr("value", gradio.props.value)}${attr("min", gradio.props.minimum)}${attr("max", gradio.props.maximum)}${attr("step", gradio.props.step)}${attr("disabled", disabled, true)} class="svelte-8epfm4"/> `);
        if (gradio.props.buttons?.includes("reset") ?? true) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button class="reset-button svelte-8epfm4"${attr("disabled", disabled, true)} aria-label="Reset to default value" data-testid="reset-button">↺</button>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div></div> <div class="slider_input_container svelte-8epfm4"><span class="min_value svelte-8epfm4">${escape_html(minimum_value)}</span> <input type="range"${attr("id", id)} name="cowbell"${attr("value", gradio.props.value)}${attr("min", gradio.props.minimum)}${attr("max", gradio.props.maximum)}${attr("step", gradio.props.step)}${attr("disabled", disabled, true)}${attr("aria-label", `range slider for ${gradio.shared.label}`)} class="svelte-8epfm4"/> <span class="max_value svelte-8epfm4">${escape_html(gradio.props.maximum)}</span></div></div>`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index30-CoESKgx2.js.map
