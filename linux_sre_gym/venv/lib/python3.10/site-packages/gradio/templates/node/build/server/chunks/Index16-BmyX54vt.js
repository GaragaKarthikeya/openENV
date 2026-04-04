import './async-D55cHugf.js';
import { c as spread_props } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import { T as Textbox } from './Textbox-T7HntrB1.js';
import { S as Static } from './index3-BSof0MuO.js';
import { B as Block } from './Block-JRVGROlG.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
export { default as BaseExample } from './Example2-8iT2brQo.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './BlockTitle-B0onSZmA.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './IconButton-D82v5nCM.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './Send-zUFiC8KE.js';
import './Square-CJzdI5ja.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './index-Cg-Pg6j3.js';
import './Clear-DH-TDCgr.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './clone-Yk88IHKV.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ..._props } = $$props;
    const gradio = new Gradio(_props);
    let label = gradio.shared.label || "Textbox";
    gradio.props.value = gradio.props.value ?? "";
    gradio.props.value;
    async function handle_input(value) {
      if (!gradio.shared || !gradio.props) return;
      gradio.props.validation_error = null;
      gradio.props.value = value;
      await tick();
      gradio.dispatch("input");
    }
    function handle_change(value) {
      if (!gradio.shared || !gradio.props) return;
      gradio.props.validation_error = null;
      gradio.props.value = value;
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        padding: gradio.shared.container,
        rtl: gradio.props.rtl,
        children: ($$renderer4) => {
          if (gradio.shared.loading_status) {
            $$renderer4.push("<!--[-->");
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                show_validation_error: false,
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          Textbox($$renderer4, {
            label,
            info: gradio.props.info,
            show_label: gradio.shared.show_label,
            lines: gradio.props.lines,
            type: gradio.props.type,
            rtl: gradio.props.rtl,
            text_align: gradio.props.text_align,
            max_lines: gradio.props.max_lines,
            placeholder: gradio.props.placeholder,
            submit_btn: gradio.props.submit_btn,
            stop_btn: gradio.props.stop_btn,
            buttons: gradio.props.buttons,
            autofocus: gradio.props.autofocus,
            container: gradio.shared.container,
            autoscroll: gradio.shared.autoscroll,
            max_length: gradio.props.max_length,
            html_attributes: gradio.props.html_attributes,
            validation_error: gradio.shared?.loading_status?.validation_error || gradio.shared?.validation_error,
            onchange: handle_change,
            oninput: handle_input,
            onsubmit: () => {
              gradio.shared.validation_error = null;
              gradio.dispatch("submit");
            },
            onblur: () => gradio.dispatch("blur"),
            onselect: (data) => gradio.dispatch("select", data),
            onfocus: () => gradio.dispatch("focus"),
            onstop: () => gradio.dispatch("stop"),
            oncopy: (data) => gradio.dispatch("copy", data),
            oncustombuttonclick: (id) => {
              gradio.dispatch("custom_button_click", { id });
            },
            disabled: !gradio.shared.interactive,
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

export { Textbox as BaseTextbox, Index as default };
//# sourceMappingURL=Index16-BmyX54vt.js.map
