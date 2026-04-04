import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, g as attr_style, s as slot, d as bind_props } from './index-u8mz_F03.js';
import { B as Block } from './Block-JRVGROlG.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-BbzONYlY.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Accordion($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { open = true, label = "", onexpand, oncollapse } = $$props;
    $$renderer2.push(`<button${attr_class("label-wrap svelte-e5lyqv", void 0, { "open": open })}><span class="svelte-e5lyqv">${escape_html(label)}</span> <span class="icon svelte-e5lyqv"${attr_style("", { transform: open ? "rotate(0)" : "rotate(90deg)" })}>▼</span></button> <div data-testid="accordion-content"${attr_style("", { display: open ? "block" : "none" })}><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { open });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    class AccordionGradio extends Gradio {
      set_data(data) {
        if ("open" in data && data.open !== this.props.open) {
          if (data.open) {
            this.dispatch("expand");
            this.dispatch("gradio_expand");
          } else {
            this.dispatch("collapse");
          }
        }
        super.set_data(data);
        this.shared.loading_status.status = "complete";
      }
    }
    const gradio = new AccordionGradio(props);
    let label = gradio.shared.label || "";
    let visibility = gradio.shared.visible === true ? true : "hidden";
    Block($$renderer2, {
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      visible: visibility,
      children: ($$renderer3) => {
        if (gradio.shared.loading_status) {
          $$renderer3.push("<!--[-->");
          Static($$renderer3, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status
          ]));
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Accordion($$renderer3, {
          label,
          open: gradio.props.open,
          onexpand: () => {
            gradio.dispatch("expand");
            gradio.dispatch("gradio_expand");
          },
          oncollapse: () => gradio.dispatch("collapse"),
          children: ($$renderer4) => {
            BaseColumn($$renderer4, {
              children: ($$renderer5) => {
                $$renderer5.push(`<!--[-->`);
                slot($$renderer5, $$props, "default", {});
                $$renderer5.push(`<!--]-->`);
              },
              $$slots: { default: true }
            });
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index32-Bne_RZGi.js.map
