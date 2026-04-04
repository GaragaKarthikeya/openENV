import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, a as attr, g as attr_style, i as stringify, s as slot, d as bind_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { S as Static } from './index3-BSof0MuO.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-BbzONYlY.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Sidebar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      open = true,
      width,
      position = "left",
      elem_classes = [],
      elem_id = "",
      onexpand = () => {
      },
      oncollapse = () => {
      }
    } = $$props;
    let _open = false;
    let width_css = typeof width === "number" ? `${width}px` : width;
    let prefersReducedMotion = false;
    let _elem_classes = elem_classes?.join(" ") || "";
    $$renderer2.push(`<div${attr_class(`sidebar ${stringify(_elem_classes)}`, "svelte-1uruprb", {
      "open": _open,
      "right": position === "right",
      "reduce-motion": prefersReducedMotion
    })}${attr("id", elem_id)}${attr_style(`width: ${stringify(width_css)}; ${stringify(position)}: calc(${stringify(width_css)} * -1)`)}><button class="toggle-button svelte-1uruprb" aria-label="Toggle Sidebar"><div class="chevron svelte-1uruprb"><span class="chevron-left svelte-1uruprb"></span></div></button> <div class="sidebar-content svelte-1uruprb"><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div></div>`);
    bind_props($$props, { open, position });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Static($$renderer3, spread_props([
        { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
        gradio.shared.loading_status
      ]));
      $$renderer3.push(`<!----> `);
      if (gradio.shared.visible) {
        $$renderer3.push("<!--[-->");
        Sidebar($$renderer3, {
          width: gradio.props.width,
          onexpand: () => gradio.dispatch("expand"),
          oncollapse: () => gradio.dispatch("collapse"),
          elem_classes: gradio.shared.elem_classes,
          elem_id: gradio.shared.elem_id,
          get open() {
            return gradio.props.open;
          },
          set open($$value) {
            gradio.props.open = $$value;
            $$settled = false;
          },
          get position() {
            return gradio.props.position;
          },
          set position($$value) {
            gradio.props.position = $$value;
            $$settled = false;
          },
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
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { Index as default };
//# sourceMappingURL=Index36-xwuDeO_Q.js.map
