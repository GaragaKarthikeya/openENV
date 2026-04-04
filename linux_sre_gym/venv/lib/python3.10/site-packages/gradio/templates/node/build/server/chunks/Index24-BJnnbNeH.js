import './async-D55cHugf.js';
import { a as attr, f as attr_class, g as attr_style, i as stringify, s as slot } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    $$renderer2.push(`<div${attr("id", gradio.shared.elem_id)}${attr_class(`gr-group ${stringify(gradio.shared.elem_classes?.join(" ") || "")}`, "svelte-1p9262q", { "hide": gradio.shared.visible === "hidden" })}><div${attr("id", gradio.shared.elem_id)}${attr_class(`gr-group ${stringify(gradio.shared.elem_classes?.join(" ") || "")}`, "svelte-1p9262q", { "hide": gradio.shared.visible === "hidden" })}><div class="styler svelte-1p9262q"${attr_style("", {
      "--block-radius": "0px",
      "--block-border-width": "0px",
      "--layout-gap": "1px",
      "--form-gap-width": "1px",
      "--button-border-width": "0px",
      "--button-large-radius": "0px",
      "--button-small-radius": "0px"
    })}><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div></div></div>`);
  });
}

export { Index as default };
//# sourceMappingURL=Index24-BJnnbNeH.js.map
