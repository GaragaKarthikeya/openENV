import './async-D55cHugf.js';
import { a as attr, f as attr_class, i as stringify, c as spread_props, s as slot } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { S as Static } from './index3-BSof0MuO.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    $$renderer2.push(`<div${attr("id", gradio.shared.elem_id)}${attr_class(`draggable ${stringify((gradio.shared.elem_classes || []).join(" "))}`, "svelte-1kr8imm", {
      "hide": !gradio.shared.visible,
      "horizontal": gradio.props.orientation === "row",
      "vertical": gradio.props.orientation === "column"
    })} role="region" aria-label="Draggable items container">`);
    if (gradio.shared.loading_status && gradio.props.show_progress) {
      $$renderer2.push("<!--[-->");
      Static($$renderer2, spread_props([
        { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
        gradio.shared.loading_status,
        {
          status: gradio.shared.loading_status ? gradio.shared.loading_status.status == "pending" ? "generating" : gradio.shared.loading_status.status : null
        }
      ]));
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div>`);
  });
}

export { Index as default };
//# sourceMappingURL=Index21-C0LViUOE.js.map
