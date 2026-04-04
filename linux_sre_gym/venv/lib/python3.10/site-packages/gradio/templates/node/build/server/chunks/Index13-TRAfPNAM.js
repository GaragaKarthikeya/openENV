import './async-D55cHugf.js';
import { a as attr, f as attr_class, j as clsx } from './index-u8mz_F03.js';
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
    (() => {
      return {
        visible: gradio.shared.visible,
        main_page_name: gradio.props.main_page_name ?? "Home",
        value: gradio.props.value
      };
    })();
    $$renderer2.push(`<div style="display: none;"${attr("id", gradio.shared.elem_id)}${attr_class(clsx(gradio.shared.elem_classes.join(" ")))}></div>`);
  });
}

export { Index as default };
//# sourceMappingURL=Index13-TRAfPNAM.js.map
