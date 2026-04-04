import './async-D55cHugf.js';
import { c as spread_props, s as slot } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-BbzONYlY.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './index3-BSof0MuO.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    BaseColumn($$renderer2, spread_props([
      gradio.shared,
      {
        children: ($$renderer3) => {
          $$renderer3.push(`<!--[-->`);
          slot($$renderer3, $$props, "default", {});
          $$renderer3.push(`<!--]-->`);
        },
        $$slots: { default: true }
      }
    ]));
  });
}

export { BaseColumn, Index as default };
//# sourceMappingURL=Index9-Bh3w6OXH.js.map
