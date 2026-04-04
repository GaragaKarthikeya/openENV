import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { I as Image } from './Image-BLyvTcTy.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

/* empty css                                            */
/* empty css                                     */
/* empty css                                       */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("container svelte-ous74z", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>`);
    Image($$renderer2, { src: value.composite?.url || value.background?.url });
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example19-Co3rLOUG.js.map
