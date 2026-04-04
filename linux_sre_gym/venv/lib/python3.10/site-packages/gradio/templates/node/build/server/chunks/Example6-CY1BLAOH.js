import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { I as Image } from './Image-BLyvTcTy.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

/* empty css                                       */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("container svelte-bs74gu", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected,
      "border": value
    })}>`);
    if (value) {
      $$renderer2.push("<!--[-->");
      Image($$renderer2, { src: value.url });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example6-CY1BLAOH.js.map
