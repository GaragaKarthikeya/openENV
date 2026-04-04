import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("svelte-164n2qi", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(value !== null ? value.toLocaleString() : "")}</div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example11-CiOo3Xzc.js.map
