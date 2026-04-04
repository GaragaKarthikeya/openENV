import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  let value = $$props["value"];
  let type = $$props["type"];
  let selected = fallback($$props["selected"], false);
  $$renderer.push(`<div${attr_class("svelte-pj4oi8", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}><pre>${escape_html(JSON.stringify(value, null, 2))}</pre></div>`);
  bind_props($$props, { value, type, selected });
}

export { Example as default };
//# sourceMappingURL=Example25-ArigFhpQ.js.map
