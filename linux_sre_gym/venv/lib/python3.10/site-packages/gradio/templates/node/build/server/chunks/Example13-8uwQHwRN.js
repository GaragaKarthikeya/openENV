import { f as fallback } from './async-D55cHugf.js';
import { g as attr_style, f as attr_class, i as stringify, d as bind_props } from './index-u8mz_F03.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  let value = $$props["value"];
  let type = $$props["type"];
  let selected = fallback($$props["selected"], false);
  $$renderer.push(`<div${attr_style(`background-color: ${stringify(value ? value : "black")}`)}${attr_class("svelte-1k1s8qu", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}></div>`);
  bind_props($$props, { value, type, selected });
}

export { Example as default };
//# sourceMappingURL=Example13-8uwQHwRN.js.map
