import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    let choices = $$props["choices"];
    let names = value.map((val) => choices.find((pair) => pair[1] === val)?.[0]).filter((name) => name !== void 0);
    let names_string = names.join(", ");
    $$renderer2.push(`<div${attr_class("svelte-25nhtv", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(names_string)}</div>`);
    bind_props($$props, { value, type, selected, choices });
  });
}

export { Example as default };
//# sourceMappingURL=Example12-C3QB1uOZ.js.map
