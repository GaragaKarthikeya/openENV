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
    let name_string;
    if (value === null) {
      name_string = "";
    } else {
      let name = choices.find((pair) => pair[1] === value);
      name_string = name ? name[0] : "";
    }
    $$renderer2.push(`<div${attr_class("svelte-g2dls0", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(name_string)}</div>`);
    bind_props($$props, { value, type, selected, choices });
  });
}

export { Example as default };
//# sourceMappingURL=Example26-4aUdI3US.js.map
