import './async-D55cHugf.js';
import { f as attr_class } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

/* empty css                                       */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value, type, selected = false, choices } = $$props;
    let value_array = value ? Array.isArray(value) ? value : [value] : [];
    let names = value_array.map((val) => choices.find((pair) => pair[1] === val)?.[0]).filter((name) => name !== void 0);
    let names_string = names.join(", ");
    $$renderer2.push(`<div${attr_class("svelte-1by696e", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(names_string)}</div>`);
  });
}

export { Example as default };
//# sourceMappingURL=Example4-CMk5LZQq.js.map
