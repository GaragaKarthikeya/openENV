import './async-D55cHugf.js';
import { f as attr_class } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

/* empty css                                       */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value, type, selected = false } = $$props;
    $$renderer2.push(`<div${attr_class("svelte-1p04unr", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(value ? Array.isArray(value) ? value.join(", ") : value : "")}</div>`);
  });
}

export { Example as default };
//# sourceMappingURL=Example5-DyYnewTx.js.map
