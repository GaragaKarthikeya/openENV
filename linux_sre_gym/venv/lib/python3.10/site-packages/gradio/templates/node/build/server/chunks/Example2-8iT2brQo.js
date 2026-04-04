import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

/* empty css                                      */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    function truncate_text(text, max_length = 60) {
      if (!text) return "";
      const str = String(text);
      if (str.length <= max_length) return str;
      return str.slice(0, max_length) + "...";
    }
    $$renderer2.push(`<div${attr_class("svelte-xxobeb", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>${escape_html(truncate_text(value))}</div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example2-8iT2brQo.js.map
