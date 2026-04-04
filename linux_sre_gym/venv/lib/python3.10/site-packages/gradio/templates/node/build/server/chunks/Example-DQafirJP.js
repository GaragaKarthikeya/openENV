import './async-D55cHugf.js';
import { f as attr_class } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

/* empty css                                       */
function Example($$renderer, $$props) {
  let { value, type, selected = false } = $$props;
  function truncate_text(text, max_length = 60) {
    if (!text) return "";
    const str = String(text);
    if (str.length <= max_length) return str;
    return str.slice(0, max_length) + "...";
  }
  $$renderer.push(`<pre${attr_class("svelte-1bbj91m", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}>${escape_html(truncate_text(value))}</pre>`);
}

export { Example as default };
//# sourceMappingURL=Example-DQafirJP.js.map
