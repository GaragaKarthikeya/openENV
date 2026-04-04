import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { M as MarkdownCode } from './MarkdownCode-BPzQM9w-.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './prism-python-uEphcfi3.js';
import './2-BAeILJ1g.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './index50-nVz3eHRG.js';
import 'path';
import 'url';
import 'fs';
import './html-CfyvkLET.js';

/* empty css                                       */
function Example($$renderer, $$props) {
  let value = $$props["value"];
  let type = $$props["type"];
  let selected = fallback($$props["selected"], false);
  let sanitize_html = $$props["sanitize_html"];
  let line_breaks = $$props["line_breaks"];
  let latex_delimiters = $$props["latex_delimiters"];
  function truncate_text(text, max_length = 60) {
    if (!text) return "";
    const str = String(text);
    if (str.length <= max_length) return str;
    return str.slice(0, max_length) + "...";
  }
  $$renderer.push(`<div${attr_class("prose svelte-11ua876", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}>`);
  MarkdownCode($$renderer, {
    message: truncate_text(value),
    latex_delimiters,
    sanitize_html,
    line_breaks,
    chatbot: false
  });
  $$renderer.push(`<!----></div>`);
  bind_props($$props, {
    value,
    type,
    selected,
    sanitize_html,
    line_breaks,
    latex_delimiters
  });
}

export { Example as default };
//# sourceMappingURL=Example7-CLPLiEhz.js.map
