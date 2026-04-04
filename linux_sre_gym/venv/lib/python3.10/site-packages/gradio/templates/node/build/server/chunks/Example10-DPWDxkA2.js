import './async-D55cHugf.js';
import { f as attr_class } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  let { value, type, selected = false } = $$props;
  $$renderer.push(`<div${attr_class("svelte-9pg6fh", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}>${escape_html(value ? value : "")}</div>`);
}

export { Example as default };
//# sourceMappingURL=Example10-DPWDxkA2.js.map
