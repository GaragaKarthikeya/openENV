import './async-D55cHugf.js';
import { d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  let title = $$props["title"];
  let x = $$props["x"];
  let y = $$props["y"];
  if (title) {
    $$renderer.push("<!--[-->");
    $$renderer.push(`${escape_html(title)}`);
  } else {
    $$renderer.push("<!--[!-->");
    $$renderer.push(`${escape_html(x)} x ${escape_html(y)}`);
  }
  $$renderer.push(`<!--]-->`);
  bind_props($$props, { title, x, y });
}

export { Example as default };
//# sourceMappingURL=Example8-CwW5sBgQ.js.map
