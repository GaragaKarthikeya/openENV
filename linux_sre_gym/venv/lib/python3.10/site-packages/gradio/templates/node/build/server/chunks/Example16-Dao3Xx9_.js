import './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value, type, selected = false } = $$props;
    $$renderer2.push(`<ul${attr_class("svelte-14aa7hi", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>`);
    if (value) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(Array.isArray(value) ? value.slice(0, 3) : [value]);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let path = each_array[$$index];
        $$renderer2.push(`<li><code>./${escape_html(path)}</code></li>`);
      }
      $$renderer2.push(`<!--]--> `);
      if (Array.isArray(value) && value.length > 3) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<li class="extra svelte-14aa7hi">...</li>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></ul>`);
  });
}

export { Example as default };
//# sourceMappingURL=Example16-Dao3Xx9_.js.map
