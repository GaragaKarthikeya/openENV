import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    let index = $$props["index"];
    let loaded = Array.isArray(value);
    let is_empty = loaded && (value.length === 0 || value[0].length === 0);
    if (loaded) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div${attr_class("svelte-wcwkqi", void 0, {
        "table": type === "table",
        "gallery": type === "gallery",
        "selected": selected
      })}>`);
      if (typeof value === "string") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`${escape_html(value)}`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (is_empty) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<table class="svelte-wcwkqi"><tbody><tr class="svelte-wcwkqi"><td class="svelte-wcwkqi">Empty</td></tr></tbody></table>`);
        } else {
          $$renderer2.push("<!--[!-->");
          $$renderer2.push(`<table class="svelte-wcwkqi"><tbody><!--[-->`);
          const each_array = ensure_array_like(value.slice(0, 3));
          for (let i = 0, $$length = each_array.length; i < $$length; i++) {
            let row = each_array[i];
            $$renderer2.push(`<tr class="svelte-wcwkqi"><!--[-->`);
            const each_array_1 = ensure_array_like(row.slice(0, 3));
            for (let j = 0, $$length2 = each_array_1.length; j < $$length2; j++) {
              let cell = each_array_1[j];
              $$renderer2.push(`<td class="svelte-wcwkqi">${escape_html(cell)}</td>`);
            }
            $$renderer2.push(`<!--]-->`);
            if (row.length > 3) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<td class="svelte-wcwkqi">…</td>`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]--></tr>`);
          }
          $$renderer2.push(`<!--]--></tbody></table> `);
          if (value.length > 3) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div${attr_class("overlay svelte-wcwkqi", void 0, {
              "odd": index % 2 != 0,
              "even": index % 2 == 0,
              "button": type === "gallery"
            })}></div>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { value, type, selected, index });
  });
}

export { Example as default };
//# sourceMappingURL=Example14-B9QZm81N.js.map
