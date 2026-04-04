import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, a as attr, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("container svelte-xds4q5", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>`);
    if (value && value.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="images-wrapper svelte-xds4q5"><!--[-->`);
      const each_array = ensure_array_like(value.slice(0, 3));
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let item = each_array[$$index];
        if ("image" in item && item.image) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="image-container svelte-xds4q5"><img${attr("src", item.image.url)}${attr("alt", item.caption || "")} class="svelte-xds4q5"/> `);
          if (item.caption) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<span class="caption svelte-xds4q5">${escape_html(item.caption)}</span>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]--></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
          if ("video" in item && item.video) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div class="image-container svelte-xds4q5"><video${attr("src", item.video.url)}${attr("controls", false, true)} muted preload="metadata" class="svelte-xds4q5"></video> `);
            if (item.caption) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<span class="caption svelte-xds4q5">${escape_html(item.caption)}</span>`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]--></div>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--> `);
      if (value.length > 3) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="more-indicator svelte-xds4q5">…</div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example17-eUqd0Yc_.js.map
