import './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, a as attr } from './index-u8mz_F03.js';
import { I as Image } from './Image-BLyvTcTy.js';
import './2-BAeILJ1g.js';
import { V as Video } from './Video2-De5bYQY6.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value = { text: "", files: [] }, type, selected = false } = $$props;
    $$renderer2.push(`<div${attr_class("container svelte-xz0m7l", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected,
      "border": value
    })}><p>${escape_html(value.text ? value.text : "")}</p> <!--[-->`);
    const each_array = ensure_array_like(value.files);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let file = each_array[$$index];
      if (file.mime_type && file.mime_type.includes("image")) {
        $$renderer2.push("<!--[-->");
        Image($$renderer2, { src: file.url });
      } else {
        $$renderer2.push("<!--[!-->");
        if (file.mime_type && file.mime_type.includes("video")) {
          $$renderer2.push("<!--[-->");
          Video($$renderer2, { src: file.url, alt: "", loop: true, is_stream: false });
        } else {
          $$renderer2.push("<!--[!-->");
          if (file.mime_type && file.mime_type.includes("audio")) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<audio${attr("src", file.url)} controls></audio>`);
          } else {
            $$renderer2.push("<!--[!-->");
            $$renderer2.push(`${escape_html(file.orig_name)}`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}

export { Example as default };
//# sourceMappingURL=Example23-Cdj-Vjy_.js.map
