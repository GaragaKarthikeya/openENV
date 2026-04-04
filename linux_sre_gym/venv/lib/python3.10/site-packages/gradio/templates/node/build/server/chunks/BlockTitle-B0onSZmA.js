import { f as fallback } from './async-D55cHugf.js';
import { a as attr, f as attr_class, s as slot, d as bind_props } from './index-u8mz_F03.js';
import { I as Info } from './Info-BeS-ygt-.js';

/* empty css                                         */
function BlockTitle($$renderer, $$props) {
  let show_label = fallback($$props["show_label"], true);
  let info = fallback($$props["info"], void 0);
  let rtl = fallback($$props["rtl"], false);
  $$renderer.push(`<span data-testid="block-info"${attr("dir", rtl ? "rtl" : "ltr")}${attr_class("svelte-jdcl7l", void 0, {
    "hide": !show_label,
    "has-info": info != null,
    "sr-only": !show_label
  })}><!--[-->`);
  slot($$renderer, $$props, "default", {});
  $$renderer.push(`<!--]--></span> `);
  if (info) {
    $$renderer.push("<!--[-->");
    Info($$renderer, { info });
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]-->`);
  bind_props($$props, { show_label, info, rtl });
}

export { BlockTitle as B };
//# sourceMappingURL=BlockTitle-B0onSZmA.js.map
