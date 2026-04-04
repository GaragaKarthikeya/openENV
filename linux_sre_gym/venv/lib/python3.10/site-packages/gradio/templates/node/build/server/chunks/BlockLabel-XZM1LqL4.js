import { f as fallback } from './async-D55cHugf.js';
import { a as attr, f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

/* empty css                                         */
function BlockLabel($$renderer, $$props) {
  let label = fallback($$props["label"], null);
  let Icon = $$props["Icon"];
  let show_label = fallback($$props["show_label"], true);
  let disable = fallback($$props["disable"], false);
  let float = fallback($$props["float"], true);
  let rtl = fallback($$props["rtl"], false);
  $$renderer.push(`<label for="" data-testid="block-label"${attr("dir", rtl ? "rtl" : "ltr")}${attr_class("svelte-19djge9", void 0, {
    "hide": !show_label,
    "sr-only": !show_label,
    "float": float,
    "hide-label": disable
  })}><span class="svelte-19djge9">`);
  Icon($$renderer, {});
  $$renderer.push(`<!----></span> ${escape_html(label)}</label>`);
  bind_props($$props, { label, Icon, show_label, disable, float, rtl });
}

export { BlockLabel as B };
//# sourceMappingURL=BlockLabel-XZM1LqL4.js.map
