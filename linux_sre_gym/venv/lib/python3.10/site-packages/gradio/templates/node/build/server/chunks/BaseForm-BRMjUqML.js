import './async-D55cHugf.js';
import { f as attr_class, a as attr, g as attr_style, s as slot } from './index-u8mz_F03.js';

function BaseForm($$renderer, $$props) {
  let { visible, scale, min_width, label = void 0 } = $$props;
  $$renderer.push(`<div${attr_class("form svelte-d5xbca", void 0, {
    "hidden": visible === false,
    "hidden-css": visible === "hidden"
  })}${attr("role", label ? "group" : void 0)}${attr("aria-label", label)}${attr_style("", {
    "flex-grow": scale,
    "min-width": `calc(min(${min_width}px, 100%))`
  })}><!--[-->`);
  slot($$renderer, $$props, "default", {});
  $$renderer.push(`<!--]--></div>`);
}

export { BaseForm as B };
//# sourceMappingURL=BaseForm-BRMjUqML.js.map
