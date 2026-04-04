import './async-D55cHugf.js';
import { f as attr_class, a as attr, g as attr_style } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

/* empty css                                         */
function IconButton($$renderer, $$props) {
  let {
    Icon,
    label = "",
    show_label = false,
    pending = false,
    size = "small",
    padded = true,
    highlight = false,
    disabled = false,
    hasPopup = false,
    color = "var(--block-label-text-color)",
    transparent = false,
    background = "var(--block-background-fill)",
    border = "transparent",
    onclick,
    children
  } = $$props;
  let _color = highlight ? "var(--color-accent)" : color;
  $$renderer.push(`<button${attr_class("icon-button svelte-3jwzs9", void 0, {
    "pending": pending,
    "padded": padded,
    "highlight": highlight,
    "transparent": transparent
  })}${attr("disabled", disabled, true)}${attr("aria-label", label)}${attr("aria-haspopup", hasPopup)}${attr("title", label)}${attr_style("", {
    "--border-color": border,
    color: !disabled && _color ? _color : "var(--block-label-text-color)",
    "--bg-color": !disabled ? background : "auto"
  })}>`);
  if (show_label) {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<span class="svelte-3jwzs9">${escape_html(label)}</span>`);
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]--> <div${attr_class("svelte-3jwzs9", void 0, {
    "x-small": size === "x-small",
    "small": size === "small",
    "large": size === "large",
    "medium": size === "medium"
  })}><!---->`);
  Icon($$renderer, {});
  $$renderer.push(`<!----> `);
  if (children) {
    $$renderer.push("<!--[-->");
    children($$renderer);
    $$renderer.push(`<!---->`);
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]--></div></button>`);
}

export { IconButton as I };
//# sourceMappingURL=IconButton-D82v5nCM.js.map
