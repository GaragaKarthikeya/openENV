import './async-D55cHugf.js';
import { a as attr, f as attr_class, g as attr_style, i as stringify } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { I as Image } from './Image-BLyvTcTy.js';

/* empty css                                         */
/* empty css                                           */
function Button($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      elem_id,
      elem_classes = [],
      visible,
      variant,
      size,
      value,
      link,
      link_target,
      icon,
      disabled,
      scale,
      min_width,
      onclick = () => {
      },
      children
    } = $$props;
    if (link && link.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<a${attr("href", link)}${attr("target", link_target)}${attr("rel", link_target === "_blank" ? "noopener noreferrer" : void 0)}${attr("aria-disabled", disabled)}${attr_class(`${stringify(size)} ${stringify(variant)} ${stringify(elem_classes.join(" "))}`, "svelte-xzq5jh", {
        "hidden": visible === false || visible === "hidden",
        "disabled": disabled
      })}${attr("id", elem_id)}${attr_style("", {
        "flex-grow": scale,
        "pointer-events": disabled ? "none" : null,
        width: scale === 0 ? "fit-content" : null,
        "min-width": typeof min_width === "number" ? `calc(min(${min_width}px, 100%))` : null
      })}>`);
      if (icon) {
        $$renderer2.push("<!--[-->");
        Image($$renderer2, {
          src: icon.url,
          restProps: { alt: `${value} icon`, class: "button-icon" }
        });
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (children) {
        $$renderer2.push("<!--[-->");
        children($$renderer2);
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></a>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<button${attr_class(`${stringify(size)} ${stringify(variant)} ${stringify(elem_classes.join(" "))}`, "svelte-xzq5jh", { "hidden": visible === false || visible === "hidden" })}${attr("id", elem_id)}${attr("disabled", disabled, true)}${attr_style("", {
        "flex-grow": scale,
        width: scale === 0 ? "fit-content" : null,
        "min-width": typeof min_width === "number" ? `calc(min(${min_width}px, 100%))` : null
      })}>`);
      if (icon) {
        $$renderer2.push("<!--[-->");
        Image($$renderer2, {
          restProps: { alt: `${value} icon` },
          class_names: [`button-icon ${value ? "right-padded" : ""}`],
          src: icon.url
        });
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (children) {
        $$renderer2.push("<!--[-->");
        children($$renderer2);
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></button>`);
    }
    $$renderer2.push(`<!--]-->`);
  });
}

export { Button as B };
//# sourceMappingURL=Button-C8geAOuh.js.map
