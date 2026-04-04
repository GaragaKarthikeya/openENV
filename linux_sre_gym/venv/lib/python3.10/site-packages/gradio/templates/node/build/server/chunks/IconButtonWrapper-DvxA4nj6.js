import './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, a as attr, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

/* empty css                                         */
function CustomButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let button = $$props["button"];
    let on_click = $$props["on_click"];
    $$renderer2.push(`<button class="custom-button svelte-gnx6f7"${attr("title", button.value || "")}${attr("aria-label", button.value || "Custom action")}>`);
    if (button.value) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="custom-button-label svelte-gnx6f7">${escape_html(button.value)}</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></button>`);
    bind_props($$props, { button, on_click });
  });
}
function IconButtonWrapper($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      top_panel = true,
      display_top_corner = false,
      show_background = true,
      buttons = null,
      on_custom_button_click = null,
      children
    } = $$props;
    $$renderer2.push(`<div${attr_class(`icon-button-wrapper ${top_panel ? "top-panel" : ""} ${display_top_corner ? "display-top-corner" : "hide-top-corner"} ${!show_background ? "no-background" : ""}`, "svelte-1pnho82")}>`);
    if (children) {
      $$renderer2.push("<!--[-->");
      children($$renderer2);
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (buttons) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(buttons);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let btn = each_array[$$index];
        if (typeof btn !== "string") {
          $$renderer2.push("<!--[-->");
          CustomButton($$renderer2, {
            button: btn,
            on_click: (id) => {
              if (on_custom_button_click) {
                on_custom_button_click(id);
              }
            }
          });
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}

export { CustomButton as C, IconButtonWrapper as I };
//# sourceMappingURL=IconButtonWrapper-DvxA4nj6.js.map
