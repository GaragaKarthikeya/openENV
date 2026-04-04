import './async-D55cHugf.js';
import { f as attr_class, a as attr, g as attr_style, d as bind_props } from './index-u8mz_F03.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { S as Send } from './Send-zUFiC8KE.js';
import { S as Square } from './Square-CJzdI5ja.js';
import './2-BAeILJ1g.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

/* empty css                                      */
function Textbox($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = "",
      value_is_output = false,
      lines = 1,
      placeholder = "",
      label,
      info = void 0,
      disabled = false,
      show_label = true,
      container = true,
      max_lines = void 0,
      type = "text",
      buttons = null,
      oncustombuttonclick = null,
      submit_btn = null,
      stop_btn = null,
      rtl = false,
      autofocus = false,
      text_align = void 0,
      autoscroll = true,
      max_length = void 0,
      html_attributes = null,
      validation_error = void 0,
      onchange,
      onsubmit,
      onstop,
      onblur,
      onselect,
      oninput,
      onfocus,
      oncopy
    } = $$props;
    let copied = false;
    let timer;
    let _max_lines = 1;
    const show_textbox_border = !submit_btn;
    async function handle_copy() {
      if ("clipboard" in navigator) {
        try {
          await navigator.clipboard.writeText(value);
          oncopy?.({ value });
        } catch (e) {
          console.error("COPYING CLIPBOARD FAILED", e);
        }
        copy_feedback();
      }
    }
    function copy_feedback() {
      copied = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(
        () => {
          copied = false;
        },
        1e3
      );
    }
    $$renderer2.push(`<label${attr_class("svelte-1hguek3", void 0, {
      "container": container,
      "show_textbox_border": show_textbox_border
    })}>`);
    if (show_label && buttons && buttons.length > 0) {
      $$renderer2.push("<!--[-->");
      IconButtonWrapper($$renderer2, {
        buttons,
        on_custom_button_click: oncustombuttonclick,
        children: ($$renderer3) => {
          if (buttons.some((btn) => typeof btn === "string" && btn === "copy")) {
            $$renderer3.push("<!--[-->");
            IconButton($$renderer3, {
              Icon: copied ? Check : Copy,
              onclick: handle_copy,
              label: copied ? "Copied" : "Copy"
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
      });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    BlockTitle($$renderer2, {
      show_label: validation_error ? true : show_label,
      info,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(label)} `);
        if (validation_error) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="validation-error svelte-1hguek3">${escape_html(validation_error)}</div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="input-container svelte-1hguek3">`);
    if (lines === 1 && _max_lines === 1) {
      $$renderer2.push("<!--[-->");
      if (type === "text") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<input data-testid="textbox" type="text"${attr_class("scroll-hide svelte-1hguek3", void 0, { "validation-error": validation_error })}${attr("dir", rtl ? "rtl" : "ltr")}${attr("value", value)}${attr("placeholder", placeholder)}${attr("disabled", disabled, true)}${attr("autofocus", autofocus, true)}${attr("maxlength", max_length)}${attr_style(text_align ? "text-align: " + text_align : "")}${attr("autocapitalize", html_attributes?.autocapitalize)}${attr("autocorrect", html_attributes?.autocorrect)}${attr("spellcheck", html_attributes?.spellcheck)}${attr("autocomplete", html_attributes?.autocomplete)}${attr("tabindex", html_attributes?.tabindex)}${attr("enterkeyhint", html_attributes?.enterkeyhint)}${attr("lang", html_attributes?.lang)}/>`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (type === "password") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<input data-testid="password" type="password"${attr_class("scroll-hide svelte-1hguek3", void 0, { "validation-error": validation_error })}${attr("value", value)}${attr("placeholder", placeholder)}${attr("disabled", disabled, true)}${attr("autofocus", autofocus, true)}${attr("maxlength", max_length)} autocomplete=""${attr("autocapitalize", html_attributes?.autocapitalize)}${attr("autocorrect", html_attributes?.autocorrect)}${attr("spellcheck", html_attributes?.spellcheck)}${attr("tabindex", html_attributes?.tabindex)}${attr("enterkeyhint", html_attributes?.enterkeyhint)}${attr("lang", html_attributes?.lang)}/>`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (type === "email") {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<input data-testid="textbox" type="email"${attr_class("scroll-hide svelte-1hguek3", void 0, { "validation-error": validation_error })}${attr("value", value)}${attr("placeholder", placeholder)}${attr("disabled", disabled, true)}${attr("autofocus", autofocus, true)}${attr("maxlength", max_length)} autocomplete="email"${attr("autocapitalize", html_attributes?.autocapitalize)}${attr("autocorrect", html_attributes?.autocorrect)}${attr("spellcheck", html_attributes?.spellcheck)}${attr("tabindex", html_attributes?.tabindex)}${attr("enterkeyhint", html_attributes?.enterkeyhint)}${attr("lang", html_attributes?.lang)}/>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<textarea data-testid="textbox"${attr("dir", rtl ? "rtl" : "ltr")}${attr("placeholder", placeholder)}${attr("rows", lines)}${attr("disabled", disabled, true)}${attr("autofocus", autofocus, true)}${attr("maxlength", max_length)}${attr_style(text_align ? "text-align: " + text_align : "")}${attr("autocapitalize", html_attributes?.autocapitalize)}${attr("autocorrect", html_attributes?.autocorrect)}${attr("spellcheck", html_attributes?.spellcheck)}${attr("autocomplete", html_attributes?.autocomplete)}${attr("tabindex", html_attributes?.tabindex)}${attr("enterkeyhint", html_attributes?.enterkeyhint)}${attr("lang", html_attributes?.lang)}${attr_class("svelte-1hguek3", void 0, {
        "no-label": !show_label && (submit_btn || stop_btn),
        "validation-error": validation_error
      })}>`);
      const $$body = escape_html(value);
      if ($$body) {
        $$renderer2.push(`${$$body}`);
      }
      $$renderer2.push(`</textarea>`);
    }
    $$renderer2.push(`<!--]--> `);
    if (submit_btn) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button${attr_class("submit-button svelte-1hguek3", void 0, { "padded-button": submit_btn !== true })} data-testid="submit-button">`);
      if (submit_btn === true) {
        $$renderer2.push("<!--[-->");
        Send($$renderer2);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`${escape_html(submit_btn)}`);
      }
      $$renderer2.push(`<!--]--></button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (stop_btn) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button${attr_class("stop-button svelte-1hguek3", void 0, { "padded-button": stop_btn !== true })} data-testid="stop-button">`);
      if (stop_btn === true) {
        $$renderer2.push("<!--[-->");
        Square($$renderer2, { fill: "none", stroke_width: 2.5 });
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`${escape_html(stop_btn)}`);
      }
      $$renderer2.push(`<!--]--></button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div></label>`);
    bind_props($$props, { value, value_is_output });
  });
}

export { Textbox as T };
//# sourceMappingURL=Textbox-T7HntrB1.js.map
