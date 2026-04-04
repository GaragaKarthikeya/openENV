import './async-D55cHugf.js';
import { f as attr_class, a as attr, g as attr_style, i as stringify } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { c as css_units } from './utils.svelte-aYomt3KL.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { M as MarkdownCode } from './MarkdownCode-BPzQM9w-.js';

function Markdown($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      elem_classes = [],
      visible = true,
      value,
      min_height = void 0,
      rtl = false,
      sanitize_html = true,
      line_breaks = false,
      latex_delimiters = [],
      header_links = false,
      height = void 0,
      show_copy_button = false,
      loading_status = void 0,
      theme_mode,
      onchange = () => {
      },
      oncopy = (val) => {
      }
    } = $$props;
    let copied = false;
    let timer;
    async function handle_copy() {
      if ("clipboard" in navigator) {
        await navigator.clipboard.writeText(value);
        oncopy({ value });
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
    $$renderer2.push(`<div${attr_class(`prose ${stringify(elem_classes?.join(" ") || "")}`, "svelte-1xjkzpp", { "hide": !visible })} data-testid="markdown"${attr("dir", rtl ? "rtl" : "ltr")}${attr_style(
      height ? `max-height: ${css_units(height)}; overflow-y: auto;` : "",
      {
        "min-height": min_height && loading_status?.status !== "pending" ? css_units(min_height) : void 0
      }
    )}>`);
    if (show_copy_button) {
      $$renderer2.push("<!--[-->");
      IconButtonWrapper($$renderer2, {
        children: ($$renderer3) => {
          IconButton($$renderer3, {
            Icon: copied ? Check : Copy,
            onclick: handle_copy,
            label: copied ? "Copied conversation" : "Copy conversation"
          });
        }
      });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    MarkdownCode($$renderer2, {
      message: value,
      latex_delimiters,
      sanitize_html,
      line_breaks,
      chatbot: false,
      header_links,
      theme_mode
    });
    $$renderer2.push(`<!----></div>`);
  });
}

export { Markdown as M };
//# sourceMappingURL=Index.svelte_svelte_type_style_lang2-y82X8y0w.js.map
