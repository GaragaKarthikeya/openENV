import './async-D55cHugf.js';
import { g as attr_style, f as attr_class, i as stringify, a as attr, e as ensure_array_like } from './index-u8mz_F03.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import './2-BAeILJ1g.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

function JSON$1($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--mdi" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path fill="currentColor" d="M5 3h2v2H5v5a2 2 0 0 1-2 2a2 2 0 0 1 2 2v5h2v2H5c-1.07-.27-2-.9-2-2v-4a2 2 0 0 0-2-2H0v-2h1a2 2 0 0 0 2-2V5a2 2 0 0 1 2-2m14 0a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h1v2h-1a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2h-2v-2h2v-5a2 2 0 0 1 2-2a2 2 0 0 1-2-2V5h-2V3h2m-7 12a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1m-4 0a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1m8 0a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1Z"></path></svg>`);
}
function JSONNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      depth = 0,
      is_root = false,
      is_last_item = true,
      key = null,
      open = false,
      theme_mode = "system",
      show_indices = false,
      interactive = true
    } = $$props;
    let collapsed = open ? false : depth >= 3;
    let child_nodes = [];
    function is_collapsible(val) {
      return val !== null && (typeof val === "object" || Array.isArray(val));
    }
    function get_collapsed_preview(val) {
      if (Array.isArray(val)) return `Array(${val.length})`;
      if (typeof val === "object" && val !== null) return `Object(${Object.keys(val).length})`;
      return String(val);
    }
    $$renderer2.push(`<div${attr_class("json-node svelte-1olemhd", void 0, { "root": is_root, "dark-mode": theme_mode === "dark" })}${attr_style(`--depth: ${stringify(depth)};`)}><div${attr_class("line svelte-1olemhd", void 0, { "collapsed": collapsed })}><span class="line-number svelte-1olemhd"></span> <span class="content svelte-1olemhd">`);
    if (is_collapsible(value)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button${attr("data-pseudo-content", interactive ? collapsed ? "▶" : "▼" : "")}${attr("aria-label", collapsed ? "Expand" : "Collapse")} class="toggle svelte-1olemhd"${attr("disabled", !interactive, true)}></button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (key !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="key svelte-1olemhd">"${escape_html(key)}"</span><span class="punctuation colon svelte-1olemhd">:</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (is_collapsible(value)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span${attr_class("punctuation bracket svelte-1olemhd", void 0, { "square-bracket": Array.isArray(value) })}>${escape_html(Array.isArray(value) ? "[" : "{")}</span> `);
      if (collapsed) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button class="preview svelte-1olemhd">${escape_html(get_collapsed_preview(value))}</button> <span${attr_class("punctuation bracket svelte-1olemhd", void 0, { "square-bracket": Array.isArray(value) })}>${escape_html(Array.isArray(value) ? "]" : "}")}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (typeof value === "string") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="value string svelte-1olemhd">"${escape_html(value)}"</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (typeof value === "number") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="value number svelte-1olemhd">${escape_html(value)}</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (typeof value === "boolean") {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<span class="value bool svelte-1olemhd">${escape_html(value.toString())}</span>`);
          } else {
            $$renderer2.push("<!--[!-->");
            if (value === null) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<span class="value null svelte-1olemhd">null</span>`);
            } else {
              $$renderer2.push("<!--[!-->");
              $$renderer2.push(`<span>${escape_html(value)}</span>`);
            }
            $$renderer2.push(`<!--]-->`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--> `);
    if (!is_last_item && (!is_collapsible(value) || collapsed)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="punctuation svelte-1olemhd">,</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></span></div> `);
    if (is_collapsible(value)) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div${attr_class("children svelte-1olemhd", void 0, { "hidden": collapsed })}><!--[-->`);
      const each_array = ensure_array_like(child_nodes);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let [subKey, subVal] = each_array[i];
        JSONNode($$renderer2, {
          value: subVal,
          depth: depth + 1,
          is_last_item: i === child_nodes.length - 1,
          key: Array.isArray(value) && !show_indices ? null : subKey,
          open,
          theme_mode,
          show_indices,
          interactive
        });
      }
      $$renderer2.push(`<!--]--> <div class="line svelte-1olemhd"><span class="line-number svelte-1olemhd"></span> <span class="content svelte-1olemhd"><span${attr_class("punctuation bracket svelte-1olemhd", void 0, { "square-bracket": Array.isArray(value) })}>${escape_html(Array.isArray(value) ? "]" : "}")}</span> `);
      if (!is_last_item) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="punctuation svelte-1olemhd">,</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></span></div></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function JSON_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = {},
      open = false,
      theme_mode = "system",
      show_indices = false,
      label_height,
      interactive = true,
      show_copy_button = true,
      buttons = null,
      on_custom_button_click = null
    } = $$props;
    let json_max_height = `calc(100% - ${label_height}px)`;
    let copied = false;
    let timer = void 0;
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
    async function handle_copy() {
      if ("clipboard" in navigator) {
        await navigator.clipboard.writeText(JSON.stringify(value, null, 2));
        copy_feedback();
      }
    }
    function is_empty(obj) {
      return obj && Object.keys(obj).length === 0 && Object.getPrototypeOf(obj) === Object.prototype && JSON.stringify(obj) === JSON.stringify({});
    }
    if (value && value !== '""' && !is_empty(value)) {
      $$renderer2.push("<!--[-->");
      if (show_copy_button || buttons && buttons.some((btn) => typeof btn !== "string")) {
        $$renderer2.push("<!--[-->");
        IconButtonWrapper($$renderer2, {
          buttons,
          on_custom_button_click,
          children: ($$renderer3) => {
            if (show_copy_button) {
              $$renderer3.push("<!--[-->");
              IconButton($$renderer3, {
                show_label: false,
                label: copied ? "Copied" : "Copy",
                Icon: copied ? Check : Copy,
                onclick: () => handle_copy()
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
      $$renderer2.push(`<!--]--> <div class="json-holder svelte-1lc38wd"${attr_style("", { "max-height": json_max_height })}>`);
      JSONNode($$renderer2, {
        value,
        depth: 0,
        is_root: true,
        open,
        theme_mode,
        show_indices,
        interactive
      });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="empty-wrapper svelte-1lc38wd">`);
      Empty($$renderer2, {
        children: ($$renderer3) => {
          JSON$1($$renderer3);
        },
        $$slots: { default: true }
      });
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
  });
}

export { JSON_1 as J, JSON$1 as a };
//# sourceMappingURL=JSON-BbQu1pP2.js.map
