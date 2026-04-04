import { f as fallback } from './async-D55cHugf.js';
import { b as store_get, o as store_set, f as attr_class, a as attr, g as attr_style, i as stringify, e as ensure_array_like, s as slot, u as unsubscribe_stores, d as bind_props } from './index-u8mz_F03.js';
import { a as setContext } from './context-CBkBucIx.js';
import { w as writable } from './index-Cg-Pg6j3.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

function OverflowIcon($$renderer) {
  $$renderer.push(`<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="2.5" cy="8" r="1.5" fill="currentColor"></circle><circle cx="8" cy="8" r="1.5" fill="currentColor"></circle><circle cx="13.5" cy="8" r="1.5" fill="currentColor"></circle></svg>`);
}
const TABS = {};
function Tabs($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let has_tabs, tab_scale;
    let visible = fallback($$props["visible"], true);
    let elem_id = fallback($$props["elem_id"], "");
    let elem_classes = fallback($$props["elem_classes"], () => [], true);
    let selected = $$props["selected"];
    let initial_tabs = $$props["initial_tabs"];
    let tabs = [...initial_tabs];
    let visible_tabs = [...initial_tabs];
    let overflow_tabs = [];
    let overflow_menu_open = false;
    let mounted_tab_orders = /* @__PURE__ */ new Set();
    function _sync_tabs(new_tabs) {
      for (let i = 0; i < new_tabs.length; i++) {
        if (new_tabs[i] && !mounted_tab_orders.has(i)) {
          tabs[i] = new_tabs[i];
        }
      }
    }
    const selected_tab = writable(selected || tabs[0]?.id || false);
    const selected_tab_index = writable(tabs.findIndex((t) => t?.id === selected) || 0);
    let overflow_has_selected_tab = false;
    setContext(TABS, {
      register_tab: (tab, order) => {
        mounted_tab_orders.add(order);
        tabs[order] = tab;
        if (store_get($$store_subs ??= {}, "$selected_tab", selected_tab) === false && tab.visible !== false && tab.interactive) {
          store_set(selected_tab, tab.id);
          store_set(selected_tab_index, order);
        }
        return order;
      },
      unregister_tab: (tab, order) => {
        mounted_tab_orders.delete(order);
        if (store_get($$store_subs ??= {}, "$selected_tab", selected_tab) === tab.id) {
          store_set(selected_tab, tabs[0]?.id || false);
        }
        tabs[order] = null;
      },
      selected_tab,
      selected_tab_index
    });
    function change_tab(id) {
      const tab_to_activate = tabs.find((t) => t?.id === id);
      if (id !== void 0 && tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible !== false && store_get($$store_subs ??= {}, "$selected_tab", selected_tab) !== tab_to_activate.id) {
        selected = id;
        store_set(selected_tab, id);
        store_set(selected_tab_index, tabs.findIndex((t) => t?.id === id));
        overflow_menu_open = false;
      }
    }
    async function handle_menu_overflow() {
      return;
    }
    function handle_overflow_has_selected_tab(selected_tab2) {
      if (selected_tab2 === false) return false;
      return overflow_tabs.some((t) => t?.id === selected_tab2);
    }
    _sync_tabs(initial_tabs);
    has_tabs = tabs.length > 0;
    selected !== null && change_tab(selected);
    handle_menu_overflow();
    overflow_has_selected_tab = handle_overflow_has_selected_tab(store_get($$store_subs ??= {}, "$selected_tab", selected_tab));
    tab_scale = tabs[store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index) >= 0 ? store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index) : 0]?.scale;
    $$renderer2.push(`<div${attr_class(`tabs ${stringify(elem_classes.join(" "))}`, "svelte-11gaq1", { "hide": visible === false, "hidden": visible === "hidden" })}${attr("id", elem_id)}${attr_style("", { "flex-grow": tab_scale })}>`);
    if (has_tabs) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="tab-wrapper svelte-11gaq1"><div class="tab-container visually-hidden svelte-11gaq1" aria-hidden="true"><!--[-->`);
      const each_array = ensure_array_like(tabs);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let t = each_array[i];
        if (t && t?.visible !== false && t?.visible !== "hidden") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<button class="svelte-11gaq1">${escape_html(t?.label)}</button>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div> <div class="tab-container svelte-11gaq1" role="tablist"><!--[-->`);
      const each_array_1 = ensure_array_like(visible_tabs);
      for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
        let t = each_array_1[i];
        if (t && t?.visible !== false) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<button role="tab"${attr("aria-selected", t.id === store_get($$store_subs ??= {}, "$selected_tab", selected_tab))}${attr("aria-controls", t.elem_id)}${attr("disabled", !t.interactive, true)}${attr("aria-disabled", !t.interactive)}${attr("id", t.elem_id ? t.elem_id + "-button" : null)}${attr("data-tab-id", t.id)}${attr_class("svelte-11gaq1", void 0, {
            "selected": t.id === store_get($$store_subs ??= {}, "$selected_tab", selected_tab)
          })}>${escape_html(t?.label !== void 0 ? t?.label : "Tab " + (i + 1))}</button>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div> <span${attr_class("overflow-menu svelte-11gaq1", void 0, {
        "hide": true
      })}><button${attr_class("svelte-11gaq1", void 0, { "overflow-item-selected": overflow_has_selected_tab })}>`);
      OverflowIcon($$renderer2);
      $$renderer2.push(`<!----></button> <div${attr_class("overflow-dropdown svelte-11gaq1", void 0, { "hide": !overflow_menu_open })}><!--[-->`);
      const each_array_2 = ensure_array_like(overflow_tabs);
      for (let i = 0, $$length = each_array_2.length; i < $$length; i++) {
        let t = each_array_2[i];
        if (t?.visible !== false) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<button${attr_class("svelte-11gaq1", void 0, {
            "selected": t?.id === store_get($$store_subs ??= {}, "$selected_tab", selected_tab)
          })}>${escape_html(t?.label)}</button>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div></span></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, { visible, elem_id, elem_classes, selected, initial_tabs });
  });
}

export { TABS as T, Tabs as a };
//# sourceMappingURL=Walkthrough.svelte_svelte_type_style_lang-D0By3Fla.js.map
