import { f as fallback } from './async-D55cHugf.js';
import { s as slot, b as store_get, o as store_set, f as attr_class, a as attr, g as attr_style, i as stringify, e as ensure_array_like, u as unsubscribe_stores, d as bind_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { a as Tabs, T as TABS } from './Walkthrough.svelte_svelte_type_style_lang-D0By3Fla.js';
import { a as setContext } from './context-CBkBucIx.js';
import { w as writable } from './index-Cg-Pg6j3.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './clone-Yk88IHKV.js';

function Walkthrough($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let has_tabs, tab_scale;
    let visible = fallback($$props["visible"], true);
    let elem_id = fallback($$props["elem_id"], "");
    let elem_classes = fallback($$props["elem_classes"], () => [], true);
    let selected = $$props["selected"];
    let initial_tabs = $$props["initial_tabs"];
    let tabs = [...initial_tabs];
    let show_labels_for_all = true;
    let label_height = 0;
    let compact = false;
    const selected_tab = writable(selected || tabs[0]?.id || false);
    const selected_tab_index = writable(tabs.findIndex((t) => t?.id === selected) || 0);
    async function check_overflow() {
      return;
    }
    setContext(TABS, {
      register_tab: (tab, order) => {
        tabs[order] = tab;
        if (store_get($$store_subs ??= {}, "$selected_tab", selected_tab) === false && tab.visible && tab.interactive) {
          store_set(selected_tab, tab.id);
          store_set(selected_tab_index, order);
        }
        return order;
      },
      unregister_tab: (tab, order) => {
        if (store_get($$store_subs ??= {}, "$selected_tab", selected_tab) === tab.id) {
          store_set(selected_tab, tabs[0]?.id || false);
        }
        tabs[order] = null;
      },
      selected_tab,
      selected_tab_index
    });
    function change_tab(id, index) {
      const tab_to_activate = tabs.find((t) => t?.id === id);
      if (id !== void 0 && tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible && store_get($$store_subs ??= {}, "$selected_tab", selected_tab) !== tab_to_activate.id) {
        selected = id;
        store_set(selected_tab, id);
        store_set(selected_tab_index, tabs.findIndex((t) => t?.id === id));
      }
    }
    has_tabs = tabs.length > 0;
    selected !== null && change_tab(selected, tabs.findIndex((t) => t?.id === selected));
    check_overflow();
    store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index), check_overflow();
    tab_scale = tabs[store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index) >= 0 ? store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index) : 0]?.scale;
    $$renderer2.push(`<div${attr_class(`stepper ${stringify(elem_classes.join(" "))}`, "svelte-1xpzbgb", { "hide": !visible, "compact": compact })}${attr("id", elem_id)}${attr_style("", { "flex-grow": tab_scale })}>`);
    if (has_tabs) {
      $$renderer2.push("<!--[-->");
      {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> <div class="stepper-wrapper svelte-1xpzbgb"${attr_style("", { "--label-height": label_height + "px" })}><div class="stepper-container svelte-1xpzbgb" role="tablist"><!--[-->`);
      const each_array = ensure_array_like(tabs);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let t = each_array[i];
        if (t?.visible) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="step-item svelte-1xpzbgb"><button role="tab"${attr_class("step-button svelte-1xpzbgb", void 0, {
            "active": t.id === store_get($$store_subs ??= {}, "$selected_tab", selected_tab),
            "completed": t.id < store_get($$store_subs ??= {}, "$selected_tab", selected_tab),
            "pending": t.id > store_get($$store_subs ??= {}, "$selected_tab", selected_tab)
          })}${attr("aria-selected", t.id === store_get($$store_subs ??= {}, "$selected_tab", selected_tab))}${attr("aria-controls", t.elem_id)}${attr("disabled", !t.interactive || i > store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index), true)}${attr("aria-disabled", !t.interactive || i > store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index))}${attr("id", t.elem_id ? t.elem_id + "-button" : null)}${attr("data-tab-id", t.id)}><span class="step-number svelte-1xpzbgb">`);
          if (t.id < store_get($$store_subs ??= {}, "$selected_tab", selected_tab)) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<svg width="12" height="10" viewBox="0 0 12 10" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 5L4.5 8.5L11 1.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`);
          } else {
            $$renderer2.push("<!--[!-->");
            $$renderer2.push(`${escape_html(i + 1)}`);
          }
          $$renderer2.push(`<!--]--></span> `);
          {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<span${attr_class("step-label svelte-1xpzbgb", void 0, {
              "visible": show_labels_for_all
            })}>${escape_html(t?.label !== void 0 ? t?.label : "Step " + (i + 1))}</span>`);
          }
          $$renderer2.push(`<!--]--></button></div> `);
          if (i < tabs.length - 1 && !compact) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div${attr_class("step-connector svelte-1xpzbgb", void 0, {
              "completed": i < store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index)
            })}></div>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div></div>`);
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
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (gradio.props.name === "walkthrough") {
        $$renderer3.push("<!--[-->");
        Walkthrough($$renderer3, {
          visible: gradio.shared.visible,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          initial_tabs: gradio.props.initial_tabs,
          get selected() {
            return gradio.props.selected;
          },
          set selected($$value) {
            gradio.props.selected = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            $$renderer4.push(`<!--[-->`);
            slot($$renderer4, $$props, "default", {});
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        Tabs($$renderer3, {
          visible: gradio.shared.visible,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          initial_tabs: gradio.props.initial_tabs,
          get selected() {
            return gradio.props.selected;
          },
          set selected($$value) {
            gradio.props.selected = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            $$renderer4.push(`<!--[-->`);
            slot($$renderer4, $$props, "default", {});
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { Tabs as BaseTabs, TABS, Index as default };
//# sourceMappingURL=Index31-X-tcmTYk.js.map
