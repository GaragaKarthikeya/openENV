import { f as fallback } from './async-D55cHugf.js';
import { b as store_get, a as attr, f as attr_class, g as attr_style, i as stringify, s as slot, u as unsubscribe_stores, d as bind_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { t as tick, c as createEventDispatcher } from './index-server-CQz6EZl_.js';
import { T as TABS } from './Walkthrough.svelte_svelte_type_style_lang-D0By3Fla.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-BbzONYlY.js';
import { g as getContext } from './context-CBkBucIx.js';
import './escaping-CBnpiEl5.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './index3-BSof0MuO.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function TabItem($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let props_json;
    let elem_id = fallback($$props["elem_id"], "");
    let elem_classes = fallback($$props["elem_classes"], () => [], true);
    let label = $$props["label"];
    let id = fallback($$props["id"], () => ({}), true);
    let visible = $$props["visible"];
    let interactive = $$props["interactive"];
    let order = $$props["order"];
    let scale = $$props["scale"];
    let component_id = $$props["component_id"];
    const dispatch = createEventDispatcher();
    const {
      register_tab,
      unregister_tab,
      selected_tab,
      selected_tab_index
    } = getContext(TABS);
    let tab_index;
    function _register_tab(obj, order2) {
      obj = JSON.parse(obj);
      return register_tab(obj, order2);
    }
    props_json = JSON.stringify({
      label,
      id,
      elem_id,
      visible,
      interactive,
      scale,
      component_id
    });
    tab_index = _register_tab(props_json, order);
    store_get($$store_subs ??= {}, "$selected_tab_index", selected_tab_index) === tab_index && tick().then(() => dispatch("select", { value: label, index: tab_index }));
    $$renderer2.push(`<div${attr("id", elem_id)}${attr_class(`tabitem ${stringify(elem_classes.join(" "))}`, "svelte-dmtrd3", { "grow-children": scale >= 1 })} role="tabpanel"${attr_style("", {
      display: store_get($$store_subs ??= {}, "$selected_tab", selected_tab) === id && visible !== false ? "flex" : "none",
      "flex-grow": scale
    })}>`);
    BaseColumn($$renderer2, {
      scale: scale >= 1 ? scale : null,
      children: ($$renderer3) => {
        $$renderer3.push(`<!--[-->`);
        slot($$renderer3, $$props, "default", {});
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, {
      elem_id,
      elem_classes,
      label,
      id,
      visible,
      interactive,
      order,
      scale,
      component_id
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    TabItem($$renderer2, {
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      label: gradio.shared.label,
      visible: gradio.shared.visible,
      interactive: gradio.shared.interactive,
      id: gradio.props.id,
      order: gradio.props.order,
      scale: gradio.props.scale,
      component_id: gradio.props.component_id,
      children: ($$renderer3) => {
        $$renderer3.push(`<!--[-->`);
        slot($$renderer3, $$props, "default", {});
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
  });
}

export { TabItem as BaseTabItem, Index as default };
//# sourceMappingURL=Index37-Bbib4BAM.js.map
