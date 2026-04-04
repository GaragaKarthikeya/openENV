import './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, a as attr, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { D as DropdownArrow, a as DropdownOptions, b as Dropdown } from './Dropdown-BhyRR6a4.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example4-CMk5LZQq.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Remove($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100%" height="100%"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path></svg>`);
}
function handle_filter(choices, input_text) {
  return choices.reduce((filtered_indices, o, index) => {
    if (input_text ? o[0].toLowerCase().includes(input_text.toLowerCase()) : true) {
      filtered_indices.push(index);
    }
    return filtered_indices;
  }, []);
}
function Multiselect($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = props.gradio;
    let filter_input;
    let input_text = "";
    let label = gradio.shared.label || "Multiselect";
    let buttons = gradio.props.buttons;
    let choices_names = (() => {
      return gradio.props.choices.map((c) => c[0]);
    })();
    let choices_values = (() => {
      return gradio.props.choices.map((c) => c[1]);
    })();
    let disabled = !gradio.shared.interactive;
    let show_options = false;
    let [filtered_indices, active_index] = (() => {
      const filtered = handle_filter(gradio.props.choices, input_text);
      return [
        filtered,
        filtered.length > 0 && !gradio.props.allow_custom_value ? filtered[0] : null
      ];
    })();
    function set_selected_indices() {
      if (gradio.props.value === void 0) {
        return [];
      } else if (Array.isArray(gradio.props.value)) {
        return gradio.props.value.map((v) => {
          const index = choices_values.indexOf(v);
          if (index !== -1) {
            return index;
          }
          if (gradio.props.allow_custom_value) {
            return v;
          }
          return void 0;
        }).filter((val) => val !== void 0);
      }
      return [];
    }
    let selected_indices = set_selected_indices();
    function remove_selected_choice(option_index) {
      selected_indices = selected_indices.filter((v) => v !== option_index);
      gradio.props.value = selected_indices.map((index) => typeof index === "number" ? choices_values[index] : index);
      gradio.dispatch("input");
      gradio.dispatch("select", {
        index: typeof option_index === "number" ? option_index : -1,
        value: typeof option_index === "number" ? choices_values[option_index] : option_index,
        selected: false
      });
    }
    function add_selected_choice(option_index) {
      if (gradio.props.max_choices == null || selected_indices.length < gradio.props.max_choices) {
        selected_indices.push(option_index);
        gradio.dispatch("select", {
          index: typeof option_index === "number" ? option_index : -1,
          value: typeof option_index === "number" ? choices_values[option_index] : option_index,
          selected: true
        });
      }
      if (selected_indices.length === gradio.props.max_choices) {
        show_options = false;
        active_index = null;
        filter_input.blur();
      }
      gradio.props.value = selected_indices.map((index) => typeof index === "number" ? choices_values[index] : index);
    }
    function handle_option_selected(index) {
      const option_index = parseInt(index);
      add_or_remove_index(option_index);
    }
    function add_or_remove_index(option_index) {
      if (selected_indices.includes(option_index)) {
        remove_selected_choice(option_index);
      } else {
        add_selected_choice(option_index);
      }
      input_text = "";
      active_index = null;
      gradio.dispatch("input");
    }
    gradio.props.value;
    function oncustom_button_click(id) {
      gradio.dispatch("custom_button_click", { id });
    }
    $$renderer2.push(`<div${attr_class("svelte-1dv2vbb", void 0, { "container": gradio.shared.container })}>`);
    if (gradio.shared.show_label && buttons && buttons.length > 0) {
      $$renderer2.push("<!--[-->");
      IconButtonWrapper($$renderer2, { buttons, on_custom_button_click: oncustom_button_click });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    BlockTitle($$renderer2, {
      show_label: gradio.shared.show_label,
      info: gradio.props.info,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(label)}`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="wrap svelte-1dv2vbb"><div${attr_class("wrap-inner svelte-1dv2vbb", void 0, { "show_options": show_options })}><!--[-->`);
    const each_array = ensure_array_like(selected_indices);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let s = each_array[$$index];
      $$renderer2.push(`<div class="token svelte-1dv2vbb"><span class="svelte-1dv2vbb">`);
      if (typeof s === "number") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`${escape_html(choices_names[s])}`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`${escape_html(s)}`);
      }
      $$renderer2.push(`<!--]--></span> `);
      if (!disabled) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="token-remove svelte-1dv2vbb" role="button" tabindex="0"${attr("title", gradio.i18n("common.remove") + " " + s)}>`);
        Remove($$renderer2);
        $$renderer2.push(`<!----></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--> <div class="secondary-wrap svelte-1dv2vbb"><input${attr_class("border-none svelte-1dv2vbb", void 0, {
      "subdued": !choices_names.includes(input_text) && !gradio.props.allow_custom_value || selected_indices.length === gradio.props.max_choices
    })}${attr("disabled", disabled, true)} autocomplete="off"${attr("value", input_text)}${attr("readonly", !gradio.props.filterable, true)}/> `);
    if (!disabled) {
      $$renderer2.push("<!--[-->");
      if (selected_indices.length > 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div role="button" tabindex="0" class="token-remove remove-all svelte-1dv2vbb"${attr("title", gradio.i18n("common.clear"))}>`);
        Remove($$renderer2);
        $$renderer2.push(`<!----></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> <span class="icon-wrap svelte-1dv2vbb">`);
      DropdownArrow($$renderer2);
      $$renderer2.push(`<!----></span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div></div> `);
    DropdownOptions($$renderer2, {
      show_options,
      choices: gradio.props.choices,
      filtered_indices,
      disabled,
      selected_indices,
      active_index,
      remember_scroll: true,
      onchange: handle_option_selected
    });
    $$renderer2.push(`<!----></div></div>`);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        padding: gradio.shared.container,
        allow_overflow: false,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (gradio.props.multiselect) {
            $$renderer4.push("<!--[-->");
            Multiselect($$renderer4, { gradio });
          } else {
            $$renderer4.push("<!--[!-->");
            Dropdown($$renderer4, {
              label: gradio.shared.label,
              info: gradio.props.info,
              choices: gradio.props.choices,
              interactive: gradio.shared.interactive,
              show_label: gradio.shared.show_label,
              container: gradio.shared.container,
              allow_custom_value: gradio.props.allow_custom_value,
              filterable: gradio.props.filterable,
              buttons: gradio.props.buttons,
              oncustom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              on_change: () => gradio.dispatch("change"),
              on_input: () => gradio.dispatch("input"),
              on_select: (data) => gradio.dispatch("select", data),
              on_focus: () => gradio.dispatch("focus"),
              on_blur: () => gradio.dispatch("blur"),
              on_key_up: (data) => gradio.dispatch("key_up", data),
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { Dropdown as BaseDropdown, DropdownOptions as BaseDropdownOptions, Multiselect as BaseMultiselect, Index as default };
//# sourceMappingURL=Index41-P4de9cQY.js.map
