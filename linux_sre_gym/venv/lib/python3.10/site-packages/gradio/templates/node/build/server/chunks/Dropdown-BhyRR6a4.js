import './async-D55cHugf.js';
import { f as attr_class, a as attr, d as bind_props, g as attr_style, e as ensure_array_like } from './index-u8mz_F03.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import './2-BAeILJ1g.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

function DropdownArrow($$renderer) {
  $$renderer.push(`<svg class="dropdown-arrow svelte-loyhyk" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 18 18"><path d="M5 8l4 4 4-4z"></path></svg>`);
}
function DropdownOptions($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      choices,
      filtered_indices,
      show_options = false,
      disabled = false,
      selected_indices = [],
      active_index = null,
      remember_scroll = false,
      offset_from_top = 0,
      from_top = false,
      onchange,
      onload
    } = $$props;
    let input_width = 0;
    let top = null;
    let bottom = null;
    let max_height = 0;
    $$renderer2.push(`<div class="reference"></div> `);
    if (show_options && !disabled) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<ul class="options svelte-1ou0lab" role="listbox"${attr_style("", {
        top,
        bottom,
        "max-height": `calc(${max_height}px - var(--window-padding))`,
        width: input_width + "px"
      })}><!--[-->`);
      const each_array = ensure_array_like(filtered_indices);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let index = each_array[$$index];
        $$renderer2.push(`<li${attr_class("item svelte-1ou0lab", void 0, {
          "selected": selected_indices.includes(index),
          "active": index === active_index,
          "bg-gray-100": index === active_index,
          "dark:bg-gray-600": index === active_index
        })}${attr("data-index", index)}${attr("aria-label", choices[index][0])} data-testid="dropdown-option" role="option"${attr("aria-selected", selected_indices.includes(index))}${attr_style("", { width: input_width + "px" })}><span${attr_class("inner-item svelte-1ou0lab", void 0, { "hide": !selected_indices.includes(index) })}>✓</span> ${escape_html(choices[index][0])}</li>`);
      }
      $$renderer2.push(`<!--]--></ul>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Dropdown($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const is_browser = typeof window !== "undefined";
    let {
      label = "Dropdown",
      info = void 0,
      value = void 0,
      choices = [],
      interactive = true,
      show_label = true,
      container = true,
      allow_custom_value = false,
      filterable = true,
      buttons = null,
      oncustom_button_click = null,
      on_change,
      on_input,
      on_select,
      on_focus,
      on_blur,
      on_key_up
    } = $$props;
    let filter_input;
    let show_options = (() => {
      return is_browser && filter_input === document.activeElement;
    })();
    let choices_names = choices.map((c) => c[0]);
    let choices_values = choices.map((c) => c[1]);
    let input_text = "";
    let selected_index = null;
    let disabled = !interactive;
    let filtered_indices = choices.map((_, i) => i);
    let active_index = null;
    let selected_indices = selected_index === null ? [] : [selected_index];
    function handle_option_selected(index) {
      selected_index = parseInt(index);
      if (isNaN(selected_index)) {
        selected_index = null;
        return;
      }
      let [_input_text, _value] = choices[selected_index];
      input_text = _input_text;
      value = _value;
      on_select?.({
        index: selected_index,
        value: choices_values[selected_index],
        selected: true
      });
      show_options = false;
      active_index = null;
      on_input?.();
      filter_input.blur();
    }
    $$renderer2.push(`<div${attr_class("svelte-1xfsv4t", void 0, { "container": container })}>`);
    if (show_label && buttons && buttons.length > 0) {
      $$renderer2.push("<!--[-->");
      IconButtonWrapper($$renderer2, { buttons, on_custom_button_click: oncustom_button_click });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    BlockTitle($$renderer2, {
      show_label,
      info,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(label)}`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="wrap svelte-1xfsv4t"><div${attr_class("wrap-inner svelte-1xfsv4t", void 0, { "show_options": show_options })}><div class="secondary-wrap svelte-1xfsv4t"><input role="listbox" aria-controls="dropdown-options"${attr("aria-expanded", show_options)}${attr("aria-label", label)}${attr_class("border-none svelte-1xfsv4t", void 0, {
      "subdued": !choices_names.includes(input_text) && !allow_custom_value
    })} autocomplete="off"${attr("disabled", disabled, true)}${attr("value", input_text)}${attr("readonly", !filterable, true)}/> `);
    if (!disabled) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="icon-wrap svelte-1xfsv4t">`);
      DropdownArrow($$renderer2);
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div></div> `);
    DropdownOptions($$renderer2, {
      show_options,
      choices,
      filtered_indices,
      disabled,
      selected_indices,
      active_index,
      onchange: handle_option_selected,
      onload: () => true
    });
    $$renderer2.push(`<!----></div></div>`);
    bind_props($$props, { value });
  });
}

export { DropdownArrow as D, DropdownOptions as a, Dropdown as b };
//# sourceMappingURL=Dropdown-BhyRR6a4.js.map
