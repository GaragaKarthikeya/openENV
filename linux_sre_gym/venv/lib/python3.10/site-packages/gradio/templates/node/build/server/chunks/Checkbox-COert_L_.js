import './async-D55cHugf.js';
import { f as attr_class, a as attr, d as bind_props } from './index-u8mz_F03.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

function Checkbox($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      label = "Checkbox",
      value = void 0,
      interactive = true,
      show_label = true,
      on_change,
      on_input,
      on_select
    } = $$props;
    let disabled = !interactive;
    $$renderer2.push(`<label${attr_class("checkbox-container svelte-1q8xtp9", void 0, { "disabled": disabled })}><input${attr("checked", value, true)}${attr("disabled", disabled, true)} type="checkbox" name="test" data-testid="checkbox" class="svelte-1q8xtp9"/> `);
    if (show_label) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="label-text svelte-1q8xtp9">${escape_html(label)}</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></label>`);
    bind_props($$props, { value });
  });
}

export { Checkbox as C };
//# sourceMappingURL=Checkbox-COert_L_.js.map
