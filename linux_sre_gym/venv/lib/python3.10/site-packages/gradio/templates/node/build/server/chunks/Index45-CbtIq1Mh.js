import './async-D55cHugf.js';
import { f as attr_class, a as attr } from './index-u8mz_F03.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { o as onDestroy } from './index-server-CQz6EZl_.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
export { default as BaseExample } from './Example3-CxHLke4-.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

function Calendar($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24"><rect x="2" y="4" width="20" height="18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></rect><line x1="2" y1="9" x2="22" y2="9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></line><line x1="7" y1="2" x2="7" y2="6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></line><line x1="17" y1="2" x2="17" y2="6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></line></svg>`);
}
const date_is_valid_format = (date, include_time) => {
  if (date == null || date === "") return true;
  const valid_regex = include_time ? /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/ : /^\d{4}-\d{2}-\d{2}$/;
  const is_valid_date = date.match(valid_regex) !== null;
  const is_valid_now = date.match(/^(?:\s*now\s*(?:-\s*\d+\s*[dmhs])?)?\s*$/) !== null;
  return is_valid_date || is_valid_now;
};
const parse_date_value = (entered_value, include_time) => {
  if (!entered_value || entered_value === "") {
    const now2 = /* @__PURE__ */ new Date();
    return {
      selected_date: now2,
      current_year: now2.getFullYear(),
      current_month: now2.getMonth(),
      selected_hour: now2.getHours(),
      selected_minute: now2.getMinutes(),
      selected_second: now2.getSeconds(),
      is_pm: now2.getHours() >= 12
    };
  }
  try {
    let date_to_parse = entered_value;
    if (!include_time && entered_value.match(/^\d{4}-\d{2}-\d{2}$/)) {
      date_to_parse += " 00:00:00";
    }
    const parsed = new Date(date_to_parse.replace(" ", "T"));
    if (!isNaN(parsed.getTime())) {
      return {
        selected_date: parsed,
        current_year: parsed.getFullYear(),
        current_month: parsed.getMonth(),
        selected_hour: parsed.getHours(),
        selected_minute: parsed.getMinutes(),
        selected_second: parsed.getSeconds(),
        is_pm: parsed.getHours() >= 12
      };
    }
  } catch (e) {
  }
  const now = /* @__PURE__ */ new Date();
  return {
    selected_date: now,
    current_year: now.getFullYear(),
    current_month: now.getMonth(),
    selected_hour: now.getHours(),
    selected_minute: now.getMinutes(),
    selected_second: now.getSeconds(),
    is_pm: now.getHours() >= 12
  };
};
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value;
    let entered_value = gradio.props.value;
    (/* @__PURE__ */ new Date()).getFullYear();
    (/* @__PURE__ */ new Date()).getMonth();
    (/* @__PURE__ */ new Date()).getHours();
    (/* @__PURE__ */ new Date()).getMinutes();
    (/* @__PURE__ */ new Date()).getSeconds();
    let valid = (() => date_is_valid_format(entered_value, gradio.props.include_time))();
    let disabled = !gradio.shared.interactive;
    const update_picker_from_value = () => {
      const parsed = parse_date_value(entered_value, gradio.props.include_time);
      parsed.selected_date;
      parsed.current_year;
      parsed.current_month;
      parsed.selected_hour;
      parsed.selected_minute;
      parsed.selected_second;
      parsed.is_pm;
    };
    const handle_click_outside = (event) => {
    };
    const handle_scroll = () => {
    };
    onDestroy(() => {
      if (typeof window !== "undefined") {
        window.removeEventListener("click", handle_click_outside);
        window.removeEventListener("scroll", handle_scroll, true);
      }
    });
    update_picker_from_value();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        padding: true,
        children: ($$renderer4) => {
          $$renderer4.push(`<div class="label-content svelte-16sct4k">`);
          if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
            $$renderer4.push("<!--[-->");
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          BlockTitle($$renderer4, {
            show_label: gradio.shared.show_label,
            info: gradio.props.info,
            children: ($$renderer5) => {
              $$renderer5.push(`<!---->${escape_html(gradio.shared.label || "Date")}`);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----></div> <div class="timebox svelte-16sct4k"><input${attr_class("time svelte-16sct4k", void 0, { "invalid": !valid })}${attr("value", entered_value)}${attr("disabled", disabled, true)}${attr("placeholder", gradio.props.include_time ? "YYYY-MM-DD HH:MM:SS" : "YYYY-MM-DD")}/> `);
          if (gradio.shared.interactive) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<button class="calendar svelte-16sct4k"${attr("disabled", disabled, true)}>`);
            Calendar($$renderer4);
            $$renderer4.push(`<!----></button>`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--></div> `);
          {
            $$renderer4.push("<!--[!-->");
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

export { Index as default };
//# sourceMappingURL=Index45-CbtIq1Mh.js.map
