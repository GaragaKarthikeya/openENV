import './async-D55cHugf.js';
import { d as bind_props, e as ensure_array_like, g as attr_style, f as attr_class, a as attr, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';

function TextHighlight($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--carbon" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path fill="currentColor" d="M12 15H5a3 3 0 0 1-3-3v-2a3 3 0 0 1 3-3h5V5a1 1 0 0 0-1-1H3V2h6a3 3 0 0 1 3 3zM5 9a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h5V9zm15 14v2a1 1 0 0 0 1 1h5v-4h-5a1 1 0 0 0-1 1z"></path><path fill="currentColor" d="M2 30h28V2Zm26-2h-7a3 3 0 0 1-3-3v-2a3 3 0 0 1 3-3h5v-2a1 1 0 0 0-1-1h-6v-2h6a3 3 0 0 1 3 3Z"></path></svg>`);
}
function is_transparent(color) {
  if (!color) return true;
  const c = color.toLowerCase().trim();
  return c === "transparent" || c.length === 9 && c.endsWith("00");
}
function merge_elements(value, merge_mode) {
  if (value.length === 0) return [];
  const result = [];
  let current_token = value[0].token;
  let current_class = value[0].class_or_confidence;
  for (let i = 1; i < value.length; i++) {
    const { token, class_or_confidence } = value[i];
    const should_merge = current_class === class_or_confidence;
    if (should_merge) {
      current_token += token;
    } else {
      result.push({ token: current_token, class_or_confidence: current_class });
      current_token = token;
      current_class = class_or_confidence;
    }
  }
  result.push({ token: current_token, class_or_confidence: current_class });
  return result;
}
function get_score_color(score) {
  if (score === null) return "";
  if (score < 0) {
    return `rgba(128, 90, 213, ${-score})`;
  }
  return `rgba(239, 68, 60, ${score})`;
}
function LabelInput($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = [],
      label_to_edit = -1,
      category,
      active_legend,
      color_map,
      label_index,
      token,
      onchange,
      is_scores_mode = false
    } = $$props;
    let input_value = category?.toString() ?? "";
    function get_background_color() {
      if (is_scores_mode) {
        const score = typeof category === "number" ? category : parseFloat(category ?? "0");
        return get_score_color(score);
      }
      if (category === null || active_legend && active_legend !== category) {
        return "";
      }
      return color_map[category]?.primary ?? "";
    }
    $$renderer2.push(`<input class="label-input svelte-14ic1z3" autofocus${attr("type", is_scores_mode ? "number" : "text")}${attr("step", is_scores_mode ? "0.1" : void 0)}${attr("placeholder", is_scores_mode ? void 0 : "label")}${attr("value", category)}${attr_style("", {
      "background-color": get_background_color(),
      width: is_scores_mode ? "7ch" : `${(input_value?.length || 4) + 4}ch`
    })}/>`);
    bind_props($$props, { value, label_to_edit });
  });
}
function HighlightedText($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = [],
      show_legend = false,
      show_inline_category = true,
      color_map = {},
      show_whitespaces = false,
      interactive = false,
      onselect,
      onchange
    } = $$props;
    let active_legend = "";
    let label_to_edit = -1;
    let resolved_color_map = {};
    function handle_value_change() {
      onchange?.(value);
      label_to_edit = -1;
    }
    function get_background_color(class_or_confidence) {
      if (class_or_confidence === null) return "";
      return resolved_color_map[class_or_confidence]?.secondary ?? "";
    }
    function get_label_color(class_or_confidence) {
      if (class_or_confidence === null) return "";
      return resolved_color_map[class_or_confidence]?.primary ?? "";
    }
    function get_text_color(class_or_confidence) {
      const bg = get_background_color(class_or_confidence);
      return is_transparent(bg) ? "" : "black";
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="container svelte-1akrbo3">`);
      {
        $$renderer3.push("<!--[-->");
        if (show_legend) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="legend svelte-1akrbo3" data-testid="highlighted-text:category-legend"><!--[-->`);
          const each_array = ensure_array_like(Object.entries(resolved_color_map));
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let [category, colors] = each_array[$$index];
            $$renderer3.push(`<button class="legend-item svelte-1akrbo3"${attr_style("", { "background-color": colors.secondary })}>${escape_html(category)}</button>`);
          }
          $$renderer3.push(`<!--]--></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <div class="textfield svelte-1akrbo3"><!--[-->`);
        const each_array_1 = ensure_array_like(value);
        for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
          let { token, class_or_confidence } = each_array_1[i];
          const lines = token.split("\n");
          $$renderer3.push(`<!--[-->`);
          const each_array_2 = ensure_array_like(lines);
          for (let j = 0, $$length2 = each_array_2.length; j < $$length2; j++) {
            let line = each_array_2[j];
            if (show_whitespaces ? line !== "" : line.trim()) {
              $$renderer3.push("<!--[-->");
              const bg_color = get_background_color(class_or_confidence);
              $$renderer3.push(`<span class="token-container svelte-1akrbo3"><span${attr_class("token svelte-1akrbo3", void 0, {
                "highlighted": class_or_confidence !== null,
                "transparent": class_or_confidence !== null && is_transparent(bg_color),
                "dimmed": active_legend
              })}${attr("role", class_or_confidence !== null ? "button" : void 0)}${attr("tabindex", class_or_confidence !== null ? 0 : void 0)}${attr_style("", {
                "background-color": bg_color,
                color: get_text_color(class_or_confidence)
              })}><span${attr_class("text svelte-1akrbo3", void 0, { "unlabeled": class_or_confidence === null })}>${escape_html(line)}</span> `);
              if (!show_legend && show_inline_category && class_or_confidence !== null && label_to_edit !== i) {
                $$renderer3.push("<!--[-->");
                $$renderer3.push(`<span class="label svelte-1akrbo3"${attr_style("", { "background-color": get_label_color(class_or_confidence) })}>${escape_html(class_or_confidence)}</span>`);
              } else {
                $$renderer3.push("<!--[!-->");
              }
              $$renderer3.push(`<!--]--> `);
              if (interactive && label_to_edit === i && class_or_confidence !== null) {
                $$renderer3.push("<!--[-->");
                LabelInput($$renderer3, {
                  category: class_or_confidence,
                  active_legend,
                  color_map: resolved_color_map,
                  label_index: i,
                  token,
                  onchange: handle_value_change,
                  get value() {
                    return value;
                  },
                  set value($$value) {
                    value = $$value;
                    $$settled = false;
                  },
                  get label_to_edit() {
                    return label_to_edit;
                  },
                  set label_to_edit($$value) {
                    label_to_edit = $$value;
                    $$settled = false;
                  }
                });
              } else {
                $$renderer3.push("<!--[!-->");
              }
              $$renderer3.push(`<!--]--></span> `);
              if (interactive && class_or_confidence !== null) {
                $$renderer3.push("<!--[-->");
                $$renderer3.push(`<button class="remove-btn svelte-1akrbo3" aria-label="Remove label">`);
                Clear($$renderer3);
                $$renderer3.push(`<!----></button>`);
              } else {
                $$renderer3.push("<!--[!-->");
              }
              $$renderer3.push(`<!--]--></span>`);
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (j < lines.length - 1) {
              $$renderer3.push("<!--[-->");
              $$renderer3.push(`<span class="line-break svelte-1akrbo3"></span>`);
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]-->`);
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]--></div>`);
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value;
    let value = (() => gradio.props.combine_adjacent ? merge_elements(gradio.props.value || []) : gradio.props.value)();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        variant: gradio.shared.interactive ? "dashed" : "solid",
        test_id: "highlighted-text",
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        padding: false,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        rtl: gradio.props.rtl,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              onclearstatus: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (gradio.shared.interactive && gradio.shared.label && gradio.shared.show_label && gradio.props.buttons?.length) {
            $$renderer4.push("<!--[-->");
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => gradio.dispatch("custom_button_click", { id })
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (gradio.shared.label && gradio.shared.show_label) {
            $$renderer4.push("<!--[-->");
            BlockLabel($$renderer4, {
              Icon: TextHighlight,
              label: gradio.shared.label || gradio.i18n("highlighted_text.highlighted_text"),
              float: false,
              disable: gradio.shared.container === false,
              show_label: gradio.shared.show_label,
              rtl: gradio.props.rtl
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (value) {
            $$renderer4.push("<!--[-->");
            HighlightedText($$renderer4, {
              interactive: gradio.shared.interactive,
              show_legend: gradio.props.show_legend,
              show_inline_category: gradio.props.show_inline_category,
              show_whitespaces: gradio.props.show_whitespaces,
              color_map: gradio.props.color_map,
              onselect: (detail) => gradio.dispatch("select", detail),
              onchange: () => {
                gradio.props.value = value;
                gradio.dispatch("change");
              },
              get value() {
                return value;
              },
              set value($$value) {
                value = $$value;
                $$settled = false;
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
            Empty($$renderer4, {
              size: gradio.shared.interactive ? "small" : "large",
              unpadded_box: gradio.shared.interactive,
              children: ($$renderer5) => {
                TextHighlight($$renderer5);
              },
              $$slots: { default: true }
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

export { HighlightedText as BaseHighlightedText, Index as default };
//# sourceMappingURL=Index48-DrPj8942.js.map
