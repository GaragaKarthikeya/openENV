import './async-D55cHugf.js';
import { f as attr_class, g as attr_style, e as ensure_array_like, a as attr, i as stringify, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { L as LineChart } from './LineChart-Bgd0tmH8.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function Label($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      color = void 0,
      selectable = false,
      show_heading = true,
      onselect
    } = $$props;
    function get_aria_referenceable_id(elem_id) {
      return elem_id.replace(/\s/g, "-");
    }
    $$renderer2.push(`<div class="container svelte-g2cwl3">`);
    if (show_heading || !value.confidences) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<h2${attr_class("output-class svelte-g2cwl3", void 0, { "no-confidence": !("confidences" in value) })} data-testid="label-output-value"${attr_style("", { "background-color": color || "transparent" })}>${escape_html(value.label)}</h2>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (typeof value === "object" && value.confidences) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(value.confidences);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let confidence_set = each_array[i];
        $$renderer2.push(`<button${attr_class("confidence-set group svelte-g2cwl3", void 0, { "selectable": selectable })}${attr("data-testid", `${confidence_set.label}-confidence-set`)}><div class="inner-wrap svelte-g2cwl3"><meter${attr("aria-labelledby", get_aria_referenceable_id(`meter-text-${confidence_set.label}`))}${attr("aria-label", confidence_set.label)}${attr("aria-valuenow", Math.round(confidence_set.confidence * 100))} aria-valuemin="0" aria-valuemax="100" class="bar svelte-g2cwl3" min="0" max="1"${attr("value", confidence_set.confidence)}${attr_style(`width: ${stringify(confidence_set.confidence * 100)}%; background: var(--stat-background-fill); `)}></meter> <dl class="label svelte-g2cwl3"><dt${attr("id", get_aria_referenceable_id(`meter-text-${confidence_set.label}`))} class="text svelte-g2cwl3">${escape_html(confidence_set.label)}</dt> <div class="line svelte-g2cwl3"></div> <dd class="confidence svelte-g2cwl3">${escape_html(Math.round(confidence_set.confidence * 100))}%</dd></dl></div></button>`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value;
    let _label = gradio.props.value.label;
    Block($$renderer2, {
      test_id: "label",
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container: gradio.shared.container,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      padding: false,
      children: ($$renderer3) => {
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
          }
        ]));
        $$renderer3.push(`<!----> `);
        if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
          $$renderer3.push("<!--[-->");
          IconButtonWrapper($$renderer3, {
            buttons: gradio.props.buttons,
            on_custom_button_click: (id) => {
              gradio.dispatch("custom_button_click", { id });
            }
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (gradio.shared.show_label) {
          $$renderer3.push("<!--[-->");
          BlockLabel($$renderer3, {
            Icon: LineChart,
            label: gradio.shared.label || gradio.i18n("label.label"),
            disable: gradio.shared.container === false,
            float: gradio.props.show_heading === true
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (_label !== void 0 && _label !== null) {
          $$renderer3.push("<!--[-->");
          Label($$renderer3, {
            onselect: (detail) => gradio.dispatch("select", detail),
            selectable: gradio.props._selectable,
            value: gradio.props.value,
            color: gradio.props.color,
            show_heading: gradio.props.show_heading
          });
        } else {
          $$renderer3.push("<!--[!-->");
          Empty($$renderer3, {
            unpadded_box: true,
            children: ($$renderer4) => {
              LineChart($$renderer4);
            },
            $$slots: { default: true }
          });
        }
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Label as BaseLabel, Index as default };
//# sourceMappingURL=Index34-D97S0Jn5.js.map
