import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, a as attr, e as ensure_array_like, g as attr_style, i as stringify } from './index-u8mz_F03.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { I as Image } from './Image2-BLbvQKZw.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FullscreenButton } from './FullscreenButton-DVAVExlR.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Maximize-B77VDSzq.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value;
    let active = null;
    let fullscreen = false;
    let label = gradio.shared.label || gradio.i18n("annotated_image.annotated_image");
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        padding: false,
        height: gradio.props.height,
        width: gradio.props.width,
        allow_overflow: false,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        get fullscreen() {
          return fullscreen;
        },
        set fullscreen($$value) {
          fullscreen = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status
          ]));
          $$renderer4.push(`<!----> `);
          BlockLabel($$renderer4, { show_label: gradio.shared.show_label, Icon: Image, label });
          $$renderer4.push(`<!----> <div class="container svelte-1oizopk">`);
          if (gradio.props.value == null) {
            $$renderer4.push("<!--[-->");
            Empty($$renderer4, {
              size: "large",
              unpadded_box: true,
              children: ($$renderer5) => {
                Image($$renderer5);
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer4.push("<!--[!-->");
            $$renderer4.push(`<div class="image-container svelte-1oizopk">`);
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons || [],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              children: ($$renderer5) => {
                if ((gradio.props.buttons || []).some((btn) => typeof btn === "string" && btn === "fullscreen")) {
                  $$renderer5.push("<!--[-->");
                  FullscreenButton($$renderer5, { fullscreen });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]-->`);
              }
            });
            $$renderer4.push(`<!----> <img${attr_class("base-image svelte-1oizopk", void 0, { "fit-height": gradio.props.height && !fullscreen })}${attr("src", gradio.props.value ? gradio.props.value.image.url : null)} alt="the base file that is annotated"/> <!--[-->`);
            const each_array = ensure_array_like(gradio.props.value ? gradio.props.value.annotations : []);
            for (let i = 0, $$length = each_array.length; i < $$length; i++) {
              let ann = each_array[i];
              $$renderer4.push(`<img${attr("alt", `segmentation mask identifying ${stringify(gradio.shared.label)} within the uploaded file`)}${attr_class("mask fit-height svelte-1oizopk", void 0, {
                "fit-height": !fullscreen,
                "active": active == ann.label,
                "inactive": active != ann.label && active != null
              })}${attr("src", ann.image.url)}${attr_style(gradio.props.color_map && ann.label in gradio.props.color_map ? null : `filter: hue-rotate(${Math.round(i * 360 / (gradio.props.value?.annotations.length ?? 1))}deg);`)}/>`);
            }
            $$renderer4.push(`<!--]--></div> `);
            if (gradio.props.show_legend && gradio.props.value) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<div class="legend svelte-1oizopk"><!--[-->`);
              const each_array_1 = ensure_array_like(gradio.props.value.annotations);
              for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
                let ann = each_array_1[i];
                $$renderer4.push(`<button class="legend-item svelte-1oizopk"${attr_style(`background-color: ${stringify(gradio.props.color_map && ann.label in gradio.props.color_map ? gradio.props.color_map[ann.label] + "88" : `hsla(${Math.round(i * 360 / gradio.props.value.annotations.length)}, 100%, 50%, 0.3)`)}`)}>${escape_html(ann.label)}</button>`);
              }
              $$renderer4.push(`<!--]--></div>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
          $$renderer4.push(`<!--]--></div>`);
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
//# sourceMappingURL=Index19-lTckZTfY.js.map
