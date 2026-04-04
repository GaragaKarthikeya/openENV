import './async-D55cHugf.js';
import { c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { D as Download } from './Download-ByiErn53.js';
import { L as LineChart } from './LineChart-Bgd0tmH8.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FullscreenButton } from './FullscreenButton-DVAVExlR.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './Maximize-B77VDSzq.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    (() => {
      if (!gradio.props.color || !gradio.props.value || gradio.props.value.datatypes[gradio.props.color] !== "nominal") {
        return [];
      }
      const color_index = gradio.props.value.columns.indexOf(gradio.props.color);
      if (color_index === -1) return [];
      return Array.from(new Set(gradio.props.value.data.map((row) => row[color_index])));
    })();
    let x_lim = gradio.props.x_lim || null;
    let y_lim = gradio.props.y_lim || null;
    x_lim?.[0] !== null ? x_lim?.[0] : void 0;
    x_lim?.[1] !== null ? x_lim?.[1] : void 0;
    y_lim?.[0] !== null ? y_lim?.[0] : void 0;
    y_lim?.[1] !== null ? y_lim?.[1] : void 0;
    let fullscreen = false;
    function reformat_sort(_sort) {
      if (_sort === "x") {
        return "ascending";
      } else if (_sort === "-x") {
        return "descending";
      } else if (_sort === "y") {
        return { field: gradio.props.y, order: "ascending" };
      } else if (_sort === "-y") {
        return { field: gradio.props.y, order: "descending" };
      } else if (_sort === null) {
        return null;
      } else if (Array.isArray(_sort)) {
        return _sort;
      }
    }
    reformat_sort(gradio.props.sort);
    gradio.props.value && gradio.props.value.datatypes[gradio.props.x] === "temporal";
    const SUFFIX_DURATION = { s: 1, m: 60, h: 60 * 60, d: 24 * 60 * 60 };
    let _x_bin = gradio.props.x_bin ? typeof gradio.props.x_bin === "string" ? 1e3 * parseInt(gradio.props.x_bin.substring(0, gradio.props.x_bin.length - 1)) * SUFFIX_DURATION[gradio.props.x_bin[gradio.props.x_bin.length - 1]] : gradio.props.x_bin : void 0;
    (() => {
      if (gradio.props.value) {
        if (gradio.props.value.mark === "point") {
          const aggregating = _x_bin !== void 0;
          return gradio.props.y_aggregate || aggregating ? "sum" : void 0;
        } else {
          return gradio.props.y_aggregate ? gradio.props.y_aggregate : "sum";
        }
      }
      return void 0;
    })();
    (() => {
      if (gradio.props.value) {
        if (gradio.props.value.mark === "point") {
          return _x_bin !== void 0;
        } else {
          return _x_bin !== void 0 || gradio.props.value.datatypes[gradio.props.x] === "nominal";
        }
      }
      return false;
    })();
    gradio.props.value;
    const is_browser = typeof window !== "undefined";
    function export_chart() {
      return;
    }
    JSON.stringify(gradio.props.color_map);
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
        height: gradio.props.height,
        get fullscreen() {
          return fullscreen;
        },
        set fullscreen($$value) {
          fullscreen = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          if (gradio.shared.loading_status) {
            $$renderer4.push("<!--[-->");
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (gradio.props.buttons?.length) {
            $$renderer4.push("<!--[-->");
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              children: ($$renderer5) => {
                if (gradio.props.buttons?.some((btn) => typeof btn === "string" && btn === "export")) {
                  $$renderer5.push("<!--[-->");
                  IconButton($$renderer5, { Icon: Download, label: "Export", onclick: export_chart });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]--> `);
                if (gradio.props.buttons?.some((btn) => typeof btn === "string" && btn === "fullscreen")) {
                  $$renderer5.push("<!--[-->");
                  FullscreenButton($$renderer5, { fullscreen });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]-->`);
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          BlockTitle($$renderer4, {
            show_label: gradio.shared.show_label,
            info: void 0,
            children: ($$renderer5) => {
              $$renderer5.push(`<!---->${escape_html(gradio.shared.label)}`);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          if (gradio.props.value && is_browser) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<div class="svelte-19utvcn"></div> `);
            if (gradio.props.caption) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<p class="caption svelte-19utvcn">${escape_html(gradio.props.caption)}</p>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          } else {
            $$renderer4.push("<!--[!-->");
            Empty($$renderer4, {
              unpadded_box: true,
              children: ($$renderer5) => {
                LineChart($$renderer5);
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

export { Index as default };
//# sourceMappingURL=Index27-G5VczEpy.js.map
