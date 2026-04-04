import { f as fallback } from './async-D55cHugf.js';
import { c as spread_props, f as attr_class, e as ensure_array_like, i as stringify, d as bind_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function JsonView($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let json = $$props["json"];
    let depth = fallback($$props["depth"], Infinity);
    let _cur = fallback($$props["_cur"], 0);
    let _last = fallback($$props["_last"], true);
    let items = [];
    let isArray = false;
    let brackets = ["", ""];
    let collapsed = false;
    function getType(i) {
      if (i === null) return "null";
      return typeof i;
    }
    function stringify$1(i) {
      return JSON.stringify(i);
    }
    function format(i) {
      switch (getType(i)) {
        case "function":
          return "f () {...}";
        case "symbol":
          return i.toString();
        default:
          return stringify$1(i);
      }
    }
    {
      items = getType(json) === "object" ? Object.keys(json) : [];
      isArray = Array.isArray(json);
      brackets = isArray ? ["[", "]"] : ["{", "}"];
    }
    collapsed = depth < _cur;
    if (!items.length) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span${attr_class("_jsonBkt empty svelte-zxj9ye", void 0, { "isArray": isArray })}>${escape_html(brackets[0])}${escape_html(brackets[1])}</span>`);
      if (!_last) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="_jsonSep svelte-zxj9ye">,</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (collapsed) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span${attr_class("_jsonBkt svelte-zxj9ye", void 0, { "isArray": isArray })} role="button" tabindex="0">${escape_html(brackets[0])}...${escape_html(brackets[1])}</span>`);
        if (!_last && collapsed) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="_jsonSep svelte-zxj9ye">,</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<span${attr_class("_jsonBkt svelte-zxj9ye", void 0, { "isArray": isArray })} role="button" tabindex="0">${escape_html(brackets[0])}</span> <ul class="_jsonList svelte-zxj9ye"><!--[-->`);
        const each_array = ensure_array_like(items);
        for (let idx = 0, $$length = each_array.length; idx < $$length; idx++) {
          let i = each_array[idx];
          $$renderer2.push(`<li>`);
          if (!isArray) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<span class="_jsonKey svelte-zxj9ye">${escape_html(stringify$1(i))}</span><span class="_jsonSep svelte-zxj9ye">:</span>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]--> `);
          if (getType(json[i]) === "object") {
            $$renderer2.push("<!--[-->");
            JsonView($$renderer2, {
              json: json[i],
              depth,
              _cur: _cur + 1,
              _last: idx === items.length - 1
            });
            $$renderer2.push(`<!---->`);
          } else {
            $$renderer2.push("<!--[!-->");
            $$renderer2.push(`<span${attr_class(`_jsonVal ${stringify(getType(json[i]))}`, "svelte-zxj9ye")}>${escape_html(format(json[i]))}</span>`);
            if (idx < items.length - 1) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<span class="_jsonSep svelte-zxj9ye">,</span>`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]-->`);
          }
          $$renderer2.push(`<!--]--></li>`);
        }
        $$renderer2.push(`<!--]--></ul> <span${attr_class("_jsonBkt svelte-zxj9ye", void 0, { "isArray": isArray })} role="button" tabindex="0">${escape_html(brackets[1])}</span>`);
        if (!_last) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="_jsonSep svelte-zxj9ye">,</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { json, depth, _cur, _last });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    const container = true;
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      children: ($$renderer3) => {
        if (gradio.shared.loading_status) {
          $$renderer3.push("<!--[-->");
          Static($$renderer3, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        JsonView($$renderer3, { json: gradio.props.value });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index39-C-nV5fyj.js.map
