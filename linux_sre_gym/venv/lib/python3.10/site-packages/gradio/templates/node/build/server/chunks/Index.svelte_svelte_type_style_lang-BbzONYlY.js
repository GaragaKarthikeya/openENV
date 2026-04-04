import './async-D55cHugf.js';
import { a as attr, f as attr_class, g as attr_style, i as stringify, c as spread_props, s as slot } from './index-u8mz_F03.js';
import { S as Static } from './index3-BSof0MuO.js';

function BaseColumn($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    let scale = props.scale ?? null;
    let min_width = props.min_width ?? 0;
    let elem_id = props.elem_id ?? "";
    let elem_classes = props.elem_classes ?? [];
    let visible = props.visible ?? true;
    let variant = props.variant ?? "default";
    let loading_status = props.loading_status;
    props.show_progress ?? false;
    $$renderer2.push(`<div${attr("id", elem_id)}${attr_class(`column ${stringify(elem_classes.join(" "))}`, "svelte-siq5d6", {
      "compact": variant === "compact",
      "panel": variant === "panel",
      "hide": !visible
    })}${attr_style("", {
      "flex-grow": scale,
      "min-width": `calc(min(${stringify(min_width)}px, 100%))`
    })}>`);
    if (loading_status && loading_status.show_progress) {
      $$renderer2.push("<!--[-->");
      Static($$renderer2, spread_props([
        { autoscroll: props.autoscroll, i18n: props.i18n },
        loading_status,
        {
          status: loading_status ? loading_status.status == "pending" ? "generating" : loading_status.status : null
        }
      ]));
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div>`);
  });
}

export { BaseColumn as B };
//# sourceMappingURL=Index.svelte_svelte_type_style_lang-BbzONYlY.js.map
