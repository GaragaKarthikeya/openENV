import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, s as slot, d as bind_props } from './index-u8mz_F03.js';

/* empty css                                         */
function Empty($$renderer, $$props) {
  let parent_height;
  let size = fallback($$props["size"], "small");
  let unpadded_box = fallback($$props["unpadded_box"], false);
  function compare_el_to_parent(el) {
    return false;
  }
  parent_height = compare_el_to_parent();
  $$renderer.push(`<div${attr_class("empty svelte-v95lt3", void 0, {
    "small": size === "small",
    "large": size === "large",
    "unpadded_box": unpadded_box,
    "small_parent": parent_height
  })} aria-label="Empty value"><div class="icon svelte-v95lt3"><!--[-->`);
  slot($$renderer, $$props, "default", {});
  $$renderer.push(`<!--]--></div></div>`);
  bind_props($$props, { size, unpadded_box });
}

export { Empty as E };
//# sourceMappingURL=Empty-BHvMDNv5.js.map
