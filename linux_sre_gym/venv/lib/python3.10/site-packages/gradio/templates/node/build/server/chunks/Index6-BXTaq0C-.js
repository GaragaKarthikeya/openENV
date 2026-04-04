import { f as fallback } from './async-D55cHugf.js';
import { d as bind_props, s as slot } from './index-u8mz_F03.js';
import { B as Block } from './Block-JRVGROlG.js';
import './2-BAeILJ1g.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Index($$renderer, $$props) {
  let elem_id = $$props["elem_id"];
  let elem_classes = $$props["elem_classes"];
  let visible = fallback($$props["visible"], true);
  Block($$renderer, {
    elem_id,
    elem_classes,
    visible,
    explicit_call: true,
    children: ($$renderer2) => {
      $$renderer2.push(`<!--[-->`);
      slot($$renderer2, $$props, "default", {});
      $$renderer2.push(`<!--]-->`);
    },
    $$slots: { default: true }
  });
  bind_props($$props, { elem_id, elem_classes, visible });
}

export { Index as default };
//# sourceMappingURL=Index6-BXTaq0C-.js.map
