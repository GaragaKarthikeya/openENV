import './async-D55cHugf.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import './2-BAeILJ1g.js';
import './index-u8mz_F03.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Plot$2($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--carbon" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><circle cx="20" cy="4" r="2" fill="currentColor"></circle><circle cx="8" cy="16" r="2" fill="currentColor"></circle><circle cx="28" cy="12" r="2" fill="currentColor"></circle><circle cx="11" cy="7" r="2" fill="currentColor"></circle><circle cx="16" cy="24" r="2" fill="currentColor"></circle><path fill="currentColor" d="M30 3.413L28.586 2L4 26.585V2H2v26a2 2 0 0 0 2 2h26v-2H5.413Z"></path></svg>`);
}
function Plot($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      theme_mode,
      caption,
      bokeh_version,
      show_actions_button,
      _selectable,
      x_lim,
      show_fullscreen_button,
      show_label,
      on_change
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      {
        $$renderer3.push("<!--[!-->");
        Empty($$renderer3, {
          unpadded_box: true,
          size: "large",
          children: ($$renderer4) => {
            Plot$2($$renderer4);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
const Plot$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: Plot
}, Symbol.toStringTag, { value: "Module" }));

export { Plot$2 as P, Plot as a, Plot$1 as b };
//# sourceMappingURL=Plot-DQRGUyXc.js.map
