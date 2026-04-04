import './async-D55cHugf.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { M as Maximize } from './Maximize-B77VDSzq.js';
import './2-BAeILJ1g.js';

function Minimize($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-minimize" width="100%" height="100%"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path></svg>`);
}
function FullscreenButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { fullscreen, onclick } = $$props;
    if (fullscreen) {
      $$renderer2.push("<!--[-->");
      IconButton($$renderer2, {
        Icon: Minimize,
        label: "Exit fullscreen mode",
        onclick: () => onclick(false)
      });
    } else {
      $$renderer2.push("<!--[!-->");
      IconButton($$renderer2, {
        Icon: Maximize,
        label: "Fullscreen",
        onclick: () => onclick(true)
      });
    }
    $$renderer2.push(`<!--]-->`);
  });
}

export { FullscreenButton as F };
//# sourceMappingURL=FullscreenButton-DVAVExlR.js.map
