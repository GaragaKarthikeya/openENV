import './async-D55cHugf.js';
import { g as attr_style, i as stringify } from './index-u8mz_F03.js';

/* empty css                                           */
function StreamingBar($$renderer, $$props) {
  let { time_limit } = $$props;
  if (time_limit) {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<div class="streaming-bar svelte-1au5sp1"${attr_style("", { "animation-duration": `${stringify(time_limit)}s` })}></div>`);
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]-->`);
}

export { StreamingBar as S };
//# sourceMappingURL=StreamingBar-B1BHh5xK.js.map
