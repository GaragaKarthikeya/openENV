import { f as fallback } from './async-D55cHugf.js';
import { a as attr, d as bind_props } from './index-u8mz_F03.js';

function Square($$renderer, $$props) {
  let fill = fallback($$props["fill"], "currentColor");
  let stroke_width = fallback($$props["stroke_width"], 1.5);
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24"${attr("fill", fill)} stroke="currentColor"${attr("stroke-width", `${stroke_width}`)} stroke-linecap="round" stroke-linejoin="round" class="feather feather-square"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`);
  bind_props($$props, { fill, stroke_width });
}

export { Square as S };
//# sourceMappingURL=Square-CJzdI5ja.js.map
