import { p as prop, a as set_attribute } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, f as from_svg } from './index-BvBk1Iap.js';

var root = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" class="feather feather-square"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`);

function Square($$anchor, $$props) {
	push($$props, false);

	let fill = prop($$props, 'fill', 12, "currentColor");
	let stroke_width = prop($$props, 'stroke_width', 12, 1.5);

	var $$exports = {
		get fill() {
			return fill();
		},

		set fill($$value) {
			fill($$value);
			flushSync();
		},

		get stroke_width() {
			return stroke_width();
		},

		set stroke_width($$value) {
			stroke_width($$value);
			flushSync();
		}
	};

	var svg = root();

	template_effect(() => {
		set_attribute(svg, 'fill', fill());
		set_attribute(svg, 'stroke-width', `${stroke_width()}`);
	});

	append($$anchor, svg);

	return pop($$exports);
}

export { Square as S };
//# sourceMappingURL=Square-B6INQV79.js.map
