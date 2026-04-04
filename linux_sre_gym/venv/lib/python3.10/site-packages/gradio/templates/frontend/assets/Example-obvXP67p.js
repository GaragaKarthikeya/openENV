import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { t as template_effect, a0 as set_text, a as append, W as from_html, V as child, Y as reset } from './index-BvBk1Iap.js';
/* empty css                                               */

var root = from_html(`<pre> </pre>`);

function Example($$anchor, $$props) {

	let selected = prop($$props, 'selected', 3, false);

	function truncate_text(text, max_length = 60) {
		if (!text) return "";

		const str = String(text);

		if (str.length <= max_length) return str;

		return str.slice(0, max_length) + "...";
	}

	var pre = root();
	let classes;
	var text_1 = child(pre, true);

	reset(pre);

	template_effect(
		($0) => {
			classes = set_class(pre, 1, 'svelte-1bbj91m', null, classes, {
				table: $$props.type === "table",
				gallery: $$props.type === "gallery",
				selected: selected()
			});

			set_text(text_1, $0);
		},
		[() => truncate_text($$props.value)]
	);

	append($$anchor, pre);
}

export { Example as default };
//# sourceMappingURL=Example-obvXP67p.js.map
