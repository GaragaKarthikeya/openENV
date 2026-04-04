import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, W as from_html, a0 as set_text, V as child, Y as reset } from './index-BvBk1Iap.js';
/* empty css                                               */

var root = from_html(`<div> </div>`);

function Example($$anchor, $$props) {
	push($$props, true);

	let selected = prop($$props, 'selected', 3, false);

	let value_array = $$props.value
		? Array.isArray($$props.value) ? $$props.value : [$$props.value]
		: [];

	let names = value_array.map((val) => $$props.choices.find((pair) => pair[1] === val)?.[0]).filter((name) => name !== undefined);
	let names_string = names.join(", ");
	var div = root();
	let classes;
	var text = child(div, true);

	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'svelte-1by696e', null, classes, {
			table: $$props.type === "table",
			gallery: $$props.type === "gallery",
			selected: selected()
		});

		set_text(text, names_string);
	});

	append($$anchor, div);
	pop();
}

export { Example as default };
//# sourceMappingURL=Example-t0cB3HBf.js.map
