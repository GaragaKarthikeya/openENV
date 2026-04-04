import { p as prop, f as set_style, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, W as from_html } from './index-BvBk1Iap.js';

var root = from_html(`<div></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get type() {
			return type();
		},

		set type($$value) {
			type($$value);
			flushSync();
		},

		get selected() {
			return selected();
		},

		set selected($$value) {
			selected($$value);
			flushSync();
		}
	};

	var div = root();
	let classes;

	template_effect(() => {
		set_style(div, `background-color: ${(value() ? value() : 'black') ?? ''}`);

		classes = set_class(div, 1, 'svelte-1k1s8qu', null, classes, {
			table: type() === "table",
			gallery: type() === "gallery",
			selected: selected()
		});
	});

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-CCEyD4GV.js.map
