import { p as prop, e as init, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, x as set, t as template_effect, a as append, T as pop, a3 as mutable_source, U as flushSync, W as from_html, a0 as set_text, w as get, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div> </div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);
	let choices = prop($$props, 'choices', 12);
	let name_string = mutable_source();

	if (value() === null) {
		set(name_string, "");
	} else {
		let name = choices().find((pair) => pair[1] === value());

		set(name_string, name ? name[0] : "");
	}

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
		},

		get choices() {
			return choices();
		},

		set choices($$value) {
			choices($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	let classes;
	var text = child(div, true);

	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'svelte-g2dls0', null, classes, {
			table: type() === "table",
			gallery: type() === "gallery",
			selected: selected()
		});

		set_text(text, get(name_string));
	});

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-GBtWCYGT.js.map
