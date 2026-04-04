import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, W as from_html, V as child, Y as reset } from './index-BvBk1Iap.js';
import { h as html } from './html-CdUEtR5E.js';

var root = from_html(`<div><!></div>`);

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
	var node = child(div);

	html(node, value);
	reset(div);

	template_effect(() => classes = set_class(div, 1, 'prose svelte-s7j0w2', null, classes, {
		table: type() === "table",
		gallery: type() === "gallery",
		selected: selected()
	}));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-BEiY6GLN.js.map
