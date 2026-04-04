import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, G as deep_read_state, z as untrack, a0 as set_text, a as append, T as pop, U as flushSync, W as from_html, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div><pre> </pre></div>`);

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
	var pre = child(div);
	var text = child(pre, true);

	reset(pre);
	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, 'svelte-pj4oi8', null, classes, {
				table: type() === "table",
				gallery: type() === "gallery",
				selected: selected()
			});

			set_text(text, $0);
		},
		[
			() => (
				deep_read_state(value()),
				untrack(() => JSON.stringify(value(), null, 2))
			)
		]
	);

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-jxpEXkTh.js.map
