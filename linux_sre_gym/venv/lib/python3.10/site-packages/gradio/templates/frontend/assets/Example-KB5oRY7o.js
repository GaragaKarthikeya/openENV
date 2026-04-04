import { p as prop, e as init, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, G as deep_read_state, z as untrack, a0 as set_text, a as append, T as pop, U as flushSync, W as from_html, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div> </div>`);

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

	init();

	var div = root();
	let classes;
	var text = child(div, true);

	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, 'svelte-164n2qi', null, classes, {
				table: type() === "table",
				gallery: type() === "gallery",
				selected: selected()
			});

			set_text(text, $0);
		},
		[
			() => (
				deep_read_state(value()),
				untrack(() => value() !== null ? value().toLocaleString() : "")
			)
		]
	);

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-KB5oRY7o.js.map
