import { p as prop, e as init, b as set_class, a as set_attribute } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, X as sibling, W as from_html, V as child, G as deep_read_state, z as untrack, a8 as next, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div><img class="svelte-ulqlw7"/> <img class="svelte-ulqlw7"/> <span class="svelte-ulqlw7"></span></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let samples_dir = prop($$props, 'samples_dir', 12);
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

		get samples_dir() {
			return samples_dir();
		},

		set samples_dir($$value) {
			samples_dir($$value);
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
	var img = child(div);
	var img_1 = sibling(img, 2);

	next(2);
	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'wrap svelte-ulqlw7', null, classes, {
			table: type() === "table",
			gallery: type() === "gallery",
			selected: selected()
		});

		set_attribute(img, 'src', (
			deep_read_state(samples_dir()),
			deep_read_state(value()),
			untrack(() => samples_dir() + value()[0])
		));

		set_attribute(img_1, 'src', (
			deep_read_state(samples_dir()),
			deep_read_state(value()),
			untrack(() => samples_dir() + value()[1])
		));
	});

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-CwgO_twd.js.map
