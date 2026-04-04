import { p as prop, e as init, i as if_block, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, W as from_html, V as child, G as deep_read_state, z as untrack, Y as reset } from './index-BvBk1Iap.js';
import { I as Image } from './Image-CDw5qb5O.js';
/* empty css                                               */
import './misc-ByVveAau.js';
/* empty css                                             */

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

	init();

	var div = root();
	let classes;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			Image($$anchor, {
				get src() {
					return (deep_read_state(value()), untrack(() => value().url));
				},
				alt: ''
			});
		};

		if_block(node, ($$render) => {
			if (value()) $$render(consequent);
		});
	}

	reset(div);

	template_effect(() => classes = set_class(div, 1, 'container svelte-bs74gu', null, classes, {
		table: type() === "table",
		gallery: type() === "gallery",
		selected: selected(),
		border: value()
	}));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-C7SZxPdR.js.map
