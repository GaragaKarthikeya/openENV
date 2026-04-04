import { p as prop, e as init } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, W as from_html, a0 as set_text, G as deep_read_state, z as untrack, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div style="display: none;"> </div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 28, () => ({ visible: true, home_page_title: "Home" }));

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	var text = child(div);

	reset(div);

	template_effect(() => set_text(text, `Navbar config: visible=${(deep_read_state(value()), untrack(() => value().visible)) ?? ''}, home_page_title="${(
		deep_read_state(value()),
		untrack(() => value().home_page_title)
	) ?? ''}"`));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-7kC6V8Ah.js.map
