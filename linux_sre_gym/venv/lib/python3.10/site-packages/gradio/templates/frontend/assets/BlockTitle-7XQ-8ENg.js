import { p as prop, s as slot, i as if_block, a as set_attribute, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, t as template_effect, a as append, T as pop, U as flushSync, V as child, W as from_html, X as sibling, Y as reset } from './index-BvBk1Iap.js';
import { I as Info } from './Info-BrkJhX6_.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root = from_html(`<span data-testid="block-info"><!></span> <!>`, 1);

function BlockTitle($$anchor, $$props) {
	push($$props, false);

	let show_label = prop($$props, 'show_label', 12, true);
	let info = prop($$props, 'info', 12, undefined);
	let rtl = prop($$props, 'rtl', 12, false);

	var $$exports = {
		get show_label() {
			return show_label();
		},

		set show_label($$value) {
			show_label($$value);
			flushSync();
		},

		get info() {
			return info();
		},

		set info($$value) {
			info($$value);
			flushSync();
		},

		get rtl() {
			return rtl();
		},

		set rtl($$value) {
			rtl($$value);
			flushSync();
		}
	};

	var fragment = root();
	var span = first_child(fragment);
	let classes;
	var node = child(span);

	slot(node, $$props, 'default', {});
	reset(span);

	var node_1 = sibling(span, 2);

	{
		var consequent = ($$anchor) => {
			Info($$anchor, {
				get info() {
					return info();
				}
			});
		};

		if_block(node_1, ($$render) => {
			if (info()) $$render(consequent);
		});
	}

	template_effect(() => {
		set_attribute(span, 'dir', rtl() ? "rtl" : "ltr");

		classes = set_class(span, 1, 'svelte-jdcl7l', null, classes, {
			hide: !show_label(),
			'has-info': info() != null,
			'sr-only': !show_label()
		});

		span.dir = span.dir;
	});

	append($$anchor, fragment);

	return pop($$exports);
}

export { BlockTitle as B };
//# sourceMappingURL=BlockTitle-7XQ-8ENg.js.map
