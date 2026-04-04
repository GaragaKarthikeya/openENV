import { p as prop, a as set_attribute, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, V as child, W as from_html, X as sibling, a0 as set_text, Y as reset } from './index-BvBk1Iap.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root = from_html(`<label for="" data-testid="block-label"><span class="svelte-19djge9"><!></span> </label>`);

function BlockLabel($$anchor, $$props) {
	push($$props, false);

	let label = prop($$props, 'label', 12, null);
	let Icon = prop($$props, 'Icon', 12);
	let show_label = prop($$props, 'show_label', 12, true);
	let disable = prop($$props, 'disable', 12, false);
	let float = prop($$props, 'float', 12, true);
	let rtl = prop($$props, 'rtl', 12, false);

	var $$exports = {
		get label() {
			return label();
		},

		set label($$value) {
			label($$value);
			flushSync();
		},

		get Icon() {
			return Icon();
		},

		set Icon($$value) {
			Icon($$value);
			flushSync();
		},

		get show_label() {
			return show_label();
		},

		set show_label($$value) {
			show_label($$value);
			flushSync();
		},

		get disable() {
			return disable();
		},

		set disable($$value) {
			disable($$value);
			flushSync();
		},

		get float() {
			return float();
		},

		set float($$value) {
			float($$value);
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

	var label_1 = root();
	let classes;
	var span = child(label_1);
	var node = child(span);

	Icon()(node, {});
	reset(span);

	var text = sibling(span);

	reset(label_1);

	template_effect(() => {
		set_attribute(label_1, 'dir', rtl() ? "rtl" : "ltr");

		classes = set_class(label_1, 1, 'svelte-19djge9', null, classes, {
			hide: !show_label(),
			'sr-only': !show_label(),
			float: float(),
			'hide-label': disable()
		});

		set_text(text, ` ${label() ?? ''}`);
		label_1.dir = label_1.dir;
	});

	append($$anchor, label_1);

	return pop($$exports);
}

export { BlockLabel as B };
//# sourceMappingURL=BlockLabel-oGSqUii_.js.map
