import { l as legacy_rest_props, p as prop, e as init, c as attribute_effect, s as slot, S as STYLE } from './i18n-CMo5lpzy.js';
import { R as push, a4 as createEventDispatcher, Z as event, a as append, T as pop, U as flushSync, z as untrack, V as child, W as from_html, w as get, a5 as user_derived, Y as reset } from './index-BvBk1Iap.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root = from_html(`<a><!></a>`);

function DownloadLink($$anchor, $$props) {
	const $$sanitized_props = legacy_rest_props($$props, ['children', '$$slots', '$$events', '$$legacy']);
	const $$restProps = legacy_rest_props($$sanitized_props, ['href', 'download']);

	push($$props, false);

	let href = prop($$props, 'href', 12, undefined);
	let download = prop($$props, 'download', 12);
	const dispatch = createEventDispatcher();

	var $$exports = {
		get href() {
			return href();
		},

		set href($$value) {
			href($$value);
			flushSync();
		},

		get download() {
			return download();
		},

		set download($$value) {
			download($$value);
			flushSync();
		}
	};

	init();

	var a = root();
	var event_handler = user_derived(() => dispatch.bind(null, "click"));

	attribute_effect(
		a,
		() => ({
			class: 'download-link',
			href: href(),
			target: (
				untrack(() => typeof window !== "undefined" && window.__is_colab__ ? "_blank" : null)
			),
			rel: 'noopener noreferrer',
			download: download(),
			...$$restProps,
			[STYLE]: { position: 'relative' }
		}),
		void 0,
		void 0,
		void 0,
		'svelte-7nkusa'
	);

	var node = child(a);

	slot(node, $$props, 'default', {});
	reset(a);

	event('click', a, function (...$$args) {
		get(event_handler)?.apply(this, $$args);
	});

	append($$anchor, a);

	return pop($$exports);
}

export { DownloadLink as D };
//# sourceMappingURL=DownloadLink-KvrFKyMn.js.map
