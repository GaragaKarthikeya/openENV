import { p as prop, e as init } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, R as push, a4 as createEventDispatcher, x as set, w as get, T as pop, a3 as mutable_source, U as flushSync, ad as derived_safe_equal, G as deep_read_state, z as untrack } from './index-BvBk1Iap.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { S as ShareError } from './utils.svelte-CYLB_jIA.js';

var root = from_svg(`<svg id="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="100%" height="100%"><path d="M23,20a5,5,0,0,0-3.89,1.89L11.8,17.32a4.46,4.46,0,0,0,0-2.64l7.31-4.57A5,5,0,1,0,18,7a4.79,4.79,0,0,0,.2,1.32l-7.31,4.57a5,5,0,1,0,0,6.22l7.31,4.57A4.79,4.79,0,0,0,18,25a5,5,0,1,0,5-5ZM23,4a3,3,0,1,1-3,3A3,3,0,0,1,23,4ZM7,19a3,3,0,1,1,3-3A3,3,0,0,1,7,19Zm16,9a3,3,0,1,1,3-3A3,3,0,0,1,23,28Z" fill="currentColor"></path></svg>`);

function Community($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

function ShareButton($$anchor, $$props) {
	push($$props, false);

	const dispatch = createEventDispatcher();
	let formatter = prop($$props, 'formatter', 12);
	let value = prop($$props, 'value', 12);
	let i18n = prop($$props, 'i18n', 12);
	let pending = mutable_source(false);

	var $$exports = {
		get formatter() {
			return formatter();
		},

		set formatter($$value) {
			formatter($$value);
			flushSync();
		},

		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get i18n() {
			return i18n();
		},

		set i18n($$value) {
			i18n($$value);
			flushSync();
		}
	};

	init();

	{
		let $0 = derived_safe_equal(() => (
			deep_read_state(i18n()),
			untrack(() => i18n()("common.share"))
		));

		IconButton($$anchor, {
			get Icon() {
				return Community;
			},

			get label() {
				return get($0);
			},

			get pending() {
				return get(pending);
			},

			onclick: async () => {
				try {
					set(pending, true);

					const formatted = await formatter()(value());

					dispatch("share", { description: formatted });
				} catch(e) {
					console.error(e);

					let message = e instanceof ShareError ? e.message : "Share failed.";

					dispatch("error", message);
				} finally {
					set(pending, false);
				}
			}
		});
	}

	return pop($$exports);
}

export { Community as C, ShareButton as S };
//# sourceMappingURL=ShareButton-Dz0y1C1K.js.map
