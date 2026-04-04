import { i as if_block } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, R as push, a6 as comment, S as first_child, T as pop } from './index-BvBk1Iap.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { M as Maximize } from './Maximize-B_43ylTs.js';

var root = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-minimize" width="100%" height="100%"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path></svg>`);

function Minimize($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

function FullscreenButton($$anchor, $$props) {
	push($$props, true);

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			IconButton($$anchor, {
				get Icon() {
					return Minimize;
				},
				label: 'Exit fullscreen mode',
				onclick: () => $$props.onclick(false)
			});
		};

		var alternate = ($$anchor) => {
			IconButton($$anchor, {
				get Icon() {
					return Maximize;
				},
				label: 'Fullscreen',
				onclick: () => $$props.onclick(true)
			});
		};

		if_block(node, ($$render) => {
			if ($$props.fullscreen) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { FullscreenButton as F };
//# sourceMappingURL=FullscreenButton-CNmhk6nd.js.map
