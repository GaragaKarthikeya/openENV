import { i as if_block, f as set_style } from './i18n-CMo5lpzy.js';
import { a6 as comment, S as first_child, a as append, t as template_effect, W as from_html } from './index-BvBk1Iap.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';

var root_1 = from_html(`<div class="streaming-bar svelte-1au5sp1"></div>`);

function StreamingBar($$anchor, $$props) {

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			var div = root_1();
			let styles;

			template_effect(() => styles = set_style(div, '', styles, { 'animation-duration': `${$$props.time_limit ?? ''}s` }));
			append($$anchor, div);
		};

		if_block(node, ($$render) => {
			if ($$props.time_limit) $$render(consequent);
		});
	}

	append($$anchor, fragment);
}

export { StreamingBar as S };
//# sourceMappingURL=StreamingBar-bVF1kkqe.js.map
