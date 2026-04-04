import { p as prop, i as if_block, b as set_class } from './i18n-CMo5lpzy.js';
import { a6 as comment, S as first_child, a as append, t as template_effect, W as from_html } from './index-BvBk1Iap.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root_1 = from_html(`<div></div>`);

function ScrollFade($$anchor, $$props) {

	let visible = prop($$props, 'visible', 3, false),
		position = prop($$props, 'position', 3, "sticky");

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			var div = root_1();
			let classes;

			template_effect(() => classes = set_class(div, 1, 'scroll-fade svelte-kmbucf', null, classes, { absolute: position() === "absolute" }));
			append($$anchor, div);
		};

		if_block(node, ($$render) => {
			if (visible()) $$render(consequent);
		});
	}

	append($$anchor, fragment);
}

export { ScrollFade as S };
//# sourceMappingURL=ScrollFade-BJkuhu7T.js.map
