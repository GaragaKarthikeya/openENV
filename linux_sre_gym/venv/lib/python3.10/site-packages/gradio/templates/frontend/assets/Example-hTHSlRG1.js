import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { t as template_effect, a as append, W as from_html, a0 as set_text, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div> </div>`);

function Example($$anchor, $$props) {

	let selected = prop($$props, 'selected', 3, false);
	var div = root();
	let classes;
	var text = child(div, true);

	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'svelte-9pg6fh', null, classes, {
			table: $$props.type === "table",
			gallery: $$props.type === "gallery",
			selected: selected()
		});

		set_text(text, $$props.value ? $$props.value : "");
	});

	append($$anchor, div);
}

export { Example as default };
//# sourceMappingURL=Example-hTHSlRG1.js.map
