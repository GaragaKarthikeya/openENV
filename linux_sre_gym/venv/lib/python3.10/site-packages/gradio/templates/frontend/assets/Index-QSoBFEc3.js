import { f as set_style, s as slot, a as set_attribute, b as set_class, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, V as child, W as from_html, Y as reset } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';

var root = from_html(`<div><div><div class="styler svelte-1p9262q"><!></div></div></div>`);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	var div = root();
	let classes;
	var div_1 = child(div);
	let classes_1;
	var div_2 = child(div_1);

	set_style(div_2, '', {}, {
		'--block-radius': '0px',
		'--block-border-width': '0px',
		'--layout-gap': '1px',
		'--form-gap-width': '1px',
		'--button-border-width': '0px',
		'--button-large-radius': '0px',
		'--button-small-radius': '0px'
	});

	var node = child(div_2);

	slot(node, $$props, 'default', {});
	reset(div_2);
	reset(div_1);
	reset(div);

	template_effect(
		($0, $1) => {
			set_attribute(div, 'id', gradio.shared.elem_id);
			classes = set_class(div, 1, `gr-group ${$0 ?? ''}`, 'svelte-1p9262q', classes, { hide: gradio.shared.visible === "hidden" });
			set_attribute(div_1, 'id', gradio.shared.elem_id);
			classes_1 = set_class(div_1, 1, `gr-group ${$1 ?? ''}`, 'svelte-1p9262q', classes_1, { hide: gradio.shared.visible === "hidden" });
		},
		[
			() => gradio.shared.elem_classes?.join(' ') || '',
			() => gradio.shared.elem_classes?.join(' ') || ''
		]
	);

	append($$anchor, div);
	pop();
}

export { Index as default };
//# sourceMappingURL=Index-QSoBFEc3.js.map
