import { n as navbar_config, h as clsx, a as set_attribute, b as set_class, r as rest_props, j as get$1 } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, y as user_effect, w as get, t as template_effect, a as append, T as pop, a5 as user_derived, W as from_html } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';

var root = from_html(`<div style="display: none;"></div>`);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	let navbar_props = user_derived(() => {
		return {
			visible: gradio.shared.visible,
			main_page_name: gradio.props.main_page_name ?? "Home",
			value: gradio.props.value
		};
	});

	onMount(() => {
		const current_store = get$1(navbar_config);

		if (!current_store) {
			navbar_config.set(get(navbar_props));
		}
	});

	user_effect(() => {
		navbar_config.set(get(navbar_props));
	});

	var div = root();

	template_effect(
		($0) => {
			set_attribute(div, 'id', gradio.shared.elem_id);
			set_class(div, 1, $0);
		},
		[() => clsx(gradio.shared.elem_classes.join(" "))]
	);

	append($$anchor, div);
	pop();
}

export { Index as default };
//# sourceMappingURL=Index-My9wgKRU.js.map
