import { r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, T as pop } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	user_effect(() => {
		gradio.props.value && gradio.dispatch("change");
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-D_ICCRmo.js.map
