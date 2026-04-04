import { r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, ae as onDestroy, T as pop } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let interval = undefined;

	user_effect(() => {
		if (interval) clearInterval(interval);

		if (gradio.props.active) {
			interval = setInterval(
				() => {
					if (document.visibilityState === "visible") {
						gradio.dispatch("tick");
					}
				},
				gradio.props.value * 1000
			);
		}
	});

	onDestroy(() => {
		if (interval) clearInterval(interval);
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-B51SJCw-.js.map
