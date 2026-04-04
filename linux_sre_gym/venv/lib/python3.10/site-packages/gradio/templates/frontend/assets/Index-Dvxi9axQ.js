import { s as slot, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, a6 as comment, S as first_child, a as append, T as pop } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as BaseForm } from './BaseForm-0cuxY3Cw.js';
import './clone-DaB-S2nH.js';

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	BaseForm($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = comment();
			var node = first_child(fragment_1);

			slot(node, $$props, 'default', {});
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { BaseForm, Index as default };
//# sourceMappingURL=Index-Dvxi9axQ.js.map
