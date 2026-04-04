import { g as spread_props, r as rest_props, s as slot } from './i18n-CMo5lpzy.js';
import { R as push, T as pop, a6 as comment, S as first_child, a as append } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-C97CbKPo.js';
import './clone-DaB-S2nH.js';
import './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './html-CdUEtR5E.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './index-C7e-J7CF.js';
import './Clear-Dgs5UhTO.js';

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	BaseColumn($$anchor, spread_props(() => gradio.shared, {
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = comment();
			var node = first_child(fragment_1);

			slot(node, $$props, 'default', {});
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	}));

	pop();
}

export { BaseColumn, Index as default };
//# sourceMappingURL=Index-BsVOYSj7.js.map
