import { r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, a7 as text, t as template_effect, a as append, w as get, T as pop, a5 as user_derived, a0 as set_text, a8 as next } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Button } from './Button-CH2w4_A4.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Image-CDw5qb5O.js';
import './misc-ByVveAau.js';
/* empty css                                             */
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
/* empty css                                                    */

function Index($$anchor, $$props) {
	push($$props, true);

	let _props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(_props);

	function handle_click() {
		gradio.dispatch("click");
	}

	{
		let $0 = user_derived(() => !gradio.shared.interactive);

		Button($$anchor, {
			get value() {
				return gradio.props.value;
			},

			get variant() {
				return gradio.props.variant;
			},

			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return gradio.shared.elem_classes;
			},

			get size() {
				return gradio.props.size;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get link() {
				return gradio.props.link;
			},

			get icon() {
				return gradio.props.icon;
			},

			get min_width() {
				return gradio.shared.min_width;
			},

			get visible() {
				return gradio.shared.visible;
			},

			get disabled() {
				return get($0);
			},

			get link_target() {
				return gradio.props.link_target;
			},
			onclick: handle_click,
			children: ($$anchor, $$slotProps) => {
				next();

				var text$1 = text();

				template_effect(() => set_text(text$1, gradio.props.value ?? ""));
				append($$anchor, text$1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { Button as BaseButton, Index as default };
//# sourceMappingURL=Index-DIRpB5DY.js.map
