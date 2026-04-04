import { g as spread_props, i as if_block, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, w as get, a as append, X as sibling, a5 as user_derived, T as pop, W as from_html } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { I as Info } from './Info-BrkJhX6_.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { C as Checkbox } from './Checkbox-BwK0lQJ8.js';
import './clone-DaB-S2nH.js';
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';
import './input-C317uaXR.js';

var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			Static(node, spread_props(
				{
					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get i18n() {
						return gradio.i18n;
					}
				},
				() => gradio.shared.loading_status,
				{
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}
			));

			var node_1 = sibling(node, 2);

			{
				var consequent = ($$anchor) => {
					IconButtonWrapper($$anchor, {
						get buttons() {
							return gradio.props.buttons;
						},

						on_custom_button_click: (id) => {
							gradio.dispatch("custom_button_click", { id });
						}
					});
				};

				if_block(node_1, ($$render) => {
					if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) $$render(consequent);
				});
			}

			var node_2 = sibling(node_1, 2);

			{
				let $0 = user_derived(() => gradio.shared.label || gradio.i18n("checkbox.checkbox"));

				Checkbox(node_2, {
					get label() {
						return get($0);
					},

					get interactive() {
						return gradio.shared.interactive;
					},

					get show_label() {
						return gradio.shared.show_label;
					},
					on_change: (val) => gradio.dispatch("change", val),
					on_input: () => gradio.dispatch("input"),
					on_select: (data) => gradio.dispatch("select", data),
					get value() {
						return gradio.props.value;
					},

					set value($$value) {
						gradio.props.value = $$value;
					}
				});
			}

			var node_3 = sibling(node_2, 2);

			{
				var consequent_1 = ($$anchor) => {
					Info($$anchor, {
						get info() {
							return gradio.props.info;
						}
					});
				};

				if_block(node_3, ($$render) => {
					if (gradio.props.info) $$render(consequent_1);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Checkbox as BaseCheckbox, Index as default };
//# sourceMappingURL=Index-BHwo3edq.js.map
