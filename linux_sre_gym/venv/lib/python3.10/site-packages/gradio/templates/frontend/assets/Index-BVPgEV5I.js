import { g as spread_props, i as if_block, r as rest_props, b as set_class, a as set_attribute, t as remove_input_defaults } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, S as first_child, t as template_effect, a as append, Z as event, X as sibling, T as pop, V as child, W as from_html, M as tick, a0 as set_text, a5 as user_derived, a8 as next, Y as reset } from './index-BvBk1Iap.js';
import { b as bind_value } from './input-C317uaXR.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

var root_4 = from_html(`<div class="validation-error svelte-16ty2ow"> </div>`);
var root_3 = from_html(` <!>`, 1);
var root_1 = from_html(`<!> <label><!> <!> <input type="number"/></label>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	gradio.props.value ??= 0;

	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			//@ts-ignore
			set(old_value, gradio.props.value, true);

			gradio.dispatch("change");
		}
	});

	async function handle_keypress(e) {
		await tick();

		if (e.key === "Enter") {
			e.preventDefault();
			gradio.dispatch("submit");
		}
	}

	const disabled = user_derived(() => !gradio.shared.interactive);

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

		get padding() {
			return gradio.shared.container;
		},
		allow_overflow: false,
		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
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
					show_validation_error: false,
					on_clear_status: () => {
						gradio.dispatch("clear_status", gradio.shared.loading_status);
					}
				}
			));

			var label = sibling(node, 2);
			let classes;
			var node_1 = child(label);

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

			BlockTitle(node_2, {
				get show_label() {
					return gradio.shared.show_label;
				},

				get info() {
					return gradio.props.info;
				},

				children: ($$anchor, $$slotProps) => {
					next();

					var fragment_3 = root_3();
					var text = first_child(fragment_3);
					var node_3 = sibling(text);

					{
						var consequent_1 = ($$anchor) => {
							var div = root_4();
							var text_1 = child(div, true);

							reset(div);
							template_effect(() => set_text(text_1, gradio.shared.loading_status?.validation_error));
							append($$anchor, div);
						};

						if_block(node_3, ($$render) => {
							if (gradio.shared.loading_status?.validation_error) $$render(consequent_1);
						});
					}

					template_effect(() => set_text(text, `${(gradio.shared.label || "Number") ?? ''} `));
					append($$anchor, fragment_3);
				},
				$$slots: { default: true }
			});

			var input = sibling(node_2, 2);

			remove_input_defaults(input);

			let classes_1;

			reset(label);

			template_effect(() => {
				classes = set_class(label, 1, 'block svelte-16ty2ow', null, classes, { container: gradio.shared.container });
				set_attribute(input, 'aria-label', gradio.shared.label || "Number");
				set_attribute(input, 'min', gradio.props.minimum);
				set_attribute(input, 'max', gradio.props.maximum);
				set_attribute(input, 'step', gradio.props.step);
				set_attribute(input, 'placeholder', gradio.props.placeholder);
				input.disabled = get(disabled);

				classes_1 = set_class(input, 1, 'svelte-16ty2ow', null, classes_1, {
					'validation-error': gradio.shared.loading_status?.validation_error
				});
			});

			bind_value(input, () => gradio.props.value, ($$value) => gradio.props.value = $$value);
			event('keypress', input, handle_keypress);
			event('input', input, () => gradio.dispatch("input"));
			event('blur', input, () => gradio.dispatch("blur"));
			event('focus', input, () => gradio.dispatch("focus"));
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-BVPgEV5I.js.map
