import { p as prop, a as set_attribute, b as set_class, t as remove_input_defaults, g as spread_props, i as if_block, k as each, r as rest_props, u as index } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, Z as event, a as append, T as pop, V as child, X as sibling, W as from_html, x as set, w as get, M as tick, a5 as user_derived, a0 as set_text, Y as reset, u as state, v as proxy, y as user_effect, S as first_child, a7 as text, at as to_array, a8 as next } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { c as bind_group } from './input-C317uaXR.js';
export { default as BaseExample } from './Example-GBtWCYGT.js';
import './clone-DaB-S2nH.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

let id = 0;
var root = from_html(`<label><input type="radio" class="svelte-19qdtil"/> <span class="svelte-19qdtil"> </span></label>`);

function Radio($$anchor, $$props) {
	push($$props, true);

	const binding_group = [];
	let selected = prop($$props, 'selected', 15);
	let is_selected = user_derived(() => selected() === $$props.internal_value);

	async function handle_input(e) {
		set(is_selected, e.target.checked);

		if (get(is_selected)) {
			await tick();
			$$props.on_input();
		}
	}

	var label = root();
	let classes;
	var input = child(label);

	remove_input_defaults(input);

	var input_value;
	var span = sibling(input, 2);
	var text = child(span, true);

	reset(span);
	reset(label);

	template_effect(() => {
		set_attribute(label, 'data-testid', `${$$props.display_value ?? ''}-radio-label`);

		classes = set_class(label, 1, 'svelte-19qdtil', null, classes, {
			disabled: $$props.disabled,
			selected: get(is_selected),
			rtl: $$props.rtl
		});

		input.disabled = $$props.disabled;
		set_attribute(input, 'name', `radio-${++id ?? ''}`);
		set_attribute(input, 'aria-checked', get(is_selected));

		if (input_value !== (input_value = $$props.internal_value)) {
			input.value = (input.__value = $$props.internal_value) ?? '';
		}

		set_text(text, $$props.display_value);
	});

	bind_group(
		binding_group,
		[],
		input,
		() => {
			$$props.internal_value;

			return selected();
		},
		selected
	);

	event('input', input, handle_input);
	append($$anchor, label);
	pop();
}

var root_1 = from_html(`<!> <!> <!> <div class="wrap svelte-e4x47i"></div>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let disabled = user_derived(() => !gradio.shared.interactive);
	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},
		type: 'fieldset',
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get container() {
			return gradio.shared.container;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},

		get rtl() {
			return gradio.props.rtl;
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

			BlockTitle(node_2, {
				get show_label() {
					return gradio.shared.show_label;
				},

				get info() {
					return gradio.props.info;
				},

				children: ($$anchor, $$slotProps) => {
					next();

					var text$1 = text();

					template_effect(($0) => set_text(text$1, $0), [() => gradio.shared.label || gradio.i18n("radio.radio")]);
					append($$anchor, text$1);
				},
				$$slots: { default: true }
			});

			var div = sibling(node_2, 2);

			each(div, 21, () => gradio.props.choices, index, ($$anchor, $$item, i) => {
				var $$array = user_derived(() => to_array(get($$item), 2));
				let display_value = () => get($$array)[0];
				let internal_value = () => get($$array)[1];

				Radio($$anchor, {
					get display_value() {
						return display_value();
					},

					get internal_value() {
						return internal_value();
					},

					get disabled() {
						return get(disabled);
					},

					get rtl() {
						return gradio.props.rtl;
					},

					on_input: () => {
						gradio.dispatch("input");
						gradio.dispatch("select", { value: internal_value(), index: i });
					},

					get selected() {
						return gradio.props.value;
					},

					set selected($$value) {
						gradio.props.value = $$value;
					}
				});
			});

			reset(div);
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Radio as BaseRadio, Index as default };
//# sourceMappingURL=Index-CKPl_BKA.js.map
