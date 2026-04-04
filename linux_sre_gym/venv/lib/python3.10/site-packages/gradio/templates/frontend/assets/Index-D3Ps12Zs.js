import { g as spread_props, d as bind_this, i as if_block, r as rest_props, a as set_attribute, t as remove_input_defaults } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, Z as event, S as first_child, a7 as text, t as template_effect, a as append, V as child, X as sibling, T as pop, a5 as user_derived, M as tick, aw as $window, W as from_html, a0 as set_text, a8 as next, Y as reset } from './index-BvBk1Iap.js';
import { b as bind_value } from './input-C317uaXR.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

let _id = 0;
var root_3 = from_html(`<button class="reset-button svelte-8epfm4" aria-label="Reset to default value" data-testid="reset-button">↺</button>`);
var root_1 = from_html(`<!> <div class="wrap svelte-8epfm4"><div class="head svelte-8epfm4"><label class="svelte-8epfm4"><!></label> <div class="tab-like-container svelte-8epfm4"><input data-testid="number-input" type="number" class="svelte-8epfm4"/> <!></div></div> <div class="slider_input_container svelte-8epfm4"><span class="min_value svelte-8epfm4"> </span> <input type="range" name="cowbell" class="svelte-8epfm4"/> <span class="max_value svelte-8epfm4"> </span></div></div>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let gradio = new Gradio(props);
	const INITIAL_VALUE = gradio.props.value;
	let old_value = state(proxy(gradio.props.value));
	let range_input;
	let number_input;
	const id = `range_id_${_id++}`;
	let minimum_value = user_derived(() => gradio.props.minimum ?? 0);

	let percentage = user_derived(() => {
		const min = gradio.props.minimum;
		const max = gradio.props.maximum;
		const val = gradio.props.value;

		if (val > max) {
			return 100;
		} else if (val < min) {
			return 0;
		}

		return (val - min) / (max - min) * 100;
	});

	user_effect(() => {
		if (gradio.props.value == null) return;

		range_input.style.setProperty("--range_progress", `${get(percentage)}%`);
		range_input.value = gradio.props.value.toString();
	});

	async function handle_change() {
		await tick();
		gradio.dispatch("change");
	}

	async function handle_release(e) {
		await tick();
		gradio.dispatch("release", gradio.props.value);
	}

	function clamp() {
		gradio.dispatch("release", gradio.props.value);
		gradio.props.value = Math.min(Math.max(gradio.props.value, gradio.props.minimum), gradio.props.maximum);
	}

	let disabled = user_derived(() => !gradio.shared.interactive);

	// When the value changes, dispatch the change event via handle_change()
	// See the docs for an explanation: https://svelte.dev/docs/svelte-components#script-3-$-marks-a-statement-as-reactive
	user_effect(() => {
		if (gradio.props.value != get(old_value)) {
			set(old_value, gradio.props.value, true);
			handle_change();
		}
	});

	function handle_resize() {
	}

	function reset_value() {
		gradio.props.value = INITIAL_VALUE;
		gradio.dispatch("change");
		gradio.dispatch("release", gradio.props.value);
	}

	async function handle_input() {
		await tick();
		gradio.dispatch("input");
	}

	event('resize', $window, handle_resize);

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

		get container() {
			return gradio.shared.container;
		},

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
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}
			));

			var div = sibling(node, 2);
			var div_1 = child(div);
			var label = child(div_1);
			var node_1 = child(label);

			BlockTitle(node_1, {
				get show_label() {
					return gradio.shared.show_label;
				},

				get info() {
					return gradio.props.info;
				},

				children: ($$anchor, $$slotProps) => {
					next();

					var text$1 = text();

					template_effect(() => set_text(text$1, gradio.shared.label || "Slider"));
					append($$anchor, text$1);
				},
				$$slots: { default: true }
			});

			reset(label);

			var div_2 = sibling(label, 2);
			var input = child(div_2);

			remove_input_defaults(input);
			bind_this(input, ($$value) => number_input = $$value, () => number_input);

			var node_2 = sibling(input, 2);

			{
				var consequent = ($$anchor) => {
					var button = root_3();

					template_effect(() => button.disabled = get(disabled));
					event('click', button, reset_value);
					append($$anchor, button);
				};

				if_block(node_2, ($$render) => {
					if (gradio.props.buttons?.includes("reset") ?? true) $$render(consequent);
				});
			}

			reset(div_2);
			reset(div_1);

			var div_3 = sibling(div_1, 2);
			var span = child(div_3);
			var text_1 = child(span, true);

			reset(span);

			var input_1 = sibling(span, 2);

			remove_input_defaults(input_1);
			bind_this(input_1, ($$value) => range_input = $$value, () => range_input);

			var span_1 = sibling(input_1, 2);
			var text_2 = child(span_1, true);

			reset(span_1);
			reset(div_3);
			reset(div);

			template_effect(() => {
				set_attribute(label, 'for', id);
				set_attribute(input, 'aria-label', `number input for ${gradio.shared.label}`);
				set_attribute(input, 'min', gradio.props.minimum);
				set_attribute(input, 'max', gradio.props.maximum);
				set_attribute(input, 'step', gradio.props.step);
				input.disabled = get(disabled);
				set_text(text_1, get(minimum_value));
				set_attribute(input_1, 'id', id);
				set_attribute(input_1, 'min', gradio.props.minimum);
				set_attribute(input_1, 'max', gradio.props.maximum);
				set_attribute(input_1, 'step', gradio.props.step);
				input_1.disabled = get(disabled);
				set_attribute(input_1, 'aria-label', `range slider for ${gradio.shared.label}`);
				set_text(text_2, gradio.props.maximum);
			});

			bind_value(input, () => gradio.props.value, ($$value) => gradio.props.value = $$value);
			event('input', input, handle_input);
			event('blur', input, clamp);
			event('pointerup', input, handle_release);
			bind_value(input_1, () => gradio.props.value, ($$value) => gradio.props.value = $$value);
			event('input', input_1, handle_input);
			event('pointerup', input_1, handle_release);
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-D3Ps12Zs.js.map
