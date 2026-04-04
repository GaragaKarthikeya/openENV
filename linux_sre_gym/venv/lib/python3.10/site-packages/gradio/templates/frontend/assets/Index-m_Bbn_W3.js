import { i as if_block, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, y as user_effect, S as first_child, w as get, a as append, X as sibling, a5 as user_derived, T as pop, x as set, M as tick, W as from_html } from './index-BvBk1Iap.js';
import { s as snapshot } from './clone-DaB-S2nH.js';
import { T as Textbox } from './Textbox-v-cNeNHQ.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
export { default as BaseExample } from './Example-a64hbcxE.js';
import './actions-Cit_UA5S.js';
import './input-C317uaXR.js';
import './BlockTitle-7XQ-8ENg.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './Check-BlnBgjLT.js';
import './Copy-DFFray7x.js';
import './Send-Ee555fP_.js';
import './Square-B6INQV79.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';
import './size-Cu4PabCL.js';
/* empty css                                               */

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let _props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(_props);
	let label = user_derived(() => gradio.shared.label || "Textbox");

	// Need to set the value to "" otherwise a change event gets
	// dispatched when the child sets it to ""
	gradio.props.value = gradio.props.value ?? "";

	let old_value = state(proxy(gradio.props.value));

	async function dispatch_change() {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			await tick();
			gradio.dispatch("change", snapshot(gradio.props.value));
		}
	}

	async function handle_input(value) {
		if (!gradio.shared || !gradio.props) return;

		gradio.props.validation_error = null;
		gradio.props.value = value;
		await tick();
		gradio.dispatch("input");
	}

	user_effect(() => {
		dispatch_change();
	});

	function handle_change(value) {
		if (!gradio.shared || !gradio.props) return;

		gradio.props.validation_error = null;
		gradio.props.value = value;
	}

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

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		allow_overflow: false,
		get padding() {
			return gradio.shared.container;
		},

		get rtl() {
			return gradio.props.rtl;
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					Static($$anchor, spread_props(
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
							on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
						}
					));
				};

				if_block(node, ($$render) => {
					if (gradio.shared.loading_status) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				let $0 = user_derived(() => gradio.shared?.loading_status?.validation_error || gradio.shared?.validation_error);
				let $1 = user_derived(() => !gradio.shared.interactive);

				Textbox(node_1, {
					get label() {
						return get(label);
					},

					get info() {
						return gradio.props.info;
					},

					get show_label() {
						return gradio.shared.show_label;
					},

					get lines() {
						return gradio.props.lines;
					},

					get type() {
						return gradio.props.type;
					},

					get rtl() {
						return gradio.props.rtl;
					},

					get text_align() {
						return gradio.props.text_align;
					},

					get max_lines() {
						return gradio.props.max_lines;
					},

					get placeholder() {
						return gradio.props.placeholder;
					},

					get submit_btn() {
						return gradio.props.submit_btn;
					},

					get stop_btn() {
						return gradio.props.stop_btn;
					},

					get buttons() {
						return gradio.props.buttons;
					},

					get autofocus() {
						return gradio.props.autofocus;
					},

					get container() {
						return gradio.shared.container;
					},

					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get max_length() {
						return gradio.props.max_length;
					},

					get html_attributes() {
						return gradio.props.html_attributes;
					},

					get validation_error() {
						return get($0);
					},
					onchange: handle_change,
					oninput: handle_input,
					onsubmit: () => {
						gradio.shared.validation_error = null;
						gradio.dispatch("submit");
					},
					onblur: () => gradio.dispatch("blur"),
					onselect: (data) => gradio.dispatch("select", data),
					onfocus: () => gradio.dispatch("focus"),
					onstop: () => gradio.dispatch("stop"),
					oncopy: (data) => gradio.dispatch("copy", data),
					oncustombuttonclick: (id) => {
						gradio.dispatch("custom_button_click", { id });
					},

					get disabled() {
						return get($1);
					},

					get value() {
						return gradio.props.value;
					},

					set value($$value) {
						gradio.props.value = $$value;
					}
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Textbox as BaseTextbox, Index as default };
//# sourceMappingURL=Index-m_Bbn_W3.js.map
