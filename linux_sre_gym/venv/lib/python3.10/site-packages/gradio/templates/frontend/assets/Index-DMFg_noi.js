import { p as prop, i as if_block, b as set_class, f as set_style, k as each, a as set_attribute, F as set_value, u as index, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, a as append, T as pop, X as sibling, W as from_html, V as child, aj as delegate, Y as reset, t as template_effect, a0 as set_text, a6 as comment, S as first_child, w as get, u as state, v as proxy, y as user_effect, x as set, a5 as user_derived } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { L as LineChart } from './LineChart-BOORCIAk.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root_1$1 = from_html(`<h2 data-testid="label-output-value"> </h2>`);
var root_3 = from_html(`<button><div class="inner-wrap svelte-g2cwl3"><meter aria-valuemin="0" aria-valuemax="100" class="bar svelte-g2cwl3" min="0" max="1"></meter> <dl class="label svelte-g2cwl3"><dt class="text svelte-g2cwl3"> </dt> <div class="line svelte-g2cwl3"></div> <dd class="confidence svelte-g2cwl3"> </dd></dl></div></button>`);
var root = from_html(`<div class="container svelte-g2cwl3"><!> <!></div>`);

function Label($$anchor, $$props) {
	push($$props, true);

	let color = prop($$props, 'color', 3, undefined),
		selectable = prop($$props, 'selectable', 3, false),
		show_heading = prop($$props, 'show_heading', 3, true);

	function get_aria_referenceable_id(elem_id) {
		// `aria-labelledby` interprets the value as a space-separated id reference list,
		// so each single id should not contain any spaces.
		// Ref: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Attributes/aria-labelledby#benefits_and_drawbacks
		return elem_id.replace(/\s/g, "-");
	}

	var div = root();
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var h2 = root_1$1();
			let classes;
			let styles;
			var text = child(h2, true);

			reset(h2);

			template_effect(() => {
				classes = set_class(h2, 1, 'output-class svelte-g2cwl3', null, classes, { 'no-confidence': !("confidences" in $$props.value) });
				styles = set_style(h2, '', styles, { 'background-color': color() || "transparent" });
				set_text(text, $$props.value.label);
			});

			append($$anchor, h2);
		};

		if_block(node, ($$render) => {
			if (show_heading() || !$$props.value.confidences) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent_1 = ($$anchor) => {
			var fragment = comment();
			var node_2 = first_child(fragment);

			each(node_2, 17, () => $$props.value.confidences, index, ($$anchor, confidence_set, i) => {
				var button = root_3();
				let classes_1;

				button.__click = () => {
					$$props.onselect?.({ index: i, value: get(confidence_set).label });
				};

				var div_1 = child(button);
				var meter = child(div_1);
				var dl = sibling(meter, 2);
				var dt = child(dl);
				var text_1 = child(dt, true);

				reset(dt);

				var dd = sibling(dt, 4);
				var text_2 = child(dd);

				reset(dd);
				reset(dl);
				reset(div_1);
				reset(button);

				template_effect(
					($0, $1, $2, $3) => {
						classes_1 = set_class(button, 1, 'confidence-set group svelte-g2cwl3', null, classes_1, { selectable: selectable() });
						set_attribute(button, 'data-testid', `${get(confidence_set).label}-confidence-set`);
						set_attribute(meter, 'aria-labelledby', $0);
						set_attribute(meter, 'aria-label', get(confidence_set).label);
						set_attribute(meter, 'aria-valuenow', $1);
						set_value(meter, get(confidence_set).confidence);

						set_style(meter, `width: ${get(confidence_set).confidence * 100}%; background: var(--stat-background-fill);
						`);

						set_attribute(dt, 'id', $2);
						set_text(text_1, get(confidence_set).label);
						set_text(text_2, `${$3 ?? ''}%`);
					},
					[
						() => get_aria_referenceable_id(`meter-text-${get(confidence_set).label}`),
						() => Math.round(get(confidence_set).confidence * 100),
						() => get_aria_referenceable_id(`meter-text-${get(confidence_set).label}`),
						() => Math.round(get(confidence_set).confidence * 100)
					]
				);

				append($$anchor, button);
			});

			append($$anchor, fragment);
		};

		if_block(node_1, ($$render) => {
			if (typeof $$props.value === "object" && $$props.value.confidences) $$render(consequent_1);
		});
	}

	reset(div);
	append($$anchor, div);
	pop();
}

delegate(['click']);

var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let old_value = state(proxy(gradio.props.value));
	let _label = user_derived(() => gradio.props.value.label);

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	Block($$anchor, {
		test_id: 'label',
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
		padding: false,
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
				var consequent_1 = ($$anchor) => {
					{
						let $0 = user_derived(() => gradio.shared.label || gradio.i18n("label.label"));
						let $1 = user_derived(() => gradio.shared.container === false);
						let $2 = user_derived(() => gradio.props.show_heading === true);

						BlockLabel($$anchor, {
							get Icon() {
								return LineChart;
							},

							get label() {
								return get($0);
							},

							get disable() {
								return get($1);
							},

							get float() {
								return get($2);
							}
						});
					}
				};

				if_block(node_2, ($$render) => {
					if (gradio.shared.show_label) $$render(consequent_1);
				});
			}

			var node_3 = sibling(node_2, 2);

			{
				var consequent_2 = ($$anchor) => {
					Label($$anchor, {
						onselect: (detail) => gradio.dispatch("select", detail),
						get selectable() {
							return gradio.props._selectable;
						},

						get value() {
							return gradio.props.value;
						},

						get color() {
							return gradio.props.color;
						},

						get show_heading() {
							return gradio.props.show_heading;
						}
					});
				};

				var alternate = ($$anchor) => {
					Empty($$anchor, {
						unpadded_box: true,
						children: ($$anchor, $$slotProps) => {
							LineChart($$anchor);
						},
						$$slots: { default: true }
					});
				};

				if_block(node_3, ($$render) => {
					if (get(_label) !== undefined && get(_label) !== null) $$render(consequent_2); else $$render(alternate, false);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Label as BaseLabel, Index as default };
//# sourceMappingURL=Index-DMFg_noi.js.map
