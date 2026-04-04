import { g as spread_props, i as if_block, r as rest_props, k as each, a as set_attribute, b as set_class, f as set_style, u as index, d as bind_this } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, S as first_child, a as append, X as sibling, T as pop, a5 as user_derived, W as from_html, V as child, a6 as comment, t as template_effect, Y as reset, a0 as set_text, Z as event } from './index-BvBk1Iap.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { I as Image } from './Image-C3F5KG1F.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './snippet-Bl5YVoCg.js';
import './clone-DaB-S2nH.js';
import './Maximize-B_43ylTs.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root_7 = from_html(`<img/>`);
var root_9 = from_html(`<button class="legend-item svelte-1oizopk"> </button>`);
var root_8 = from_html(`<div class="legend svelte-1oizopk"></div>`);
var root_4 = from_html(`<div class="image-container svelte-1oizopk"><!> <img alt="the base file that is annotated"/> <!></div> <!>`, 1);
var root_1 = from_html(`<!> <!> <div class="container svelte-1oizopk"><!></div>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let old_value = state(proxy(gradio.props.value));
	let active = state(null);
	let image_container;
	let fullscreen = state(false);
	let label = user_derived(() => gradio.shared.label || gradio.i18n("annotated_image.annotated_image"));

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	function handle_mouseover(_label) {
		set(active, _label, true);
	}

	function handle_mouseout() {
		set(active, null);
	}

	function handle_click(i, value) {
		gradio.dispatch("select", { value, index: i });
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
		padding: false,
		get height() {
			return gradio.props.height;
		},

		get width() {
			return gradio.props.width;
		},
		allow_overflow: false,
		get container() {
			return gradio.shared.container;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},

		get fullscreen() {
			return get(fullscreen);
		},

		set fullscreen($$value) {
			set(fullscreen, $$value, true);
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
				() => gradio.shared.loading_status
			));

			var node_1 = sibling(node, 2);

			BlockLabel(node_1, {
				get show_label() {
					return gradio.shared.show_label;
				},

				get Icon() {
					return Image;
				},

				get label() {
					return get(label);
				}
			});

			var div = sibling(node_1, 2);
			var node_2 = child(div);

			{
				var consequent = ($$anchor) => {
					Empty($$anchor, {
						size: 'large',
						unpadded_box: true,
						children: ($$anchor, $$slotProps) => {
							Image($$anchor);
						},
						$$slots: { default: true }
					});
				};

				var alternate = ($$anchor) => {
					var fragment_4 = root_4();
					var div_1 = first_child(fragment_4);
					var node_3 = child(div_1);

					{
						let $0 = user_derived(() => gradio.props.buttons || []);

						IconButtonWrapper(node_3, {
							get buttons() {
								return get($0);
							},

							on_custom_button_click: (id) => {
								gradio.dispatch("custom_button_click", { id });
							},

							children: ($$anchor, $$slotProps) => {
								var fragment_5 = comment();
								var node_4 = first_child(fragment_5);

								{
									var consequent_1 = ($$anchor) => {
										FullscreenButton($$anchor, {
											get fullscreen() {
												return get(fullscreen);
											},

											$$events: {
												fullscreen: ({ detail }) => {
													set(fullscreen, detail, true);
												}
											}
										});
									};

									if_block(node_4, ($$render) => {
										if ((gradio.props.buttons || []).some((btn) => typeof btn === "string" && btn === "fullscreen")) $$render(consequent_1);
									});
								}

								append($$anchor, fragment_5);
							},
							$$slots: { default: true }
						});
					}

					var img = sibling(node_3, 2);
					let classes;
					var node_5 = sibling(img, 2);

					each(node_5, 17, () => gradio.props.value ? gradio.props.value.annotations : [], index, ($$anchor, ann, i) => {
						var img_1 = root_7();
						let classes_1;

						template_effect(
							($0) => {
								set_attribute(img_1, 'alt', `segmentation mask identifying ${gradio.shared.label ?? ''} within the uploaded file`);

								classes_1 = set_class(img_1, 1, 'mask fit-height svelte-1oizopk', null, classes_1, {
									'fit-height': !get(fullscreen),
									active: get(active) == get(ann).label,
									inactive: get(active) != get(ann).label && get(active) != null
								});

								set_attribute(img_1, 'src', get(ann).image.url);
								set_style(img_1, $0);
							},
							[
								() => gradio.props.color_map && get(ann).label in gradio.props.color_map
									? null
									: `filter: hue-rotate(${Math.round(i * 360 / (gradio.props.value?.annotations.length ?? 1))}deg);`
							]
						);

						append($$anchor, img_1);
					});

					reset(div_1);
					bind_this(div_1, ($$value) => image_container = $$value, () => image_container);

					var node_6 = sibling(div_1, 2);

					{
						var consequent_2 = ($$anchor) => {
							var div_2 = root_8();

							each(div_2, 21, () => gradio.props.value.annotations, index, ($$anchor, ann, i) => {
								var button = root_9();
								var text = child(button, true);

								reset(button);

								template_effect(
									($0) => {
										set_style(button, `background-color: ${$0 ?? ''}`);
										set_text(text, get(ann).label);
									},
									[
										() => gradio.props.color_map && get(ann).label in gradio.props.color_map
											? gradio.props.color_map[get(ann).label] + '88'
											: `hsla(${Math.round(i * 360 / gradio.props.value.annotations.length)}, 100%, 50%, 0.3)`
									]
								);

								event('mouseover', button, () => handle_mouseover(get(ann).label));
								event('focus', button, () => handle_mouseover(get(ann).label));
								event('mouseout', button, () => handle_mouseout());
								event('blur', button, () => handle_mouseout());
								event('click', button, () => handle_click(i, get(ann).label));
								append($$anchor, button);
							});

							reset(div_2);
							append($$anchor, div_2);
						};

						if_block(node_6, ($$render) => {
							if (gradio.props.show_legend && gradio.props.value) $$render(consequent_2);
						});
					}

					template_effect(() => {
						classes = set_class(img, 1, 'base-image svelte-1oizopk', null, classes, { 'fit-height': gradio.props.height && !get(fullscreen) });
						set_attribute(img, 'src', gradio.props.value ? gradio.props.value.image.url : null);
					});

					append($$anchor, fragment_4);
				};

				if_block(node_2, ($$render) => {
					if (gradio.props.value == null) $$render(consequent); else $$render(alternate, false);
				});
			}

			reset(div);
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-DWC_ST0U.js.map
