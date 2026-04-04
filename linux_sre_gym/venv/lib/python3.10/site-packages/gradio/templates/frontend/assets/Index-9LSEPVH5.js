import { i as if_block, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, w as get, a as append, x as set, a5 as user_derived, X as sibling, T as pop, u as state, W as from_html, a6 as comment } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { P as Plot, a as Plot$1 } from './Plot-B28lJT0D.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './key-BV46G5Fv.js';
import './misc-ByVveAau.js';
import './Empty-CHa-gxv7.js';
import './snippet-Bl5YVoCg.js';
import './Maximize-B_43ylTs.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let fullscreen = state(false);

	Block($$anchor, {
		padding: false,
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get visible() {
			return gradio.shared.visible;
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
		allow_overflow: false,
		get fullscreen() {
			return get(fullscreen);
		},

		set fullscreen($$value) {
			set(fullscreen, $$value, true);
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				let $0 = user_derived(() => gradio.shared.label || gradio.i18n("plot.plot"));

				BlockLabel(node, {
					get show_label() {
						return gradio.shared.show_label;
					},

					get label() {
						return get($0);
					},

					get Icon() {
						return Plot;
					}
				});
			}

			var node_1 = sibling(node, 2);

			{
				var consequent_1 = ($$anchor) => {
					{
						let $0 = user_derived(() => gradio.props.buttons ?? []);

						IconButtonWrapper($$anchor, {
							get buttons() {
								return get($0);
							},

							on_custom_button_click: (id) => {
								gradio.dispatch("custom_button_click", { id });
							},

							children: ($$anchor, $$slotProps) => {
								var fragment_3 = comment();
								var node_2 = first_child(fragment_3);

								{
									var consequent = ($$anchor) => {
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

									if_block(node_2, ($$render) => {
										if (gradio.props.show_fullscreen_button) $$render(consequent);
									});
								}

								append($$anchor, fragment_3);
							},
							$$slots: { default: true }
						});
					}
				};

				if_block(node_1, ($$render) => {
					if (gradio.props.buttons && gradio.props.buttons.length > 0 || gradio.props.show_fullscreen_button) $$render(consequent_1);
				});
			}

			var node_3 = sibling(node_1, 2);

			Static(node_3, spread_props(
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

			var node_4 = sibling(node_3, 2);

			Plot$1(node_4, {
				get value() {
					return gradio.props.value;
				},

				get theme_mode() {
					return gradio.props.theme_mode;
				},

				get show_label() {
					return gradio.shared.show_label;
				},

				get caption() {
					return gradio.props.caption;
				},

				get bokeh_version() {
					return gradio.props.bokeh_version;
				},

				get show_actions_button() {
					return gradio.props.show_actions_button;
				},

				get _selectable() {
					return gradio.props._selectable;
				},

				get x_lim() {
					return gradio.props.x_lim;
				},

				get show_fullscreen_button() {
					return gradio.props.show_fullscreen_button;
				},
				on_change: () => gradio.dispatch("change")
			});

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Plot$1 as BasePlot, Index as default };
//# sourceMappingURL=Index-9LSEPVH5.js.map
