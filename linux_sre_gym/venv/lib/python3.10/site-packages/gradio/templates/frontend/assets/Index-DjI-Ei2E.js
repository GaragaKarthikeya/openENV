import { p as prop, w as writable, x as store_set, e as init, i as if_block, s as slot, b as set_class, a as set_attribute, f as set_style, q as store_get, m as setup_stores, k as each, d as bind_this, u as index, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, a3 as mutable_source, w as get, a4 as createEventDispatcher, aa as onMount, x as set, ax as setContext, as as mutate, a1 as legacy_pre_effect, G as deep_read_state, a2 as legacy_pre_effect_reset, Z as event, aw as $window, V as child, X as sibling, Y as reset, t as template_effect, z as untrack, a as append, T as pop, M as tick, U as flushSync, S as first_child, a6 as comment, W as from_html, a0 as set_text, a7 as text, f as from_svg, y as user_effect } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { T as TABS, a as Tabs } from './Walkthrough.svelte_svelte_type_style_lang-Bq8jBNe2.js';
import './clone-DaB-S2nH.js';
import './event-modifiers-DanhKw3_.js';

var root_2 = from_html(`<p class="step-title"><strong> </strong> </p>`);
var root_5 = from_svg(`<svg width="12" height="10" viewBox="0 0 12 10" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 5L4.5 8.5L11 1.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`);
var root_7 = from_html(`<span> </span>`);
var root_8 = from_html(`<div></div>`);
var root_4 = from_html(`<div class="step-item svelte-1xpzbgb"><button role="tab"><span class="step-number svelte-1xpzbgb"><!></span> <!></button></div> <!>`, 1);
var root_1 = from_html(`<!> <div class="stepper-wrapper svelte-1xpzbgb"><div class="stepper-container svelte-1xpzbgb" role="tablist"></div></div>`, 1);
var root = from_html(`<div><!> <!></div>`);

function Walkthrough($$anchor, $$props) {
	push($$props, false);

	const $selected_tab = () => store_get(selected_tab, '$selected_tab', $$stores);
	const $selected_tab_index = () => store_get(selected_tab_index, '$selected_tab_index', $$stores);
	const [$$stores, $$cleanup] = setup_stores();
	const has_tabs = mutable_source();
	const tab_scale = mutable_source();
	let visible = prop($$props, 'visible', 12, true);
	let elem_id = prop($$props, 'elem_id', 12, "");
	let elem_classes = prop($$props, 'elem_classes', 28, () => []);
	let selected = prop($$props, 'selected', 12);
	let initial_tabs = prop($$props, 'initial_tabs', 12);
	let tabs = mutable_source([...initial_tabs()]);
	let stepper_container = mutable_source();
	let show_labels_for_all = mutable_source(true);
	let measurement_container = mutable_source();
	let step_buttons = mutable_source([]);
	let step_labels = mutable_source([]);
	let label_height = mutable_source(0);
	let compact = mutable_source(false);
	let recompute_overflow = true;
	const selected_tab = writable(selected() || get(tabs)[0]?.id || false);
	const selected_tab_index = writable(get(tabs).findIndex((t) => t?.id === selected()) || 0);
	const dispatch = createEventDispatcher();

	async function check_overflow() {
		if (!get(stepper_container) || !get(measurement_container) || !recompute_overflow) return;

		recompute_overflow = false;
		await tick();

		// First, show all labels to measure
		set(show_labels_for_all, true);

		await tick();

		const SEP_WIDTH = 50;
		const button_width = get(step_buttons)[0].getBoundingClientRect().width * get(step_buttons).length + SEP_WIDTH * (get(step_buttons).length - 1);
		const containerWidth = get(stepper_container).getBoundingClientRect().width;
		const does_it_fit = button_width < containerWidth;

		if (!does_it_fit) {
			set(show_labels_for_all, false);
			set(compact, true);

			return;
		}

		let max_height = 0;
		let is_overlapping = false;
		let last_right = 0;

		for (const label of get(step_labels)) {
			const { height, width, left, right } = label.getBoundingClientRect();

			if (height > max_height) {
				max_height = height;
			}

			if (last_right && left - 10 < last_right && !is_overlapping) {
				is_overlapping = true;
			}

			last_right = right;
		}

		set(label_height, max_height);

		if (is_overlapping) {
			set(show_labels_for_all, false);
		}
	}

	let last_width = 0;

	onMount(() => {
		check_overflow();

		const observer = new ResizeObserver((entries) => {
			if (entries[0].contentRect.width === last_width) return;

			last_width = entries[0].contentRect.width;
			set(compact, false);
			recompute_overflow = true;
			check_overflow();
		});

		if (get(stepper_container)) {
			observer.observe(get(stepper_container));
		}

		return () => {
			observer.disconnect();
		};
	});

	setContext(TABS, {
		register_tab: (tab, order) => {
			mutate(tabs, get(tabs)[order] = tab);

			if ($selected_tab() === false && tab.visible && tab.interactive) {
				store_set(selected_tab, tab.id);
				store_set(selected_tab_index, order);
			}

			return order;
		},

		unregister_tab: (tab, order) => {
			if ($selected_tab() === tab.id) {
				store_set(selected_tab, get(tabs)[0]?.id || false);
			}

			mutate(tabs, get(tabs)[order] = null);
		},
		selected_tab,
		selected_tab_index
	});

	function change_tab(id, index) {
		const tab_to_activate = get(tabs).find((t) => t?.id === id);

		if (id !== undefined && tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible && $selected_tab() !== tab_to_activate.id) {
			selected(id);
			store_set(selected_tab, id);
			store_set(selected_tab_index, get(tabs).findIndex((t) => t?.id === id));
			dispatch("change");
		}
	}

	legacy_pre_effect(() => (get(tabs)), () => {
		set(has_tabs, get(tabs).length > 0);
	});

	legacy_pre_effect(() => (get(tabs), deep_read_state(selected())), () => {
		(
			get(tabs),
			selected() !== null && change_tab(selected(), get(tabs).findIndex((t) => t?.id === selected()))
		);
	});

	legacy_pre_effect(() => (get(tabs)), () => {
		(get(tabs), check_overflow());
	});

	legacy_pre_effect(() => ($selected_tab_index()), () => {
		($selected_tab_index(), check_overflow());
	});

	legacy_pre_effect(() => (get(tabs), $selected_tab_index()), () => {
		set(tab_scale, get(tabs)[$selected_tab_index() >= 0 ? $selected_tab_index() : 0]?.scale);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get visible() {
			return visible();
		},

		set visible($$value) {
			visible($$value);
			flushSync();
		},

		get elem_id() {
			return elem_id();
		},

		set elem_id($$value) {
			elem_id($$value);
			flushSync();
		},

		get elem_classes() {
			return elem_classes();
		},

		set elem_classes($$value) {
			elem_classes($$value);
			flushSync();
		},

		get selected() {
			return selected();
		},

		set selected($$value) {
			selected($$value);
			flushSync();
		},

		get initial_tabs() {
			return initial_tabs();
		},

		set initial_tabs($$value) {
			initial_tabs($$value);
			flushSync();
		}
	};

	init();

	var div = root();

	event('resize', $window, check_overflow);

	let classes;
	let styles;
	var node = child(div);

	{
		var consequent_5 = ($$anchor) => {
			var fragment = root_1();
			var node_1 = first_child(fragment);

			{
				var consequent = ($$anchor) => {
					var p = root_2();
					var strong = child(p);
					var text = child(strong);

					reset(strong);

					var text_1 = sibling(strong);

					reset(p);

					template_effect(() => {
						set_text(text, `Step ${($selected_tab_index() || 0) + 1}/${(get(tabs), untrack(() => get(tabs).length)) ?? ''}:`);

						set_text(text_1, ` ${(
							get(tabs),
							$selected_tab_index(),
							untrack(() => get(tabs)[$selected_tab_index()]?.label || "Walkthrough")
						) ?? ''}`);
					});

					append($$anchor, p);
				};

				if_block(node_1, ($$render) => {
					if (get(compact)) $$render(consequent);
				});
			}

			var div_1 = sibling(node_1, 2);
			let styles_1;
			var div_2 = child(div_1);

			each(div_2, 5, () => get(tabs), index, ($$anchor, t, i) => {
				var fragment_1 = comment();
				var node_2 = first_child(fragment_1);

				{
					var consequent_4 = ($$anchor) => {
						var fragment_2 = root_4();
						var div_3 = first_child(fragment_2);
						var button = child(div_3);
						let classes_1;
						var span = child(button);
						var node_3 = child(span);

						{
							var consequent_1 = ($$anchor) => {
								var svg = root_5();

								append($$anchor, svg);
							};

							var alternate = ($$anchor) => {
								var text_2 = text();

								text_2.nodeValue = i + 1;
								append($$anchor, text_2);
							};

							if_block(node_3, ($$render) => {
								if ((
									get(t),
									$selected_tab(),
									untrack(() => get(t).id < $selected_tab())
								)) $$render(consequent_1); else $$render(alternate, false);
							});
						}

						reset(span);

						var node_4 = sibling(span, 2);

						{
							var consequent_2 = ($$anchor) => {
								var span_1 = root_7();
								let classes_2;
								var text_3 = child(span_1, true);

								reset(span_1);
								bind_this(span_1, ($$value, i) => mutate(step_labels, get(step_labels)[i] = $$value), (i) => get(step_labels)?.[i], () => [i]);

								template_effect(() => {
									classes_2 = set_class(span_1, 1, 'step-label svelte-1xpzbgb', null, classes_2, {
										visible: get(show_labels_for_all) || i === $selected_tab_index()
									});

									set_text(text_3, (
										get(t),
										untrack(() => get(t)?.label !== undefined ? get(t)?.label : "Step " + (i + 1))
									));
								});

								append($$anchor, span_1);
							};

							if_block(node_4, ($$render) => {
								if (!get(compact)) $$render(consequent_2);
							});
						}

						reset(button);
						bind_this(button, ($$value, i) => mutate(step_buttons, get(step_buttons)[i] = $$value), (i) => get(step_buttons)?.[i], () => [i]);
						reset(div_3);

						var node_5 = sibling(div_3, 2);

						{
							var consequent_3 = ($$anchor) => {
								var div_4 = root_8();
								let classes_3;

								template_effect(() => classes_3 = set_class(div_4, 1, 'step-connector svelte-1xpzbgb', null, classes_3, { completed: i < $selected_tab_index() }));
								append($$anchor, div_4);
							};

							if_block(node_5, ($$render) => {
								if ((
									get(tabs),
									get(compact),
									untrack(() => i < get(tabs).length - 1 && !get(compact))
								)) $$render(consequent_3);
							});
						}

						template_effect(() => {
							classes_1 = set_class(button, 1, 'step-button svelte-1xpzbgb', null, classes_1, {
								active: get(t).id === $selected_tab(),
								completed: get(t).id < $selected_tab(),
								pending: get(t).id > $selected_tab()
							});

							set_attribute(button, 'aria-selected', (
								get(t),
								$selected_tab(),
								untrack(() => get(t).id === $selected_tab())
							));

							set_attribute(button, 'aria-controls', (get(t), untrack(() => get(t).elem_id)));

							button.disabled = (
								get(t),
								$selected_tab_index(),
								untrack(() => !get(t).interactive || i > $selected_tab_index())
							);

							set_attribute(button, 'aria-disabled', (
								get(t),
								$selected_tab_index(),
								untrack(() => !get(t).interactive || i > $selected_tab_index())
							));

							set_attribute(button, 'id', (
								get(t),
								untrack(() => get(t).elem_id ? get(t).elem_id + "-button" : null)
							));

							set_attribute(button, 'data-tab-id', (get(t), untrack(() => get(t).id)));
						});

						event('click', button, () => {
							if (i <= $selected_tab_index() && get(t).id !== $selected_tab()) {
								change_tab(get(t).id);
								dispatch("select", { value: get(t).label, index: i });
							}
						});

						append($$anchor, fragment_2);
					};

					if_block(node_2, ($$render) => {
						if ((get(t), untrack(() => get(t)?.visible))) $$render(consequent_4);
					});
				}

				append($$anchor, fragment_1);
			});

			reset(div_2);
			bind_this(div_2, ($$value) => set(measurement_container, $$value), () => get(measurement_container));
			reset(div_1);
			bind_this(div_1, ($$value) => set(stepper_container, $$value), () => get(stepper_container));
			template_effect(() => styles_1 = set_style(div_1, '', styles_1, { '--label-height': get(label_height) + "px" }));
			append($$anchor, fragment);
		};

		if_block(node, ($$render) => {
			if (get(has_tabs)) $$render(consequent_5);
		});
	}

	var node_6 = sibling(node, 2);

	slot(node_6, $$props, 'default', {});
	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, `stepper ${$0 ?? ''}`, 'svelte-1xpzbgb', classes, { hide: !visible(), compact: get(compact) });
			set_attribute(div, 'id', elem_id());
			styles = set_style(div, '', styles, { 'flex-grow': get(tab_scale) });
		},
		[
			() => (
				deep_read_state(elem_classes()),
				untrack(() => elem_classes().join(' '))
			)
		]
	);

	append($$anchor, div);

	var $$pop = pop($$exports);

	$$cleanup();

	return $$pop;
}

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	user_effect(() => {
		if (gradio.props.selected) {
			untrack(() => {
				const i = gradio.props.initial_tabs.findIndex((t) => t.id === gradio.props.selected);

				gradio.dispatch("gradio_tab_select", {
					value: gradio.props.initial_tabs[i].label,
					index: i,
					id: gradio.props.initial_tabs[i].id,
					component_id: gradio.props.initial_tabs[i].component_id
				});
			});
		}
	});

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			Walkthrough($$anchor, {
				get visible() {
					return gradio.shared.visible;
				},

				get elem_id() {
					return gradio.shared.elem_id;
				},

				get elem_classes() {
					return gradio.shared.elem_classes;
				},

				get initial_tabs() {
					return gradio.props.initial_tabs;
				},

				get selected() {
					return gradio.props.selected;
				},

				set selected($$value) {
					gradio.props.selected = $$value;
				},

				$$events: {
					change: () => gradio.dispatch("change"),
					select: (e) => {
						gradio.dispatch("select", e.detail);
						gradio.dispatch("gradio_tab_select", e.detail);
					}
				},

				children: ($$anchor, $$slotProps) => {
					var fragment_2 = comment();
					var node_1 = first_child(fragment_2);

					slot(node_1, $$props, 'default', {});
					append($$anchor, fragment_2);
				},
				$$slots: { default: true }
			});
		};

		var alternate = ($$anchor) => {
			Tabs($$anchor, {
				get visible() {
					return gradio.shared.visible;
				},

				get elem_id() {
					return gradio.shared.elem_id;
				},

				get elem_classes() {
					return gradio.shared.elem_classes;
				},

				get initial_tabs() {
					return gradio.props.initial_tabs;
				},

				get selected() {
					return gradio.props.selected;
				},

				set selected($$value) {
					gradio.props.selected = $$value;
				},

				$$events: {
					change: () => gradio.dispatch("change"),
					select: (e) => {
						gradio.dispatch("select", e.detail);
						gradio.dispatch("gradio_tab_select", e.detail);
					}
				},

				children: ($$anchor, $$slotProps) => {
					var fragment_4 = comment();
					var node_2 = first_child(fragment_4);

					slot(node_2, $$props, 'default', {});
					append($$anchor, fragment_4);
				},
				$$slots: { default: true }
			});
		};

		if_block(node, ($$render) => {
			if (gradio.props.name === "walkthrough") $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { Tabs as BaseTabs, TABS, Index as default };
//# sourceMappingURL=Index-DjI-Ei2E.js.map
