import { p as prop, w as writable, e as init, i as if_block, s as slot, b as set_class, a as set_attribute, f as set_style, q as store_get, m as setup_stores, x as store_set, k as each, d as bind_this, u as index } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, R as push, a3 as mutable_source, w as get, a4 as createEventDispatcher, aa as onMount, ax as setContext, a1 as legacy_pre_effect, G as deep_read_state, a2 as legacy_pre_effect_reset, Z as event, t as template_effect, z as untrack, T as pop, U as flushSync, X as sibling, W as from_html, as as mutate, aw as $window, M as tick, x as set, V as child, a6 as comment, S as first_child, Y as reset, a0 as set_text } from './index-BvBk1Iap.js';
import { s as stopPropagation } from './event-modifiers-DanhKw3_.js';

var root$1 = from_svg(`<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="2.5" cy="8" r="1.5" fill="currentColor"></circle><circle cx="8" cy="8" r="1.5" fill="currentColor"></circle><circle cx="13.5" cy="8" r="1.5" fill="currentColor"></circle></svg>`);

function OverflowIcon($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

const TABS = {};

var root_3 = from_html(`<button class="svelte-11gaq1"> </button>`);
var root_5 = from_html(`<button role="tab"> </button>`);
var root_7 = from_html(`<button> </button>`);
var root_1 = from_html(`<div class="tab-wrapper svelte-11gaq1"><div class="tab-container visually-hidden svelte-11gaq1" aria-hidden="true"></div> <div class="tab-container svelte-11gaq1" role="tablist"></div> <span><button><!></button> <div></div></span></div>`);
var root = from_html(`<div><!> <!></div>`);

function Tabs($$anchor, $$props) {
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
	let visible_tabs = mutable_source([...initial_tabs()]);
	let overflow_tabs = mutable_source([]);
	let overflow_menu_open = mutable_source(false);
	let overflow_menu = mutable_source();

	// Track which tab orders have been registered by mounted TabItem components.
	// Once a TabItem mounts and calls register_tab, it manages its own tab entry
	// via _set_data -> register_tab, so _sync_tabs should not overwrite it.
	let mounted_tab_orders = new Set();

	// When initial_tabs changes (e.g. a non-mounted tab's props were updated),
	// sync the internal tabs array so the tab buttons reflect the new state.
	// Using a function call so the $: dependency is only on initial_tabs,
	// not on tabs (which would cause a loop with register_tab).
	function _sync_tabs(new_tabs) {
		for (let i = 0; i < new_tabs.length; i++) {
			if (new_tabs[i] && !mounted_tab_orders.has(i)) {
				mutate(tabs, get(tabs)[i] = new_tabs[i]);
			}
		}
	}

	let tab_nav_el = mutable_source();
	const selected_tab = writable(selected() || get(tabs)[0]?.id || false);
	const selected_tab_index = writable(get(tabs).findIndex((t) => t?.id === selected()) || 0);
	const dispatch = createEventDispatcher();
	let is_overflowing = mutable_source(false);
	let overflow_has_selected_tab = mutable_source(false);
	let tab_els = mutable_source({});

	onMount(() => {
		if (!get(tab_nav_el)) return;

		const observer = new IntersectionObserver((entries) => {
			handle_menu_overflow();
		});

		observer.observe(get(tab_nav_el));
	});

	setContext(TABS, {
		register_tab: (tab, order) => {
			mounted_tab_orders.add(order);
			mutate(tabs, get(tabs)[order] = tab);

			if ($selected_tab() === false && tab.visible !== false && tab.interactive) {
				store_set(selected_tab, tab.id);
				store_set(selected_tab_index, order);
			}

			return order;
		},

		unregister_tab: (tab, order) => {
			mounted_tab_orders.delete(order);

			if ($selected_tab() === tab.id) {
				store_set(selected_tab, get(tabs)[0]?.id || false);
			}

			mutate(tabs, get(tabs)[order] = null);
		},
		selected_tab,
		selected_tab_index
	});

	function change_tab(id) {
		const tab_to_activate = get(tabs).find((t) => t?.id === id);

		if (id !== undefined && tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible !== false && $selected_tab() !== tab_to_activate.id) {
			selected(id);
			store_set(selected_tab, id);
			store_set(selected_tab_index, get(tabs).findIndex((t) => t?.id === id));
			dispatch("change");
			set(overflow_menu_open, false);
		}
	}

	function handle_outside_click(event) {
		if (get(overflow_menu_open) && get(overflow_menu) && !get(overflow_menu).contains(event.target)) {
			set(overflow_menu_open, false);
		}
	}

	async function handle_menu_overflow() {
		if (!get(tab_nav_el)) return;

		await tick();

		const tab_nav_size = get(tab_nav_el).getBoundingClientRect();
		let max_width = tab_nav_size.width;
		const tab_sizes = get_tab_sizes(get(tabs), get(tab_els));
		let last_visible_index = 0;
		const offset = tab_nav_size.left;

		for (let i = get(tabs).length - 1; i >= 0; i--) {
			const tab = get(tabs)[i];

			if (!tab) continue;

			const tab_rect = tab_sizes[tab.id];

			if (!tab_rect) continue;

			if (tab_rect.right - offset < max_width) {
				last_visible_index = i;

				break;
			}
		}

		set(overflow_tabs, get(tabs).slice(last_visible_index + 1));
		set(visible_tabs, get(tabs).slice(0, last_visible_index + 1));
		set(overflow_has_selected_tab, handle_overflow_has_selected_tab($selected_tab()));
		set(is_overflowing, get(overflow_tabs).length > 0);
	}

	function handle_overflow_has_selected_tab(selected_tab) {
		if (selected_tab === false) return false;

		return get(overflow_tabs).some((t) => t?.id === selected_tab);
	}

	function get_tab_sizes(tabs, tab_els) {
		const tab_sizes = {};

		tabs.forEach((tab) => {
			if (!tab) return;

			tab_sizes[tab.id] = tab_els[tab.id]?.getBoundingClientRect();
		});

		return tab_sizes;
	}

	legacy_pre_effect(() => (deep_read_state(initial_tabs())), () => {
		_sync_tabs(initial_tabs());
	});

	legacy_pre_effect(() => (get(tabs)), () => {
		set(has_tabs, get(tabs).length > 0);
	});

	legacy_pre_effect(() => (get(tabs), deep_read_state(selected())), () => {
		(get(tabs), selected() !== null && change_tab(selected()));
	});

	legacy_pre_effect(() => (get(tabs), get(tab_nav_el), get(tab_els)), () => {
		(
			get(tabs),
			get(tab_nav_el),
			get(tab_els),
			handle_menu_overflow()
		);
	});

	legacy_pre_effect(() => ($selected_tab()), () => {
		set(overflow_has_selected_tab, handle_overflow_has_selected_tab($selected_tab()));
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

	event('resize', $window, handle_menu_overflow);
	event('click', $window, handle_outside_click);

	let classes;
	let styles;
	var node = child(div);

	{
		var consequent_3 = ($$anchor) => {
			var div_1 = root_1();
			var div_2 = child(div_1);

			each(div_2, 5, () => get(tabs), index, ($$anchor, t) => {
				var fragment = comment();
				var node_1 = first_child(fragment);

				{
					var consequent = ($$anchor) => {
						var button = root_3();
						var text = child(button, true);

						reset(button);
						bind_this(button, ($$value, t) => mutate(tab_els, get(tab_els)[t.id] = $$value), (t) => get(tab_els)?.[t.id], () => [get(t)]);
						template_effect(() => set_text(text, (get(t), untrack(() => get(t)?.label))));
						append($$anchor, button);
					};

					if_block(node_1, ($$render) => {
						if ((
							get(t),
							untrack(() => get(t) && get(t)?.visible !== false && get(t)?.visible !== "hidden")
						)) $$render(consequent);
					});
				}

				append($$anchor, fragment);
			});

			reset(div_2);

			var div_3 = sibling(div_2, 2);

			each(div_3, 5, () => get(visible_tabs), index, ($$anchor, t, i) => {
				var fragment_1 = comment();
				var node_2 = first_child(fragment_1);

				{
					var consequent_1 = ($$anchor) => {
						var button_1 = root_5();
						let classes_1;
						var text_1 = child(button_1, true);

						reset(button_1);

						template_effect(() => {
							set_attribute(button_1, 'aria-selected', (
								get(t),
								$selected_tab(),
								untrack(() => get(t).id === $selected_tab())
							));

							set_attribute(button_1, 'aria-controls', (get(t), untrack(() => get(t).elem_id)));
							button_1.disabled = (get(t), untrack(() => !get(t).interactive));
							set_attribute(button_1, 'aria-disabled', (get(t), untrack(() => !get(t).interactive)));

							set_attribute(button_1, 'id', (
								get(t),
								untrack(() => get(t).elem_id ? get(t).elem_id + "-button" : null)
							));

							set_attribute(button_1, 'data-tab-id', (get(t), untrack(() => get(t).id)));
							classes_1 = set_class(button_1, 1, 'svelte-11gaq1', null, classes_1, { selected: get(t).id === $selected_tab() });

							set_text(text_1, (
								get(t),
								untrack(() => get(t)?.label !== undefined ? get(t)?.label : "Tab " + (i + 1))
							));
						});

						event('click', button_1, () => {
							if (get(t).id !== $selected_tab()) {
								change_tab(get(t).id);

								dispatch("select", {
									value: get(t).label,
									index: i,
									id: get(t).id,
									component_id: get(t).component_id
								});
							}
						});

						append($$anchor, button_1);
					};

					if_block(node_2, ($$render) => {
						if ((
							get(t),
							untrack(() => get(t) && get(t)?.visible !== false)
						)) $$render(consequent_1);
					});
				}

				append($$anchor, fragment_1);
			});

			reset(div_3);
			bind_this(div_3, ($$value) => set(tab_nav_el, $$value), () => get(tab_nav_el));

			var span = sibling(div_3, 2);
			let classes_2;
			var button_2 = child(span);
			let classes_3;
			var node_3 = child(button_2);

			OverflowIcon(node_3);
			reset(button_2);

			var div_4 = sibling(button_2, 2);
			let classes_4;

			each(div_4, 5, () => get(overflow_tabs), index, ($$anchor, t, i) => {
				var fragment_2 = comment();
				var node_4 = first_child(fragment_2);

				{
					var consequent_2 = ($$anchor) => {
						var button_3 = root_7();
						let classes_5;
						var text_2 = child(button_3, true);

						reset(button_3);

						template_effect(() => {
							classes_5 = set_class(button_3, 1, 'svelte-11gaq1', null, classes_5, { selected: get(t)?.id === $selected_tab() });
							set_text(text_2, (get(t), untrack(() => get(t)?.label)));
						});

						event('click', button_3, () => {
							change_tab(get(t)?.id);

							dispatch("select", {
								value: get(t).label,
								index: i,
								id: get(t).id,
								component_id: get(t).component_id
							});
						});

						append($$anchor, button_3);
					};

					if_block(node_4, ($$render) => {
						if ((get(t), untrack(() => get(t)?.visible !== false))) $$render(consequent_2);
					});
				}

				append($$anchor, fragment_2);
			});

			reset(div_4);
			reset(span);
			bind_this(span, ($$value) => set(overflow_menu, $$value), () => get(overflow_menu));
			reset(div_1);

			template_effect(
				($0) => {
					classes_2 = set_class(span, 1, 'overflow-menu svelte-11gaq1', null, classes_2, $0);
					classes_3 = set_class(button_2, 1, 'svelte-11gaq1', null, classes_3, { 'overflow-item-selected': get(overflow_has_selected_tab) });
					classes_4 = set_class(div_4, 1, 'overflow-dropdown svelte-11gaq1', null, classes_4, { hide: !get(overflow_menu_open) });
				},
				[
					() => ({
						hide: !get(is_overflowing) || !get(overflow_tabs).some((t) => t?.visible !== false)
					})
				]
			);

			event('click', button_2, stopPropagation(() => set(overflow_menu_open, !get(overflow_menu_open))));
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (get(has_tabs)) $$render(consequent_3);
		});
	}

	var node_5 = sibling(node, 2);

	slot(node_5, $$props, 'default', {});
	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, `tabs ${$0 ?? ''}`, 'svelte-11gaq1', classes, { hide: visible() === false, hidden: visible() === "hidden" });
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

export { TABS as T, Tabs as a };
//# sourceMappingURL=Walkthrough.svelte_svelte_type_style_lang-Bq8jBNe2.js.map
