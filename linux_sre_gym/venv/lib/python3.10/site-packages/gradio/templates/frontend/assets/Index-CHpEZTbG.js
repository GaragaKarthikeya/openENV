import { p as prop, e as init, s as slot, a as set_attribute, b as set_class, f as set_style, q as store_get, m as setup_stores, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, a4 as createEventDispatcher, ay as getContext, aa as onMount, a1 as legacy_pre_effect, G as deep_read_state, w as get, a2 as legacy_pre_effect_reset, a6 as comment, S as first_child, a as append, t as template_effect, z as untrack, T as pop, a3 as mutable_source, U as flushSync, V as child, ad as derived_safe_equal, W as from_html, x as set, M as tick, Y as reset } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { T as TABS } from './Walkthrough.svelte_svelte_type_style_lang-Bq8jBNe2.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-C97CbKPo.js';
import './clone-DaB-S2nH.js';
import './event-modifiers-DanhKw3_.js';
import './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './html-CdUEtR5E.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './index-C7e-J7CF.js';
import './Clear-Dgs5UhTO.js';

var root = from_html(`<div role="tabpanel"><!></div>`);

function TabItem($$anchor, $$props) {
	push($$props, false);

	const $selected_tab_index = () => store_get(selected_tab_index, '$selected_tab_index', $$stores);
	const $selected_tab = () => store_get(selected_tab, '$selected_tab', $$stores);
	const [$$stores, $$cleanup] = setup_stores();
	const props_json = mutable_source();
	let elem_id = prop($$props, 'elem_id', 12, "");
	let elem_classes = prop($$props, 'elem_classes', 28, () => []);
	let label = prop($$props, 'label', 12);
	let id = prop($$props, 'id', 28, () => ({}));
	let visible = prop($$props, 'visible', 12);
	let interactive = prop($$props, 'interactive', 12);
	let order = prop($$props, 'order', 12);
	let scale = prop($$props, 'scale', 12);
	let component_id = prop($$props, 'component_id', 12);
	const dispatch = createEventDispatcher();

	const {
		register_tab,
		unregister_tab,
		selected_tab,
		selected_tab_index
	} = getContext(TABS);

	let tab_index = mutable_source();

	function _register_tab(obj, order) {
		obj = JSON.parse(obj);

		return register_tab(obj, order);
	}

	onMount(() => {
		return () => unregister_tab({ label: label(), id: id(), elem_id: elem_id() }, order());
	});

	legacy_pre_effect(
		() => (
			deep_read_state(label()),
			deep_read_state(id()),
			deep_read_state(elem_id()),
			deep_read_state(visible()),
			deep_read_state(interactive()),
			deep_read_state(scale()),
			deep_read_state(component_id())
		),
		() => {
			set(props_json, JSON.stringify({
				label: label(),
				id: id(),
				elem_id: elem_id(),
				visible: visible(),
				interactive: interactive(),
				scale: scale(),
				component_id: component_id()
			}));
		}
	);

	legacy_pre_effect(() => (get(props_json), deep_read_state(order())), () => {
		set(tab_index, _register_tab(get(props_json), order()));
	});

	legacy_pre_effect(
		() => (
			$selected_tab_index(),
			get(tab_index),
			deep_read_state(label())
		),
		() => {
			$selected_tab_index() === get(tab_index) && tick().then(() => dispatch("select", { value: label(), index: get(tab_index) }));
		}
	);

	legacy_pre_effect_reset();

	var $$exports = {
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

		get label() {
			return label();
		},

		set label($$value) {
			label($$value);
			flushSync();
		},

		get id() {
			return id();
		},

		set id($$value) {
			id($$value);
			flushSync();
		},

		get visible() {
			return visible();
		},

		set visible($$value) {
			visible($$value);
			flushSync();
		},

		get interactive() {
			return interactive();
		},

		set interactive($$value) {
			interactive($$value);
			flushSync();
		},

		get order() {
			return order();
		},

		set order($$value) {
			order($$value);
			flushSync();
		},

		get scale() {
			return scale();
		},

		set scale($$value) {
			scale($$value);
			flushSync();
		},

		get component_id() {
			return component_id();
		},

		set component_id($$value) {
			component_id($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	let classes;
	let styles;
	var node = child(div);

	{
		let $0 = derived_safe_equal(() => scale() >= 1 ? scale() : null);

		BaseColumn(node, {
			get scale() {
				return get($0);
			},

			children: ($$anchor, $$slotProps) => {
				var fragment = comment();
				var node_1 = first_child(fragment);

				slot(node_1, $$props, 'default', {});
				append($$anchor, fragment);
			},
			$$slots: { default: true }
		});
	}

	reset(div);

	template_effect(
		($0) => {
			set_attribute(div, 'id', elem_id());
			classes = set_class(div, 1, `tabitem ${$0 ?? ''}`, 'svelte-dmtrd3', classes, { 'grow-children': scale() >= 1 });

			styles = set_style(div, '', styles, {
				display: $selected_tab() === id() && visible() !== false ? "flex" : "none",
				'flex-grow': scale()
			});
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

	TabItem($$anchor, {
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get label() {
			return gradio.shared.label;
		},

		get visible() {
			return gradio.shared.visible;
		},

		get interactive() {
			return gradio.shared.interactive;
		},

		get id() {
			return gradio.props.id;
		},

		get order() {
			return gradio.props.order;
		},

		get scale() {
			return gradio.props.scale;
		},

		get component_id() {
			return gradio.props.component_id;
		},
		$$events: { select: ({ detail }) => gradio.dispatch("select", detail) },
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = comment();
			var node = first_child(fragment_1);

			slot(node, $$props, 'default', {});
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { TabItem as BaseTabItem, Index as default };
//# sourceMappingURL=Index-CHpEZTbG.js.map
