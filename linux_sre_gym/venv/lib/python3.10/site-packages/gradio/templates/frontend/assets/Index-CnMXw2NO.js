import { i as if_block, k as each, d as bind_this, b as set_class, u as index, a as set_attribute, t as remove_input_defaults, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, R as push, u as state, v as proxy, y as user_effect, w as get, x as set, a7 as text, t as template_effect, Z as event, T as pop, a5 as user_derived, X as sibling, V as child, W as from_html, at as to_array, a0 as set_text, a8 as next, Y as reset, S as first_child } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { b as bind_value } from './input-C317uaXR.js';
import { p as preventDefault } from './event-modifiers-DanhKw3_.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { D as DropdownArrow } from './DropdownArrow-Dw1NZKCM.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { D as DropdownOptions, h as handle_shared_keys, a as handle_filter, b as Dropdown } from './Dropdown-DUCTuRQm.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
export { default as BaseExample } from './Example-t0cB3HBf.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './window-BGXjFJtG.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';
/* empty css                                               */

var root$1 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100%" height="100%"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path></svg>`);

function Remove($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

var root_6 = from_html(`<div class="token-remove svelte-1dv2vbb" role="button" tabindex="0"><!></div>`);
var root_3 = from_html(`<div class="token svelte-1dv2vbb"><span class="svelte-1dv2vbb"><!></span> <!></div>`);
var root_8 = from_html(`<div role="button" tabindex="0" class="token-remove remove-all svelte-1dv2vbb"><!></div>`);
var root_7 = from_html(`<!> <span class="icon-wrap svelte-1dv2vbb"><!></span>`, 1);
var root = from_html(`<div><!> <!> <div class="wrap svelte-1dv2vbb"><div><!> <div class="secondary-wrap svelte-1dv2vbb"><input autocomplete="off"/> <!></div></div> <!></div></div>`);

function Multiselect($$anchor, $$props) {
	push($$props, true);
	const gradio = $$props.gradio;
	let filter_input;
	let input_text = state("");
	let label = user_derived(() => gradio.shared.label || "Multiselect");
	let buttons = user_derived(() => gradio.props.buttons);

	let choices_names = user_derived(() => {
		return gradio.props.choices.map((c) => c[0]);
	});

	let choices_values = user_derived(() => {
		return gradio.props.choices.map((c) => c[1]);
	});

	let disabled = user_derived(() => !gradio.shared.interactive);
	let show_options = state(false);

	// All of these are indices with respect to the choices array
	let $$d = user_derived(() => {
			const filtered = handle_filter(gradio.props.choices, get(input_text));

			return [
				filtered,
				filtered.length > 0 && !gradio.props.allow_custom_value ? filtered[0] : null
			];
		}),
		$$array = user_derived(() => to_array(get($$d), 2)),
		filtered_indices = user_derived(() => get($$array)[0]),
		active_index = user_derived(() => get($$array)[1]);

	function set_selected_indices() {
		if (gradio.props.value === undefined) {
			return [];
		} else if (Array.isArray(gradio.props.value)) {
			return gradio.props.value.map((v) => {
				const index = get(choices_values).indexOf(v);

				if (index !== -1) {
					return index;
				}

				if (gradio.props.allow_custom_value) {
					return v;
				}

				// Instead of returning null, skip this iteration
				return undefined;
			}).filter((val) => val !== undefined);
		}

		return [];
	}

	let selected_indices = user_derived(set_selected_indices);

	function handle_blur() {
		if (!gradio.props.allow_custom_value) {
			set(input_text, "");
		}

		if (gradio.props.allow_custom_value && get(input_text) !== "") {
			add_selected_choice(get(input_text));
			set(input_text, "");
		}

		gradio.dispatch("blur");
		set(show_options, false);
		set(active_index, null);
	}

	function remove_selected_choice(option_index) {
		set(selected_indices, get(selected_indices).filter((v) => v !== option_index));
		gradio.props.value = get(selected_indices).map((index) => typeof index === "number" ? get(choices_values)[index] : index);
		gradio.dispatch("input");

		gradio.dispatch("select", {
			index: typeof option_index === "number" ? option_index : -1,
			value: typeof option_index === "number" ? get(choices_values)[option_index] : option_index,
			selected: false
		});
	}

	function add_selected_choice(option_index) {
		if (gradio.props.max_choices == null || get(selected_indices).length < gradio.props.max_choices) {
			get(selected_indices).push(option_index);

			gradio.dispatch("select", {
				index: typeof option_index === "number" ? option_index : -1,
				value: typeof option_index === "number" ? get(choices_values)[option_index] : option_index,
				selected: true
			});
		}

		if (get(selected_indices).length === gradio.props.max_choices) {
			set(show_options, false);
			set(active_index, null);
			filter_input.blur();
		}

		gradio.props.value = get(selected_indices).map((index) => typeof index === "number" ? get(choices_values)[index] : index);
	}

	function handle_option_selected(index) {
		const option_index = parseInt(index);

		add_or_remove_index(option_index);
	}

	function add_or_remove_index(option_index) {
		if (get(selected_indices).includes(option_index)) {
			remove_selected_choice(option_index);
		} else {
			add_selected_choice(option_index);
		}

		set(input_text, "");
		set(active_index, null);
		gradio.dispatch("input");
	}

	function remove_all(e) {
		set(selected_indices, []);
		set(input_text, "");
		gradio.props.value = [];
		e.preventDefault();
	}

	function handle_focus(e) {
		set(filtered_indices, gradio.props.choices.map((_, i) => i));

		if (gradio.props.max_choices === null || get(selected_indices).length < gradio.props.max_choices) {
			set(show_options, true);
		}

		gradio.dispatch("focus");
		set(show_options, true);
	}

	function handle_key_down(e) {
		(($$value) => {
			var $$array_1 = to_array($$value, 2);

			set(show_options, $$array_1[0], true);
			set(active_index, $$array_1[1]);
		})(handle_shared_keys(e, get(active_index), get(filtered_indices)));

		if (e.key === "Enter") {
			if (get(active_index) !== null) {
				add_or_remove_index(get(active_index));
			} else {
				if (gradio.props.allow_custom_value) {
					add_selected_choice(get(input_text));
					set(input_text, "");
				}
			}
		}

		if (e.key === "Backspace" && get(input_text) === "") {
			set(selected_indices, [...get(selected_indices).slice(0, -1)]);
		}

		if (get(selected_indices).length === gradio.props.max_choices) {
			set(show_options, false);
			set(active_index, null);
		}
	}

	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	function oncustom_button_click(id) {
		gradio.dispatch("custom_button_click", { id });
	}

	var div = root();
	let classes;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				get buttons() {
					return get(buttons);
				},
				on_custom_button_click: oncustom_button_click
			});
		};

		if_block(node, ($$render) => {
			if (gradio.shared.show_label && get(buttons) && get(buttons).length > 0) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

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

			template_effect(() => set_text(text$1, get(label)));
			append($$anchor, text$1);
		},
		$$slots: { default: true }
	});

	var div_1 = sibling(node_1, 2);
	var div_2 = child(div_1);
	let classes_1;
	var node_2 = child(div_2);

	each(node_2, 17, () => get(selected_indices), index, ($$anchor, s) => {
		var div_3 = root_3();
		var span = child(div_3);
		var node_3 = child(span);

		{
			var consequent_1 = ($$anchor) => {
				var text_1 = text();

				template_effect(() => set_text(text_1, get(choices_names)[get(s)]));
				append($$anchor, text_1);
			};

			var alternate = ($$anchor) => {
				var text_2 = text();

				template_effect(() => set_text(text_2, get(s)));
				append($$anchor, text_2);
			};

			if_block(node_3, ($$render) => {
				if (typeof get(s) === "number") $$render(consequent_1); else $$render(alternate, false);
			});
		}

		reset(span);

		var node_4 = sibling(span, 2);

		{
			var consequent_2 = ($$anchor) => {
				var div_4 = root_6();
				var node_5 = child(div_4);

				Remove(node_5);
				reset(div_4);
				template_effect(($0) => set_attribute(div_4, 'title', $0), [() => gradio.i18n("common.remove") + " " + get(s)]);
				event('click', div_4, preventDefault(() => remove_selected_choice(get(s))));

				event('keydown', div_4, (event) => {
					if (event.key === "Enter") {
						remove_selected_choice(get(s));
					}
				});

				append($$anchor, div_4);
			};

			if_block(node_4, ($$render) => {
				if (!get(disabled)) $$render(consequent_2);
			});
		}

		reset(div_3);
		append($$anchor, div_3);
	});

	var div_5 = sibling(node_2, 2);
	var input = child(div_5);

	remove_input_defaults(input);

	let classes_2;

	bind_this(input, ($$value) => filter_input = $$value, () => filter_input);

	var node_6 = sibling(input, 2);

	{
		var consequent_4 = ($$anchor) => {
			var fragment_4 = root_7();
			var node_7 = first_child(fragment_4);

			{
				var consequent_3 = ($$anchor) => {
					var div_6 = root_8();
					var node_8 = child(div_6);

					Remove(node_8);
					reset(div_6);
					template_effect(($0) => set_attribute(div_6, 'title', $0), [() => gradio.i18n("common.clear")]);
					event('click', div_6, remove_all);

					event('keydown', div_6, (event) => {
						if (event.key === "Enter") {
							remove_all(event);
						}
					});

					append($$anchor, div_6);
				};

				if_block(node_7, ($$render) => {
					if (get(selected_indices).length > 0) $$render(consequent_3);
				});
			}

			var span_1 = sibling(node_7, 2);
			var node_9 = child(span_1);

			DropdownArrow(node_9);
			reset(span_1);
			append($$anchor, fragment_4);
		};

		if_block(node_6, ($$render) => {
			if (!get(disabled)) $$render(consequent_4);
		});
	}

	reset(div_5);
	reset(div_2);

	var node_10 = sibling(div_2, 2);

	DropdownOptions(node_10, {
		get show_options() {
			return get(show_options);
		},

		get choices() {
			return gradio.props.choices;
		},

		get filtered_indices() {
			return get(filtered_indices);
		},

		get disabled() {
			return get(disabled);
		},

		get selected_indices() {
			return get(selected_indices);
		},

		get active_index() {
			return get(active_index);
		},
		remember_scroll: true,
		onchange: handle_option_selected
	});

	reset(div_1);
	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, 'svelte-1dv2vbb', null, classes, { container: gradio.shared.container });
			classes_1 = set_class(div_2, 1, 'wrap-inner svelte-1dv2vbb', null, classes_1, { show_options: get(show_options) });
			classes_2 = set_class(input, 1, 'border-none svelte-1dv2vbb', null, classes_2, $0);
			input.disabled = get(disabled);
			input.readOnly = !gradio.props.filterable;
		},
		[
			() => ({
				subdued: !get(choices_names).includes(get(input_text)) && !gradio.props.allow_custom_value || get(selected_indices).length === gradio.props.max_choices
			})
		]
	);

	bind_value(input, () => get(input_text), ($$value) => set(input_text, $$value));
	event('keydown', input, handle_key_down);

	event('keyup', input, (e) => {
		gradio.dispatch("key_up", { key: e.key, input_value: get(input_text) });
	});

	event('blur', input, handle_blur);
	event('focus', input, handle_focus);
	append($$anchor, div);
	pop();
}

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

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
					on_clear_status: () => gradio.dispatch("clear_status", loading_status)
				}
			));

			var node_1 = sibling(node, 2);

			{
				var consequent = ($$anchor) => {
					Multiselect($$anchor, {
						get gradio() {
							return gradio;
						}
					});
				};

				var alternate = ($$anchor) => {
					Dropdown($$anchor, {
						get label() {
							return gradio.shared.label;
						},

						get info() {
							return gradio.props.info;
						},

						get choices() {
							return gradio.props.choices;
						},

						get interactive() {
							return gradio.shared.interactive;
						},

						get show_label() {
							return gradio.shared.show_label;
						},

						get container() {
							return gradio.shared.container;
						},

						get allow_custom_value() {
							return gradio.props.allow_custom_value;
						},

						get filterable() {
							return gradio.props.filterable;
						},

						get buttons() {
							return gradio.props.buttons;
						},

						oncustom_button_click: (id) => {
							gradio.dispatch("custom_button_click", { id });
						},
						on_change: () => gradio.dispatch("change"),
						on_input: () => gradio.dispatch("input"),
						on_select: (data) => gradio.dispatch("select", data),
						on_focus: () => gradio.dispatch("focus"),
						on_blur: () => gradio.dispatch("blur"),
						on_key_up: (data) => gradio.dispatch("key_up", data),
						get value() {
							return gradio.props.value;
						},

						set value($$value) {
							gradio.props.value = $$value;
						}
					});
				};

				if_block(node_1, ($$render) => {
					if (gradio.props.multiselect) $$render(consequent); else $$render(alternate, false);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Dropdown as BaseDropdown, DropdownOptions as BaseDropdownOptions, Multiselect as BaseMultiselect, Index as default };
//# sourceMappingURL=Index-CnMXw2NO.js.map
