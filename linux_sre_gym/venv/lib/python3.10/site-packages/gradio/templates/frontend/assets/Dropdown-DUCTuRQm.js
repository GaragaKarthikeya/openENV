import { p as prop, d as bind_this, i as if_block, k as each, b as set_class, a as set_attribute, f as set_style, u as index, t as remove_input_defaults } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, w as get, x as set, Z as event, S as first_child, a as append, T as pop, u as state, aw as $window, X as sibling, W as from_html, aj as delegate, V as child, Y as reset, t as template_effect, a0 as set_text, v as proxy, a7 as text, a5 as user_derived, M as tick, at as to_array, a8 as next } from './index-BvBk1Iap.js';
import { b as bind_value } from './input-C317uaXR.js';
import { t as transition, a as fly } from './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { b as bind_window_size } from './window-BGXjFJtG.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { D as DropdownArrow } from './DropdownArrow-Dw1NZKCM.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';

var root_2 = from_html(`<li data-testid="dropdown-option" role="option"><span>✓</span> </li>`);
var root_1 = from_html(`<ul class="options svelte-1ou0lab" role="listbox"></ul>`);
var root$1 = from_html(`<div class="reference"></div> <!>`, 1);

function DropdownOptions($$anchor, $$props) {
	push($$props, true);

	let show_options = prop($$props, 'show_options', 3, false),
		disabled = prop($$props, 'disabled', 3, false),
		selected_indices = prop($$props, 'selected_indices', 19, () => []),
		active_index = prop($$props, 'active_index', 3, null),
		remember_scroll = prop($$props, 'remember_scroll', 3, false),
		offset_from_top = prop($$props, 'offset_from_top', 3, 0),
		from_top = prop($$props, 'from_top', 3, false);

	let distance_from_top = state(0);
	let distance_from_bottom = state(0);
	let input_height = state(0);
	let input_width = state(0);
	let refElement;
	let listElement;
	let top = state(null);
	let bottom = state(null);
	let max_height = state(0);
	let innerHeight = state(0);
	let list_scroll_y = 0;

	function calculate_window_distance() {
		const { top: ref_top, bottom: ref_bottom } = refElement.getBoundingClientRect();

		if (from_top()) {
			set(distance_from_top, offset_from_top());
		} else {
			set(distance_from_top, ref_top, true);
		}

		set(distance_from_bottom, get(innerHeight) - ref_bottom);
	}

	let scroll_timeout = null;

	function scroll_listener() {
		if (!show_options()) return;

		if (scroll_timeout !== null) {
			clearTimeout(scroll_timeout);
		}

		scroll_timeout = setTimeout(
			() => {
				calculate_window_distance();
				scroll_timeout = null;
			},
			10
		);
	}

	function restore_last_scroll() {
		listElement?.scrollTo?.(0, list_scroll_y);
	}

	user_effect(() => {
		if (show_options() && refElement) {
			if (remember_scroll()) {
				restore_last_scroll();
			} else {
				if (listElement && selected_indices().length > 0) {
					let elements = listElement.querySelectorAll("li");

					for (const element of Array.from(elements)) {
						if (element.getAttribute("data-index") === selected_indices()[0].toString()) {
							listElement?.scrollTo?.(0, element.offsetTop);

							break;
						}
					}
				}
			}

			calculate_window_distance();

			const rect = refElement.parentElement?.getBoundingClientRect();

			set(input_height, rect?.height || 0, true);
			set(input_width, rect?.width || 0, true);
			$$props.onload?.();
		}

		if (get(distance_from_bottom) > get(distance_from_top) || from_top()) {
			set(top, `${get(distance_from_top)}px`);
			set(max_height, get(distance_from_bottom), true);
			set(bottom, null);
		} else {
			set(bottom, `${get(distance_from_bottom) + get(input_height)}px`);
			set(max_height, get(distance_from_top) - get(input_height));
			set(top, null);
		}
	});

	var fragment = root$1();

	event('scroll', $window, scroll_listener);

	var div = first_child(fragment);

	bind_this(div, ($$value) => refElement = $$value, () => refElement);

	var node = sibling(div, 2);

	{
		var consequent = ($$anchor) => {
			var ul = root_1();

			ul.__mousedown = (e) => {
				e.preventDefault();
				$$props.onchange?.(e.target.dataset.index);
			};

			let styles;

			each(ul, 21, () => $$props.filtered_indices, index, ($$anchor, index) => {
				var li = root_2();
				let classes;
				let styles_1;
				var span = child(li);
				let classes_1;
				var text = sibling(span);

				reset(li);

				template_effect(
					($0, $1, $2) => {
						classes = set_class(li, 1, 'item svelte-1ou0lab', null, classes, $0);
						set_attribute(li, 'data-index', get(index));
						set_attribute(li, 'aria-label', $$props.choices[get(index)][0]);
						set_attribute(li, 'aria-selected', $1);
						styles_1 = set_style(li, '', styles_1, { width: get(input_width) + "px" });
						classes_1 = set_class(span, 1, 'inner-item svelte-1ou0lab', null, classes_1, $2);
						set_text(text, ` ${$$props.choices[get(index)][0] ?? ''}`);
					},
					[
						() => ({
							selected: selected_indices().includes(get(index)),
							active: get(index) === active_index(),
							'bg-gray-100': get(index) === active_index(),
							'dark:bg-gray-600': get(index) === active_index()
						}),
						() => selected_indices().includes(get(index)),
						() => ({ hide: !selected_indices().includes(get(index)) })
					]
				);

				append($$anchor, li);
			});

			reset(ul);
			bind_this(ul, ($$value) => listElement = $$value, () => listElement);

			template_effect(() => styles = set_style(ul, '', styles, {
				top: get(top),
				bottom: get(bottom),
				'max-height': `calc(${get(max_height)}px - var(--window-padding))`,
				width: get(input_width) + "px"
			}));

			event('scroll', ul, (e) => list_scroll_y = e.currentTarget.scrollTop);
			transition(3, ul, () => fly, () => ({ duration: 200, y: 5 }));
			append($$anchor, ul);
		};

		if_block(node, ($$render) => {
			if (show_options() && !disabled()) $$render(consequent);
		});
	}

	bind_window_size('innerHeight', ($$value) => set(innerHeight, $$value, true));
	append($$anchor, fragment);
	pop();
}

delegate(['mousedown']);

function positive_mod(n, m) {
  return (n % m + m) % m;
}
function handle_filter(choices, input_text) {
  return choices.reduce((filtered_indices, o, index) => {
    if (input_text ? o[0].toLowerCase().includes(input_text.toLowerCase()) : true) {
      filtered_indices.push(index);
    }
    return filtered_indices;
  }, []);
}
function handle_shared_keys(e, active_index, filtered_indices) {
  if (e.key === "Escape") {
    return [false, active_index];
  }
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    if (filtered_indices.length > 0) {
      if (active_index === null) {
        active_index = e.key === "ArrowDown" ? filtered_indices[0] : filtered_indices[filtered_indices.length - 1];
      } else {
        const index_in_filtered = filtered_indices.indexOf(active_index);
        const increment = e.key === "ArrowUp" ? -1 : 1;
        active_index = filtered_indices[positive_mod(index_in_filtered + increment, filtered_indices.length)];
      }
    }
  }
  return [true, active_index];
}

var root_3 = from_html(`<div class="icon-wrap svelte-1xfsv4t"><!></div>`);
var root = from_html(`<div><!> <!> <div class="wrap svelte-1xfsv4t"><div><div class="secondary-wrap svelte-1xfsv4t"><input role="listbox" aria-controls="dropdown-options" autocomplete="off"/> <!></div></div> <!></div></div>`);

function Dropdown($$anchor, $$props) {
	push($$props, true);

	const is_browser = typeof window !== "undefined";

	let label = prop($$props, 'label', 3, "Dropdown"),
		info = prop($$props, 'info', 3, undefined),
		value = prop($$props, 'value', 15),
		choices = prop($$props, 'choices', 19, () => []),
		interactive = prop($$props, 'interactive', 3, true),
		show_label = prop($$props, 'show_label', 3, true),
		container = prop($$props, 'container', 3, true),
		allow_custom_value = prop($$props, 'allow_custom_value', 3, false),
		filterable = prop($$props, 'filterable', 3, true),
		buttons = prop($$props, 'buttons', 3, null),
		oncustom_button_click = prop($$props, 'oncustom_button_click', 3, null);

	let filter_input;

	let show_options = user_derived(() => {
		return is_browser && filter_input === document.activeElement;
	});

	let choices_names = user_derived(() => choices().map((c) => c[0]));
	let choices_values = user_derived(() => choices().map((c) => c[1]));
	let input_text = state("");
	let selected_index = state(null);

	user_effect(() => {
		if (value() === undefined || value() === null || Array.isArray(value()) && value().length === 0) {
			set(input_text, "");
			set(selected_index, null);
		} else if (get(choices_values).includes(value())) {
			set(input_text, get(choices_names)[get(choices_values).indexOf(value())], true);
			set(selected_index, get(choices_values).indexOf(value()), true);
		} else if (allow_custom_value()) {
			set(input_text, value(), true);
			set(selected_index, null);
		} else {
			set(input_text, "");
			set(selected_index, null);
		}
	});

	// Use last_typed_value to track when the user has typed
	// on_blur we only want to update value if the user has typed
	let last_typed_value = get(input_text);

	let initialized = state(false);
	let disabled = user_derived(() => !interactive());

	// All of these are indices with respect to the choices array
	let filtered_indices = state(proxy(choices().map((_, i) => i)));

	let active_index = state(null);
	let selected_indices = user_derived(() => get(selected_index) === null ? [] : [get(selected_index)]);

	function handle_option_selected(index) {
		set(selected_index, parseInt(index), true);

		if (isNaN(get(selected_index))) {
			// This is the case when the user clicks on the scrollbar
			set(selected_index, null);

			return;
		}

		let [_input_text, _value] = choices()[get(selected_index)];

		set(input_text, _input_text, true);
		last_typed_value = get(input_text);
		value(_value);

		$$props.on_select?.({
			index: get(selected_index),
			value: get(choices_values)[get(selected_index)],
			selected: true
		});

		set(show_options, false);
		set(active_index, null);
		$$props.on_input?.();
		filter_input.blur();
	}

	function handle_focus(e) {
		set(filtered_indices, choices().map((_, i) => i), true);
		set(show_options, true);
		$$props.on_focus?.();
	}

	function handle_blur() {
		if (!allow_custom_value()) {
			set(input_text, get(choices_names)[get(choices_values).indexOf(value())], true);
		} else {
			if (get(choices_names).includes(get(input_text))) {
				set(selected_index, get(choices_names).indexOf(get(input_text)), true);
				value(get(choices_values)[get(selected_index)]);
			} else if (get(input_text) !== last_typed_value) {
				value(get(input_text));
				set(selected_index, null);
			}
		}

		set(show_options, false);
		set(active_index, null);
		set(filtered_indices, choices().map((_, i) => i), true);
		$$props.on_blur?.();
		$$props.on_input?.();
	}

	async function handle_key_down(e) {
		await tick();
		set(filtered_indices, handle_filter(choices(), get(input_text)), true);
		set(active_index, get(filtered_indices).length > 0 ? get(filtered_indices)[0] : null, true);

		(($$value) => {
			var $$array = to_array($$value, 2);

			set(show_options, $$array[0]);
			set(active_index, $$array[1], true);
		})(handle_shared_keys(e, get(active_index), get(filtered_indices)));

		if (e.key === "Enter") {
			last_typed_value = get(input_text);

			if (get(active_index) !== null) {
				set(selected_index, get(active_index), true);
				value(get(choices_values)[get(active_index)]);
				set(show_options, false);
				filter_input.blur();
				set(active_index, null);
			} else if (get(choices_names).includes(get(input_text))) {
				set(selected_index, get(choices_names).indexOf(get(input_text)), true);
				value(get(choices_values)[get(selected_index)]);
				set(show_options, false);
				set(active_index, null);
				filter_input.blur();
			} else if (allow_custom_value()) {
				value(get(input_text));
				set(selected_index, null);
				set(show_options, false);
				set(active_index, null);
				filter_input.blur();
			}
		}
	}

	let old_value = state(proxy(value()));

	user_effect(() => {
		if (get(old_value) !== value()) {
			set(old_value, value(), true);
			$$props.on_change?.(value());
		}
	});

	var div = root();
	let classes;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				get buttons() {
					return buttons();
				},

				get on_custom_button_click() {
					return oncustom_button_click();
				}
			});
		};

		if_block(node, ($$render) => {
			if (show_label() && buttons() && buttons().length > 0) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	BlockTitle(node_1, {
		get show_label() {
			return show_label();
		},

		get info() {
			return info();
		},

		children: ($$anchor, $$slotProps) => {
			next();

			var text$1 = text();

			template_effect(() => set_text(text$1, label()));
			append($$anchor, text$1);
		},
		$$slots: { default: true }
	});

	var div_1 = sibling(node_1, 2);
	var div_2 = child(div_1);
	let classes_1;
	var div_3 = child(div_2);
	var input = child(div_3);

	remove_input_defaults(input);

	let classes_2;

	input.__keydown = handle_key_down;

	input.__keyup = (e) => {
		$$props.on_key_up?.({ key: e.key, input_value: get(input_text) });
	};

	bind_this(input, ($$value) => filter_input = $$value, () => filter_input);

	var node_2 = sibling(input, 2);

	{
		var consequent_1 = ($$anchor) => {
			var div_4 = root_3();
			var node_3 = child(div_4);

			DropdownArrow(node_3);
			reset(div_4);
			append($$anchor, div_4);
		};

		if_block(node_2, ($$render) => {
			if (!get(disabled)) $$render(consequent_1);
		});
	}

	reset(div_3);
	reset(div_2);

	var node_4 = sibling(div_2, 2);

	DropdownOptions(node_4, {
		get show_options() {
			return get(show_options);
		},

		get choices() {
			return choices();
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
		onchange: handle_option_selected,
		onload: () => set(initialized, true)
	});

	reset(div_1);
	reset(div);

	template_effect(
		($0) => {
			classes = set_class(div, 1, 'svelte-1xfsv4t', null, classes, { container: container() });
			classes_1 = set_class(div_2, 1, 'wrap-inner svelte-1xfsv4t', null, classes_1, { show_options: get(show_options) });
			set_attribute(input, 'aria-expanded', get(show_options));
			set_attribute(input, 'aria-label', label());
			classes_2 = set_class(input, 1, 'border-none svelte-1xfsv4t', null, classes_2, $0);
			input.disabled = get(disabled);
			input.readOnly = !filterable();
		},
		[
			() => ({
				subdued: !get(choices_names).includes(get(input_text)) && !allow_custom_value()
			})
		]
	);

	event('blur', input, handle_blur);
	event('focus', input, handle_focus);
	bind_value(input, () => get(input_text), ($$value) => set(input_text, $$value));
	append($$anchor, div);
	pop();
}

delegate(['keydown', 'keyup']);

export { DropdownOptions as D, handle_filter as a, Dropdown as b, handle_shared_keys as h };
//# sourceMappingURL=Dropdown-DUCTuRQm.js.map
