import { M as colors, p as prop, t as remove_input_defaults, a as set_attribute, F as set_value, f as set_style, i as if_block, k as each, u as index, b as set_class, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, aj as delegate, R as push, v as proxy, u as state, al as autofocus, x as set, t as template_effect, w as get, Z as event, T as pop, W as from_html, y as user_effect, aa as onMount, V as child, S as first_child, X as sibling, a6 as comment, a5 as user_derived, Y as reset, a0 as set_text, at as to_array } from './index-BvBk1Iap.js';
import { g as get_next_color } from './color-Tpcfv14B.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './snippet-Bl5YVoCg.js';
import './clone-DaB-S2nH.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root$2 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--carbon" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path fill="currentColor" d="M12 15H5a3 3 0 0 1-3-3v-2a3 3 0 0 1 3-3h5V5a1 1 0 0 0-1-1H3V2h6a3 3 0 0 1 3 3zM5 9a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h5V9zm15 14v2a1 1 0 0 0 1 1h5v-4h-5a1 1 0 0 0-1 1z"></path><path fill="currentColor" d="M2 30h28V2Zm26-2h-7a3 3 0 0 1-3-3v-2a3 3 0 0 1 3-3h5v-2a1 1 0 0 0-1-1h-6v-2h6a3 3 0 0 1 3 3Z"></path></svg>`);

function TextHighlight($$anchor) {
	var svg = root$2();

	append($$anchor, svg);
}

function name_to_rgba(name, alpha) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = name;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
function is_transparent(color) {
  if (!color) return true;
  const c = color.toLowerCase().trim();
  return c === "transparent" || c.length === 9 && c.endsWith("00");
}
function generate_color_map(color_map, is_browser) {
  const result = {};
  for (const key in color_map) {
    const color = color_map[key].trim();
    if (color in colors) {
      result[key] = colors[color];
    } else if (is_transparent(color)) {
      result[key] = {
        primary: "transparent",
        secondary: "transparent"
      };
    } else {
      result[key] = {
        primary: is_browser ? name_to_rgba(color, 1) : color,
        secondary: is_browser ? name_to_rgba(color, 0.5) : color
      };
    }
  }
  return result;
}
function merge_elements(value, merge_mode) {
  if (value.length === 0) return [];
  const result = [];
  let current_token = value[0].token;
  let current_class = value[0].class_or_confidence;
  for (let i = 1; i < value.length; i++) {
    const { token, class_or_confidence } = value[i];
    const should_merge = current_class === class_or_confidence;
    if (should_merge) {
      current_token += token;
    } else {
      result.push({ token: current_token, class_or_confidence: current_class });
      current_token = token;
      current_class = class_or_confidence;
    }
  }
  result.push({ token: current_token, class_or_confidence: current_class });
  return result;
}
function get_score_color(score) {
  if (score === null) return "";
  if (score < 0) {
    return `rgba(128, 90, 213, ${-score})`;
  }
  return `rgba(239, 68, 60, ${score})`;
}

var root$1 = from_html(`<input class="label-input svelte-14ic1z3"/>`);

function LabelInput($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 31, () => proxy([])),
		label_to_edit = prop($$props, 'label_to_edit', 31, () => -1),
		is_scores_mode = prop($$props, 'is_scores_mode', 3, false);

	let input_value = state(proxy($$props.category?.toString() ?? ""));

	function get_background_color() {
		if (is_scores_mode()) {
			const score = typeof $$props.category === "number"
				? $$props.category
				: parseFloat($$props.category ?? "0");

			return get_score_color(score);
		}

		if ($$props.category === null || $$props.active_legend && $$props.active_legend !== $$props.category) {
			return "";
		}

		return $$props.color_map[$$props.category]?.primary ?? "";
	}

	function update_value(e) {
		const target = e.target;
		const new_value = target.value.trim();

		value([
			...value().slice(0, $$props.label_index),
			{
				token: $$props.token,
				class_or_confidence: new_value === ""
					? null
					: is_scores_mode() ? Number(new_value) : new_value
			},
			...value().slice($$props.label_index + 1)
		]);

		$$props.onchange();
	}

	function handle_keydown(e) {
		if (e.key === "Enter") {
			update_value(e);
			label_to_edit(-1);
		}
	}

	var input = root$1();

	remove_input_defaults(input);
	autofocus(input, true);

	input.__input = (e) => {
		set(input_value, e.target.value, true);
	};

	input.__keydown = handle_keydown;

	let styles;

	template_effect(
		($0) => {
			set_attribute(input, 'type', is_scores_mode() ? "number" : "text");
			set_attribute(input, 'step', is_scores_mode() ? "0.1" : undefined);
			set_attribute(input, 'placeholder', is_scores_mode() ? undefined : "label");
			set_value(input, $$props.category);
			styles = set_style(input, '', styles, $0);
		},
		[
			() => ({
				'background-color': get_background_color(),
				width: is_scores_mode() ? "7ch" : `${(get(input_value)?.length || 4) + 4}ch`
			})
		]
	);

	event('blur', input, update_value);
	append($$anchor, input);
	pop();
}

delegate(['input', 'keydown']);

var root_3 = from_html(`<button class="legend-item svelte-1akrbo3"> </button>`);
var root_2 = from_html(`<div class="legend svelte-1akrbo3" data-testid="highlighted-text:category-legend"></div>`);
var root_7 = from_html(`<span class="label svelte-1akrbo3"> </span>`);
var root_9 = from_html(`<button class="remove-btn svelte-1akrbo3" aria-label="Remove label"><!></button>`);
var root_6 = from_html(`<span class="token-container svelte-1akrbo3"><span><span> </span> <!> <!></span> <!></span>`);
var root_10 = from_html(`<span class="line-break svelte-1akrbo3"></span>`);
var root_5 = from_html(`<!> <!>`, 1);
var root_1$1 = from_html(`<!> <div class="textfield svelte-1akrbo3"></div>`, 1);
var root_12 = from_html(`<div class="score-legend svelte-1akrbo3" data-testid="highlighted-text:color-legend"><span>-1</span> <span>0</span> <span>+1</span></div>`);
var root_15 = from_html(`<button class="remove-btn svelte-1akrbo3" aria-label="Remove label"><!></button>`);
var root_13 = from_html(`<span class="token-container svelte-1akrbo3"><span role="button"><span class="text svelte-1akrbo3"> </span> <!></span> <!></span>`);
var root_11 = from_html(`<!> <div class="textfield svelte-1akrbo3" data-testid="highlighted-text:textfield"></div>`, 1);
var root = from_html(`<div class="container svelte-1akrbo3"><!></div>`);

function HighlightedText($$anchor, $$props) {
	push($$props, true);

	const is_browser = typeof window !== "undefined";

	let value = prop($$props, 'value', 31, () => proxy([])),
		show_legend = prop($$props, 'show_legend', 3, false),
		show_inline_category = prop($$props, 'show_inline_category', 3, true),
		color_map = prop($$props, 'color_map', 19, () => ({})),
		show_whitespaces = prop($$props, 'show_whitespaces', 3, false),
		interactive = prop($$props, 'interactive', 3, false);

	let active_element_index = state(-1);
	let active_legend = state("");
	let label_to_edit = state(-1);
	let selection = state(null);
	let mode = state("categories");
	let resolved_color_map = state(proxy({}));

	user_effect(() => {
		let local_colors = { ...color_map() };

		for (const entry of value()) {
			if (entry.class_or_confidence === null) continue;

			if (typeof entry.class_or_confidence === "string") {
				set(mode, "categories");

				if (!(entry.class_or_confidence in local_colors)) {
					local_colors[entry.class_or_confidence] = get_next_color(Object.keys(local_colors).length);
				}
			} else {
				set(mode, "scores");
			}
		}

		set(resolved_color_map, generate_color_map(local_colors, is_browser), true);
	});

	onMount(() => {
		if (!interactive()) return;

		const on_mouse_up = () => {
			set(selection, window.getSelection(), true);
			handle_selection_complete();
			window.removeEventListener("mouseup", on_mouse_up);
		};

		window.addEventListener("mousedown", () => {
			window.addEventListener("mouseup", on_mouse_up);
		});
	});

	function handle_selection_complete() {
		if (!get(selection)) return;

		const text = get(selection).toString();

		if (!text) return;
		if (!show_whitespaces() && !text.trim()) return;

		const start = get(selection).getRangeAt(0).startOffset;
		const end = get(selection).getRangeAt(0).endOffset;

		handle_text_selected(start, end);
	}

	function handle_text_selected(start, end) {
		if (!get(selection)?.toString() || get(active_element_index) === -1 || !value()[get(active_element_index)].token.includes(get(selection).toString())) {
			return;
		}

		const str = value()[get(active_element_index)].token;

		const new_entries = [
			{ token: str.substring(0, start), class_or_confidence: null },
			{
				token: str.substring(start, end),
				class_or_confidence: get(mode) === "scores" ? 1 : "label"
			},
			{ token: str.substring(end), class_or_confidence: null }
		].filter((e) => show_whitespaces() ? e.token !== "" : e.token.trim() !== "");

		value([
			...value().slice(0, get(active_element_index)),
			...new_entries,
			...value().slice(get(active_element_index) + 1)
		]);

		set(label_to_edit, value().findIndex((v, i) => i >= get(active_element_index) && v.token === str.substring(start, end) && v.class_or_confidence !== null), true);
		handle_value_change();
		document.getElementById(`label-input-${get(label_to_edit)}`)?.focus();
	}

	function remove_highlight(index) {
		if (index < 0 || index >= value().length) return;

		value(value()[index].class_or_confidence = null, true);
		value(merge_elements(value()));
		handle_value_change();
		window.getSelection()?.empty();
	}

	function handle_value_change() {
		$$props.onchange?.(value());
		set(label_to_edit, -1);
	}

	function get_background_color(class_or_confidence) {
		if (class_or_confidence === null) return "";
		if (get(active_legend) && get(active_legend) !== class_or_confidence) return "";

		return get(resolved_color_map)[class_or_confidence]?.secondary ?? "";
	}

	function get_label_color(class_or_confidence) {
		if (class_or_confidence === null) return "";
		if (get(active_legend) && get(active_legend) !== class_or_confidence) return "";

		return get(resolved_color_map)[class_or_confidence]?.primary ?? "";
	}

	function get_text_color(class_or_confidence) {
		const bg = get_background_color(class_or_confidence);

		return is_transparent(bg) ? "" : "black";
	}

	var div = root();
	var node = child(div);

	{
		var consequent_6 = ($$anchor) => {
			var fragment = root_1$1();
			var node_1 = first_child(fragment);

			{
				var consequent = ($$anchor) => {
					var div_1 = root_2();

					each(div_1, 21, () => Object.entries(get(resolved_color_map)), index, ($$anchor, $$item) => {
						var $$array = user_derived(() => to_array(get($$item), 2));
						let category = () => get($$array)[0];
						let colors = () => get($$array)[1];
						var button = root_3();
						let styles;
						var text_1 = child(button, true);

						reset(button);

						template_effect(() => {
							styles = set_style(button, '', styles, { 'background-color': colors().secondary });
							set_text(text_1, category());
						});

						event('mouseenter', button, () => set(active_legend, category(), true));
						event('mouseleave', button, () => set(active_legend, ""));
						event('focus', button, () => set(active_legend, category(), true));
						event('blur', button, () => set(active_legend, ""));
						append($$anchor, button);
					});

					reset(div_1);
					append($$anchor, div_1);
				};

				if_block(node_1, ($$render) => {
					if (show_legend()) $$render(consequent);
				});
			}

			var div_2 = sibling(node_1, 2);

			each(div_2, 21, value, index, ($$anchor, $$item, i) => {
				let token = () => get($$item).token;
				let class_or_confidence = () => get($$item).class_or_confidence;
				const lines = user_derived(() => token().split("\n"));
				var fragment_1 = comment();
				var node_2 = first_child(fragment_1);

				each(node_2, 17, () => get(lines), index, ($$anchor, line, j) => {
					var fragment_2 = root_5();
					var node_3 = first_child(fragment_2);

					{
						var consequent_4 = ($$anchor) => {
							const bg_color = user_derived(() => get_background_color(class_or_confidence()));
							var span = root_6();
							var span_1 = child(span);
							let classes;

							span_1.__click = () => {
								if (class_or_confidence() === null) return;

								if (interactive()) {
									$$props.onselect?.({ index: i, value: [token(), class_or_confidence()] });
									set(label_to_edit, i, true);
								} else {
									$$props.onselect?.({ index: i, value: [token(), class_or_confidence()] });
								}
							};

							span_1.__keydown = (e) => {
								if (!interactive()) return;

								if (e.key === "Enter" && class_or_confidence() !== null) {
									set(label_to_edit, i, true);
								}
							};

							let styles_1;
							var span_2 = child(span_1);
							let classes_1;
							var text_2 = child(span_2, true);

							reset(span_2);

							var node_4 = sibling(span_2, 2);

							{
								var consequent_1 = ($$anchor) => {
									var span_3 = root_7();
									let styles_2;
									var text_3 = child(span_3, true);

									reset(span_3);

									template_effect(
										($0) => {
											styles_2 = set_style(span_3, '', styles_2, $0);
											set_text(text_3, class_or_confidence());
										},
										[
											() => ({ 'background-color': get_label_color(class_or_confidence()) })
										]
									);

									append($$anchor, span_3);
								};

								if_block(node_4, ($$render) => {
									if (!show_legend() && show_inline_category() && class_or_confidence() !== null && get(label_to_edit) !== i) $$render(consequent_1);
								});
							}

							var node_5 = sibling(node_4, 2);

							{
								var consequent_2 = ($$anchor) => {
									LabelInput($$anchor, {
										get category() {
											return class_or_confidence();
										},

										get active_legend() {
											return get(active_legend);
										},

										get color_map() {
											return get(resolved_color_map);
										},
										label_index: i,
										get token() {
											return token();
										},
										onchange: handle_value_change,
										get value() {
											return value();
										},

										set value($$value) {
											value($$value);
										},

										get label_to_edit() {
											return get(label_to_edit);
										},

										set label_to_edit($$value) {
											set(label_to_edit, $$value, true);
										}
									});
								};

								if_block(node_5, ($$render) => {
									if (interactive() && get(label_to_edit) === i && class_or_confidence() !== null) $$render(consequent_2);
								});
							}

							reset(span_1);

							var node_6 = sibling(span_1, 2);

							{
								var consequent_3 = ($$anchor) => {
									var button_1 = root_9();

									button_1.__click = () => remove_highlight(i);

									var node_7 = child(button_1);

									Clear(node_7);
									reset(button_1);
									append($$anchor, button_1);
								};

								if_block(node_6, ($$render) => {
									if (interactive() && class_or_confidence() !== null) $$render(consequent_3);
								});
							}

							reset(span);

							template_effect(
								($0, $1) => {
									classes = set_class(span_1, 1, 'token svelte-1akrbo3', null, classes, $0);
									set_attribute(span_1, 'role', class_or_confidence() !== null ? "button" : undefined);
									set_attribute(span_1, 'tabindex', class_or_confidence() !== null ? 0 : undefined);
									styles_1 = set_style(span_1, '', styles_1, $1);
									classes_1 = set_class(span_2, 1, 'text svelte-1akrbo3', null, classes_1, { unlabeled: class_or_confidence() === null });
									set_text(text_2, get(line));
								},
								[
									() => ({
										highlighted: class_or_confidence() !== null,
										transparent: class_or_confidence() !== null && is_transparent(get(bg_color)),
										dimmed: get(active_legend) && get(active_legend) !== class_or_confidence()
									}),

									() => ({
										'background-color': get(bg_color),
										color: get_text_color(class_or_confidence())
									})
								]
							);

							event('focus', span_1, () => set(active_element_index, i, true));
							event('mouseenter', span_1, () => set(active_element_index, i, true));
							append($$anchor, span);
						};

						if_block(node_3, ($$render) => {
							if (show_whitespaces() ? get(line) !== "" : get(line).trim()) $$render(consequent_4);
						});
					}

					var node_8 = sibling(node_3, 2);

					{
						var consequent_5 = ($$anchor) => {
							var span_4 = root_10();

							append($$anchor, span_4);
						};

						if_block(node_8, ($$render) => {
							if (j < get(lines).length - 1) $$render(consequent_5);
						});
					}

					append($$anchor, fragment_2);
				});

				append($$anchor, fragment_1);
			});

			reset(div_2);
			append($$anchor, fragment);
		};

		var alternate = ($$anchor) => {
			var fragment_4 = root_11();
			var node_9 = first_child(fragment_4);

			{
				var consequent_7 = ($$anchor) => {
					var div_3 = root_12();

					append($$anchor, div_3);
				};

				if_block(node_9, ($$render) => {
					if (show_legend()) $$render(consequent_7);
				});
			}

			var div_4 = sibling(node_9, 2);

			each(div_4, 21, value, index, ($$anchor, $$item, i) => {
				let token = () => get($$item).token;
				let class_or_confidence = () => get($$item).class_or_confidence;

				const score = user_derived(() => typeof class_or_confidence() === "string"
					? parseFloat(class_or_confidence())
					: class_or_confidence());

				var span_5 = root_13();
				var span_6 = child(span_5);
				let classes_2;

				set_attribute(span_6, 'tabindex', 0);

				span_6.__click = () => {
					if (interactive()) {
						set(label_to_edit, i, true);
					} else {
						$$props.onselect?.({ index: i, value: [token(), class_or_confidence()] });
					}
				};

				span_6.__keydown = (e) => {
					if (e.key === "Enter") {
						if (interactive()) {
							set(label_to_edit, i, true);
						} else {
							$$props.onselect?.({ index: i, value: [token(), class_or_confidence()] });
						}
					}
				};

				let styles_3;
				var span_7 = child(span_6);
				var text_4 = child(span_7, true);

				reset(span_7);

				var node_10 = sibling(span_7, 2);

				{
					var consequent_8 = ($$anchor) => {
						LabelInput($$anchor, {
							get category() {
								return class_or_confidence();
							},

							get active_legend() {
								return get(active_legend);
							},

							get color_map() {
								return get(resolved_color_map);
							},
							label_index: i,
							get token() {
								return token();
							},
							onchange: handle_value_change,
							is_scores_mode: true,
							get value() {
								return value();
							},

							set value($$value) {
								value($$value);
							},

							get label_to_edit() {
								return get(label_to_edit);
							},

							set label_to_edit($$value) {
								set(label_to_edit, $$value, true);
							}
						});
					};

					if_block(node_10, ($$render) => {
						if (interactive() && class_or_confidence() !== null && get(label_to_edit) === i) $$render(consequent_8);
					});
				}

				reset(span_6);

				var node_11 = sibling(span_6, 2);

				{
					var consequent_9 = ($$anchor) => {
						var button_2 = root_15();

						button_2.__click = () => remove_highlight(i);

						var node_12 = child(button_2);

						Clear(node_12);
						reset(button_2);
						append($$anchor, button_2);
					};

					if_block(node_11, ($$render) => {
						if (interactive() && class_or_confidence() !== null && get(active_element_index) === i) $$render(consequent_9);
					});
				}

				reset(span_5);

				template_effect(
					($0) => {
						classes_2 = set_class(span_6, 1, 'token score-token svelte-1akrbo3', null, classes_2, { highlighted: get(score) !== null });
						styles_3 = set_style(span_6, '', styles_3, $0);
						set_text(text_4, token());
					},
					[
						() => ({ 'background-color': get_score_color(get(score)) })
					]
				);

				event('mouseenter', span_6, () => set(active_element_index, i, true));
				event('focus', span_6, () => set(active_element_index, i, true));
				append($$anchor, span_5);
			});

			reset(div_4);
			append($$anchor, fragment_4);
		};

		if_block(node, ($$render) => {
			if (get(mode) === "categories") $$render(consequent_6); else $$render(alternate, false);
		});
	}

	reset(div);
	append($$anchor, div);
	pop();
}

delegate(['click', 'keydown']);

var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	let value = user_derived(() => gradio.props.combine_adjacent
		? merge_elements(gradio.props.value || [])
		: gradio.props.value);

	{
		let $0 = user_derived(() => gradio.shared.interactive ? "dashed" : "solid");

		Block($$anchor, {
			get variant() {
				return get($0);
			},
			test_id: 'highlighted-text',
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
			get container() {
				return gradio.shared.container;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get min_width() {
				return gradio.shared.min_width;
			},

			get rtl() {
				return gradio.props.rtl;
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
						onclearstatus: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
					}
				));

				var node_1 = sibling(node, 2);

				{
					var consequent = ($$anchor) => {
						IconButtonWrapper($$anchor, {
							get buttons() {
								return gradio.props.buttons;
							},
							on_custom_button_click: (id) => gradio.dispatch("custom_button_click", { id })
						});
					};

					if_block(node_1, ($$render) => {
						if (gradio.shared.interactive && gradio.shared.label && gradio.shared.show_label && gradio.props.buttons?.length) $$render(consequent);
					});
				}

				var node_2 = sibling(node_1, 2);

				{
					var consequent_1 = ($$anchor) => {
						{
							let $0 = user_derived(() => gradio.shared.label || gradio.i18n("highlighted_text.highlighted_text"));
							let $1 = user_derived(() => gradio.shared.container === false);

							BlockLabel($$anchor, {
								get Icon() {
									return TextHighlight;
								},

								get label() {
									return get($0);
								},
								float: false,
								get disable() {
									return get($1);
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get rtl() {
									return gradio.props.rtl;
								}
							});
						}
					};

					if_block(node_2, ($$render) => {
						if (gradio.shared.label && gradio.shared.show_label) $$render(consequent_1);
					});
				}

				var node_3 = sibling(node_2, 2);

				{
					var consequent_2 = ($$anchor) => {
						HighlightedText($$anchor, {
							get interactive() {
								return gradio.shared.interactive;
							},

							get show_legend() {
								return gradio.props.show_legend;
							},

							get show_inline_category() {
								return gradio.props.show_inline_category;
							},

							get show_whitespaces() {
								return gradio.props.show_whitespaces;
							},

							get color_map() {
								return gradio.props.color_map;
							},
							onselect: (detail) => gradio.dispatch("select", detail),
							onchange: () => {
								gradio.props.value = get(value);
								gradio.dispatch("change");
							},

							get value() {
								return get(value);
							},

							set value($$value) {
								set(value, $$value);
							}
						});
					};

					var alternate = ($$anchor) => {
						{
							let $0 = user_derived(() => gradio.shared.interactive ? "small" : "large");

							Empty($$anchor, {
								get size() {
									return get($0);
								},

								get unpadded_box() {
									return gradio.shared.interactive;
								},

								children: ($$anchor, $$slotProps) => {
									TextHighlight($$anchor);
								},
								$$slots: { default: true }
							});
						}
					};

					if_block(node_3, ($$render) => {
						if (get(value)) $$render(consequent_2); else $$render(alternate, false);
					});
				}

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { HighlightedText as BaseHighlightedText, Index as default };
//# sourceMappingURL=Index-DS7IaHHH.js.map
