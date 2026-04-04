import { p as prop, i as if_block, b as set_class, d as bind_this, a as set_attribute, f as set_style, t as remove_input_defaults } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, x as set, ai as user_pre_effect, w as get, S as first_child, t as template_effect, a as append, T as pop, u as state, M as tick, X as sibling, a5 as user_derived, W as from_html, aj as delegate, V as child, a0 as set_text, a6 as comment, a8 as next, Y as reset, ak as remove_textarea_child, al as autofocus, A as effect, Z as event, a7 as text } from './index-BvBk1Iap.js';
import { a as action } from './actions-Cit_UA5S.js';
import { b as bind_value } from './input-C317uaXR.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { S as Send } from './Send-Ee555fP_.js';
import { S as Square } from './Square-B6INQV79.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';

var root_5 = from_html(`<div class="validation-error svelte-1hguek3"> </div>`);
var root_4 = from_html(` <!>`, 1);
var root_7 = from_html(`<input data-testid="textbox" type="text"/>`);
var root_9 = from_html(`<input data-testid="password" type="password" autocomplete=""/>`);
var root_11 = from_html(`<input data-testid="textbox" type="email" autocomplete="email"/>`);
var root_12 = from_html(`<textarea data-testid="textbox"></textarea>`);
var root_13 = from_html(`<button data-testid="submit-button"><!></button>`);
var root_16 = from_html(`<button data-testid="stop-button"><!></button>`);
var root = from_html(`<label><!> <!> <div class="input-container svelte-1hguek3"><!> <!> <!></div></label>`);

function Textbox($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, ""),
		value_is_output = prop($$props, 'value_is_output', 15, false),
		lines = prop($$props, 'lines', 3, 1),
		placeholder = prop($$props, 'placeholder', 3, ""),
		info = prop($$props, 'info', 3, undefined),
		disabled = prop($$props, 'disabled', 3, false),
		show_label = prop($$props, 'show_label', 3, true),
		container = prop($$props, 'container', 3, true),
		max_lines = prop($$props, 'max_lines', 3, undefined),
		type = prop($$props, 'type', 3, "text"),
		buttons = prop($$props, 'buttons', 3, null),
		oncustombuttonclick = prop($$props, 'oncustombuttonclick', 3, null),
		submit_btn = prop($$props, 'submit_btn', 3, null),
		stop_btn = prop($$props, 'stop_btn', 3, null),
		rtl = prop($$props, 'rtl', 3, false),
		autofocus$1 = prop($$props, 'autofocus', 3, false),
		text_align = prop($$props, 'text_align', 3, undefined),
		autoscroll = prop($$props, 'autoscroll', 3, true),
		max_length = prop($$props, 'max_length', 3, undefined),
		html_attributes = prop($$props, 'html_attributes', 3, null),
		validation_error = prop($$props, 'validation_error', 3, undefined);

	// svelte-ignore non_reactive_update
	let el;

	let copied = state(false);
	let timer;
	let can_scroll = state(false);
	let previous_scroll_top = state(0);
	let user_has_scrolled_up = state(false);
	let _max_lines = state(1);

	// svelte-ignore state_referenced_locally
	const show_textbox_border = !submit_btn();

	user_effect(() => {
		if (max_lines() === undefined || max_lines() === null) {
			if (type() === "text") {
				set(_max_lines, Math.max(lines(), 20), true);
			} else {
				set(_max_lines, 1);
			}
		} else {
			set(_max_lines, Math.max(max_lines(), lines()), true);
		}
	});

	user_effect(() => {
		value();
		validation_error();

		if (el && lines() !== get(_max_lines) && get(_max_lines) > 1) {
			resize({ target: el });
		}
	});

	user_effect(() => {
		if (value() === null) value("");
	});

	user_pre_effect(() => {
		if (!get(user_has_scrolled_up) && el && el.offsetHeight + el.scrollTop > el.scrollHeight - 100) {
			set(can_scroll, true);
		}
	});

	const scroll = () => {
		if (get(can_scroll) && autoscroll() && !get(user_has_scrolled_up)) {
			el.scrollTo(0, el.scrollHeight);
		}
	};

	async function handle_change() {
		await tick();
		$$props.onchange?.(value());
	}

	user_effect(() => {
		if (autofocus$1() && el) {
			el.focus();
		}

		if (get(can_scroll) && autoscroll()) {
			scroll();
		}

		value_is_output(false);
	});

	user_effect(() => {
		value();
		handle_change();
	});

	async function handle_copy() {
		if ("clipboard" in navigator) {
			try {
				await navigator.clipboard.writeText(value());
				$$props.oncopy?.({ value: value() });
			} catch(e) {
				console.error("COPYING CLIPBOARD FAILED", e);
			}

			copy_feedback();
		}
	}

	function copy_feedback() {
		set(copied, true);

		if (timer) clearTimeout(timer);

		timer = setTimeout(
			() => {
				set(copied, false);
			},
			1000
		);
	}

	function handle_select(event) {
		const target = event.target;
		const text = target.value;
		const index = [target.selectionStart, target.selectionEnd];

		$$props.onselect?.({ value: text.substring(...index), index });
	}

	async function handle_keypress(e) {
		if (e.key === "Enter" && e.shiftKey && lines() > 1) {
			e.preventDefault();
			await tick();
			$$props.onsubmit?.();
		} else if (e.key === "Enter" && !e.shiftKey && lines() === 1 && get(_max_lines) >= 1) {
			e.preventDefault();
			await tick();
			$$props.onsubmit?.();
		}

		await tick();
		$$props.oninput?.(value());
	}

	function handle_scroll(event) {
		const target = event.target;
		const current_scroll_top = target.scrollTop;

		if (current_scroll_top < get(previous_scroll_top)) {
			set(user_has_scrolled_up, true);
		}

		set(previous_scroll_top, current_scroll_top, true);

		const max_scroll_top = target.scrollHeight - target.clientHeight;
		const user_has_scrolled_to_bottom = current_scroll_top >= max_scroll_top;

		if (user_has_scrolled_to_bottom) {
			set(user_has_scrolled_up, false);
		}
	}

	function handle_stop() {
		$$props.onstop?.();
	}

	function handle_submit() {
		$$props.onsubmit?.();
	}

	async function resize(event) {
		await tick();

		if (lines() === get(_max_lines)) return;

		const target = event.target;
		const computed_styles = window.getComputedStyle(target);
		const padding_top = parseFloat(computed_styles.paddingTop);
		const padding_bottom = parseFloat(computed_styles.paddingBottom);
		const line_height = parseFloat(computed_styles.lineHeight);

		let max = get(_max_lines) === undefined
			? false
			: padding_top + padding_bottom + line_height * get(_max_lines);

		let min = padding_top + padding_bottom + lines() * line_height;

		target.style.height = "1px";

		let scroll_height;

		if (max && target.scrollHeight > max) {
			scroll_height = max;
		} else if (target.scrollHeight < min) {
			scroll_height = min;
		} else {
			scroll_height = target.scrollHeight;
		}

		target.style.height = `${scroll_height}px`;
		update_scrollbar_visibility(target);
	}

	function update_scrollbar_visibility(textarea) {
		// Using "auto" scroll does not work, as the scrollbar is visible even
		// when the content is about the same height as the textarea height. So
		// here, we add the scrollbar if the content is longer than a threshold
		// of 1 line height beyond the textarea height.
		const content_height = textarea.scrollHeight;

		const visible_height = textarea.clientHeight;
		const line_height = parseFloat(window.getComputedStyle(textarea).lineHeight);

		if (content_height > visible_height + line_height) {
			textarea.style.overflowY = "scroll";
		} else {
			textarea.style.overflowY = "hidden";
		}
	}

	function text_area_resize(_el, _value) {
		if (lines() === get(_max_lines) || lines() === 1 && get(_max_lines) === 1) return;

		_el.addEventListener("input", resize);

		if (_value.trim()) {
			_el.style.overflowY = "scroll";
			resize({ target: _el });
		}

		return { destroy: () => _el.removeEventListener("input", resize) };
	}

	var label_1 = root();
	let classes;
	var node = child(label_1);

	{
		var consequent_1 = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				get buttons() {
					return buttons();
				},

				get on_custom_button_click() {
					return oncustombuttonclick();
				},

				children: ($$anchor, $$slotProps) => {
					var fragment_1 = comment();
					var node_1 = first_child(fragment_1);

					{
						var consequent = ($$anchor) => {
							{
								let $0 = user_derived(() => get(copied) ? Check : Copy);
								let $1 = user_derived(() => get(copied) ? "Copied" : "Copy");

								IconButton($$anchor, {
									get Icon() {
										return get($0);
									},
									onclick: handle_copy,
									get label() {
										return get($1);
									}
								});
							}
						};

						if_block(node_1, ($$render) => {
							if (buttons().some((btn) => typeof btn === "string" && btn === "copy")) $$render(consequent);
						});
					}

					append($$anchor, fragment_1);
				},
				$$slots: { default: true }
			});
		};

		if_block(node, ($$render) => {
			if (show_label() && buttons() && buttons().length > 0) $$render(consequent_1);
		});
	}

	var node_2 = sibling(node, 2);

	{
		let $0 = user_derived(() => validation_error() ? true : show_label());

		BlockTitle(node_2, {
			get show_label() {
				return get($0);
			},

			get info() {
				return info();
			},

			children: ($$anchor, $$slotProps) => {
				next();

				var fragment_3 = root_4();
				var text_1 = first_child(fragment_3);
				var node_3 = sibling(text_1);

				{
					var consequent_2 = ($$anchor) => {
						var div = root_5();
						var text_2 = child(div, true);

						reset(div);
						template_effect(() => set_text(text_2, validation_error()));
						append($$anchor, div);
					};

					if_block(node_3, ($$render) => {
						if (validation_error()) $$render(consequent_2);
					});
				}

				template_effect(() => set_text(text_1, `${$$props.label ?? ''} `));
				append($$anchor, fragment_3);
			},
			$$slots: { default: true }
		});
	}

	var div_1 = sibling(node_2, 2);
	var node_4 = child(div_1);

	{
		var consequent_6 = ($$anchor) => {
			var fragment_4 = comment();
			var node_5 = first_child(fragment_4);

			{
				var consequent_3 = ($$anchor) => {
					var input = root_7();

					remove_input_defaults(input);

					let classes_1;

					autofocus(input, autofocus$1());
					bind_this(input, ($$value) => el = $$value, () => el);

					template_effect(() => {
						classes_1 = set_class(input, 1, 'scroll-hide svelte-1hguek3', null, classes_1, { 'validation-error': validation_error() });
						set_attribute(input, 'dir', rtl() ? "rtl" : "ltr");
						set_attribute(input, 'placeholder', placeholder());
						input.disabled = disabled();
						set_attribute(input, 'maxlength', max_length());
						set_style(input, text_align() ? "text-align: " + text_align() : "");
						set_attribute(input, 'autocapitalize', html_attributes()?.autocapitalize);
						set_attribute(input, 'autocorrect', html_attributes()?.autocorrect);
						set_attribute(input, 'spellcheck', html_attributes()?.spellcheck);
						set_attribute(input, 'autocomplete', html_attributes()?.autocomplete);
						set_attribute(input, 'tabindex', html_attributes()?.tabindex);
						set_attribute(input, 'enterkeyhint', html_attributes()?.enterkeyhint);
						set_attribute(input, 'lang', html_attributes()?.lang);
						input.dir = input.dir;
					});

					event('keypress', input, handle_keypress);
					event('blur', input, () => $$props.onblur?.());
					event('select', input, handle_select);
					event('focus', input, () => $$props.onfocus?.());
					bind_value(input, value);
					append($$anchor, input);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_5 = comment();
					var node_6 = first_child(fragment_5);

					{
						var consequent_4 = ($$anchor) => {
							var input_1 = root_9();

							remove_input_defaults(input_1);

							let classes_2;

							autofocus(input_1, autofocus$1());
							bind_this(input_1, ($$value) => el = $$value, () => el);

							template_effect(() => {
								classes_2 = set_class(input_1, 1, 'scroll-hide svelte-1hguek3', null, classes_2, { 'validation-error': validation_error() });
								set_attribute(input_1, 'placeholder', placeholder());
								input_1.disabled = disabled();
								set_attribute(input_1, 'maxlength', max_length());
								set_attribute(input_1, 'autocapitalize', html_attributes()?.autocapitalize);
								set_attribute(input_1, 'autocorrect', html_attributes()?.autocorrect);
								set_attribute(input_1, 'spellcheck', html_attributes()?.spellcheck);
								set_attribute(input_1, 'tabindex', html_attributes()?.tabindex);
								set_attribute(input_1, 'enterkeyhint', html_attributes()?.enterkeyhint);
								set_attribute(input_1, 'lang', html_attributes()?.lang);
							});

							event('keypress', input_1, handle_keypress);
							event('blur', input_1, () => $$props.onblur?.());
							event('select', input_1, handle_select);
							event('focus', input_1, () => $$props.onfocus?.());
							bind_value(input_1, value);
							append($$anchor, input_1);
						};

						var alternate = ($$anchor) => {
							var fragment_6 = comment();
							var node_7 = first_child(fragment_6);

							{
								var consequent_5 = ($$anchor) => {
									var input_2 = root_11();

									remove_input_defaults(input_2);

									let classes_3;

									autofocus(input_2, autofocus$1());
									bind_this(input_2, ($$value) => el = $$value, () => el);

									template_effect(() => {
										classes_3 = set_class(input_2, 1, 'scroll-hide svelte-1hguek3', null, classes_3, { 'validation-error': validation_error() });
										set_attribute(input_2, 'placeholder', placeholder());
										input_2.disabled = disabled();
										set_attribute(input_2, 'maxlength', max_length());
										set_attribute(input_2, 'autocapitalize', html_attributes()?.autocapitalize);
										set_attribute(input_2, 'autocorrect', html_attributes()?.autocorrect);
										set_attribute(input_2, 'spellcheck', html_attributes()?.spellcheck);
										set_attribute(input_2, 'tabindex', html_attributes()?.tabindex);
										set_attribute(input_2, 'enterkeyhint', html_attributes()?.enterkeyhint);
										set_attribute(input_2, 'lang', html_attributes()?.lang);
									});

									event('keypress', input_2, handle_keypress);
									event('blur', input_2, () => $$props.onblur?.());
									event('select', input_2, handle_select);
									event('focus', input_2, () => $$props.onfocus?.());
									bind_value(input_2, value);
									append($$anchor, input_2);
								};

								if_block(
									node_7,
									($$render) => {
										if (type() === "email") $$render(consequent_5);
									},
									true
								);
							}

							append($$anchor, fragment_6);
						};

						if_block(
							node_6,
							($$render) => {
								if (type() === "password") $$render(consequent_4); else $$render(alternate, false);
							},
							true
						);
					}

					append($$anchor, fragment_5);
				};

				if_block(node_5, ($$render) => {
					if (type() === "text") $$render(consequent_3); else $$render(alternate_1, false);
				});
			}

			append($$anchor, fragment_4);
		};

		var alternate_2 = ($$anchor) => {
			var textarea_1 = root_12();

			remove_textarea_child(textarea_1);
			autofocus(textarea_1, autofocus$1());

			let classes_4;

			action(textarea_1, ($$node, $$action_arg) => text_area_resize?.($$node, $$action_arg), value);
			effect(() => bind_value(textarea_1, value));
			bind_this(textarea_1, ($$value) => el = $$value, () => el);

			template_effect(() => {
				set_attribute(textarea_1, 'dir', rtl() ? "rtl" : "ltr");
				set_attribute(textarea_1, 'placeholder', placeholder());
				set_attribute(textarea_1, 'rows', lines());
				textarea_1.disabled = disabled();
				set_attribute(textarea_1, 'maxlength', max_length());
				set_style(textarea_1, text_align() ? "text-align: " + text_align() : "");
				set_attribute(textarea_1, 'autocapitalize', html_attributes()?.autocapitalize);
				set_attribute(textarea_1, 'autocorrect', html_attributes()?.autocorrect);
				set_attribute(textarea_1, 'spellcheck', html_attributes()?.spellcheck);
				set_attribute(textarea_1, 'autocomplete', html_attributes()?.autocomplete);
				set_attribute(textarea_1, 'tabindex', html_attributes()?.tabindex);
				set_attribute(textarea_1, 'enterkeyhint', html_attributes()?.enterkeyhint);
				set_attribute(textarea_1, 'lang', html_attributes()?.lang);

				classes_4 = set_class(textarea_1, 1, 'svelte-1hguek3', null, classes_4, {
					'no-label': !show_label() && (submit_btn() || stop_btn()),
					'validation-error': validation_error()
				});

				textarea_1.dir = textarea_1.dir;
			});

			event('keypress', textarea_1, handle_keypress);
			event('blur', textarea_1, () => $$props.onblur?.());
			event('select', textarea_1, handle_select);
			event('focus', textarea_1, () => $$props.onfocus?.());
			event('scroll', textarea_1, handle_scroll);
			append($$anchor, textarea_1);
		};

		if_block(node_4, ($$render) => {
			if (lines() === 1 && get(_max_lines) === 1) $$render(consequent_6); else $$render(alternate_2, false);
		});
	}

	var node_8 = sibling(node_4, 2);

	{
		var consequent_8 = ($$anchor) => {
			var button = root_13();
			let classes_5;

			button.__click = handle_submit;

			var node_9 = child(button);

			{
				var consequent_7 = ($$anchor) => {
					Send($$anchor);
				};

				var alternate_3 = ($$anchor) => {
					var text_3 = text();

					template_effect(() => set_text(text_3, submit_btn()));
					append($$anchor, text_3);
				};

				if_block(node_9, ($$render) => {
					if (submit_btn() === true) $$render(consequent_7); else $$render(alternate_3, false);
				});
			}

			reset(button);
			template_effect(() => classes_5 = set_class(button, 1, 'submit-button svelte-1hguek3', null, classes_5, { 'padded-button': submit_btn() !== true }));
			append($$anchor, button);
		};

		if_block(node_8, ($$render) => {
			if (submit_btn()) $$render(consequent_8);
		});
	}

	var node_10 = sibling(node_8, 2);

	{
		var consequent_10 = ($$anchor) => {
			var button_1 = root_16();
			let classes_6;

			button_1.__click = handle_stop;

			var node_11 = child(button_1);

			{
				var consequent_9 = ($$anchor) => {
					Square($$anchor, { fill: 'none', stroke_width: 2.5 });
				};

				var alternate_4 = ($$anchor) => {
					var text_4 = text();

					template_effect(() => set_text(text_4, stop_btn()));
					append($$anchor, text_4);
				};

				if_block(node_11, ($$render) => {
					if (stop_btn() === true) $$render(consequent_9); else $$render(alternate_4, false);
				});
			}

			reset(button_1);
			template_effect(() => classes_6 = set_class(button_1, 1, 'stop-button svelte-1hguek3', null, classes_6, { 'padded-button': stop_btn() !== true }));
			append($$anchor, button_1);
		};

		if_block(node_10, ($$render) => {
			if (stop_btn()) $$render(consequent_10);
		});
	}

	reset(div_1);
	reset(label_1);
	template_effect(() => classes = set_class(label_1, 1, 'svelte-1hguek3', null, classes, { container: container(), show_textbox_border }));
	append($$anchor, label_1);
	pop();
}

delegate(['click']);

export { Textbox as T };
//# sourceMappingURL=Textbox-v-cNeNHQ.js.map
