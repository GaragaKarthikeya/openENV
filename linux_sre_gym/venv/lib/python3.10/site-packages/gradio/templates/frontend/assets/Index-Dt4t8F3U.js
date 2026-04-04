import { p as prop, i as if_block, f as set_style, d as bind_this, t as remove_input_defaults, k as each, b as set_class, u as index, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, y as user_effect, Z as event, S as first_child, a7 as text, t as template_effect, a as append, x as set, w as get, T as pop, u as state, M as tick, aw as $window, X as sibling, W as from_html, aj as delegate, a0 as set_text, a8 as next, V as child, Y as reset, a5 as user_derived, at as to_array, v as proxy } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { a as action } from './actions-Cit_UA5S.js';
import { b as bind_value } from './input-C317uaXR.js';
import { t as tinycolor, E as Eyedropper } from './tinycolor-NLEY4io3.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
export { default as BaseExample } from './Example-CCEyD4GV.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

function click_outside(node, callback) {
  const handle_click = (event) => {
    if (node && !node.contains(event.target) && !event.defaultPrevented) {
      callback(event);
    }
  };
  document.addEventListener("mousedown", handle_click, true);
  return {
    destroy() {
      document.removeEventListener("mousedown", handle_click, true);
    }
  };
}

function hsva_to_rgba(hsva) {
  const saturation = hsva.s;
  const value = hsva.v;
  let chroma = saturation * value;
  const hue_by_60 = hsva.h / 60;
  let x = chroma * (1 - Math.abs(hue_by_60 % 2 - 1));
  const m = value - chroma;
  chroma = chroma + m;
  x = x + m;
  const index = Math.floor(hue_by_60) % 6;
  const red = [chroma, x, m, m, x, chroma][index];
  const green = [x, chroma, chroma, x, m, m][index];
  const blue = [m, m, x, chroma, chroma, x][index];
  return tinycolor({
    r: red * 255,
    g: green * 255,
    b: blue * 255,
    a: hsva.a
  }).toHexString();
}
function normalize_color(color) {
  const tc = tinycolor(color);
  if (tc.isValid()) {
    return tc.toHexString();
  }
  return color;
}
function format_color(color, mode) {
  if (mode === "hex") {
    return tinycolor(color).toHexString();
  } else if (mode === "rgb") {
    return tinycolor(color).toRgbString();
  }
  return tinycolor(color).toHslString();
}

var root_4 = from_html(`<button> </button>`);
var root_2 = from_html(`<div class="color-picker svelte-nbn1m9"><div class="color-gradient svelte-nbn1m9"><div class="marker svelte-nbn1m9"></div></div> <div class="hue-slider svelte-nbn1m9"><div class="marker svelte-nbn1m9"></div></div> <div class="input svelte-nbn1m9"><button class="swatch svelte-nbn1m9"></button> <div><div class="input-wrap svelte-nbn1m9"><input type="text" class="svelte-nbn1m9"/> <button class="eyedropper svelte-nbn1m9"><!></button></div> <div class="buttons svelte-nbn1m9"></div></div></div></div>`);
var root = from_html(`<!> <button class="dialog-button svelte-nbn1m9"></button> <!>`, 1);

function Colorpicker($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15),
		on_input = prop($$props, 'on_input', 3, () => {}),
		on_submit = prop($$props, 'on_submit', 3, () => {}),
		on_blur = prop($$props, 'on_blur', 3, () => {}),
		on_focus = prop($$props, 'on_focus', 3, () => {});

	let dialog_open = state(false);
	let current_mode = state("hex");
	let eyedropper_supported = false;
	let sl_wrap;
	let hue_wrap;
	let sl_marker_pos = [0, 0];
	let sl_rect = null;
	let sl_moving = false;
	let sl = [0, 0];
	let hue = 0;
	let hue_marker_pos = 0;
	let hue_rect = null;
	let hue_moving = false;

	function handle_hue_down(event) {
		hue_rect = event.currentTarget.getBoundingClientRect();
		hue_moving = true;
		update_hue_from_mouse(event.clientX);
	}

	function update_hue_from_mouse(x) {
		if (!hue_rect) return;

		const _x = Math.max(0, Math.min(x - hue_rect.left, hue_rect.width)); // Get the x-coordinate relative to the box

		hue_marker_pos = _x;

		const _hue = _x / hue_rect.width * 360; // Scale the x position to a hue value (0-360)

		hue = _hue;
		value(hsva_to_rgba({ h: _hue, s: sl[0], v: sl[1], a: 1 }));
		on_input()();
	}

	function update_color_from_mouse(x, y) {
		if (!sl_rect) return;

		const _x = Math.max(0, Math.min(x - sl_rect.left, sl_rect.width));
		const _y = Math.max(0, Math.min(y - sl_rect.top, sl_rect.height));

		sl_marker_pos = [_x, _y];

		const _hsva = {
			h: hue * 1,
			s: _x / sl_rect.width,
			v: 1 - _y / sl_rect.height,
			a: 1
		};

		sl = [_hsva.s, _hsva.v];
		value(hsva_to_rgba(_hsva));
		on_input()();
	}

	function handle_sl_down(event) {
		sl_moving = true;
		sl_rect = event.currentTarget.getBoundingClientRect();
		update_color_from_mouse(event.clientX, event.clientY);
	}

	function handle_move(event) {
		if (sl_moving) update_color_from_mouse(event.clientX, event.clientY);
		if (hue_moving) update_hue_from_mouse(event.clientX);
	}

	function handle_end() {
		sl_moving = false;
		hue_moving = false;
	}

	async function update_mouse_from_color(color) {
		if (sl_moving || hue_moving) return;

		await tick();

		if (!color) return;

		if (!sl_rect && sl_wrap) {
			sl_rect = sl_wrap.getBoundingClientRect();
		}

		if (!hue_rect && hue_wrap) {
			hue_rect = hue_wrap.getBoundingClientRect();
		}

		// Exit if we still don't have valid rectangles
		if (!sl_rect || !hue_rect) return;

		const hsva = tinycolor(color).toHsv();
		const _x = hsva.s * sl_rect.width;
		const _y = (1 - hsva.v) * sl_rect.height;

		sl_marker_pos = [_x, _y];
		sl = [hsva.s, hsva.v];
		hue = hsva.h;
		hue_marker_pos = hsva.h / 360 * hue_rect.width;
	}

	function request_eyedropper() {
		// @ts-ignore
		const eyeDropper = new EyeDropper();

		eyeDropper.open().then((result) => {
			value(result.sRGBHex);
		});

		on_input()();
	}

	const modes = [["Hex", "hex"], ["RGB", "rgb"], ["HSL", "hsl"]];
	let color_string = user_derived(() => format_color(value(), get(current_mode)));

	onMount(async () => {
		// @ts-ignore
		eyedropper_supported = window !== undefined && !!window.EyeDropper;
	});

	function handle_click_outside() {
		set(dialog_open, false);
	}

	user_effect(() => {
		update_mouse_from_color(value());
	});

	var fragment = root();

	event('mousemove', $window, handle_move);
	event('mouseup', $window, handle_end);

	var node = first_child(fragment);

	BlockTitle(node, {
		get show_label() {
			return $$props.show_label;
		},

		get info() {
			return $$props.info;
		},

		children: ($$anchor, $$slotProps) => {
			next();

			var text$1 = text();

			template_effect(() => set_text(text$1, $$props.label));
			append($$anchor, text$1);
		},
		$$slots: { default: true }
	});

	var button = sibling(node, 2);

	button.__click = () => {
		update_mouse_from_color(value());
		set(dialog_open, !get(dialog_open));
	};

	let styles;
	var node_1 = sibling(button, 2);

	{
		var consequent_1 = ($$anchor) => {
			var div = root_2();
			var div_1 = child(div);

			div_1.__mousedown = handle_sl_down;

			var div_2 = child(div_1);
			let styles_1;

			reset(div_1);
			bind_this(div_1, ($$value) => sl_wrap = $$value, () => sl_wrap);

			var div_3 = sibling(div_1, 2);

			div_3.__mousedown = handle_hue_down;

			var div_4 = child(div_3);
			let styles_2;

			reset(div_3);
			bind_this(div_3, ($$value) => hue_wrap = $$value, () => hue_wrap);

			var div_5 = sibling(div_3, 2);
			var button_1 = child(div_5);
			let styles_3;
			var div_6 = sibling(button_1, 2);
			var div_7 = child(div_6);
			var input = child(div_7);

			remove_input_defaults(input);

			input.__change = (e) => {
				value(normalize_color(e.currentTarget.value));
			};

			input.__keydown = (e) => {
				if (e.key === "Enter") {
					on_submit()();
				}
			};

			var button_2 = sibling(input, 2);

			button_2.__click = request_eyedropper;

			var node_2 = child(button_2);

			{
				var consequent = ($$anchor) => {
					Eyedropper($$anchor);
				};

				if_block(node_2, ($$render) => {
					if (eyedropper_supported) $$render(consequent);
				});
			}

			reset(button_2);
			reset(div_7);

			var div_8 = sibling(div_7, 2);

			each(div_8, 21, () => modes, index, ($$anchor, $$item, $$index, $$array) => {
				var $$array_1 = user_derived(() => to_array(get($$item), 2));
				let label = () => get($$array_1)[0];
				let value = () => get($$array_1)[1];
				var button_3 = root_4();
				let classes;

				button_3.__click = () => {
					set(current_mode, value(), true);
				};

				var text_1 = child(button_3, true);

				reset(button_3);

				template_effect(() => {
					classes = set_class(button_3, 1, 'button svelte-nbn1m9', null, classes, { active: get(current_mode) === value() });
					set_text(text_1, label());
				});

				append($$anchor, button_3);
			});

			reset(div_8);
			reset(div_6);
			reset(div_5);
			reset(div);
			action(div, ($$node, $$action_arg) => click_outside?.($$node, $$action_arg), () => handle_click_outside);

			template_effect(() => {
				set_style(div_1, `--hue:${hue ?? ''}`);

				styles_1 = set_style(div_2, '', styles_1, {
					transform: `translate(${sl_marker_pos[0] ?? ''}px,${sl_marker_pos[1] ?? ''}px)`,
					background: value()
				});

				styles_2 = set_style(div_4, '', styles_2, {
					background: "hsl(" + hue + ", 100%, 50%)",
					transform: `translateX(${hue_marker_pos ?? ''}px)`
				});

				styles_3 = set_style(button_1, '', styles_3, { background: value() });
			});

			bind_value(input, () => get(color_string), ($$value) => set(color_string, $$value));
			append($$anchor, div);
		};

		if_block(node_1, ($$render) => {
			if (get(dialog_open)) $$render(consequent_1);
		});
	}

	template_effect(() => {
		button.disabled = $$props.disabled;
		styles = set_style(button, '', styles, { background: value() });
	});

	event('focus', button, function (...$$args) {
		on_focus()?.apply(this, $$args);
	});

	event('blur', button, function (...$$args) {
		on_blur()?.apply(this, $$args);
	});

	append($$anchor, fragment);
	pop();
}

delegate(['click', 'mousedown', 'change', 'keydown']);

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props, { value: "#000000" });
	let old_value = state(proxy(gradio.props.value));
	let label = user_derived(() => gradio.shared.label || gradio.i18n("color_picker.color_picker"));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

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

		get container() {
			return gradio.shared.container;
		},

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
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}
			));

			var node_1 = sibling(node, 2);

			{
				let $0 = user_derived(() => !gradio.shared.interactive);

				Colorpicker(node_1, {
					get label() {
						return get(label);
					},

					get info() {
						return gradio.props.info;
					},

					get show_label() {
						return gradio.shared.show_label;
					},

					get disabled() {
						return get($0);
					},
					on_input: () => gradio.dispatch("input"),
					on_submit: () => gradio.dispatch("submit"),
					on_blur: () => gradio.dispatch("blur"),
					on_focus: () => gradio.dispatch("focus"),
					get value() {
						return gradio.props.value;
					},

					set value($$value) {
						gradio.props.value = $$value;
					}
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Colorpicker as BaseColorPicker, Index as default };
//# sourceMappingURL=Index-Dt4t8F3U.js.map
