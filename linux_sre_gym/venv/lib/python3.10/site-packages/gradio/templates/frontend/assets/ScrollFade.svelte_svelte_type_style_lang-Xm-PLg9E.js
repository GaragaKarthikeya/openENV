import { B as BranchManager, p as prop, i as if_block, b as set_class, a as set_attribute, f as set_style } from './i18n-CMo5lpzy.js';
import { p as block, E as EFFECT_TRANSPARENT, h as hydrating, e as hydrate_next, t as template_effect, a as append, X as sibling, W as from_html, V as child, aj as delegate, w as get, a5 as user_derived, a0 as set_text, a6 as comment, S as first_child, Y as reset, f as from_svg, R as push, T as pop } from './index-BvBk1Iap.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';

/** @import { TemplateNode, Dom } from '#client' */

/**
 * @template P
 * @template {(props: P) => void} C
 * @param {TemplateNode} node
 * @param {() => C} get_component
 * @param {(anchor: TemplateNode, component: C) => Dom | void} render_fn
 * @returns {void}
 */
function component(node, get_component, render_fn) {
	if (hydrating) {
		hydrate_next();
	}

	var branches = new BranchManager(node);

	block(() => {
		var component = get_component() ?? null;
		branches.ensure(component, component && ((target) => render_fn(target, component)));
	}, EFFECT_TRANSPARENT);
}

var root_1$1 = from_html(`<span class="svelte-3jwzs9"> </span>`);
var root$5 = from_html(`<button><!> <div><!> <!></div></button>`);

function IconButton($$anchor, $$props) {
	let label = prop($$props, 'label', 3, ""),
		show_label = prop($$props, 'show_label', 3, false),
		pending = prop($$props, 'pending', 3, false),
		size = prop($$props, 'size', 3, "small"),
		padded = prop($$props, 'padded', 3, true),
		highlight = prop($$props, 'highlight', 3, false),
		disabled = prop($$props, 'disabled', 3, false),
		hasPopup = prop($$props, 'hasPopup', 3, false),
		color = prop($$props, 'color', 3, "var(--block-label-text-color)"),
		transparent = prop($$props, 'transparent', 3, false),
		background = prop($$props, 'background', 3, "var(--block-background-fill)"),
		border = prop($$props, 'border', 3, "transparent");

	let _color = user_derived(() => highlight() ? "var(--color-accent)" : color());
	var button = root$5();
	let classes;

	button.__click = function (...$$args) {
		$$props.onclick?.apply(this, $$args);
	};

	let styles;
	var node = child(button);

	{
		var consequent = ($$anchor) => {
			var span = root_1$1();
			var text = child(span, true);

			reset(span);
			template_effect(() => set_text(text, label()));
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if (show_label()) $$render(consequent);
		});
	}

	var div = sibling(node, 2);
	let classes_1;
	var node_1 = child(div);

	component(node_1, () => $$props.Icon, ($$anchor, Icon_1) => {
		Icon_1($$anchor, {});
	});

	var node_2 = sibling(node_1, 2);

	{
		var consequent_1 = ($$anchor) => {
			var fragment = comment();
			var node_3 = first_child(fragment);

			snippet(node_3, () => $$props.children);
			append($$anchor, fragment);
		};

		if_block(node_2, ($$render) => {
			if ($$props.children) $$render(consequent_1);
		});
	}

	reset(div);
	reset(button);

	template_effect(() => {
		classes = set_class(button, 1, 'icon-button svelte-3jwzs9', null, classes, {
			pending: pending(),
			padded: padded(),
			highlight: highlight(),
			transparent: transparent()
		});

		button.disabled = disabled();
		set_attribute(button, 'aria-label', label());
		set_attribute(button, 'aria-haspopup', hasPopup());
		set_attribute(button, 'title', label());

		styles = set_style(button, '', styles, {
			'--border-color': border(),
			color: !disabled() && get(_color) ? get(_color) : "var(--block-label-text-color)",
			'--bg-color': !disabled() ? background() : "auto"
		});

		classes_1 = set_class(div, 1, 'svelte-3jwzs9', null, classes_1, {
			'x-small': size() === "x-small",
			small: size() === "small",
			large: size() === "large",
			medium: size() === "medium"
		});
	});

	append($$anchor, button);
}

delegate(['click']);

var root$4 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 256 256"><path fill="currentColor" d="M200 32h-36.26a47.92 47.92 0 0 0-71.48 0H56a16 16 0 0 0-16 16v168a16 16 0 0 0 16 16h144a16 16 0 0 0 16-16V48a16 16 0 0 0-16-16m-72 0a32 32 0 0 1 32 32H96a32 32 0 0 1 32-32m72 184H56V48h26.75A47.9 47.9 0 0 0 80 64v8a8 8 0 0 0 8 8h80a8 8 0 0 0 8-8v-8a47.9 47.9 0 0 0-2.75-16H200Z"></path></svg>`);

function ImagePaste($$anchor) {
	var svg = root$4();

	append($$anchor, svg);
}

var root$3 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-mic"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`);

function Microphone($$anchor) {
	var svg = root$3();

	append($$anchor, svg);
}

var root$2 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="90%" height="90%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`);

function Upload($$anchor) {
	var svg = root$2();

	append($$anchor, svg);
}

var root$1 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-video"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>`);

function Video($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

var root = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2c-4.963 0-9 4.038-9 9c0 3.328 1.82 6.232 4.513 7.79l-2.067 1.378A1 1 0 0 0 6 22h12a1 1 0 0 0 .555-1.832l-2.067-1.378C19.18 17.232 21 14.328 21 11c0-4.962-4.037-9-9-9zm0 16c-3.859 0-7-3.141-7-7c0-3.86 3.141-7 7-7s7 3.14 7 7c0 3.859-3.141 7-7 7z"></path><path fill="currentColor" d="M12 6c-2.757 0-5 2.243-5 5s2.243 5 5 5s5-2.243 5-5s-2.243-5-5-5zm0 8c-1.654 0-3-1.346-3-3s1.346-3 3-3s3 1.346 3 3s-1.346 3-3 3z"></path></svg>`);

function Webcam($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

var root_2 = from_html(`<button aria-label="Upload file"><!></button>`);
var root_3 = from_html(`<button aria-label="Record audio"><!></button>`);
var root_4 = from_html(`<button aria-label="Capture from camera"><!></button>`);
var root_5 = from_html(`<button aria-label="Record video from camera"><!></button>`);
var root_6 = from_html(`<button aria-label="Paste from clipboard"><!></button>`);
var root_1 = from_html(`<span class="source-selection svelte-exvkcd" data-testid="source-select"><!> <!> <!> <!> <!></span>`);

function SelectSource($$anchor, $$props) {
	push($$props, true);

	let active_source = prop($$props, 'active_source', 15),
		handle_clear = prop($$props, 'handle_clear', 3, () => {}),
		handle_select = prop($$props, 'handle_select', 3, () => {});

	let unique_sources = user_derived(() => [...new Set($$props.sources)]);

	async function handle_select_source(source) {
		handle_clear()();
		active_source(source);
		handle_select()(source);
	}

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_5 = ($$anchor) => {
			var span = root_1();
			var node_1 = child(span);

			{
				var consequent = ($$anchor) => {
					var button = root_2();
					let classes;

					button.__click = () => handle_select_source("upload");

					var node_2 = child(button);

					Upload(node_2);
					reset(button);
					template_effect(() => classes = set_class(button, 1, 'icon svelte-exvkcd', null, classes, { selected: active_source() === "upload" || !active_source() }));
					append($$anchor, button);
				};

				if_block(node_1, ($$render) => {
					if ($$props.sources.includes("upload")) $$render(consequent);
				});
			}

			var node_3 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var button_1 = root_3();
					let classes_1;

					button_1.__click = () => handle_select_source("microphone");

					var node_4 = child(button_1);

					Microphone(node_4);
					reset(button_1);
					template_effect(() => classes_1 = set_class(button_1, 1, 'icon svelte-exvkcd', null, classes_1, { selected: active_source() === "microphone" }));
					append($$anchor, button_1);
				};

				if_block(node_3, ($$render) => {
					if ($$props.sources.includes("microphone")) $$render(consequent_1);
				});
			}

			var node_5 = sibling(node_3, 2);

			{
				var consequent_2 = ($$anchor) => {
					var button_2 = root_4();
					let classes_2;

					button_2.__click = () => handle_select_source("webcam");

					var node_6 = child(button_2);

					Webcam(node_6);
					reset(button_2);
					template_effect(() => classes_2 = set_class(button_2, 1, 'icon svelte-exvkcd', null, classes_2, { selected: active_source() === "webcam" }));
					append($$anchor, button_2);
				};

				if_block(node_5, ($$render) => {
					if ($$props.sources.includes("webcam")) $$render(consequent_2);
				});
			}

			var node_7 = sibling(node_5, 2);

			{
				var consequent_3 = ($$anchor) => {
					var button_3 = root_5();
					let classes_3;

					button_3.__click = () => handle_select_source("webcam-video");

					var node_8 = child(button_3);

					Video(node_8);
					reset(button_3);
					template_effect(() => classes_3 = set_class(button_3, 1, 'icon svelte-exvkcd', null, classes_3, { selected: active_source() === "webcam-video" }));
					append($$anchor, button_3);
				};

				if_block(node_7, ($$render) => {
					if ($$props.sources.includes("webcam-video")) $$render(consequent_3);
				});
			}

			var node_9 = sibling(node_7, 2);

			{
				var consequent_4 = ($$anchor) => {
					var button_4 = root_6();
					let classes_4;

					button_4.__click = () => handle_select_source("clipboard");

					var node_10 = child(button_4);

					ImagePaste(node_10);
					reset(button_4);
					template_effect(() => classes_4 = set_class(button_4, 1, 'icon svelte-exvkcd', null, classes_4, { selected: active_source() === "clipboard" }));
					append($$anchor, button_4);
				};

				if_block(node_9, ($$render) => {
					if ($$props.sources.includes("clipboard")) $$render(consequent_4);
				});
			}

			reset(span);
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if (get(unique_sources).length > 1 || $$props.sources.includes("clipboard")) $$render(consequent_5);
		});
	}

	append($$anchor, fragment);
	pop();
}

delegate(['click']);

export { IconButton as I, Microphone as M, SelectSource as S, Upload as U, Video as V, Webcam as W, ImagePaste as a, component as c };
//# sourceMappingURL=ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js.map
