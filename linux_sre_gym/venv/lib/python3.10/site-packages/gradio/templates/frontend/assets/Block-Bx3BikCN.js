import { B as BranchManager, p as prop, e as init, i as if_block, d as bind_this, c as attribute_effect, S as STYLE, C as CLASS, s as slot, f as set_style } from './i18n-CMo5lpzy.js';
import { p as block, E as EFFECT_TRANSPARENT, am as set_should_intro, an as NAMESPACE_SVG, d as assign_nodes, ao as create_text, b as active_effect, N as teardown, i as hydrate_node, h as hydrating, e as hydrate_next, ap as ELEMENT_NODE, aq as is_raw_text_element, g as get_first_child, ar as set_hydrating, s as set_hydrate_node, R as push, a3 as mutable_source, a1 as legacy_pre_effect, G as deep_read_state, w as get, a2 as legacy_pre_effect_reset, a6 as comment, S as first_child, a as append, T as pop, U as flushSync, x as set, X as sibling, W as from_html, z as untrack, Z as event, t as template_effect, f as from_svg, as as mutate } from './index-BvBk1Iap.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

/** @import { Effect, EffectNodes, TemplateNode } from '#client' */

/**
 * @param {Comment | Element} node
 * @param {() => string} get_tag
 * @param {boolean} is_svg
 * @param {undefined | ((element: Element, anchor: Node | null) => void)} render_fn,
 * @param {undefined | (() => string)} get_namespace
 * @param {undefined | [number, number]} location
 * @returns {void}
 */
function element(node, get_tag, is_svg, render_fn, get_namespace, location) {
	let was_hydrating = hydrating;

	if (hydrating) {
		hydrate_next();
	}

	/** @type {null | Element} */
	var element = null;

	if (hydrating && hydrate_node.nodeType === ELEMENT_NODE) {
		element = /** @type {Element} */ (hydrate_node);
		hydrate_next();
	}

	var anchor = /** @type {TemplateNode} */ (hydrating ? hydrate_node : node);

	var branches = new BranchManager(anchor, false);

	block(() => {
		const next_tag = get_tag() || null;
		var ns = next_tag === 'svg' ? NAMESPACE_SVG : null;

		if (next_tag === null) {
			branches.ensure(null, null);
			set_should_intro(true);
			return;
		}

		branches.ensure(next_tag, (anchor) => {
			if (next_tag) {
				element = hydrating
					? /** @type {Element} */ (element)
					: ns
						? document.createElementNS(ns, next_tag)
						: document.createElement(next_tag);

				assign_nodes(element, element);

				if (render_fn) {
					if (hydrating && is_raw_text_element(next_tag)) {
						// prevent hydration glitches
						element.append(document.createComment(''));
					}

					// If hydrating, use the existing ssr comment as the anchor so that the
					// inner open and close methods can pick up the existing nodes correctly
					var child_anchor = hydrating
						? get_first_child(element)
						: element.appendChild(create_text());

					if (hydrating) {
						if (child_anchor === null) {
							set_hydrating(false);
						} else {
							set_hydrate_node(child_anchor);
						}
					}

					// `child_anchor` is undefined if this is a void element, but we still
					// need to call `render_fn` in order to run actions etc. If the element
					// contains children, it's a user error (which is warned on elsewhere)
					// and the DOM will be silently discarded
					render_fn(element, child_anchor);
				}

				// we do this after calling `render_fn` so that child effects don't override `nodes.end`
				/** @type {Effect & { nodes: EffectNodes }} */ (active_effect).nodes.end = element;

				anchor.before(element);
			}

			if (hydrating) {
				set_hydrate_node(anchor);
			}
		});

		// revert to the default state after the effect has been created
		set_should_intro(true);

		return () => {
			if (next_tag) {
				// if we're in this callback because we're re-running the effect,
				// disable intros (unless no element is currently displayed)
				set_should_intro(false);
			}
		};
	}, EFFECT_TRANSPARENT);

	teardown(() => {
		set_should_intro(true);
	});

	if (was_hydrating) {
		set_hydrating(true);
		set_hydrate_node(anchor);
	}
}

var root_3 = from_svg(`<svg class="resize-handle svelte-1plpy97" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1plpy97"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1plpy97"></line></svg>`);
var root_2 = from_html(`<!> <!>`, 1);
var root_4 = from_html(`<div class="placeholder svelte-1plpy97"></div>`);
var root_1 = from_html(`<!> <!>`, 1);

function Block($$anchor, $$props) {
	push($$props, false);

	let height = prop($$props, 'height', 12, undefined);
	let min_height = prop($$props, 'min_height', 12, undefined);
	let max_height = prop($$props, 'max_height', 12, undefined);
	let width = prop($$props, 'width', 12, undefined);
	let elem_id = prop($$props, 'elem_id', 12, "");
	let elem_classes = prop($$props, 'elem_classes', 28, () => []);
	let variant = prop($$props, 'variant', 12, "solid");
	let border_mode = prop($$props, 'border_mode', 12, "base");
	let padding = prop($$props, 'padding', 12, true);
	let type = prop($$props, 'type', 12, "normal");
	let test_id = prop($$props, 'test_id', 12, undefined);
	let explicit_call = prop($$props, 'explicit_call', 12, false);
	let container = prop($$props, 'container', 12, true);
	let visible = prop($$props, 'visible', 12, true);
	let allow_overflow = prop($$props, 'allow_overflow', 12, true);
	let overflow_behavior = prop($$props, 'overflow_behavior', 12, "auto");
	let scale = prop($$props, 'scale', 12, null);
	let min_width = prop($$props, 'min_width', 12, 0);
	let flex = prop($$props, 'flex', 12, false);
	let resizable = prop($$props, 'resizable', 12, false);
	let rtl = prop($$props, 'rtl', 12, false);
	let fullscreen = prop($$props, 'fullscreen', 12, false);
	let label = prop($$props, 'label', 12, undefined);
	let old_fullscreen = mutable_source(fullscreen());
	let element$1 = mutable_source();
	let tag = type() === "fieldset" ? "fieldset" : "div";
	let placeholder_height = mutable_source(0);
	let placeholder_width = mutable_source(0);
	let preexpansionBoundingRect = mutable_source(null);

	function handleKeydown(event) {
		if (fullscreen() && event.key === "Escape") {
			fullscreen(false);
		}
	}

	const get_dimension = (dimension_value) => {
		if (dimension_value === undefined) {
			return undefined;
		}

		if (typeof dimension_value === "number") {
			return dimension_value + "px";
		} else if (typeof dimension_value === "string") {
			return dimension_value;
		}
	};

	const resize = (e) => {
		let prevY = e.clientY;

		const onMouseMove = (e) => {
			const dy = e.clientY - prevY;

			prevY = e.clientY;
			mutate(element$1, get(element$1).style.height = `${get(element$1).offsetHeight + dy}px`);
		};

		const onMouseUp = () => {
			window.removeEventListener("mousemove", onMouseMove);
			window.removeEventListener("mouseup", onMouseUp);
		};

		window.addEventListener("mousemove", onMouseMove);
		window.addEventListener("mouseup", onMouseUp);
	};

	legacy_pre_effect(
		() => (
			deep_read_state(fullscreen()),
			get(old_fullscreen),
			get(element$1)
		),
		() => {
			if (fullscreen() !== get(old_fullscreen)) {
				set(old_fullscreen, fullscreen());

				if (fullscreen()) {
					set(preexpansionBoundingRect, get(element$1).getBoundingClientRect());
					set(placeholder_height, get(element$1).offsetHeight);
					set(placeholder_width, get(element$1).offsetWidth);
					window.addEventListener("keydown", handleKeydown);
				} else {
					set(preexpansionBoundingRect, null);
					window.removeEventListener("keydown", handleKeydown);
				}
			}
		}
	);

	legacy_pre_effect(() => (deep_read_state(visible())), () => {
		if (!visible()) {
			flex(false);
		}
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get height() {
			return height();
		},

		set height($$value) {
			height($$value);
			flushSync();
		},

		get min_height() {
			return min_height();
		},

		set min_height($$value) {
			min_height($$value);
			flushSync();
		},

		get max_height() {
			return max_height();
		},

		set max_height($$value) {
			max_height($$value);
			flushSync();
		},

		get width() {
			return width();
		},

		set width($$value) {
			width($$value);
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

		get variant() {
			return variant();
		},

		set variant($$value) {
			variant($$value);
			flushSync();
		},

		get border_mode() {
			return border_mode();
		},

		set border_mode($$value) {
			border_mode($$value);
			flushSync();
		},

		get padding() {
			return padding();
		},

		set padding($$value) {
			padding($$value);
			flushSync();
		},

		get type() {
			return type();
		},

		set type($$value) {
			type($$value);
			flushSync();
		},

		get test_id() {
			return test_id();
		},

		set test_id($$value) {
			test_id($$value);
			flushSync();
		},

		get explicit_call() {
			return explicit_call();
		},

		set explicit_call($$value) {
			explicit_call($$value);
			flushSync();
		},

		get container() {
			return container();
		},

		set container($$value) {
			container($$value);
			flushSync();
		},

		get visible() {
			return visible();
		},

		set visible($$value) {
			visible($$value);
			flushSync();
		},

		get allow_overflow() {
			return allow_overflow();
		},

		set allow_overflow($$value) {
			allow_overflow($$value);
			flushSync();
		},

		get overflow_behavior() {
			return overflow_behavior();
		},

		set overflow_behavior($$value) {
			overflow_behavior($$value);
			flushSync();
		},

		get scale() {
			return scale();
		},

		set scale($$value) {
			scale($$value);
			flushSync();
		},

		get min_width() {
			return min_width();
		},

		set min_width($$value) {
			min_width($$value);
			flushSync();
		},

		get flex() {
			return flex();
		},

		set flex($$value) {
			flex($$value);
			flushSync();
		},

		get resizable() {
			return resizable();
		},

		set resizable($$value) {
			resizable($$value);
			flushSync();
		},

		get rtl() {
			return rtl();
		},

		set rtl($$value) {
			rtl($$value);
			flushSync();
		},

		get fullscreen() {
			return fullscreen();
		},

		set fullscreen($$value) {
			fullscreen($$value);
			flushSync();
		},

		get label() {
			return label();
		},

		set label($$value) {
			label($$value);
			flushSync();
		}
	};

	init();

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_2 = ($$anchor) => {
			var fragment_1 = root_1();
			var node_1 = first_child(fragment_1);

			element(node_1, () => tag, false, ($$element, $$anchor) => {
				bind_this($$element, ($$value) => set(element$1, $$value), () => get(element$1));

				attribute_effect(
					$$element,
					($0, $1) => ({
						'data-testid': test_id(),
						id: elem_id(),
						class: `block ${$0 ?? ''}`,
						dir: rtl() ? "rtl" : "ltr",
						'aria-label': label(),
						style: '',
						[CLASS]: {
							hidden: visible() === "hidden",
							padded: padding(),
							flex: flex(),
							border_focus: border_mode() === "focus",
							border_contrast: border_mode() === "contrast",
							'hide-container': !explicit_call() && !container(),
							fullscreen: fullscreen(),
							animating: fullscreen() && get(preexpansionBoundingRect) !== null,
							'auto-margin': scale() === null
						},
						[STYLE]: $1
					}),
					[
						() => (
							deep_read_state(elem_classes()),
							untrack(() => elem_classes()?.join(' ') || '')
						),

						() => ({
							height: (
								deep_read_state(fullscreen()),
								deep_read_state(height()),
								untrack(() => fullscreen() ? undefined : get_dimension(height()))
							),

							'min-height': (
								deep_read_state(fullscreen()),
								deep_read_state(min_height()),
								untrack(() => fullscreen() ? undefined : get_dimension(min_height()))
							),

							'max-height': (
								deep_read_state(fullscreen()),
								deep_read_state(max_height()),
								untrack(() => fullscreen() ? undefined : get_dimension(max_height()))
							),

							'--start-top': (
								get(preexpansionBoundingRect),
								untrack(() => get(preexpansionBoundingRect) ? `${get(preexpansionBoundingRect).top}px` : "0px")
							),

							'--start-left': (
								get(preexpansionBoundingRect),
								untrack(() => get(preexpansionBoundingRect) ? `${get(preexpansionBoundingRect).left}px` : "0px")
							),

							'--start-width': (
								get(preexpansionBoundingRect),
								untrack(() => get(preexpansionBoundingRect) ? `${get(preexpansionBoundingRect).width}px` : "0px")
							),

							'--start-height': (
								get(preexpansionBoundingRect),
								untrack(() => get(preexpansionBoundingRect) ? `${get(preexpansionBoundingRect).height}px` : "0px")
							),

							width: (
								deep_read_state(fullscreen()),
								deep_read_state(width()),
								untrack(() => fullscreen()
									? undefined
									: typeof width() === "number"
										? `calc(min(${width()}px, 100%))`
										: get_dimension(width()))
							),
							'border-style': variant(),
							overflow: allow_overflow() ? overflow_behavior() : "hidden",
							'flex-grow': scale(),
							'min-width': `calc(min(${min_width()}px, 100%))`
						})
					],
					void 0,
					void 0,
					'svelte-1plpy97'
				);

				var fragment_2 = root_2();
				var node_2 = first_child(fragment_2);

				slot(node_2, $$props, 'default', {});

				var node_3 = sibling(node_2, 2);

				{
					var consequent = ($$anchor) => {
						var svg = root_3();

						event('mousedown', svg, resize);
						append($$anchor, svg);
					};

					if_block(node_3, ($$render) => {
						if (resizable()) $$render(consequent);
					});
				}

				append($$anchor, fragment_2);
			});

			var node_4 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var div = root_4();
					let styles;

					template_effect(() => styles = set_style(div, '', styles, {
						height: get(placeholder_height) + "px",
						width: get(placeholder_width) + "px"
					}));

					append($$anchor, div);
				};

				if_block(node_4, ($$render) => {
					if (fullscreen()) $$render(consequent_1);
				});
			}

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if (visible() === true || visible() === "hidden") $$render(consequent_2);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

export { Block as B };
//# sourceMappingURL=Block-Bx3BikCN.js.map
