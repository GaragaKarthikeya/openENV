import { p as prop, i as if_block, d as bind_this, b as set_class, f as set_style, k as each } from './i18n-CMo5lpzy.js';
import { b as active_effect, aW as should_intro, E as EFFECT_TRANSPARENT, aX as BLOCK_EFFECT, aY as EFFECT_RAN, A as effect, z as untrack, aZ as TRANSITION_GLOBAL, a_ as TRANSITION_IN, a$ as TRANSITION_OUT, ac as without_reactive_context, b0 as is_function, O as queue_micro_task, b1 as noop, a as append, f as from_svg, D as DEV, b2 as transition_slide_display, aj as delegate, R as push, y as user_effect, w as get, X as sibling, t as template_effect, a0 as set_text, T as pop, u as state, a5 as user_derived, x as set, V as child, a6 as comment, S as first_child, W as from_html, Y as reset } from './index-BvBk1Iap.js';
import { h as html } from './html-CdUEtR5E.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { A as Amuchina } from './index-C7e-J7CF.js';

/** @import { Raf } from '#client' */

const now = () => performance.now() ;

/** @type {Raf} */
const raf = {
	// don't access requestAnimationFrame eagerly outside method
	// this allows basic testing of user code without JSDOM
	// bunder will eval and remove ternary when the user's app is built
	tick: /** @param {any} _ */ (_) => (requestAnimationFrame )(_),
	now: () => now(),
	tasks: new Set()
};

/** @import { TaskCallback, Task, TaskEntry } from '#client' */

// TODO move this into timing.js where it probably belongs

/**
 * @returns {void}
 */
function run_tasks() {
	// use `raf.now()` instead of the `requestAnimationFrame` callback argument, because
	// otherwise things can get wonky https://github.com/sveltejs/svelte/pull/14541
	const now = raf.now();

	raf.tasks.forEach((task) => {
		if (!task.c(now)) {
			raf.tasks.delete(task);
			task.f();
		}
	});

	if (raf.tasks.size !== 0) {
		raf.tick(run_tasks);
	}
}

/**
 * Creates a new task that runs on each raf frame
 * until it returns a falsy value or is aborted
 * @param {TaskCallback} callback
 * @returns {Task}
 */
function loop(callback) {
	/** @type {TaskEntry} */
	let task;

	if (raf.tasks.size === 0) {
		raf.tick(run_tasks);
	}

	return {
		promise: new Promise((fulfill) => {
			raf.tasks.add((task = { c: callback, f: fulfill }));
		}),
		abort() {
			raf.tasks.delete(task);
		}
	};
}

/** @import { AnimateFn, Animation, AnimationConfig, EachItem, Effect, EffectNodes, TransitionFn, TransitionManager } from '#client' */

/**
 * @param {Element} element
 * @param {'introstart' | 'introend' | 'outrostart' | 'outroend'} type
 * @returns {void}
 */
function dispatch_event(element, type) {
	without_reactive_context(() => {
		element.dispatchEvent(new CustomEvent(type));
	});
}

/**
 * Converts a property to the camel-case format expected by Element.animate(), KeyframeEffect(), and KeyframeEffect.setKeyframes().
 * @param {string} style
 * @returns {string}
 */
function css_property_to_camelcase(style) {
	// in compliance with spec
	if (style === 'float') return 'cssFloat';
	if (style === 'offset') return 'cssOffset';

	// do not rename custom @properties
	if (style.startsWith('--')) return style;

	const parts = style.split('-');
	if (parts.length === 1) return parts[0];
	return (
		parts[0] +
		parts
			.slice(1)
			.map(/** @param {any} word */ (word) => word[0].toUpperCase() + word.slice(1))
			.join('')
	);
}

/**
 * @param {string} css
 * @returns {Keyframe}
 */
function css_to_keyframe(css) {
	/** @type {Keyframe} */
	const keyframe = {};
	const parts = css.split(';');
	for (const part of parts) {
		const [property, value] = part.split(':');
		if (!property || value === undefined) break;

		const formatted_property = css_property_to_camelcase(property.trim());
		keyframe[formatted_property] = value.trim();
	}
	return keyframe;
}

/** @param {number} t */
const linear$1 = (t) => t;

/**
 * Called inside block effects as `$.transition(...)`. This creates a transition manager and
 * attaches it to the current effect — later, inside `pause_effect` and `resume_effect`, we
 * use this to create `intro` and `outro` transitions.
 * @template P
 * @param {number} flags
 * @param {HTMLElement} element
 * @param {() => TransitionFn<P | undefined>} get_fn
 * @param {(() => P) | null} get_params
 * @returns {void}
 */
function transition(flags, element, get_fn, get_params) {
	var is_intro = (flags & TRANSITION_IN) !== 0;
	var is_outro = (flags & TRANSITION_OUT) !== 0;
	var is_both = is_intro && is_outro;
	var is_global = (flags & TRANSITION_GLOBAL) !== 0;

	/** @type {'in' | 'out' | 'both'} */
	var direction = is_both ? 'both' : is_intro ? 'in' : 'out';

	/** @type {AnimationConfig | ((opts: { direction: 'in' | 'out' }) => AnimationConfig) | undefined} */
	var current_options;

	var inert = element.inert;

	/**
	 * The default overflow style, stashed so we can revert changes during the transition
	 * that are necessary to work around a Safari <18 bug
	 * TODO 6.0 remove this, if older versions of Safari have died out enough
	 */
	var overflow = element.style.overflow;

	/** @type {Animation | undefined} */
	var intro;

	/** @type {Animation | undefined} */
	var outro;

	function get_options() {
		return without_reactive_context(() => {
			// If a transition is still ongoing, we use the existing options rather than generating
			// new ones. This ensures that reversible transitions reverse smoothly, rather than
			// jumping to a new spot because (for example) a different `duration` was used
			return (current_options ??= get_fn()(element, get_params?.() ?? /** @type {P} */ ({}), {
				direction
			}));
		});
	}

	/** @type {TransitionManager} */
	var transition = {
		is_global,
		in() {
			element.inert = inert;

			if (!is_intro) {
				outro?.abort();
				outro?.reset?.();
				return;
			}

			if (!is_outro) {
				// if we intro then outro then intro again, we want to abort the first intro,
				// if it's not a bidirectional transition
				intro?.abort();
			}

			dispatch_event(element, 'introstart');

			intro = animate(element, get_options(), outro, 1, () => {
				dispatch_event(element, 'introend');

				// Ensure we cancel the animation to prevent leaking
				intro?.abort();
				intro = current_options = undefined;

				element.style.overflow = overflow;
			});
		},
		out(fn) {
			if (!is_outro) {
				fn?.();
				current_options = undefined;
				return;
			}

			element.inert = true;

			dispatch_event(element, 'outrostart');

			outro = animate(element, get_options(), intro, 0, () => {
				dispatch_event(element, 'outroend');
				fn?.();
			});
		},
		stop: () => {
			intro?.abort();
			outro?.abort();
		}
	};

	var e = /** @type {Effect & { nodes: EffectNodes }} */ (active_effect);

	(e.nodes.t ??= []).push(transition);

	// if this is a local transition, we only want to run it if the parent (branch) effect's
	// parent (block) effect is where the state change happened. we can determine that by
	// looking at whether the block effect is currently initializing
	if (is_intro && should_intro) {
		var run = is_global;

		if (!run) {
			var block = /** @type {Effect | null} */ (e.parent);

			// skip over transparent blocks (e.g. snippets, else-if blocks)
			while (block && (block.f & EFFECT_TRANSPARENT) !== 0) {
				while ((block = block.parent)) {
					if ((block.f & BLOCK_EFFECT) !== 0) break;
				}
			}

			run = !block || (block.f & EFFECT_RAN) !== 0;
		}

		if (run) {
			effect(() => {
				untrack(() => transition.in());
			});
		}
	}
}

/**
 * Animates an element, according to the provided configuration
 * @param {Element} element
 * @param {AnimationConfig | ((opts: { direction: 'in' | 'out' }) => AnimationConfig)} options
 * @param {Animation | undefined} counterpart The corresponding intro/outro to this outro/intro
 * @param {number} t2 The target `t` value — `1` for intro, `0` for outro
 * @param {(() => void)} on_finish Called after successfully completing the animation
 * @returns {Animation}
 */
function animate(element, options, counterpart, t2, on_finish) {
	var is_intro = t2 === 1;

	if (is_function(options)) {
		// In the case of a deferred transition (such as `crossfade`), `option` will be
		// a function rather than an `AnimationConfig`. We need to call this function
		// once the DOM has been updated...
		/** @type {Animation} */
		var a;
		var aborted = false;

		queue_micro_task(() => {
			if (aborted) return;
			var o = options({ direction: is_intro ? 'in' : 'out' });
			a = animate(element, o, counterpart, t2, on_finish);
		});

		// ...but we want to do so without using `async`/`await` everywhere, so
		// we return a facade that allows everything to remain synchronous
		return {
			abort: () => {
				aborted = true;
				a?.abort();
			},
			deactivate: () => a.deactivate(),
			reset: () => a.reset(),
			t: () => a.t()
		};
	}

	counterpart?.deactivate();

	if (!options?.duration) {
		on_finish();

		return {
			abort: noop,
			deactivate: noop,
			reset: noop,
			t: () => t2
		};
	}

	const { delay = 0, css, tick, easing = linear$1 } = options;

	var keyframes = [];

	if (is_intro && counterpart === undefined) {
		if (tick) {
			tick(0, 1); // TODO put in nested effect, to avoid interleaved reads/writes?
		}

		if (css) {
			var styles = css_to_keyframe(css(0, 1));
			keyframes.push(styles, styles);
		}
	}

	var get_t = () => 1 - t2;

	// create a dummy animation that lasts as long as the delay (but with whatever devtools
	// multiplier is in effect). in the common case that it is `0`, we keep it anyway so that
	// the CSS keyframes aren't created until the DOM is updated
	//
	// fill forwards to prevent the element from rendering without styles applied
	// see https://github.com/sveltejs/svelte/issues/14732
	var animation = element.animate(keyframes, { duration: delay, fill: 'forwards' });

	animation.onfinish = () => {
		// remove dummy animation from the stack to prevent conflict with main animation
		animation.cancel();

		// for bidirectional transitions, we start from the current position,
		// rather than doing a full intro/outro
		var t1 = counterpart?.t() ?? 1 - t2;
		counterpart?.abort();

		var delta = t2 - t1;
		var duration = /** @type {number} */ (options.duration) * Math.abs(delta);
		var keyframes = [];

		if (duration > 0) {
			/**
			 * Whether or not the CSS includes `overflow: hidden`, in which case we need to
			 * add it as an inline style to work around a Safari <18 bug
			 * TODO 6.0 remove this, if possible
			 */
			var needs_overflow_hidden = false;

			if (css) {
				var n = Math.ceil(duration / (1000 / 60)); // `n` must be an integer, or we risk missing the `t2` value

				for (var i = 0; i <= n; i += 1) {
					var t = t1 + delta * easing(i / n);
					var styles = css_to_keyframe(css(t, 1 - t));
					keyframes.push(styles);

					needs_overflow_hidden ||= styles.overflow === 'hidden';
				}
			}

			if (needs_overflow_hidden) {
				/** @type {HTMLElement} */ (element).style.overflow = 'hidden';
			}

			get_t = () => {
				var time = /** @type {number} */ (
					/** @type {globalThis.Animation} */ (animation).currentTime
				);

				return t1 + delta * easing(time / duration);
			};

			if (tick) {
				loop(() => {
					if (animation.playState !== 'running') return false;

					var t = get_t();
					tick(t, 1 - t);

					return true;
				});
			}
		}

		animation = element.animate(keyframes, { duration, fill: 'forwards' });

		animation.onfinish = () => {
			get_t = () => t2;
			tick?.(t2, 1 - t2);
			on_finish();
		};
	};

	return {
		abort: () => {
			if (animation) {
				animation.cancel();
				// This prevents memory leaks in Chromium
				animation.effect = null;
				// This prevents onfinish to be launched after cancel(),
				// which can happen in some rare cases
				// see https://github.com/sveltejs/svelte/issues/13681
				animation.onfinish = noop;
			}
		},
		deactivate: () => {
			on_finish = noop;
		},
		reset: () => {
			if (t2 === 0) {
				tick?.(1, 0);
			}
		},
		t: () => get_t()
	};
}

var root$5 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6L8 10L12 6"></path></svg>`);

function ChevronDown($$anchor) {
	var svg = root$5();

	append($$anchor, svg);
}

var root$4 = from_svg(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"></path></svg>`);

function Error($$anchor) {
	var svg = root$4();

	append($$anchor, svg);
}

var root$3 = from_svg(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"></path></svg>`);

function Info($$anchor) {
	var svg = root$3();

	append($$anchor, svg);
}

var root$2 = from_svg(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`);

function Success($$anchor) {
	var svg = root$2();

	append($$anchor, svg);
}

var root$1 = from_svg(`<svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"></path></svg>`);

function Warning($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

const is_external_url = (link, root = location.href) => {
  try {
    return !!link && new URL(link).origin !== new URL(root).origin;
  } catch (e) {
    return false;
  }
};
function sanitize(source) {
  const amuchina = new Amuchina();
  const node = new DOMParser().parseFromString(source, "text/html");
  walk_nodes(node.body, "A", (node2) => {
    if (node2 instanceof HTMLElement && "target" in node2) {
      if (is_external_url(node2.getAttribute("href"), location.href)) {
        node2.setAttribute("target", "_blank");
        node2.setAttribute("rel", "noopener noreferrer");
      }
    }
  });
  return amuchina.sanitize(node).body.innerHTML;
}
function walk_nodes(node, test, callback) {
  if (node && (node.nodeName === test || typeof test === "function")) {
    callback(node);
  }
  const children = node?.childNodes || [];
  for (let i = 0; i < children.length; i++) {
    walk_nodes(children[i], test, callback);
  }
}

/** @import { BlurParams, CrossfadeParams, DrawParams, FadeParams, FlyParams, ScaleParams, SlideParams, TransitionConfig } from './public' */


/** @param {number} x */
const linear = (x) => x;

/** @param {number} t */
function cubic_out(t) {
	const f = t - 1.0;
	return f * f * f + 1.0;
}

/** @param {number | string} value
 * @returns {[number, string]}
 */
function split_css_unit(value) {
	const split = typeof value === 'string' && value.match(/^\s*(-?[\d.]+)([^\s]*)\s*$/);
	return split ? [parseFloat(split[1]), split[2] || 'px'] : [/** @type {number} */ (value), 'px'];
}

/**
 * Animates the opacity of an element from 0 to the current opacity for `in` transitions and from the current opacity to 0 for `out` transitions.
 *
 * @param {Element} node
 * @param {FadeParams} [params]
 * @returns {TransitionConfig}
 */
function fade(node, { delay = 0, duration = 400, easing = linear } = {}) {
	const o = +getComputedStyle(node).opacity;
	return {
		delay,
		duration,
		easing,
		css: (t) => `opacity: ${t * o}`
	};
}

/**
 * Animates the x and y positions and the opacity of an element. `in` transitions animate from the provided values, passed as parameters to the element's default values. `out` transitions animate from the element's default values to the provided values.
 *
 * @param {Element} node
 * @param {FlyParams} [params]
 * @returns {TransitionConfig}
 */
function fly(
	node,
	{ delay = 0, duration = 400, easing = cubic_out, x = 0, y = 0, opacity = 0 } = {}
) {
	const style = getComputedStyle(node);
	const target_opacity = +style.opacity;
	const transform = style.transform === 'none' ? '' : style.transform;
	const od = target_opacity * (1 - opacity);
	const [x_value, x_unit] = split_css_unit(x);
	const [y_value, y_unit] = split_css_unit(y);
	return {
		delay,
		duration,
		easing,
		css: (t, u) => `
			transform: ${transform} translate(${(1 - t) * x_value}${x_unit}, ${(1 - t) * y_value}${y_unit});
			opacity: ${target_opacity - od * u}`
	};
}

var slide_warning = false;

/**
 * Slides an element in and out.
 *
 * @param {Element} node
 * @param {SlideParams} [params]
 * @returns {TransitionConfig}
 */
function slide(node, { delay = 0, duration = 400, easing = cubic_out, axis = 'y' } = {}) {
	const style = getComputedStyle(node);

	if (DEV && !slide_warning && /(contents|inline|table)/.test(style.display)) {
		slide_warning = true;
		Promise.resolve().then(() => (slide_warning = false));
		transition_slide_display(style.display);
	}

	const opacity = +style.opacity;
	const primary_property = axis === 'y' ? 'height' : 'width';
	const primary_property_value = parseFloat(style[primary_property]);
	const secondary_properties = axis === 'y' ? ['top', 'bottom'] : ['left', 'right'];
	const capitalized_secondary_properties = secondary_properties.map(
		(e) => /** @type {'Left' | 'Right' | 'Top' | 'Bottom'} */ (`${e[0].toUpperCase()}${e.slice(1)}`)
	);
	const padding_start_value = parseFloat(style[`padding${capitalized_secondary_properties[0]}`]);
	const padding_end_value = parseFloat(style[`padding${capitalized_secondary_properties[1]}`]);
	const margin_start_value = parseFloat(style[`margin${capitalized_secondary_properties[0]}`]);
	const margin_end_value = parseFloat(style[`margin${capitalized_secondary_properties[1]}`]);
	const border_width_start_value = parseFloat(
		style[`border${capitalized_secondary_properties[0]}Width`]
	);
	const border_width_end_value = parseFloat(
		style[`border${capitalized_secondary_properties[1]}Width`]
	);
	return {
		delay,
		duration,
		easing,
		css: (t) =>
			'overflow: hidden;' +
			`opacity: ${Math.min(t * 20, 1) * opacity};` +
			`${primary_property}: ${t * primary_property_value}px;` +
			`padding-${secondary_properties[0]}: ${t * padding_start_value}px;` +
			`padding-${secondary_properties[1]}: ${t * padding_end_value}px;` +
			`margin-${secondary_properties[0]}: ${t * margin_start_value}px;` +
			`margin-${secondary_properties[1]}: ${t * margin_end_value}px;` +
			`border-${secondary_properties[0]}-width: ${t * border_width_start_value}px;` +
			`border-${secondary_properties[1]}-width: ${t * border_width_end_value}px;` +
			`min-${primary_property}: 0`
	};
}

var root_8 = from_html(`<span class="toast-count svelte-irmu64"> </span>`);
var root_11 = from_html(`<div class="toast-separator svelte-irmu64"></div>`);
var root_10 = from_html(`<div><div data-testid="toast-text"><!></div></div> <!>`, 1);
var root_9 = from_html(`<div class="toast-messages svelte-irmu64"></div>`);
var root_12 = from_html(`<div></div>`);
var root = from_html(`<div role="status" aria-live="polite" data-testid="toast-body"><div class="toast-header svelte-irmu64" role="button" tabindex="0"><div><!></div> <div class="toast-title-row svelte-irmu64"><span> <!></span> <div><!></div></div> <button type="button" aria-label="Close" data-testid="toast-close"><span aria-hidden="true">&#215;</span></button></div> <!> <!></div>`);

function ToastContent($$anchor, $$props) {
	push($$props, true);

	let messages = prop($$props, 'messages', 19, () => []),
		expanded = prop($$props, 'expanded', 3, true);

	let touch_start_x = state(0);
	let touch_start_y = state(0);
	let offset_x = state(0);
	let is_dragging = state(false);
	let toast_element;

	const default_messages = {
		error: "An error occurred.",
		warning: "Warning.",
		success: "Success.",
		info: "Info."
	};

	let count = user_derived(() => messages().length);
	let first_message = user_derived(() => messages()[0]);
	let type_label = user_derived(() => $$props.type.charAt(0).toUpperCase() + $$props.type.slice(1));
	let has_duration = user_derived(() => get(first_message)?.duration !== null);
	let timer_duration = user_derived(() => get(has_duration) ? `${get(first_message).duration}s` : "0s");

	function handle_toggle() {
		$$props.ontoggle?.();
	}

	function close_all() {
		messages().forEach((msg) => {
			$$props.onclose?.(msg.id);
		});
	}

	function handle_touch_start(e) {
		set(touch_start_x, e.touches[0].clientX, true);
		set(touch_start_y, e.touches[0].clientY, true);
		set(is_dragging, true);
	}

	function handle_touch_move(e) {
		if (!get(is_dragging)) return;

		const touch_x = e.touches[0].clientX;
		const touch_y = e.touches[0].clientY;
		const delta_x = touch_x - get(touch_start_x);
		const delta_y = touch_y - get(touch_start_y);

		if (Math.abs(delta_x) > Math.abs(delta_y) && Math.abs(delta_x) > 10) {
			e.preventDefault();
			set(offset_x, delta_x);
		}
	}

	function handle_touch_end() {
		if (!get(is_dragging)) return;

		if (Math.abs(get(offset_x)) > 100) {
			close_all();
		} else {
			set(offset_x, 0);
		}

		set(is_dragging, false);
	}

	user_effect(() => {
		if (get(has_duration) && messages().length === 1) {
			setTimeout(close_all, get(first_message).duration * 1000);
		}
	});

	var div = root();

	div.__touchstart = handle_touch_start;
	div.__touchmove = handle_touch_move;
	div.__touchend = handle_touch_end;

	var div_1 = child(div);

	div_1.__click = handle_toggle;

	div_1.__keydown = (e) => {
		if (e.key === "Enter" || e.key === " ") {
			handle_toggle();
		}
	};

	var div_2 = child(div_1);
	var node = child(div_2);

	{
		var consequent = ($$anchor) => {
			Warning($$anchor);
		};

		var alternate_2 = ($$anchor) => {
			var fragment_1 = comment();
			var node_1 = first_child(fragment_1);

			{
				var consequent_1 = ($$anchor) => {
					Info($$anchor);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_3 = comment();
					var node_2 = first_child(fragment_3);

					{
						var consequent_2 = ($$anchor) => {
							Success($$anchor);
						};

						var alternate = ($$anchor) => {
							var fragment_5 = comment();
							var node_3 = first_child(fragment_5);

							{
								var consequent_3 = ($$anchor) => {
									Error($$anchor);
								};

								if_block(
									node_3,
									($$render) => {
										if ($$props.type === "error") $$render(consequent_3);
									},
									true
								);
							}

							append($$anchor, fragment_5);
						};

						if_block(
							node_2,
							($$render) => {
								if ($$props.type === "success") $$render(consequent_2); else $$render(alternate, false);
							},
							true
						);
					}

					append($$anchor, fragment_3);
				};

				if_block(
					node_1,
					($$render) => {
						if ($$props.type === "info") $$render(consequent_1); else $$render(alternate_1, false);
					},
					true
				);
			}

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if ($$props.type === "warning") $$render(consequent); else $$render(alternate_2, false);
		});
	}

	reset(div_2);

	var div_3 = sibling(div_2, 2);
	var span = child(div_3);
	var text = child(span);
	var node_4 = sibling(text);

	{
		var consequent_4 = ($$anchor) => {
			var span_1 = root_8();
			var text_1 = child(span_1);

			reset(span_1);
			template_effect(() => set_text(text_1, `(${get(count) ?? ''})`));
			append($$anchor, span_1);
		};

		if_block(node_4, ($$render) => {
			if (get(count) > 1) $$render(consequent_4);
		});
	}

	reset(span);

	var div_4 = sibling(span, 2);
	let classes;
	var node_5 = child(div_4);

	ChevronDown(node_5);
	reset(div_4);
	reset(div_3);

	var button = sibling(div_3, 2);

	button.__click = (e) => {
		e.stopPropagation();
		close_all();
	};

	reset(div_1);

	var node_6 = sibling(div_1, 2);

	{
		var consequent_6 = ($$anchor) => {
			var div_5 = root_9();

			each(div_5, 23, messages, (message) => message.id, ($$anchor, message, i) => {
				var fragment_7 = root_10();
				var div_6 = first_child(fragment_7);
				var div_7 = child(div_6);
				var node_7 = child(div_7);

				html(node_7, () => sanitize(get(message).message || default_messages[$$props.type]));
				reset(div_7);
				reset(div_6);

				var node_8 = sibling(div_6, 2);

				{
					var consequent_5 = ($$anchor) => {
						var div_8 = root_11();

						append($$anchor, div_8);
					};

					if_block(node_8, ($$render) => {
						if (get(i) < messages().length - 1) $$render(consequent_5);
					});
				}

				template_effect(() => {
					set_class(div_6, 1, `toast-message-item ${$$props.type ?? ''}`, 'svelte-irmu64');
					set_class(div_7, 1, `toast-message-text ${$$props.type ?? ''}`, 'svelte-irmu64');
				});

				append($$anchor, fragment_7);
			});

			reset(div_5);
			transition(3, div_5, () => slide, () => ({ duration: 200 }));
			append($$anchor, div_5);
		};

		if_block(node_6, ($$render) => {
			if (expanded()) $$render(consequent_6);
		});
	}

	var node_9 = sibling(node_6, 2);

	{
		var consequent_7 = ($$anchor) => {
			var div_9 = root_12();

			template_effect(() => {
				set_class(div_9, 1, `timer ${$$props.type ?? ''}`, 'svelte-irmu64');
				set_style(div_9, `animation-duration: ${get(timer_duration) ?? ''}`);
			});

			append($$anchor, div_9);
		};

		if_block(node_9, ($$render) => {
			if (get(has_duration)) $$render(consequent_7);
		});
	}

	reset(div);
	bind_this(div, ($$value) => toast_element = $$value, () => toast_element);

	template_effect(
		($0) => {
			set_class(div, 1, `toast-body ${$$props.type ?? ''}`, 'svelte-irmu64');
			set_style(div, `transform: translateX(${get(offset_x) ?? ''}px); opacity: ${$0 ?? ''};`);
			set_class(div_2, 1, `toast-icon ${$$props.type ?? ''}`, 'svelte-irmu64');
			set_class(span, 1, `toast-title ${$$props.type ?? ''}`, 'svelte-irmu64');
			set_text(text, `${get(type_label) ?? ''} `);
			classes = set_class(div_4, 1, 'chevron svelte-irmu64', null, classes, { expanded: expanded(), visible: get(count) > 0 });
			set_class(button, 1, `toast-close ${$$props.type ?? ''}`, 'svelte-irmu64');
		},
		[() => 1 - Math.abs(get(offset_x)) / 300]
	);

	transition(1, div, () => fade, () => ({ duration: 200, delay: 100 }));
	transition(2, div, () => fade, () => ({ duration: 200 }));
	append($$anchor, div);
	pop();
}

delegate(['touchstart', 'touchmove', 'touchend', 'click', 'keydown']);

export { ToastContent as T, fly as a, fade as f, loop as l, raf as r, slide as s, transition as t };
//# sourceMappingURL=StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js.map
