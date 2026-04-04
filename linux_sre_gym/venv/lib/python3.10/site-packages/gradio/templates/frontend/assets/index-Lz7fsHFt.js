import { w as writable, p as prop, m as setup_stores, b as set_class, f as set_style, q as store_get, i as if_block, d as bind_this, k as each, u as index, s as slot } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, x as set, t as template_effect, a as append, T as pop, u as state, X as sibling, W as from_html, w as get, V as child, Y as reset, z as untrack, a5 as user_derived, a0 as set_text, S as first_child, a6 as comment, M as tick, a7 as text } from './index-BvBk1Iap.js';
import { r as raf$1, l as loop } from './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';

function pretty_si(num) {
  let units = ["", "k", "M", "G", "T", "P", "E", "Z"];
  let i = 0;
  while (num > 1e3 && i < units.length - 1) {
    num /= 1e3;
    i++;
  }
  let unit = units[i];
  return (Number.isInteger(num) ? num : num.toFixed(1)) + unit;
}

/**
 * @param {any} obj
 * @returns {obj is Date}
 */
function is_date(obj) {
	return Object.prototype.toString.call(obj) === '[object Date]';
}

/** @import { Task } from '#client' */
/** @import { SpringOpts, SpringUpdateOpts, TickContext } from './private.js' */
/** @import { Spring as SpringStore } from './public.js' */

/**
 * @template T
 * @param {TickContext} ctx
 * @param {T} last_value
 * @param {T} current_value
 * @param {T} target_value
 * @returns {T}
 */
function tick_spring(ctx, last_value, current_value, target_value) {
	if (typeof current_value === 'number' || is_date(current_value)) {
		// @ts-ignore
		const delta = target_value - current_value;
		// @ts-ignore
		const velocity = (current_value - last_value) / (ctx.dt || 1 / 60); // guard div by 0
		const spring = ctx.opts.stiffness * delta;
		const damper = ctx.opts.damping * velocity;
		const acceleration = (spring - damper) * ctx.inv_mass;
		const d = (velocity + acceleration) * ctx.dt;
		if (Math.abs(d) < ctx.opts.precision && Math.abs(delta) < ctx.opts.precision) {
			return target_value; // settled
		} else {
			ctx.settled = false; // signal loop to keep ticking
			// @ts-ignore
			return is_date(current_value) ? new Date(current_value.getTime() + d) : current_value + d;
		}
	} else if (Array.isArray(current_value)) {
		// @ts-ignore
		return current_value.map((_, i) =>
			// @ts-ignore
			tick_spring(ctx, last_value[i], current_value[i], target_value[i])
		);
	} else if (typeof current_value === 'object') {
		const next_value = {};
		for (const k in current_value) {
			// @ts-ignore
			next_value[k] = tick_spring(ctx, last_value[k], current_value[k], target_value[k]);
		}
		// @ts-ignore
		return next_value;
	} else {
		throw new Error(`Cannot spring ${typeof current_value} values`);
	}
}

/**
 * The spring function in Svelte creates a store whose value is animated, with a motion that simulates the behavior of a spring. This means when the value changes, instead of transitioning at a steady rate, it "bounces" like a spring would, depending on the physics parameters provided. This adds a level of realism to the transitions and can enhance the user experience.
 *
 * @deprecated Use [`Spring`](https://svelte.dev/docs/svelte/svelte-motion#Spring) instead
 * @template [T=any]
 * @param {T} [value]
 * @param {SpringOpts} [opts]
 * @returns {SpringStore<T>}
 */
function spring(value, opts = {}) {
	const store = writable(value);
	const { stiffness = 0.15, damping = 0.8, precision = 0.01 } = opts;
	/** @type {number} */
	let last_time;
	/** @type {Task | null} */
	let task;
	/** @type {object} */
	let current_token;

	let last_value = /** @type {T} */ (value);
	let target_value = /** @type {T | undefined} */ (value);

	let inv_mass = 1;
	let inv_mass_recovery_rate = 0;
	let cancel_task = false;
	/**
	 * @param {T} new_value
	 * @param {SpringUpdateOpts} opts
	 * @returns {Promise<void>}
	 */
	function set(new_value, opts = {}) {
		target_value = new_value;
		const token = (current_token = {});
		if (value == null || opts.hard || (spring.stiffness >= 1 && spring.damping >= 1)) {
			cancel_task = true; // cancel any running animation
			last_time = raf$1.now();
			last_value = new_value;
			store.set((value = target_value));
			return Promise.resolve();
		} else if (opts.soft) {
			const rate = opts.soft === true ? 0.5 : +opts.soft;
			inv_mass_recovery_rate = 1 / (rate * 60);
			inv_mass = 0; // infinite mass, unaffected by spring forces
		}
		if (!task) {
			last_time = raf$1.now();
			cancel_task = false;
			task = loop((now) => {
				if (cancel_task) {
					cancel_task = false;
					task = null;
					return false;
				}
				inv_mass = Math.min(inv_mass + inv_mass_recovery_rate, 1);

				// clamp elapsed time to 1/30th of a second, so that longer pauses
				// (blocked thread or inactive tab) don't cause the spring to go haywire
				const elapsed = Math.min(now - last_time, 1000 / 30);

				/** @type {TickContext} */
				const ctx = {
					inv_mass,
					opts: spring,
					settled: true,
					dt: (elapsed * 60) / 1000
				};
				// @ts-ignore
				const next_value = tick_spring(ctx, last_value, value, target_value);
				last_time = now;
				last_value = /** @type {T} */ (value);
				store.set((value = /** @type {T} */ (next_value)));
				if (ctx.settled) {
					task = null;
				}
				return !ctx.settled;
			});
		}
		return new Promise((fulfil) => {
			/** @type {Task} */ (task).promise.then(() => {
				if (token === current_token) fulfil();
			});
		});
	}
	/** @type {SpringStore<T>} */
	// @ts-expect-error - class-only properties are missing
	const spring = {
		set,
		update: (fn, opts) => set(fn(/** @type {T} */ (target_value), /** @type {T} */ (value)), opts),
		subscribe: store.subscribe,
		stiffness,
		damping,
		precision
	};
	return spring;
}

var root$1 = from_html(`<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-1vhirvf"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-1vhirvf"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-1vhirvf"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-1vhirvf"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-1vhirvf"></path></g></svg></div>`);

function Loader($$anchor, $$props) {
	push($$props, true);

	const $top = () => store_get(top, '$top', $$stores);
	const $bottom = () => store_get(bottom, '$bottom', $$stores);
	const [$$stores, $$cleanup] = setup_stores();
	let margin = prop($$props, 'margin', 3, true);
	const top = spring([0, 0]);
	const bottom = spring([0, 0]);
	let dismounted = state(false);

	async function animate() {
		await Promise.all([top.set([125, 140]), bottom.set([-125, -140])]);
		await Promise.all([top.set([-125, 140]), bottom.set([125, -140])]);
		await Promise.all([top.set([-125, 0]), bottom.set([125, -0])]);
		await Promise.all([top.set([125, 0]), bottom.set([-125, 0])]);
	}

	async function run() {
		await animate();

		if (!get(dismounted)) run();
	}

	async function loading() {
		await Promise.all([top.set([125, 0]), bottom.set([-125, 0])]);
		run();
	}

	user_effect(() => {
		loading();

		return () => {
			set(dismounted, true);
		};
	});

	var div = root$1();
	let classes;
	var svg = child(div);
	var g = child(svg);
	var g_1 = sibling(g);

	reset(svg);
	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'svelte-1vhirvf', null, classes, { margin: margin() });
		set_style(g, `transform: translate(${$top()[0] ?? ''}px, ${$top()[1] ?? ''}px);`);
		set_style(g_1, `transform: translate(${$bottom()[0] ?? ''}px, ${$bottom()[1] ?? ''}px);`);
	});

	append($$anchor, div);
	pop();
	$$cleanup();
}

let items = [];
let called = false;
const is_browser = typeof window !== "undefined";
const raf = is_browser ? window.requestAnimationFrame : (cb) => {};

async function scroll_into_view(el, enable = true) {
	if (window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && enable !== true) {
		return;
	}

	items.push(el);

	if (!called) called = true; else return;

	await tick();

	raf(() => {
		let min = [0, 0];

		for (let i = 0; i < items.length; i++) {
			const element = items[i];
			const box = element.getBoundingClientRect();

			if (i === 0 || box.top + window.scrollY <= min[0]) {
				min[0] = box.top + window.scrollY;
				min[1] = i;
			}
		}

		window.scrollTo({ top: min[0] - 20, behavior: "smooth" });
		called = false;
		items = [];
	});
}

var root_1 = from_html(`<div class="validation-error svelte-1uj8rng"> <button class="svelte-1uj8rng"><!></button></div>`);
var root_3 = from_html(`<div class="eta-bar svelte-1uj8rng"></div>`);
var root_6 = from_html(`<!> `, 1);
var root_17 = from_html(`<!> <!> <!> <!>`, 1);
var root_14 = from_html(`<div class="progress-level svelte-1uj8rng"><div class="progress-level-inner svelte-1uj8rng"><!></div> <div class="progress-bar-wrap svelte-1uj8rng"><div class="progress-bar svelte-1uj8rng"></div></div></div>`);
var root_24 = from_html(`<p class="loading svelte-1uj8rng"> </p> <!>`, 1);
var root_2 = from_html(`<!> <div><!> <!></div> <!> <!>`, 1);
var root_26 = from_html(`<div class="clear-status svelte-1uj8rng"><!></div> <span class="error svelte-1uj8rng"> </span> <!>`, 1);
var root = from_html(`<div data-testid="status-tracker"><!> <!></div>`);

function Static($$anchor, $$props) {
	push($$props, true);

	let eta = prop($$props, 'eta', 7, null),
		scroll_to_output = prop($$props, 'scroll_to_output', 3, false),
		timer = prop($$props, 'timer', 3, true),
		show_progress = prop($$props, 'show_progress', 3, "full"),
		message = prop($$props, 'message', 3, null),
		progress = prop($$props, 'progress', 3, null),
		variant = prop($$props, 'variant', 3, "default"),
		loading_text = prop($$props, 'loading_text', 3, "Loading..."),
		absolute = prop($$props, 'absolute', 3, true),
		translucent = prop($$props, 'translucent', 3, false),
		border = prop($$props, 'border', 3, false),
		validation_error = prop($$props, 'validation_error', 7, null),
		show_validation_error = prop($$props, 'show_validation_error', 3, true),
		type = prop($$props, 'type', 3, null);

	let el;
	let _timer = false;
	let timer_start = state(0);
	let old_eta = state(null);
	let eta_from_start = state(null);
	let message_visible = state(false);
	let formatted_eta = state(null);
	const should_hide = user_derived(() => type() === "input" || !$$props.status || $$props.status === "complete" || show_progress() === "hidden" || $$props.status == "streaming" || !!(show_validation_error() && validation_error()));
	let timer_diff = state(0);

	const eta_level = user_derived(() => get(eta_from_start) === null || get(eta_from_start) <= 0 || !get(timer_diff)
		? 0
		: Math.min(get(timer_diff) / get(eta_from_start), 1));

	const formatted_timer = user_derived(() => get(timer_diff).toFixed(1));
	let show_eta_bar = user_derived(() => progress() != null ? false : true);

	function run() {
		raf(() => {
			set(timer_diff, (performance.now() - get(timer_start)) / 1000);

			if (_timer) run();
		});
	}

	let progress_level = user_derived(() => {
		let _progress_level = null;

		if (progress() != null) {
			_progress_level = progress().map((p) => {
				if (p.index != null && p.length != null) {
					return p.index / p.length;
				} else if (p.progress != null) {
					return p.progress;
				}

				return undefined;
			});
		} else {
			_progress_level = null;
		}

		let _last_progress_level;
		let transition = "";

		if (_progress_level) {
			_last_progress_level = _progress_level[_progress_level.length - 1];

			if (_last_progress_level === 0) {
				transition = "0";
			} else {
				transition = "150ms";
			}
		} else {
			_last_progress_level = undefined;
		}

		return {
			progress_level: _progress_level,
			last_progress_level: _last_progress_level,
			progress_bar_transition: transition
		};
	});

	function start_timer() {
		if (_timer) return;

		set(old_eta, set(formatted_eta, null), true);
		set(timer_start, performance.now(), true);
		_timer = true;
		run();
	}

	function stop_timer() {
		set(old_eta, set(formatted_eta, null), true);

		if (!_timer) return;

		_timer = false;
	}

	user_effect(() => {
		if ($$props.status === "pending") {
			start_timer();
		} else {
			untrack(() => {
				stop_timer();
			});
		}
	});

	user_effect(() => {
		if (el && scroll_to_output() && ($$props.status === "pending" || $$props.status === "complete")) {
			scroll_into_view(el, $$props.autoscroll);
		}
	});

	user_effect(() => {
		if (eta() === null) {
			eta(get(old_eta));
		}

		if (eta() != null && get(old_eta) !== eta()) {
			set(eta_from_start, (performance.now() - get(timer_start)) / 1000 + eta());
			set(formatted_eta, get(eta_from_start).toFixed(1), true);
			set(old_eta, eta(), true);
		}
	});

	function close_message() {
		set(message_visible, false);
	}

	user_effect(() => {
		untrack(() => {
			close_message();
		});

		if ($$props.status === "error" && message()) {
			set(message_visible, true);
		}
	});

	var div = root();
	let classes;
	let styles;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_1();
			var text = child(div_1);
			var button = sibling(text);
			var node_1 = child(button);

			{
				let $0 = user_derived(() => $$props.i18n ? $$props.i18n("common.clear") : "Clear");

				IconButton(node_1, {
					get Icon() {
						return Clear;
					},

					get label() {
						return get($0);
					},
					disabled: false,
					size: 'x-small',
					background: 'var(--background-fill-primary)',
					color: 'var(--error-background-text)',
					border: 'var(--border-color-primary)',
					onclick: () => validation_error(null)
				});
			}

			reset(button);
			reset(div_1);
			template_effect(() => set_text(text, `${validation_error() ?? ''} `));
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (validation_error() && show_validation_error()) $$render(consequent);
		});
	}

	var node_2 = sibling(node, 2);

	{
		var consequent_17 = ($$anchor) => {
			var fragment = root_2();
			var node_3 = first_child(fragment);

			{
				var consequent_1 = ($$anchor) => {
					var div_2 = root_3();
					let styles_1;

					template_effect(() => styles_1 = set_style(div_2, '', styles_1, {
						transform: `translateX(${(get(eta_level) || 0) * 100 - 100}%)`
					}));

					append($$anchor, div_2);
				};

				if_block(node_3, ($$render) => {
					if (variant() === "default" && get(show_eta_bar) && show_progress() === "full") $$render(consequent_1);
				});
			}

			var div_3 = sibling(node_3, 2);
			let classes_1;
			var node_4 = child(div_3);

			{
				var consequent_4 = ($$anchor) => {
					var fragment_1 = comment();
					var node_5 = first_child(fragment_1);

					each(node_5, 17, progress, index, ($$anchor, p) => {
						var fragment_2 = comment();
						var node_6 = first_child(fragment_2);

						{
							var consequent_3 = ($$anchor) => {
								var fragment_3 = root_6();
								var node_7 = first_child(fragment_3);

								{
									var consequent_2 = ($$anchor) => {
										var text_1 = text();

										template_effect(($0, $1) => set_text(text_1, `${$0 ?? ''}/${$1 ?? ''}`), [
											() => pretty_si(get(p).index || 0),
											() => pretty_si(get(p).length)
										]);

										append($$anchor, text_1);
									};

									var alternate = ($$anchor) => {
										var text_2 = text();

										template_effect(($0) => set_text(text_2, $0), [() => pretty_si(get(p).index || 0)]);
										append($$anchor, text_2);
									};

									if_block(node_7, ($$render) => {
										if (get(p).length != null) $$render(consequent_2); else $$render(alternate, false);
									});
								}

								var text_3 = sibling(node_7);

								template_effect(() => set_text(text_3, ` ${get(p).unit ?? ''} |  `));
								append($$anchor, fragment_3);
							};

							if_block(node_6, ($$render) => {
								if (get(p).index != null) $$render(consequent_3);
							});
						}

						append($$anchor, fragment_2);
					});

					append($$anchor, fragment_1);
				};

				var alternate_2 = ($$anchor) => {
					var fragment_6 = comment();
					var node_8 = first_child(fragment_6);

					{
						var consequent_5 = ($$anchor) => {
							var text_4 = text();

							template_effect(() => set_text(text_4, `queue: ${$$props.queue_position + 1}/${$$props.queue_size ?? ''} |`));
							append($$anchor, text_4);
						};

						var alternate_1 = ($$anchor) => {
							var fragment_8 = comment();
							var node_9 = first_child(fragment_8);

							{
								var consequent_6 = ($$anchor) => {
									var text_5 = text('processing |');

									append($$anchor, text_5);
								};

								if_block(
									node_9,
									($$render) => {
										if ($$props.queue_position === 0) $$render(consequent_6);
									},
									true
								);
							}

							append($$anchor, fragment_8);
						};

						if_block(
							node_8,
							($$render) => {
								if ($$props.queue_position !== null && $$props.queue_size !== undefined && $$props.queue_position >= 0) $$render(consequent_5); else $$render(alternate_1, false);
							},
							true
						);
					}

					append($$anchor, fragment_6);
				};

				if_block(node_4, ($$render) => {
					if (progress()) $$render(consequent_4); else $$render(alternate_2, false);
				});
			}

			var node_10 = sibling(node_4, 2);

			{
				var consequent_7 = ($$anchor) => {
					var text_6 = text();

					template_effect(() => set_text(text_6, `${get(formatted_timer) ?? ''}${eta() ? `/${get(formatted_eta)}` : ""}s`));
					append($$anchor, text_6);
				};

				if_block(node_10, ($$render) => {
					if (timer()) $$render(consequent_7);
				});
			}

			reset(div_3);

			var node_11 = sibling(div_3, 2);

			{
				var consequent_14 = ($$anchor) => {
					var div_4 = root_14();
					var div_5 = child(div_4);
					var node_12 = child(div_5);

					{
						var consequent_13 = ($$anchor) => {
							var fragment_10 = comment();
							var node_13 = first_child(fragment_10);

							each(node_13, 17, progress, index, ($$anchor, p, i) => {
								var fragment_11 = comment();
								var node_14 = first_child(fragment_11);

								{
									var consequent_12 = ($$anchor) => {
										var fragment_12 = root_17();
										var node_15 = first_child(fragment_12);

										{
											var consequent_8 = ($$anchor) => {
												var text_7 = text(' /');

												append($$anchor, text_7);
											};

											if_block(node_15, ($$render) => {
												if (i !== 0) $$render(consequent_8);
											});
										}

										var node_16 = sibling(node_15, 2);

										{
											var consequent_9 = ($$anchor) => {
												var text_8 = text();

												template_effect(() => set_text(text_8, get(p).desc));
												append($$anchor, text_8);
											};

											if_block(node_16, ($$render) => {
												if (get(p).desc != null) $$render(consequent_9);
											});
										}

										var node_17 = sibling(node_16, 2);

										{
											var consequent_10 = ($$anchor) => {
												var text_9 = text('-');

												append($$anchor, text_9);
											};

											if_block(node_17, ($$render) => {
												if (get(p).desc != null && get(progress_level).progress_level && get(progress_level).progress_level[i] != null) $$render(consequent_10);
											});
										}

										var node_18 = sibling(node_17, 2);

										{
											var consequent_11 = ($$anchor) => {
												var text_10 = text();

												template_effect(($0) => set_text(text_10, `${$0 ?? ''}%`), [
													() => (100 * (get(progress_level).progress_level[i] || 0)).toFixed(1)
												]);

												append($$anchor, text_10);
											};

											if_block(node_18, ($$render) => {
												if (get(progress_level).progress_level != null) $$render(consequent_11);
											});
										}

										append($$anchor, fragment_12);
									};

									if_block(node_14, ($$render) => {
										if (get(p).desc != null || get(progress_level).progress_level && get(progress_level).progress_level[i] != null) $$render(consequent_12);
									});
								}

								append($$anchor, fragment_11);
							});

							append($$anchor, fragment_10);
						};

						if_block(node_12, ($$render) => {
							if (progress() != null) $$render(consequent_13);
						});
					}

					reset(div_5);

					var div_6 = sibling(div_5, 2);
					var div_7 = child(div_6);
					let styles_2;

					reset(div_6);
					reset(div_4);

					template_effect(() => styles_2 = set_style(div_7, '', styles_2, {
						width: `${get(progress_level).last_progress_level * 100}%`,
						transition: get(progress_level).progress_bar_transition
					}));

					append($$anchor, div_4);
				};

				var alternate_3 = ($$anchor) => {
					var fragment_15 = comment();
					var node_19 = first_child(fragment_15);

					{
						var consequent_15 = ($$anchor) => {
							{
								let $0 = user_derived(() => variant() === "default");

								Loader($$anchor, {
									get margin() {
										return get($0);
									}
								});
							}
						};

						if_block(
							node_19,
							($$render) => {
								if (show_progress() === "full") $$render(consequent_15);
							},
							true
						);
					}

					append($$anchor, fragment_15);
				};

				if_block(node_11, ($$render) => {
					if (get(progress_level).last_progress_level != null) $$render(consequent_14); else $$render(alternate_3, false);
				});
			}

			var node_20 = sibling(node_11, 2);

			{
				var consequent_16 = ($$anchor) => {
					var fragment_17 = root_24();
					var p_1 = first_child(fragment_17);
					var text_11 = child(p_1, true);

					reset(p_1);

					var node_21 = sibling(p_1, 2);

					slot(node_21, $$props, 'additional-loading-text', {});
					template_effect(() => set_text(text_11, loading_text()));
					append($$anchor, fragment_17);
				};

				if_block(node_20, ($$render) => {
					if (!timer()) $$render(consequent_16);
				});
			}

			template_effect(() => classes_1 = set_class(div_3, 1, 'progress-text svelte-1uj8rng', null, classes_1, {
				'meta-text-center': variant() === "center",
				'meta-text': variant() === "default"
			}));

			append($$anchor, fragment);
		};

		var alternate_4 = ($$anchor) => {
			var fragment_18 = comment();
			var node_22 = first_child(fragment_18);

			{
				var consequent_18 = ($$anchor) => {
					var fragment_19 = root_26();
					var div_8 = first_child(fragment_19);
					var node_23 = child(div_8);

					{
						let $0 = user_derived(() => $$props.i18n("common.clear"));

						IconButton(node_23, {
							get Icon() {
								return Clear;
							},

							get label() {
								return get($0);
							},
							disabled: false,
							$$events: {
								click: () => {
									$$props.on_clear_status?.();
								}
							}
						});
					}

					reset(div_8);

					var span = sibling(div_8, 2);
					var text_12 = child(span, true);

					reset(span);

					var node_24 = sibling(span, 2);

					slot(node_24, $$props, 'error', {});
					template_effect(($0) => set_text(text_12, $0), [() => $$props.i18n("common.error")]);
					append($$anchor, fragment_19);
				};

				if_block(
					node_22,
					($$render) => {
						if ($$props.status === "error") $$render(consequent_18);
					},
					true
				);
			}

			append($$anchor, fragment_18);
		};

		if_block(node_2, ($$render) => {
			if ($$props.status === "pending") $$render(consequent_17); else $$render(alternate_4, false);
		});
	}

	reset(div);
	bind_this(div, ($$value) => el = $$value, () => el);

	template_effect(() => {
		classes = set_class(div, 1, `wrap ${variant() ?? ''} ${show_progress() ?? ''}`, 'svelte-1uj8rng', classes, {
			'no-click': validation_error() && show_validation_error(),
			hide: get(should_hide),
			translucent: variant() === "center" && ($$props.status === "pending" || $$props.status === "error") || translucent() || show_progress() === "minimal" || validation_error(),
			generating: $$props.status === "generating" && show_progress() === "full",
			border: border()
		});

		styles = set_style(div, '', styles, {
			position: absolute() ? "absolute" : "static",
			padding: absolute() ? "0" : "var(--size-8) 0"
		});
	});

	append($$anchor, div);
	pop();
}

export { Loader as L, Static as S, is_date as i, spring as s };
//# sourceMappingURL=index-Lz7fsHFt.js.map
