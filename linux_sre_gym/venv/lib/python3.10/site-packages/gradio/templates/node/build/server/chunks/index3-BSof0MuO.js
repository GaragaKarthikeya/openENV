import { n as noop } from './async-D55cHugf.js';
import { f as attr_class, g as attr_style, i as stringify, e as ensure_array_like, s as slot, b as store_get, u as unsubscribe_stores } from './index-u8mz_F03.js';
import { w as writable } from './index-Cg-Pg6j3.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import './2-BAeILJ1g.js';

/** @import { Raf } from '#client' */

const now = () => Date.now();

/** @type {Raf} */
const raf = {
	// don't access requestAnimationFrame eagerly outside method
	// this allows basic testing of user code without JSDOM
	// bunder will eval and remove ternary when the user's app is built
	tick: /** @param {any} _ */ (_) => (noop)(),
	now: () => now(),
	tasks: new Set()
};

/** @import { TaskCallback, Task, TaskEntry } from '#client' */

/**
 * Creates a new task that runs on each raf frame
 * until it returns a falsy value or is aborted
 * @param {TaskCallback} callback
 * @returns {Task}
 */
function loop(callback) {
	/** @type {TaskEntry} */
	let task;

	if (raf.tasks.size === 0) ;

	return {
		promise: new Promise((fulfill) => {
			raf.tasks.add((task = { c: callback, f: fulfill }));
		}),
		abort() {
			raf.tasks.delete(task);
		}
	};
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
			last_time = raf.now();
			last_value = new_value;
			store.set((value = target_value));
			return Promise.resolve();
		} else if (opts.soft) {
			const rate = opts.soft === true ? 0.5 : +opts.soft;
			inv_mass_recovery_rate = 1 / (rate * 60);
			inv_mass = 0; // infinite mass, unaffected by spring forces
		}
		if (!task) {
			last_time = raf.now();
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

function Loader($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let { margin = true } = $$props;
    const top = spring([0, 0]);
    const bottom = spring([0, 0]);
    $$renderer2.push(`<div${attr_class("svelte-1vhirvf", void 0, { "margin": margin })}><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-1vhirvf"><g${attr_style(`transform: translate(${stringify(store_get($$store_subs ??= {}, "$top", top)[0])}px, ${stringify(store_get($$store_subs ??= {}, "$top", top)[1])}px);`)}><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-1vhirvf"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-1vhirvf"></path></g><g${attr_style(`transform: translate(${stringify(store_get($$store_subs ??= {}, "$bottom", bottom)[0])}px, ${stringify(store_get($$store_subs ??= {}, "$bottom", bottom)[1])}px);`)}><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-1vhirvf"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1vhirvf"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-1vhirvf"></path></g></svg></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
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
function Static($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      i18n,
      eta = null,
      queue_position,
      queue_size,
      status,
      scroll_to_output = false,
      timer = true,
      show_progress = "full",
      message = null,
      progress = null,
      variant = "default",
      loading_text = "Loading...",
      absolute = true,
      translucent = false,
      border = false,
      autoscroll,
      validation_error = null,
      show_validation_error = true,
      type = null,
      on_clear_status
    } = $$props;
    let formatted_eta = null;
    const should_hide = type === "input" || !status || status === "complete" || show_progress === "hidden" || status == "streaming" || !!(show_validation_error && validation_error);
    let timer_diff = 0;
    const formatted_timer = timer_diff.toFixed(1);
    let show_eta_bar = progress != null ? false : true;
    let progress_level = (() => {
      let _progress_level = null;
      if (progress != null) {
        _progress_level = progress.map((p) => {
          if (p.index != null && p.length != null) {
            return p.index / p.length;
          } else if (p.progress != null) {
            return p.progress;
          }
          return void 0;
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
        _last_progress_level = void 0;
      }
      return {
        progress_level: _progress_level,
        last_progress_level: _last_progress_level,
        progress_bar_transition: transition
      };
    })();
    $$renderer2.push(`<div${attr_class(`wrap ${stringify(variant)} ${stringify(show_progress)}`, "svelte-1uj8rng", {
      "no-click": validation_error && show_validation_error,
      "hide": should_hide,
      "translucent": variant === "center" && (status === "pending" || status === "error") || translucent || show_progress === "minimal" || validation_error,
      "generating": status === "generating" && show_progress === "full",
      "border": border
    })} data-testid="status-tracker"${attr_style("", {
      position: absolute ? "absolute" : "static",
      padding: absolute ? "0" : "var(--size-8) 0"
    })}>`);
    if (validation_error && show_validation_error) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="validation-error svelte-1uj8rng">${escape_html(validation_error)} <button class="svelte-1uj8rng">`);
      IconButton($$renderer2, {
        Icon: Clear,
        label: i18n ? i18n("common.clear") : "Clear",
        disabled: false,
        size: "x-small",
        background: "var(--background-fill-primary)",
        color: "var(--error-background-text)",
        border: "var(--border-color-primary)",
        onclick: () => validation_error = null
      });
      $$renderer2.push(`<!----></button></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (status === "pending") {
      $$renderer2.push("<!--[-->");
      if (variant === "default" && show_eta_bar && show_progress === "full") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="eta-bar svelte-1uj8rng"${attr_style("", {
          transform: `translateX(${stringify(0 * 100 - 100)}%)`
        })}></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> <div${attr_class("progress-text svelte-1uj8rng", void 0, {
        "meta-text-center": variant === "center",
        "meta-text": variant === "default"
      })}>`);
      if (progress) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<!--[-->`);
        const each_array = ensure_array_like(progress);
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let p = each_array[$$index];
          if (p.index != null) {
            $$renderer2.push("<!--[-->");
            if (p.length != null) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`${escape_html(pretty_si(p.index || 0))}/${escape_html(pretty_si(p.length))}`);
            } else {
              $$renderer2.push("<!--[!-->");
              $$renderer2.push(`${escape_html(pretty_si(p.index || 0))}`);
            }
            $$renderer2.push(`<!--]--> ${escape_html(p.unit)} |  `);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (queue_position !== null && queue_size !== void 0 && queue_position >= 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`queue: ${escape_html(queue_position + 1)}/${escape_html(queue_size)} |`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (queue_position === 0) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`processing |`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--> `);
      if (timer) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`${escape_html(formatted_timer)}${escape_html(eta ? `/${formatted_eta}` : "")}s`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div> `);
      if (progress_level.last_progress_level != null) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="progress-level svelte-1uj8rng"><div class="progress-level-inner svelte-1uj8rng">`);
        if (progress != null) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<!--[-->`);
          const each_array_1 = ensure_array_like(progress);
          for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
            let p = each_array_1[i];
            if (p.desc != null || progress_level.progress_level && progress_level.progress_level[i] != null) {
              $$renderer2.push("<!--[-->");
              if (i !== 0) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(` /`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]--> `);
              if (p.desc != null) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`${escape_html(p.desc)}`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]--> `);
              if (p.desc != null && progress_level.progress_level && progress_level.progress_level[i] != null) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`-`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]--> `);
              if (progress_level.progress_level != null) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`${escape_html((100 * (progress_level.progress_level[i] || 0)).toFixed(1))}%`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]-->`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]-->`);
          }
          $$renderer2.push(`<!--]-->`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div> <div class="progress-bar-wrap svelte-1uj8rng"><div class="progress-bar svelte-1uj8rng"${attr_style("", {
          width: `${stringify(progress_level.last_progress_level * 100)}%`,
          transition: progress_level.progress_bar_transition
        })}></div></div></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (show_progress === "full") {
          $$renderer2.push("<!--[-->");
          Loader($$renderer2, { margin: variant === "default" });
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--> `);
      if (!timer) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<p class="loading svelte-1uj8rng">${escape_html(loading_text)}</p> <!--[-->`);
        slot($$renderer2, $$props, "additional-loading-text", {});
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (status === "error") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="clear-status svelte-1uj8rng">`);
        IconButton($$renderer2, { Icon: Clear, label: i18n("common.clear"), disabled: false });
        $$renderer2.push(`<!----></div> <span class="error svelte-1uj8rng">${escape_html(i18n("common.error"))}</span> <!--[-->`);
        slot($$renderer2, $$props, "error", {});
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}

export { Loader as L, Static as S, is_date as i, loop as l, raf as r, spring as s };
//# sourceMappingURL=index3-BSof0MuO.js.map
