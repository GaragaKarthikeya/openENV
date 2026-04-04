import { f as fallback } from './async-D55cHugf.js';
import { c as spread_props, d as bind_props, f as attr_class, s as slot, a as attr, g as attr_style, i as stringify } from './index-u8mz_F03.js';
import { F as FileData, a as prepare_files, g as getDefaultExportFromCjs } from './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import ImagePreview from './ImagePreview-CjXAuu8E.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
import { I as Image$1 } from './Image2-BLbvQKZw.js';
import { t as tinycolor } from './tinycolor-DVnMVnIq.js';
import { C as Check } from './Check-B-uwlXei.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './DownloadLink-BMNxSkQ8.js';
import './IconButton-D82v5nCM.js';
import './Empty-BHvMDNv5.js';
import './ShareButton-DnAPtcgb.js';
import './index-server-CQz6EZl_.js';
import './Download-ByiErn53.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './FullscreenButton-DVAVExlR.js';
import './Maximize-B77VDSzq.js';
import './Image-BLyvTcTy.js';
import './Clear-DH-TDCgr.js';

function IconButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let _color;
    let Icon = $$props["Icon"];
    let label = fallback($$props["label"], "");
    let show_label = fallback($$props["show_label"], false);
    let pending = fallback($$props["pending"], false);
    let size = fallback($$props["size"], "small");
    let padded = fallback($$props["padded"], true);
    let highlight = fallback($$props["highlight"], false);
    let disabled = fallback($$props["disabled"], false);
    let hasPopup = fallback($$props["hasPopup"], false);
    let color = fallback($$props["color"], "var(--block-label-text-color)");
    let transparent = fallback($$props["transparent"], false);
    let background = fallback($$props["background"], "var(--background-fill-primary)");
    let offset = fallback($$props["offset"], 0);
    let label_position = fallback($$props["label_position"], "left");
    let roundedness = fallback($$props["roundedness"], "quite");
    let onclick = fallback($$props["onclick"], void 0);
    _color = highlight ? "var(--color-accent)" : color.toString();
    $$renderer2.push(`<button${attr("disabled", disabled, true)}${attr("aria-label", label)}${attr("aria-haspopup", hasPopup)}${attr("title", label)}${attr_class(`${stringify(roundedness)}-round`, "svelte-1lrnmgy", {
      "pending": pending,
      "padded": padded,
      "highlight": highlight,
      "transparent": transparent
    })}${attr_style("", {
      color: !disabled && _color ? _color : "var(--block-label-text-color)",
      "--bg-color": !disabled ? background : "auto",
      "margin-left": offset + "px"
    })}>`);
    if (show_label && label_position === "left") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span style="margin-left: 4px;" class="svelte-1lrnmgy">${escape_html(label)}</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <div${attr_class("svelte-1lrnmgy", void 0, {
      "small": size === "small",
      "large": size === "large",
      "medium": size === "medium"
    })}>`);
    Icon($$renderer2, {});
    $$renderer2.push(`<!----></div> `);
    if (show_label && label_position === "right") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span style="margin-right: 4px;" class="svelte-1lrnmgy">${escape_html(label)}</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></button>`);
    bind_props($$props, {
      Icon,
      label,
      show_label,
      pending,
      size,
      padded,
      highlight,
      disabled,
      hasPopup,
      color,
      transparent,
      background,
      offset,
      label_position,
      roundedness,
      onclick
    });
  });
}
var ExtensionType = /* @__PURE__ */ ((ExtensionType2) => {
  ExtensionType2["Application"] = "application";
  ExtensionType2["WebGLPipes"] = "webgl-pipes";
  ExtensionType2["WebGLPipesAdaptor"] = "webgl-pipes-adaptor";
  ExtensionType2["WebGLSystem"] = "webgl-system";
  ExtensionType2["WebGPUPipes"] = "webgpu-pipes";
  ExtensionType2["WebGPUPipesAdaptor"] = "webgpu-pipes-adaptor";
  ExtensionType2["WebGPUSystem"] = "webgpu-system";
  ExtensionType2["CanvasSystem"] = "canvas-system";
  ExtensionType2["CanvasPipesAdaptor"] = "canvas-pipes-adaptor";
  ExtensionType2["CanvasPipes"] = "canvas-pipes";
  ExtensionType2["Asset"] = "asset";
  ExtensionType2["LoadParser"] = "load-parser";
  ExtensionType2["ResolveParser"] = "resolve-parser";
  ExtensionType2["CacheParser"] = "cache-parser";
  ExtensionType2["DetectionParser"] = "detection-parser";
  ExtensionType2["MaskEffect"] = "mask-effect";
  ExtensionType2["BlendMode"] = "blend-mode";
  ExtensionType2["TextureSource"] = "texture-source";
  ExtensionType2["Environment"] = "environment";
  ExtensionType2["ShapeBuilder"] = "shape-builder";
  ExtensionType2["Batcher"] = "batcher";
  return ExtensionType2;
})(ExtensionType || {});
const normalizeExtension = (ext) => {
  if (typeof ext === "function" || typeof ext === "object" && ext.extension) {
    if (!ext.extension) {
      throw new Error("Extension class must have an extension object");
    }
    const metadata = typeof ext.extension !== "object" ? { type: ext.extension } : ext.extension;
    ext = { ...metadata, ref: ext };
  }
  if (typeof ext === "object") {
    ext = { ...ext };
  } else {
    throw new Error("Invalid extension type");
  }
  if (typeof ext.type === "string") {
    ext.type = [ext.type];
  }
  return ext;
};
const normalizeExtensionPriority = (ext, defaultPriority) => normalizeExtension(ext).priority ?? defaultPriority;
const extensions = {
  /** @ignore */
  _addHandlers: {},
  /** @ignore */
  _removeHandlers: {},
  /** @ignore */
  _queue: {},
  /**
   * Remove extensions from PixiJS.
   * @param extensions - Extensions to be removed. Can be:
   * - Extension class with static `extension` property
   * - Extension format object with `type` and `ref`
   * - Multiple extensions as separate arguments
   * @returns {extensions} this for chaining
   * @example
   * ```ts
   * // Remove a single extension
   * extensions.remove(MyRendererPlugin);
   *
   * // Remove multiple extensions
   * extensions.remove(
   *     MyRendererPlugin,
   *     MySystemPlugin
   * );
   * ```
   * @see {@link ExtensionType} For available extension types
   * @see {@link ExtensionFormat} For extension format details
   */
  remove(...extensions2) {
    extensions2.map(normalizeExtension).forEach((ext) => {
      ext.type.forEach((type) => this._removeHandlers[type]?.(ext));
    });
    return this;
  },
  /**
   * Register new extensions with PixiJS. Extensions can be registered in multiple formats:
   * - As a class with a static `extension` property
   * - As an extension format object
   * - As multiple extensions passed as separate arguments
   * @param extensions - Extensions to add to PixiJS. Each can be:
   * - A class with static `extension` property
   * - An extension format object with `type` and `ref`
   * - Multiple extensions as separate arguments
   * @returns This extensions instance for chaining
   * @example
   * ```ts
   * // Register a simple extension
   * extensions.add(MyRendererPlugin);
   *
   * // Register multiple extensions
   * extensions.add(
   *     MyRendererPlugin,
   *     MySystemPlugin,
   * });
   * ```
   * @see {@link ExtensionType} For available extension types
   * @see {@link ExtensionFormat} For extension format details
   * @see {@link extensions.remove} For removing registered extensions
   */
  add(...extensions2) {
    extensions2.map(normalizeExtension).forEach((ext) => {
      ext.type.forEach((type) => {
        const handlers = this._addHandlers;
        const queue = this._queue;
        if (!handlers[type]) {
          queue[type] = queue[type] || [];
          queue[type]?.push(ext);
        } else {
          handlers[type]?.(ext);
        }
      });
    });
    return this;
  },
  /**
   * Internal method to handle extensions by name.
   * @param type - The extension type.
   * @param onAdd  - Function handler when extensions are added/registered {@link StrictExtensionFormat}.
   * @param onRemove  - Function handler when extensions are removed/unregistered {@link StrictExtensionFormat}.
   * @returns this for chaining.
   * @internal
   * @ignore
   */
  handle(type, onAdd, onRemove) {
    const addHandlers = this._addHandlers;
    const removeHandlers = this._removeHandlers;
    if (addHandlers[type] || removeHandlers[type]) {
      throw new Error(`Extension type ${type} already has a handler`);
    }
    addHandlers[type] = onAdd;
    removeHandlers[type] = onRemove;
    const queue = this._queue;
    if (queue[type]) {
      queue[type]?.forEach((ext) => onAdd(ext));
      delete queue[type];
    }
    return this;
  },
  /**
   * Handle a type, but using a map by `name` property.
   * @param type - Type of extension to handle.
   * @param map - The object map of named extensions.
   * @returns this for chaining.
   * @ignore
   */
  handleByMap(type, map) {
    return this.handle(
      type,
      (extension) => {
        if (extension.name) {
          map[extension.name] = extension.ref;
        }
      },
      (extension) => {
        if (extension.name) {
          delete map[extension.name];
        }
      }
    );
  },
  /**
   * Handle a type, but using a list of extensions with a `name` property.
   * @param type - Type of extension to handle.
   * @param map - The array of named extensions.
   * @param defaultPriority - Fallback priority if none is defined.
   * @returns this for chaining.
   * @ignore
   */
  handleByNamedList(type, map, defaultPriority = -1) {
    return this.handle(
      type,
      (extension) => {
        const index = map.findIndex((item) => item.name === extension.name);
        if (index >= 0)
          return;
        map.push({ name: extension.name, value: extension.ref });
        map.sort((a2, b2) => normalizeExtensionPriority(b2.value, defaultPriority) - normalizeExtensionPriority(a2.value, defaultPriority));
      },
      (extension) => {
        const index = map.findIndex((item) => item.name === extension.name);
        if (index !== -1) {
          map.splice(index, 1);
        }
      }
    );
  },
  /**
   * Handle a type, but using a list of extensions.
   * @param type - Type of extension to handle.
   * @param list - The list of extensions.
   * @param defaultPriority - The default priority to use if none is specified.
   * @returns this for chaining.
   * @ignore
   */
  handleByList(type, list, defaultPriority = -1) {
    return this.handle(
      type,
      (extension) => {
        if (list.includes(extension.ref)) {
          return;
        }
        list.push(extension.ref);
        list.sort((a2, b2) => normalizeExtensionPriority(b2, defaultPriority) - normalizeExtensionPriority(a2, defaultPriority));
      },
      (extension) => {
        const index = list.indexOf(extension.ref);
        if (index !== -1) {
          list.splice(index, 1);
        }
      }
    );
  },
  /**
   * Mixin the source object(s) properties into the target class's prototype.
   * Copies all property descriptors from source objects to the target's prototype.
   * @param Target - The target class to mix properties into
   * @param sources - One or more source objects containing properties to mix in
   * @example
   * ```ts
   * // Create a mixin with shared properties
   * const moveable = {
   *     x: 0,
   *     y: 0,
   *     move(x: number, y: number) {
   *         this.x += x;
   *         this.y += y;
   *     }
   * };
   *
   * // Create a mixin with computed properties
   * const scalable = {
   *     scale: 1,
   *     get scaled() {
   *         return this.scale > 1;
   *     }
   * };
   *
   * // Apply mixins to a class
   * extensions.mixin(Sprite, moveable, scalable);
   *
   * // Use mixed-in properties
   * const sprite = new Sprite();
   * sprite.move(10, 20);
   * console.log(sprite.x, sprite.y); // 10, 20
   * ```
   * @remarks
   * - Copies all properties including getters/setters
   * - Does not modify source objects
   * - Preserves property descriptors
   * @see {@link Object.defineProperties} For details on property descriptors
   * @see {@link Object.getOwnPropertyDescriptors} For details on property copying
   */
  mixin(Target, ...sources2) {
    for (const source2 of sources2) {
      Object.defineProperties(Target.prototype, Object.getOwnPropertyDescriptors(source2));
    }
  }
};
const browserExt = {
  extension: {
    type: ExtensionType.Environment,
    name: "browser",
    priority: -1
  },
  test: () => true,
  load: async () => {
    await import('./browserAll-D3spK9tT.js');
  }
};
const webworkerExt = {
  extension: {
    type: ExtensionType.Environment,
    name: "webworker",
    priority: 0
  },
  test: () => typeof self !== "undefined" && self.WorkerGlobalScope !== void 0,
  load: async () => {
    await import('./webworkerAll-D45vWcki.js');
  }
};
class ObservablePoint {
  /**
   * Creates a new `ObservablePoint`
   * @param observer - Observer to pass to listen for change events.
   * @param {number} [x=0] - position of the point on the x axis
   * @param {number} [y=0] - position of the point on the y axis
   */
  constructor(observer, x2, y2) {
    this._x = x2 || 0;
    this._y = y2 || 0;
    this._observer = observer;
  }
  /**
   * Creates a clone of this point.
   * @example
   * ```ts
   * // Basic cloning
   * const point = new ObservablePoint(observer, 100, 200);
   * const copy = point.clone();
   *
   * // Clone with new observer
   * const newObserver = {
   *     _onUpdate: (p) => console.log(`Clone updated: (${p.x}, ${p.y})`)
   * };
   * const watched = point.clone(newObserver);
   *
   * // Verify independence
   * watched.set(300, 400); // Only triggers new observer
   * ```
   * @param observer - Optional observer to pass to the new observable point
   * @returns A copy of this observable point
   * @see {@link ObservablePoint.copyFrom} For copying into existing point
   * @see {@link Observer} For observer interface details
   */
  clone(observer) {
    return new ObservablePoint(observer ?? this._observer, this._x, this._y);
  }
  /**
   * Sets the point to a new x and y position.
   *
   * If y is omitted, both x and y will be set to x.
   * @example
   * ```ts
   * // Basic position setting
   * const point = new ObservablePoint(observer);
   * point.set(100, 200);
   *
   * // Set both x and y to same value
   * point.set(50); // x=50, y=50
   * ```
   * @param x - Position on the x axis
   * @param y - Position on the y axis, defaults to x
   * @returns The point instance itself
   * @see {@link ObservablePoint.copyFrom} For copying from another point
   * @see {@link ObservablePoint.equals} For comparing positions
   */
  set(x2 = 0, y2 = x2) {
    if (this._x !== x2 || this._y !== y2) {
      this._x = x2;
      this._y = y2;
      this._observer._onUpdate(this);
    }
    return this;
  }
  /**
   * Copies x and y from the given point into this point.
   * @example
   * ```ts
   * // Basic copying
   * const source = new ObservablePoint(observer, 100, 200);
   * const target = new ObservablePoint();
   * target.copyFrom(source);
   *
   * // Copy and chain operations
   * const point = new ObservablePoint()
   *     .copyFrom(source)
   *     .set(x + 50, y + 50);
   *
   * // Copy from any PointData
   * const data = { x: 10, y: 20 };
   * point.copyFrom(data);
   * ```
   * @param p - The point to copy from
   * @returns The point instance itself
   * @see {@link ObservablePoint.copyTo} For copying to another point
   * @see {@link ObservablePoint.clone} For creating new point copy
   */
  copyFrom(p2) {
    if (this._x !== p2.x || this._y !== p2.y) {
      this._x = p2.x;
      this._y = p2.y;
      this._observer._onUpdate(this);
    }
    return this;
  }
  /**
   * Copies this point's x and y into the given point.
   * @example
   * ```ts
   * // Basic copying
   * const source = new ObservablePoint(100, 200);
   * const target = new ObservablePoint();
   * source.copyTo(target);
   * ```
   * @param p - The point to copy to. Can be any type that is or extends `PointLike`
   * @returns The point (`p`) with values updated
   * @see {@link ObservablePoint.copyFrom} For copying from another point
   * @see {@link ObservablePoint.clone} For creating new point copy
   */
  copyTo(p2) {
    p2.set(this._x, this._y);
    return p2;
  }
  /**
   * Checks if another point is equal to this point.
   *
   * Compares x and y values using strict equality.
   * @example
   * ```ts
   * // Basic equality check
   * const p1 = new ObservablePoint(100, 200);
   * const p2 = new ObservablePoint(100, 200);
   * console.log(p1.equals(p2)); // true
   *
   * // Compare with PointData
   * const data = { x: 100, y: 200 };
   * console.log(p1.equals(data)); // true
   *
   * // Check different points
   * const p3 = new ObservablePoint(200, 300);
   * console.log(p1.equals(p3)); // false
   * ```
   * @param p - The point to check
   * @returns `true` if both `x` and `y` are equal
   * @see {@link ObservablePoint.copyFrom} For making points equal
   * @see {@link PointData} For point data interface
   */
  equals(p2) {
    return p2.x === this._x && p2.y === this._y;
  }
  toString() {
    return `[pixi.js/math:ObservablePoint x=${this._x} y=${this._y} scope=${this._observer}]`;
  }
  /**
   * Position of the observable point on the x axis.
   * Triggers observer callback when value changes.
   * @example
   * ```ts
   * // Basic x position
   * const point = new ObservablePoint(observer);
   * point.x = 100; // Triggers observer
   *
   * // Use in calculations
   * const width = rightPoint.x - leftPoint.x;
   * ```
   * @default 0
   */
  get x() {
    return this._x;
  }
  set x(value) {
    if (this._x !== value) {
      this._x = value;
      this._observer._onUpdate(this);
    }
  }
  /**
   * Position of the observable point on the y axis.
   * Triggers observer callback when value changes.
   * @example
   * ```ts
   * // Basic y position
   * const point = new ObservablePoint(observer);
   * point.y = 200; // Triggers observer
   *
   * // Use in calculations
   * const height = bottomPoint.y - topPoint.y;
   * ```
   * @default 0
   */
  get y() {
    return this._y;
  }
  set y(value) {
    if (this._y !== value) {
      this._y = value;
      this._observer._onUpdate(this);
    }
  }
}
var eventemitter3 = { exports: {} };
var hasRequiredEventemitter3;
function requireEventemitter3() {
  if (hasRequiredEventemitter3) return eventemitter3.exports;
  hasRequiredEventemitter3 = 1;
  (function(module) {
    var has = Object.prototype.hasOwnProperty, prefix = "~";
    function Events() {
    }
    if (Object.create) {
      Events.prototype = /* @__PURE__ */ Object.create(null);
      if (!new Events().__proto__) prefix = false;
    }
    function EE(fn, context2, once) {
      this.fn = fn;
      this.context = context2;
      this.once = once || false;
    }
    function addListener(emitter, event, fn, context2, once) {
      if (typeof fn !== "function") {
        throw new TypeError("The listener must be a function");
      }
      var listener = new EE(fn, context2 || emitter, once), evt = prefix ? prefix + event : event;
      if (!emitter._events[evt]) emitter._events[evt] = listener, emitter._eventsCount++;
      else if (!emitter._events[evt].fn) emitter._events[evt].push(listener);
      else emitter._events[evt] = [emitter._events[evt], listener];
      return emitter;
    }
    function clearEvent(emitter, evt) {
      if (--emitter._eventsCount === 0) emitter._events = new Events();
      else delete emitter._events[evt];
    }
    function EventEmitter2() {
      this._events = new Events();
      this._eventsCount = 0;
    }
    EventEmitter2.prototype.eventNames = function eventNames() {
      var names = [], events, name;
      if (this._eventsCount === 0) return names;
      for (name in events = this._events) {
        if (has.call(events, name)) names.push(prefix ? name.slice(1) : name);
      }
      if (Object.getOwnPropertySymbols) {
        return names.concat(Object.getOwnPropertySymbols(events));
      }
      return names;
    };
    EventEmitter2.prototype.listeners = function listeners(event) {
      var evt = prefix ? prefix + event : event, handlers = this._events[evt];
      if (!handlers) return [];
      if (handlers.fn) return [handlers.fn];
      for (var i2 = 0, l2 = handlers.length, ee = new Array(l2); i2 < l2; i2++) {
        ee[i2] = handlers[i2].fn;
      }
      return ee;
    };
    EventEmitter2.prototype.listenerCount = function listenerCount(event) {
      var evt = prefix ? prefix + event : event, listeners = this._events[evt];
      if (!listeners) return 0;
      if (listeners.fn) return 1;
      return listeners.length;
    };
    EventEmitter2.prototype.emit = function emit(event, a1, a2, a3, a4, a5) {
      var evt = prefix ? prefix + event : event;
      if (!this._events[evt]) return false;
      var listeners = this._events[evt], len = arguments.length, args, i2;
      if (listeners.fn) {
        if (listeners.once) this.removeListener(event, listeners.fn, void 0, true);
        switch (len) {
          case 1:
            return listeners.fn.call(listeners.context), true;
          case 2:
            return listeners.fn.call(listeners.context, a1), true;
          case 3:
            return listeners.fn.call(listeners.context, a1, a2), true;
          case 4:
            return listeners.fn.call(listeners.context, a1, a2, a3), true;
          case 5:
            return listeners.fn.call(listeners.context, a1, a2, a3, a4), true;
          case 6:
            return listeners.fn.call(listeners.context, a1, a2, a3, a4, a5), true;
        }
        for (i2 = 1, args = new Array(len - 1); i2 < len; i2++) {
          args[i2 - 1] = arguments[i2];
        }
        listeners.fn.apply(listeners.context, args);
      } else {
        var length = listeners.length, j2;
        for (i2 = 0; i2 < length; i2++) {
          if (listeners[i2].once) this.removeListener(event, listeners[i2].fn, void 0, true);
          switch (len) {
            case 1:
              listeners[i2].fn.call(listeners[i2].context);
              break;
            case 2:
              listeners[i2].fn.call(listeners[i2].context, a1);
              break;
            case 3:
              listeners[i2].fn.call(listeners[i2].context, a1, a2);
              break;
            case 4:
              listeners[i2].fn.call(listeners[i2].context, a1, a2, a3);
              break;
            default:
              if (!args) for (j2 = 1, args = new Array(len - 1); j2 < len; j2++) {
                args[j2 - 1] = arguments[j2];
              }
              listeners[i2].fn.apply(listeners[i2].context, args);
          }
        }
      }
      return true;
    };
    EventEmitter2.prototype.on = function on(event, fn, context2) {
      return addListener(this, event, fn, context2, false);
    };
    EventEmitter2.prototype.once = function once(event, fn, context2) {
      return addListener(this, event, fn, context2, true);
    };
    EventEmitter2.prototype.removeListener = function removeListener(event, fn, context2, once) {
      var evt = prefix ? prefix + event : event;
      if (!this._events[evt]) return this;
      if (!fn) {
        clearEvent(this, evt);
        return this;
      }
      var listeners = this._events[evt];
      if (listeners.fn) {
        if (listeners.fn === fn && (!once || listeners.once) && (!context2 || listeners.context === context2)) {
          clearEvent(this, evt);
        }
      } else {
        for (var i2 = 0, events = [], length = listeners.length; i2 < length; i2++) {
          if (listeners[i2].fn !== fn || once && !listeners[i2].once || context2 && listeners[i2].context !== context2) {
            events.push(listeners[i2]);
          }
        }
        if (events.length) this._events[evt] = events.length === 1 ? events[0] : events;
        else clearEvent(this, evt);
      }
      return this;
    };
    EventEmitter2.prototype.removeAllListeners = function removeAllListeners(event) {
      var evt;
      if (event) {
        evt = prefix ? prefix + event : event;
        if (this._events[evt]) clearEvent(this, evt);
      } else {
        this._events = new Events();
        this._eventsCount = 0;
      }
      return this;
    };
    EventEmitter2.prototype.off = EventEmitter2.prototype.removeListener;
    EventEmitter2.prototype.addListener = EventEmitter2.prototype.on;
    EventEmitter2.prefixed = prefix;
    EventEmitter2.EventEmitter = EventEmitter2;
    {
      module.exports = EventEmitter2;
    }
  })(eventemitter3);
  return eventemitter3.exports;
}
var eventemitter3Exports = requireEventemitter3();
const EventEmitter = /* @__PURE__ */ getDefaultExportFromCjs(eventemitter3Exports);
const PI_2 = Math.PI * 2;
const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;
class Point {
  /**
   * Creates a new `Point`
   * @param {number} [x=0] - position of the point on the x axis
   * @param {number} [y=0] - position of the point on the y axis
   */
  constructor(x2 = 0, y2 = 0) {
    this.x = 0;
    this.y = 0;
    this.x = x2;
    this.y = y2;
  }
  /**
   * Creates a clone of this point, which is a new instance with the same `x` and `y` values.
   * @example
   * ```ts
   * // Basic point cloning
   * const original = new Point(100, 200);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.set(300, 400);
   *
   * // Verify independence
   * console.log(original); // Point(100, 200)
   * console.log(modified); // Point(300, 400)
   * ```
   * @remarks
   * - Creates new Point instance
   * - Deep copies x and y values
   * - Independent from original
   * - Useful for preserving values
   * @returns A clone of this point
   * @see {@link Point.copyFrom} For copying into existing point
   * @see {@link Point.copyTo} For copying to existing point
   */
  clone() {
    return new Point(this.x, this.y);
  }
  /**
   * Copies x and y from the given point into this point.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Point(100, 200);
   * const target = new Point();
   * target.copyFrom(source);
   *
   * // Copy and chain operations
   * const point = new Point()
   *     .copyFrom(source)
   *     .set(x + 50, y + 50);
   *
   * // Copy from any PointData
   * const data = { x: 10, y: 20 };
   * point.copyFrom(data);
   * ```
   * @param p - The point to copy from
   * @returns The point instance itself
   * @see {@link Point.copyTo} For copying to another point
   * @see {@link Point.clone} For creating new point copy
   */
  copyFrom(p2) {
    this.set(p2.x, p2.y);
    return this;
  }
  /**
   * Copies this point's x and y into the given point.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Point(100, 200);
   * const target = new Point();
   * source.copyTo(target);
   * ```
   * @param p - The point to copy to. Can be any type that is or extends `PointLike`
   * @returns The point (`p`) with values updated
   * @see {@link Point.copyFrom} For copying from another point
   * @see {@link Point.clone} For creating new point copy
   */
  copyTo(p2) {
    p2.set(this.x, this.y);
    return p2;
  }
  /**
   * Checks if another point is equal to this point.
   *
   * Compares x and y values using strict equality.
   * @example
   * ```ts
   * // Basic equality check
   * const p1 = new Point(100, 200);
   * const p2 = new Point(100, 200);
   * console.log(p1.equals(p2)); // true
   *
   * // Compare with PointData
   * const data = { x: 100, y: 200 };
   * console.log(p1.equals(data)); // true
   *
   * // Check different points
   * const p3 = new Point(200, 300);
   * console.log(p1.equals(p3)); // false
   * ```
   * @param p - The point to check
   * @returns `true` if both `x` and `y` are equal
   * @see {@link Point.copyFrom} For making points equal
   * @see {@link PointData} For point data interface
   */
  equals(p2) {
    return p2.x === this.x && p2.y === this.y;
  }
  /**
   * Sets the point to a new x and y position.
   *
   * If y is omitted, both x and y will be set to x.
   * @example
   * ```ts
   * // Basic position setting
   * const point = new Point();
   * point.set(100, 200);
   *
   * // Set both x and y to same value
   * point.set(50); // x=50, y=50
   *
   * // Chain with other operations
   * point
   *     .set(10, 20)
   *     .copyTo(otherPoint);
   * ```
   * @param x - Position on the x axis
   * @param y - Position on the y axis, defaults to x
   * @returns The point instance itself
   * @see {@link Point.copyFrom} For copying from another point
   * @see {@link Point.equals} For comparing positions
   */
  set(x2 = 0, y2 = x2) {
    this.x = x2;
    this.y = y2;
    return this;
  }
  toString() {
    return `[pixi.js/math:Point x=${this.x} y=${this.y}]`;
  }
  /**
   * A static Point object with `x` and `y` values of `0`.
   *
   * This shared instance is reset to zero values when accessed.
   *
   * > [!IMPORTANT] This point is shared and temporary. Do not store references to it.
   * @example
   * ```ts
   * // Use for temporary calculations
   * const tempPoint = Point.shared;
   * tempPoint.set(100, 200);
   * matrix.apply(tempPoint);
   *
   * // Will be reset to (0,0) on next access
   * const fresh = Point.shared; // x=0, y=0
   * ```
   * @readonly
   * @returns A fresh zeroed point for temporary use
   * @see {@link Point.constructor} For creating new points
   * @see {@link PointData} For basic point interface
   */
  static get shared() {
    tempPoint.x = 0;
    tempPoint.y = 0;
    return tempPoint;
  }
}
const tempPoint = new Point();
class Matrix {
  /**
   * @param a - x scale
   * @param b - y skew
   * @param c - x skew
   * @param d - y scale
   * @param tx - x translation
   * @param ty - y translation
   */
  constructor(a2 = 1, b2 = 0, c2 = 0, d2 = 1, tx = 0, ty = 0) {
    this.array = null;
    this.a = a2;
    this.b = b2;
    this.c = c2;
    this.d = d2;
    this.tx = tx;
    this.ty = ty;
  }
  /**
   * Creates a Matrix object based on the given array.
   * Populates matrix components from a flat array in column-major order.
   *
   * > [!NOTE] Array mapping order:
   * > ```
   * > array[0] = a  (x scale)
   * > array[1] = b  (y skew)
   * > array[2] = tx (x translation)
   * > array[3] = c  (x skew)
   * > array[4] = d  (y scale)
   * > array[5] = ty (y translation)
   * > ```
   * @example
   * ```ts
   * // Create matrix from array
   * const matrix = new Matrix();
   * matrix.fromArray([
   *     2, 0,  100,  // a, b, tx
   *     0, 2,  100   // c, d, ty
   * ]);
   *
   * // Create matrix from typed array
   * const float32Array = new Float32Array([
   *     1, 0, 0,     // Scale x1, no skew
   *     0, 1, 0      // No skew, scale x1
   * ]);
   * matrix.fromArray(float32Array);
   * ```
   * @param array - The array to populate the matrix from
   * @see {@link Matrix.toArray} For converting matrix to array
   * @see {@link Matrix.set} For setting values directly
   */
  fromArray(array) {
    this.a = array[0];
    this.b = array[1];
    this.c = array[3];
    this.d = array[4];
    this.tx = array[2];
    this.ty = array[5];
  }
  /**
   * Sets the matrix properties directly.
   * All matrix components can be set in one call.
   * @example
   * ```ts
   * // Set to identity matrix
   * matrix.set(1, 0, 0, 1, 0, 0);
   *
   * // Set to scale matrix
   * matrix.set(2, 0, 0, 2, 0, 0); // Scale 2x
   *
   * // Set to translation matrix
   * matrix.set(1, 0, 0, 1, 100, 50); // Move 100,50
   * ```
   * @param a - Scale on x axis
   * @param b - Shear on y axis
   * @param c - Shear on x axis
   * @param d - Scale on y axis
   * @param tx - Translation on x axis
   * @param ty - Translation on y axis
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.identity} For resetting to identity
   * @see {@link Matrix.fromArray} For setting from array
   */
  set(a2, b2, c2, d2, tx, ty) {
    this.a = a2;
    this.b = b2;
    this.c = c2;
    this.d = d2;
    this.tx = tx;
    this.ty = ty;
    return this;
  }
  /**
   * Creates an array from the current Matrix object.
   *
   * > [!NOTE] The array format is:
   * > ```
   * > Non-transposed:
   * > [a, c, tx,
   * > b, d, ty,
   * > 0, 0, 1]
   * >
   * > Transposed:
   * > [a, b, 0,
   * > c, d, 0,
   * > tx,ty,1]
   * > ```
   * @example
   * ```ts
   * // Basic array conversion
   * const matrix = new Matrix(2, 0, 0, 2, 100, 100);
   * const array = matrix.toArray();
   *
   * // Using existing array
   * const float32Array = new Float32Array(9);
   * matrix.toArray(false, float32Array);
   *
   * // Get transposed array
   * const transposed = matrix.toArray(true);
   * ```
   * @param transpose - Whether to transpose the matrix
   * @param out - Optional Float32Array to store the result
   * @returns The array containing the matrix values
   * @see {@link Matrix.fromArray} For creating matrix from array
   * @see {@link Matrix.array} For cached array storage
   */
  toArray(transpose, out2) {
    if (!this.array) {
      this.array = new Float32Array(9);
    }
    const array = out2 || this.array;
    if (transpose) {
      array[0] = this.a;
      array[1] = this.b;
      array[2] = 0;
      array[3] = this.c;
      array[4] = this.d;
      array[5] = 0;
      array[6] = this.tx;
      array[7] = this.ty;
      array[8] = 1;
    } else {
      array[0] = this.a;
      array[1] = this.c;
      array[2] = this.tx;
      array[3] = this.b;
      array[4] = this.d;
      array[5] = this.ty;
      array[6] = 0;
      array[7] = 0;
      array[8] = 1;
    }
    return array;
  }
  /**
   * Get a new position with the current transformation applied.
   *
   * Can be used to go from a child's coordinate space to the world coordinate space. (e.g. rendering)
   * @example
   * ```ts
   * // Basic point transformation
   * const matrix = new Matrix().translate(100, 50).rotate(Math.PI / 4);
   * const point = new Point(10, 20);
   * const transformed = matrix.apply(point);
   *
   * // Reuse existing point
   * const output = new Point();
   * matrix.apply(point, output);
   * ```
   * @param pos - The origin point to transform
   * @param newPos - Optional point to store the result
   * @returns The transformed point
   * @see {@link Matrix.applyInverse} For inverse transformation
   * @see {@link Point} For point operations
   */
  apply(pos, newPos) {
    newPos = newPos || new Point();
    const x2 = pos.x;
    const y2 = pos.y;
    newPos.x = this.a * x2 + this.c * y2 + this.tx;
    newPos.y = this.b * x2 + this.d * y2 + this.ty;
    return newPos;
  }
  /**
   * Get a new position with the inverse of the current transformation applied.
   *
   * Can be used to go from the world coordinate space to a child's coordinate space. (e.g. input)
   * @example
   * ```ts
   * // Basic inverse transformation
   * const matrix = new Matrix().translate(100, 50).rotate(Math.PI / 4);
   * const worldPoint = new Point(150, 100);
   * const localPoint = matrix.applyInverse(worldPoint);
   *
   * // Reuse existing point
   * const output = new Point();
   * matrix.applyInverse(worldPoint, output);
   *
   * // Convert mouse position to local space
   * const mousePoint = new Point(mouseX, mouseY);
   * const localMouse = matrix.applyInverse(mousePoint);
   * ```
   * @param pos - The origin point to inverse-transform
   * @param newPos - Optional point to store the result
   * @returns The inverse-transformed point
   * @see {@link Matrix.apply} For forward transformation
   * @see {@link Matrix.invert} For getting inverse matrix
   */
  applyInverse(pos, newPos) {
    newPos = newPos || new Point();
    const a2 = this.a;
    const b2 = this.b;
    const c2 = this.c;
    const d2 = this.d;
    const tx = this.tx;
    const ty = this.ty;
    const id = 1 / (a2 * d2 + c2 * -b2);
    const x2 = pos.x;
    const y2 = pos.y;
    newPos.x = d2 * id * x2 + -c2 * id * y2 + (ty * c2 - tx * d2) * id;
    newPos.y = a2 * id * y2 + -b2 * id * x2 + (-ty * a2 + tx * b2) * id;
    return newPos;
  }
  /**
   * Translates the matrix on the x and y axes.
   * Adds to the position values while preserving scale, rotation and skew.
   * @example
   * ```ts
   * // Basic translation
   * const matrix = new Matrix();
   * matrix.translate(100, 50); // Move right 100, down 50
   *
   * // Chain with other transformations
   * matrix
   *     .scale(2, 2)
   *     .translate(100, 0)
   *     .rotate(Math.PI / 4);
   * ```
   * @param x - How much to translate on the x axis
   * @param y - How much to translate on the y axis
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.set} For setting position directly
   * @see {@link Matrix.setTransform} For complete transform setup
   */
  translate(x2, y2) {
    this.tx += x2;
    this.ty += y2;
    return this;
  }
  /**
   * Applies a scale transformation to the matrix.
   * Multiplies the scale values with existing matrix components.
   * @example
   * ```ts
   * // Basic scaling
   * const matrix = new Matrix();
   * matrix.scale(2, 3); // Scale 2x horizontally, 3x vertically
   *
   * // Chain with other transformations
   * matrix
   *     .translate(100, 100)
   *     .scale(2, 2)     // Scales after translation
   *     .rotate(Math.PI / 4);
   * ```
   * @param x - The amount to scale horizontally
   * @param y - The amount to scale vertically
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.setTransform} For setting scale directly
   * @see {@link Matrix.append} For combining transformations
   */
  scale(x2, y2) {
    this.a *= x2;
    this.d *= y2;
    this.c *= x2;
    this.b *= y2;
    this.tx *= x2;
    this.ty *= y2;
    return this;
  }
  /**
   * Applies a rotation transformation to the matrix.
   *
   * Rotates around the origin (0,0) by the given angle in radians.
   * @example
   * ```ts
   * // Basic rotation
   * const matrix = new Matrix();
   * matrix.rotate(Math.PI / 4); // Rotate 45 degrees
   *
   * // Chain with other transformations
   * matrix
   *     .translate(100, 100) // Move to rotation center
   *     .rotate(Math.PI)     // Rotate 180 degrees
   *     .scale(2, 2);        // Scale after rotation
   *
   * // Common angles
   * matrix.rotate(Math.PI / 2);  // 90 degrees
   * matrix.rotate(Math.PI);      // 180 degrees
   * matrix.rotate(Math.PI * 2);  // 360 degrees
   * ```
   * @remarks
   * - Rotates around origin point (0,0)
   * - Affects position if translation was set
   * - Uses counter-clockwise rotation
   * - Order of operations matters when chaining
   * @param angle - The angle in radians
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.setTransform} For setting rotation directly
   * @see {@link Matrix.append} For combining transformations
   */
  rotate(angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const a1 = this.a;
    const c1 = this.c;
    const tx1 = this.tx;
    this.a = a1 * cos - this.b * sin;
    this.b = a1 * sin + this.b * cos;
    this.c = c1 * cos - this.d * sin;
    this.d = c1 * sin + this.d * cos;
    this.tx = tx1 * cos - this.ty * sin;
    this.ty = tx1 * sin + this.ty * cos;
    return this;
  }
  /**
   * Appends the given Matrix to this Matrix.
   * Combines two matrices by multiplying them together: this = this * matrix
   * @example
   * ```ts
   * // Basic matrix combination
   * const matrix = new Matrix();
   * const other = new Matrix().translate(100, 0).rotate(Math.PI / 4);
   * matrix.append(other);
   * ```
   * @remarks
   * - Order matters: A.append(B) !== B.append(A)
   * - Modifies current matrix
   * - Preserves transformation order
   * - Commonly used for combining transforms
   * @param matrix - The matrix to append
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.prepend} For prepending transformations
   * @see {@link Matrix.appendFrom} For appending two external matrices
   */
  append(matrix) {
    const a1 = this.a;
    const b1 = this.b;
    const c1 = this.c;
    const d1 = this.d;
    this.a = matrix.a * a1 + matrix.b * c1;
    this.b = matrix.a * b1 + matrix.b * d1;
    this.c = matrix.c * a1 + matrix.d * c1;
    this.d = matrix.c * b1 + matrix.d * d1;
    this.tx = matrix.tx * a1 + matrix.ty * c1 + this.tx;
    this.ty = matrix.tx * b1 + matrix.ty * d1 + this.ty;
    return this;
  }
  /**
   * Appends two matrices and sets the result to this matrix.
   * Performs matrix multiplication: this = A * B
   * @example
   * ```ts
   * // Basic matrix multiplication
   * const result = new Matrix();
   * const matrixA = new Matrix().scale(2, 2);
   * const matrixB = new Matrix().rotate(Math.PI / 4);
   * result.appendFrom(matrixA, matrixB);
   * ```
   * @remarks
   * - Order matters: A * B !== B * A
   * - Creates a new transformation from two others
   * - More efficient than append() for multiple operations
   * - Does not modify input matrices
   * @param a - The first matrix to multiply
   * @param b - The second matrix to multiply
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.append} For single matrix combination
   * @see {@link Matrix.prepend} For reverse order multiplication
   */
  appendFrom(a2, b2) {
    const a1 = a2.a;
    const b1 = a2.b;
    const c1 = a2.c;
    const d1 = a2.d;
    const tx = a2.tx;
    const ty = a2.ty;
    const a22 = b2.a;
    const b22 = b2.b;
    const c2 = b2.c;
    const d2 = b2.d;
    this.a = a1 * a22 + b1 * c2;
    this.b = a1 * b22 + b1 * d2;
    this.c = c1 * a22 + d1 * c2;
    this.d = c1 * b22 + d1 * d2;
    this.tx = tx * a22 + ty * c2 + b2.tx;
    this.ty = tx * b22 + ty * d2 + b2.ty;
    return this;
  }
  /**
   * Sets the matrix based on all the available properties.
   * Combines position, scale, rotation, skew and pivot in a single operation.
   * @example
   * ```ts
   * // Basic transform setup
   * const matrix = new Matrix();
   * matrix.setTransform(
   *     100, 100,    // position
   *     0, 0,        // pivot
   *     2, 2,        // scale
   *     Math.PI / 4, // rotation (45 degrees)
   *     0, 0         // skew
   * );
   * ```
   * @remarks
   * - Updates all matrix components at once
   * - More efficient than separate transform calls
   * - Uses radians for rotation and skew
   * - Pivot affects rotation center
   * @param x - Position on the x axis
   * @param y - Position on the y axis
   * @param pivotX - Pivot on the x axis
   * @param pivotY - Pivot on the y axis
   * @param scaleX - Scale on the x axis
   * @param scaleY - Scale on the y axis
   * @param rotation - Rotation in radians
   * @param skewX - Skew on the x axis
   * @param skewY - Skew on the y axis
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.decompose} For extracting transform properties
   * @see {@link TransformableObject} For transform data structure
   */
  setTransform(x2, y2, pivotX, pivotY, scaleX, scaleY, rotation, skewX, skewY) {
    this.a = Math.cos(rotation + skewY) * scaleX;
    this.b = Math.sin(rotation + skewY) * scaleX;
    this.c = -Math.sin(rotation - skewX) * scaleY;
    this.d = Math.cos(rotation - skewX) * scaleY;
    this.tx = x2 - (pivotX * this.a + pivotY * this.c);
    this.ty = y2 - (pivotX * this.b + pivotY * this.d);
    return this;
  }
  /**
   * Prepends the given Matrix to this Matrix.
   * Combines two matrices by multiplying them together: this = matrix * this
   * @example
   * ```ts
   * // Basic matrix prepend
   * const matrix = new Matrix().scale(2, 2);
   * const other = new Matrix().translate(100, 0);
   * matrix.prepend(other); // Translation happens before scaling
   * ```
   * @remarks
   * - Order matters: A.prepend(B) !== B.prepend(A)
   * - Modifies current matrix
   * - Reverses transformation order compared to append()
   * @param matrix - The matrix to prepend
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.append} For appending transformations
   * @see {@link Matrix.appendFrom} For combining external matrices
   */
  prepend(matrix) {
    const tx1 = this.tx;
    if (matrix.a !== 1 || matrix.b !== 0 || matrix.c !== 0 || matrix.d !== 1) {
      const a1 = this.a;
      const c1 = this.c;
      this.a = a1 * matrix.a + this.b * matrix.c;
      this.b = a1 * matrix.b + this.b * matrix.d;
      this.c = c1 * matrix.a + this.d * matrix.c;
      this.d = c1 * matrix.b + this.d * matrix.d;
    }
    this.tx = tx1 * matrix.a + this.ty * matrix.c + matrix.tx;
    this.ty = tx1 * matrix.b + this.ty * matrix.d + matrix.ty;
    return this;
  }
  /**
   * Decomposes the matrix into its individual transform components.
   * Extracts position, scale, rotation and skew values from the matrix.
   * @example
   * ```ts
   * // Basic decomposition
   * const matrix = new Matrix()
   *     .translate(100, 100)
   *     .rotate(Math.PI / 4)
   *     .scale(2, 2);
   *
   * const transform = {
   *     position: new Point(),
   *     scale: new Point(),
   *     pivot: new Point(),
   *     skew: new Point(),
   *     rotation: 0
   * };
   *
   * matrix.decompose(transform);
   * console.log(transform.position); // Point(100, 100)
   * console.log(transform.rotation); // ~0.785 (PI/4)
   * console.log(transform.scale); // Point(2, 2)
   * ```
   * @remarks
   * - Handles combined transformations
   * - Accounts for pivot points
   * - Chooses between rotation/skew based on transform type
   * - Uses radians for rotation and skew
   * @param transform - The transform object to store the decomposed values
   * @returns The transform with the newly applied properties
   * @see {@link Matrix.setTransform} For composing from components
   * @see {@link TransformableObject} For transform structure
   */
  decompose(transform) {
    const a2 = this.a;
    const b2 = this.b;
    const c2 = this.c;
    const d2 = this.d;
    const pivot = transform.pivot;
    const skewX = -Math.atan2(-c2, d2);
    const skewY = Math.atan2(b2, a2);
    const delta = Math.abs(skewX + skewY);
    if (delta < 1e-5 || Math.abs(PI_2 - delta) < 1e-5) {
      transform.rotation = skewY;
      transform.skew.x = transform.skew.y = 0;
    } else {
      transform.rotation = 0;
      transform.skew.x = skewX;
      transform.skew.y = skewY;
    }
    transform.scale.x = Math.sqrt(a2 * a2 + b2 * b2);
    transform.scale.y = Math.sqrt(c2 * c2 + d2 * d2);
    transform.position.x = this.tx + (pivot.x * a2 + pivot.y * c2);
    transform.position.y = this.ty + (pivot.x * b2 + pivot.y * d2);
    return transform;
  }
  /**
   * Inverts this matrix.
   * Creates the matrix that when multiplied with this matrix results in an identity matrix.
   * @example
   * ```ts
   * // Basic matrix inversion
   * const matrix = new Matrix()
   *     .translate(100, 50)
   *     .scale(2, 2);
   *
   * matrix.invert(); // Now transforms in opposite direction
   *
   * // Verify inversion
   * const point = new Point(50, 50);
   * const transformed = matrix.apply(point);
   * const original = matrix.invert().apply(transformed);
   * // original ≈ point
   * ```
   * @remarks
   * - Modifies the current matrix
   * - Useful for reversing transformations
   * - Cannot invert matrices with zero determinant
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.identity} For resetting to identity
   * @see {@link Matrix.applyInverse} For inverse transformations
   */
  invert() {
    const a1 = this.a;
    const b1 = this.b;
    const c1 = this.c;
    const d1 = this.d;
    const tx1 = this.tx;
    const n2 = a1 * d1 - b1 * c1;
    this.a = d1 / n2;
    this.b = -b1 / n2;
    this.c = -c1 / n2;
    this.d = a1 / n2;
    this.tx = (c1 * this.ty - d1 * tx1) / n2;
    this.ty = -(a1 * this.ty - b1 * tx1) / n2;
    return this;
  }
  /**
   * Checks if this matrix is an identity matrix.
   *
   * An identity matrix has no transformations applied (default state).
   * @example
   * ```ts
   * // Check if matrix is identity
   * const matrix = new Matrix();
   * console.log(matrix.isIdentity()); // true
   *
   * // Check after transformations
   * matrix.translate(100, 0);
   * console.log(matrix.isIdentity()); // false
   *
   * // Reset and verify
   * matrix.identity();
   * console.log(matrix.isIdentity()); // true
   * ```
   * @remarks
   * - Verifies a = 1, d = 1 (no scale)
   * - Verifies b = 0, c = 0 (no skew)
   * - Verifies tx = 0, ty = 0 (no translation)
   * @returns True if matrix has no transformations
   * @see {@link Matrix.identity} For resetting to identity
   * @see {@link Matrix.IDENTITY} For constant identity matrix
   */
  isIdentity() {
    return this.a === 1 && this.b === 0 && this.c === 0 && this.d === 1 && this.tx === 0 && this.ty === 0;
  }
  /**
   * Resets this Matrix to an identity (default) matrix.
   * Sets all components to their default values: scale=1, no skew, no translation.
   * @example
   * ```ts
   * // Reset transformed matrix
   * const matrix = new Matrix()
   *     .scale(2, 2)
   *     .rotate(Math.PI / 4);
   * matrix.identity(); // Back to default state
   *
   * // Chain after reset
   * matrix
   *     .identity()
   *     .translate(100, 100)
   *     .scale(2, 2);
   *
   * // Compare with identity constant
   * const isDefault = matrix.equals(Matrix.IDENTITY);
   * ```
   * @remarks
   * - Sets a=1, d=1 (default scale)
   * - Sets b=0, c=0 (no skew)
   * - Sets tx=0, ty=0 (no translation)
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.IDENTITY} For constant identity matrix
   * @see {@link Matrix.isIdentity} For checking identity state
   */
  identity() {
    this.a = 1;
    this.b = 0;
    this.c = 0;
    this.d = 1;
    this.tx = 0;
    this.ty = 0;
    return this;
  }
  /**
   * Creates a new Matrix object with the same values as this one.
   * @returns A copy of this matrix. Good for chaining method calls.
   */
  clone() {
    const matrix = new Matrix();
    matrix.a = this.a;
    matrix.b = this.b;
    matrix.c = this.c;
    matrix.d = this.d;
    matrix.tx = this.tx;
    matrix.ty = this.ty;
    return matrix;
  }
  /**
   * Creates a new Matrix object with the same values as this one.
   * @param matrix
   * @example
   * ```ts
   * // Basic matrix cloning
   * const matrix = new Matrix()
   *     .translate(100, 100)
   *     .rotate(Math.PI / 4);
   * const copy = matrix.clone();
   *
   * // Clone and modify
   * const modified = matrix.clone()
   *     .scale(2, 2);
   *
   * // Compare matrices
   * console.log(matrix.equals(copy));     // true
   * console.log(matrix.equals(modified)); // false
   * ```
   * @returns A copy of this matrix. Good for chaining method calls.
   * @see {@link Matrix.copyTo} For copying to existing matrix
   * @see {@link Matrix.copyFrom} For copying from another matrix
   */
  copyTo(matrix) {
    matrix.a = this.a;
    matrix.b = this.b;
    matrix.c = this.c;
    matrix.d = this.d;
    matrix.tx = this.tx;
    matrix.ty = this.ty;
    return matrix;
  }
  /**
   * Changes the values of the matrix to be the same as the ones in given matrix.
   * @example
   * ```ts
   * // Basic matrix copying
   * const source = new Matrix()
   *     .translate(100, 100)
   *     .rotate(Math.PI / 4);
   * const target = new Matrix();
   * target.copyFrom(source);
   * ```
   * @param matrix - The matrix to copy from
   * @returns This matrix. Good for chaining method calls.
   * @see {@link Matrix.clone} For creating new matrix copy
   * @see {@link Matrix.copyTo} For copying to another matrix
   */
  copyFrom(matrix) {
    this.a = matrix.a;
    this.b = matrix.b;
    this.c = matrix.c;
    this.d = matrix.d;
    this.tx = matrix.tx;
    this.ty = matrix.ty;
    return this;
  }
  /**
   * Checks if this matrix equals another matrix.
   * Compares all components for exact equality.
   * @example
   * ```ts
   * // Basic equality check
   * const m1 = new Matrix();
   * const m2 = new Matrix();
   * console.log(m1.equals(m2)); // true
   *
   * // Compare transformed matrices
   * const transform = new Matrix()
   *     .translate(100, 100)
   * const clone = new Matrix()
   *     .scale(2, 2);
   * console.log(transform.equals(clone)); // false
   * ```
   * @param matrix - The matrix to compare to
   * @returns True if matrices are identical
   * @see {@link Matrix.copyFrom} For copying matrix values
   * @see {@link Matrix.isIdentity} For identity comparison
   */
  equals(matrix) {
    return matrix.a === this.a && matrix.b === this.b && matrix.c === this.c && matrix.d === this.d && matrix.tx === this.tx && matrix.ty === this.ty;
  }
  toString() {
    return `[pixi.js:Matrix a=${this.a} b=${this.b} c=${this.c} d=${this.d} tx=${this.tx} ty=${this.ty}]`;
  }
  /**
   * A default (identity) matrix with no transformations applied.
   *
   * > [!IMPORTANT] This is a shared read-only object. Create a new Matrix if you need to modify it.
   * @example
   * ```ts
   * // Get identity matrix reference
   * const identity = Matrix.IDENTITY;
   * console.log(identity.isIdentity()); // true
   *
   * // Compare with identity
   * const matrix = new Matrix();
   * console.log(matrix.equals(Matrix.IDENTITY)); // true
   *
   * // Create new matrix instead of modifying IDENTITY
   * const transform = new Matrix()
   *     .copyFrom(Matrix.IDENTITY)
   *     .translate(100, 100);
   * ```
   * @readonly
   * @returns A read-only identity matrix
   * @see {@link Matrix.shared} For temporary calculations
   * @see {@link Matrix.identity} For resetting matrices
   */
  static get IDENTITY() {
    return identityMatrix$1.identity();
  }
  /**
   * A static Matrix that can be used to avoid creating new objects.
   * Will always ensure the matrix is reset to identity when requested.
   *
   * > [!IMPORTANT] This matrix is shared and temporary. Do not store references to it.
   * @example
   * ```ts
   * // Use for temporary calculations
   * const tempMatrix = Matrix.shared;
   * tempMatrix.translate(100, 100).rotate(Math.PI / 4);
   * const point = tempMatrix.apply({ x: 10, y: 20 });
   *
   * // Will be reset to identity on next access
   * const fresh = Matrix.shared; // Back to identity
   * ```
   * @remarks
   * - Always returns identity matrix
   * - Safe to modify temporarily
   * - Not safe to store references
   * - Useful for one-off calculations
   * @readonly
   * @returns A fresh identity matrix for temporary use
   * @see {@link Matrix.IDENTITY} For immutable identity matrix
   * @see {@link Matrix.identity} For resetting matrices
   */
  static get shared() {
    return tempMatrix$3.identity();
  }
}
const tempMatrix$3 = new Matrix();
const identityMatrix$1 = new Matrix();
const ux = [1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1];
const uy = [0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1];
const vx = [0, -1, -1, -1, 0, 1, 1, 1, 0, 1, 1, 1, 0, -1, -1, -1];
const vy = [1, 1, 0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 1, 1, 0, -1];
const rotationCayley = [];
const rotationMatrices = [];
const signum = Math.sign;
function init() {
  for (let i2 = 0; i2 < 16; i2++) {
    const row = [];
    rotationCayley.push(row);
    for (let j2 = 0; j2 < 16; j2++) {
      const _ux = signum(ux[i2] * ux[j2] + vx[i2] * uy[j2]);
      const _uy = signum(uy[i2] * ux[j2] + vy[i2] * uy[j2]);
      const _vx = signum(ux[i2] * vx[j2] + vx[i2] * vy[j2]);
      const _vy = signum(uy[i2] * vx[j2] + vy[i2] * vy[j2]);
      for (let k2 = 0; k2 < 16; k2++) {
        if (ux[k2] === _ux && uy[k2] === _uy && vx[k2] === _vx && vy[k2] === _vy) {
          row.push(k2);
          break;
        }
      }
    }
  }
  for (let i2 = 0; i2 < 16; i2++) {
    const mat = new Matrix();
    mat.set(ux[i2], uy[i2], vx[i2], vy[i2], 0, 0);
    rotationMatrices.push(mat);
  }
}
init();
const groupD8 = {
  /**
   * | Rotation | Direction |
   * |----------|-----------|
   * | 0°       | East      |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  E: 0,
  /**
   * | Rotation | Direction |
   * |----------|-----------|
   * | 45°↻     | Southeast |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  SE: 1,
  /**
   * | Rotation | Direction |
   * |----------|-----------|
   * | 90°↻     | South     |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  S: 2,
  /**
   * | Rotation | Direction |
   * |----------|-----------|
   * | 135°↻    | Southwest |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  SW: 3,
  /**
   * | Rotation | Direction |
   * |----------|-----------|
   * | 180°     | West      |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  W: 4,
  /**
   * | Rotation    | Direction    |
   * |-------------|--------------|
   * | -135°/225°↻ | Northwest    |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  NW: 5,
  /**
   * | Rotation    | Direction    |
   * |-------------|--------------|
   * | -90°/270°↻  | North        |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  N: 6,
  /**
   * | Rotation    | Direction    |
   * |-------------|--------------|
   * | -45°/315°↻  | Northeast    |
   * @group groupD8
   * @type {GD8Symmetry}
   */
  NE: 7,
  /**
   * Reflection about Y-axis.
   * @group groupD8
   * @type {GD8Symmetry}
   */
  MIRROR_VERTICAL: 8,
  /**
   * Reflection about the main diagonal.
   * @group groupD8
   * @type {GD8Symmetry}
   */
  MAIN_DIAGONAL: 10,
  /**
   * Reflection about X-axis.
   * @group groupD8
   * @type {GD8Symmetry}
   */
  MIRROR_HORIZONTAL: 12,
  /**
   * Reflection about reverse diagonal.
   * @group groupD8
   * @type {GD8Symmetry}
   */
  REVERSE_DIAGONAL: 14,
  /**
   * @group groupD8
   * @param {GD8Symmetry} ind - sprite rotation angle.
   * @returns {GD8Symmetry} The X-component of the U-axis
   *    after rotating the axes.
   */
  uX: (ind) => ux[ind],
  /**
   * @group groupD8
   * @param {GD8Symmetry} ind - sprite rotation angle.
   * @returns {GD8Symmetry} The Y-component of the U-axis
   *    after rotating the axes.
   */
  uY: (ind) => uy[ind],
  /**
   * @group groupD8
   * @param {GD8Symmetry} ind - sprite rotation angle.
   * @returns {GD8Symmetry} The X-component of the V-axis
   *    after rotating the axes.
   */
  vX: (ind) => vx[ind],
  /**
   * @group groupD8
   * @param {GD8Symmetry} ind - sprite rotation angle.
   * @returns {GD8Symmetry} The Y-component of the V-axis
   *    after rotating the axes.
   */
  vY: (ind) => vy[ind],
  /**
   * @group groupD8
   * @param {GD8Symmetry} rotation - symmetry whose opposite
   *   is needed. Only rotations have opposite symmetries while
   *   reflections don't.
   * @returns {GD8Symmetry} The opposite symmetry of `rotation`
   */
  inv: (rotation) => {
    if (rotation & 8) {
      return rotation & 15;
    }
    return -rotation & 7;
  },
  /**
   * Composes the two D8 operations.
   *
   * Taking `^` as reflection:
   *
   * |       | E=0 | S=2 | W=4 | N=6 | E^=8 | S^=10 | W^=12 | N^=14 |
   * |-------|-----|-----|-----|-----|------|-------|-------|-------|
   * | E=0   | E   | S   | W   | N   | E^   | S^    | W^    | N^    |
   * | S=2   | S   | W   | N   | E   | S^   | W^    | N^    | E^    |
   * | W=4   | W   | N   | E   | S   | W^   | N^    | E^    | S^    |
   * | N=6   | N   | E   | S   | W   | N^   | E^    | S^    | W^    |
   * | E^=8  | E^  | N^  | W^  | S^  | E    | N     | W     | S     |
   * | S^=10 | S^  | E^  | N^  | W^  | S    | E     | N     | W     |
   * | W^=12 | W^  | S^  | E^  | N^  | W    | S     | E     | N     |
   * | N^=14 | N^  | W^  | S^  | E^  | N    | W     | S     | E     |
   *
   * [This is a Cayley table]{@link https://en.wikipedia.org/wiki/Cayley_table}
   * @group groupD8
   * @param {GD8Symmetry} rotationSecond - Second operation, which
   *   is the row in the above cayley table.
   * @param {GD8Symmetry} rotationFirst - First operation, which
   *   is the column in the above cayley table.
   * @returns {GD8Symmetry} Composed operation
   */
  add: (rotationSecond, rotationFirst) => rotationCayley[rotationSecond][rotationFirst],
  /**
   * Reverse of `add`.
   * @group groupD8
   * @param {GD8Symmetry} rotationSecond - Second operation
   * @param {GD8Symmetry} rotationFirst - First operation
   * @returns {GD8Symmetry} Result
   */
  sub: (rotationSecond, rotationFirst) => rotationCayley[rotationSecond][groupD8.inv(rotationFirst)],
  /**
   * Adds 180 degrees to rotation, which is a commutative
   * operation.
   * @group groupD8
   * @param {number} rotation - The number to rotate.
   * @returns {number} Rotated number
   */
  rotate180: (rotation) => rotation ^ 4,
  /**
   * Checks if the rotation angle is vertical, i.e. south
   * or north. It doesn't work for reflections.
   * @group groupD8
   * @param {GD8Symmetry} rotation - The number to check.
   * @returns {boolean} Whether or not the direction is vertical
   */
  isVertical: (rotation) => (rotation & 3) === 2,
  // rotation % 4 === 2
  /**
   * Approximates the vector `V(dx,dy)` into one of the
   * eight directions provided by `groupD8`.
   * @group groupD8
   * @param {number} dx - X-component of the vector
   * @param {number} dy - Y-component of the vector
   * @returns {GD8Symmetry} Approximation of the vector into
   *  one of the eight symmetries.
   */
  byDirection: (dx, dy) => {
    if (Math.abs(dx) * 2 <= Math.abs(dy)) {
      if (dy >= 0) {
        return groupD8.S;
      }
      return groupD8.N;
    } else if (Math.abs(dy) * 2 <= Math.abs(dx)) {
      if (dx > 0) {
        return groupD8.E;
      }
      return groupD8.W;
    } else if (dy > 0) {
      if (dx > 0) {
        return groupD8.SE;
      }
      return groupD8.SW;
    } else if (dx > 0) {
      return groupD8.NE;
    }
    return groupD8.NW;
  },
  /**
   * Helps sprite to compensate texture packer rotation.
   * @group groupD8
   * @param {Matrix} matrix - sprite world matrix
   * @param {GD8Symmetry} rotation - The rotation factor to use.
   * @param {number} tx - sprite anchoring
   * @param {number} ty - sprite anchoring
   */
  matrixAppendRotationInv: (matrix, rotation, tx = 0, ty = 0) => {
    const mat = rotationMatrices[groupD8.inv(rotation)];
    mat.tx = tx;
    mat.ty = ty;
    matrix.append(mat);
  },
  /**
   * Transforms rectangle coordinates based on texture packer rotation.
   * Used when texture atlas pages are rotated and coordinates need to be adjusted.
   * @group groupD8
   * @param {RectangleLike} rect - Rectangle with original coordinates to transform
   * @param {RectangleLike} sourceFrame - Source texture frame (includes offset and dimensions)
   * @param {GD8Symmetry} rotation - The groupD8 rotation value
   * @param {Rectangle} out - Rectangle to store the result
   * @returns {Rectangle} Transformed coordinates (includes source frame offset)
   */
  transformRectCoords: (rect, sourceFrame, rotation, out2) => {
    const { x: x2, y: y2, width, height } = rect;
    const { x: frameX, y: frameY, width: frameWidth, height: frameHeight } = sourceFrame;
    if (rotation === groupD8.E) {
      out2.set(x2 + frameX, y2 + frameY, width, height);
      return out2;
    } else if (rotation === groupD8.S) {
      return out2.set(
        frameWidth - y2 - height + frameX,
        x2 + frameY,
        height,
        width
      );
    } else if (rotation === groupD8.W) {
      return out2.set(
        frameWidth - x2 - width + frameX,
        frameHeight - y2 - height + frameY,
        width,
        height
      );
    } else if (rotation === groupD8.N) {
      return out2.set(
        y2 + frameX,
        frameHeight - x2 - width + frameY,
        height,
        width
      );
    }
    return out2.set(x2 + frameX, y2 + frameY, width, height);
  }
};
const tempPoints = [new Point(), new Point(), new Point(), new Point()];
class Rectangle {
  /**
   * @param x - The X coordinate of the upper-left corner of the rectangle
   * @param y - The Y coordinate of the upper-left corner of the rectangle
   * @param width - The overall width of the rectangle
   * @param height - The overall height of the rectangle
   */
  constructor(x2 = 0, y2 = 0, width = 0, height = 0) {
    this.type = "rectangle";
    this.x = Number(x2);
    this.y = Number(y2);
    this.width = Number(width);
    this.height = Number(height);
  }
  /**
   * Returns the left edge (x-coordinate) of the rectangle.
   * @example
   * ```ts
   * // Get left edge position
   * const rect = new Rectangle(100, 100, 200, 150);
   * console.log(rect.left); // 100
   *
   * // Use in alignment calculations
   * sprite.x = rect.left + padding;
   *
   * // Compare positions
   * if (point.x > rect.left) {
   *     console.log('Point is right of rectangle');
   * }
   * ```
   * @readonly
   * @returns The x-coordinate of the left edge
   * @see {@link Rectangle.right} For right edge position
   * @see {@link Rectangle.x} For direct x-coordinate access
   */
  get left() {
    return this.x;
  }
  /**
   * Returns the right edge (x + width) of the rectangle.
   * @example
   * ```ts
   * // Get right edge position
   * const rect = new Rectangle(100, 100, 200, 150);
   * console.log(rect.right); // 300
   *
   * // Align to right edge
   * sprite.x = rect.right - sprite.width;
   *
   * // Check boundaries
   * if (point.x < rect.right) {
   *     console.log('Point is inside right bound');
   * }
   * ```
   * @readonly
   * @returns The x-coordinate of the right edge
   * @see {@link Rectangle.left} For left edge position
   * @see {@link Rectangle.width} For width value
   */
  get right() {
    return this.x + this.width;
  }
  /**
   * Returns the top edge (y-coordinate) of the rectangle.
   * @example
   * ```ts
   * // Get top edge position
   * const rect = new Rectangle(100, 100, 200, 150);
   * console.log(rect.top); // 100
   *
   * // Position above rectangle
   * sprite.y = rect.top - sprite.height;
   *
   * // Check vertical position
   * if (point.y > rect.top) {
   *     console.log('Point is below top edge');
   * }
   * ```
   * @readonly
   * @returns The y-coordinate of the top edge
   * @see {@link Rectangle.bottom} For bottom edge position
   * @see {@link Rectangle.y} For direct y-coordinate access
   */
  get top() {
    return this.y;
  }
  /**
   * Returns the bottom edge (y + height) of the rectangle.
   * @example
   * ```ts
   * // Get bottom edge position
   * const rect = new Rectangle(100, 100, 200, 150);
   * console.log(rect.bottom); // 250
   *
   * // Stack below rectangle
   * sprite.y = rect.bottom + margin;
   *
   * // Check vertical bounds
   * if (point.y < rect.bottom) {
   *     console.log('Point is above bottom edge');
   * }
   * ```
   * @readonly
   * @returns The y-coordinate of the bottom edge
   * @see {@link Rectangle.top} For top edge position
   * @see {@link Rectangle.height} For height value
   */
  get bottom() {
    return this.y + this.height;
  }
  /**
   * Determines whether the Rectangle is empty (has no area).
   * @example
   * ```ts
   * // Check zero dimensions
   * const rect = new Rectangle(100, 100, 0, 50);
   * console.log(rect.isEmpty()); // true
   * ```
   * @returns True if the rectangle has no area
   * @see {@link Rectangle.width} For width value
   * @see {@link Rectangle.height} For height value
   */
  isEmpty() {
    return this.left === this.right || this.top === this.bottom;
  }
  /**
   * A constant empty rectangle. This is a new object every time the property is accessed.
   * @example
   * ```ts
   * // Get fresh empty rectangle
   * const empty = Rectangle.EMPTY;
   * console.log(empty.isEmpty()); // true
   * ```
   * @returns A new empty rectangle instance
   * @see {@link Rectangle.isEmpty} For empty state testing
   */
  static get EMPTY() {
    return new Rectangle(0, 0, 0, 0);
  }
  /**
   * Creates a clone of this Rectangle
   * @example
   * ```ts
   * // Basic cloning
   * const original = new Rectangle(100, 100, 200, 150);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.width *= 2;
   * modified.height += 50;
   *
   * // Verify independence
   * console.log(original.width);  // 200
   * console.log(modified.width);  // 400
   * ```
   * @returns A copy of the rectangle
   * @see {@link Rectangle.copyFrom} For copying into existing rectangle
   * @see {@link Rectangle.copyTo} For copying to another rectangle
   */
  clone() {
    return new Rectangle(this.x, this.y, this.width, this.height);
  }
  /**
   * Converts a Bounds object to a Rectangle object.
   * @example
   * ```ts
   * // Convert bounds to rectangle
   * const bounds = container.getBounds();
   * const rect = new Rectangle().copyFromBounds(bounds);
   * ```
   * @param bounds - The bounds to copy and convert to a rectangle
   * @returns Returns itself
   * @see {@link Bounds} For bounds object structure
   * @see {@link Rectangle.getBounds} For getting rectangle bounds
   */
  copyFromBounds(bounds) {
    this.x = bounds.minX;
    this.y = bounds.minY;
    this.width = bounds.maxX - bounds.minX;
    this.height = bounds.maxY - bounds.minY;
    return this;
  }
  /**
   * Copies another rectangle to this one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Rectangle(100, 100, 200, 150);
   * const target = new Rectangle();
   * target.copyFrom(source);
   *
   * // Chain with other operations
   * const rect = new Rectangle()
   *     .copyFrom(source)
   *     .pad(10);
   * ```
   * @param rectangle - The rectangle to copy from
   * @returns Returns itself
   * @see {@link Rectangle.copyTo} For copying to another rectangle
   * @see {@link Rectangle.clone} For creating new rectangle copy
   */
  copyFrom(rectangle) {
    this.x = rectangle.x;
    this.y = rectangle.y;
    this.width = rectangle.width;
    this.height = rectangle.height;
    return this;
  }
  /**
   * Copies this rectangle to another one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Rectangle(100, 100, 200, 150);
   * const target = new Rectangle();
   * source.copyTo(target);
   *
   * // Chain with other operations
   * const result = source
   *     .copyTo(new Rectangle())
   *     .getBounds();
   * ```
   * @param rectangle - The rectangle to copy to
   * @returns Returns given parameter
   * @see {@link Rectangle.copyFrom} For copying from another rectangle
   * @see {@link Rectangle.clone} For creating new rectangle copy
   */
  copyTo(rectangle) {
    rectangle.copyFrom(this);
    return rectangle;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this Rectangle
   * @example
   * ```ts
   * // Basic containment check
   * const rect = new Rectangle(100, 100, 200, 150);
   * const isInside = rect.contains(150, 125); // true
   * // Check edge cases
   * console.log(rect.contains(100, 100)); // true (on edge)
   * console.log(rect.contains(300, 250)); // false (outside)
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @returns Whether the x/y coordinates are within this Rectangle
   * @see {@link Rectangle.containsRect} For rectangle containment
   * @see {@link Rectangle.strokeContains} For checking stroke intersection
   */
  contains(x2, y2) {
    if (this.width <= 0 || this.height <= 0) {
      return false;
    }
    if (x2 >= this.x && x2 < this.x + this.width) {
      if (y2 >= this.y && y2 < this.y + this.height) {
        return true;
      }
    }
    return false;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this rectangle including the stroke.
   * @example
   * ```ts
   * // Basic stroke check
   * const rect = new Rectangle(100, 100, 200, 150);
   * const isOnStroke = rect.strokeContains(150, 100, 4); // 4px line width
   *
   * // Check with different alignments
   * const innerStroke = rect.strokeContains(150, 100, 4, 1);   // Inside
   * const centerStroke = rect.strokeContains(150, 100, 4, 0.5); // Centered
   * const outerStroke = rect.strokeContains(150, 100, 4, 0);   // Outside
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @param strokeWidth - The width of the line to check
   * @param alignment - The alignment of the stroke (1 = inner, 0.5 = centered, 0 = outer)
   * @returns Whether the x/y coordinates are within this rectangle's stroke
   * @see {@link Rectangle.contains} For checking fill containment
   * @see {@link Rectangle.getBounds} For getting stroke bounds
   */
  strokeContains(x2, y2, strokeWidth, alignment = 0.5) {
    const { width, height } = this;
    if (width <= 0 || height <= 0)
      return false;
    const _x = this.x;
    const _y = this.y;
    const strokeWidthOuter = strokeWidth * (1 - alignment);
    const strokeWidthInner = strokeWidth - strokeWidthOuter;
    const outerLeft = _x - strokeWidthOuter;
    const outerRight = _x + width + strokeWidthOuter;
    const outerTop = _y - strokeWidthOuter;
    const outerBottom = _y + height + strokeWidthOuter;
    const innerLeft = _x + strokeWidthInner;
    const innerRight = _x + width - strokeWidthInner;
    const innerTop = _y + strokeWidthInner;
    const innerBottom = _y + height - strokeWidthInner;
    return x2 >= outerLeft && x2 <= outerRight && y2 >= outerTop && y2 <= outerBottom && !(x2 > innerLeft && x2 < innerRight && y2 > innerTop && y2 < innerBottom);
  }
  /**
   * Determines whether the `other` Rectangle transformed by `transform` intersects with `this` Rectangle object.
   * Returns true only if the area of the intersection is >0, this means that Rectangles
   * sharing a side are not overlapping. Another side effect is that an arealess rectangle
   * (width or height equal to zero) can't intersect any other rectangle.
   * @param {Rectangle} other - The Rectangle to intersect with `this`.
   * @param {Matrix} transform - The transformation matrix of `other`.
   * @returns {boolean} A value of `true` if the transformed `other` Rectangle intersects with `this`; otherwise `false`.
   */
  /**
   * Determines whether the `other` Rectangle transformed by `transform` intersects with `this` Rectangle object.
   *
   * Returns true only if the area of the intersection is greater than 0.
   * This means that rectangles sharing only a side are not considered intersecting.
   * @example
   * ```ts
   * // Basic intersection check
   * const rect1 = new Rectangle(0, 0, 100, 100);
   * const rect2 = new Rectangle(50, 50, 100, 100);
   * console.log(rect1.intersects(rect2)); // true
   *
   * // With transformation matrix
   * const matrix = new Matrix();
   * matrix.rotate(Math.PI / 4); // 45 degrees
   * console.log(rect1.intersects(rect2, matrix)); // Checks with rotation
   *
   * // Edge cases
   * const zeroWidth = new Rectangle(0, 0, 0, 100);
   * console.log(rect1.intersects(zeroWidth)); // false (no area)
   * ```
   * @remarks
   * - Returns true only if intersection area is > 0
   * - Rectangles sharing only a side are not intersecting
   * - Zero-area rectangles cannot intersect anything
   * - Supports optional transformation matrix
   * @param other - The Rectangle to intersect with `this`
   * @param transform - Optional transformation matrix of `other`
   * @returns True if the transformed `other` Rectangle intersects with `this`
   * @see {@link Rectangle.containsRect} For containment testing
   * @see {@link Rectangle.contains} For point testing
   */
  intersects(other, transform) {
    if (!transform) {
      const x02 = this.x < other.x ? other.x : this.x;
      const x12 = this.right > other.right ? other.right : this.right;
      if (x12 <= x02) {
        return false;
      }
      const y02 = this.y < other.y ? other.y : this.y;
      const y12 = this.bottom > other.bottom ? other.bottom : this.bottom;
      return y12 > y02;
    }
    const x0 = this.left;
    const x1 = this.right;
    const y0 = this.top;
    const y1 = this.bottom;
    if (x1 <= x0 || y1 <= y0) {
      return false;
    }
    const lt = tempPoints[0].set(other.left, other.top);
    const lb = tempPoints[1].set(other.left, other.bottom);
    const rt = tempPoints[2].set(other.right, other.top);
    const rb = tempPoints[3].set(other.right, other.bottom);
    if (rt.x <= lt.x || lb.y <= lt.y) {
      return false;
    }
    const s2 = Math.sign(transform.a * transform.d - transform.b * transform.c);
    if (s2 === 0) {
      return false;
    }
    transform.apply(lt, lt);
    transform.apply(lb, lb);
    transform.apply(rt, rt);
    transform.apply(rb, rb);
    if (Math.max(lt.x, lb.x, rt.x, rb.x) <= x0 || Math.min(lt.x, lb.x, rt.x, rb.x) >= x1 || Math.max(lt.y, lb.y, rt.y, rb.y) <= y0 || Math.min(lt.y, lb.y, rt.y, rb.y) >= y1) {
      return false;
    }
    const nx = s2 * (lb.y - lt.y);
    const ny = s2 * (lt.x - lb.x);
    const n00 = nx * x0 + ny * y0;
    const n10 = nx * x1 + ny * y0;
    const n01 = nx * x0 + ny * y1;
    const n11 = nx * x1 + ny * y1;
    if (Math.max(n00, n10, n01, n11) <= nx * lt.x + ny * lt.y || Math.min(n00, n10, n01, n11) >= nx * rb.x + ny * rb.y) {
      return false;
    }
    const mx = s2 * (lt.y - rt.y);
    const my = s2 * (rt.x - lt.x);
    const m00 = mx * x0 + my * y0;
    const m10 = mx * x1 + my * y0;
    const m01 = mx * x0 + my * y1;
    const m11 = mx * x1 + my * y1;
    if (Math.max(m00, m10, m01, m11) <= mx * lt.x + my * lt.y || Math.min(m00, m10, m01, m11) >= mx * rb.x + my * rb.y) {
      return false;
    }
    return true;
  }
  /**
   * Pads the rectangle making it grow in all directions.
   *
   * If paddingY is omitted, both paddingX and paddingY will be set to paddingX.
   * @example
   * ```ts
   * // Basic padding
   * const rect = new Rectangle(100, 100, 200, 150);
   * rect.pad(10); // Adds 10px padding on all sides
   *
   * // Different horizontal and vertical padding
   * const uiRect = new Rectangle(0, 0, 100, 50);
   * uiRect.pad(20, 10); // 20px horizontal, 10px vertical
   * ```
   * @remarks
   * - Adjusts x/y by subtracting padding
   * - Increases width/height by padding * 2
   * - Common in UI layout calculations
   * - Chainable with other methods
   * @param paddingX - The horizontal padding amount
   * @param paddingY - The vertical padding amount
   * @returns Returns itself
   * @see {@link Rectangle.enlarge} For growing to include another rectangle
   * @see {@link Rectangle.fit} For shrinking to fit within another rectangle
   */
  pad(paddingX = 0, paddingY = paddingX) {
    this.x -= paddingX;
    this.y -= paddingY;
    this.width += paddingX * 2;
    this.height += paddingY * 2;
    return this;
  }
  /**
   * Fits this rectangle around the passed one.
   * @example
   * ```ts
   * // Basic fitting
   * const container = new Rectangle(0, 0, 100, 100);
   * const content = new Rectangle(25, 25, 200, 200);
   * content.fit(container); // Clips to container bounds
   * ```
   * @param rectangle - The rectangle to fit around
   * @returns Returns itself
   * @see {@link Rectangle.enlarge} For growing to include another rectangle
   * @see {@link Rectangle.pad} For adding padding around the rectangle
   */
  fit(rectangle) {
    const x1 = Math.max(this.x, rectangle.x);
    const x2 = Math.min(this.x + this.width, rectangle.x + rectangle.width);
    const y1 = Math.max(this.y, rectangle.y);
    const y2 = Math.min(this.y + this.height, rectangle.y + rectangle.height);
    this.x = x1;
    this.width = Math.max(x2 - x1, 0);
    this.y = y1;
    this.height = Math.max(y2 - y1, 0);
    return this;
  }
  /**
   * Enlarges rectangle so that its corners lie on a grid defined by resolution.
   * @example
   * ```ts
   * // Basic grid alignment
   * const rect = new Rectangle(10.2, 10.6, 100.8, 100.4);
   * rect.ceil(); // Aligns to whole pixels
   *
   * // Custom resolution grid
   * const uiRect = new Rectangle(5.3, 5.7, 50.2, 50.8);
   * uiRect.ceil(0.5); // Aligns to half pixels
   *
   * // Use with precision value
   * const preciseRect = new Rectangle(20.001, 20.999, 100.001, 100.999);
   * preciseRect.ceil(1, 0.01); // Handles small decimal variations
   * ```
   * @param resolution - The grid size to align to (1 = whole pixels)
   * @param eps - Small number to prevent floating point errors
   * @returns Returns itself
   * @see {@link Rectangle.fit} For constraining to bounds
   * @see {@link Rectangle.enlarge} For growing dimensions
   */
  ceil(resolution = 1, eps = 1e-3) {
    const x2 = Math.ceil((this.x + this.width - eps) * resolution) / resolution;
    const y2 = Math.ceil((this.y + this.height - eps) * resolution) / resolution;
    this.x = Math.floor((this.x + eps) * resolution) / resolution;
    this.y = Math.floor((this.y + eps) * resolution) / resolution;
    this.width = x2 - this.x;
    this.height = y2 - this.y;
    return this;
  }
  /**
   * Scales the rectangle's dimensions and position by the specified factors.
   * @example
   * ```ts
   * const rect = new Rectangle(50, 50, 100, 100);
   *
   * // Scale uniformly
   * rect.scale(0.5, 0.5);
   * // rect is now: x=25, y=25, width=50, height=50
   *
   * // non-uniformly
   * rect.scale(0.5, 1);
   * // rect is now: x=25, y=50, width=50, height=100
   * ```
   * @param x - The factor by which to scale the horizontal properties (x, width).
   * @param y - The factor by which to scale the vertical properties (y, height).
   * @returns Returns itself
   */
  scale(x2, y2 = x2) {
    this.x *= x2;
    this.y *= y2;
    this.width *= x2;
    this.height *= y2;
    return this;
  }
  /**
   * Enlarges this rectangle to include the passed rectangle.
   * @example
   * ```ts
   * // Basic enlargement
   * const rect = new Rectangle(50, 50, 100, 100);
   * const other = new Rectangle(0, 0, 200, 75);
   * rect.enlarge(other);
   * // rect is now: x=0, y=0, width=200, height=150
   *
   * // Use for bounding box calculation
   * const bounds = new Rectangle();
   * objects.forEach((obj) => {
   *     bounds.enlarge(obj.getBounds());
   * });
   * ```
   * @param rectangle - The rectangle to include
   * @returns Returns itself
   * @see {@link Rectangle.fit} For shrinking to fit within another rectangle
   * @see {@link Rectangle.pad} For adding padding around the rectangle
   */
  enlarge(rectangle) {
    const x1 = Math.min(this.x, rectangle.x);
    const x2 = Math.max(this.x + this.width, rectangle.x + rectangle.width);
    const y1 = Math.min(this.y, rectangle.y);
    const y2 = Math.max(this.y + this.height, rectangle.y + rectangle.height);
    this.x = x1;
    this.width = x2 - x1;
    this.y = y1;
    this.height = y2 - y1;
    return this;
  }
  /**
   * Returns the framing rectangle of the rectangle as a Rectangle object
   * @example
   * ```ts
   * // Basic bounds retrieval
   * const rect = new Rectangle(100, 100, 200, 150);
   * const bounds = rect.getBounds();
   *
   * // Reuse existing rectangle
   * const out = new Rectangle();
   * rect.getBounds(out);
   * ```
   * @param out - Optional rectangle to store the result
   * @returns The framing rectangle
   * @see {@link Rectangle.copyFrom} For direct copying
   * @see {@link Rectangle.clone} For creating new copy
   */
  getBounds(out2) {
    out2 || (out2 = new Rectangle());
    out2.copyFrom(this);
    return out2;
  }
  /**
   * Determines whether another Rectangle is fully contained within this Rectangle.
   *
   * Rectangles that occupy the same space are considered to be containing each other.
   *
   * Rectangles without area (width or height equal to zero) can't contain anything,
   * not even other arealess rectangles.
   * @example
   * ```ts
   * // Check if one rectangle contains another
   * const container = new Rectangle(0, 0, 100, 100);
   * const inner = new Rectangle(25, 25, 50, 50);
   *
   * console.log(container.containsRect(inner)); // true
   *
   * // Check overlapping rectangles
   * const partial = new Rectangle(75, 75, 50, 50);
   * console.log(container.containsRect(partial)); // false
   *
   * // Zero-area rectangles
   * const empty = new Rectangle(0, 0, 0, 100);
   * console.log(container.containsRect(empty)); // false
   * ```
   * @param other - The Rectangle to check for containment
   * @returns True if other is fully contained within this Rectangle
   * @see {@link Rectangle.contains} For point containment
   * @see {@link Rectangle.intersects} For overlap testing
   */
  containsRect(other) {
    if (this.width <= 0 || this.height <= 0)
      return false;
    const x1 = other.x;
    const y1 = other.y;
    const x2 = other.x + other.width;
    const y2 = other.y + other.height;
    return x1 >= this.x && x1 < this.x + this.width && y1 >= this.y && y1 < this.y + this.height && x2 >= this.x && x2 < this.x + this.width && y2 >= this.y && y2 < this.y + this.height;
  }
  /**
   * Sets the position and dimensions of the rectangle.
   * @example
   * ```ts
   * // Basic usage
   * const rect = new Rectangle();
   * rect.set(100, 100, 200, 150);
   *
   * // Chain with other operations
   * const bounds = new Rectangle()
   *     .set(0, 0, 100, 100)
   *     .pad(10);
   * ```
   * @param x - The X coordinate of the upper-left corner of the rectangle
   * @param y - The Y coordinate of the upper-left corner of the rectangle
   * @param width - The overall width of the rectangle
   * @param height - The overall height of the rectangle
   * @returns Returns itself for method chaining
   * @see {@link Rectangle.copyFrom} For copying from another rectangle
   * @see {@link Rectangle.clone} For creating a new copy
   */
  set(x2, y2, width, height) {
    this.x = x2;
    this.y = y2;
    this.width = width;
    this.height = height;
    return this;
  }
  toString() {
    return `[pixi.js/math:Rectangle x=${this.x} y=${this.y} width=${this.width} height=${this.height}]`;
  }
}
const uidCache = {
  default: -1
};
function uid(name = "default") {
  if (uidCache[name] === void 0) {
    uidCache[name] = -1;
  }
  return ++uidCache[name];
}
const warnings = /* @__PURE__ */ new Set();
const v8_0_0 = "8.0.0";
const v8_3_4 = "8.3.4";
const deprecationState = {
  quiet: false,
  noColor: false
};
const deprecation = (version, message, ignoreDepth = 3) => {
  if (deprecationState.quiet || warnings.has(message))
    return;
  let stack = new Error().stack;
  const deprecationMessage = `${message}
Deprecated since v${version}`;
  const useGroup = typeof console.groupCollapsed === "function" && !deprecationState.noColor;
  if (typeof stack === "undefined") {
    console.warn("PixiJS Deprecation Warning: ", deprecationMessage);
  } else {
    stack = stack.split("\n").splice(ignoreDepth).join("\n");
    if (useGroup) {
      console.groupCollapsed(
        "%cPixiJS Deprecation Warning: %c%s",
        "color:#614108;background:#fffbe6",
        "font-weight:normal;color:#614108;background:#fffbe6",
        deprecationMessage
      );
      console.warn(stack);
      console.groupEnd();
    } else {
      console.warn("PixiJS Deprecation Warning: ", deprecationMessage);
      console.warn(stack);
    }
  }
  warnings.add(message);
};
Object.defineProperties(deprecation, {
  quiet: {
    get: () => deprecationState.quiet,
    set: (value) => {
      deprecationState.quiet = value;
    },
    enumerable: true,
    configurable: false
  },
  noColor: {
    get: () => deprecationState.noColor,
    set: (value) => {
      deprecationState.noColor = value;
    },
    enumerable: true,
    configurable: false
  }
});
const NOOP = () => {
};
function nextPow2(v2) {
  v2 += v2 === 0 ? 1 : 0;
  --v2;
  v2 |= v2 >>> 1;
  v2 |= v2 >>> 2;
  v2 |= v2 >>> 4;
  v2 |= v2 >>> 8;
  v2 |= v2 >>> 16;
  return v2 + 1;
}
function isPow2(v2) {
  return !(v2 & v2 - 1) && !!v2;
}
function definedProps(obj) {
  const result = {};
  for (const key in obj) {
    if (obj[key] !== void 0) {
      result[key] = obj[key];
    }
  }
  return result;
}
const idHash$1 = /* @__PURE__ */ Object.create(null);
function createResourceIdFromString(value) {
  const id = idHash$1[value];
  if (id === void 0) {
    idHash$1[value] = uid("resource");
  }
  return id;
}
const _TextureStyle = class _TextureStyle2 extends EventEmitter {
  /**
   * @param options - options for the style
   */
  constructor(options = {}) {
    super();
    this._resourceType = "textureSampler";
    this._touched = 0;
    this._maxAnisotropy = 1;
    this.destroyed = false;
    options = { ..._TextureStyle2.defaultOptions, ...options };
    this.addressMode = options.addressMode;
    this.addressModeU = options.addressModeU ?? this.addressModeU;
    this.addressModeV = options.addressModeV ?? this.addressModeV;
    this.addressModeW = options.addressModeW ?? this.addressModeW;
    this.scaleMode = options.scaleMode;
    this.magFilter = options.magFilter ?? this.magFilter;
    this.minFilter = options.minFilter ?? this.minFilter;
    this.mipmapFilter = options.mipmapFilter ?? this.mipmapFilter;
    this.lodMinClamp = options.lodMinClamp;
    this.lodMaxClamp = options.lodMaxClamp;
    this.compare = options.compare;
    this.maxAnisotropy = options.maxAnisotropy ?? 1;
  }
  set addressMode(value) {
    this.addressModeU = value;
    this.addressModeV = value;
    this.addressModeW = value;
  }
  /** setting this will set wrapModeU,wrapModeV and wrapModeW all at once! */
  get addressMode() {
    return this.addressModeU;
  }
  set wrapMode(value) {
    deprecation(v8_0_0, "TextureStyle.wrapMode is now TextureStyle.addressMode");
    this.addressMode = value;
  }
  get wrapMode() {
    return this.addressMode;
  }
  set scaleMode(value) {
    this.magFilter = value;
    this.minFilter = value;
    this.mipmapFilter = value;
  }
  /** setting this will set magFilter,minFilter and mipmapFilter all at once!  */
  get scaleMode() {
    return this.magFilter;
  }
  /** Specifies the maximum anisotropy value clamp used by the sampler. */
  set maxAnisotropy(value) {
    this._maxAnisotropy = Math.min(value, 16);
    if (this._maxAnisotropy > 1) {
      this.scaleMode = "linear";
    }
  }
  get maxAnisotropy() {
    return this._maxAnisotropy;
  }
  // TODO - move this to WebGL?
  get _resourceId() {
    return this._sharedResourceId || this._generateResourceId();
  }
  update() {
    this._sharedResourceId = null;
    this.emit("change", this);
  }
  _generateResourceId() {
    const bigKey = `${this.addressModeU}-${this.addressModeV}-${this.addressModeW}-${this.magFilter}-${this.minFilter}-${this.mipmapFilter}-${this.lodMinClamp}-${this.lodMaxClamp}-${this.compare}-${this._maxAnisotropy}`;
    this._sharedResourceId = createResourceIdFromString(bigKey);
    return this._resourceId;
  }
  /** Destroys the style */
  destroy() {
    this.destroyed = true;
    this.emit("destroy", this);
    this.emit("change", this);
    this.removeAllListeners();
  }
};
_TextureStyle.defaultOptions = {
  addressMode: "clamp-to-edge",
  scaleMode: "linear"
};
let TextureStyle = _TextureStyle;
const _TextureSource = class _TextureSource2 extends EventEmitter {
  /**
   * @param options - options for creating a new TextureSource
   */
  constructor(options = {}) {
    super();
    this.options = options;
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this._gcLastUsed = -1;
    this.uid = uid("textureSource");
    this._resourceType = "textureSource";
    this._resourceId = uid("resource");
    this.uploadMethodId = "unknown";
    this._resolution = 1;
    this.pixelWidth = 1;
    this.pixelHeight = 1;
    this.width = 1;
    this.height = 1;
    this.sampleCount = 1;
    this.mipLevelCount = 1;
    this.autoGenerateMipmaps = false;
    this.format = "rgba8unorm";
    this.dimension = "2d";
    this.antialias = false;
    this._touched = 0;
    this._batchTick = -1;
    this._textureBindLocation = -1;
    options = { ..._TextureSource2.defaultOptions, ...options };
    this.label = options.label ?? "";
    this.resource = options.resource;
    this.autoGarbageCollect = options.autoGarbageCollect;
    this._resolution = options.resolution;
    if (options.width) {
      this.pixelWidth = options.width * this._resolution;
    } else {
      this.pixelWidth = this.resource ? this.resourceWidth ?? 1 : 1;
    }
    if (options.height) {
      this.pixelHeight = options.height * this._resolution;
    } else {
      this.pixelHeight = this.resource ? this.resourceHeight ?? 1 : 1;
    }
    this.width = this.pixelWidth / this._resolution;
    this.height = this.pixelHeight / this._resolution;
    this.format = options.format;
    this.dimension = options.dimensions;
    this.mipLevelCount = options.mipLevelCount;
    this.autoGenerateMipmaps = options.autoGenerateMipmaps;
    this.sampleCount = options.sampleCount;
    this.antialias = options.antialias;
    this.alphaMode = options.alphaMode;
    this.style = new TextureStyle(definedProps(options));
    this.destroyed = false;
    this._refreshPOT();
  }
  /** returns itself */
  get source() {
    return this;
  }
  /** the style of the texture */
  get style() {
    return this._style;
  }
  set style(value) {
    if (this.style === value)
      return;
    this._style?.off("change", this._onStyleChange, this);
    this._style = value;
    this._style?.on("change", this._onStyleChange, this);
    this._onStyleChange();
  }
  /** Specifies the maximum anisotropy value clamp used by the sampler. */
  set maxAnisotropy(value) {
    this._style.maxAnisotropy = value;
  }
  get maxAnisotropy() {
    return this._style.maxAnisotropy;
  }
  /** setting this will set wrapModeU, wrapModeV and wrapModeW all at once! */
  get addressMode() {
    return this._style.addressMode;
  }
  set addressMode(value) {
    this._style.addressMode = value;
  }
  /** setting this will set wrapModeU, wrapModeV and wrapModeW all at once! */
  get repeatMode() {
    return this._style.addressMode;
  }
  set repeatMode(value) {
    this._style.addressMode = value;
  }
  /** Specifies the sampling behavior when the sample footprint is smaller than or equal to one texel. */
  get magFilter() {
    return this._style.magFilter;
  }
  set magFilter(value) {
    this._style.magFilter = value;
  }
  /** Specifies the sampling behavior when the sample footprint is larger than one texel. */
  get minFilter() {
    return this._style.minFilter;
  }
  set minFilter(value) {
    this._style.minFilter = value;
  }
  /** Specifies behavior for sampling between mipmap levels. */
  get mipmapFilter() {
    return this._style.mipmapFilter;
  }
  set mipmapFilter(value) {
    this._style.mipmapFilter = value;
  }
  /** Specifies the minimum and maximum levels of detail, respectively, used internally when sampling a texture. */
  get lodMinClamp() {
    return this._style.lodMinClamp;
  }
  set lodMinClamp(value) {
    this._style.lodMinClamp = value;
  }
  /** Specifies the minimum and maximum levels of detail, respectively, used internally when sampling a texture. */
  get lodMaxClamp() {
    return this._style.lodMaxClamp;
  }
  set lodMaxClamp(value) {
    this._style.lodMaxClamp = value;
  }
  _onStyleChange() {
    this.emit("styleChange", this);
  }
  /** call this if you have modified the texture outside of the constructor */
  update() {
    if (this.resource) {
      const resolution = this._resolution;
      const didResize = this.resize(this.resourceWidth / resolution, this.resourceHeight / resolution);
      if (didResize)
        return;
    }
    this.emit("update", this);
  }
  /** Destroys this texture source */
  destroy() {
    this.destroyed = true;
    this.unload();
    this.emit("destroy", this);
    if (this._style) {
      this._style.destroy();
      this._style = null;
    }
    this.uploadMethodId = null;
    this.resource = null;
    this.removeAllListeners();
  }
  /**
   * This will unload the Texture source from the GPU. This will free up the GPU memory
   * As soon as it is required fore rendering, it will be re-uploaded.
   */
  unload() {
    this._resourceId = uid("resource");
    this.emit("change", this);
    this.emit("unload", this);
    for (const key in this._gpuData) {
      this._gpuData[key]?.destroy?.();
    }
    this._gpuData = /* @__PURE__ */ Object.create(null);
  }
  /** the width of the resource. This is the REAL pure number, not accounting resolution   */
  get resourceWidth() {
    const { resource } = this;
    return resource.naturalWidth || resource.videoWidth || resource.displayWidth || resource.width;
  }
  /** the height of the resource. This is the REAL pure number, not accounting resolution */
  get resourceHeight() {
    const { resource } = this;
    return resource.naturalHeight || resource.videoHeight || resource.displayHeight || resource.height;
  }
  /**
   * the resolution of the texture. Changing this number, will not change the number of pixels in the actual texture
   * but will the size of the texture when rendered.
   *
   * changing the resolution of this texture to 2 for example will make it appear twice as small when rendered (as pixel
   * density will have increased)
   */
  get resolution() {
    return this._resolution;
  }
  set resolution(resolution) {
    if (this._resolution === resolution)
      return;
    this._resolution = resolution;
    this.width = this.pixelWidth / resolution;
    this.height = this.pixelHeight / resolution;
  }
  /**
   * Resize the texture, this is handy if you want to use the texture as a render texture
   * @param width - the new width of the texture
   * @param height - the new height of the texture
   * @param resolution - the new resolution of the texture
   * @returns - if the texture was resized
   */
  resize(width, height, resolution) {
    resolution || (resolution = this._resolution);
    width || (width = this.width);
    height || (height = this.height);
    const newPixelWidth = Math.round(width * resolution);
    const newPixelHeight = Math.round(height * resolution);
    this.width = newPixelWidth / resolution;
    this.height = newPixelHeight / resolution;
    this._resolution = resolution;
    if (this.pixelWidth === newPixelWidth && this.pixelHeight === newPixelHeight) {
      return false;
    }
    this._refreshPOT();
    this.pixelWidth = newPixelWidth;
    this.pixelHeight = newPixelHeight;
    this.emit("resize", this);
    this._resourceId = uid("resource");
    this.emit("change", this);
    return true;
  }
  /**
   * Lets the renderer know that this texture has been updated and its mipmaps should be re-generated.
   * This is only important for RenderTexture instances, as standard Texture instances will have their
   * mipmaps generated on upload. You should call this method after you make any change to the texture
   *
   * The reason for this is is can be quite expensive to update mipmaps for a texture. So by default,
   * We want you, the developer to specify when this action should happen.
   *
   * Generally you don't want to have mipmaps generated on Render targets that are changed every frame,
   */
  updateMipmaps() {
    if (this.autoGenerateMipmaps && this.mipLevelCount > 1) {
      this.emit("updateMipmaps", this);
    }
  }
  set wrapMode(value) {
    this._style.wrapMode = value;
  }
  get wrapMode() {
    return this._style.wrapMode;
  }
  set scaleMode(value) {
    this._style.scaleMode = value;
  }
  /** setting this will set magFilter,minFilter and mipmapFilter all at once!  */
  get scaleMode() {
    return this._style.scaleMode;
  }
  /**
   * Refresh check for isPowerOfTwo texture based on size
   * @private
   */
  _refreshPOT() {
    this.isPowerOfTwo = isPow2(this.pixelWidth) && isPow2(this.pixelHeight);
  }
  static test(_resource) {
    throw new Error("Unimplemented");
  }
};
_TextureSource.defaultOptions = {
  resolution: 1,
  format: "bgra8unorm",
  alphaMode: "premultiply-alpha-on-upload",
  dimensions: "2d",
  mipLevelCount: 1,
  autoGenerateMipmaps: false,
  sampleCount: 1,
  antialias: false,
  autoGarbageCollect: false
};
let TextureSource = _TextureSource;
class BufferImageSource extends TextureSource {
  constructor(options) {
    const buffer = options.resource || new Float32Array(options.width * options.height * 4);
    let format = options.format;
    if (!format) {
      if (buffer instanceof Float32Array) {
        format = "rgba32float";
      } else if (buffer instanceof Int32Array) {
        format = "rgba32uint";
      } else if (buffer instanceof Uint32Array) {
        format = "rgba32uint";
      } else if (buffer instanceof Int16Array) {
        format = "rgba16uint";
      } else if (buffer instanceof Uint16Array) {
        format = "rgba16uint";
      } else if (buffer instanceof Int8Array) {
        format = "bgra8unorm";
      } else {
        format = "bgra8unorm";
      }
    }
    super({
      ...options,
      resource: buffer,
      format
    });
    this.uploadMethodId = "buffer";
  }
  static test(resource) {
    return resource instanceof Int8Array || resource instanceof Uint8Array || resource instanceof Uint8ClampedArray || resource instanceof Int16Array || resource instanceof Uint16Array || resource instanceof Int32Array || resource instanceof Uint32Array || resource instanceof Float32Array;
  }
}
BufferImageSource.extension = ExtensionType.TextureSource;
const tempMat = new Matrix();
class TextureMatrix {
  /**
   * @param texture - observed texture
   * @param clampMargin - Changes frame clamping, 0.5 by default. Use -0.5 for extra border.
   */
  constructor(texture, clampMargin) {
    this.mapCoord = new Matrix();
    this.uClampFrame = new Float32Array(4);
    this.uClampOffset = new Float32Array(2);
    this._textureID = -1;
    this._updateID = 0;
    this.clampOffset = 0;
    if (typeof clampMargin === "undefined") {
      this.clampMargin = texture.width < 10 ? 0 : 0.5;
    } else {
      this.clampMargin = clampMargin;
    }
    this.isSimple = false;
    this.texture = texture;
  }
  /** Texture property. */
  get texture() {
    return this._texture;
  }
  set texture(value) {
    if (this.texture === value)
      return;
    this._texture?.removeListener("update", this.update, this);
    this._texture = value;
    this._texture.addListener("update", this.update, this);
    this.update();
  }
  /**
   * Multiplies uvs array to transform
   * @param uvs - mesh uvs
   * @param [out=uvs] - output
   * @returns - output
   */
  multiplyUvs(uvs, out2) {
    if (out2 === void 0) {
      out2 = uvs;
    }
    const mat = this.mapCoord;
    for (let i2 = 0; i2 < uvs.length; i2 += 2) {
      const x2 = uvs[i2];
      const y2 = uvs[i2 + 1];
      out2[i2] = x2 * mat.a + y2 * mat.c + mat.tx;
      out2[i2 + 1] = x2 * mat.b + y2 * mat.d + mat.ty;
    }
    return out2;
  }
  /**
   * Updates matrices if texture was changed
   * @returns - whether or not it was updated
   */
  update() {
    const tex = this._texture;
    this._updateID++;
    const uvs = tex.uvs;
    this.mapCoord.set(uvs.x1 - uvs.x0, uvs.y1 - uvs.y0, uvs.x3 - uvs.x0, uvs.y3 - uvs.y0, uvs.x0, uvs.y0);
    const orig = tex.orig;
    const trim = tex.trim;
    if (trim) {
      tempMat.set(
        orig.width / trim.width,
        0,
        0,
        orig.height / trim.height,
        -trim.x / trim.width,
        -trim.y / trim.height
      );
      this.mapCoord.append(tempMat);
    }
    const texBase = tex.source;
    const frame = this.uClampFrame;
    const margin = this.clampMargin / texBase._resolution;
    const offset = this.clampOffset / texBase._resolution;
    frame[0] = (tex.frame.x + margin + offset) / texBase.width;
    frame[1] = (tex.frame.y + margin + offset) / texBase.height;
    frame[2] = (tex.frame.x + tex.frame.width - margin + offset) / texBase.width;
    frame[3] = (tex.frame.y + tex.frame.height - margin + offset) / texBase.height;
    this.uClampOffset[0] = this.clampOffset / texBase.pixelWidth;
    this.uClampOffset[1] = this.clampOffset / texBase.pixelHeight;
    this.isSimple = tex.frame.width === texBase.width && tex.frame.height === texBase.height && tex.rotate === 0;
    return true;
  }
}
class Texture extends EventEmitter {
  /**
   * @param {TextureOptions} options - Options for the texture
   */
  constructor({
    source: source2,
    label,
    frame,
    orig,
    trim,
    defaultAnchor,
    defaultBorders,
    rotate,
    dynamic
  } = {}) {
    super();
    this.uid = uid("texture");
    this.uvs = { x0: 0, y0: 0, x1: 0, y1: 0, x2: 0, y2: 0, x3: 0, y3: 0 };
    this.frame = new Rectangle();
    this.noFrame = false;
    this.dynamic = false;
    this.isTexture = true;
    this.label = label;
    this.source = source2?.source ?? new TextureSource();
    this.noFrame = !frame;
    if (frame) {
      this.frame.copyFrom(frame);
    } else {
      const { width, height } = this._source;
      this.frame.width = width;
      this.frame.height = height;
    }
    this.orig = orig || this.frame;
    this.trim = trim;
    this.rotate = rotate ?? 0;
    this.defaultAnchor = defaultAnchor;
    this.defaultBorders = defaultBorders;
    this.destroyed = false;
    this.dynamic = dynamic || false;
    this.updateUvs();
  }
  set source(value) {
    if (this._source) {
      this._source.off("resize", this.update, this);
    }
    this._source = value;
    value.on("resize", this.update, this);
    this.emit("update", this);
  }
  /** the underlying source of the texture (equivalent of baseTexture in v7) */
  get source() {
    return this._source;
  }
  /** returns a TextureMatrix instance for this texture. By default, that object is not created because its heavy. */
  get textureMatrix() {
    if (!this._textureMatrix) {
      this._textureMatrix = new TextureMatrix(this);
    }
    return this._textureMatrix;
  }
  /** The width of the Texture in pixels. */
  get width() {
    return this.orig.width;
  }
  /** The height of the Texture in pixels. */
  get height() {
    return this.orig.height;
  }
  /** Call this function when you have modified the frame of this texture. */
  updateUvs() {
    const { uvs, frame } = this;
    const { width, height } = this._source;
    const nX = frame.x / width;
    const nY = frame.y / height;
    const nW = frame.width / width;
    const nH = frame.height / height;
    let rotate = this.rotate;
    if (rotate) {
      const w2 = nW / 2;
      const h2 = nH / 2;
      const cX = nX + w2;
      const cY = nY + h2;
      rotate = groupD8.add(rotate, groupD8.NW);
      uvs.x0 = cX + w2 * groupD8.uX(rotate);
      uvs.y0 = cY + h2 * groupD8.uY(rotate);
      rotate = groupD8.add(rotate, 2);
      uvs.x1 = cX + w2 * groupD8.uX(rotate);
      uvs.y1 = cY + h2 * groupD8.uY(rotate);
      rotate = groupD8.add(rotate, 2);
      uvs.x2 = cX + w2 * groupD8.uX(rotate);
      uvs.y2 = cY + h2 * groupD8.uY(rotate);
      rotate = groupD8.add(rotate, 2);
      uvs.x3 = cX + w2 * groupD8.uX(rotate);
      uvs.y3 = cY + h2 * groupD8.uY(rotate);
    } else {
      uvs.x0 = nX;
      uvs.y0 = nY;
      uvs.x1 = nX + nW;
      uvs.y1 = nY;
      uvs.x2 = nX + nW;
      uvs.y2 = nY + nH;
      uvs.x3 = nX;
      uvs.y3 = nY + nH;
    }
  }
  /**
   * Destroys this texture
   * @param destroySource - Destroy the source when the texture is destroyed.
   */
  destroy(destroySource = false) {
    if (this._source) {
      this._source.off("resize", this.update, this);
      if (destroySource) {
        this._source.destroy();
        this._source = null;
      }
    }
    this._textureMatrix = null;
    this.destroyed = true;
    this.emit("destroy", this);
    this.removeAllListeners();
  }
  /**
   * Call this if you have modified the `texture outside` of the constructor.
   *
   * If you have modified this texture's source, you must separately call `texture.source.update()` to see those changes.
   */
  update() {
    if (this.noFrame) {
      this.frame.width = this._source.width;
      this.frame.height = this._source.height;
    }
    this.updateUvs();
    this.emit("update", this);
  }
  /** @deprecated since 8.0.0 */
  get baseTexture() {
    deprecation(v8_0_0, "Texture.baseTexture is now Texture.source");
    return this._source;
  }
}
Texture.EMPTY = new Texture({
  label: "EMPTY",
  source: new TextureSource({
    label: "EMPTY"
  })
});
Texture.EMPTY.destroy = NOOP;
Texture.WHITE = new Texture({
  source: new BufferImageSource({
    resource: new Uint8Array([255, 255, 255, 255]),
    width: 1,
    height: 1,
    alphaMode: "premultiply-alpha-on-upload",
    label: "WHITE"
  }),
  label: "WHITE"
});
Texture.WHITE.destroy = NOOP;
function updateQuadBounds(bounds, anchor, texture) {
  const { width, height } = texture.orig;
  const trim = texture.trim;
  if (trim) {
    const sourceWidth = trim.width;
    const sourceHeight = trim.height;
    bounds.minX = trim.x - anchor._x * width;
    bounds.maxX = bounds.minX + sourceWidth;
    bounds.minY = trim.y - anchor._y * height;
    bounds.maxY = bounds.minY + sourceHeight;
  } else {
    bounds.minX = -anchor._x * width;
    bounds.maxX = bounds.minX + width;
    bounds.minY = -anchor._y * height;
    bounds.maxY = bounds.minY + height;
  }
}
const defaultMatrix = new Matrix();
class Bounds {
  /**
   * Creates a new Bounds object.
   * @param minX - The minimum X coordinate of the bounds.
   * @param minY - The minimum Y coordinate of the bounds.
   * @param maxX - The maximum X coordinate of the bounds.
   * @param maxY - The maximum Y coordinate of the bounds.
   */
  constructor(minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity) {
    this.minX = Infinity;
    this.minY = Infinity;
    this.maxX = -Infinity;
    this.maxY = -Infinity;
    this.matrix = defaultMatrix;
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  }
  /**
   * Checks if bounds are empty, meaning either width or height is zero or negative.
   * Empty bounds occur when min values exceed max values on either axis.
   * @example
   * ```ts
   * const bounds = new Bounds();
   *
   * // Check if newly created bounds are empty
   * console.log(bounds.isEmpty()); // true, default bounds are empty
   *
   * // Add frame and check again
   * bounds.addFrame(0, 0, 100, 100);
   * console.log(bounds.isEmpty()); // false, bounds now have area
   *
   * // Clear bounds
   * bounds.clear();
   * console.log(bounds.isEmpty()); // true, bounds are empty again
   * ```
   * @returns True if bounds are empty (have no area)
   * @see {@link Bounds#clear} For resetting bounds
   * @see {@link Bounds#isValid} For checking validity
   */
  isEmpty() {
    return this.minX > this.maxX || this.minY > this.maxY;
  }
  /**
   * The bounding rectangle representation of these bounds.
   * Lazily creates and updates a Rectangle instance based on the current bounds.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   *
   * // Get rectangle representation
   * const rect = bounds.rectangle;
   * console.log(rect.x, rect.y, rect.width, rect.height);
   *
   * // Use for hit testing
   * if (bounds.rectangle.contains(mouseX, mouseY)) {
   *     console.log('Mouse is inside bounds!');
   * }
   * ```
   * @see {@link Rectangle} For rectangle methods
   * @see {@link Bounds.isEmpty} For bounds validation
   */
  get rectangle() {
    if (!this._rectangle) {
      this._rectangle = new Rectangle();
    }
    const rectangle = this._rectangle;
    if (this.minX > this.maxX || this.minY > this.maxY) {
      rectangle.x = 0;
      rectangle.y = 0;
      rectangle.width = 0;
      rectangle.height = 0;
    } else {
      rectangle.copyFromBounds(this);
    }
    return rectangle;
  }
  /**
   * Clears the bounds and resets all coordinates to their default values.
   * Resets the transformation matrix back to identity.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * console.log(bounds.isEmpty()); // false
   * // Clear the bounds
   * bounds.clear();
   * console.log(bounds.isEmpty()); // true
   * ```
   * @returns This bounds object for chaining
   */
  clear() {
    this.minX = Infinity;
    this.minY = Infinity;
    this.maxX = -Infinity;
    this.maxY = -Infinity;
    this.matrix = defaultMatrix;
    return this;
  }
  /**
   * Sets the bounds directly using coordinate values.
   * Provides a way to set all bounds values at once.
   * @example
   * ```ts
   * const bounds = new Bounds();
   * bounds.set(0, 0, 100, 100);
   * ```
   * @param x0 - Left X coordinate of frame
   * @param y0 - Top Y coordinate of frame
   * @param x1 - Right X coordinate of frame
   * @param y1 - Bottom Y coordinate of frame
   * @see {@link Bounds#addFrame} For matrix-aware bounds setting
   * @see {@link Bounds#clear} For resetting bounds
   */
  set(x0, y0, x1, y1) {
    this.minX = x0;
    this.minY = y0;
    this.maxX = x1;
    this.maxY = y1;
  }
  /**
   * Adds a rectangular frame to the bounds, optionally transformed by a matrix.
   * Updates the bounds to encompass the new frame coordinates.
   * @example
   * ```ts
   * const bounds = new Bounds();
   * bounds.addFrame(0, 0, 100, 100);
   *
   * // Add transformed frame
   * const matrix = new Matrix()
   *     .translate(50, 50)
   *     .rotate(Math.PI / 4);
   * bounds.addFrame(0, 0, 100, 100, matrix);
   * ```
   * @param x0 - Left X coordinate of frame
   * @param y0 - Top Y coordinate of frame
   * @param x1 - Right X coordinate of frame
   * @param y1 - Bottom Y coordinate of frame
   * @param matrix - Optional transformation matrix
   * @see {@link Bounds#addRect} For adding Rectangle objects
   * @see {@link Bounds#addBounds} For adding other Bounds
   */
  addFrame(x0, y0, x1, y1, matrix) {
    matrix || (matrix = this.matrix);
    const a2 = matrix.a;
    const b2 = matrix.b;
    const c2 = matrix.c;
    const d2 = matrix.d;
    const tx = matrix.tx;
    const ty = matrix.ty;
    let minX = this.minX;
    let minY = this.minY;
    let maxX = this.maxX;
    let maxY = this.maxY;
    let x2 = a2 * x0 + c2 * y0 + tx;
    let y2 = b2 * x0 + d2 * y0 + ty;
    if (x2 < minX)
      minX = x2;
    if (y2 < minY)
      minY = y2;
    if (x2 > maxX)
      maxX = x2;
    if (y2 > maxY)
      maxY = y2;
    x2 = a2 * x1 + c2 * y0 + tx;
    y2 = b2 * x1 + d2 * y0 + ty;
    if (x2 < minX)
      minX = x2;
    if (y2 < minY)
      minY = y2;
    if (x2 > maxX)
      maxX = x2;
    if (y2 > maxY)
      maxY = y2;
    x2 = a2 * x0 + c2 * y1 + tx;
    y2 = b2 * x0 + d2 * y1 + ty;
    if (x2 < minX)
      minX = x2;
    if (y2 < minY)
      minY = y2;
    if (x2 > maxX)
      maxX = x2;
    if (y2 > maxY)
      maxY = y2;
    x2 = a2 * x1 + c2 * y1 + tx;
    y2 = b2 * x1 + d2 * y1 + ty;
    if (x2 < minX)
      minX = x2;
    if (y2 < minY)
      minY = y2;
    if (x2 > maxX)
      maxX = x2;
    if (y2 > maxY)
      maxY = y2;
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  }
  /**
   * Adds a rectangle to the bounds, optionally transformed by a matrix.
   * Updates the bounds to encompass the given rectangle.
   * @example
   * ```ts
   * const bounds = new Bounds();
   * // Add simple rectangle
   * const rect = new Rectangle(0, 0, 100, 100);
   * bounds.addRect(rect);
   *
   * // Add transformed rectangle
   * const matrix = new Matrix()
   *     .translate(50, 50)
   *     .rotate(Math.PI / 4);
   * bounds.addRect(rect, matrix);
   * ```
   * @param rect - The rectangle to be added
   * @param matrix - Optional transformation matrix
   * @see {@link Bounds#addFrame} For adding raw coordinates
   * @see {@link Bounds#addBounds} For adding other bounds
   */
  addRect(rect, matrix) {
    this.addFrame(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, matrix);
  }
  /**
   * Adds another bounds object to this one, optionally transformed by a matrix.
   * Expands the bounds to include the given bounds' area.
   * @example
   * ```ts
   * const bounds = new Bounds();
   *
   * // Add child bounds
   * const childBounds = sprite.getBounds();
   * bounds.addBounds(childBounds);
   *
   * // Add transformed bounds
   * const matrix = new Matrix()
   *     .scale(2, 2);
   * bounds.addBounds(childBounds, matrix);
   * ```
   * @param bounds - The bounds to be added
   * @param matrix - Optional transformation matrix
   * @see {@link Bounds#addFrame} For adding raw coordinates
   * @see {@link Bounds#addRect} For adding rectangles
   */
  addBounds(bounds, matrix) {
    this.addFrame(bounds.minX, bounds.minY, bounds.maxX, bounds.maxY, matrix);
  }
  /**
   * Adds other Bounds as a mask, creating an intersection of the two bounds.
   * Only keeps the overlapping region between current bounds and mask bounds.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Create mask bounds
   * const mask = new Bounds();
   * mask.addFrame(50, 50, 150, 150);
   * // Apply mask - results in bounds of (50,50,100,100)
   * bounds.addBoundsMask(mask);
   * ```
   * @param mask - The Bounds to use as a mask
   * @see {@link Bounds#addBounds} For union operation
   * @see {@link Bounds#fit} For fitting to rectangle
   */
  addBoundsMask(mask) {
    this.minX = this.minX > mask.minX ? this.minX : mask.minX;
    this.minY = this.minY > mask.minY ? this.minY : mask.minY;
    this.maxX = this.maxX < mask.maxX ? this.maxX : mask.maxX;
    this.maxY = this.maxY < mask.maxY ? this.maxY : mask.maxY;
  }
  /**
   * Applies a transformation matrix to the bounds, updating its coordinates.
   * Transforms all corners of the bounds using the given matrix.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Apply translation
   * const translateMatrix = new Matrix()
   *     .translate(50, 50);
   * bounds.applyMatrix(translateMatrix);
   * ```
   * @param matrix - The matrix to apply to the bounds
   * @see {@link Matrix} For matrix operations
   * @see {@link Bounds#addFrame} For adding transformed frames
   */
  applyMatrix(matrix) {
    const minX = this.minX;
    const minY = this.minY;
    const maxX = this.maxX;
    const maxY = this.maxY;
    const { a: a2, b: b2, c: c2, d: d2, tx, ty } = matrix;
    let x2 = a2 * minX + c2 * minY + tx;
    let y2 = b2 * minX + d2 * minY + ty;
    this.minX = x2;
    this.minY = y2;
    this.maxX = x2;
    this.maxY = y2;
    x2 = a2 * maxX + c2 * minY + tx;
    y2 = b2 * maxX + d2 * minY + ty;
    this.minX = x2 < this.minX ? x2 : this.minX;
    this.minY = y2 < this.minY ? y2 : this.minY;
    this.maxX = x2 > this.maxX ? x2 : this.maxX;
    this.maxY = y2 > this.maxY ? y2 : this.maxY;
    x2 = a2 * minX + c2 * maxY + tx;
    y2 = b2 * minX + d2 * maxY + ty;
    this.minX = x2 < this.minX ? x2 : this.minX;
    this.minY = y2 < this.minY ? y2 : this.minY;
    this.maxX = x2 > this.maxX ? x2 : this.maxX;
    this.maxY = y2 > this.maxY ? y2 : this.maxY;
    x2 = a2 * maxX + c2 * maxY + tx;
    y2 = b2 * maxX + d2 * maxY + ty;
    this.minX = x2 < this.minX ? x2 : this.minX;
    this.minY = y2 < this.minY ? y2 : this.minY;
    this.maxX = x2 > this.maxX ? x2 : this.maxX;
    this.maxY = y2 > this.maxY ? y2 : this.maxY;
  }
  /**
   * Resizes the bounds object to fit within the given rectangle.
   * Clips the bounds if they extend beyond the rectangle's edges.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 200, 200);
   * // Fit within viewport
   * const viewport = new Rectangle(50, 50, 100, 100);
   * bounds.fit(viewport);
   * // bounds are now (50, 50, 150, 150)
   * ```
   * @param rect - The rectangle to fit within
   * @returns This bounds object for chaining
   * @see {@link Bounds#addBoundsMask} For intersection
   * @see {@link Bounds#pad} For expanding bounds
   */
  fit(rect) {
    if (this.minX < rect.left)
      this.minX = rect.left;
    if (this.maxX > rect.right)
      this.maxX = rect.right;
    if (this.minY < rect.top)
      this.minY = rect.top;
    if (this.maxY > rect.bottom)
      this.maxY = rect.bottom;
    return this;
  }
  /**
   * Resizes the bounds object to include the given bounds.
   * Similar to fit() but works with raw coordinate values instead of a Rectangle.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 200, 200);
   * // Fit to specific coordinates
   * bounds.fitBounds(50, 150, 50, 150);
   * // bounds are now (50, 50, 150, 150)
   * ```
   * @param left - The left value of the bounds
   * @param right - The right value of the bounds
   * @param top - The top value of the bounds
   * @param bottom - The bottom value of the bounds
   * @returns This bounds object for chaining
   * @see {@link Bounds#fit} For fitting to Rectangle
   * @see {@link Bounds#addBoundsMask} For intersection
   */
  fitBounds(left, right, top, bottom) {
    if (this.minX < left)
      this.minX = left;
    if (this.maxX > right)
      this.maxX = right;
    if (this.minY < top)
      this.minY = top;
    if (this.maxY > bottom)
      this.maxY = bottom;
    return this;
  }
  /**
   * Pads bounds object, making it grow in all directions.
   * If paddingY is omitted, both paddingX and paddingY will be set to paddingX.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   *
   * // Add equal padding
   * bounds.pad(10);
   * // bounds are now (-10, -10, 110, 110)
   *
   * // Add different padding for x and y
   * bounds.pad(20, 10);
   * // bounds are now (-30, -20, 130, 120)
   * ```
   * @param paddingX - The horizontal padding amount
   * @param paddingY - The vertical padding amount
   * @returns This bounds object for chaining
   * @see {@link Bounds#fit} For constraining bounds
   * @see {@link Bounds#scale} For uniform scaling
   */
  pad(paddingX, paddingY = paddingX) {
    this.minX -= paddingX;
    this.maxX += paddingX;
    this.minY -= paddingY;
    this.maxY += paddingY;
    return this;
  }
  /**
   * Ceils the bounds by rounding up max values and rounding down min values.
   * Useful for pixel-perfect calculations and avoiding fractional pixels.
   * @example
   * ```ts
   * const bounds = new Bounds();
   * bounds.set(10.2, 10.9, 50.1, 50.8);
   *
   * // Round to whole pixels
   * bounds.ceil();
   * // bounds are now (10, 10, 51, 51)
   * ```
   * @returns This bounds object for chaining
   * @see {@link Bounds#scale} For size adjustments
   * @see {@link Bounds#fit} For constraining bounds
   */
  ceil() {
    this.minX = Math.floor(this.minX);
    this.minY = Math.floor(this.minY);
    this.maxX = Math.ceil(this.maxX);
    this.maxY = Math.ceil(this.maxY);
    return this;
  }
  /**
   * Creates a new Bounds instance with the same values.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   *
   * // Create a copy
   * const copy = bounds.clone();
   *
   * // Original and copy are independent
   * bounds.pad(10);
   * console.log(copy.width === bounds.width); // false
   * ```
   * @returns A new Bounds instance with the same values
   * @see {@link Bounds#copyFrom} For reusing existing bounds
   */
  clone() {
    return new Bounds(this.minX, this.minY, this.maxX, this.maxY);
  }
  /**
   * Scales the bounds by the given values, adjusting all edges proportionally.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   *
   * // Scale uniformly
   * bounds.scale(2);
   * // bounds are now (0, 0, 200, 200)
   *
   * // Scale non-uniformly
   * bounds.scale(0.5, 2);
   * // bounds are now (0, 0, 100, 400)
   * ```
   * @param x - The X value to scale by
   * @param y - The Y value to scale by (defaults to x)
   * @returns This bounds object for chaining
   * @see {@link Bounds#pad} For adding padding
   * @see {@link Bounds#fit} For constraining size
   */
  scale(x2, y2 = x2) {
    this.minX *= x2;
    this.minY *= y2;
    this.maxX *= x2;
    this.maxY *= y2;
    return this;
  }
  /**
   * The x position of the bounds in local space.
   * Setting this value will move the bounds while maintaining its width.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Get x position
   * console.log(bounds.x); // 0
   *
   * // Move bounds horizontally
   * bounds.x = 50;
   * console.log(bounds.minX, bounds.maxX); // 50, 150
   *
   * // Width stays the same
   * console.log(bounds.width); // Still 100
   * ```
   */
  get x() {
    return this.minX;
  }
  set x(value) {
    const width = this.maxX - this.minX;
    this.minX = value;
    this.maxX = value + width;
  }
  /**
   * The y position of the bounds in local space.
   * Setting this value will move the bounds while maintaining its height.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Get y position
   * console.log(bounds.y); // 0
   *
   * // Move bounds vertically
   * bounds.y = 50;
   * console.log(bounds.minY, bounds.maxY); // 50, 150
   *
   * // Height stays the same
   * console.log(bounds.height); // Still 100
   * ```
   */
  get y() {
    return this.minY;
  }
  set y(value) {
    const height = this.maxY - this.minY;
    this.minY = value;
    this.maxY = value + height;
  }
  /**
   * The width value of the bounds.
   * Represents the distance between minX and maxX coordinates.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Get width
   * console.log(bounds.width); // 100
   * // Resize width
   * bounds.width = 200;
   * console.log(bounds.maxX - bounds.minX); // 200
   * ```
   */
  get width() {
    return this.maxX - this.minX;
  }
  set width(value) {
    this.maxX = this.minX + value;
  }
  /**
   * The height value of the bounds.
   * Represents the distance between minY and maxY coordinates.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Get height
   * console.log(bounds.height); // 100
   * // Resize height
   * bounds.height = 150;
   * console.log(bounds.maxY - bounds.minY); // 150
   * ```
   */
  get height() {
    return this.maxY - this.minY;
  }
  set height(value) {
    this.maxY = this.minY + value;
  }
  /**
   * The left edge coordinate of the bounds.
   * Alias for minX.
   * @example
   * ```ts
   * const bounds = new Bounds(50, 0, 150, 100);
   * console.log(bounds.left); // 50
   * console.log(bounds.left === bounds.minX); // true
   * ```
   * @readonly
   */
  get left() {
    return this.minX;
  }
  /**
   * The right edge coordinate of the bounds.
   * Alias for maxX.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * console.log(bounds.right); // 100
   * console.log(bounds.right === bounds.maxX); // true
   * ```
   * @readonly
   */
  get right() {
    return this.maxX;
  }
  /**
   * The top edge coordinate of the bounds.
   * Alias for minY.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 25, 100, 125);
   * console.log(bounds.top); // 25
   * console.log(bounds.top === bounds.minY); // true
   * ```
   * @readonly
   */
  get top() {
    return this.minY;
  }
  /**
   * The bottom edge coordinate of the bounds.
   * Alias for maxY.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 200);
   * console.log(bounds.bottom); // 200
   * console.log(bounds.bottom === bounds.maxY); // true
   * ```
   * @readonly
   */
  get bottom() {
    return this.maxY;
  }
  /**
   * Whether the bounds has positive width and height.
   * Checks if both dimensions are greater than zero.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Check if bounds are positive
   * console.log(bounds.isPositive); // true
   *
   * // Negative bounds
   * bounds.maxX = bounds.minX;
   * console.log(bounds.isPositive); // false, width is 0
   * ```
   * @readonly
   * @see {@link Bounds#isEmpty} For checking empty state
   * @see {@link Bounds#isValid} For checking validity
   */
  get isPositive() {
    return this.maxX - this.minX > 0 && this.maxY - this.minY > 0;
  }
  /**
   * Whether the bounds has valid coordinates.
   * Checks if the bounds has been initialized with real values.
   * @example
   * ```ts
   * const bounds = new Bounds();
   * console.log(bounds.isValid); // false, default state
   *
   * // Set valid bounds
   * bounds.addFrame(0, 0, 100, 100);
   * console.log(bounds.isValid); // true
   * ```
   * @readonly
   * @see {@link Bounds#isEmpty} For checking empty state
   * @see {@link Bounds#isPositive} For checking dimensions
   */
  get isValid() {
    return this.minX + this.minY !== Infinity;
  }
  /**
   * Adds vertices from a Float32Array to the bounds, optionally transformed by a matrix.
   * Used for efficiently updating bounds from raw vertex data.
   * @example
   * ```ts
   * const bounds = new Bounds();
   *
   * // Add vertices from geometry
   * const vertices = new Float32Array([
   *     0, 0,    // Vertex 1
   *     100, 0,  // Vertex 2
   *     100, 100 // Vertex 3
   * ]);
   * bounds.addVertexData(vertices, 0, 6);
   *
   * // Add transformed vertices
   * const matrix = new Matrix()
   *     .translate(50, 50)
   *     .rotate(Math.PI / 4);
   * bounds.addVertexData(vertices, 0, 6, matrix);
   *
   * // Add subset of vertices
   * bounds.addVertexData(vertices, 2, 4); // Only second vertex
   * ```
   * @param vertexData - The array of vertices to add
   * @param beginOffset - Starting index in the vertex array
   * @param endOffset - Ending index in the vertex array (excluded)
   * @param matrix - Optional transformation matrix
   * @see {@link Bounds#addFrame} For adding rectangular frames
   * @see {@link Matrix} For transformation details
   */
  addVertexData(vertexData, beginOffset, endOffset, matrix) {
    let minX = this.minX;
    let minY = this.minY;
    let maxX = this.maxX;
    let maxY = this.maxY;
    matrix || (matrix = this.matrix);
    const a2 = matrix.a;
    const b2 = matrix.b;
    const c2 = matrix.c;
    const d2 = matrix.d;
    const tx = matrix.tx;
    const ty = matrix.ty;
    for (let i2 = beginOffset; i2 < endOffset; i2 += 2) {
      const localX = vertexData[i2];
      const localY = vertexData[i2 + 1];
      const x2 = a2 * localX + c2 * localY + tx;
      const y2 = b2 * localX + d2 * localY + ty;
      minX = x2 < minX ? x2 : minX;
      minY = y2 < minY ? y2 : minY;
      maxX = x2 > maxX ? x2 : maxX;
      maxY = y2 > maxY ? y2 : maxY;
    }
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  }
  /**
   * Checks if a point is contained within the bounds.
   * Returns true if the point's coordinates fall within the bounds' area.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * // Basic point check
   * console.log(bounds.containsPoint(50, 50)); // true
   * console.log(bounds.containsPoint(150, 150)); // false
   *
   * // Check edges
   * console.log(bounds.containsPoint(0, 0));   // true, includes edges
   * console.log(bounds.containsPoint(100, 100)); // true, includes edges
   * ```
   * @param x - x coordinate to check
   * @param y - y coordinate to check
   * @returns True if the point is inside the bounds
   * @see {@link Bounds#isPositive} For valid bounds check
   * @see {@link Bounds#rectangle} For Rectangle representation
   */
  containsPoint(x2, y2) {
    if (this.minX <= x2 && this.minY <= y2 && this.maxX >= x2 && this.maxY >= y2) {
      return true;
    }
    return false;
  }
  /**
   * Returns a string representation of the bounds.
   * Useful for debugging and logging bounds information.
   * @example
   * ```ts
   * const bounds = new Bounds(0, 0, 100, 100);
   * console.log(bounds.toString()); // "[pixi.js:Bounds minX=0 minY=0 maxX=100 maxY=100 width=100 height=100]"
   * ```
   * @returns A string describing the bounds
   * @see {@link Bounds#copyFrom} For copying bounds
   * @see {@link Bounds#clone} For creating a new instance
   */
  toString() {
    return `[pixi.js:Bounds minX=${this.minX} minY=${this.minY} maxX=${this.maxX} maxY=${this.maxY} width=${this.width} height=${this.height}]`;
  }
  /**
   * Copies the bounds from another bounds object.
   * Useful for reusing bounds objects and avoiding allocations.
   * @example
   * ```ts
   * const sourceBounds = new Bounds(0, 0, 100, 100);
   * // Copy bounds
   * const targetBounds = new Bounds();
   * targetBounds.copyFrom(sourceBounds);
   * ```
   * @param bounds - The bounds to copy from
   * @returns This bounds object for chaining
   * @see {@link Bounds#clone} For creating new instances
   */
  copyFrom(bounds) {
    this.minX = bounds.minX;
    this.minY = bounds.minY;
    this.maxX = bounds.maxX;
    this.maxY = bounds.maxY;
    return this;
  }
}
var r = { grad: 0.9, turn: 360, rad: 360 / (2 * Math.PI) }, t = function(r2) {
  return "string" == typeof r2 ? r2.length > 0 : "number" == typeof r2;
}, n = function(r2, t2, n2) {
  return void 0 === t2 && (t2 = 0), void 0 === n2 && (n2 = Math.pow(10, t2)), Math.round(n2 * r2) / n2 + 0;
}, e = function(r2, t2, n2) {
  return void 0 === t2 && (t2 = 0), void 0 === n2 && (n2 = 1), r2 > n2 ? n2 : r2 > t2 ? r2 : t2;
}, u = function(r2) {
  return (r2 = isFinite(r2) ? r2 % 360 : 0) > 0 ? r2 : r2 + 360;
}, a = function(r2) {
  return { r: e(r2.r, 0, 255), g: e(r2.g, 0, 255), b: e(r2.b, 0, 255), a: e(r2.a) };
}, o = function(r2) {
  return { r: n(r2.r), g: n(r2.g), b: n(r2.b), a: n(r2.a, 3) };
}, i = /^#([0-9a-f]{3,8})$/i, s = function(r2) {
  var t2 = r2.toString(16);
  return t2.length < 2 ? "0" + t2 : t2;
}, h = function(r2) {
  var t2 = r2.r, n2 = r2.g, e2 = r2.b, u2 = r2.a, a2 = Math.max(t2, n2, e2), o2 = a2 - Math.min(t2, n2, e2), i2 = o2 ? a2 === t2 ? (n2 - e2) / o2 : a2 === n2 ? 2 + (e2 - t2) / o2 : 4 + (t2 - n2) / o2 : 0;
  return { h: 60 * (i2 < 0 ? i2 + 6 : i2), s: a2 ? o2 / a2 * 100 : 0, v: a2 / 255 * 100, a: u2 };
}, b = function(r2) {
  var t2 = r2.h, n2 = r2.s, e2 = r2.v, u2 = r2.a;
  t2 = t2 / 360 * 6, n2 /= 100, e2 /= 100;
  var a2 = Math.floor(t2), o2 = e2 * (1 - n2), i2 = e2 * (1 - (t2 - a2) * n2), s2 = e2 * (1 - (1 - t2 + a2) * n2), h2 = a2 % 6;
  return { r: 255 * [e2, i2, o2, o2, s2, e2][h2], g: 255 * [s2, e2, e2, i2, o2, o2][h2], b: 255 * [o2, o2, s2, e2, e2, i2][h2], a: u2 };
}, g = function(r2) {
  return { h: u(r2.h), s: e(r2.s, 0, 100), l: e(r2.l, 0, 100), a: e(r2.a) };
}, d = function(r2) {
  return { h: n(r2.h), s: n(r2.s), l: n(r2.l), a: n(r2.a, 3) };
}, f = function(r2) {
  return b((n2 = (t2 = r2).s, { h: t2.h, s: (n2 *= ((e2 = t2.l) < 50 ? e2 : 100 - e2) / 100) > 0 ? 2 * n2 / (e2 + n2) * 100 : 0, v: e2 + n2, a: t2.a }));
  var t2, n2, e2;
}, c = function(r2) {
  return { h: (t2 = h(r2)).h, s: (u2 = (200 - (n2 = t2.s)) * (e2 = t2.v) / 100) > 0 && u2 < 200 ? n2 * e2 / 100 / (u2 <= 100 ? u2 : 200 - u2) * 100 : 0, l: u2 / 2, a: t2.a };
  var t2, n2, e2, u2;
}, l = /^hsla?\(\s*([+-]?\d*\.?\d+)(deg|rad|grad|turn)?\s*,\s*([+-]?\d*\.?\d+)%\s*,\s*([+-]?\d*\.?\d+)%\s*(?:,\s*([+-]?\d*\.?\d+)(%)?\s*)?\)$/i, p = /^hsla?\(\s*([+-]?\d*\.?\d+)(deg|rad|grad|turn)?\s+([+-]?\d*\.?\d+)%\s+([+-]?\d*\.?\d+)%\s*(?:\/\s*([+-]?\d*\.?\d+)(%)?\s*)?\)$/i, v = /^rgba?\(\s*([+-]?\d*\.?\d+)(%)?\s*,\s*([+-]?\d*\.?\d+)(%)?\s*,\s*([+-]?\d*\.?\d+)(%)?\s*(?:,\s*([+-]?\d*\.?\d+)(%)?\s*)?\)$/i, m = /^rgba?\(\s*([+-]?\d*\.?\d+)(%)?\s+([+-]?\d*\.?\d+)(%)?\s+([+-]?\d*\.?\d+)(%)?\s*(?:\/\s*([+-]?\d*\.?\d+)(%)?\s*)?\)$/i, y = { string: [[function(r2) {
  var t2 = i.exec(r2);
  return t2 ? (r2 = t2[1]).length <= 4 ? { r: parseInt(r2[0] + r2[0], 16), g: parseInt(r2[1] + r2[1], 16), b: parseInt(r2[2] + r2[2], 16), a: 4 === r2.length ? n(parseInt(r2[3] + r2[3], 16) / 255, 2) : 1 } : 6 === r2.length || 8 === r2.length ? { r: parseInt(r2.substr(0, 2), 16), g: parseInt(r2.substr(2, 2), 16), b: parseInt(r2.substr(4, 2), 16), a: 8 === r2.length ? n(parseInt(r2.substr(6, 2), 16) / 255, 2) : 1 } : null : null;
}, "hex"], [function(r2) {
  var t2 = v.exec(r2) || m.exec(r2);
  return t2 ? t2[2] !== t2[4] || t2[4] !== t2[6] ? null : a({ r: Number(t2[1]) / (t2[2] ? 100 / 255 : 1), g: Number(t2[3]) / (t2[4] ? 100 / 255 : 1), b: Number(t2[5]) / (t2[6] ? 100 / 255 : 1), a: void 0 === t2[7] ? 1 : Number(t2[7]) / (t2[8] ? 100 : 1) }) : null;
}, "rgb"], [function(t2) {
  var n2 = l.exec(t2) || p.exec(t2);
  if (!n2) return null;
  var e2, u2, a2 = g({ h: (e2 = n2[1], u2 = n2[2], void 0 === u2 && (u2 = "deg"), Number(e2) * (r[u2] || 1)), s: Number(n2[3]), l: Number(n2[4]), a: void 0 === n2[5] ? 1 : Number(n2[5]) / (n2[6] ? 100 : 1) });
  return f(a2);
}, "hsl"]], object: [[function(r2) {
  var n2 = r2.r, e2 = r2.g, u2 = r2.b, o2 = r2.a, i2 = void 0 === o2 ? 1 : o2;
  return t(n2) && t(e2) && t(u2) ? a({ r: Number(n2), g: Number(e2), b: Number(u2), a: Number(i2) }) : null;
}, "rgb"], [function(r2) {
  var n2 = r2.h, e2 = r2.s, u2 = r2.l, a2 = r2.a, o2 = void 0 === a2 ? 1 : a2;
  if (!t(n2) || !t(e2) || !t(u2)) return null;
  var i2 = g({ h: Number(n2), s: Number(e2), l: Number(u2), a: Number(o2) });
  return f(i2);
}, "hsl"], [function(r2) {
  var n2 = r2.h, a2 = r2.s, o2 = r2.v, i2 = r2.a, s2 = void 0 === i2 ? 1 : i2;
  if (!t(n2) || !t(a2) || !t(o2)) return null;
  var h2 = (function(r3) {
    return { h: u(r3.h), s: e(r3.s, 0, 100), v: e(r3.v, 0, 100), a: e(r3.a) };
  })({ h: Number(n2), s: Number(a2), v: Number(o2), a: Number(s2) });
  return b(h2);
}, "hsv"]] }, N = function(r2, t2) {
  for (var n2 = 0; n2 < t2.length; n2++) {
    var e2 = t2[n2][0](r2);
    if (e2) return [e2, t2[n2][1]];
  }
  return [null, void 0];
}, x = function(r2) {
  return "string" == typeof r2 ? N(r2.trim(), y.string) : "object" == typeof r2 && null !== r2 ? N(r2, y.object) : [null, void 0];
}, M = function(r2, t2) {
  var n2 = c(r2);
  return { h: n2.h, s: e(n2.s + 100 * t2, 0, 100), l: n2.l, a: n2.a };
}, H = function(r2) {
  return (299 * r2.r + 587 * r2.g + 114 * r2.b) / 1e3 / 255;
}, $ = function(r2, t2) {
  var n2 = c(r2);
  return { h: n2.h, s: n2.s, l: e(n2.l + 100 * t2, 0, 100), a: n2.a };
}, j = (function() {
  function r2(r3) {
    this.parsed = x(r3)[0], this.rgba = this.parsed || { r: 0, g: 0, b: 0, a: 1 };
  }
  return r2.prototype.isValid = function() {
    return null !== this.parsed;
  }, r2.prototype.brightness = function() {
    return n(H(this.rgba), 2);
  }, r2.prototype.isDark = function() {
    return H(this.rgba) < 0.5;
  }, r2.prototype.isLight = function() {
    return H(this.rgba) >= 0.5;
  }, r2.prototype.toHex = function() {
    return r3 = o(this.rgba), t2 = r3.r, e2 = r3.g, u2 = r3.b, i2 = (a2 = r3.a) < 1 ? s(n(255 * a2)) : "", "#" + s(t2) + s(e2) + s(u2) + i2;
    var r3, t2, e2, u2, a2, i2;
  }, r2.prototype.toRgb = function() {
    return o(this.rgba);
  }, r2.prototype.toRgbString = function() {
    return r3 = o(this.rgba), t2 = r3.r, n2 = r3.g, e2 = r3.b, (u2 = r3.a) < 1 ? "rgba(" + t2 + ", " + n2 + ", " + e2 + ", " + u2 + ")" : "rgb(" + t2 + ", " + n2 + ", " + e2 + ")";
    var r3, t2, n2, e2, u2;
  }, r2.prototype.toHsl = function() {
    return d(c(this.rgba));
  }, r2.prototype.toHslString = function() {
    return r3 = d(c(this.rgba)), t2 = r3.h, n2 = r3.s, e2 = r3.l, (u2 = r3.a) < 1 ? "hsla(" + t2 + ", " + n2 + "%, " + e2 + "%, " + u2 + ")" : "hsl(" + t2 + ", " + n2 + "%, " + e2 + "%)";
    var r3, t2, n2, e2, u2;
  }, r2.prototype.toHsv = function() {
    return r3 = h(this.rgba), { h: n(r3.h), s: n(r3.s), v: n(r3.v), a: n(r3.a, 3) };
    var r3;
  }, r2.prototype.invert = function() {
    return w({ r: 255 - (r3 = this.rgba).r, g: 255 - r3.g, b: 255 - r3.b, a: r3.a });
    var r3;
  }, r2.prototype.saturate = function(r3) {
    return void 0 === r3 && (r3 = 0.1), w(M(this.rgba, r3));
  }, r2.prototype.desaturate = function(r3) {
    return void 0 === r3 && (r3 = 0.1), w(M(this.rgba, -r3));
  }, r2.prototype.grayscale = function() {
    return w(M(this.rgba, -1));
  }, r2.prototype.lighten = function(r3) {
    return void 0 === r3 && (r3 = 0.1), w($(this.rgba, r3));
  }, r2.prototype.darken = function(r3) {
    return void 0 === r3 && (r3 = 0.1), w($(this.rgba, -r3));
  }, r2.prototype.rotate = function(r3) {
    return void 0 === r3 && (r3 = 15), this.hue(this.hue() + r3);
  }, r2.prototype.alpha = function(r3) {
    return "number" == typeof r3 ? w({ r: (t2 = this.rgba).r, g: t2.g, b: t2.b, a: r3 }) : n(this.rgba.a, 3);
    var t2;
  }, r2.prototype.hue = function(r3) {
    var t2 = c(this.rgba);
    return "number" == typeof r3 ? w({ h: r3, s: t2.s, l: t2.l, a: t2.a }) : n(t2.h);
  }, r2.prototype.isEqual = function(r3) {
    return this.toHex() === w(r3).toHex();
  }, r2;
})(), w = function(r2) {
  return r2 instanceof j ? r2 : new j(r2);
}, S = [], k = function(r2) {
  r2.forEach(function(r3) {
    S.indexOf(r3) < 0 && (r3(j, y), S.push(r3));
  });
};
function namesPlugin(e2, f2) {
  var a2 = { white: "#ffffff", bisque: "#ffe4c4", blue: "#0000ff", cadetblue: "#5f9ea0", chartreuse: "#7fff00", chocolate: "#d2691e", coral: "#ff7f50", antiquewhite: "#faebd7", aqua: "#00ffff", azure: "#f0ffff", whitesmoke: "#f5f5f5", papayawhip: "#ffefd5", plum: "#dda0dd", blanchedalmond: "#ffebcd", black: "#000000", gold: "#ffd700", goldenrod: "#daa520", gainsboro: "#dcdcdc", cornsilk: "#fff8dc", cornflowerblue: "#6495ed", burlywood: "#deb887", aquamarine: "#7fffd4", beige: "#f5f5dc", crimson: "#dc143c", cyan: "#00ffff", darkblue: "#00008b", darkcyan: "#008b8b", darkgoldenrod: "#b8860b", darkkhaki: "#bdb76b", darkgray: "#a9a9a9", darkgreen: "#006400", darkgrey: "#a9a9a9", peachpuff: "#ffdab9", darkmagenta: "#8b008b", darkred: "#8b0000", darkorchid: "#9932cc", darkorange: "#ff8c00", darkslateblue: "#483d8b", gray: "#808080", darkslategray: "#2f4f4f", darkslategrey: "#2f4f4f", deeppink: "#ff1493", deepskyblue: "#00bfff", wheat: "#f5deb3", firebrick: "#b22222", floralwhite: "#fffaf0", ghostwhite: "#f8f8ff", darkviolet: "#9400d3", magenta: "#ff00ff", green: "#008000", dodgerblue: "#1e90ff", grey: "#808080", honeydew: "#f0fff0", hotpink: "#ff69b4", blueviolet: "#8a2be2", forestgreen: "#228b22", lawngreen: "#7cfc00", indianred: "#cd5c5c", indigo: "#4b0082", fuchsia: "#ff00ff", brown: "#a52a2a", maroon: "#800000", mediumblue: "#0000cd", lightcoral: "#f08080", darkturquoise: "#00ced1", lightcyan: "#e0ffff", ivory: "#fffff0", lightyellow: "#ffffe0", lightsalmon: "#ffa07a", lightseagreen: "#20b2aa", linen: "#faf0e6", mediumaquamarine: "#66cdaa", lemonchiffon: "#fffacd", lime: "#00ff00", khaki: "#f0e68c", mediumseagreen: "#3cb371", limegreen: "#32cd32", mediumspringgreen: "#00fa9a", lightskyblue: "#87cefa", lightblue: "#add8e6", midnightblue: "#191970", lightpink: "#ffb6c1", mistyrose: "#ffe4e1", moccasin: "#ffe4b5", mintcream: "#f5fffa", lightslategray: "#778899", lightslategrey: "#778899", navajowhite: "#ffdead", navy: "#000080", mediumvioletred: "#c71585", powderblue: "#b0e0e6", palegoldenrod: "#eee8aa", oldlace: "#fdf5e6", paleturquoise: "#afeeee", mediumturquoise: "#48d1cc", mediumorchid: "#ba55d3", rebeccapurple: "#663399", lightsteelblue: "#b0c4de", mediumslateblue: "#7b68ee", thistle: "#d8bfd8", tan: "#d2b48c", orchid: "#da70d6", mediumpurple: "#9370db", purple: "#800080", pink: "#ffc0cb", skyblue: "#87ceeb", springgreen: "#00ff7f", palegreen: "#98fb98", red: "#ff0000", yellow: "#ffff00", slateblue: "#6a5acd", lavenderblush: "#fff0f5", peru: "#cd853f", palevioletred: "#db7093", violet: "#ee82ee", teal: "#008080", slategray: "#708090", slategrey: "#708090", aliceblue: "#f0f8ff", darkseagreen: "#8fbc8f", darkolivegreen: "#556b2f", greenyellow: "#adff2f", seagreen: "#2e8b57", seashell: "#fff5ee", tomato: "#ff6347", silver: "#c0c0c0", sienna: "#a0522d", lavender: "#e6e6fa", lightgreen: "#90ee90", orange: "#ffa500", orangered: "#ff4500", steelblue: "#4682b4", royalblue: "#4169e1", turquoise: "#40e0d0", yellowgreen: "#9acd32", salmon: "#fa8072", saddlebrown: "#8b4513", sandybrown: "#f4a460", rosybrown: "#bc8f8f", darksalmon: "#e9967a", lightgoldenrodyellow: "#fafad2", snow: "#fffafa", lightgrey: "#d3d3d3", lightgray: "#d3d3d3", dimgray: "#696969", dimgrey: "#696969", olivedrab: "#6b8e23", olive: "#808000" }, r2 = {};
  for (var d2 in a2) r2[a2[d2]] = d2;
  var l2 = {};
  e2.prototype.toName = function(f3) {
    if (!(this.rgba.a || this.rgba.r || this.rgba.g || this.rgba.b)) return "transparent";
    var d3, i2, n2 = r2[this.toHex()];
    if (n2) return n2;
    if (null == f3 ? void 0 : f3.closest) {
      var o2 = this.toRgb(), t2 = 1 / 0, b2 = "black";
      if (!l2.length) for (var c2 in a2) l2[c2] = new e2(a2[c2]).toRgb();
      for (var g2 in a2) {
        var u2 = (d3 = o2, i2 = l2[g2], Math.pow(d3.r - i2.r, 2) + Math.pow(d3.g - i2.g, 2) + Math.pow(d3.b - i2.b, 2));
        u2 < t2 && (t2 = u2, b2 = g2);
      }
      return b2;
    }
  };
  f2.string.push([function(f3) {
    var r3 = f3.toLowerCase(), d3 = "transparent" === r3 ? "#0000" : a2[r3];
    return d3 ? new e2(d3).toRgb() : null;
  }, "name"]);
}
k([namesPlugin]);
const _Color = class _Color2 {
  /**
   * @param {ColorSource} value - Optional value to use, if not provided, white is used.
   */
  constructor(value = 16777215) {
    this._value = null;
    this._components = new Float32Array(4);
    this._components.fill(1);
    this._int = 16777215;
    this.value = value;
  }
  /**
   * Get the red component of the color, normalized between 0 and 1.
   * @example
   * ```ts
   * const color = new Color('red');
   * console.log(color.red); // 1
   *
   * const green = new Color('#00ff00');
   * console.log(green.red); // 0
   * ```
   */
  get red() {
    return this._components[0];
  }
  /**
   * Get the green component of the color, normalized between 0 and 1.
   * @example
   * ```ts
   * const color = new Color('lime');
   * console.log(color.green); // 1
   *
   * const red = new Color('#ff0000');
   * console.log(red.green); // 0
   * ```
   */
  get green() {
    return this._components[1];
  }
  /**
   * Get the blue component of the color, normalized between 0 and 1.
   * @example
   * ```ts
   * const color = new Color('blue');
   * console.log(color.blue); // 1
   *
   * const yellow = new Color('#ffff00');
   * console.log(yellow.blue); // 0
   * ```
   */
  get blue() {
    return this._components[2];
  }
  /**
   * Get the alpha component of the color, normalized between 0 and 1.
   * @example
   * ```ts
   * const color = new Color('red');
   * console.log(color.alpha); // 1 (fully opaque)
   *
   * const transparent = new Color('rgba(255, 0, 0, 0.5)');
   * console.log(transparent.alpha); // 0.5 (semi-transparent)
   * ```
   */
  get alpha() {
    return this._components[3];
  }
  /**
   * Sets the color value and returns the instance for chaining.
   *
   * This is a chainable version of setting the `value` property.
   * @param value - The color to set. Accepts various formats:
   * - Hex strings/numbers (e.g., '#ff0000', 0xff0000)
   * - RGB/RGBA values (arrays, objects)
   * - CSS color names
   * - HSL/HSLA values
   * - HSV/HSVA values
   * @returns The Color instance for chaining
   * @example
   * ```ts
   * // Basic usage
   * const color = new Color();
   * color.setValue('#ff0000')
   *     .setAlpha(0.5)
   *     .premultiply(0.8);
   *
   * // Different formats
   * color.setValue(0xff0000);          // Hex number
   * color.setValue('#ff0000');         // Hex string
   * color.setValue([1, 0, 0]);         // RGB array
   * color.setValue([1, 0, 0, 0.5]);    // RGBA array
   * color.setValue({ r: 1, g: 0, b: 0 }); // RGB object
   *
   * // Copy from another color
   * const red = new Color('red');
   * color.setValue(red);
   * ```
   * @throws {Error} If the color value is invalid or null
   * @see {@link Color.value} For the underlying value property
   */
  setValue(value) {
    this.value = value;
    return this;
  }
  /**
   * The current color source. This property allows getting and setting the color value
   * while preserving the original format where possible.
   * @remarks
   * When setting:
   * - Setting to a `Color` instance copies its source and components
   * - Setting to other valid sources normalizes and stores the value
   * - Setting to `null` throws an Error
   * - The color remains unchanged if normalization fails
   *
   * When getting:
   * - Returns `null` if color was modified by {@link Color.multiply} or {@link Color.premultiply}
   * - Otherwise returns the original color source
   * @example
   * ```ts
   * // Setting different color formats
   * const color = new Color();
   *
   * color.value = 0xff0000;         // Hex number
   * color.value = '#ff0000';        // Hex string
   * color.value = [1, 0, 0];        // RGB array
   * color.value = [1, 0, 0, 0.5];   // RGBA array
   * color.value = { r: 1, g: 0, b: 0 }; // RGB object
   *
   * // Copying from another color
   * const red = new Color('red');
   * color.value = red;  // Copies red's components
   *
   * // Getting the value
   * console.log(color.value);  // Returns original format
   *
   * // After modifications
   * color.multiply([0.5, 0.5, 0.5]);
   * console.log(color.value);  // Returns null
   * ```
   * @throws {Error} When attempting to set `null`
   */
  set value(value) {
    if (value instanceof _Color2) {
      this._value = this._cloneSource(value._value);
      this._int = value._int;
      this._components.set(value._components);
    } else if (value === null) {
      throw new Error("Cannot set Color#value to null");
    } else if (this._value === null || !this._isSourceEqual(this._value, value)) {
      this._value = this._cloneSource(value);
      this._normalize(this._value);
    }
  }
  get value() {
    return this._value;
  }
  /**
   * Copy a color source internally.
   * @param value - Color source
   */
  _cloneSource(value) {
    if (typeof value === "string" || typeof value === "number" || value instanceof Number || value === null) {
      return value;
    } else if (Array.isArray(value) || ArrayBuffer.isView(value)) {
      return value.slice(0);
    } else if (typeof value === "object" && value !== null) {
      return { ...value };
    }
    return value;
  }
  /**
   * Equality check for color sources.
   * @param value1 - First color source
   * @param value2 - Second color source
   * @returns `true` if the color sources are equal, `false` otherwise.
   */
  _isSourceEqual(value1, value2) {
    const type1 = typeof value1;
    const type2 = typeof value2;
    if (type1 !== type2) {
      return false;
    } else if (type1 === "number" || type1 === "string" || value1 instanceof Number) {
      return value1 === value2;
    } else if (Array.isArray(value1) && Array.isArray(value2) || ArrayBuffer.isView(value1) && ArrayBuffer.isView(value2)) {
      if (value1.length !== value2.length) {
        return false;
      }
      return value1.every((v2, i2) => v2 === value2[i2]);
    } else if (value1 !== null && value2 !== null) {
      const keys1 = Object.keys(value1);
      const keys2 = Object.keys(value2);
      if (keys1.length !== keys2.length) {
        return false;
      }
      return keys1.every((key) => value1[key] === value2[key]);
    }
    return value1 === value2;
  }
  /**
   * Convert to a RGBA color object with normalized components (0-1).
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Convert colors to RGBA objects
   * new Color('white').toRgba();     // returns { r: 1, g: 1, b: 1, a: 1 }
   * new Color('#ff0000').toRgba();   // returns { r: 1, g: 0, b: 0, a: 1 }
   *
   * // With transparency
   * new Color('rgba(255,0,0,0.5)').toRgba(); // returns { r: 1, g: 0, b: 0, a: 0.5 }
   * ```
   * @returns An RGBA object with normalized components
   */
  toRgba() {
    const [r2, g2, b2, a2] = this._components;
    return { r: r2, g: g2, b: b2, a: a2 };
  }
  /**
   * Convert to a RGB color object with normalized components (0-1).
   *
   * Alpha component is omitted in the output.
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Convert colors to RGB objects
   * new Color('white').toRgb();     // returns { r: 1, g: 1, b: 1 }
   * new Color('#ff0000').toRgb();   // returns { r: 1, g: 0, b: 0 }
   *
   * // Alpha is ignored
   * new Color('rgba(255,0,0,0.5)').toRgb(); // returns { r: 1, g: 0, b: 0 }
   * ```
   * @returns An RGB object with normalized components
   */
  toRgb() {
    const [r2, g2, b2] = this._components;
    return { r: r2, g: g2, b: b2 };
  }
  /**
   * Convert to a CSS-style rgba string representation.
   *
   * RGB components are scaled to 0-255 range, alpha remains 0-1.
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Convert colors to RGBA strings
   * new Color('white').toRgbaString();     // returns "rgba(255,255,255,1)"
   * new Color('#ff0000').toRgbaString();   // returns "rgba(255,0,0,1)"
   *
   * // With transparency
   * new Color([1, 0, 0, 0.5]).toRgbaString(); // returns "rgba(255,0,0,0.5)"
   * ```
   * @returns A CSS-compatible rgba string
   */
  toRgbaString() {
    const [r2, g2, b2] = this.toUint8RgbArray();
    return `rgba(${r2},${g2},${b2},${this.alpha})`;
  }
  /**
   * Convert to an [R, G, B] array of clamped uint8 values (0 to 255).
   * @param {number[]|Uint8Array|Uint8ClampedArray} [out] - Optional output array. If not provided,
   * a cached array will be used and returned.
   * @returns Array containing RGB components as integers between 0-255
   * @example
   * ```ts
   * // Basic usage
   * new Color('white').toUint8RgbArray(); // returns [255, 255, 255]
   * new Color('#ff0000').toUint8RgbArray(); // returns [255, 0, 0]
   *
   * // Using custom output array
   * const rgb = new Uint8Array(3);
   * new Color('blue').toUint8RgbArray(rgb); // rgb is now [0, 0, 255]
   *
   * // Using different array types
   * new Color('red').toUint8RgbArray(new Uint8ClampedArray(3)); // [255, 0, 0]
   * new Color('red').toUint8RgbArray([]); // [255, 0, 0]
   * ```
   * @remarks
   * - Output values are always clamped between 0-255
   * - Alpha component is not included in output
   * - Reuses internal cache array if no output array provided
   */
  toUint8RgbArray(out2) {
    const [r2, g2, b2] = this._components;
    if (!this._arrayRgb) {
      this._arrayRgb = [];
    }
    out2 || (out2 = this._arrayRgb);
    out2[0] = Math.round(r2 * 255);
    out2[1] = Math.round(g2 * 255);
    out2[2] = Math.round(b2 * 255);
    return out2;
  }
  /**
   * Convert to an [R, G, B, A] array of normalized floats (numbers from 0.0 to 1.0).
   * @param {number[]|Float32Array} [out] - Optional output array. If not provided,
   * a cached array will be used and returned.
   * @returns Array containing RGBA components as floats between 0-1
   * @example
   * ```ts
   * // Basic usage
   * new Color('white').toArray();  // returns [1, 1, 1, 1]
   * new Color('red').toArray();    // returns [1, 0, 0, 1]
   *
   * // With alpha
   * new Color('rgba(255,0,0,0.5)').toArray(); // returns [1, 0, 0, 0.5]
   *
   * // Using custom output array
   * const rgba = new Float32Array(4);
   * new Color('blue').toArray(rgba); // rgba is now [0, 0, 1, 1]
   * ```
   * @remarks
   * - Output values are normalized between 0-1
   * - Includes alpha component as the fourth value
   * - Reuses internal cache array if no output array provided
   */
  toArray(out2) {
    if (!this._arrayRgba) {
      this._arrayRgba = [];
    }
    out2 || (out2 = this._arrayRgba);
    const [r2, g2, b2, a2] = this._components;
    out2[0] = r2;
    out2[1] = g2;
    out2[2] = b2;
    out2[3] = a2;
    return out2;
  }
  /**
   * Convert to an [R, G, B] array of normalized floats (numbers from 0.0 to 1.0).
   * @param {number[]|Float32Array} [out] - Optional output array. If not provided,
   * a cached array will be used and returned.
   * @returns Array containing RGB components as floats between 0-1
   * @example
   * ```ts
   * // Basic usage
   * new Color('white').toRgbArray(); // returns [1, 1, 1]
   * new Color('red').toRgbArray();   // returns [1, 0, 0]
   *
   * // Using custom output array
   * const rgb = new Float32Array(3);
   * new Color('blue').toRgbArray(rgb); // rgb is now [0, 0, 1]
   * ```
   * @remarks
   * - Output values are normalized between 0-1
   * - Alpha component is omitted from output
   * - Reuses internal cache array if no output array provided
   */
  toRgbArray(out2) {
    if (!this._arrayRgb) {
      this._arrayRgb = [];
    }
    out2 || (out2 = this._arrayRgb);
    const [r2, g2, b2] = this._components;
    out2[0] = r2;
    out2[1] = g2;
    out2[2] = b2;
    return out2;
  }
  /**
   * Convert to a hexadecimal number.
   * @returns The color as a 24-bit RGB integer
   * @example
   * ```ts
   * // Basic usage
   * new Color('white').toNumber(); // returns 0xffffff
   * new Color('red').toNumber();   // returns 0xff0000
   *
   * // Store as hex
   * const color = new Color('blue');
   * const hex = color.toNumber(); // 0x0000ff
   * ```
   */
  toNumber() {
    return this._int;
  }
  /**
   * Convert to a BGR number.
   *
   * Useful for platforms that expect colors in BGR format.
   * @returns The color as a 24-bit BGR integer
   * @example
   * ```ts
   * // Convert RGB to BGR
   * new Color(0xffcc99).toBgrNumber(); // returns 0x99ccff
   *
   * // Common use case: platform-specific color format
   * const color = new Color('orange');
   * const bgrColor = color.toBgrNumber(); // Color with swapped R/B channels
   * ```
   * @remarks
   * This swaps the red and blue channels compared to the normal RGB format:
   * - RGB 0xRRGGBB becomes BGR 0xBBGGRR
   */
  toBgrNumber() {
    const [r2, g2, b2] = this.toUint8RgbArray();
    return (b2 << 16) + (g2 << 8) + r2;
  }
  /**
   * Convert to a hexadecimal number in little endian format (e.g., BBGGRR).
   *
   * Useful for platforms that expect colors in little endian byte order.
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Convert RGB color to little endian format
   * new Color(0xffcc99).toLittleEndianNumber(); // returns 0x99ccff
   *
   * // Common use cases:
   * const color = new Color('orange');
   * const leColor = color.toLittleEndianNumber(); // Swaps byte order for LE systems
   *
   * // Multiple conversions
   * const colors = {
   *     normal: 0xffcc99,
   *     littleEndian: new Color(0xffcc99).toLittleEndianNumber(), // 0x99ccff
   *     backToNormal: new Color(0x99ccff).toLittleEndianNumber()  // 0xffcc99
   * };
   * ```
   * @remarks
   * - Swaps R and B channels in the color value
   * - RGB 0xRRGGBB becomes 0xBBGGRR
   * - Useful for systems that use little endian byte order
   * - Can be used to convert back and forth between formats
   * @returns The color as a number in little endian format (BBGGRR)
   * @see {@link Color.toBgrNumber} For BGR format without byte swapping
   */
  toLittleEndianNumber() {
    const value = this._int;
    return (value >> 16) + (value & 65280) + ((value & 255) << 16);
  }
  /**
   * Multiply with another color.
   *
   * This action is destructive and modifies the original color.
   * @param {ColorSource} value - The color to multiply by. Accepts any valid color format:
   * - Hex strings/numbers (e.g., '#ff0000', 0xff0000)
   * - RGB/RGBA arrays ([1, 0, 0], [1, 0, 0, 1])
   * - Color objects ({ r: 1, g: 0, b: 0 })
   * - CSS color names ('red', 'blue')
   * @returns this - The Color instance for chaining
   * @example
   * ```ts
   * // Basic multiplication
   * const color = new Color('#ff0000');
   * color.multiply(0x808080); // 50% darker red
   *
   * // With transparency
   * color.multiply([1, 1, 1, 0.5]); // 50% transparent
   *
   * // Chain operations
   * color
   *     .multiply('#808080')
   *     .multiply({ r: 1, g: 1, b: 1, a: 0.5 });
   * ```
   * @remarks
   * - Multiplies each RGB component and alpha separately
   * - Values are clamped between 0-1
   * - Original color format is lost (value becomes null)
   * - Operation cannot be undone
   */
  multiply(value) {
    const [r2, g2, b2, a2] = _Color2._temp.setValue(value)._components;
    this._components[0] *= r2;
    this._components[1] *= g2;
    this._components[2] *= b2;
    this._components[3] *= a2;
    this._refreshInt();
    this._value = null;
    return this;
  }
  /**
   * Converts color to a premultiplied alpha format.
   *
   * This action is destructive and modifies the original color.
   * @param alpha - The alpha value to multiply by (0-1)
   * @param {boolean} [applyToRGB=true] - Whether to premultiply RGB channels
   * @returns {Color} The Color instance for chaining
   * @example
   * ```ts
   * // Basic premultiplication
   * const color = new Color('red');
   * color.premultiply(0.5); // 50% transparent red with premultiplied RGB
   *
   * // Alpha only (RGB unchanged)
   * color.premultiply(0.5, false); // 50% transparent, original RGB
   *
   * // Chain with other operations
   * color
   *     .multiply(0x808080)
   *     .premultiply(0.5)
   *     .toNumber();
   * ```
   * @remarks
   * - RGB channels are multiplied by alpha when applyToRGB is true
   * - Alpha is always set to the provided value
   * - Values are clamped between 0-1
   * - Original color format is lost (value becomes null)
   * - Operation cannot be undone
   */
  premultiply(alpha, applyToRGB = true) {
    if (applyToRGB) {
      this._components[0] *= alpha;
      this._components[1] *= alpha;
      this._components[2] *= alpha;
    }
    this._components[3] = alpha;
    this._refreshInt();
    this._value = null;
    return this;
  }
  /**
   * Returns the color as a 32-bit premultiplied alpha integer.
   *
   * Format: 0xAARRGGBB
   * @param {number} alpha - The alpha value to multiply by (0-1)
   * @param {boolean} [applyToRGB=true] - Whether to premultiply RGB channels
   * @returns {number} The premultiplied color as a 32-bit integer
   * @example
   * ```ts
   * // Convert to premultiplied format
   * const color = new Color('red');
   *
   * // Full opacity (0xFFRRGGBB)
   * color.toPremultiplied(1.0); // 0xFFFF0000
   *
   * // 50% transparency with premultiplied RGB
   * color.toPremultiplied(0.5); // 0x7F7F0000
   *
   * // 50% transparency without RGB premultiplication
   * color.toPremultiplied(0.5, false); // 0x7FFF0000
   * ```
   * @remarks
   * - Returns full opacity (0xFF000000) when alpha is 1.0
   * - Returns 0 when alpha is 0.0 and applyToRGB is true
   * - RGB values are rounded during premultiplication
   */
  toPremultiplied(alpha, applyToRGB = true) {
    if (alpha === 1) {
      return (255 << 24) + this._int;
    }
    if (alpha === 0) {
      return applyToRGB ? 0 : this._int;
    }
    let r2 = this._int >> 16 & 255;
    let g2 = this._int >> 8 & 255;
    let b2 = this._int & 255;
    if (applyToRGB) {
      r2 = r2 * alpha + 0.5 | 0;
      g2 = g2 * alpha + 0.5 | 0;
      b2 = b2 * alpha + 0.5 | 0;
    }
    return (alpha * 255 << 24) + (r2 << 16) + (g2 << 8) + b2;
  }
  /**
   * Convert to a hexadecimal string (6 characters).
   * @returns A CSS-compatible hex color string (e.g., "#ff0000")
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Basic colors
   * new Color('red').toHex();    // returns "#ff0000"
   * new Color('white').toHex();  // returns "#ffffff"
   * new Color('black').toHex();  // returns "#000000"
   *
   * // From different formats
   * new Color(0xff0000).toHex(); // returns "#ff0000"
   * new Color([1, 0, 0]).toHex(); // returns "#ff0000"
   * new Color({ r: 1, g: 0, b: 0 }).toHex(); // returns "#ff0000"
   * ```
   * @remarks
   * - Always returns a 6-character hex string
   * - Includes leading "#" character
   * - Alpha channel is ignored
   * - Values are rounded to nearest hex value
   */
  toHex() {
    const hexString = this._int.toString(16);
    return `#${"000000".substring(0, 6 - hexString.length) + hexString}`;
  }
  /**
   * Convert to a hexadecimal string with alpha (8 characters).
   * @returns A CSS-compatible hex color string with alpha (e.g., "#ff0000ff")
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // Fully opaque colors
   * new Color('red').toHexa();   // returns "#ff0000ff"
   * new Color('white').toHexa(); // returns "#ffffffff"
   *
   * // With transparency
   * new Color('rgba(255, 0, 0, 0.5)').toHexa(); // returns "#ff00007f"
   * new Color([1, 0, 0, 0]).toHexa(); // returns "#ff000000"
   * ```
   * @remarks
   * - Returns an 8-character hex string
   * - Includes leading "#" character
   * - Alpha is encoded in last two characters
   * - Values are rounded to nearest hex value
   */
  toHexa() {
    const alphaValue = Math.round(this._components[3] * 255);
    const alphaString = alphaValue.toString(16);
    return this.toHex() + "00".substring(0, 2 - alphaString.length) + alphaString;
  }
  /**
   * Set alpha (transparency) value while preserving color components.
   *
   * Provides a chainable interface for setting alpha.
   * @param alpha - Alpha value between 0 (fully transparent) and 1 (fully opaque)
   * @returns The Color instance for chaining
   * @example
   * ```ts
   * // Basic alpha setting
   * const color = new Color('red');
   * color.setAlpha(0.5);  // 50% transparent red
   *
   * // Chain with other operations
   * color
   *     .setValue('#ff0000')
   *     .setAlpha(0.8)    // 80% opaque
   *     .premultiply(0.5); // Further modify alpha
   *
   * // Reset to fully opaque
   * color.setAlpha(1);
   * ```
   * @remarks
   * - Alpha value is clamped between 0-1
   * - Can be chained with other color operations
   */
  setAlpha(alpha) {
    this._components[3] = this._clamp(alpha);
    return this;
  }
  /**
   * Normalize the input value into rgba
   * @param value - Input value
   */
  _normalize(value) {
    let r2;
    let g2;
    let b2;
    let a2;
    if ((typeof value === "number" || value instanceof Number) && value >= 0 && value <= 16777215) {
      const int = value;
      r2 = (int >> 16 & 255) / 255;
      g2 = (int >> 8 & 255) / 255;
      b2 = (int & 255) / 255;
      a2 = 1;
    } else if ((Array.isArray(value) || value instanceof Float32Array) && value.length >= 3 && value.length <= 4) {
      value = this._clamp(value);
      [r2, g2, b2, a2 = 1] = value;
    } else if ((value instanceof Uint8Array || value instanceof Uint8ClampedArray) && value.length >= 3 && value.length <= 4) {
      value = this._clamp(value, 0, 255);
      [r2, g2, b2, a2 = 255] = value;
      r2 /= 255;
      g2 /= 255;
      b2 /= 255;
      a2 /= 255;
    } else if (typeof value === "string" || typeof value === "object") {
      if (typeof value === "string") {
        const match = _Color2.HEX_PATTERN.exec(value);
        if (match) {
          value = `#${match[2]}`;
        }
      }
      const color = w(value);
      if (color.isValid()) {
        ({ r: r2, g: g2, b: b2, a: a2 } = color.rgba);
        r2 /= 255;
        g2 /= 255;
        b2 /= 255;
      }
    }
    if (r2 !== void 0) {
      this._components[0] = r2;
      this._components[1] = g2;
      this._components[2] = b2;
      this._components[3] = a2;
      this._refreshInt();
    } else {
      throw new Error(`Unable to convert color ${value}`);
    }
  }
  /** Refresh the internal color rgb number */
  _refreshInt() {
    this._clamp(this._components);
    const [r2, g2, b2] = this._components;
    this._int = (r2 * 255 << 16) + (g2 * 255 << 8) + (b2 * 255 | 0);
  }
  /**
   * Clamps values to a range. Will override original values
   * @param value - Value(s) to clamp
   * @param min - Minimum value
   * @param max - Maximum value
   */
  _clamp(value, min = 0, max = 1) {
    if (typeof value === "number") {
      return Math.min(Math.max(value, min), max);
    }
    value.forEach((v2, i2) => {
      value[i2] = Math.min(Math.max(v2, min), max);
    });
    return value;
  }
  /**
   * Check if a value can be interpreted as a valid color format.
   * Supports all color formats that can be used with the Color class.
   * @param value - Value to check
   * @returns True if the value can be used as a color
   * @example
   * ```ts
   * import { Color } from 'pixi.js';
   *
   * // CSS colors and hex values
   * Color.isColorLike('red');          // true
   * Color.isColorLike('#ff0000');      // true
   * Color.isColorLike(0xff0000);       // true
   *
   * // Arrays (RGB/RGBA)
   * Color.isColorLike([1, 0, 0]);      // true
   * Color.isColorLike([1, 0, 0, 0.5]); // true
   *
   * // TypedArrays
   * Color.isColorLike(new Float32Array([1, 0, 0]));          // true
   * Color.isColorLike(new Uint8Array([255, 0, 0]));          // true
   * Color.isColorLike(new Uint8ClampedArray([255, 0, 0]));   // true
   *
   * // Object formats
   * Color.isColorLike({ r: 1, g: 0, b: 0 });            // true (RGB)
   * Color.isColorLike({ r: 1, g: 0, b: 0, a: 0.5 });    // true (RGBA)
   * Color.isColorLike({ h: 0, s: 100, l: 50 });         // true (HSL)
   * Color.isColorLike({ h: 0, s: 100, l: 50, a: 0.5 }); // true (HSLA)
   * Color.isColorLike({ h: 0, s: 100, v: 100 });        // true (HSV)
   * Color.isColorLike({ h: 0, s: 100, v: 100, a: 0.5 });// true (HSVA)
   *
   * // Color instances
   * Color.isColorLike(new Color('red')); // true
   *
   * // Invalid values
   * Color.isColorLike(null);           // false
   * Color.isColorLike(undefined);      // false
   * Color.isColorLike({});             // false
   * Color.isColorLike([]);             // false
   * Color.isColorLike('not-a-color');  // false
   * ```
   * @remarks
   * Checks for the following formats:
   * - Numbers (0x000000 to 0xffffff)
   * - CSS color strings
   * - RGB/RGBA arrays and objects
   * - HSL/HSLA objects
   * - HSV/HSVA objects
   * - TypedArrays (Float32Array, Uint8Array, Uint8ClampedArray)
   * - Color instances
   * @see {@link ColorSource} For supported color format types
   * @see {@link Color.setValue} For setting color values
   * @category utility
   */
  static isColorLike(value) {
    return typeof value === "number" || typeof value === "string" || value instanceof Number || value instanceof _Color2 || Array.isArray(value) || value instanceof Uint8Array || value instanceof Uint8ClampedArray || value instanceof Float32Array || value.r !== void 0 && value.g !== void 0 && value.b !== void 0 || value.r !== void 0 && value.g !== void 0 && value.b !== void 0 && value.a !== void 0 || value.h !== void 0 && value.s !== void 0 && value.l !== void 0 || value.h !== void 0 && value.s !== void 0 && value.l !== void 0 && value.a !== void 0 || value.h !== void 0 && value.s !== void 0 && value.v !== void 0 || value.h !== void 0 && value.s !== void 0 && value.v !== void 0 && value.a !== void 0;
  }
};
_Color.shared = new _Color();
_Color._temp = new _Color();
_Color.HEX_PATTERN = /^(#|0x)?(([a-f0-9]{3}){1,2}([a-f0-9]{2})?)$/i;
let Color = _Color;
const cullingMixin = {
  cullArea: null,
  cullable: false,
  cullableChildren: true
};
let warnCount = 0;
const maxWarnings = 500;
function warn(...args) {
  if (warnCount === maxWarnings)
    return;
  warnCount++;
  if (warnCount === maxWarnings) {
    console.warn("PixiJS Warning: too many warnings, no more warnings will be reported to the console by PixiJS.");
  } else {
    console.warn("PixiJS Warning: ", ...args);
  }
}
const GlobalResourceRegistry = {
  /**
   * Set of registered pools and cleanable objects.
   * @private
   */
  _registeredResources: /* @__PURE__ */ new Set(),
  /**
   * Registers a pool or cleanable object for cleanup.
   * @param {Cleanable} pool - The pool or object to register.
   */
  register(pool) {
    this._registeredResources.add(pool);
  },
  /**
   * Unregisters a pool or cleanable object from cleanup.
   * @param {Cleanable} pool - The pool or object to unregister.
   */
  unregister(pool) {
    this._registeredResources.delete(pool);
  },
  /** Clears all registered pools and cleanable objects. This will call clear() on each registered item. */
  release() {
    this._registeredResources.forEach((pool) => pool.clear());
  },
  /**
   * Gets the number of registered pools and cleanable objects.
   * @returns {number} The count of registered items.
   */
  get registeredCount() {
    return this._registeredResources.size;
  },
  /**
   * Checks if a specific pool or cleanable object is registered.
   * @param {Cleanable} pool - The pool or object to check.
   * @returns {boolean} True if the item is registered, false otherwise.
   */
  isRegistered(pool) {
    return this._registeredResources.has(pool);
  },
  /**
   * Removes all registrations without clearing the pools.
   * Useful if you want to reset the collector without affecting the pools.
   */
  reset() {
    this._registeredResources.clear();
  }
};
class Pool {
  /**
   * Constructs a new Pool.
   * @param ClassType - The constructor of the items in the pool.
   * @param {number} [initialSize] - The initial size of the pool.
   */
  constructor(ClassType, initialSize) {
    this._pool = [];
    this._count = 0;
    this._index = 0;
    this._classType = ClassType;
    if (initialSize) {
      this.prepopulate(initialSize);
    }
  }
  /**
   * Prepopulates the pool with a given number of items.
   * @param total - The number of items to add to the pool.
   */
  prepopulate(total) {
    for (let i2 = 0; i2 < total; i2++) {
      this._pool[this._index++] = new this._classType();
    }
    this._count += total;
  }
  /**
   * Gets an item from the pool. Calls the item's `init` method if it exists.
   * If there are no items left in the pool, a new one will be created.
   * @param {unknown} [data] - Optional data to pass to the item's constructor.
   * @returns {T} The item from the pool.
   */
  get(data) {
    let item;
    if (this._index > 0) {
      item = this._pool[--this._index];
    } else {
      item = new this._classType();
      this._count++;
    }
    item.init?.(data);
    return item;
  }
  /**
   * Returns an item to the pool. Calls the item's `reset` method if it exists.
   * @param {T} item - The item to return to the pool.
   */
  return(item) {
    item.reset?.();
    this._pool[this._index++] = item;
  }
  /**
   * Gets the number of items in the pool.
   * @readonly
   */
  get totalSize() {
    return this._count;
  }
  /**
   * Gets the number of items in the pool that are free to use without needing to create more.
   * @readonly
   */
  get totalFree() {
    return this._index;
  }
  /**
   * Gets the number of items in the pool that are currently in use.
   * @readonly
   */
  get totalUsed() {
    return this._count - this._index;
  }
  /** clears the pool */
  clear() {
    if (this._pool.length > 0 && this._pool[0].destroy) {
      for (let i2 = 0; i2 < this._index; i2++) {
        this._pool[i2].destroy();
      }
    }
    this._pool.length = 0;
    this._count = 0;
    this._index = 0;
  }
}
class PoolGroupClass {
  constructor() {
    this._poolsByClass = /* @__PURE__ */ new Map();
  }
  /**
   * Prepopulates a specific pool with a given number of items.
   * @template T The type of items in the pool. Must extend PoolItem.
   * @param {PoolItemConstructor<T>} Class - The constructor of the items in the pool.
   * @param {number} total - The number of items to add to the pool.
   */
  prepopulate(Class, total) {
    const classPool = this.getPool(Class);
    classPool.prepopulate(total);
  }
  /**
   * Gets an item from a specific pool.
   * @template T The type of items in the pool. Must extend PoolItem.
   * @param {PoolItemConstructor<T>} Class - The constructor of the items in the pool.
   * @param {unknown} [data] - Optional data to pass to the item's constructor.
   * @returns {T} The item from the pool.
   */
  get(Class, data) {
    const pool = this.getPool(Class);
    return pool.get(data);
  }
  /**
   * Returns an item to its respective pool.
   * @param {PoolItem} item - The item to return to the pool.
   */
  return(item) {
    const pool = this.getPool(item.constructor);
    pool.return(item);
  }
  /**
   * Gets a specific pool based on the class type.
   * @template T The type of items in the pool. Must extend PoolItem.
   * @param {PoolItemConstructor<T>} ClassType - The constructor of the items in the pool.
   * @returns {Pool<T>} The pool of the given class type.
   */
  getPool(ClassType) {
    if (!this._poolsByClass.has(ClassType)) {
      this._poolsByClass.set(ClassType, new Pool(ClassType));
    }
    return this._poolsByClass.get(ClassType);
  }
  /** gets the usage stats of each pool in the system */
  stats() {
    const stats = {};
    this._poolsByClass.forEach((pool) => {
      const name = stats[pool._classType.name] ? pool._classType.name + pool._classType.ID : pool._classType.name;
      stats[name] = {
        free: pool.totalFree,
        used: pool.totalUsed,
        size: pool.totalSize
      };
    });
    return stats;
  }
  /** Clears all pools in the group. This will reset all pools and free their resources. */
  clear() {
    this._poolsByClass.forEach((pool) => pool.clear());
    this._poolsByClass.clear();
  }
}
const BigPool = new PoolGroupClass();
GlobalResourceRegistry.register(BigPool);
const cacheAsTextureMixin = {
  get isCachedAsTexture() {
    return !!this.renderGroup?.isCachedAsTexture;
  },
  cacheAsTexture(val) {
    if (typeof val === "boolean" && val === false) {
      this.disableRenderGroup();
    } else {
      this.enableRenderGroup();
      this.renderGroup.enableCacheAsTexture(val === true ? {} : val);
    }
  },
  updateCacheTexture() {
    this.renderGroup?.updateCacheTexture();
  },
  get cacheAsBitmap() {
    return this.isCachedAsTexture;
  },
  set cacheAsBitmap(val) {
    deprecation("v8.6.0", "cacheAsBitmap is deprecated, use cacheAsTexture instead.");
    this.cacheAsTexture(val);
  }
};
function removeItems(arr, startIdx, removeCount) {
  const length = arr.length;
  let i2;
  if (startIdx >= length || removeCount === 0) {
    return;
  }
  removeCount = startIdx + removeCount > length ? length - startIdx : removeCount;
  const len = length - removeCount;
  for (i2 = startIdx; i2 < len; ++i2) {
    arr[i2] = arr[i2 + removeCount];
  }
  arr.length = len;
}
const childrenHelperMixin = {
  allowChildren: true,
  removeChildren(beginIndex = 0, endIndex) {
    const end = endIndex ?? this.children.length;
    const range = end - beginIndex;
    const removed = [];
    if (range > 0 && range <= end) {
      for (let i2 = end - 1; i2 >= beginIndex; i2--) {
        const child = this.children[i2];
        if (!child)
          continue;
        removed.push(child);
        child.parent = null;
      }
      removeItems(this.children, beginIndex, end);
      const renderGroup = this.renderGroup || this.parentRenderGroup;
      if (renderGroup) {
        renderGroup.removeChildren(removed);
      }
      for (let i2 = 0; i2 < removed.length; ++i2) {
        const child = removed[i2];
        child.parentRenderLayer?.detach(child);
        this.emit("childRemoved", child, this, i2);
        removed[i2].emit("removed", this);
      }
      if (removed.length > 0) {
        this._didViewChangeTick++;
      }
      return removed;
    } else if (range === 0 && this.children.length === 0) {
      return removed;
    }
    throw new RangeError("removeChildren: numeric values are outside the acceptable range.");
  },
  removeChildAt(index) {
    const child = this.getChildAt(index);
    return this.removeChild(child);
  },
  getChildAt(index) {
    if (index < 0 || index >= this.children.length) {
      throw new Error(`getChildAt: Index (${index}) does not exist.`);
    }
    return this.children[index];
  },
  setChildIndex(child, index) {
    if (index < 0 || index >= this.children.length) {
      throw new Error(`The index ${index} supplied is out of bounds ${this.children.length}`);
    }
    this.getChildIndex(child);
    this.addChildAt(child, index);
  },
  getChildIndex(child) {
    const index = this.children.indexOf(child);
    if (index === -1) {
      throw new Error("The supplied Container must be a child of the caller");
    }
    return index;
  },
  addChildAt(child, index) {
    if (!this.allowChildren) {
      deprecation(v8_0_0, "addChildAt: Only Containers will be allowed to add children in v8.0.0");
    }
    const { children } = this;
    if (index < 0 || index > children.length) {
      throw new Error(`${child}addChildAt: The index ${index} supplied is out of bounds ${children.length}`);
    }
    if (child.parent) {
      const currentIndex = child.parent.children.indexOf(child);
      if (child.parent === this && currentIndex === index) {
        return child;
      }
      if (currentIndex !== -1) {
        child.parent.children.splice(currentIndex, 1);
      }
    }
    if (index === children.length) {
      children.push(child);
    } else {
      children.splice(index, 0, child);
    }
    child.parent = this;
    child.didChange = true;
    child._updateFlags = 15;
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (renderGroup) {
      renderGroup.addChild(child);
    }
    if (this.sortableChildren)
      this.sortDirty = true;
    this.emit("childAdded", child, this, index);
    child.emit("added", this);
    return child;
  },
  swapChildren(child, child2) {
    if (child === child2) {
      return;
    }
    const index1 = this.getChildIndex(child);
    const index2 = this.getChildIndex(child2);
    this.children[index1] = child2;
    this.children[index2] = child;
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (renderGroup) {
      renderGroup.structureDidChange = true;
    }
    this._didContainerChangeTick++;
  },
  removeFromParent() {
    this.parent?.removeChild(this);
  },
  reparentChild(...child) {
    if (child.length === 1) {
      return this.reparentChildAt(child[0], this.children.length);
    }
    child.forEach((c2) => this.reparentChildAt(c2, this.children.length));
    return child[0];
  },
  reparentChildAt(child, index) {
    if (child.parent === this) {
      this.setChildIndex(child, index);
      return child;
    }
    const childMat = child.worldTransform.clone();
    child.removeFromParent();
    this.addChildAt(child, index);
    const newMatrix = this.worldTransform.clone();
    newMatrix.invert();
    childMat.prepend(newMatrix);
    child.setFromMatrix(childMat);
    return child;
  },
  replaceChild(oldChild, newChild) {
    oldChild.updateLocalTransform();
    this.addChildAt(newChild, this.getChildIndex(oldChild));
    newChild.setFromMatrix(oldChild.localTransform);
    newChild.updateLocalTransform();
    this.removeChild(oldChild);
  }
};
const collectRenderablesMixin = {
  collectRenderables(instructionSet, renderer, currentLayer) {
    if (this.parentRenderLayer && this.parentRenderLayer !== currentLayer || this.globalDisplayStatus < 7 || !this.includeInBuild)
      return;
    if (this.sortableChildren) {
      this.sortChildren();
    }
    if (this.isSimple) {
      this.collectRenderablesSimple(instructionSet, renderer, currentLayer);
    } else if (this.renderGroup) {
      renderer.renderPipes.renderGroup.addRenderGroup(this.renderGroup, instructionSet);
    } else {
      this.collectRenderablesWithEffects(instructionSet, renderer, currentLayer);
    }
  },
  collectRenderablesSimple(instructionSet, renderer, currentLayer) {
    const children = this.children;
    const length = children.length;
    for (let i2 = 0; i2 < length; i2++) {
      children[i2].collectRenderables(instructionSet, renderer, currentLayer);
    }
  },
  collectRenderablesWithEffects(instructionSet, renderer, currentLayer) {
    const { renderPipes } = renderer;
    for (let i2 = 0; i2 < this.effects.length; i2++) {
      const effect = this.effects[i2];
      const pipe = renderPipes[effect.pipe];
      pipe.push(effect, this, instructionSet);
    }
    this.collectRenderablesSimple(instructionSet, renderer, currentLayer);
    for (let i2 = this.effects.length - 1; i2 >= 0; i2--) {
      const effect = this.effects[i2];
      const pipe = renderPipes[effect.pipe];
      pipe.pop(effect, this, instructionSet);
    }
  }
};
class FilterEffect {
  constructor() {
    this.pipe = "filter";
    this.priority = 1;
  }
  destroy() {
    for (let i2 = 0; i2 < this.filters.length; i2++) {
      this.filters[i2].destroy();
    }
    this.filters = null;
    this.filterArea = null;
  }
}
class MaskEffectManagerClass {
  constructor() {
    this._effectClasses = [];
    this._tests = [];
    this._initialized = false;
  }
  init() {
    if (this._initialized)
      return;
    this._initialized = true;
    this._effectClasses.forEach((test) => {
      this.add({
        test: test.test,
        maskClass: test
      });
    });
  }
  add(test) {
    this._tests.push(test);
  }
  getMaskEffect(item) {
    if (!this._initialized)
      this.init();
    for (let i2 = 0; i2 < this._tests.length; i2++) {
      const test = this._tests[i2];
      if (test.test(item)) {
        return BigPool.get(test.maskClass, item);
      }
    }
    return item;
  }
  returnMaskEffect(effect) {
    BigPool.return(effect);
  }
}
const MaskEffectManager = new MaskEffectManagerClass();
extensions.handleByList(ExtensionType.MaskEffect, MaskEffectManager._effectClasses);
const effectsMixin = {
  _maskEffect: null,
  _maskOptions: {
    inverse: false
  },
  _filterEffect: null,
  effects: [],
  _markStructureAsChanged() {
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (renderGroup) {
      renderGroup.structureDidChange = true;
    }
  },
  addEffect(effect) {
    const index = this.effects.indexOf(effect);
    if (index !== -1)
      return;
    this.effects.push(effect);
    this.effects.sort((a2, b2) => a2.priority - b2.priority);
    this._markStructureAsChanged();
    this._updateIsSimple();
  },
  removeEffect(effect) {
    const index = this.effects.indexOf(effect);
    if (index === -1)
      return;
    this.effects.splice(index, 1);
    this._markStructureAsChanged();
    this._updateIsSimple();
  },
  set mask(value) {
    const effect = this._maskEffect;
    if (effect?.mask === value)
      return;
    if (effect) {
      this.removeEffect(effect);
      MaskEffectManager.returnMaskEffect(effect);
      this._maskEffect = null;
    }
    if (value === null || value === void 0)
      return;
    this._maskEffect = MaskEffectManager.getMaskEffect(value);
    this.addEffect(this._maskEffect);
  },
  get mask() {
    return this._maskEffect?.mask;
  },
  setMask(options) {
    this._maskOptions = {
      ...this._maskOptions,
      ...options
    };
    if (options.mask) {
      this.mask = options.mask;
    }
    this._markStructureAsChanged();
  },
  set filters(value) {
    if (!Array.isArray(value) && value)
      value = [value];
    const effect = this._filterEffect || (this._filterEffect = new FilterEffect());
    value = value;
    const hasFilters = value?.length > 0;
    const hadFilters = effect.filters?.length > 0;
    const didChange = hasFilters !== hadFilters;
    value = Array.isArray(value) ? value.slice(0) : value;
    effect.filters = Object.freeze(value);
    if (didChange) {
      if (hasFilters) {
        this.addEffect(effect);
      } else {
        this.removeEffect(effect);
        effect.filters = value ?? null;
      }
    }
  },
  get filters() {
    return this._filterEffect?.filters;
  },
  set filterArea(value) {
    this._filterEffect || (this._filterEffect = new FilterEffect());
    this._filterEffect.filterArea = value;
  },
  get filterArea() {
    return this._filterEffect?.filterArea;
  }
};
const findMixin = {
  label: null,
  get name() {
    deprecation(v8_0_0, "Container.name property has been removed, use Container.label instead");
    return this.label;
  },
  set name(value) {
    deprecation(v8_0_0, "Container.name property has been removed, use Container.label instead");
    this.label = value;
  },
  getChildByName(name, deep = false) {
    return this.getChildByLabel(name, deep);
  },
  getChildByLabel(label, deep = false) {
    const children = this.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      const child = children[i2];
      if (child.label === label || label instanceof RegExp && label.test(child.label))
        return child;
    }
    if (deep) {
      for (let i2 = 0; i2 < children.length; i2++) {
        const child = children[i2];
        const found = child.getChildByLabel(label, true);
        if (found) {
          return found;
        }
      }
    }
    return null;
  },
  getChildrenByLabel(label, deep = false, out2 = []) {
    const children = this.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      const child = children[i2];
      if (child.label === label || label instanceof RegExp && label.test(child.label)) {
        out2.push(child);
      }
    }
    if (deep) {
      for (let i2 = 0; i2 < children.length; i2++) {
        children[i2].getChildrenByLabel(label, true, out2);
      }
    }
    return out2;
  }
};
const matrixPool = BigPool.getPool(Matrix);
const boundsPool = BigPool.getPool(Bounds);
const tempMatrix$2 = new Matrix();
const getFastGlobalBoundsMixin = {
  getFastGlobalBounds(factorRenderLayers, bounds) {
    bounds || (bounds = new Bounds());
    bounds.clear();
    this._getGlobalBoundsRecursive(!!factorRenderLayers, bounds, this.parentRenderLayer);
    if (!bounds.isValid) {
      bounds.set(0, 0, 0, 0);
    }
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    bounds.applyMatrix(renderGroup.worldTransform);
    return bounds;
  },
  _getGlobalBoundsRecursive(factorRenderLayers, bounds, currentLayer) {
    let localBounds = bounds;
    if (factorRenderLayers && this.parentRenderLayer && this.parentRenderLayer !== currentLayer)
      return;
    if (this.localDisplayStatus !== 7 || !this.measurable) {
      return;
    }
    const manageEffects = !!this.effects.length;
    if (this.renderGroup || manageEffects) {
      localBounds = boundsPool.get().clear();
    }
    if (this.boundsArea) {
      bounds.addRect(this.boundsArea, this.worldTransform);
    } else {
      if (this.renderPipeId) {
        const viewBounds = this.bounds;
        localBounds.addFrame(
          viewBounds.minX,
          viewBounds.minY,
          viewBounds.maxX,
          viewBounds.maxY,
          this.groupTransform
        );
      }
      const children = this.children;
      for (let i2 = 0; i2 < children.length; i2++) {
        children[i2]._getGlobalBoundsRecursive(factorRenderLayers, localBounds, currentLayer);
      }
    }
    if (manageEffects) {
      let advanced = false;
      const renderGroup = this.renderGroup || this.parentRenderGroup;
      for (let i2 = 0; i2 < this.effects.length; i2++) {
        if (this.effects[i2].addBounds) {
          if (!advanced) {
            advanced = true;
            localBounds.applyMatrix(renderGroup.worldTransform);
          }
          this.effects[i2].addBounds(localBounds, true);
        }
      }
      if (advanced) {
        localBounds.applyMatrix(renderGroup.worldTransform.copyTo(tempMatrix$2).invert());
      }
      bounds.addBounds(localBounds);
      boundsPool.return(localBounds);
    } else if (this.renderGroup) {
      bounds.addBounds(localBounds, this.relativeGroupTransform);
      boundsPool.return(localBounds);
    }
  }
};
function getGlobalBounds(target, skipUpdateTransform, bounds) {
  bounds.clear();
  let parentTransform;
  let pooledMatrix;
  if (target.parent) {
    if (!skipUpdateTransform) {
      pooledMatrix = matrixPool.get().identity();
      parentTransform = updateTransformBackwards(target, pooledMatrix);
    } else {
      parentTransform = target.parent.worldTransform;
    }
  } else {
    parentTransform = Matrix.IDENTITY;
  }
  _getGlobalBounds(target, bounds, parentTransform, skipUpdateTransform);
  if (pooledMatrix) {
    matrixPool.return(pooledMatrix);
  }
  if (!bounds.isValid) {
    bounds.set(0, 0, 0, 0);
  }
  return bounds;
}
function _getGlobalBounds(target, bounds, parentTransform, skipUpdateTransform) {
  if (!target.visible || !target.measurable)
    return;
  let worldTransform;
  if (!skipUpdateTransform) {
    target.updateLocalTransform();
    worldTransform = matrixPool.get();
    worldTransform.appendFrom(target.localTransform, parentTransform);
  } else {
    worldTransform = target.worldTransform;
  }
  const parentBounds = bounds;
  const preserveBounds = !!target.effects.length;
  if (preserveBounds) {
    bounds = boundsPool.get().clear();
  }
  if (target.boundsArea) {
    bounds.addRect(target.boundsArea, worldTransform);
  } else {
    const renderableBounds = target.bounds;
    if (renderableBounds && !renderableBounds.isEmpty()) {
      bounds.matrix = worldTransform;
      bounds.addBounds(renderableBounds);
    }
    for (let i2 = 0; i2 < target.children.length; i2++) {
      _getGlobalBounds(target.children[i2], bounds, worldTransform, skipUpdateTransform);
    }
  }
  if (preserveBounds) {
    for (let i2 = 0; i2 < target.effects.length; i2++) {
      target.effects[i2].addBounds?.(bounds);
    }
    parentBounds.addBounds(bounds, Matrix.IDENTITY);
    boundsPool.return(bounds);
  }
  if (!skipUpdateTransform) {
    matrixPool.return(worldTransform);
  }
}
function updateTransformBackwards(target, parentTransform) {
  const parent = target.parent;
  if (parent) {
    updateTransformBackwards(parent, parentTransform);
    parent.updateLocalTransform();
    parentTransform.append(parent.localTransform);
  }
  return parentTransform;
}
function multiplyHexColors(color1, color2) {
  if (color1 === 16777215 || !color2)
    return color2;
  if (color2 === 16777215 || !color1)
    return color1;
  const r1 = color1 >> 16 & 255;
  const g1 = color1 >> 8 & 255;
  const b1 = color1 & 255;
  const r2 = color2 >> 16 & 255;
  const g2 = color2 >> 8 & 255;
  const b2 = color2 & 255;
  const r3 = r1 * r2 / 255 | 0;
  const g3 = g1 * g2 / 255 | 0;
  const b3 = b1 * b2 / 255 | 0;
  return (r3 << 16) + (g3 << 8) + b3;
}
const WHITE_BGR = 16777215;
function multiplyColors(localBGRColor, parentBGRColor) {
  if (localBGRColor === WHITE_BGR) {
    return parentBGRColor;
  }
  if (parentBGRColor === WHITE_BGR) {
    return localBGRColor;
  }
  return multiplyHexColors(localBGRColor, parentBGRColor);
}
function bgr2rgb(color) {
  return ((color & 255) << 16) + (color & 65280) + (color >> 16 & 255);
}
const getGlobalMixin = {
  getGlobalAlpha(skipUpdate) {
    if (skipUpdate) {
      if (this.renderGroup) {
        return this.renderGroup.worldAlpha;
      }
      if (this.parentRenderGroup) {
        return this.parentRenderGroup.worldAlpha * this.alpha;
      }
      return this.alpha;
    }
    let alpha = this.alpha;
    let current = this.parent;
    while (current) {
      alpha *= current.alpha;
      current = current.parent;
    }
    return alpha;
  },
  getGlobalTransform(matrix = new Matrix(), skipUpdate) {
    if (skipUpdate) {
      return matrix.copyFrom(this.worldTransform);
    }
    this.updateLocalTransform();
    const parentTransform = updateTransformBackwards(this, matrixPool.get().identity());
    matrix.appendFrom(this.localTransform, parentTransform);
    matrixPool.return(parentTransform);
    return matrix;
  },
  getGlobalTint(skipUpdate) {
    if (skipUpdate) {
      if (this.renderGroup) {
        return bgr2rgb(this.renderGroup.worldColor);
      }
      if (this.parentRenderGroup) {
        return bgr2rgb(
          multiplyColors(this.localColor, this.parentRenderGroup.worldColor)
        );
      }
      return this.tint;
    }
    let color = this.localColor;
    let parent = this.parent;
    while (parent) {
      color = multiplyColors(color, parent.localColor);
      parent = parent.parent;
    }
    return bgr2rgb(color);
  }
};
function getLocalBounds(target, bounds, relativeMatrix) {
  bounds.clear();
  relativeMatrix || (relativeMatrix = Matrix.IDENTITY);
  _getLocalBounds(target, bounds, relativeMatrix, target, true);
  if (!bounds.isValid) {
    bounds.set(0, 0, 0, 0);
  }
  return bounds;
}
function _getLocalBounds(target, bounds, parentTransform, rootContainer, isRoot) {
  let relativeTransform;
  if (!isRoot) {
    if (!target.visible || !target.measurable)
      return;
    target.updateLocalTransform();
    const localTransform = target.localTransform;
    relativeTransform = matrixPool.get();
    relativeTransform.appendFrom(localTransform, parentTransform);
  } else {
    relativeTransform = matrixPool.get();
    relativeTransform = parentTransform.copyTo(relativeTransform);
  }
  const parentBounds = bounds;
  const preserveBounds = !!target.effects.length;
  if (preserveBounds) {
    bounds = boundsPool.get().clear();
  }
  if (target.boundsArea) {
    bounds.addRect(target.boundsArea, relativeTransform);
  } else {
    if (target.renderPipeId) {
      bounds.matrix = relativeTransform;
      bounds.addBounds(target.bounds);
    }
    const children = target.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      _getLocalBounds(children[i2], bounds, relativeTransform, rootContainer, false);
    }
  }
  if (preserveBounds) {
    for (let i2 = 0; i2 < target.effects.length; i2++) {
      target.effects[i2].addLocalBounds?.(bounds, rootContainer);
    }
    parentBounds.addBounds(bounds, Matrix.IDENTITY);
    boundsPool.return(bounds);
  }
  matrixPool.return(relativeTransform);
}
function checkChildrenDidChange(container, previousData) {
  const children = container.children;
  for (let i2 = 0; i2 < children.length; i2++) {
    const child = children[i2];
    const uid2 = child.uid;
    const didChange = (child._didViewChangeTick & 65535) << 16 | child._didContainerChangeTick & 65535;
    const index = previousData.index;
    if (previousData.data[index] !== uid2 || previousData.data[index + 1] !== didChange) {
      previousData.data[previousData.index] = uid2;
      previousData.data[previousData.index + 1] = didChange;
      previousData.didChange = true;
    }
    previousData.index = index + 2;
    if (child.children.length) {
      checkChildrenDidChange(child, previousData);
    }
  }
  return previousData.didChange;
}
const tempMatrix$1 = new Matrix();
const measureMixin = {
  _localBoundsCacheId: -1,
  _localBoundsCacheData: null,
  _setWidth(value, localWidth) {
    const sign2 = Math.sign(this.scale.x) || 1;
    if (localWidth !== 0) {
      this.scale.x = value / localWidth * sign2;
    } else {
      this.scale.x = sign2;
    }
  },
  _setHeight(value, localHeight) {
    const sign2 = Math.sign(this.scale.y) || 1;
    if (localHeight !== 0) {
      this.scale.y = value / localHeight * sign2;
    } else {
      this.scale.y = sign2;
    }
  },
  getLocalBounds() {
    if (!this._localBoundsCacheData) {
      this._localBoundsCacheData = {
        data: [],
        index: 1,
        didChange: false,
        localBounds: new Bounds()
      };
    }
    const localBoundsCacheData = this._localBoundsCacheData;
    localBoundsCacheData.index = 1;
    localBoundsCacheData.didChange = false;
    if (localBoundsCacheData.data[0] !== this._didViewChangeTick) {
      localBoundsCacheData.didChange = true;
      localBoundsCacheData.data[0] = this._didViewChangeTick;
    }
    checkChildrenDidChange(this, localBoundsCacheData);
    if (localBoundsCacheData.didChange) {
      getLocalBounds(this, localBoundsCacheData.localBounds, tempMatrix$1);
    }
    return localBoundsCacheData.localBounds;
  },
  getBounds(skipUpdate, bounds) {
    return getGlobalBounds(this, skipUpdate, bounds || new Bounds());
  }
};
const onRenderMixin = {
  _onRender: null,
  set onRender(func) {
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (!func) {
      if (this._onRender) {
        renderGroup?.removeOnRender(this);
      }
      this._onRender = null;
      return;
    }
    if (!this._onRender) {
      renderGroup?.addOnRender(this);
    }
    this._onRender = func;
  },
  get onRender() {
    return this._onRender;
  }
};
const sortMixin = {
  _zIndex: 0,
  sortDirty: false,
  sortableChildren: false,
  get zIndex() {
    return this._zIndex;
  },
  set zIndex(value) {
    if (this._zIndex === value)
      return;
    this._zIndex = value;
    this.depthOfChildModified();
  },
  depthOfChildModified() {
    if (this.parent) {
      this.parent.sortableChildren = true;
      this.parent.sortDirty = true;
    }
    if (this.parentRenderGroup) {
      this.parentRenderGroup.structureDidChange = true;
    }
  },
  sortChildren() {
    if (!this.sortDirty)
      return;
    this.sortDirty = false;
    this.children.sort(sortChildren);
  }
};
function sortChildren(a2, b2) {
  return a2._zIndex - b2._zIndex;
}
const toLocalGlobalMixin = {
  getGlobalPosition(point = new Point(), skipUpdate = false) {
    if (this.parent) {
      this.parent.toGlobal(this._position, point, skipUpdate);
    } else {
      point.x = this._position.x;
      point.y = this._position.y;
    }
    return point;
  },
  toGlobal(position, point, skipUpdate = false) {
    const globalMatrix = this.getGlobalTransform(matrixPool.get(), skipUpdate);
    point = globalMatrix.apply(position, point);
    matrixPool.return(globalMatrix);
    return point;
  },
  toLocal(position, from, point, skipUpdate) {
    if (from) {
      position = from.toGlobal(position, point, skipUpdate);
    }
    const globalMatrix = this.getGlobalTransform(matrixPool.get(), skipUpdate);
    point = globalMatrix.applyInverse(position, point);
    matrixPool.return(globalMatrix);
    return point;
  }
};
class InstructionSet {
  constructor() {
    this.uid = uid("instructionSet");
    this.instructions = [];
    this.instructionSize = 0;
    this.renderables = [];
    this.gcTick = 0;
  }
  /** reset the instruction set so it can be reused set size back to 0 */
  reset() {
    this.instructionSize = 0;
  }
  /**
   * Destroy the instruction set, clearing the instructions and renderables.
   * @internal
   */
  destroy() {
    this.instructions.length = 0;
    this.renderables.length = 0;
    this.renderPipes = null;
    this.gcTick = 0;
  }
  /**
   * Add an instruction to the set
   * @param instruction - add an instruction to the set
   */
  add(instruction) {
    this.instructions[this.instructionSize++] = instruction;
  }
  /**
   * Log the instructions to the console (for debugging)
   * @internal
   */
  log() {
    this.instructions.length = this.instructionSize;
    console.table(this.instructions, ["type", "action"]);
  }
}
let count = 0;
class TexturePoolClass {
  /**
   * @param textureOptions - options that will be passed to BaseRenderTexture constructor
   * @param {SCALE_MODE} [textureOptions.scaleMode] - See {@link SCALE_MODE} for possible values.
   */
  constructor(textureOptions) {
    this._poolKeyHash = /* @__PURE__ */ Object.create(null);
    this._texturePool = {};
    this.textureOptions = textureOptions || {};
    this.enableFullScreen = false;
    this.textureStyle = new TextureStyle(this.textureOptions);
  }
  /**
   * Creates texture with params that were specified in pool constructor.
   * @param pixelWidth - Width of texture in pixels.
   * @param pixelHeight - Height of texture in pixels.
   * @param antialias
   */
  createTexture(pixelWidth, pixelHeight, antialias) {
    const textureSource = new TextureSource({
      ...this.textureOptions,
      width: pixelWidth,
      height: pixelHeight,
      resolution: 1,
      antialias,
      autoGarbageCollect: false
    });
    return new Texture({
      source: textureSource,
      label: `texturePool_${count++}`
    });
  }
  /**
   * Gets a Power-of-Two render texture or fullScreen texture
   * @param frameWidth - The minimum width of the render texture.
   * @param frameHeight - The minimum height of the render texture.
   * @param resolution - The resolution of the render texture.
   * @param antialias
   * @returns The new render texture.
   */
  getOptimalTexture(frameWidth, frameHeight, resolution = 1, antialias) {
    let po2Width = Math.ceil(frameWidth * resolution - 1e-6);
    let po2Height = Math.ceil(frameHeight * resolution - 1e-6);
    po2Width = nextPow2(po2Width);
    po2Height = nextPow2(po2Height);
    const key = (po2Width << 17) + (po2Height << 1) + (antialias ? 1 : 0);
    if (!this._texturePool[key]) {
      this._texturePool[key] = [];
    }
    let texture = this._texturePool[key].pop();
    if (!texture) {
      texture = this.createTexture(po2Width, po2Height, antialias);
    }
    texture.source._resolution = resolution;
    texture.source.width = po2Width / resolution;
    texture.source.height = po2Height / resolution;
    texture.source.pixelWidth = po2Width;
    texture.source.pixelHeight = po2Height;
    texture.frame.x = 0;
    texture.frame.y = 0;
    texture.frame.width = frameWidth;
    texture.frame.height = frameHeight;
    texture.updateUvs();
    this._poolKeyHash[texture.uid] = key;
    return texture;
  }
  /**
   * Gets extra texture of the same size as input renderTexture
   * @param texture - The texture to check what size it is.
   * @param antialias - Whether to use antialias.
   * @returns A texture that is a power of two
   */
  getSameSizeTexture(texture, antialias = false) {
    const source2 = texture.source;
    return this.getOptimalTexture(texture.width, texture.height, source2._resolution, antialias);
  }
  /**
   * Place a render texture back into the pool. Optionally reset the style of the texture to the default texture style.
   * useful if you modified the style of the texture after getting it from the pool.
   * @param renderTexture - The renderTexture to free
   * @param resetStyle - Whether to reset the style of the texture to the default texture style
   */
  returnTexture(renderTexture, resetStyle = false) {
    const key = this._poolKeyHash[renderTexture.uid];
    if (resetStyle) {
      renderTexture.source.style = this.textureStyle;
    }
    this._texturePool[key].push(renderTexture);
  }
  /**
   * Clears the pool.
   * @param destroyTextures - Destroy all stored textures.
   */
  clear(destroyTextures) {
    destroyTextures = destroyTextures !== false;
    if (destroyTextures) {
      for (const i2 in this._texturePool) {
        const textures = this._texturePool[i2];
        if (textures) {
          for (let j2 = 0; j2 < textures.length; j2++) {
            textures[j2].destroy(true);
          }
        }
      }
    }
    this._texturePool = {};
  }
}
const TexturePool = new TexturePoolClass();
GlobalResourceRegistry.register(TexturePool);
class RenderGroup {
  constructor() {
    this.renderPipeId = "renderGroup";
    this.root = null;
    this.canBundle = false;
    this.renderGroupParent = null;
    this.renderGroupChildren = [];
    this.worldTransform = new Matrix();
    this.worldColorAlpha = 4294967295;
    this.worldColor = 16777215;
    this.worldAlpha = 1;
    this.childrenToUpdate = /* @__PURE__ */ Object.create(null);
    this.updateTick = 0;
    this.gcTick = 0;
    this.childrenRenderablesToUpdate = { list: [], index: 0 };
    this.structureDidChange = true;
    this.instructionSet = new InstructionSet();
    this._onRenderContainers = [];
    this.textureNeedsUpdate = true;
    this.isCachedAsTexture = false;
    this._matrixDirty = 7;
  }
  init(root) {
    this.root = root;
    if (root._onRender)
      this.addOnRender(root);
    root.didChange = true;
    const children = root.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      const child = children[i2];
      child._updateFlags = 15;
      this.addChild(child);
    }
  }
  enableCacheAsTexture(options = {}) {
    this.textureOptions = options;
    this.isCachedAsTexture = true;
    this.textureNeedsUpdate = true;
  }
  disableCacheAsTexture() {
    this.isCachedAsTexture = false;
    if (this.texture) {
      TexturePool.returnTexture(this.texture, true);
      this.texture = null;
    }
  }
  updateCacheTexture() {
    this.textureNeedsUpdate = true;
    const cachedParent = this._parentCacheAsTextureRenderGroup;
    if (cachedParent && !cachedParent.textureNeedsUpdate) {
      cachedParent.updateCacheTexture();
    }
  }
  reset() {
    this.renderGroupChildren.length = 0;
    for (const i2 in this.childrenToUpdate) {
      const childrenAtDepth = this.childrenToUpdate[i2];
      childrenAtDepth.list.fill(null);
      childrenAtDepth.index = 0;
    }
    this.childrenRenderablesToUpdate.index = 0;
    this.childrenRenderablesToUpdate.list.fill(null);
    this.root = null;
    this.updateTick = 0;
    this.structureDidChange = true;
    this._onRenderContainers.length = 0;
    this.renderGroupParent = null;
    this.disableCacheAsTexture();
  }
  get localTransform() {
    return this.root.localTransform;
  }
  addRenderGroupChild(renderGroupChild) {
    if (renderGroupChild.renderGroupParent) {
      renderGroupChild.renderGroupParent._removeRenderGroupChild(renderGroupChild);
    }
    renderGroupChild.renderGroupParent = this;
    this.renderGroupChildren.push(renderGroupChild);
  }
  _removeRenderGroupChild(renderGroupChild) {
    const index = this.renderGroupChildren.indexOf(renderGroupChild);
    if (index > -1) {
      this.renderGroupChildren.splice(index, 1);
    }
    renderGroupChild.renderGroupParent = null;
  }
  addChild(child) {
    this.structureDidChange = true;
    child.parentRenderGroup = this;
    child.updateTick = -1;
    if (child.parent === this.root) {
      child.relativeRenderGroupDepth = 1;
    } else {
      child.relativeRenderGroupDepth = child.parent.relativeRenderGroupDepth + 1;
    }
    child.didChange = true;
    this.onChildUpdate(child);
    if (child.renderGroup) {
      this.addRenderGroupChild(child.renderGroup);
      return;
    }
    if (child._onRender)
      this.addOnRender(child);
    const children = child.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      this.addChild(children[i2]);
    }
  }
  removeChild(child) {
    this.structureDidChange = true;
    if (child._onRender) {
      if (!child.renderGroup) {
        this.removeOnRender(child);
      }
    }
    child.parentRenderGroup = null;
    if (child.renderGroup) {
      this._removeRenderGroupChild(child.renderGroup);
      return;
    }
    const children = child.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      this.removeChild(children[i2]);
    }
  }
  removeChildren(children) {
    for (let i2 = 0; i2 < children.length; i2++) {
      this.removeChild(children[i2]);
    }
  }
  onChildUpdate(child) {
    let childrenToUpdate = this.childrenToUpdate[child.relativeRenderGroupDepth];
    if (!childrenToUpdate) {
      childrenToUpdate = this.childrenToUpdate[child.relativeRenderGroupDepth] = {
        index: 0,
        list: []
      };
    }
    childrenToUpdate.list[childrenToUpdate.index++] = child;
  }
  updateRenderable(renderable) {
    if (renderable.globalDisplayStatus < 7)
      return;
    this.instructionSet.renderPipes[renderable.renderPipeId].updateRenderable(renderable);
    renderable.didViewUpdate = false;
  }
  onChildViewUpdate(child) {
    this.childrenRenderablesToUpdate.list[this.childrenRenderablesToUpdate.index++] = child;
  }
  get isRenderable() {
    return this.root.localDisplayStatus === 7 && this.worldAlpha > 0;
  }
  /**
   * adding a container to the onRender list will make sure the user function
   * passed in to the user defined 'onRender` callBack
   * @param container - the container to add to the onRender list
   */
  addOnRender(container) {
    this._onRenderContainers.push(container);
  }
  removeOnRender(container) {
    this._onRenderContainers.splice(this._onRenderContainers.indexOf(container), 1);
  }
  runOnRender(renderer) {
    for (let i2 = 0; i2 < this._onRenderContainers.length; i2++) {
      this._onRenderContainers[i2]._onRender(renderer);
    }
  }
  destroy() {
    this.disableCacheAsTexture();
    this.renderGroupParent = null;
    this.root = null;
    this.childrenRenderablesToUpdate = null;
    this.childrenToUpdate = null;
    this.renderGroupChildren = null;
    this._onRenderContainers = null;
    this.instructionSet = null;
  }
  getChildren(out2 = []) {
    const children = this.root.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      this._getChildren(children[i2], out2);
    }
    return out2;
  }
  _getChildren(container, out2 = []) {
    out2.push(container);
    if (container.renderGroup)
      return out2;
    const children = container.children;
    for (let i2 = 0; i2 < children.length; i2++) {
      this._getChildren(children[i2], out2);
    }
    return out2;
  }
  invalidateMatrices() {
    this._matrixDirty = 7;
  }
  /**
   * Returns the inverse of the world transform matrix.
   * @returns {Matrix} The inverse of the world transform matrix.
   */
  get inverseWorldTransform() {
    if ((this._matrixDirty & 1) === 0)
      return this._inverseWorldTransform;
    this._matrixDirty &= -2;
    this._inverseWorldTransform || (this._inverseWorldTransform = new Matrix());
    return this._inverseWorldTransform.copyFrom(this.worldTransform).invert();
  }
  /**
   * Returns the inverse of the texture offset transform matrix.
   * @returns {Matrix} The inverse of the texture offset transform matrix.
   */
  get textureOffsetInverseTransform() {
    if ((this._matrixDirty & 2) === 0)
      return this._textureOffsetInverseTransform;
    this._matrixDirty &= -3;
    this._textureOffsetInverseTransform || (this._textureOffsetInverseTransform = new Matrix());
    return this._textureOffsetInverseTransform.copyFrom(this.inverseWorldTransform).translate(
      -this._textureBounds.x,
      -this._textureBounds.y
    );
  }
  /**
   * Returns the inverse of the parent texture transform matrix.
   * This is used to properly transform coordinates when rendering into cached textures.
   * @returns {Matrix} The inverse of the parent texture transform matrix.
   */
  get inverseParentTextureTransform() {
    if ((this._matrixDirty & 4) === 0)
      return this._inverseParentTextureTransform;
    this._matrixDirty &= -5;
    const parentCacheAsTexture = this._parentCacheAsTextureRenderGroup;
    if (parentCacheAsTexture) {
      this._inverseParentTextureTransform || (this._inverseParentTextureTransform = new Matrix());
      return this._inverseParentTextureTransform.copyFrom(this.worldTransform).prepend(parentCacheAsTexture.inverseWorldTransform).translate(
        -parentCacheAsTexture._textureBounds.x,
        -parentCacheAsTexture._textureBounds.y
      );
    }
    return this.worldTransform;
  }
  /**
   * Returns a matrix that transforms coordinates to the correct coordinate space of the texture being rendered to.
   * This is the texture offset inverse transform of the closest parent RenderGroup that is cached as a texture.
   * @returns {Matrix | null} The transform matrix for the cached texture coordinate space,
   * or null if no parent is cached as texture.
   */
  get cacheToLocalTransform() {
    if (this.isCachedAsTexture) {
      return this.textureOffsetInverseTransform;
    }
    if (!this._parentCacheAsTextureRenderGroup)
      return null;
    return this._parentCacheAsTextureRenderGroup.textureOffsetInverseTransform;
  }
}
function assignWithIgnore(target, options, ignore = {}) {
  for (const key in options) {
    if (!ignore[key] && options[key] !== void 0) {
      target[key] = options[key];
    }
  }
}
const defaultSkew = new ObservablePoint(null);
const defaultPivot = new ObservablePoint(null);
const defaultScale = new ObservablePoint(null, 1, 1);
const defaultOrigin = new ObservablePoint(null);
const UPDATE_COLOR = 1;
const UPDATE_BLEND = 2;
const UPDATE_VISIBLE = 4;
class Container extends EventEmitter {
  constructor(options = {}) {
    super();
    this.uid = uid("renderable");
    this._updateFlags = 15;
    this.renderGroup = null;
    this.parentRenderGroup = null;
    this.parentRenderGroupIndex = 0;
    this.didChange = false;
    this.didViewUpdate = false;
    this.relativeRenderGroupDepth = 0;
    this.children = [];
    this.parent = null;
    this.includeInBuild = true;
    this.measurable = true;
    this.isSimple = true;
    this.parentRenderLayer = null;
    this.updateTick = -1;
    this.localTransform = new Matrix();
    this.relativeGroupTransform = new Matrix();
    this.groupTransform = this.relativeGroupTransform;
    this.destroyed = false;
    this._position = new ObservablePoint(this, 0, 0);
    this._scale = defaultScale;
    this._pivot = defaultPivot;
    this._origin = defaultOrigin;
    this._skew = defaultSkew;
    this._cx = 1;
    this._sx = 0;
    this._cy = 0;
    this._sy = 1;
    this._rotation = 0;
    this.localColor = 16777215;
    this.localAlpha = 1;
    this.groupAlpha = 1;
    this.groupColor = 16777215;
    this.groupColorAlpha = 4294967295;
    this.localBlendMode = "inherit";
    this.groupBlendMode = "normal";
    this.localDisplayStatus = 7;
    this.globalDisplayStatus = 7;
    this._didContainerChangeTick = 0;
    this._didViewChangeTick = 0;
    this._didLocalTransformChangeId = -1;
    this.effects = [];
    assignWithIgnore(this, options, {
      children: true,
      parent: true,
      effects: true
    });
    options.children?.forEach((child) => this.addChild(child));
    options.parent?.addChild(this);
  }
  /**
   * Mixes all enumerable properties and methods from a source object to Container.
   * @param source - The source of properties and methods to mix in.
   * @deprecated since 8.8.0
   */
  static mixin(source2) {
    deprecation("8.8.0", "Container.mixin is deprecated, please use extensions.mixin instead.");
    extensions.mixin(Container, source2);
  }
  // = 'default';
  /**
   * We now use the _didContainerChangeTick and _didViewChangeTick to track changes
   * @deprecated since 8.2.6
   * @ignore
   */
  set _didChangeId(value) {
    this._didViewChangeTick = value >> 12 & 4095;
    this._didContainerChangeTick = value & 4095;
  }
  /** @ignore */
  get _didChangeId() {
    return this._didContainerChangeTick & 4095 | (this._didViewChangeTick & 4095) << 12;
  }
  /**
   * Adds one or more children to the container.
   * The children will be rendered as part of this container's display list.
   * @example
   * ```ts
   * // Add a single child
   * container.addChild(sprite);
   *
   * // Add multiple children
   * container.addChild(background, player, foreground);
   *
   * // Add with type checking
   * const sprite = container.addChild<Sprite>(new Sprite(texture));
   * sprite.tint = 'red';
   * ```
   * @param children - The Container(s) to add to the container
   * @returns The first child that was added
   * @see {@link Container#removeChild} For removing children
   * @see {@link Container#addChildAt} For adding at specific index
   */
  addChild(...children) {
    if (!this.allowChildren) {
      deprecation(v8_0_0, "addChild: Only Containers will be allowed to add children in v8.0.0");
    }
    if (children.length > 1) {
      for (let i2 = 0; i2 < children.length; i2++) {
        this.addChild(children[i2]);
      }
      return children[0];
    }
    const child = children[0];
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (child.parent === this) {
      this.children.splice(this.children.indexOf(child), 1);
      this.children.push(child);
      if (renderGroup) {
        renderGroup.structureDidChange = true;
      }
      return child;
    }
    if (child.parent) {
      child.parent.removeChild(child);
    }
    this.children.push(child);
    if (this.sortableChildren)
      this.sortDirty = true;
    child.parent = this;
    child.didChange = true;
    child._updateFlags = 15;
    if (renderGroup) {
      renderGroup.addChild(child);
    }
    this.emit("childAdded", child, this, this.children.length - 1);
    child.emit("added", this);
    this._didViewChangeTick++;
    if (child._zIndex !== 0) {
      child.depthOfChildModified();
    }
    return child;
  }
  /**
   * Removes one or more children from the container.
   * When removing multiple children, events will be triggered for each child in sequence.
   * @example
   * ```ts
   * // Remove a single child
   * const removed = container.removeChild(sprite);
   *
   * // Remove multiple children
   * const bg = container.removeChild(background, player, userInterface);
   *
   * // Remove with type checking
   * const sprite = container.removeChild<Sprite>(childSprite);
   * sprite.texture = newTexture;
   * ```
   * @param children - The Container(s) to remove
   * @returns The first child that was removed
   * @see {@link Container#addChild} For adding children
   * @see {@link Container#removeChildren} For removing multiple children
   */
  removeChild(...children) {
    if (children.length > 1) {
      for (let i2 = 0; i2 < children.length; i2++) {
        this.removeChild(children[i2]);
      }
      return children[0];
    }
    const child = children[0];
    const index = this.children.indexOf(child);
    if (index > -1) {
      this._didViewChangeTick++;
      this.children.splice(index, 1);
      if (this.renderGroup) {
        this.renderGroup.removeChild(child);
      } else if (this.parentRenderGroup) {
        this.parentRenderGroup.removeChild(child);
      }
      if (child.parentRenderLayer) {
        child.parentRenderLayer.detach(child);
      }
      child.parent = null;
      this.emit("childRemoved", child, this, index);
      child.emit("removed", this);
    }
    return child;
  }
  /** @ignore */
  _onUpdate(point) {
    if (point) {
      if (point === this._skew) {
        this._updateSkew();
      }
    }
    this._didContainerChangeTick++;
    if (this.didChange)
      return;
    this.didChange = true;
    if (this.parentRenderGroup) {
      this.parentRenderGroup.onChildUpdate(this);
    }
  }
  set isRenderGroup(value) {
    if (!!this.renderGroup === value)
      return;
    if (value) {
      this.enableRenderGroup();
    } else {
      this.disableRenderGroup();
    }
  }
  /**
   * Returns true if this container is a render group.
   * This means that it will be rendered as a separate pass, with its own set of instructions
   * @advanced
   */
  get isRenderGroup() {
    return !!this.renderGroup;
  }
  /**
   * Calling this enables a render group for this container.
   * This means it will be rendered as a separate set of instructions.
   * The transform of the container will also be handled on the GPU rather than the CPU.
   * @advanced
   */
  enableRenderGroup() {
    if (this.renderGroup)
      return;
    const parentRenderGroup = this.parentRenderGroup;
    parentRenderGroup?.removeChild(this);
    this.renderGroup = BigPool.get(RenderGroup, this);
    this.groupTransform = Matrix.IDENTITY;
    parentRenderGroup?.addChild(this);
    this._updateIsSimple();
  }
  /**
   * This will disable the render group for this container.
   * @advanced
   */
  disableRenderGroup() {
    if (!this.renderGroup)
      return;
    const parentRenderGroup = this.parentRenderGroup;
    parentRenderGroup?.removeChild(this);
    BigPool.return(this.renderGroup);
    this.renderGroup = null;
    this.groupTransform = this.relativeGroupTransform;
    parentRenderGroup?.addChild(this);
    this._updateIsSimple();
  }
  /** @ignore */
  _updateIsSimple() {
    this.isSimple = !this.renderGroup && this.effects.length === 0;
  }
  /**
   * Current transform of the object based on world (parent) factors.
   *
   * This matrix represents the absolute transformation in the scene graph.
   * @example
   * ```ts
   * // Get world position
   * const worldPos = container.worldTransform;
   * console.log(`World position: (${worldPos.tx}, ${worldPos.ty})`);
   * ```
   * @readonly
   * @see {@link Container#localTransform} For local space transform
   */
  get worldTransform() {
    this._worldTransform || (this._worldTransform = new Matrix());
    if (this.renderGroup) {
      this._worldTransform.copyFrom(this.renderGroup.worldTransform);
    } else if (this.parentRenderGroup) {
      this._worldTransform.appendFrom(this.relativeGroupTransform, this.parentRenderGroup.worldTransform);
    }
    return this._worldTransform;
  }
  /**
   * The position of the container on the x axis relative to the local coordinates of the parent.
   *
   * An alias to position.x
   * @example
   * ```ts
   * // Basic position
   * container.x = 100;
   * ```
   */
  get x() {
    return this._position.x;
  }
  set x(value) {
    this._position.x = value;
  }
  /**
   * The position of the container on the y axis relative to the local coordinates of the parent.
   *
   * An alias to position.y
   * @example
   * ```ts
   * // Basic position
   * container.y = 200;
   * ```
   */
  get y() {
    return this._position.y;
  }
  set y(value) {
    this._position.y = value;
  }
  /**
   * The coordinate of the object relative to the local coordinates of the parent.
   * @example
   * ```ts
   * // Basic position setting
   * container.position.set(100, 200);
   * container.position.set(100); // Sets both x and y to 100
   * // Using point data
   * container.position = { x: 50, y: 75 };
   * ```
   * @since 4.0.0
   */
  get position() {
    return this._position;
  }
  set position(value) {
    this._position.copyFrom(value);
  }
  /**
   * The rotation of the object in radians.
   *
   * > [!NOTE] 'rotation' and 'angle' have the same effect on a display object;
   * > rotation is in radians, angle is in degrees.
   * @example
   * ```ts
   * // Basic rotation
   * container.rotation = Math.PI / 4; // 45 degrees
   *
   * // Convert from degrees
   * const degrees = 45;
   * container.rotation = degrees * Math.PI / 180;
   *
   * // Rotate around center
   * container.pivot.set(container.width / 2, container.height / 2);
   * container.rotation = Math.PI; // 180 degrees
   *
   * // Rotate around center with origin
   * container.origin.set(container.width / 2, container.height / 2);
   * container.rotation = Math.PI; // 180 degrees
   * ```
   */
  get rotation() {
    return this._rotation;
  }
  set rotation(value) {
    if (this._rotation !== value) {
      this._rotation = value;
      this._onUpdate(this._skew);
    }
  }
  /**
   * The angle of the object in degrees.
   *
   * > [!NOTE] 'rotation' and 'angle' have the same effect on a display object;
   * > rotation is in radians, angle is in degrees.
   * @example
   * ```ts
   * // Basic angle rotation
   * sprite.angle = 45; // 45 degrees
   *
   * // Rotate around center
   * sprite.pivot.set(sprite.width / 2, sprite.height / 2);
   * sprite.angle = 180; // Half rotation
   *
   * // Rotate around center with origin
   * sprite.origin.set(sprite.width / 2, sprite.height / 2);
   * sprite.angle = 180; // Half rotation
   *
   * // Reset rotation
   * sprite.angle = 0;
   * ```
   */
  get angle() {
    return this.rotation * RAD_TO_DEG;
  }
  set angle(value) {
    this.rotation = value * DEG_TO_RAD;
  }
  /**
   * The center of rotation, scaling, and skewing for this display object in its local space.
   * The `position` is the projection of `pivot` in the parent's local space.
   *
   * By default, the pivot is the origin (0, 0).
   * @example
   * ```ts
   * // Rotate around center
   * container.pivot.set(container.width / 2, container.height / 2);
   * container.rotation = Math.PI; // Rotates around center
   * ```
   * @since 4.0.0
   */
  get pivot() {
    if (this._pivot === defaultPivot) {
      this._pivot = new ObservablePoint(this, 0, 0);
    }
    return this._pivot;
  }
  set pivot(value) {
    if (this._pivot === defaultPivot) {
      this._pivot = new ObservablePoint(this, 0, 0);
      if (this._origin !== defaultOrigin) {
        warn(`Setting both a pivot and origin on a Container is not recommended. This can lead to unexpected behavior if not handled carefully.`);
      }
    }
    typeof value === "number" ? this._pivot.set(value) : this._pivot.copyFrom(value);
  }
  /**
   * The skew factor for the object in radians. Skewing is a transformation that distorts
   * the object by rotating it differently at each point, creating a non-uniform shape.
   * @example
   * ```ts
   * // Basic skewing
   * container.skew.set(0.5, 0); // Skew horizontally
   * container.skew.set(0, 0.5); // Skew vertically
   *
   * // Skew with point data
   * container.skew = { x: 0.3, y: 0.3 }; // Diagonal skew
   *
   * // Reset skew
   * container.skew.set(0, 0);
   *
   * // Animate skew
   * app.ticker.add(() => {
   *     // Create wave effect
   *     container.skew.x = Math.sin(Date.now() / 1000) * 0.3;
   * });
   *
   * // Combine with rotation
   * container.rotation = Math.PI / 4; // 45 degrees
   * container.skew.set(0.2, 0.2); // Skew the rotated object
   * ```
   * @since 4.0.0
   * @type {ObservablePoint} Point-like object with x/y properties in radians
   * @default {x: 0, y: 0}
   */
  get skew() {
    if (this._skew === defaultSkew) {
      this._skew = new ObservablePoint(this, 0, 0);
    }
    return this._skew;
  }
  set skew(value) {
    if (this._skew === defaultSkew) {
      this._skew = new ObservablePoint(this, 0, 0);
    }
    this._skew.copyFrom(value);
  }
  /**
   * The scale factors of this object along the local coordinate axes.
   *
   * The default scale is (1, 1).
   * @example
   * ```ts
   * // Basic scaling
   * container.scale.set(2, 2); // Scales to double size
   * container.scale.set(2); // Scales uniformly to double size
   * container.scale = 2; // Scales uniformly to double size
   * // Scale to a specific width and height
   * container.setSize(200, 100); // Sets width to 200 and height to 100
   * ```
   * @since 4.0.0
   */
  get scale() {
    if (this._scale === defaultScale) {
      this._scale = new ObservablePoint(this, 1, 1);
    }
    return this._scale;
  }
  set scale(value) {
    if (this._scale === defaultScale) {
      this._scale = new ObservablePoint(this, 0, 0);
    }
    if (typeof value === "string") {
      value = parseFloat(value);
    }
    typeof value === "number" ? this._scale.set(value) : this._scale.copyFrom(value);
  }
  /**
   * @experimental
   * The origin point around which the container rotates and scales without affecting its position.
   * Unlike pivot, changing the origin will not move the container's position.
   * @example
   * ```ts
   * // Rotate around center point
   * container.origin.set(container.width / 2, container.height / 2);
   * container.rotation = Math.PI; // Rotates around center
   *
   * // Reset origin
   * container.origin.set(0, 0);
   * ```
   */
  get origin() {
    if (this._origin === defaultOrigin) {
      this._origin = new ObservablePoint(this, 0, 0);
    }
    return this._origin;
  }
  set origin(value) {
    if (this._origin === defaultOrigin) {
      this._origin = new ObservablePoint(this, 0, 0);
      if (this._pivot !== defaultPivot) {
        warn(`Setting both a pivot and origin on a Container is not recommended. This can lead to unexpected behavior if not handled carefully.`);
      }
    }
    typeof value === "number" ? this._origin.set(value) : this._origin.copyFrom(value);
  }
  /**
   * The width of the Container, setting this will actually modify the scale to achieve the value set.
   * > [!NOTE] Changing the width will adjust the scale.x property of the container while maintaining its aspect ratio.
   * > [!NOTE] If you want to set both width and height at the same time, use {@link Container#setSize}
   * as it is more optimized by not recalculating the local bounds twice.
   * @example
   * ```ts
   * // Basic width setting
   * container.width = 100;
   * // Optimized width setting
   * container.setSize(100, 100);
   * ```
   */
  get width() {
    return Math.abs(this.scale.x * this.getLocalBounds().width);
  }
  set width(value) {
    const localWidth = this.getLocalBounds().width;
    this._setWidth(value, localWidth);
  }
  /**
   * The height of the Container,
   * > [!NOTE] Changing the height will adjust the scale.y property of the container while maintaining its aspect ratio.
   * > [!NOTE] If you want to set both width and height at the same time, use {@link Container#setSize}
   * as it is more optimized by not recalculating the local bounds twice.
   * @example
   * ```ts
   * // Basic height setting
   * container.height = 200;
   * // Optimized height setting
   * container.setSize(100, 200);
   * ```
   */
  get height() {
    return Math.abs(this.scale.y * this.getLocalBounds().height);
  }
  set height(value) {
    const localHeight = this.getLocalBounds().height;
    this._setHeight(value, localHeight);
  }
  /**
   * Retrieves the size of the container as a [Size]{@link Size} object.
   *
   * This is faster than get the width and height separately.
   * @example
   * ```ts
   * // Basic size retrieval
   * const size = container.getSize();
   * console.log(`Size: ${size.width}x${size.height}`);
   *
   * // Reuse existing size object
   * const reuseSize = { width: 0, height: 0 };
   * container.getSize(reuseSize);
   * ```
   * @param out - Optional object to store the size in.
   * @returns The size of the container.
   */
  getSize(out2) {
    if (!out2) {
      out2 = {};
    }
    const bounds = this.getLocalBounds();
    out2.width = Math.abs(this.scale.x * bounds.width);
    out2.height = Math.abs(this.scale.y * bounds.height);
    return out2;
  }
  /**
   * Sets the size of the container to the specified width and height.
   * This is more efficient than setting width and height separately as it only recalculates bounds once.
   * @example
   * ```ts
   * // Basic size setting
   * container.setSize(100, 200);
   *
   * // Set uniform size
   * container.setSize(100); // Sets both width and height to 100
   * ```
   * @param value - This can be either a number or a [Size]{@link Size} object.
   * @param height - The height to set. Defaults to the value of `width` if not provided.
   */
  setSize(value, height) {
    const size = this.getLocalBounds();
    if (typeof value === "object") {
      height = value.height ?? value.width;
      value = value.width;
    } else {
      height ?? (height = value);
    }
    value !== void 0 && this._setWidth(value, size.width);
    height !== void 0 && this._setHeight(height, size.height);
  }
  /** Called when the skew or the rotation changes. */
  _updateSkew() {
    const rotation = this._rotation;
    const skew = this._skew;
    this._cx = Math.cos(rotation + skew._y);
    this._sx = Math.sin(rotation + skew._y);
    this._cy = -Math.sin(rotation - skew._x);
    this._sy = Math.cos(rotation - skew._x);
  }
  /**
   * Updates the transform properties of the container.
   * Allows partial updates of transform properties for optimized manipulation.
   * @example
   * ```ts
   * // Basic transform update
   * container.updateTransform({
   *     x: 100,
   *     y: 200,
   *     rotation: Math.PI / 4
   * });
   *
   * // Scale and rotate around center
   * sprite.updateTransform({
   *     pivotX: sprite.width / 2,
   *     pivotY: sprite.height / 2,
   *     scaleX: 2,
   *     scaleY: 2,
   *     rotation: Math.PI
   * });
   *
   * // Update position only
   * button.updateTransform({
   *     x: button.x + 10, // Move right
   *     y: button.y      // Keep same y
   * });
   * ```
   * @param opts - Transform options to update
   * @param opts.x - The x position
   * @param opts.y - The y position
   * @param opts.scaleX - The x-axis scale factor
   * @param opts.scaleY - The y-axis scale factor
   * @param opts.rotation - The rotation in radians
   * @param opts.skewX - The x-axis skew factor
   * @param opts.skewY - The y-axis skew factor
   * @param opts.pivotX - The x-axis pivot point
   * @param opts.pivotY - The y-axis pivot point
   * @returns This container, for chaining
   * @see {@link Container#setFromMatrix} For matrix-based transforms
   * @see {@link Container#position} For direct position access
   */
  updateTransform(opts) {
    this.position.set(
      typeof opts.x === "number" ? opts.x : this.position.x,
      typeof opts.y === "number" ? opts.y : this.position.y
    );
    this.scale.set(
      typeof opts.scaleX === "number" ? opts.scaleX || 1 : this.scale.x,
      typeof opts.scaleY === "number" ? opts.scaleY || 1 : this.scale.y
    );
    this.rotation = typeof opts.rotation === "number" ? opts.rotation : this.rotation;
    this.skew.set(
      typeof opts.skewX === "number" ? opts.skewX : this.skew.x,
      typeof opts.skewY === "number" ? opts.skewY : this.skew.y
    );
    this.pivot.set(
      typeof opts.pivotX === "number" ? opts.pivotX : this.pivot.x,
      typeof opts.pivotY === "number" ? opts.pivotY : this.pivot.y
    );
    this.origin.set(
      typeof opts.originX === "number" ? opts.originX : this.origin.x,
      typeof opts.originY === "number" ? opts.originY : this.origin.y
    );
    return this;
  }
  /**
   * Updates the local transform properties by decomposing the given matrix.
   * Extracts position, scale, rotation, and skew from a transformation matrix.
   * @example
   * ```ts
   * // Basic matrix transform
   * const matrix = new Matrix()
   *     .translate(100, 100)
   *     .rotate(Math.PI / 4)
   *     .scale(2, 2);
   *
   * container.setFromMatrix(matrix);
   *
   * // Copy transform from another container
   * const source = new Container();
   * source.position.set(100, 100);
   * source.rotation = Math.PI / 2;
   *
   * target.setFromMatrix(source.localTransform);
   *
   * // Reset transform
   * container.setFromMatrix(Matrix.IDENTITY);
   * ```
   * @param matrix - The matrix to use for updating the transform
   * @see {@link Container#updateTransform} For property-based updates
   * @see {@link Matrix#decompose} For matrix decomposition details
   */
  setFromMatrix(matrix) {
    matrix.decompose(this);
  }
  /** Updates the local transform. */
  updateLocalTransform() {
    const localTransformChangeId = this._didContainerChangeTick;
    if (this._didLocalTransformChangeId === localTransformChangeId)
      return;
    this._didLocalTransformChangeId = localTransformChangeId;
    const lt = this.localTransform;
    const scale = this._scale;
    const pivot = this._pivot;
    const origin = this._origin;
    const position = this._position;
    const sx = scale._x;
    const sy = scale._y;
    const px = pivot._x;
    const py = pivot._y;
    const ox = -origin._x;
    const oy = -origin._y;
    lt.a = this._cx * sx;
    lt.b = this._sx * sx;
    lt.c = this._cy * sy;
    lt.d = this._sy * sy;
    lt.tx = position._x - (px * lt.a + py * lt.c) + (ox * lt.a + oy * lt.c) - ox;
    lt.ty = position._y - (px * lt.b + py * lt.d) + (ox * lt.b + oy * lt.d) - oy;
  }
  // / ///// color related stuff
  set alpha(value) {
    if (value === this.localAlpha)
      return;
    this.localAlpha = value;
    this._updateFlags |= UPDATE_COLOR;
    this._onUpdate();
  }
  /**
   * The opacity of the object relative to its parent's opacity.
   * Value ranges from 0 (fully transparent) to 1 (fully opaque).
   * @example
   * ```ts
   * // Basic transparency
   * sprite.alpha = 0.5; // 50% opacity
   *
   * // Inherited opacity
   * container.alpha = 0.5;
   * const child = new Sprite(texture);
   * child.alpha = 0.5;
   * container.addChild(child);
   * // child's effective opacity is 0.25 (0.5 * 0.5)
   * ```
   * @default 1
   * @see {@link Container#visible} For toggling visibility
   * @see {@link Container#renderable} For render control
   */
  get alpha() {
    return this.localAlpha;
  }
  set tint(value) {
    const tempColor = Color.shared.setValue(value ?? 16777215);
    const bgr = tempColor.toBgrNumber();
    if (bgr === this.localColor)
      return;
    this.localColor = bgr;
    this._updateFlags |= UPDATE_COLOR;
    this._onUpdate();
  }
  /**
   * The tint applied to the sprite.
   *
   * This can be any valid {@link ColorSource}.
   * @example
   * ```ts
   * // Basic color tinting
   * container.tint = 0xff0000; // Red tint
   * container.tint = 'red';    // Same as above
   * container.tint = '#00ff00'; // Green
   * container.tint = 'rgb(0,0,255)'; // Blue
   *
   * // Remove tint
   * container.tint = 0xffffff; // White = no tint
   * container.tint = null;     // Also removes tint
   * ```
   * @default 0xFFFFFF
   * @see {@link Container#alpha} For transparency
   * @see {@link Container#visible} For visibility control
   */
  get tint() {
    return bgr2rgb(this.localColor);
  }
  // / //////////////// blend related stuff
  set blendMode(value) {
    if (this.localBlendMode === value)
      return;
    if (this.parentRenderGroup) {
      this.parentRenderGroup.structureDidChange = true;
    }
    this._updateFlags |= UPDATE_BLEND;
    this.localBlendMode = value;
    this._onUpdate();
  }
  /**
   * The blend mode to be applied to the sprite. Controls how pixels are blended when rendering.
   *
   * Setting to 'normal' will reset to default blending.
   * > [!NOTE] More blend modes are available after importing the `pixi.js/advanced-blend-modes` sub-export.
   * @example
   * ```ts
   * // Basic blend modes
   * sprite.blendMode = 'add';        // Additive blending
   * sprite.blendMode = 'multiply';   // Multiply colors
   * sprite.blendMode = 'screen';     // Screen blend
   *
   * // Reset blend mode
   * sprite.blendMode = 'normal';     // Normal blending
   * ```
   * @default 'normal'
   * @see {@link Container#alpha} For transparency
   * @see {@link Container#tint} For color adjustments
   */
  get blendMode() {
    return this.localBlendMode;
  }
  // / ///////// VISIBILITY / RENDERABLE /////////////////
  /**
   * The visibility of the object. If false the object will not be drawn,
   * and the transform will not be updated.
   * @example
   * ```ts
   * // Basic visibility toggle
   * sprite.visible = false; // Hide sprite
   * sprite.visible = true;  // Show sprite
   * ```
   * @default true
   * @see {@link Container#renderable} For render-only control
   * @see {@link Container#alpha} For transparency
   */
  get visible() {
    return !!(this.localDisplayStatus & 2);
  }
  set visible(value) {
    const valueNumber = value ? 2 : 0;
    if ((this.localDisplayStatus & 2) === valueNumber)
      return;
    if (this.parentRenderGroup) {
      this.parentRenderGroup.structureDidChange = true;
    }
    this._updateFlags |= UPDATE_VISIBLE;
    this.localDisplayStatus ^= 2;
    this._onUpdate();
  }
  /** @ignore */
  get culled() {
    return !(this.localDisplayStatus & 4);
  }
  /** @ignore */
  set culled(value) {
    const valueNumber = value ? 0 : 4;
    if ((this.localDisplayStatus & 4) === valueNumber)
      return;
    if (this.parentRenderGroup) {
      this.parentRenderGroup.structureDidChange = true;
    }
    this._updateFlags |= UPDATE_VISIBLE;
    this.localDisplayStatus ^= 4;
    this._onUpdate();
  }
  /**
   * Controls whether this object can be rendered. If false the object will not be drawn,
   * but the transform will still be updated. This is different from visible, which skips
   * transform updates.
   * @example
   * ```ts
   * // Basic render control
   * sprite.renderable = false; // Skip rendering
   * sprite.renderable = true;  // Enable rendering
   * ```
   * @default true
   * @see {@link Container#visible} For skipping transform updates
   * @see {@link Container#alpha} For transparency
   */
  get renderable() {
    return !!(this.localDisplayStatus & 1);
  }
  set renderable(value) {
    const valueNumber = value ? 1 : 0;
    if ((this.localDisplayStatus & 1) === valueNumber)
      return;
    this._updateFlags |= UPDATE_VISIBLE;
    this.localDisplayStatus ^= 1;
    if (this.parentRenderGroup) {
      this.parentRenderGroup.structureDidChange = true;
    }
    this._onUpdate();
  }
  /**
   * Whether or not the object should be rendered.
   * @advanced
   */
  get isRenderable() {
    return this.localDisplayStatus === 7 && this.groupAlpha > 0;
  }
  /**
   * Removes all internal references and listeners as well as removes children from the display list.
   * Do not use a Container after calling `destroy`.
   * @param options - Options parameter. A boolean will act as if all options
   *  have been set to that value
   * @example
   * ```ts
   * container.destroy();
   * container.destroy(true);
   * container.destroy({ children: true });
   * container.destroy({ children: true, texture: true, textureSource: true });
   * ```
   */
  destroy(options = false) {
    if (this.destroyed)
      return;
    this.destroyed = true;
    let oldChildren;
    if (this.children.length) {
      oldChildren = this.removeChildren(0, this.children.length);
    }
    this.removeFromParent();
    this.parent = null;
    this._maskEffect = null;
    this._filterEffect = null;
    this.effects = null;
    this._position = null;
    this._scale = null;
    this._pivot = null;
    this._origin = null;
    this._skew = null;
    this.emit("destroyed", this);
    this.removeAllListeners();
    const destroyChildren = typeof options === "boolean" ? options : options?.children;
    if (destroyChildren && oldChildren) {
      for (let i2 = 0; i2 < oldChildren.length; ++i2) {
        oldChildren[i2].destroy(options);
      }
    }
    this.renderGroup?.destroy();
    this.renderGroup = null;
  }
}
extensions.mixin(
  Container,
  childrenHelperMixin,
  getFastGlobalBoundsMixin,
  toLocalGlobalMixin,
  onRenderMixin,
  measureMixin,
  effectsMixin,
  findMixin,
  sortMixin,
  cullingMixin,
  cacheAsTextureMixin,
  getGlobalMixin,
  collectRenderablesMixin
);
class ViewContainer extends Container {
  constructor(options) {
    super(options);
    this.canBundle = true;
    this.allowChildren = false;
    this._roundPixels = 0;
    this._lastUsed = -1;
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this.autoGarbageCollect = true;
    this._gcLastUsed = -1;
    this._bounds = new Bounds(0, 1, 0, 0);
    this._boundsDirty = true;
    this.autoGarbageCollect = options.autoGarbageCollect ?? true;
  }
  /**
   * The local bounds of the view in its own coordinate space.
   * Bounds are automatically updated when the view's content changes.
   * @example
   * ```ts
   * // Get bounds dimensions
   * const bounds = view.bounds;
   * console.log(`Width: ${bounds.maxX - bounds.minX}`);
   * console.log(`Height: ${bounds.maxY - bounds.minY}`);
   * ```
   * @returns The rectangular bounds of the view
   * @see {@link Bounds} For bounds operations
   */
  get bounds() {
    if (!this._boundsDirty)
      return this._bounds;
    this.updateBounds();
    this._boundsDirty = false;
    return this._bounds;
  }
  /**
   * Whether or not to round the x/y position of the sprite.
   * @example
   * ```ts
   * // Enable pixel rounding for crisp rendering
   * view.roundPixels = true;
   * ```
   * @default false
   */
  get roundPixels() {
    return !!this._roundPixels;
  }
  set roundPixels(value) {
    this._roundPixels = value ? 1 : 0;
  }
  /**
   * Checks if the object contains the given point in local coordinates.
   * Uses the view's bounds for hit testing.
   * @example
   * ```ts
   * // Basic point check
   * const localPoint = { x: 50, y: 25 };
   * const contains = view.containsPoint(localPoint);
   * console.log('Point is inside:', contains);
   * ```
   * @param point - The point to check in local coordinates
   * @returns True if the point is within the view's bounds
   * @see {@link ViewContainer#bounds} For the bounds used in hit testing
   * @see {@link Container#toLocal} For converting global coordinates to local
   */
  containsPoint(point) {
    const bounds = this.bounds;
    const { x: x2, y: y2 } = point;
    return x2 >= bounds.minX && x2 <= bounds.maxX && y2 >= bounds.minY && y2 <= bounds.maxY;
  }
  /** @private */
  onViewUpdate() {
    this._didViewChangeTick++;
    this._boundsDirty = true;
    if (this.didViewUpdate)
      return;
    this.didViewUpdate = true;
    const renderGroup = this.renderGroup || this.parentRenderGroup;
    if (renderGroup) {
      renderGroup.onChildViewUpdate(this);
    }
  }
  /** Unloads the GPU data from the view. */
  unload() {
    this.emit("unload", this);
    for (const key in this._gpuData) {
      this._gpuData[key]?.destroy();
    }
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this.onViewUpdate();
  }
  destroy(options) {
    this.unload();
    super.destroy(options);
    this._bounds = null;
  }
  /**
   * Collects renderables for the view container.
   * @param instructionSet - The instruction set to collect renderables for.
   * @param renderer - The renderer to collect renderables for.
   * @param currentLayer - The current render layer.
   * @internal
   */
  collectRenderablesSimple(instructionSet, renderer, currentLayer) {
    const { renderPipes } = renderer;
    renderPipes.blendMode.pushBlendMode(this, this.groupBlendMode, instructionSet);
    const rp = renderPipes;
    rp[this.renderPipeId].addRenderable(this, instructionSet);
    this.didViewUpdate = false;
    const children = this.children;
    const length = children.length;
    for (let i2 = 0; i2 < length; i2++) {
      children[i2].collectRenderables(instructionSet, renderer, currentLayer);
    }
    renderPipes.blendMode.popBlendMode(instructionSet);
  }
}
class Sprite extends ViewContainer {
  /**
   * @param options - The options for creating the sprite.
   */
  constructor(options = Texture.EMPTY) {
    if (options instanceof Texture) {
      options = { texture: options };
    }
    const { texture = Texture.EMPTY, anchor, roundPixels, width, height, ...rest } = options;
    super({
      label: "Sprite",
      ...rest
    });
    this.renderPipeId = "sprite";
    this.batched = true;
    this._visualBounds = { minX: 0, maxX: 1, minY: 0, maxY: 0 };
    this._anchor = new ObservablePoint(
      {
        _onUpdate: () => {
          this.onViewUpdate();
        }
      }
    );
    if (anchor) {
      this.anchor = anchor;
    } else if (texture.defaultAnchor) {
      this.anchor = texture.defaultAnchor;
    }
    this.texture = texture;
    this.allowChildren = false;
    this.roundPixels = roundPixels ?? false;
    if (width !== void 0)
      this.width = width;
    if (height !== void 0)
      this.height = height;
  }
  /**
   * Creates a new sprite based on a source texture, image, video, or canvas element.
   * This is a convenience method that automatically creates and manages textures.
   * @example
   * ```ts
   * // Create from path or URL
   * const sprite = Sprite.from('assets/image.png');
   *
   * // Create from existing texture
   * const sprite = Sprite.from(texture);
   *
   * // Create from canvas
   * const canvas = document.createElement('canvas');
   * const sprite = Sprite.from(canvas, true); // Skip caching new texture
   * ```
   * @param source - The source to create the sprite from. Can be a path to an image, a texture,
   * or any valid texture source (canvas, video, etc.)
   * @param skipCache - Whether to skip adding to the texture cache when creating a new texture
   * @returns A new sprite based on the source
   * @see {@link Texture.from} For texture creation details
   * @see {@link Assets} For asset loading and management
   */
  static from(source2, skipCache = false) {
    if (source2 instanceof Texture) {
      return new Sprite(source2);
    }
    return new Sprite(Texture.from(source2, skipCache));
  }
  set texture(value) {
    value || (value = Texture.EMPTY);
    const currentTexture = this._texture;
    if (currentTexture === value)
      return;
    if (currentTexture && currentTexture.dynamic)
      currentTexture.off("update", this.onViewUpdate, this);
    if (value.dynamic)
      value.on("update", this.onViewUpdate, this);
    this._texture = value;
    if (this._width) {
      this._setWidth(this._width, this._texture.orig.width);
    }
    if (this._height) {
      this._setHeight(this._height, this._texture.orig.height);
    }
    this.onViewUpdate();
  }
  /**
   * The texture that is displayed by the sprite. When changed, automatically updates
   * the sprite dimensions and manages texture event listeners.
   * @example
   * ```ts
   * // Create sprite with texture
   * const sprite = new Sprite({
   *     texture: Texture.from('sprite.png')
   * });
   *
   * // Update texture
   * sprite.texture = Texture.from('newSprite.png');
   *
   * // Use texture from spritesheet
   * const sheet = await Assets.load('spritesheet.json');
   * sprite.texture = sheet.textures['frame1.png'];
   *
   * // Reset to empty texture
   * sprite.texture = Texture.EMPTY;
   * ```
   * @see {@link Texture} For texture creation and management
   * @see {@link Assets} For asset loading
   */
  get texture() {
    return this._texture;
  }
  /**
   * The bounds of the sprite, taking into account the texture's trim area.
   * @example
   * ```ts
   * const texture = new Texture({
   *     source: new TextureSource({ width: 300, height: 300 }),
   *     frame: new Rectangle(196, 66, 58, 56),
   *     trim: new Rectangle(4, 4, 58, 56),
   *     orig: new Rectangle(0, 0, 64, 64),
   *     rotate: 2,
   * });
   * const sprite = new Sprite(texture);
   * const visualBounds = sprite.visualBounds;
   * // console.log(visualBounds); // { minX: -4, maxX: 62, minY: -4, maxY: 60 }
   */
  get visualBounds() {
    updateQuadBounds(this._visualBounds, this._anchor, this._texture);
    return this._visualBounds;
  }
  /**
   * @deprecated
   * @ignore
   */
  get sourceBounds() {
    deprecation("8.6.1", "Sprite.sourceBounds is deprecated, use visualBounds instead.");
    return this.visualBounds;
  }
  /** @private */
  updateBounds() {
    const anchor = this._anchor;
    const texture = this._texture;
    const bounds = this._bounds;
    const { width, height } = texture.orig;
    bounds.minX = -anchor._x * width;
    bounds.maxX = bounds.minX + width;
    bounds.minY = -anchor._y * height;
    bounds.maxY = bounds.minY + height;
  }
  /**
   * Destroys this sprite renderable and optionally its texture.
   * @param options - Options parameter. A boolean will act as if all options
   *  have been set to that value
   * @example
   * sprite.destroy();
   * sprite.destroy(true);
   * sprite.destroy({ texture: true, textureSource: true });
   */
  destroy(options = false) {
    super.destroy(options);
    const destroyTexture = typeof options === "boolean" ? options : options?.texture;
    if (destroyTexture) {
      const destroyTextureSource = typeof options === "boolean" ? options : options?.textureSource;
      this._texture.destroy(destroyTextureSource);
    }
    this._texture = null;
    this._visualBounds = null;
    this._bounds = null;
    this._anchor = null;
  }
  /**
   * The anchor sets the origin point of the sprite. The default value is taken from the {@link Texture}
   * and passed to the constructor.
   *
   * - The default is `(0,0)`, this means the sprite's origin is the top left.
   * - Setting the anchor to `(0.5,0.5)` means the sprite's origin is centered.
   * - Setting the anchor to `(1,1)` would mean the sprite's origin point will be the bottom right corner.
   *
   * If you pass only single parameter, it will set both x and y to the same value as shown in the example below.
   * @example
   * ```ts
   * // Center the anchor point
   * sprite.anchor = 0.5; // Sets both x and y to 0.5
   * sprite.position.set(400, 300); // Sprite will be centered at this position
   *
   * // Set specific x/y anchor points
   * sprite.anchor = {
   *     x: 1, // Right edge
   *     y: 0  // Top edge
   * };
   *
   * // Using individual coordinates
   * sprite.anchor.set(0.5, 1); // Center-bottom
   *
   * // For rotation around center
   * sprite.anchor.set(0.5);
   * sprite.rotation = Math.PI / 4; // 45 degrees around center
   *
   * // For scaling from center
   * sprite.anchor.set(0.5);
   * sprite.scale.set(2); // Scales from center point
   * ```
   */
  get anchor() {
    return this._anchor;
  }
  set anchor(value) {
    typeof value === "number" ? this._anchor.set(value) : this._anchor.copyFrom(value);
  }
  /**
   * The width of the sprite, setting this will actually modify the scale to achieve the value set.
   * @example
   * ```ts
   * // Set width directly
   * sprite.width = 200;
   * console.log(sprite.scale.x); // Scale adjusted to match width
   *
   * // Set width while preserving aspect ratio
   * const ratio = sprite.height / sprite.width;
   * sprite.width = 300;
   * sprite.height = 300 * ratio;
   *
   * // For better performance when setting both width and height
   * sprite.setSize(300, 400); // Avoids recalculating bounds twice
   *
   * // Reset to original texture size
   * sprite.width = sprite.texture.orig.width;
   * ```
   */
  get width() {
    return Math.abs(this.scale.x) * this._texture.orig.width;
  }
  set width(value) {
    this._setWidth(value, this._texture.orig.width);
    this._width = value;
  }
  /**
   * The height of the sprite, setting this will actually modify the scale to achieve the value set.
   * @example
   * ```ts
   * // Set height directly
   * sprite.height = 150;
   * console.log(sprite.scale.y); // Scale adjusted to match height
   *
   * // Set height while preserving aspect ratio
   * const ratio = sprite.width / sprite.height;
   * sprite.height = 200;
   * sprite.width = 200 * ratio;
   *
   * // For better performance when setting both width and height
   * sprite.setSize(300, 400); // Avoids recalculating bounds twice
   *
   * // Reset to original texture size
   * sprite.height = sprite.texture.orig.height;
   * ```
   */
  get height() {
    return Math.abs(this.scale.y) * this._texture.orig.height;
  }
  set height(value) {
    this._setHeight(value, this._texture.orig.height);
    this._height = value;
  }
  /**
   * Retrieves the size of the Sprite as a [Size]{@link Size} object based on the texture dimensions and scale.
   * This is faster than getting width and height separately as it only calculates the bounds once.
   * @example
   * ```ts
   * // Basic size retrieval
   * const sprite = new Sprite(Texture.from('sprite.png'));
   * const size = sprite.getSize();
   * console.log(`Size: ${size.width}x${size.height}`);
   *
   * // Reuse existing size object
   * const reuseSize = { width: 0, height: 0 };
   * sprite.getSize(reuseSize);
   * ```
   * @param out - Optional object to store the size in, to avoid allocating a new object
   * @returns The size of the Sprite
   * @see {@link Sprite#width} For getting just the width
   * @see {@link Sprite#height} For getting just the height
   * @see {@link Sprite#setSize} For setting both width and height
   */
  getSize(out2) {
    out2 || (out2 = {});
    out2.width = Math.abs(this.scale.x) * this._texture.orig.width;
    out2.height = Math.abs(this.scale.y) * this._texture.orig.height;
    return out2;
  }
  /**
   * Sets the size of the Sprite to the specified width and height.
   * This is faster than setting width and height separately as it only recalculates bounds once.
   * @example
   * ```ts
   * // Basic size setting
   * const sprite = new Sprite(Texture.from('sprite.png'));
   * sprite.setSize(100, 200); // Width: 100, Height: 200
   *
   * // Set uniform size
   * sprite.setSize(100); // Sets both width and height to 100
   *
   * // Set size with object
   * sprite.setSize({
   *     width: 200,
   *     height: 300
   * });
   *
   * // Reset to texture size
   * sprite.setSize(
   *     sprite.texture.orig.width,
   *     sprite.texture.orig.height
   * );
   * ```
   * @param value - This can be either a number or a {@link Size} object
   * @param height - The height to set. Defaults to the value of `width` if not provided
   * @see {@link Sprite#width} For setting width only
   * @see {@link Sprite#height} For setting height only
   * @see {@link Sprite#texture} For the source dimensions
   */
  setSize(value, height) {
    if (typeof value === "object") {
      height = value.height ?? value.width;
      value = value.width;
    } else {
      height ?? (height = value);
    }
    value !== void 0 && this._setWidth(value, this._texture.orig.width);
    height !== void 0 && this._setHeight(height, this._texture.orig.height);
  }
}
const tempBounds = new Bounds();
function addMaskBounds(mask, bounds, skipUpdateTransform) {
  const boundsToMask = tempBounds;
  mask.measurable = true;
  getGlobalBounds(mask, skipUpdateTransform, boundsToMask);
  bounds.addBoundsMask(boundsToMask);
  mask.measurable = false;
}
function addMaskLocalBounds(mask, bounds, localRoot) {
  const boundsToMask = boundsPool.get();
  mask.measurable = true;
  const tempMatrix2 = matrixPool.get().identity();
  const relativeMask = getMatrixRelativeToParent(mask, localRoot, tempMatrix2);
  getLocalBounds(mask, boundsToMask, relativeMask);
  mask.measurable = false;
  bounds.addBoundsMask(boundsToMask);
  matrixPool.return(tempMatrix2);
  boundsPool.return(boundsToMask);
}
function getMatrixRelativeToParent(target, root, matrix) {
  if (!target) {
    warn("Mask bounds, renderable is not inside the root container");
    return matrix;
  }
  if (target !== root) {
    getMatrixRelativeToParent(target.parent, root, matrix);
    target.updateLocalTransform();
    matrix.append(target.localTransform);
  }
  return matrix;
}
class AlphaMask {
  constructor(options) {
    this.priority = 0;
    this.inverse = false;
    this.pipe = "alphaMask";
    if (options?.mask) {
      this.init(options.mask);
    }
  }
  init(mask) {
    this.mask = mask;
    this.renderMaskToTexture = !(mask instanceof Sprite);
    this.mask.renderable = this.renderMaskToTexture;
    this.mask.includeInBuild = !this.renderMaskToTexture;
    this.mask.measurable = false;
  }
  reset() {
    if (this.mask === null)
      return;
    this.mask.measurable = true;
    this.mask = null;
  }
  addBounds(bounds, skipUpdateTransform) {
    if (!this.inverse) {
      addMaskBounds(this.mask, bounds, skipUpdateTransform);
    }
  }
  addLocalBounds(bounds, localRoot) {
    addMaskLocalBounds(this.mask, bounds, localRoot);
  }
  containsPoint(point, hitTestFn) {
    const mask = this.mask;
    return hitTestFn(mask, point);
  }
  destroy() {
    this.reset();
  }
  static test(mask) {
    return mask instanceof Sprite;
  }
}
AlphaMask.extension = ExtensionType.MaskEffect;
class ColorMask {
  constructor(options) {
    this.priority = 0;
    this.pipe = "colorMask";
    if (options?.mask) {
      this.init(options.mask);
    }
  }
  init(mask) {
    this.mask = mask;
  }
  destroy() {
  }
  static test(mask) {
    return typeof mask === "number";
  }
}
ColorMask.extension = ExtensionType.MaskEffect;
class StencilMask {
  constructor(options) {
    this.priority = 0;
    this.pipe = "stencilMask";
    if (options?.mask) {
      this.init(options.mask);
    }
  }
  init(mask) {
    this.mask = mask;
    this.mask.includeInBuild = false;
    this.mask.measurable = false;
  }
  reset() {
    if (this.mask === null)
      return;
    this.mask.measurable = true;
    this.mask.includeInBuild = true;
    this.mask = null;
  }
  addBounds(bounds, skipUpdateTransform) {
    addMaskBounds(this.mask, bounds, skipUpdateTransform);
  }
  addLocalBounds(bounds, localRoot) {
    addMaskLocalBounds(this.mask, bounds, localRoot);
  }
  containsPoint(point, hitTestFn) {
    const mask = this.mask;
    return hitTestFn(mask, point);
  }
  destroy() {
    this.reset();
  }
  static test(mask) {
    return mask instanceof Container;
  }
}
StencilMask.extension = ExtensionType.MaskEffect;
const BrowserAdapter = {
  createCanvas: (width, height) => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  },
  createImage: () => new Image(),
  getCanvasRenderingContext2D: () => CanvasRenderingContext2D,
  getWebGLRenderingContext: () => WebGLRenderingContext,
  getNavigator: () => navigator,
  getBaseUrl: () => document.baseURI ?? window.location.href,
  getFontFaceSet: () => document.fonts,
  fetch: (url, options) => fetch(url, options),
  parseXML: (xml) => {
    const parser = new DOMParser();
    return parser.parseFromString(xml, "text/xml");
  }
};
let currentAdapter = BrowserAdapter;
const DOMAdapter = {
  /**
   * Returns the current adapter.
   * @returns {environment.Adapter} The current adapter.
   */
  get() {
    return currentAdapter;
  },
  /**
   * Sets the current adapter.
   * @param adapter - The new adapter.
   */
  set(adapter) {
    currentAdapter = adapter;
  }
};
class CanvasSource extends TextureSource {
  constructor(options) {
    if (!options.resource) {
      options.resource = DOMAdapter.get().createCanvas();
    }
    if (!options.width) {
      options.width = options.resource.width;
      if (!options.autoDensity) {
        options.width /= options.resolution;
      }
    }
    if (!options.height) {
      options.height = options.resource.height;
      if (!options.autoDensity) {
        options.height /= options.resolution;
      }
    }
    super(options);
    this.uploadMethodId = "image";
    this.autoDensity = options.autoDensity;
    this.resizeCanvas();
    this.transparent = !!options.transparent;
  }
  resizeCanvas() {
    if (this.autoDensity && "style" in this.resource) {
      this.resource.style.width = `${this.width}px`;
      this.resource.style.height = `${this.height}px`;
    }
    if (this.resource.width !== this.pixelWidth || this.resource.height !== this.pixelHeight) {
      this.resource.width = this.pixelWidth;
      this.resource.height = this.pixelHeight;
    }
  }
  resize(width = this.width, height = this.height, resolution = this._resolution) {
    const didResize = super.resize(width, height, resolution);
    if (didResize) {
      this.resizeCanvas();
    }
    return didResize;
  }
  static test(resource) {
    return globalThis.HTMLCanvasElement && resource instanceof HTMLCanvasElement || globalThis.OffscreenCanvas && resource instanceof OffscreenCanvas;
  }
  /**
   * Returns the 2D rendering context for the canvas.
   * Caches the context after creating it.
   * @returns The 2D rendering context of the canvas.
   */
  get context2D() {
    return this._context2D || (this._context2D = this.resource.getContext("2d"));
  }
}
CanvasSource.extension = ExtensionType.TextureSource;
class ImageSource extends TextureSource {
  constructor(options) {
    super(options);
    this.uploadMethodId = "image";
    this.autoGarbageCollect = true;
  }
  static test(resource) {
    return globalThis.HTMLImageElement && resource instanceof HTMLImageElement || typeof ImageBitmap !== "undefined" && resource instanceof ImageBitmap || globalThis.VideoFrame && resource instanceof VideoFrame;
  }
}
ImageSource.extension = ExtensionType.TextureSource;
var UPDATE_PRIORITY = /* @__PURE__ */ ((UPDATE_PRIORITY2) => {
  UPDATE_PRIORITY2[UPDATE_PRIORITY2["INTERACTION"] = 50] = "INTERACTION";
  UPDATE_PRIORITY2[UPDATE_PRIORITY2["HIGH"] = 25] = "HIGH";
  UPDATE_PRIORITY2[UPDATE_PRIORITY2["NORMAL"] = 0] = "NORMAL";
  UPDATE_PRIORITY2[UPDATE_PRIORITY2["LOW"] = -25] = "LOW";
  UPDATE_PRIORITY2[UPDATE_PRIORITY2["UTILITY"] = -50] = "UTILITY";
  return UPDATE_PRIORITY2;
})(UPDATE_PRIORITY || {});
class TickerListener {
  /**
   * Constructor
   * @private
   * @param fn - The listener function to be added for one update
   * @param context - The listener context
   * @param priority - The priority for emitting
   * @param once - If the handler should fire once
   */
  constructor(fn, context2 = null, priority = 0, once = false) {
    this.next = null;
    this.previous = null;
    this._destroyed = false;
    this._fn = fn;
    this._context = context2;
    this.priority = priority;
    this._once = once;
  }
  /**
   * Simple compare function to figure out if a function and context match.
   * @param fn - The listener function to be added for one update
   * @param context - The listener context
   * @returns `true` if the listener match the arguments
   */
  match(fn, context2 = null) {
    return this._fn === fn && this._context === context2;
  }
  /**
   * Emit by calling the current function.
   * @param ticker - The ticker emitting.
   * @returns Next ticker
   */
  emit(ticker) {
    if (this._fn) {
      if (this._context) {
        this._fn.call(this._context, ticker);
      } else {
        this._fn(ticker);
      }
    }
    const redirect = this.next;
    if (this._once) {
      this.destroy(true);
    }
    if (this._destroyed) {
      this.next = null;
    }
    return redirect;
  }
  /**
   * Connect to the list.
   * @param previous - Input node, previous listener
   */
  connect(previous) {
    this.previous = previous;
    if (previous.next) {
      previous.next.previous = this;
    }
    this.next = previous.next;
    previous.next = this;
  }
  /**
   * Destroy and don't use after this.
   * @param hard - `true` to remove the `next` reference, this
   *        is considered a hard destroy. Soft destroy maintains the next reference.
   * @returns The listener to redirect while emitting or removing.
   */
  destroy(hard = false) {
    this._destroyed = true;
    this._fn = null;
    this._context = null;
    if (this.previous) {
      this.previous.next = this.next;
    }
    if (this.next) {
      this.next.previous = this.previous;
    }
    const redirect = this.next;
    this.next = hard ? null : redirect;
    this.previous = null;
    return redirect;
  }
}
const _Ticker = class _Ticker2 {
  constructor() {
    this.autoStart = false;
    this.deltaTime = 1;
    this.lastTime = -1;
    this.speed = 1;
    this.started = false;
    this._requestId = null;
    this._maxElapsedMS = 100;
    this._minElapsedMS = 0;
    this._protected = false;
    this._lastFrame = -1;
    this._head = new TickerListener(null, null, Infinity);
    this.deltaMS = 1 / _Ticker2.targetFPMS;
    this.elapsedMS = 1 / _Ticker2.targetFPMS;
    this._tick = (time) => {
      this._requestId = null;
      if (this.started) {
        this.update(time);
        if (this.started && this._requestId === null && this._head.next) {
          this._requestId = requestAnimationFrame(this._tick);
        }
      }
    };
  }
  /**
   * Conditionally requests a new animation frame.
   * If a frame has not already been requested, and if the internal
   * emitter has listeners, a new frame is requested.
   */
  _requestIfNeeded() {
    if (this._requestId === null && this._head.next) {
      this.lastTime = performance.now();
      this._lastFrame = this.lastTime;
      this._requestId = requestAnimationFrame(this._tick);
    }
  }
  /** Conditionally cancels a pending animation frame. */
  _cancelIfNeeded() {
    if (this._requestId !== null) {
      cancelAnimationFrame(this._requestId);
      this._requestId = null;
    }
  }
  /**
   * Conditionally requests a new animation frame.
   * If the ticker has been started it checks if a frame has not already
   * been requested, and if the internal emitter has listeners. If these
   * conditions are met, a new frame is requested. If the ticker has not
   * been started, but autoStart is `true`, then the ticker starts now,
   * and continues with the previous conditions to request a new frame.
   */
  _startIfPossible() {
    if (this.started) {
      this._requestIfNeeded();
    } else if (this.autoStart) {
      this.start();
    }
  }
  /**
   * Register a handler for tick events.
   * @param fn - The listener function to add. Receives the Ticker instance as parameter
   * @param context - The context for the listener
   * @param priority - The priority of the listener
   * @example
   * ```ts
   * // Access time properties through the ticker parameter
   * ticker.add((ticker) => {
   *     // Use deltaTime (dimensionless scalar) for frame-independent animations
   *     sprite.rotation += 0.1 * ticker.deltaTime;
   *
   *     // Use deltaMS (milliseconds) for time-based calculations
   *     const progress = ticker.deltaMS / animationDuration;
   *
   *     // Use elapsedMS for raw timing measurements
   *     console.log(`Raw frame time: ${ticker.elapsedMS}ms`);
   * });
   * ```
   */
  add(fn, context2, priority = UPDATE_PRIORITY.NORMAL) {
    return this._addListener(new TickerListener(fn, context2, priority));
  }
  /**
   * Add a handler for the tick event which is only executed once on the next frame.
   * @example
   * ```ts
   * // Basic one-time update
   * ticker.addOnce(() => {
   *     console.log('Runs next frame only');
   * });
   *
   * // With specific context
   * const game = {
   *     init(ticker) {
   *         this.loadResources();
   *         console.log('Game initialized');
   *     }
   * };
   * ticker.addOnce(game.init, game);
   *
   * // With priority
   * ticker.addOnce(
   *     () => {
   *         // High priority one-time setup
   *         physics.init();
   *     },
   *     undefined,
   *     UPDATE_PRIORITY.HIGH
   * );
   * ```
   * @param fn - The listener function to be added for one update
   * @param context - The listener context
   * @param priority - The priority for emitting (default: UPDATE_PRIORITY.NORMAL)
   * @returns This instance of a ticker
   * @see {@link Ticker#add} For continuous updates
   * @see {@link Ticker#remove} For removing handlers
   */
  addOnce(fn, context2, priority = UPDATE_PRIORITY.NORMAL) {
    return this._addListener(new TickerListener(fn, context2, priority, true));
  }
  /**
   * Internally adds the event handler so that it can be sorted by priority.
   * Priority allows certain handler (user, AnimatedSprite, Interaction) to be run
   * before the rendering.
   * @private
   * @param listener - Current listener being added.
   * @returns This instance of a ticker
   */
  _addListener(listener) {
    let current = this._head.next;
    let previous = this._head;
    if (!current) {
      listener.connect(previous);
    } else {
      while (current) {
        if (listener.priority > current.priority) {
          listener.connect(previous);
          break;
        }
        previous = current;
        current = current.next;
      }
      if (!listener.previous) {
        listener.connect(previous);
      }
    }
    this._startIfPossible();
    return this;
  }
  /**
   * Removes any handlers matching the function and context parameters.
   * If no handlers are left after removing, then it cancels the animation frame.
   * @example
   * ```ts
   * // Basic removal
   * const onTick = () => {
   *     sprite.rotation += 0.1;
   * };
   * ticker.add(onTick);
   * ticker.remove(onTick);
   *
   * // Remove with context
   * const game = {
   *     update(ticker) {
   *         this.physics.update(ticker.deltaTime);
   *     }
   * };
   * ticker.add(game.update, game);
   * ticker.remove(game.update, game);
   *
   * // Remove all matching handlers
   * // (if same function was added multiple times)
   * ticker.add(onTick);
   * ticker.add(onTick);
   * ticker.remove(onTick); // Removes all instances
   * ```
   * @param fn - The listener function to be removed
   * @param context - The listener context to be removed
   * @returns This instance of a ticker
   * @see {@link Ticker#add} For adding handlers
   * @see {@link Ticker#addOnce} For one-time handlers
   */
  remove(fn, context2) {
    let listener = this._head.next;
    while (listener) {
      if (listener.match(fn, context2)) {
        listener = listener.destroy();
      } else {
        listener = listener.next;
      }
    }
    if (!this._head.next) {
      this._cancelIfNeeded();
    }
    return this;
  }
  /**
   * The number of listeners on this ticker, calculated by walking through linked list.
   * @example
   * ```ts
   * // Check number of active listeners
   * const ticker = new Ticker();
   * console.log(ticker.count); // 0
   *
   * // Add some listeners
   * ticker.add(() => {});
   * ticker.add(() => {});
   * console.log(ticker.count); // 2
   *
   * // Check after cleanup
   * ticker.destroy();
   * console.log(ticker.count); // 0
   * ```
   * @readonly
   * @see {@link Ticker#add} For adding listeners
   * @see {@link Ticker#remove} For removing listeners
   */
  get count() {
    if (!this._head) {
      return 0;
    }
    let count2 = 0;
    let current = this._head;
    while (current = current.next) {
      count2++;
    }
    return count2;
  }
  /**
   * Starts the ticker. If the ticker has listeners a new animation frame is requested at this point.
   * @example
   * ```ts
   * // Basic manual start
   * const ticker = new Ticker();
   * ticker.add(() => {
   *     // Animation code here
   * });
   * ticker.start();
   * ```
   * @see {@link Ticker#stop} For stopping the ticker
   * @see {@link Ticker#autoStart} For automatic starting
   * @see {@link Ticker#started} For checking ticker state
   */
  start() {
    if (!this.started) {
      this.started = true;
      this._requestIfNeeded();
    }
  }
  /**
   * Stops the ticker. If the ticker has requested an animation frame it is canceled at this point.
   * @example
   * ```ts
   * // Basic stop
   * const ticker = new Ticker();
   * ticker.stop();
   * ```
   * @see {@link Ticker#start} For starting the ticker
   * @see {@link Ticker#started} For checking ticker state
   * @see {@link Ticker#destroy} For cleaning up the ticker
   */
  stop() {
    if (this.started) {
      this.started = false;
      this._cancelIfNeeded();
    }
  }
  /**
   * Destroy the ticker and don't use after this. Calling this method removes all references to internal events.
   * @example
   * ```ts
   * // Clean up with active listeners
   * const ticker = new Ticker();
   * ticker.add(() => {});
   * ticker.destroy(); // Removes all listeners
   * ```
   * @see {@link Ticker#stop} For stopping without destroying
   * @see {@link Ticker#remove} For removing specific listeners
   */
  destroy() {
    if (!this._protected) {
      this.stop();
      let listener = this._head.next;
      while (listener) {
        listener = listener.destroy(true);
      }
      this._head.destroy();
      this._head = null;
    }
  }
  /**
   * Triggers an update.
   *
   * An update entails setting the
   * current {@link Ticker#elapsedMS|elapsedMS},
   * the current {@link Ticker#deltaTime|deltaTime},
   * invoking all listeners with current deltaTime,
   * and then finally setting {@link Ticker#lastTime|lastTime}
   * with the value of currentTime that was provided.
   *
   * This method will be called automatically by animation
   * frame callbacks if the ticker instance has been started
   * and listeners are added.
   * @example
   * ```ts
   * // Basic manual update
   * const ticker = new Ticker();
   * ticker.update(performance.now());
   * ```
   * @param currentTime - The current time of execution (defaults to performance.now())
   * @see {@link Ticker#deltaTime} For frame delta value
   * @see {@link Ticker#elapsedMS} For raw elapsed time
   */
  update(currentTime = performance.now()) {
    let elapsedMS;
    if (currentTime > this.lastTime) {
      elapsedMS = this.elapsedMS = currentTime - this.lastTime;
      if (elapsedMS > this._maxElapsedMS) {
        elapsedMS = this._maxElapsedMS;
      }
      elapsedMS *= this.speed;
      if (this._minElapsedMS) {
        const delta = currentTime - this._lastFrame | 0;
        if (delta < this._minElapsedMS) {
          return;
        }
        this._lastFrame = currentTime - delta % this._minElapsedMS;
      }
      this.deltaMS = elapsedMS;
      this.deltaTime = this.deltaMS * _Ticker2.targetFPMS;
      const head = this._head;
      let listener = head.next;
      while (listener) {
        listener = listener.emit(this);
      }
      if (!head.next) {
        this._cancelIfNeeded();
      }
    } else {
      this.deltaTime = this.deltaMS = this.elapsedMS = 0;
    }
    this.lastTime = currentTime;
  }
  /**
   * The frames per second at which this ticker is running.
   * The default is approximately 60 in most modern browsers.
   * > [!NOTE] This does not factor in the value of
   * > {@link Ticker#speed|speed}, which is specific
   * > to scaling {@link Ticker#deltaTime|deltaTime}.
   * @example
   * ```ts
   * // Basic FPS monitoring
   * ticker.add(() => {
   *     console.log(`Current FPS: ${Math.round(ticker.FPS)}`);
   * });
   * ```
   * @readonly
   */
  get FPS() {
    return 1e3 / this.elapsedMS;
  }
  /**
   * Manages the maximum amount of milliseconds allowed to
   * elapse between invoking {@link Ticker#update|update}.
   *
   * This value is used to cap {@link Ticker#deltaTime|deltaTime},
   * but does not effect the measured value of {@link Ticker#FPS|FPS}.
   *
   * When setting this property it is clamped to a value between
   * `0` and `Ticker.targetFPMS * 1000`.
   * @example
   * ```ts
   * // Set minimum acceptable frame rate
   * const ticker = new Ticker();
   * ticker.minFPS = 30; // Never go below 30 FPS
   *
   * // Use with maxFPS for frame rate clamping
   * ticker.minFPS = 30;
   * ticker.maxFPS = 60;
   *
   * // Monitor delta capping
   * ticker.add(() => {
   *     // Delta time will be capped based on minFPS
   *     console.log(`Delta time: ${ticker.deltaTime}`);
   * });
   * ```
   * @default 10
   */
  get minFPS() {
    return 1e3 / this._maxElapsedMS;
  }
  set minFPS(fps) {
    const minFPS = Math.min(this.maxFPS, fps);
    const minFPMS = Math.min(Math.max(0, minFPS) / 1e3, _Ticker2.targetFPMS);
    this._maxElapsedMS = 1 / minFPMS;
  }
  /**
   * Manages the minimum amount of milliseconds required to
   * elapse between invoking {@link Ticker#update|update}.
   *
   * This will effect the measured value of {@link Ticker#FPS|FPS}.
   *
   * If it is set to `0`, then there is no limit; PixiJS will render as many frames as it can.
   * Otherwise it will be at least `minFPS`
   * @example
   * ```ts
   * // Set minimum acceptable frame rate
   * const ticker = new Ticker();
   * ticker.maxFPS = 60; // Never go above 60 FPS
   *
   * // Use with maxFPS for frame rate clamping
   * ticker.minFPS = 30;
   * ticker.maxFPS = 60;
   *
   * // Monitor delta capping
   * ticker.add(() => {
   *     // Delta time will be capped based on maxFPS
   *     console.log(`Delta time: ${ticker.deltaTime}`);
   * });
   * ```
   * @default 0
   */
  get maxFPS() {
    if (this._minElapsedMS) {
      return Math.round(1e3 / this._minElapsedMS);
    }
    return 0;
  }
  set maxFPS(fps) {
    if (fps === 0) {
      this._minElapsedMS = 0;
    } else {
      const maxFPS = Math.max(this.minFPS, fps);
      this._minElapsedMS = 1 / (maxFPS / 1e3);
    }
  }
  /**
   * The shared ticker instance used by {@link AnimatedSprite} and by
   * {@link VideoSource} to update animation frames / video textures.
   *
   * It may also be used by {@link Application} if created with the `sharedTicker` option property set to true.
   *
   * The property {@link Ticker#autoStart|autoStart} is set to `true` for this instance.
   * Please follow the examples for usage, including how to opt-out of auto-starting the shared ticker.
   * @example
   * import { Ticker } from 'pixi.js';
   *
   * const ticker = Ticker.shared;
   * // Set this to prevent starting this ticker when listeners are added.
   * // By default this is true only for the Ticker.shared instance.
   * ticker.autoStart = false;
   *
   * // FYI, call this to ensure the ticker is stopped. It should be stopped
   * // if you have not attempted to render anything yet.
   * ticker.stop();
   *
   * // Call this when you are ready for a running shared ticker.
   * ticker.start();
   * @example
   * import { autoDetectRenderer, Container } from 'pixi.js';
   *
   * // You may use the shared ticker to render...
   * const renderer = autoDetectRenderer();
   * const stage = new Container();
   * document.body.appendChild(renderer.view);
   * ticker.add((time) => renderer.render(stage));
   *
   * // Or you can just update it manually.
   * ticker.autoStart = false;
   * ticker.stop();
   * const animate = (time) => {
   *     ticker.update(time);
   *     renderer.render(stage);
   *     requestAnimationFrame(animate);
   * };
   * animate(performance.now());
   * @type {Ticker}
   * @readonly
   */
  static get shared() {
    if (!_Ticker2._shared) {
      const shared = _Ticker2._shared = new _Ticker2();
      shared.autoStart = true;
      shared._protected = true;
    }
    return _Ticker2._shared;
  }
  /**
   * The system ticker instance used by {@link PrepareBase} for core timing
   * functionality that shouldn't usually need to be paused, unlike the `shared`
   * ticker which drives visual animations and rendering which may want to be paused.
   *
   * The property {@link Ticker#autoStart|autoStart} is set to `true` for this instance.
   * @type {Ticker}
   * @readonly
   * @advanced
   */
  static get system() {
    if (!_Ticker2._system) {
      const system = _Ticker2._system = new _Ticker2();
      system.autoStart = true;
      system._protected = true;
    }
    return _Ticker2._system;
  }
};
_Ticker.targetFPMS = 0.06;
let Ticker = _Ticker;
let promise;
async function detectVideoAlphaMode() {
  promise ?? (promise = (async () => {
    const canvas = DOMAdapter.get().createCanvas(1, 1);
    const gl = canvas.getContext("webgl");
    if (!gl) {
      return "premultiply-alpha-on-upload";
    }
    const video = await new Promise((resolve) => {
      const video2 = document.createElement("video");
      video2.onloadeddata = () => resolve(video2);
      video2.onerror = () => resolve(null);
      video2.autoplay = false;
      video2.crossOrigin = "anonymous";
      video2.preload = "auto";
      video2.src = "data:video/webm;base64,GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAHTEU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHGTbuMU6uEElTDZ1OsggEXTbuMU6uEHFO7a1OsggG97AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmoCrXsYMPQkBNgIRMYXZmV0GETGF2ZkSJiEBEAAAAAAAAFlSua8yuAQAAAAAAAEPXgQFzxYgAAAAAAAAAAZyBACK1nIN1bmSIgQCGhVZfVlA5g4EBI+ODhAJiWgDglLCBArqBApqBAlPAgQFVsIRVuYEBElTDZ9Vzc9JjwItjxYgAAAAAAAAAAWfInEWjh0VOQ09ERVJEh49MYXZjIGxpYnZweC12cDlnyKJFo4hEVVJBVElPTkSHlDAwOjAwOjAwLjA0MDAwMDAwMAAAH0O2dcfngQCgwqGggQAAAIJJg0IAABAAFgA4JBwYSgAAICAAEb///4r+AAB1oZ2mm+6BAaWWgkmDQgAAEAAWADgkHBhKAAAgIABIQBxTu2uRu4+zgQC3iveBAfGCAXHwgQM=";
      video2.load();
    });
    if (!video) {
      return "premultiply-alpha-on-upload";
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    const pixel = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    gl.deleteFramebuffer(framebuffer);
    gl.deleteTexture(texture);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return pixel[0] <= pixel[3] ? "premultiplied-alpha" : "premultiply-alpha-on-upload";
  })());
  return promise;
}
const _VideoSource = class _VideoSource2 extends TextureSource {
  constructor(options) {
    super(options);
    this.isReady = false;
    this.uploadMethodId = "video";
    options = {
      ..._VideoSource2.defaultOptions,
      ...options
    };
    this._autoUpdate = true;
    this._isConnectedToTicker = false;
    this._updateFPS = options.updateFPS || 0;
    this._msToNextUpdate = 0;
    this.autoPlay = options.autoPlay !== false;
    this.alphaMode = options.alphaMode ?? "premultiply-alpha-on-upload";
    this._videoFrameRequestCallback = this._videoFrameRequestCallback.bind(this);
    this._videoFrameRequestCallbackHandle = null;
    this._load = null;
    this._resolve = null;
    this._reject = null;
    this._onCanPlay = this._onCanPlay.bind(this);
    this._onCanPlayThrough = this._onCanPlayThrough.bind(this);
    this._onError = this._onError.bind(this);
    this._onPlayStart = this._onPlayStart.bind(this);
    this._onPlayStop = this._onPlayStop.bind(this);
    this._onSeeked = this._onSeeked.bind(this);
    if (options.autoLoad !== false) {
      void this.load();
    }
  }
  /** Update the video frame if the source is not destroyed and meets certain conditions. */
  updateFrame() {
    if (this.destroyed) {
      return;
    }
    if (this._updateFPS) {
      const elapsedMS = Ticker.shared.elapsedMS * this.resource.playbackRate;
      this._msToNextUpdate = Math.floor(this._msToNextUpdate - elapsedMS);
    }
    if (!this._updateFPS || this._msToNextUpdate <= 0) {
      this._msToNextUpdate = this._updateFPS ? Math.floor(1e3 / this._updateFPS) : 0;
    }
    if (this.isValid) {
      this.update();
    }
  }
  /** Callback to update the video frame and potentially request the next frame update. */
  _videoFrameRequestCallback() {
    this.updateFrame();
    if (this.destroyed) {
      this._videoFrameRequestCallbackHandle = null;
    } else {
      this._videoFrameRequestCallbackHandle = this.resource.requestVideoFrameCallback(
        this._videoFrameRequestCallback
      );
    }
  }
  /**
   * Checks if the resource has valid dimensions.
   * @returns {boolean} True if width and height are set, otherwise false.
   */
  get isValid() {
    return !!this.resource.videoWidth && !!this.resource.videoHeight;
  }
  /**
   * Start preloading the video resource.
   * @returns {Promise<this>} Handle the validate event
   */
  async load() {
    if (this._load) {
      return this._load;
    }
    const source2 = this.resource;
    const options = this.options;
    if ((source2.readyState === source2.HAVE_ENOUGH_DATA || source2.readyState === source2.HAVE_FUTURE_DATA) && source2.width && source2.height) {
      source2.complete = true;
    }
    source2.addEventListener("play", this._onPlayStart);
    source2.addEventListener("pause", this._onPlayStop);
    source2.addEventListener("seeked", this._onSeeked);
    if (!this._isSourceReady()) {
      if (!options.preload) {
        source2.addEventListener("canplay", this._onCanPlay);
      }
      source2.addEventListener("canplaythrough", this._onCanPlayThrough);
      source2.addEventListener("error", this._onError, true);
    } else {
      this._mediaReady();
    }
    this.alphaMode = await detectVideoAlphaMode();
    this._load = new Promise((resolve, reject) => {
      if (this.isValid) {
        resolve(this);
      } else {
        this._resolve = resolve;
        this._reject = reject;
        if (options.preloadTimeoutMs !== void 0) {
          this._preloadTimeout = setTimeout(() => {
            this._onError(new ErrorEvent(`Preload exceeded timeout of ${options.preloadTimeoutMs}ms`));
          });
        }
        source2.load();
      }
    });
    return this._load;
  }
  /**
   * Handle video error events.
   * @param event - The error event
   */
  _onError(event) {
    this.resource.removeEventListener("error", this._onError, true);
    this.emit("error", event);
    if (this._reject) {
      this._reject(event);
      this._reject = null;
      this._resolve = null;
    }
  }
  /**
   * Checks if the underlying source is playing.
   * @returns True if playing.
   */
  _isSourcePlaying() {
    const source2 = this.resource;
    return !source2.paused && !source2.ended;
  }
  /**
   * Checks if the underlying source is ready for playing.
   * @returns True if ready.
   */
  _isSourceReady() {
    const source2 = this.resource;
    return source2.readyState > 2;
  }
  /** Runs the update loop when the video is ready to play. */
  _onPlayStart() {
    if (!this.isValid) {
      this._mediaReady();
    }
    this._configureAutoUpdate();
  }
  /** Stops the update loop when a pause event is triggered. */
  _onPlayStop() {
    this._configureAutoUpdate();
  }
  /** Handles behavior when the video completes seeking to the current playback position. */
  _onSeeked() {
    if (this._autoUpdate && !this._isSourcePlaying()) {
      this._msToNextUpdate = 0;
      this.updateFrame();
      this._msToNextUpdate = 0;
    }
  }
  _onCanPlay() {
    const source2 = this.resource;
    source2.removeEventListener("canplay", this._onCanPlay);
    this._mediaReady();
  }
  _onCanPlayThrough() {
    const source2 = this.resource;
    source2.removeEventListener("canplaythrough", this._onCanPlay);
    if (this._preloadTimeout) {
      clearTimeout(this._preloadTimeout);
      this._preloadTimeout = void 0;
    }
    this._mediaReady();
  }
  /** Fired when the video is loaded and ready to play. */
  _mediaReady() {
    const source2 = this.resource;
    if (this.isValid) {
      this.isReady = true;
      this.resize(source2.videoWidth, source2.videoHeight);
    }
    this._msToNextUpdate = 0;
    this.updateFrame();
    this._msToNextUpdate = 0;
    if (this._resolve) {
      this._resolve(this);
      this._resolve = null;
      this._reject = null;
    }
    if (this._isSourcePlaying()) {
      this._onPlayStart();
    } else if (this.autoPlay) {
      void this.resource.play();
    }
  }
  /** Cleans up resources and event listeners associated with this texture. */
  destroy() {
    this._configureAutoUpdate();
    const source2 = this.resource;
    if (source2) {
      source2.removeEventListener("play", this._onPlayStart);
      source2.removeEventListener("pause", this._onPlayStop);
      source2.removeEventListener("seeked", this._onSeeked);
      source2.removeEventListener("canplay", this._onCanPlay);
      source2.removeEventListener("canplaythrough", this._onCanPlayThrough);
      source2.removeEventListener("error", this._onError, true);
      source2.pause();
      source2.src = "";
      source2.load();
    }
    super.destroy();
  }
  /** Should the base texture automatically update itself, set to true by default. */
  get autoUpdate() {
    return this._autoUpdate;
  }
  set autoUpdate(value) {
    if (value !== this._autoUpdate) {
      this._autoUpdate = value;
      this._configureAutoUpdate();
    }
  }
  /**
   * How many times a second to update the texture from the video.
   * Leave at 0 to update at every render.
   * A lower fps can help performance, as updating the texture at 60fps on a 30ps video may not be efficient.
   */
  get updateFPS() {
    return this._updateFPS;
  }
  set updateFPS(value) {
    if (value !== this._updateFPS) {
      this._updateFPS = value;
      this._configureAutoUpdate();
    }
  }
  /**
   * Configures the updating mechanism based on the current state and settings.
   *
   * This method decides between using the browser's native video frame callback or a custom ticker
   * for updating the video frame. It ensures optimal performance and responsiveness
   * based on the video's state, playback status, and the desired frames-per-second setting.
   *
   * - If `_autoUpdate` is enabled and the video source is playing:
   *   - It will prefer the native video frame callback if available and no specific FPS is set.
   *   - Otherwise, it will use a custom ticker for manual updates.
   * - If `_autoUpdate` is disabled or the video isn't playing, any active update mechanisms are halted.
   */
  _configureAutoUpdate() {
    if (this._autoUpdate && this._isSourcePlaying()) {
      if (!this._updateFPS && this.resource.requestVideoFrameCallback) {
        if (this._isConnectedToTicker) {
          Ticker.shared.remove(this.updateFrame, this);
          this._isConnectedToTicker = false;
          this._msToNextUpdate = 0;
        }
        if (this._videoFrameRequestCallbackHandle === null) {
          this._videoFrameRequestCallbackHandle = this.resource.requestVideoFrameCallback(
            this._videoFrameRequestCallback
          );
        }
      } else {
        if (this._videoFrameRequestCallbackHandle !== null) {
          this.resource.cancelVideoFrameCallback(this._videoFrameRequestCallbackHandle);
          this._videoFrameRequestCallbackHandle = null;
        }
        if (!this._isConnectedToTicker) {
          Ticker.shared.add(this.updateFrame, this);
          this._isConnectedToTicker = true;
          this._msToNextUpdate = 0;
        }
      }
    } else {
      if (this._videoFrameRequestCallbackHandle !== null) {
        this.resource.cancelVideoFrameCallback(this._videoFrameRequestCallbackHandle);
        this._videoFrameRequestCallbackHandle = null;
      }
      if (this._isConnectedToTicker) {
        Ticker.shared.remove(this.updateFrame, this);
        this._isConnectedToTicker = false;
        this._msToNextUpdate = 0;
      }
    }
  }
  static test(resource) {
    return globalThis.HTMLVideoElement && resource instanceof HTMLVideoElement;
  }
};
_VideoSource.extension = ExtensionType.TextureSource;
_VideoSource.defaultOptions = {
  ...TextureSource.defaultOptions,
  /** If true, the video will start loading immediately. */
  autoLoad: true,
  /** If true, the video will start playing as soon as it is loaded. */
  autoPlay: true,
  /** The number of times a second to update the texture from the video. Leave at 0 to update at every render. */
  updateFPS: 0,
  /** If true, the video will be loaded with the `crossorigin` attribute. */
  crossorigin: true,
  /** If true, the video will loop when it ends. */
  loop: false,
  /** If true, the video will be muted. */
  muted: true,
  /** If true, the video will play inline. */
  playsinline: true,
  /** If true, the video will be preloaded. */
  preload: false
};
_VideoSource.MIME_TYPES = {
  ogv: "video/ogg",
  mov: "video/quicktime",
  m4v: "video/mp4"
};
let VideoSource = _VideoSource;
const convertToList = (input, transform, forceTransform = false) => {
  if (!Array.isArray(input)) {
    input = [input];
  }
  if (!transform) {
    return input;
  }
  return input.map((item) => {
    if (typeof item === "string" || forceTransform) {
      return transform(item);
    }
    return item;
  });
};
class CacheClass {
  constructor() {
    this._parsers = [];
    this._cache = /* @__PURE__ */ new Map();
    this._cacheMap = /* @__PURE__ */ new Map();
  }
  /** Clear all entries. */
  reset() {
    this._cacheMap.clear();
    this._cache.clear();
  }
  /**
   * Check if the key exists
   * @param key - The key to check
   */
  has(key) {
    return this._cache.has(key);
  }
  /**
   * Fetch entry by key
   * @param key - The key of the entry to get
   */
  get(key) {
    const result = this._cache.get(key);
    if (!result) {
      warn(`[Assets] Asset id ${key} was not found in the Cache`);
    }
    return result;
  }
  /**
   * Set a value by key or keys name
   * @param key - The key or keys to set
   * @param value - The value to store in the cache or from which cacheable assets will be derived.
   */
  set(key, value) {
    const keys = convertToList(key);
    let cacheableAssets;
    for (let i2 = 0; i2 < this.parsers.length; i2++) {
      const parser = this.parsers[i2];
      if (parser.test(value)) {
        cacheableAssets = parser.getCacheableAssets(keys, value);
        break;
      }
    }
    const cacheableMap = new Map(Object.entries(cacheableAssets || {}));
    if (!cacheableAssets) {
      keys.forEach((key2) => {
        cacheableMap.set(key2, value);
      });
    }
    const cacheKeys = [...cacheableMap.keys()];
    const cachedAssets = {
      cacheKeys,
      keys
    };
    keys.forEach((key2) => {
      this._cacheMap.set(key2, cachedAssets);
    });
    cacheKeys.forEach((key2) => {
      const val = cacheableAssets ? cacheableAssets[key2] : value;
      if (this._cache.has(key2) && this._cache.get(key2) !== val) {
        warn("[Cache] already has key:", key2);
      }
      this._cache.set(key2, cacheableMap.get(key2));
    });
  }
  /**
   * Remove entry by key
   *
   * This function will also remove any associated alias from the cache also.
   * @param key - The key of the entry to remove
   */
  remove(key) {
    if (!this._cacheMap.has(key)) {
      warn(`[Assets] Asset id ${key} was not found in the Cache`);
      return;
    }
    const cacheMap2 = this._cacheMap.get(key);
    const cacheKeys = cacheMap2.cacheKeys;
    cacheKeys.forEach((key2) => {
      this._cache.delete(key2);
    });
    cacheMap2.keys.forEach((key2) => {
      this._cacheMap.delete(key2);
    });
  }
  /**
   * All loader parsers registered
   * @advanced
   */
  get parsers() {
    return this._parsers;
  }
}
const Cache = new CacheClass();
const sources = [];
extensions.handleByList(ExtensionType.TextureSource, sources);
function textureSourceFrom(options = {}) {
  const hasResource = options && options.resource;
  const res = hasResource ? options.resource : options;
  const opts = hasResource ? options : { resource: options };
  for (let i2 = 0; i2 < sources.length; i2++) {
    const Source = sources[i2];
    if (Source.test(res)) {
      return new Source(opts);
    }
  }
  throw new Error(`Could not find a source type for resource: ${opts.resource}`);
}
function resourceToTexture(options = {}, skipCache = false) {
  const hasResource = options && options.resource;
  const resource = hasResource ? options.resource : options;
  const opts = hasResource ? options : { resource: options };
  if (!skipCache && Cache.has(resource)) {
    return Cache.get(resource);
  }
  const texture = new Texture({ source: textureSourceFrom(opts) });
  texture.on("destroy", () => {
    if (Cache.has(resource)) {
      Cache.remove(resource);
    }
  });
  if (!skipCache) {
    Cache.set(resource, texture);
  }
  return texture;
}
function textureFrom(id, skipCache = false) {
  if (typeof id === "string") {
    return Cache.get(id);
  } else if (id instanceof TextureSource) {
    return new Texture({ source: id });
  }
  return resourceToTexture(id, skipCache);
}
Texture.from = textureFrom;
TextureSource.from = textureSourceFrom;
extensions.add(AlphaMask, ColorMask, StencilMask, VideoSource, ImageSource, CanvasSource, BufferImageSource);
var LoaderParserPriority = /* @__PURE__ */ ((LoaderParserPriority2) => {
  LoaderParserPriority2[LoaderParserPriority2["Low"] = 0] = "Low";
  LoaderParserPriority2[LoaderParserPriority2["Normal"] = 1] = "Normal";
  LoaderParserPriority2[LoaderParserPriority2["High"] = 2] = "High";
  return LoaderParserPriority2;
})(LoaderParserPriority || {});
function assertPath(path2) {
  if (typeof path2 !== "string") {
    throw new TypeError(`Path must be a string. Received ${JSON.stringify(path2)}`);
  }
}
function removeUrlParams(url) {
  const re = url.split("?")[0];
  return re.split("#")[0];
}
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function replaceAll(str, find, replace) {
  return str.replace(new RegExp(escapeRegExp(find), "g"), replace);
}
function normalizeStringPosix(path2, allowAboveRoot) {
  let res = "";
  let lastSegmentLength = 0;
  let lastSlash = -1;
  let dots = 0;
  let code = -1;
  for (let i2 = 0; i2 <= path2.length; ++i2) {
    if (i2 < path2.length) {
      code = path2.charCodeAt(i2);
    } else if (code === 47) {
      break;
    } else {
      code = 47;
    }
    if (code === 47) {
      if (lastSlash === i2 - 1 || dots === 1) ;
      else if (lastSlash !== i2 - 1 && dots === 2) {
        if (res.length < 2 || lastSegmentLength !== 2 || res.charCodeAt(res.length - 1) !== 46 || res.charCodeAt(res.length - 2) !== 46) {
          if (res.length > 2) {
            const lastSlashIndex = res.lastIndexOf("/");
            if (lastSlashIndex !== res.length - 1) {
              if (lastSlashIndex === -1) {
                res = "";
                lastSegmentLength = 0;
              } else {
                res = res.slice(0, lastSlashIndex);
                lastSegmentLength = res.length - 1 - res.lastIndexOf("/");
              }
              lastSlash = i2;
              dots = 0;
              continue;
            }
          } else if (res.length === 2 || res.length === 1) {
            res = "";
            lastSegmentLength = 0;
            lastSlash = i2;
            dots = 0;
            continue;
          }
        }
      } else {
        if (res.length > 0) {
          res += `/${path2.slice(lastSlash + 1, i2)}`;
        } else {
          res = path2.slice(lastSlash + 1, i2);
        }
        lastSegmentLength = i2 - lastSlash - 1;
      }
      lastSlash = i2;
      dots = 0;
    } else if (code === 46 && dots !== -1) {
      ++dots;
    } else {
      dots = -1;
    }
  }
  return res;
}
const path = {
  /**
   * Converts a path to posix format.
   * @param path - The path to convert to posix
   * @example
   * ```ts
   * // Convert a Windows path to POSIX format
   * path.toPosix('C:\\Users\\User\\Documents\\file.txt');
   * // -> 'C:/Users/User/Documents/file.txt'
   * ```
   */
  toPosix(path2) {
    return replaceAll(path2, "\\", "/");
  },
  /**
   * Checks if the path is a URL e.g. http://, https://
   * @param path - The path to check
   * @example
   * ```ts
   * // Check if a path is a URL
   * path.isUrl('http://www.example.com');
   * // -> true
   * path.isUrl('C:/Users/User/Documents/file.txt');
   * // -> false
   * ```
   */
  isUrl(path2) {
    return /^https?:/.test(this.toPosix(path2));
  },
  /**
   * Checks if the path is a data URL
   * @param path - The path to check
   * @example
   * ```ts
   * // Check if a path is a data URL
   * path.isDataUrl('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...');
   * // -> true
   * ```
   */
  isDataUrl(path2) {
    return /^data:([a-z]+\/[a-z0-9-+.]+(;[a-z0-9-.!#$%*+.{}|~`]+=[a-z0-9-.!#$%*+.{}()_|~`]+)*)?(;base64)?,([a-z0-9!$&',()*+;=\-._~:@\/?%\s<>]*?)$/i.test(path2);
  },
  /**
   * Checks if the path is a blob URL
   * @param path - The path to check
   * @example
   * ```ts
   * // Check if a path is a blob URL
   * path.isBlobUrl('blob:http://www.example.com/12345678-1234-1234-1234-123456789012');
   * // -> true
   * ```
   */
  isBlobUrl(path2) {
    return path2.startsWith("blob:");
  },
  /**
   * Checks if the path has a protocol e.g. http://, https://, file:///, data:, blob:, C:/
   * This will return true for windows file paths
   * @param path - The path to check
   * @example
   * ```ts
   * // Check if a path has a protocol
   * path.hasProtocol('http://www.example.com');
   * // -> true
   * path.hasProtocol('C:/Users/User/Documents/file.txt');
   * // -> true
   * ```
   */
  hasProtocol(path2) {
    return /^[^/:]+:/.test(this.toPosix(path2));
  },
  /**
   * Returns the protocol of the path e.g. http://, https://, file:///, data:, blob:, C:/
   * @param path - The path to get the protocol from
   * @example
   * ```ts
   * // Get the protocol from a URL
   * path.getProtocol('http://www.example.com/path/to/resource');
   * // -> 'http://'
   * // Get the protocol from a file path
   * path.getProtocol('C:/Users/User/Documents/file.txt');
   * // -> 'C:/'
   * ```
   */
  getProtocol(path2) {
    assertPath(path2);
    path2 = this.toPosix(path2);
    const matchFile = /^file:\/\/\//.exec(path2);
    if (matchFile) {
      return matchFile[0];
    }
    const matchProtocol = /^[^/:]+:\/{0,2}/.exec(path2);
    if (matchProtocol) {
      return matchProtocol[0];
    }
    return "";
  },
  /**
   * Converts URL to an absolute path.
   * When loading from a Web Worker, we must use absolute paths.
   * If the URL is already absolute we return it as is
   * If it's not, we convert it
   * @param url - The URL to test
   * @param customBaseUrl - The base URL to use
   * @param customRootUrl - The root URL to use
   * @example
   * ```ts
   * // Convert a relative URL to an absolute path
   * path.toAbsolute('images/texture.png', 'http://example.com/assets/');
   * // -> 'http://example.com/assets/images/texture.png'
   * ```
   */
  toAbsolute(url, customBaseUrl, customRootUrl) {
    assertPath(url);
    if (this.isDataUrl(url) || this.isBlobUrl(url))
      return url;
    const baseUrl = removeUrlParams(this.toPosix(customBaseUrl ?? DOMAdapter.get().getBaseUrl()));
    const rootUrl = removeUrlParams(this.toPosix(customRootUrl ?? this.rootname(baseUrl)));
    url = this.toPosix(url);
    if (url.startsWith("/")) {
      return path.join(rootUrl, url.slice(1));
    }
    const absolutePath = this.isAbsolute(url) ? url : this.join(baseUrl, url);
    return absolutePath;
  },
  /**
   * Normalizes the given path, resolving '..' and '.' segments
   * @param path - The path to normalize
   * @example
   * ```ts
   * // Normalize a path with relative segments
   * path.normalize('http://www.example.com/foo/bar/../baz');
   * // -> 'http://www.example.com/foo/baz'
   * // Normalize a file path with relative segments
   * path.normalize('C:\\Users\\User\\Documents\\..\\file.txt');
   * // -> 'C:/Users/User/file.txt'
   * ```
   */
  normalize(path2) {
    assertPath(path2);
    if (path2.length === 0)
      return ".";
    if (this.isDataUrl(path2) || this.isBlobUrl(path2))
      return path2;
    path2 = this.toPosix(path2);
    let protocol = "";
    const isAbsolute = path2.startsWith("/");
    if (this.hasProtocol(path2)) {
      protocol = this.rootname(path2);
      path2 = path2.slice(protocol.length);
    }
    const trailingSeparator = path2.endsWith("/");
    path2 = normalizeStringPosix(path2);
    if (path2.length > 0 && trailingSeparator)
      path2 += "/";
    if (isAbsolute)
      return `/${path2}`;
    return protocol + path2;
  },
  /**
   * Determines if path is an absolute path.
   * Absolute paths can be urls, data urls, or paths on disk
   * @param path - The path to test
   * @example
   * ```ts
   * // Check if a path is absolute
   * path.isAbsolute('http://www.example.com/foo/bar');
   * // -> true
   * path.isAbsolute('C:/Users/User/Documents/file.txt');
   * // -> true
   * ```
   */
  isAbsolute(path2) {
    assertPath(path2);
    path2 = this.toPosix(path2);
    if (this.hasProtocol(path2))
      return true;
    return path2.startsWith("/");
  },
  /**
   * Joins all given path segments together using the platform-specific separator as a delimiter,
   * then normalizes the resulting path
   * @param segments - The segments of the path to join
   * @example
   * ```ts
   * // Join multiple path segments
   * path.join('assets', 'images', 'sprite.png');
   * // -> 'assets/images/sprite.png'
   * // Join with relative segments
   * path.join('assets', 'images', '../textures', 'sprite.png');
   * // -> 'assets/textures/sprite.png'
   * ```
   */
  join(...segments) {
    if (segments.length === 0) {
      return ".";
    }
    let joined;
    for (let i2 = 0; i2 < segments.length; ++i2) {
      const arg = segments[i2];
      assertPath(arg);
      if (arg.length > 0) {
        if (joined === void 0)
          joined = arg;
        else {
          const prevArg = segments[i2 - 1] ?? "";
          if (this.joinExtensions.includes(this.extname(prevArg).toLowerCase())) {
            joined += `/../${arg}`;
          } else {
            joined += `/${arg}`;
          }
        }
      }
    }
    if (joined === void 0) {
      return ".";
    }
    return this.normalize(joined);
  },
  /**
   * Returns the directory name of a path
   * @param path - The path to parse
   * @example
   * ```ts
   * // Get the directory name of a path
   * path.dirname('http://www.example.com/foo/bar/baz.png');
   * // -> 'http://www.example.com/foo/bar'
   * // Get the directory name of a file path
   * path.dirname('C:/Users/User/Documents/file.txt');
   * // -> 'C:/Users/User/Documents'
   * ```
   */
  dirname(path2) {
    assertPath(path2);
    if (path2.length === 0)
      return ".";
    path2 = this.toPosix(path2);
    let code = path2.charCodeAt(0);
    const hasRoot = code === 47;
    let end = -1;
    let matchedSlash = true;
    const proto = this.getProtocol(path2);
    const origpath = path2;
    path2 = path2.slice(proto.length);
    for (let i2 = path2.length - 1; i2 >= 1; --i2) {
      code = path2.charCodeAt(i2);
      if (code === 47) {
        if (!matchedSlash) {
          end = i2;
          break;
        }
      } else {
        matchedSlash = false;
      }
    }
    if (end === -1)
      return hasRoot ? "/" : this.isUrl(origpath) ? proto + path2 : proto;
    if (hasRoot && end === 1)
      return "//";
    return proto + path2.slice(0, end);
  },
  /**
   * Returns the root of the path e.g. /, C:/, file:///, http://domain.com/
   * @param path - The path to parse
   * @example
   * ```ts
   * // Get the root of a URL
   * path.rootname('http://www.example.com/foo/bar/baz.png');
   * // -> 'http://www.example.com/'
   * // Get the root of a file path
   * path.rootname('C:/Users/User/Documents/file.txt');
   * // -> 'C:/'
   * ```
   */
  rootname(path2) {
    assertPath(path2);
    path2 = this.toPosix(path2);
    let root = "";
    if (path2.startsWith("/"))
      root = "/";
    else {
      root = this.getProtocol(path2);
    }
    if (this.isUrl(path2)) {
      const index = path2.indexOf("/", root.length);
      if (index !== -1) {
        root = path2.slice(0, index);
      } else
        root = path2;
      if (!root.endsWith("/"))
        root += "/";
    }
    return root;
  },
  /**
   * Returns the last portion of a path
   * @param path - The path to test
   * @param ext - Optional extension to remove
   * @example
   * ```ts
   * // Get the basename of a URL
   * path.basename('http://www.example.com/foo/bar/baz.png');
   * // -> 'baz.png'
   * // Get the basename of a file path
   * path.basename('C:/Users/User/Documents/file.txt');
   * // -> 'file.txt'
   * ```
   */
  basename(path2, ext) {
    assertPath(path2);
    if (ext)
      assertPath(ext);
    path2 = removeUrlParams(this.toPosix(path2));
    let start = 0;
    let end = -1;
    let matchedSlash = true;
    let i2;
    if (ext !== void 0 && ext.length > 0 && ext.length <= path2.length) {
      if (ext.length === path2.length && ext === path2)
        return "";
      let extIdx = ext.length - 1;
      let firstNonSlashEnd = -1;
      for (i2 = path2.length - 1; i2 >= 0; --i2) {
        const code = path2.charCodeAt(i2);
        if (code === 47) {
          if (!matchedSlash) {
            start = i2 + 1;
            break;
          }
        } else {
          if (firstNonSlashEnd === -1) {
            matchedSlash = false;
            firstNonSlashEnd = i2 + 1;
          }
          if (extIdx >= 0) {
            if (code === ext.charCodeAt(extIdx)) {
              if (--extIdx === -1) {
                end = i2;
              }
            } else {
              extIdx = -1;
              end = firstNonSlashEnd;
            }
          }
        }
      }
      if (start === end)
        end = firstNonSlashEnd;
      else if (end === -1)
        end = path2.length;
      return path2.slice(start, end);
    }
    for (i2 = path2.length - 1; i2 >= 0; --i2) {
      if (path2.charCodeAt(i2) === 47) {
        if (!matchedSlash) {
          start = i2 + 1;
          break;
        }
      } else if (end === -1) {
        matchedSlash = false;
        end = i2 + 1;
      }
    }
    if (end === -1)
      return "";
    return path2.slice(start, end);
  },
  /**
   * Returns the extension of the path, from the last occurrence of the . (period) character to end of string in the last
   * portion of the path. If there is no . in the last portion of the path, or if there are no . characters other than
   * the first character of the basename of path, an empty string is returned.
   * @param path - The path to parse
   * @example
   * ```ts
   * // Get the extension of a URL
   * path.extname('http://www.example.com/foo/bar/baz.png');
   * // -> '.png'
   * // Get the extension of a file path
   * path.extname('C:/Users/User/Documents/file.txt');
   * // -> '.txt'
   * ```
   */
  extname(path2) {
    assertPath(path2);
    path2 = removeUrlParams(this.toPosix(path2));
    let startDot = -1;
    let startPart = 0;
    let end = -1;
    let matchedSlash = true;
    let preDotState = 0;
    for (let i2 = path2.length - 1; i2 >= 0; --i2) {
      const code = path2.charCodeAt(i2);
      if (code === 47) {
        if (!matchedSlash) {
          startPart = i2 + 1;
          break;
        }
        continue;
      }
      if (end === -1) {
        matchedSlash = false;
        end = i2 + 1;
      }
      if (code === 46) {
        if (startDot === -1)
          startDot = i2;
        else if (preDotState !== 1)
          preDotState = 1;
      } else if (startDot !== -1) {
        preDotState = -1;
      }
    }
    if (startDot === -1 || end === -1 || preDotState === 0 || preDotState === 1 && startDot === end - 1 && startDot === startPart + 1) {
      return "";
    }
    return path2.slice(startDot, end);
  },
  /**
   * Parses a path into an object containing the 'root', `dir`, `base`, `ext`, and `name` properties.
   * @param path - The path to parse
   * @example
   * ```ts
   * // Parse a URL
   * const parsed = path.parse('http://www.example.com/foo/bar/baz.png');
   * // -> {
   * //   root: 'http://www.example.com/',
   * //   dir: 'http://www.example.com/foo/bar',
   * //   base: 'baz.png',
   * //   ext: '.png',
   * //   name: 'baz'
   * // }
   * // Parse a file path
   * const parsedFile = path.parse('C:/Users/User/Documents/file.txt');
   * // -> {
   * //   root: 'C:/',
   * //   dir: 'C:/Users/User/Documents',
   * //   base: 'file.txt',
   * //   ext: '.txt',
   * //   name: 'file'
   * // }
   * ```
   */
  parse(path2) {
    assertPath(path2);
    const ret = { root: "", dir: "", base: "", ext: "", name: "" };
    if (path2.length === 0)
      return ret;
    path2 = removeUrlParams(this.toPosix(path2));
    let code = path2.charCodeAt(0);
    const isAbsolute = this.isAbsolute(path2);
    let start;
    ret.root = this.rootname(path2);
    if (isAbsolute || this.hasProtocol(path2)) {
      start = 1;
    } else {
      start = 0;
    }
    let startDot = -1;
    let startPart = 0;
    let end = -1;
    let matchedSlash = true;
    let i2 = path2.length - 1;
    let preDotState = 0;
    for (; i2 >= start; --i2) {
      code = path2.charCodeAt(i2);
      if (code === 47) {
        if (!matchedSlash) {
          startPart = i2 + 1;
          break;
        }
        continue;
      }
      if (end === -1) {
        matchedSlash = false;
        end = i2 + 1;
      }
      if (code === 46) {
        if (startDot === -1)
          startDot = i2;
        else if (preDotState !== 1)
          preDotState = 1;
      } else if (startDot !== -1) {
        preDotState = -1;
      }
    }
    if (startDot === -1 || end === -1 || preDotState === 0 || preDotState === 1 && startDot === end - 1 && startDot === startPart + 1) {
      if (end !== -1) {
        if (startPart === 0 && isAbsolute)
          ret.base = ret.name = path2.slice(1, end);
        else
          ret.base = ret.name = path2.slice(startPart, end);
      }
    } else {
      if (startPart === 0 && isAbsolute) {
        ret.name = path2.slice(1, startDot);
        ret.base = path2.slice(1, end);
      } else {
        ret.name = path2.slice(startPart, startDot);
        ret.base = path2.slice(startPart, end);
      }
      ret.ext = path2.slice(startDot, end);
    }
    ret.dir = this.dirname(path2);
    return ret;
  },
  sep: "/",
  delimiter: ":",
  joinExtensions: [".html"]
};
function processX(base, ids, depth, result, tags) {
  const id = ids[depth];
  for (let i2 = 0; i2 < id.length; i2++) {
    const value = id[i2];
    if (depth < ids.length - 1) {
      processX(base.replace(result[depth], value), ids, depth + 1, result, tags);
    } else {
      tags.push(base.replace(result[depth], value));
    }
  }
}
function createStringVariations(string) {
  const regex = /\{(.*?)\}/g;
  const result = string.match(regex);
  const tags = [];
  if (result) {
    const ids = [];
    result.forEach((vars) => {
      const split = vars.substring(1, vars.length - 1).split(",");
      ids.push(split);
    });
    processX(string, ids, 0, result, tags);
  } else {
    tags.push(string);
  }
  return tags;
}
const isSingleItem = (item) => !Array.isArray(item);
class Resolver {
  constructor() {
    this._defaultBundleIdentifierOptions = {
      connector: "-",
      createBundleAssetId: (bundleId, assetId) => `${bundleId}${this._bundleIdConnector}${assetId}`,
      extractAssetIdFromBundle: (bundleId, assetBundleId) => assetBundleId.replace(`${bundleId}${this._bundleIdConnector}`, "")
    };
    this._bundleIdConnector = this._defaultBundleIdentifierOptions.connector;
    this._createBundleAssetId = this._defaultBundleIdentifierOptions.createBundleAssetId;
    this._extractAssetIdFromBundle = this._defaultBundleIdentifierOptions.extractAssetIdFromBundle;
    this._assetMap = {};
    this._preferredOrder = [];
    this._parsers = [];
    this._resolverHash = {};
    this._bundles = {};
  }
  /**
   * Override how the resolver deals with generating bundle ids.
   * must be called before any bundles are added
   * @param bundleIdentifier - the bundle identifier options
   */
  setBundleIdentifier(bundleIdentifier) {
    this._bundleIdConnector = bundleIdentifier.connector ?? this._bundleIdConnector;
    this._createBundleAssetId = bundleIdentifier.createBundleAssetId ?? this._createBundleAssetId;
    this._extractAssetIdFromBundle = bundleIdentifier.extractAssetIdFromBundle ?? this._extractAssetIdFromBundle;
    if (this._extractAssetIdFromBundle("foo", this._createBundleAssetId("foo", "bar")) !== "bar") {
      throw new Error("[Resolver] GenerateBundleAssetId are not working correctly");
    }
  }
  /**
   * Let the resolver know which assets you prefer to use when resolving assets.
   * Multiple prefer user defined rules can be added.
   * @example
   * resolver.prefer({
   *     // first look for something with the correct format, and then then correct resolution
   *     priority: ['format', 'resolution'],
   *     params:{
   *         format:'webp', // prefer webp images
   *         resolution: 2, // prefer a resolution of 2
   *     }
   * })
   * resolver.add('foo', ['bar@2x.webp', 'bar@2x.png', 'bar.webp', 'bar.png']);
   * resolver.resolveUrl('foo') // => 'bar@2x.webp'
   * @param preferOrders - the prefer options
   */
  prefer(...preferOrders) {
    preferOrders.forEach((prefer) => {
      this._preferredOrder.push(prefer);
      if (!prefer.priority) {
        prefer.priority = Object.keys(prefer.params);
      }
    });
    this._resolverHash = {};
  }
  /**
   * Set the base path to prepend to all urls when resolving
   * @example
   * resolver.basePath = 'https://home.com/';
   * resolver.add('foo', 'bar.ong');
   * resolver.resolveUrl('foo', 'bar.png'); // => 'https://home.com/bar.png'
   * @param basePath - the base path to use
   */
  set basePath(basePath) {
    this._basePath = basePath;
  }
  get basePath() {
    return this._basePath;
  }
  /**
   * Set the root path for root-relative URLs. By default the `basePath`'s root is used. If no `basePath` is set, then the
   * default value for browsers is `window.location.origin`
   * @example
   * // Application hosted on https://home.com/some-path/index.html
   * resolver.basePath = 'https://home.com/some-path/';
   * resolver.rootPath = 'https://home.com/';
   * resolver.add('foo', '/bar.png');
   * resolver.resolveUrl('foo', '/bar.png'); // => 'https://home.com/bar.png'
   * @param rootPath - the root path to use
   */
  set rootPath(rootPath) {
    this._rootPath = rootPath;
  }
  get rootPath() {
    return this._rootPath;
  }
  /**
   * All the active URL parsers that help the parser to extract information and create
   * an asset object-based on parsing the URL itself.
   *
   * Can be added using the extensions API
   * @example
   * resolver.add('foo', [
   *     {
   *         resolution: 2,
   *         format: 'png',
   *         src: 'image@2x.png',
   *     },
   *     {
   *         resolution:1,
   *         format:'png',
   *         src: 'image.png',
   *     },
   * ]);
   *
   * // With a url parser the information such as resolution and file format could extracted from the url itself:
   * extensions.add({
   *     extension: ExtensionType.ResolveParser,
   *     test: loadTextures.test, // test if url ends in an image
   *     parse: (value: string) =>
   *     ({
   *         resolution: parseFloat(Resolver.RETINA_PREFIX.exec(value)?.[1] ?? '1'),
   *         format: value.split('.').pop(),
   *         src: value,
   *     }),
   * });
   *
   * // Now resolution and format can be extracted from the url
   * resolver.add('foo', [
   *     'image@2x.png',
   *     'image.png',
   * ]);
   */
  get parsers() {
    return this._parsers;
  }
  /** Used for testing, this resets the resolver to its initial state */
  reset() {
    this.setBundleIdentifier(this._defaultBundleIdentifierOptions);
    this._assetMap = {};
    this._preferredOrder = [];
    this._resolverHash = {};
    this._rootPath = null;
    this._basePath = null;
    this._manifest = null;
    this._bundles = {};
    this._defaultSearchParams = null;
  }
  /**
   * Sets the default URL search parameters for the URL resolver. The urls can be specified as a string or an object.
   * @param searchParams - the default url parameters to append when resolving urls
   */
  setDefaultSearchParams(searchParams) {
    if (typeof searchParams === "string") {
      this._defaultSearchParams = searchParams;
    } else {
      const queryValues = searchParams;
      this._defaultSearchParams = Object.keys(queryValues).map((key) => `${encodeURIComponent(key)}=${encodeURIComponent(queryValues[key])}`).join("&");
    }
  }
  /**
   * Returns the aliases for a given asset
   * @param asset - the asset to get the aliases for
   */
  getAlias(asset) {
    const { alias, src } = asset;
    const aliasesToUse = convertToList(
      alias || src,
      (value) => {
        if (typeof value === "string")
          return value;
        if (Array.isArray(value))
          return value.map((v2) => v2?.src ?? v2);
        if (value?.src)
          return value.src;
        return value;
      },
      true
    );
    return aliasesToUse;
  }
  /**
   * Add a manifest to the asset resolver. This is a nice way to add all the asset information in one go.
   * generally a manifest would be built using a tool.
   * @param manifest - the manifest to add to the resolver
   */
  addManifest(manifest) {
    if (this._manifest) {
      warn("[Resolver] Manifest already exists, this will be overwritten");
    }
    this._manifest = manifest;
    manifest.bundles.forEach((bundle) => {
      this.addBundle(bundle.name, bundle.assets);
    });
  }
  /**
   * This adds a bundle of assets in one go so that you can resolve them as a group.
   * For example you could add a bundle for each screen in you pixi app
   * @example
   * resolver.addBundle('animals', [
   *  { alias: 'bunny', src: 'bunny.png' },
   *  { alias: 'chicken', src: 'chicken.png' },
   *  { alias: 'thumper', src: 'thumper.png' },
   * ]);
   * // or
   * resolver.addBundle('animals', {
   *     bunny: 'bunny.png',
   *     chicken: 'chicken.png',
   *     thumper: 'thumper.png',
   * });
   *
   * const resolvedAssets = await resolver.resolveBundle('animals');
   * @param bundleId - The id of the bundle to add
   * @param assets - A record of the asset or assets that will be chosen from when loading via the specified key
   */
  addBundle(bundleId, assets) {
    const assetNames = [];
    let convertedAssets = assets;
    if (!Array.isArray(assets)) {
      convertedAssets = Object.entries(assets).map(([alias, src]) => {
        if (typeof src === "string" || Array.isArray(src)) {
          return { alias, src };
        }
        return { alias, ...src };
      });
    }
    convertedAssets.forEach((asset) => {
      const srcs = asset.src;
      const aliases = asset.alias;
      let ids;
      if (typeof aliases === "string") {
        const bundleAssetId = this._createBundleAssetId(bundleId, aliases);
        assetNames.push(bundleAssetId);
        ids = [aliases, bundleAssetId];
      } else {
        const bundleIds = aliases.map((name) => this._createBundleAssetId(bundleId, name));
        assetNames.push(...bundleIds);
        ids = [...aliases, ...bundleIds];
      }
      this.add({
        ...asset,
        ...{
          alias: ids,
          src: srcs
        }
      });
    });
    this._bundles[bundleId] = assetNames;
  }
  /**
   * Tells the resolver what keys are associated with witch asset.
   * The most important thing the resolver does
   * @example
   * // Single key, single asset:
   * resolver.add({alias: 'foo', src: 'bar.png');
   * resolver.resolveUrl('foo') // => 'bar.png'
   *
   * // Multiple keys, single asset:
   * resolver.add({alias: ['foo', 'boo'], src: 'bar.png'});
   * resolver.resolveUrl('foo') // => 'bar.png'
   * resolver.resolveUrl('boo') // => 'bar.png'
   *
   * // Multiple keys, multiple assets:
   * resolver.add({alias: ['foo', 'boo'], src: ['bar.png', 'bar.webp']});
   * resolver.resolveUrl('foo') // => 'bar.png'
   *
   * // Add custom data attached to the resolver
   * Resolver.add({
   *     alias: 'bunnyBooBooSmooth',
   *     src: 'bunny{png,webp}',
   *     data: { scaleMode:SCALE_MODES.NEAREST }, // Base texture options
   * });
   *
   * resolver.resolve('bunnyBooBooSmooth') // => { src: 'bunny.png', data: { scaleMode: SCALE_MODES.NEAREST } }
   * @param aliases - the UnresolvedAsset or array of UnresolvedAssets to add to the resolver
   */
  add(aliases) {
    const assets = [];
    if (Array.isArray(aliases)) {
      assets.push(...aliases);
    } else {
      assets.push(aliases);
    }
    let keyCheck;
    keyCheck = (key) => {
      if (this.hasKey(key)) {
        warn(`[Resolver] already has key: ${key} overwriting`);
      }
    };
    const assetArray = convertToList(assets);
    assetArray.forEach((asset) => {
      const { src } = asset;
      let {
        data,
        format,
        loadParser: userDefinedLoadParser,
        parser: userDefinedParser
      } = asset;
      const srcsToUse = convertToList(src).map((src2) => {
        if (typeof src2 === "string") {
          return createStringVariations(src2);
        }
        return Array.isArray(src2) ? src2 : [src2];
      });
      const aliasesToUse = this.getAlias(asset);
      Array.isArray(aliasesToUse) ? aliasesToUse.forEach(keyCheck) : keyCheck(aliasesToUse);
      const resolvedAssets = [];
      const parseUrl = (url) => {
        const parser = this._parsers.find((p2) => p2.test(url));
        return {
          ...parser?.parse(url),
          src: url
        };
      };
      srcsToUse.forEach((srcs) => {
        srcs.forEach((src2) => {
          let formattedAsset = {};
          if (typeof src2 !== "object") {
            formattedAsset = parseUrl(src2);
          } else {
            data = src2.data ?? data;
            format = src2.format ?? format;
            if (src2.loadParser || src2.parser) {
              userDefinedLoadParser = src2.loadParser ?? userDefinedLoadParser;
              userDefinedParser = src2.parser ?? userDefinedParser;
            }
            formattedAsset = {
              ...parseUrl(src2.src),
              ...src2
            };
          }
          if (!aliasesToUse) {
            throw new Error(`[Resolver] alias is undefined for this asset: ${formattedAsset.src}`);
          }
          formattedAsset = this._buildResolvedAsset(formattedAsset, {
            aliases: aliasesToUse,
            data,
            format,
            loadParser: userDefinedLoadParser,
            parser: userDefinedParser,
            progressSize: asset.progressSize
          });
          resolvedAssets.push(formattedAsset);
        });
      });
      aliasesToUse.forEach((alias) => {
        this._assetMap[alias] = resolvedAssets;
      });
    });
  }
  // TODO: this needs an overload like load did in Assets
  /**
   * If the resolver has had a manifest set via setManifest, this will return the assets urls for
   * a given bundleId or bundleIds.
   * @example
   * // Manifest Example
   * const manifest = {
   *     bundles: [
   *         {
   *             name: 'load-screen',
   *             assets: [
   *                 {
   *                     alias: 'background',
   *                     src: 'sunset.png',
   *                 },
   *                 {
   *                     alias: 'bar',
   *                     src: 'load-bar.{png,webp}',
   *                 },
   *             ],
   *         },
   *         {
   *             name: 'game-screen',
   *             assets: [
   *                 {
   *                     alias: 'character',
   *                     src: 'robot.png',
   *                 },
   *                 {
   *                     alias: 'enemy',
   *                     src: 'bad-guy.png',
   *                 },
   *             ],
   *         },
   *     ]
   * };
   *
   * resolver.setManifest(manifest);
   * const resolved = resolver.resolveBundle('load-screen');
   * @param bundleIds - The bundle ids to resolve
   * @returns All the bundles assets or a hash of assets for each bundle specified
   */
  resolveBundle(bundleIds) {
    const singleAsset = isSingleItem(bundleIds);
    bundleIds = convertToList(bundleIds);
    const out2 = {};
    bundleIds.forEach((bundleId) => {
      const assetNames = this._bundles[bundleId];
      if (assetNames) {
        const results = this.resolve(assetNames);
        const assets = {};
        for (const key in results) {
          const asset = results[key];
          assets[this._extractAssetIdFromBundle(bundleId, key)] = asset;
        }
        out2[bundleId] = assets;
      }
    });
    return singleAsset ? out2[bundleIds[0]] : out2;
  }
  /**
   * Does exactly what resolve does, but returns just the URL rather than the whole asset object
   * @param key - The key or keys to resolve
   * @returns - The URLs associated with the key(s)
   */
  resolveUrl(key) {
    const result = this.resolve(key);
    if (typeof key !== "string") {
      const out2 = {};
      for (const i2 in result) {
        out2[i2] = result[i2].src;
      }
      return out2;
    }
    return result.src;
  }
  resolve(keys) {
    const singleAsset = isSingleItem(keys);
    keys = convertToList(keys);
    const result = {};
    keys.forEach((key) => {
      if (!this._resolverHash[key]) {
        if (this._assetMap[key]) {
          let assets = this._assetMap[key];
          const preferredOrder = this._getPreferredOrder(assets);
          preferredOrder?.priority.forEach((priorityKey) => {
            preferredOrder.params[priorityKey].forEach((value) => {
              const filteredAssets = assets.filter((asset) => {
                if (asset[priorityKey]) {
                  return asset[priorityKey] === value;
                }
                return false;
              });
              if (filteredAssets.length) {
                assets = filteredAssets;
              }
            });
          });
          this._resolverHash[key] = assets[0];
        } else {
          this._resolverHash[key] = this._buildResolvedAsset({
            alias: [key],
            src: key
          }, {});
        }
      }
      result[key] = this._resolverHash[key];
    });
    return singleAsset ? result[keys[0]] : result;
  }
  /**
   * Checks if an asset with a given key exists in the resolver
   * @param key - The key of the asset
   */
  hasKey(key) {
    return !!this._assetMap[key];
  }
  /**
   * Checks if a bundle with the given key exists in the resolver
   * @param key - The key of the bundle
   */
  hasBundle(key) {
    return !!this._bundles[key];
  }
  /**
   * Internal function for figuring out what prefer criteria an asset should use.
   * @param assets
   */
  _getPreferredOrder(assets) {
    for (let i2 = 0; i2 < assets.length; i2++) {
      const asset = assets[i2];
      const preferred = this._preferredOrder.find((preference) => preference.params.format.includes(asset.format));
      if (preferred) {
        return preferred;
      }
    }
    return this._preferredOrder[0];
  }
  /**
   * Appends the default url parameters to the url
   * @param url - The url to append the default parameters to
   * @returns - The url with the default parameters appended
   */
  _appendDefaultSearchParams(url) {
    if (!this._defaultSearchParams)
      return url;
    const paramConnector = /\?/.test(url) ? "&" : "?";
    return `${url}${paramConnector}${this._defaultSearchParams}`;
  }
  _buildResolvedAsset(formattedAsset, data) {
    const { aliases, data: assetData, loadParser, parser, format, progressSize } = data;
    if (this._basePath || this._rootPath) {
      formattedAsset.src = path.toAbsolute(formattedAsset.src, this._basePath, this._rootPath);
    }
    formattedAsset.alias = aliases ?? formattedAsset.alias ?? [formattedAsset.src];
    formattedAsset.src = this._appendDefaultSearchParams(formattedAsset.src);
    formattedAsset.data = { ...assetData || {}, ...formattedAsset.data };
    formattedAsset.loadParser = loadParser ?? formattedAsset.loadParser;
    formattedAsset.parser = parser ?? formattedAsset.parser;
    formattedAsset.format = format ?? formattedAsset.format ?? getUrlExtension(formattedAsset.src);
    if (progressSize !== void 0) {
      formattedAsset.progressSize = progressSize;
    }
    return formattedAsset;
  }
}
Resolver.RETINA_PREFIX = /@([0-9\.]+)x/;
function getUrlExtension(url) {
  return url.split(".").pop().split("?").shift().split("#").shift();
}
const copySearchParams = (targetUrl, sourceUrl) => {
  const searchParams = sourceUrl.split("?")[1];
  if (searchParams) {
    targetUrl += `?${searchParams}`;
  }
  return targetUrl;
};
const _Spritesheet = class _Spritesheet2 {
  constructor(optionsOrTexture, arg1) {
    this.linkedSheets = [];
    let options = optionsOrTexture;
    if (optionsOrTexture?.source instanceof TextureSource) {
      options = {
        texture: optionsOrTexture,
        data: arg1
      };
    }
    const { texture, data, cachePrefix = "" } = options;
    this.cachePrefix = cachePrefix;
    this._texture = texture instanceof Texture ? texture : null;
    this.textureSource = texture.source;
    this.textures = {};
    this.animations = {};
    this.data = data;
    const metaResolution = parseFloat(data.meta.scale);
    if (metaResolution) {
      this.resolution = metaResolution;
      texture.source.resolution = this.resolution;
    } else {
      this.resolution = texture.source._resolution;
    }
    this._frames = this.data.frames;
    this._frameKeys = Object.keys(this._frames);
    this._batchIndex = 0;
    this._callback = null;
  }
  /**
   * Parser spritesheet from loaded data. This is done asynchronously
   * to prevent creating too many Texture within a single process.
   */
  parse() {
    return new Promise((resolve) => {
      this._callback = resolve;
      this._batchIndex = 0;
      if (this._frameKeys.length <= _Spritesheet2.BATCH_SIZE) {
        this._processFrames(0);
        this._processAnimations();
        this._parseComplete();
      } else {
        this._nextBatch();
      }
    });
  }
  /**
   * Process a batch of frames
   * @param initialFrameIndex - The index of frame to start.
   */
  _processFrames(initialFrameIndex) {
    let frameIndex = initialFrameIndex;
    const maxFrames = _Spritesheet2.BATCH_SIZE;
    while (frameIndex - initialFrameIndex < maxFrames && frameIndex < this._frameKeys.length) {
      const i2 = this._frameKeys[frameIndex];
      const data = this._frames[i2];
      const rect = data.frame;
      if (rect) {
        let frame = null;
        let trim = null;
        const sourceSize = data.trimmed !== false && data.sourceSize ? data.sourceSize : data.frame;
        const orig = new Rectangle(
          0,
          0,
          Math.floor(sourceSize.w) / this.resolution,
          Math.floor(sourceSize.h) / this.resolution
        );
        if (data.rotated) {
          frame = new Rectangle(
            Math.floor(rect.x) / this.resolution,
            Math.floor(rect.y) / this.resolution,
            Math.floor(rect.h) / this.resolution,
            Math.floor(rect.w) / this.resolution
          );
        } else {
          frame = new Rectangle(
            Math.floor(rect.x) / this.resolution,
            Math.floor(rect.y) / this.resolution,
            Math.floor(rect.w) / this.resolution,
            Math.floor(rect.h) / this.resolution
          );
        }
        if (data.trimmed !== false && data.spriteSourceSize) {
          trim = new Rectangle(
            Math.floor(data.spriteSourceSize.x) / this.resolution,
            Math.floor(data.spriteSourceSize.y) / this.resolution,
            Math.floor(rect.w) / this.resolution,
            Math.floor(rect.h) / this.resolution
          );
        }
        this.textures[i2] = new Texture({
          source: this.textureSource,
          frame,
          orig,
          trim,
          rotate: data.rotated ? 2 : 0,
          defaultAnchor: data.anchor,
          defaultBorders: data.borders,
          label: i2.toString()
        });
      }
      frameIndex++;
    }
  }
  /** Parse animations config. */
  _processAnimations() {
    const animations = this.data.animations || {};
    for (const animName in animations) {
      this.animations[animName] = [];
      for (let i2 = 0; i2 < animations[animName].length; i2++) {
        const frameName = animations[animName][i2];
        this.animations[animName].push(this.textures[frameName]);
      }
    }
  }
  /** The parse has completed. */
  _parseComplete() {
    const callback = this._callback;
    this._callback = null;
    this._batchIndex = 0;
    callback.call(this, this.textures);
  }
  /** Begin the next batch of textures. */
  _nextBatch() {
    this._processFrames(this._batchIndex * _Spritesheet2.BATCH_SIZE);
    this._batchIndex++;
    setTimeout(() => {
      if (this._batchIndex * _Spritesheet2.BATCH_SIZE < this._frameKeys.length) {
        this._nextBatch();
      } else {
        this._processAnimations();
        this._parseComplete();
      }
    }, 0);
  }
  /**
   * Destroy Spritesheet and don't use after this.
   * @param {boolean} [destroyBase=false] - Whether to destroy the base texture as well
   */
  destroy(destroyBase = false) {
    for (const i2 in this.textures) {
      this.textures[i2].destroy();
    }
    this._frames = null;
    this._frameKeys = null;
    this.data = null;
    this.textures = null;
    if (destroyBase) {
      this._texture?.destroy();
      this.textureSource.destroy();
    }
    this._texture = null;
    this.textureSource = null;
    this.linkedSheets = [];
  }
};
_Spritesheet.BATCH_SIZE = 1e3;
let Spritesheet = _Spritesheet;
const validImages = [
  "jpg",
  "png",
  "jpeg",
  "avif",
  "webp",
  "basis",
  "etc2",
  "bc7",
  "bc6h",
  "bc5",
  "bc4",
  "bc3",
  "bc2",
  "bc1",
  "eac",
  "astc"
];
function getCacheableAssets(keys, asset, ignoreMultiPack) {
  const out2 = {};
  keys.forEach((key) => {
    out2[key] = asset;
  });
  Object.keys(asset.textures).forEach((key) => {
    out2[`${asset.cachePrefix}${key}`] = asset.textures[key];
  });
  if (!ignoreMultiPack) {
    const basePath = path.dirname(keys[0]);
    asset.linkedSheets.forEach((item, i2) => {
      const out22 = getCacheableAssets([`${basePath}/${asset.data.meta.related_multi_packs[i2]}`], item, true);
      Object.assign(out2, out22);
    });
  }
  return out2;
}
const spritesheetAsset = {
  extension: ExtensionType.Asset,
  /** Handle the caching of the related Spritesheet Textures */
  cache: {
    test: (asset) => asset instanceof Spritesheet,
    getCacheableAssets: (keys, asset) => getCacheableAssets(keys, asset, false)
  },
  /** Resolve the resolution of the asset. */
  resolver: {
    extension: {
      type: ExtensionType.ResolveParser,
      name: "resolveSpritesheet"
    },
    test: (value) => {
      const tempURL = value.split("?")[0];
      const split = tempURL.split(".");
      const extension = split.pop();
      const format = split.pop();
      return extension === "json" && validImages.includes(format);
    },
    parse: (value) => {
      const split = value.split(".");
      return {
        resolution: parseFloat(Resolver.RETINA_PREFIX.exec(value)?.[1] ?? "1"),
        format: split[split.length - 2],
        src: value
      };
    }
  },
  /**
   * Loader plugin that parses sprite sheets!
   * once the JSON has been loaded this checks to see if the JSON is spritesheet data.
   * If it is, we load the spritesheets image and parse the data into Spritesheet
   * All textures in the sprite sheet are then added to the cache
   */
  loader: {
    /** used for deprecation purposes */
    name: "spritesheetLoader",
    id: "spritesheet",
    extension: {
      type: ExtensionType.LoadParser,
      priority: LoaderParserPriority.Normal,
      name: "spritesheetLoader"
    },
    async testParse(asset, options) {
      return path.extname(options.src).toLowerCase() === ".json" && !!asset.frames;
    },
    async parse(asset, options, loader) {
      const {
        texture: imageTexture,
        // if user need to use preloaded texture
        imageFilename,
        // if user need to use custom filename (not from jsonFile.meta.image)
        textureOptions,
        // if user need to set texture options on texture
        cachePrefix
        // if user need to use custom cache prefix
      } = options?.data ?? {};
      let basePath = path.dirname(options.src);
      if (basePath && basePath.lastIndexOf("/") !== basePath.length - 1) {
        basePath += "/";
      }
      let texture;
      if (imageTexture instanceof Texture) {
        texture = imageTexture;
      } else {
        const imagePath = copySearchParams(basePath + (imageFilename ?? asset.meta.image), options.src);
        const assets = await loader.load([{ src: imagePath, data: textureOptions }]);
        texture = assets[imagePath];
      }
      const spritesheet = new Spritesheet({
        texture: texture.source,
        data: asset,
        cachePrefix
      });
      await spritesheet.parse();
      const multiPacks = asset?.meta?.related_multi_packs;
      if (Array.isArray(multiPacks)) {
        const promises = [];
        for (const item of multiPacks) {
          if (typeof item !== "string") {
            continue;
          }
          let itemUrl = basePath + item;
          if (options.data?.ignoreMultiPack) {
            continue;
          }
          itemUrl = copySearchParams(itemUrl, options.src);
          promises.push(loader.load({
            src: itemUrl,
            data: {
              textureOptions,
              ignoreMultiPack: true
            }
          }));
        }
        const res = await Promise.all(promises);
        spritesheet.linkedSheets = res;
        res.forEach((item) => {
          item.linkedSheets = [spritesheet].concat(spritesheet.linkedSheets.filter((sp) => sp !== item));
        });
      }
      return spritesheet;
    },
    async unload(spritesheet, _resolvedAsset, loader) {
      await loader.unload(spritesheet.textureSource._sourceOrigin);
      spritesheet.destroy(false);
    }
  }
};
extensions.add(spritesheetAsset);
const idCounts = /* @__PURE__ */ Object.create(null);
const idHash = /* @__PURE__ */ Object.create(null);
function createIdFromString(value, groupId) {
  let id = idHash[value];
  if (id === void 0) {
    if (idCounts[groupId] === void 0) {
      idCounts[groupId] = 1;
    }
    idHash[value] = id = idCounts[groupId]++;
  }
  return id;
}
let context;
function getTestContext() {
  if (!context || context?.isContextLost()) {
    const canvas = DOMAdapter.get().createCanvas();
    context = canvas.getContext("webgl", {});
  }
  return context;
}
let maxFragmentPrecision;
function getMaxFragmentPrecision() {
  if (!maxFragmentPrecision) {
    maxFragmentPrecision = "mediump";
    const gl = getTestContext();
    if (gl) {
      if (gl.getShaderPrecisionFormat) {
        const shaderFragment = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
        maxFragmentPrecision = shaderFragment.precision ? "highp" : "mediump";
      }
    }
  }
  return maxFragmentPrecision;
}
function addProgramDefines(src, isES300, isFragment) {
  if (isES300)
    return src;
  if (isFragment) {
    src = src.replace("out vec4 finalColor;", "");
    return `

        #ifdef GL_ES // This checks if it is WebGL1
        #define in varying
        #define finalColor gl_FragColor
        #define texture texture2D
        #endif
        ${src}
        `;
  }
  return `

        #ifdef GL_ES // This checks if it is WebGL1
        #define in attribute
        #define out varying
        #endif
        ${src}
        `;
}
function ensurePrecision(src, options, isFragment) {
  const maxSupportedPrecision = isFragment ? options.maxSupportedFragmentPrecision : options.maxSupportedVertexPrecision;
  if (src.substring(0, 9) !== "precision") {
    let precision = isFragment ? options.requestedFragmentPrecision : options.requestedVertexPrecision;
    if (precision === "highp" && maxSupportedPrecision !== "highp") {
      precision = "mediump";
    }
    return `precision ${precision} float;
${src}`;
  } else if (maxSupportedPrecision !== "highp" && src.substring(0, 15) === "precision highp") {
    return src.replace("precision highp", "precision mediump");
  }
  return src;
}
function insertVersion(src, isES300) {
  if (!isES300)
    return src;
  return `#version 300 es
${src}`;
}
const fragmentNameCache = {};
const VertexNameCache = {};
function setProgramName(src, { name = `pixi-program` }, isFragment = true) {
  name = name.replace(/\s+/g, "-");
  name += isFragment ? "-fragment" : "-vertex";
  const nameCache = isFragment ? fragmentNameCache : VertexNameCache;
  if (nameCache[name]) {
    nameCache[name]++;
    name += `-${nameCache[name]}`;
  } else {
    nameCache[name] = 1;
  }
  if (src.indexOf("#define SHADER_NAME") !== -1)
    return src;
  const shaderName = `#define SHADER_NAME ${name}`;
  return `${shaderName}
${src}`;
}
function stripVersion(src, isES300) {
  if (!isES300)
    return src;
  return src.replace("#version 300 es", "");
}
const processes = {
  // strips any version headers..
  stripVersion,
  // adds precision string if not already present
  ensurePrecision,
  // add some defines if WebGL1 to make it more compatible with WebGL2 shaders
  addProgramDefines,
  // add the program name to the shader
  setProgramName,
  // add the version string to the shader header
  insertVersion
};
const programCache$1 = /* @__PURE__ */ Object.create(null);
const _GlProgram = class _GlProgram2 {
  /**
   * Creates a shiny new GlProgram. Used by WebGL renderer.
   * @param options - The options for the program.
   */
  constructor(options) {
    options = { ..._GlProgram2.defaultOptions, ...options };
    const isES300 = options.fragment.indexOf("#version 300 es") !== -1;
    const preprocessorOptions = {
      stripVersion: isES300,
      ensurePrecision: {
        requestedFragmentPrecision: options.preferredFragmentPrecision,
        requestedVertexPrecision: options.preferredVertexPrecision,
        maxSupportedVertexPrecision: "highp",
        maxSupportedFragmentPrecision: getMaxFragmentPrecision()
      },
      setProgramName: {
        name: options.name
      },
      addProgramDefines: isES300,
      insertVersion: isES300
    };
    let fragment2 = options.fragment;
    let vertex2 = options.vertex;
    Object.keys(processes).forEach((processKey) => {
      const processOptions = preprocessorOptions[processKey];
      fragment2 = processes[processKey](fragment2, processOptions, true);
      vertex2 = processes[processKey](vertex2, processOptions, false);
    });
    this.fragment = fragment2;
    this.vertex = vertex2;
    this.transformFeedbackVaryings = options.transformFeedbackVaryings;
    this._key = createIdFromString(`${this.vertex}:${this.fragment}`, "gl-program");
  }
  /** destroys the program */
  destroy() {
    this.fragment = null;
    this.vertex = null;
    this._attributeData = null;
    this._uniformData = null;
    this._uniformBlockData = null;
    this.transformFeedbackVaryings = null;
    programCache$1[this._cacheKey] = null;
  }
  /**
   * Helper function that creates a program for a given source.
   * It will check the program cache if the program has already been created.
   * If it has that one will be returned, if not a new one will be created and cached.
   * @param options - The options for the program.
   * @returns A program using the same source
   */
  static from(options) {
    const key = `${options.vertex}:${options.fragment}`;
    if (!programCache$1[key]) {
      programCache$1[key] = new _GlProgram2(options);
      programCache$1[key]._cacheKey = key;
    }
    return programCache$1[key];
  }
};
_GlProgram.defaultOptions = {
  preferredVertexPrecision: "highp",
  preferredFragmentPrecision: "mediump"
};
let GlProgram = _GlProgram;
const attributeFormatData = {
  uint8x2: { size: 2, stride: 2, normalised: false },
  uint8x4: { size: 4, stride: 4, normalised: false },
  sint8x2: { size: 2, stride: 2, normalised: false },
  sint8x4: { size: 4, stride: 4, normalised: false },
  unorm8x2: { size: 2, stride: 2, normalised: true },
  unorm8x4: { size: 4, stride: 4, normalised: true },
  snorm8x2: { size: 2, stride: 2, normalised: true },
  snorm8x4: { size: 4, stride: 4, normalised: true },
  uint16x2: { size: 2, stride: 4, normalised: false },
  uint16x4: { size: 4, stride: 8, normalised: false },
  sint16x2: { size: 2, stride: 4, normalised: false },
  sint16x4: { size: 4, stride: 8, normalised: false },
  unorm16x2: { size: 2, stride: 4, normalised: true },
  unorm16x4: { size: 4, stride: 8, normalised: true },
  snorm16x2: { size: 2, stride: 4, normalised: true },
  snorm16x4: { size: 4, stride: 8, normalised: true },
  float16x2: { size: 2, stride: 4, normalised: false },
  float16x4: { size: 4, stride: 8, normalised: false },
  float32: { size: 1, stride: 4, normalised: false },
  float32x2: { size: 2, stride: 8, normalised: false },
  float32x3: { size: 3, stride: 12, normalised: false },
  float32x4: { size: 4, stride: 16, normalised: false },
  uint32: { size: 1, stride: 4, normalised: false },
  uint32x2: { size: 2, stride: 8, normalised: false },
  uint32x3: { size: 3, stride: 12, normalised: false },
  uint32x4: { size: 4, stride: 16, normalised: false },
  sint32: { size: 1, stride: 4, normalised: false },
  sint32x2: { size: 2, stride: 8, normalised: false },
  sint32x3: { size: 3, stride: 12, normalised: false },
  sint32x4: { size: 4, stride: 16, normalised: false }
};
function getAttributeInfoFromFormat(format) {
  return attributeFormatData[format] ?? attributeFormatData.float32;
}
const WGSL_TO_VERTEX_TYPES = {
  f32: "float32",
  "vec2<f32>": "float32x2",
  "vec3<f32>": "float32x3",
  "vec4<f32>": "float32x4",
  vec2f: "float32x2",
  vec3f: "float32x3",
  vec4f: "float32x4",
  i32: "sint32",
  "vec2<i32>": "sint32x2",
  "vec3<i32>": "sint32x3",
  "vec4<i32>": "sint32x4",
  u32: "uint32",
  "vec2<u32>": "uint32x2",
  "vec3<u32>": "uint32x3",
  "vec4<u32>": "uint32x4",
  bool: "uint32",
  "vec2<bool>": "uint32x2",
  "vec3<bool>": "uint32x3",
  "vec4<bool>": "uint32x4"
};
function extractAttributesFromGpuProgram({ source: source2, entryPoint }) {
  const results = {};
  const mainVertStart = source2.indexOf(`fn ${entryPoint}`);
  if (mainVertStart !== -1) {
    const arrowFunctionStart = source2.indexOf("->", mainVertStart);
    if (arrowFunctionStart !== -1) {
      const functionArgsSubstring = source2.substring(mainVertStart, arrowFunctionStart);
      const inputsRegex = /@location\((\d+)\)\s+([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_<>]+)(?:,|\s|$)/g;
      let match;
      while ((match = inputsRegex.exec(functionArgsSubstring)) !== null) {
        const format = WGSL_TO_VERTEX_TYPES[match[3]] ?? "float32";
        results[match[2]] = {
          location: parseInt(match[1], 10),
          format,
          stride: getAttributeInfoFromFormat(format).stride,
          offset: 0,
          instance: false,
          start: 0
        };
      }
    }
  }
  return results;
}
function extractStructAndGroups(wgsl) {
  const linePattern = /(^|[^/])@(group|binding)\(\d+\)[^;]+;/g;
  const groupPattern = /@group\((\d+)\)/;
  const bindingPattern = /@binding\((\d+)\)/;
  const namePattern = /var(<[^>]+>)? (\w+)/;
  const typePattern = /:\s*(\w+)/;
  const structPattern = /struct\s+(\w+)\s*{([^}]+)}/g;
  const structMemberPattern = /(\w+)\s*:\s*([\w\<\>]+)/g;
  const structName = /struct\s+(\w+)/;
  const groups = wgsl.match(linePattern)?.map((item) => ({
    group: parseInt(item.match(groupPattern)[1], 10),
    binding: parseInt(item.match(bindingPattern)[1], 10),
    name: item.match(namePattern)[2],
    isUniform: item.match(namePattern)[1] === "<uniform>",
    type: item.match(typePattern)[1]
  }));
  if (!groups) {
    return {
      groups: [],
      structs: []
    };
  }
  const structs = wgsl.match(structPattern)?.map((struct) => {
    const name = struct.match(structName)[1];
    const members = struct.match(structMemberPattern).reduce((acc, member) => {
      const [name2, type] = member.split(":");
      acc[name2.trim()] = type.trim();
      return acc;
    }, {});
    if (!members) {
      return null;
    }
    return { name, members };
  }).filter(({ name }) => groups.some((group) => group.type === name)) ?? [];
  return {
    groups,
    structs
  };
}
var ShaderStage = /* @__PURE__ */ ((ShaderStage2) => {
  ShaderStage2[ShaderStage2["VERTEX"] = 1] = "VERTEX";
  ShaderStage2[ShaderStage2["FRAGMENT"] = 2] = "FRAGMENT";
  ShaderStage2[ShaderStage2["COMPUTE"] = 4] = "COMPUTE";
  return ShaderStage2;
})(ShaderStage || {});
function generateGpuLayoutGroups({ groups }) {
  const layout = [];
  for (let i2 = 0; i2 < groups.length; i2++) {
    const group = groups[i2];
    if (!layout[group.group]) {
      layout[group.group] = [];
    }
    if (group.isUniform) {
      layout[group.group].push({
        binding: group.binding,
        visibility: ShaderStage.VERTEX | ShaderStage.FRAGMENT,
        buffer: {
          type: "uniform"
        }
      });
    } else if (group.type === "sampler") {
      layout[group.group].push({
        binding: group.binding,
        visibility: ShaderStage.FRAGMENT,
        sampler: {
          type: "filtering"
        }
      });
    } else if (group.type === "texture_2d") {
      layout[group.group].push({
        binding: group.binding,
        visibility: ShaderStage.FRAGMENT,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
          multisampled: false
        }
      });
    }
  }
  return layout;
}
function generateLayoutHash({ groups }) {
  const layout = [];
  for (let i2 = 0; i2 < groups.length; i2++) {
    const group = groups[i2];
    if (!layout[group.group]) {
      layout[group.group] = {};
    }
    layout[group.group][group.name] = group.binding;
  }
  return layout;
}
function removeStructAndGroupDuplicates(vertexStructsAndGroups, fragmentStructsAndGroups) {
  const structNameSet = /* @__PURE__ */ new Set();
  const dupeGroupKeySet = /* @__PURE__ */ new Set();
  const structs = [...vertexStructsAndGroups.structs, ...fragmentStructsAndGroups.structs].filter((struct) => {
    if (structNameSet.has(struct.name)) {
      return false;
    }
    structNameSet.add(struct.name);
    return true;
  });
  const groups = [...vertexStructsAndGroups.groups, ...fragmentStructsAndGroups.groups].filter((group) => {
    const key = `${group.name}-${group.binding}`;
    if (dupeGroupKeySet.has(key)) {
      return false;
    }
    dupeGroupKeySet.add(key);
    return true;
  });
  return { structs, groups };
}
const programCache = /* @__PURE__ */ Object.create(null);
class GpuProgram {
  /**
   * Create a new GpuProgram
   * @param options - The options for the gpu program
   */
  constructor(options) {
    this._layoutKey = 0;
    this._attributeLocationsKey = 0;
    const { fragment: fragment2, vertex: vertex2, layout, gpuLayout, name } = options;
    this.name = name;
    this.fragment = fragment2;
    this.vertex = vertex2;
    if (fragment2.source === vertex2.source) {
      const structsAndGroups = extractStructAndGroups(fragment2.source);
      this.structsAndGroups = structsAndGroups;
    } else {
      const vertexStructsAndGroups = extractStructAndGroups(vertex2.source);
      const fragmentStructsAndGroups = extractStructAndGroups(fragment2.source);
      this.structsAndGroups = removeStructAndGroupDuplicates(vertexStructsAndGroups, fragmentStructsAndGroups);
    }
    this.layout = layout ?? generateLayoutHash(this.structsAndGroups);
    this.gpuLayout = gpuLayout ?? generateGpuLayoutGroups(this.structsAndGroups);
    this.autoAssignGlobalUniforms = !!(this.layout[0]?.globalUniforms !== void 0);
    this.autoAssignLocalUniforms = !!(this.layout[1]?.localUniforms !== void 0);
    this._generateProgramKey();
  }
  // TODO maker this pure
  _generateProgramKey() {
    const { vertex: vertex2, fragment: fragment2 } = this;
    const bigKey = vertex2.source + fragment2.source + vertex2.entryPoint + fragment2.entryPoint;
    this._layoutKey = createIdFromString(bigKey, "program");
  }
  get attributeData() {
    this._attributeData ?? (this._attributeData = extractAttributesFromGpuProgram(this.vertex));
    return this._attributeData;
  }
  /** destroys the program */
  destroy() {
    this.gpuLayout = null;
    this.layout = null;
    this.structsAndGroups = null;
    this.fragment = null;
    this.vertex = null;
    programCache[this._cacheKey] = null;
  }
  /**
   * Helper function that creates a program for a given source.
   * It will check the program cache if the program has already been created.
   * If it has that one will be returned, if not a new one will be created and cached.
   * @param options - The options for the program.
   * @returns A program using the same source
   */
  static from(options) {
    const key = `${options.vertex.source}:${options.fragment.source}:${options.fragment.entryPoint}:${options.vertex.entryPoint}`;
    if (!programCache[key]) {
      programCache[key] = new GpuProgram(options);
      programCache[key]._cacheKey = key;
    }
    return programCache[key];
  }
}
const UNIFORM_TYPES_VALUES = [
  "f32",
  "i32",
  "vec2<f32>",
  "vec3<f32>",
  "vec4<f32>",
  "mat2x2<f32>",
  "mat3x3<f32>",
  "mat4x4<f32>",
  "mat3x2<f32>",
  "mat4x2<f32>",
  "mat2x3<f32>",
  "mat4x3<f32>",
  "mat2x4<f32>",
  "mat3x4<f32>",
  "vec2<i32>",
  "vec3<i32>",
  "vec4<i32>"
];
const UNIFORM_TYPES_MAP = UNIFORM_TYPES_VALUES.reduce((acc, type) => {
  acc[type] = true;
  return acc;
}, {});
function getDefaultUniformValue(type, size) {
  switch (type) {
    case "f32":
      return 0;
    case "vec2<f32>":
      return new Float32Array(2 * size);
    case "vec3<f32>":
      return new Float32Array(3 * size);
    case "vec4<f32>":
      return new Float32Array(4 * size);
    case "mat2x2<f32>":
      return new Float32Array([
        1,
        0,
        0,
        1
      ]);
    case "mat3x3<f32>":
      return new Float32Array([
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1
      ]);
    case "mat4x4<f32>":
      return new Float32Array([
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1
      ]);
  }
  return null;
}
const _UniformGroup = class _UniformGroup2 {
  /**
   * Create a new Uniform group
   * @param uniformStructures - The structures of the uniform group
   * @param options - The optional parameters of this uniform group
   */
  constructor(uniformStructures, options) {
    this._touched = 0;
    this.uid = uid("uniform");
    this._resourceType = "uniformGroup";
    this._resourceId = uid("resource");
    this.isUniformGroup = true;
    this._dirtyId = 0;
    this.destroyed = false;
    options = { ..._UniformGroup2.defaultOptions, ...options };
    this.uniformStructures = uniformStructures;
    const uniforms = {};
    for (const i2 in uniformStructures) {
      const uniformData = uniformStructures[i2];
      uniformData.name = i2;
      uniformData.size = uniformData.size ?? 1;
      if (!UNIFORM_TYPES_MAP[uniformData.type]) {
        const arrayMatch = uniformData.type.match(/^array<(\w+(?:<\w+>)?),\s*(\d+)>$/);
        if (arrayMatch) {
          const [, innerType, size] = arrayMatch;
          throw new Error(
            `Uniform type ${uniformData.type} is not supported. Use type: '${innerType}', size: ${size} instead.`
          );
        }
        throw new Error(`Uniform type ${uniformData.type} is not supported. Supported uniform types are: ${UNIFORM_TYPES_VALUES.join(", ")}`);
      }
      uniformData.value ?? (uniformData.value = getDefaultUniformValue(uniformData.type, uniformData.size));
      uniforms[i2] = uniformData.value;
    }
    this.uniforms = uniforms;
    this._dirtyId = 1;
    this.ubo = options.ubo;
    this.isStatic = options.isStatic;
    this._signature = createIdFromString(Object.keys(uniforms).map(
      (i2) => `${i2}-${uniformStructures[i2].type}`
    ).join("-"), "uniform-group");
  }
  /** Call this if you want the uniform groups data to be uploaded to the GPU only useful if `isStatic` is true. */
  update() {
    this._dirtyId++;
  }
};
_UniformGroup.defaultOptions = {
  /** if true the UniformGroup is handled as an Uniform buffer object. */
  ubo: false,
  /** if true, then you are responsible for when the data is uploaded to the GPU by calling `update()` */
  isStatic: false
};
let UniformGroup = _UniformGroup;
class BindGroup {
  /**
   * Create a new instance eof the Bind Group.
   * @param resources - The resources that are bound together for use by a shader.
   */
  constructor(resources) {
    this.resources = /* @__PURE__ */ Object.create(null);
    this._dirty = true;
    let index = 0;
    for (const i2 in resources) {
      const resource = resources[i2];
      this.setResource(resource, index++);
    }
    this._updateKey();
  }
  /**
   * Updates the key if its flagged as dirty. This is used internally to
   * match this bind group to a WebGPU BindGroup.
   * @internal
   */
  _updateKey() {
    if (!this._dirty)
      return;
    this._dirty = false;
    const keyParts = [];
    let index = 0;
    for (const i2 in this.resources) {
      keyParts[index++] = this.resources[i2]._resourceId;
    }
    this._key = keyParts.join("|");
  }
  /**
   * Set a resource at a given index. this function will
   * ensure that listeners will be removed from the current resource
   * and added to the new resource.
   * @param resource - The resource to set.
   * @param index - The index to set the resource at.
   */
  setResource(resource, index) {
    const currentResource = this.resources[index];
    if (resource === currentResource)
      return;
    if (currentResource) {
      resource.off?.("change", this.onResourceChange, this);
    }
    resource.on?.("change", this.onResourceChange, this);
    this.resources[index] = resource;
    this._dirty = true;
  }
  /**
   * Returns the resource at the current specified index.
   * @param index - The index of the resource to get.
   * @returns - The resource at the specified index.
   */
  getResource(index) {
    return this.resources[index];
  }
  /**
   * Used internally to 'touch' each resource, to ensure that the GC
   * knows that all resources in this bind group are still being used.
   * @param now - The current time in milliseconds.
   * @param tick - The current tick.
   * @internal
   */
  _touch(now, tick2) {
    const resources = this.resources;
    for (const i2 in resources) {
      resources[i2]._gcLastUsed = now;
      resources[i2]._touched = tick2;
    }
  }
  /** Destroys this bind group and removes all listeners. */
  destroy() {
    const resources = this.resources;
    for (const i2 in resources) {
      const resource = resources[i2];
      resource?.off?.("change", this.onResourceChange, this);
    }
    this.resources = null;
  }
  onResourceChange(resource) {
    this._dirty = true;
    if (resource.destroyed) {
      const resources = this.resources;
      for (const i2 in resources) {
        if (resources[i2] === resource) {
          resources[i2] = null;
        }
      }
    } else {
      this._updateKey();
    }
  }
}
var RendererType = /* @__PURE__ */ ((RendererType2) => {
  RendererType2[RendererType2["WEBGL"] = 1] = "WEBGL";
  RendererType2[RendererType2["WEBGPU"] = 2] = "WEBGPU";
  RendererType2[RendererType2["BOTH"] = 3] = "BOTH";
  return RendererType2;
})(RendererType || {});
class Shader extends EventEmitter {
  constructor(options) {
    super();
    this.uid = uid("shader");
    this._uniformBindMap = /* @__PURE__ */ Object.create(null);
    this._ownedBindGroups = [];
    this._destroyed = false;
    let {
      gpuProgram,
      glProgram,
      groups,
      resources,
      compatibleRenderers,
      groupMap
    } = options;
    this.gpuProgram = gpuProgram;
    this.glProgram = glProgram;
    if (compatibleRenderers === void 0) {
      compatibleRenderers = 0;
      if (gpuProgram)
        compatibleRenderers |= RendererType.WEBGPU;
      if (glProgram)
        compatibleRenderers |= RendererType.WEBGL;
    }
    this.compatibleRenderers = compatibleRenderers;
    const nameHash = {};
    if (!resources && !groups) {
      resources = {};
    }
    if (resources && groups) {
      throw new Error("[Shader] Cannot have both resources and groups");
    } else if (!gpuProgram && groups && !groupMap) {
      throw new Error("[Shader] No group map or WebGPU shader provided - consider using resources instead.");
    } else if (!gpuProgram && groups && groupMap) {
      for (const i2 in groupMap) {
        for (const j2 in groupMap[i2]) {
          const uniformName = groupMap[i2][j2];
          nameHash[uniformName] = {
            group: i2,
            binding: j2,
            name: uniformName
          };
        }
      }
    } else if (gpuProgram && groups && !groupMap) {
      const groupData = gpuProgram.structsAndGroups.groups;
      groupMap = {};
      groupData.forEach((data) => {
        groupMap[data.group] = groupMap[data.group] || {};
        groupMap[data.group][data.binding] = data.name;
        nameHash[data.name] = data;
      });
    } else if (resources) {
      groups = {};
      groupMap = {};
      if (gpuProgram) {
        const groupData = gpuProgram.structsAndGroups.groups;
        groupData.forEach((data) => {
          groupMap[data.group] = groupMap[data.group] || {};
          groupMap[data.group][data.binding] = data.name;
          nameHash[data.name] = data;
        });
      }
      let bindTick = 0;
      for (const i2 in resources) {
        if (nameHash[i2])
          continue;
        if (!groups[99]) {
          groups[99] = new BindGroup();
          this._ownedBindGroups.push(groups[99]);
        }
        nameHash[i2] = { group: 99, binding: bindTick, name: i2 };
        groupMap[99] = groupMap[99] || {};
        groupMap[99][bindTick] = i2;
        bindTick++;
      }
      for (const i2 in resources) {
        const name = i2;
        let value = resources[i2];
        if (!value.source && !value._resourceType) {
          value = new UniformGroup(value);
        }
        const data = nameHash[name];
        if (data) {
          if (!groups[data.group]) {
            groups[data.group] = new BindGroup();
            this._ownedBindGroups.push(groups[data.group]);
          }
          groups[data.group].setResource(value, data.binding);
        }
      }
    }
    this.groups = groups;
    this._uniformBindMap = groupMap;
    this.resources = this._buildResourceAccessor(groups, nameHash);
  }
  /**
   * Sometimes a resource group will be provided later (for example global uniforms)
   * In such cases, this method can be used to let the shader know about the group.
   * @param name - the name of the resource group
   * @param groupIndex - the index of the group (should match the webGPU shader group location)
   * @param bindIndex - the index of the bind point (should match the webGPU shader bind point)
   */
  addResource(name, groupIndex, bindIndex) {
    var _a, _b;
    (_a = this._uniformBindMap)[groupIndex] || (_a[groupIndex] = {});
    (_b = this._uniformBindMap[groupIndex])[bindIndex] || (_b[bindIndex] = name);
    if (!this.groups[groupIndex]) {
      this.groups[groupIndex] = new BindGroup();
      this._ownedBindGroups.push(this.groups[groupIndex]);
    }
  }
  _buildResourceAccessor(groups, nameHash) {
    const uniformsOut = {};
    for (const i2 in nameHash) {
      const data = nameHash[i2];
      Object.defineProperty(uniformsOut, data.name, {
        get() {
          return groups[data.group].getResource(data.binding);
        },
        set(value) {
          groups[data.group].setResource(value, data.binding);
        }
      });
    }
    return uniformsOut;
  }
  /**
   * Use to destroy the shader when its not longer needed.
   * It will destroy the resources and remove listeners.
   * @param destroyPrograms - if the programs should be destroyed as well.
   * Make sure its not being used by other shaders!
   */
  destroy(destroyPrograms = false) {
    if (this._destroyed)
      return;
    this._destroyed = true;
    this.emit("destroy", this);
    if (destroyPrograms) {
      this.gpuProgram?.destroy();
      this.glProgram?.destroy();
    }
    this.gpuProgram = null;
    this.glProgram = null;
    this.removeAllListeners();
    this._uniformBindMap = null;
    this._ownedBindGroups.forEach((bindGroup) => {
      bindGroup.destroy();
    });
    this._ownedBindGroups = null;
    this.resources = null;
    this.groups = null;
  }
  static from(options) {
    const { gpu, gl, ...rest } = options;
    let gpuProgram;
    let glProgram;
    if (gpu) {
      gpuProgram = GpuProgram.from(gpu);
    }
    if (gl) {
      glProgram = GlProgram.from(gl);
    }
    return new Shader({
      gpuProgram,
      glProgram,
      ...rest
    });
  }
}
const blendModeIds = {
  normal: 0,
  add: 1,
  multiply: 2,
  screen: 3,
  overlay: 4,
  erase: 5,
  "normal-npm": 6,
  "add-npm": 7,
  "screen-npm": 8,
  min: 9,
  max: 10
};
const BLEND = 0;
const OFFSET = 1;
const CULLING = 2;
const DEPTH_TEST = 3;
const WINDING = 4;
const DEPTH_MASK = 5;
const _State = class _State2 {
  constructor() {
    this.data = 0;
    this.blendMode = "normal";
    this.polygonOffset = 0;
    this.blend = true;
    this.depthMask = true;
  }
  /**
   * Activates blending of the computed fragment color values.
   * @default true
   */
  get blend() {
    return !!(this.data & 1 << BLEND);
  }
  set blend(value) {
    if (!!(this.data & 1 << BLEND) !== value) {
      this.data ^= 1 << BLEND;
    }
  }
  /**
   * Activates adding an offset to depth values of polygon's fragments
   * @default false
   */
  get offsets() {
    return !!(this.data & 1 << OFFSET);
  }
  set offsets(value) {
    if (!!(this.data & 1 << OFFSET) !== value) {
      this.data ^= 1 << OFFSET;
    }
  }
  /** The culling settings for this state none - No culling back - Back face culling front - Front face culling */
  set cullMode(value) {
    if (value === "none") {
      this.culling = false;
      return;
    }
    this.culling = true;
    this.clockwiseFrontFace = value === "front";
  }
  get cullMode() {
    if (!this.culling) {
      return "none";
    }
    return this.clockwiseFrontFace ? "front" : "back";
  }
  /**
   * Activates culling of polygons.
   * @default false
   */
  get culling() {
    return !!(this.data & 1 << CULLING);
  }
  set culling(value) {
    if (!!(this.data & 1 << CULLING) !== value) {
      this.data ^= 1 << CULLING;
    }
  }
  /**
   * Activates depth comparisons and updates to the depth buffer.
   * @default false
   */
  get depthTest() {
    return !!(this.data & 1 << DEPTH_TEST);
  }
  set depthTest(value) {
    if (!!(this.data & 1 << DEPTH_TEST) !== value) {
      this.data ^= 1 << DEPTH_TEST;
    }
  }
  /**
   * Enables or disables writing to the depth buffer.
   * @default true
   */
  get depthMask() {
    return !!(this.data & 1 << DEPTH_MASK);
  }
  set depthMask(value) {
    if (!!(this.data & 1 << DEPTH_MASK) !== value) {
      this.data ^= 1 << DEPTH_MASK;
    }
  }
  /**
   * Specifies whether or not front or back-facing polygons can be culled.
   * @default false
   */
  get clockwiseFrontFace() {
    return !!(this.data & 1 << WINDING);
  }
  set clockwiseFrontFace(value) {
    if (!!(this.data & 1 << WINDING) !== value) {
      this.data ^= 1 << WINDING;
    }
  }
  /**
   * The blend mode to be applied when this state is set. Apply a value of `normal` to reset the blend mode.
   * Setting this mode to anything other than NO_BLEND will automatically switch blending on.
   * @default 'normal'
   */
  get blendMode() {
    return this._blendMode;
  }
  set blendMode(value) {
    this.blend = value !== "none";
    this._blendMode = value;
    this._blendModeId = blendModeIds[value] || 0;
  }
  /**
   * The polygon offset. Setting this property to anything other than 0 will automatically enable polygon offset fill.
   * @default 0
   */
  get polygonOffset() {
    return this._polygonOffset;
  }
  set polygonOffset(value) {
    this.offsets = !!value;
    this._polygonOffset = value;
  }
  toString() {
    return `[pixi.js/core:State blendMode=${this.blendMode} clockwiseFrontFace=${this.clockwiseFrontFace} culling=${this.culling} depthMask=${this.depthMask} polygonOffset=${this.polygonOffset}]`;
  }
  /**
   * A quickly getting an instance of a State that is configured for 2d rendering.
   * @returns a new State with values set for 2d rendering
   */
  static for2d() {
    const state = new _State2();
    state.depthTest = false;
    state.blend = true;
    return state;
  }
};
_State.default2d = _State.for2d();
let State = _State;
const _Filter = class _Filter2 extends Shader {
  /**
   * @param options - The optional parameters of this filter.
   */
  constructor(options) {
    options = { ..._Filter2.defaultOptions, ...options };
    super(options);
    this.enabled = true;
    this._state = State.for2d();
    this.blendMode = options.blendMode;
    this.padding = options.padding;
    if (typeof options.antialias === "boolean") {
      this.antialias = options.antialias ? "on" : "off";
    } else {
      this.antialias = options.antialias;
    }
    this.resolution = options.resolution;
    this.blendRequired = options.blendRequired;
    this.clipToViewport = options.clipToViewport;
    this.addResource("uTexture", 0, 1);
    if (options.blendRequired) {
      this.addResource("uBackTexture", 0, 3);
    }
  }
  /**
   * Applies the filter
   * @param filterManager - The renderer to retrieve the filter from
   * @param input - The input render target.
   * @param output - The target to output to.
   * @param clearMode - Should the output be cleared before rendering to it
   */
  apply(filterManager, input, output, clearMode) {
    filterManager.applyFilter(this, input, output, clearMode);
  }
  /**
   * Get the blend mode of the filter.
   * @default "normal"
   */
  get blendMode() {
    return this._state.blendMode;
  }
  /** Sets the blend mode of the filter. */
  set blendMode(value) {
    this._state.blendMode = value;
  }
  /**
   * A short hand function to create a filter based of a vertex and fragment shader src.
   * @param options
   * @returns A shiny new PixiJS filter!
   */
  static from(options) {
    const { gpu, gl, ...rest } = options;
    let gpuProgram;
    let glProgram;
    if (gpu) {
      gpuProgram = GpuProgram.from(gpu);
    }
    if (gl) {
      glProgram = GlProgram.from(gl);
    }
    return new _Filter2({
      gpuProgram,
      glProgram,
      ...rest
    });
  }
};
_Filter.defaultOptions = {
  blendMode: "normal",
  resolution: 1,
  padding: 0,
  antialias: "off",
  blendRequired: false,
  clipToViewport: true
};
let Filter = _Filter;
const environments = [];
extensions.handleByNamedList(ExtensionType.Environment, environments);
async function loadEnvironmentExtensions(skip) {
  if (skip)
    return;
  for (let i2 = 0; i2 < environments.length; i2++) {
    const env = environments[i2];
    if (env.value.test()) {
      await env.value.load();
      return;
    }
  }
}
let unsafeEval;
function unsafeEvalSupported() {
  if (typeof unsafeEval === "boolean") {
    return unsafeEval;
  }
  try {
    const func = new Function("param1", "param2", "param3", "return param1[param2] === param3;");
    unsafeEval = func({ a: "b" }, "a", "b") === true;
  } catch (_e) {
    unsafeEval = false;
  }
  return unsafeEval;
}
function earcut$1(data, holeIndices, dim = 2) {
  const hasHoles = holeIndices && holeIndices.length;
  const outerLen = hasHoles ? holeIndices[0] * dim : data.length;
  let outerNode = linkedList(data, 0, outerLen, dim, true);
  const triangles = [];
  if (!outerNode || outerNode.next === outerNode.prev) return triangles;
  let minX, minY, invSize;
  if (hasHoles) outerNode = eliminateHoles(data, holeIndices, outerNode, dim);
  if (data.length > 80 * dim) {
    minX = data[0];
    minY = data[1];
    let maxX = minX;
    let maxY = minY;
    for (let i2 = dim; i2 < outerLen; i2 += dim) {
      const x2 = data[i2];
      const y2 = data[i2 + 1];
      if (x2 < minX) minX = x2;
      if (y2 < minY) minY = y2;
      if (x2 > maxX) maxX = x2;
      if (y2 > maxY) maxY = y2;
    }
    invSize = Math.max(maxX - minX, maxY - minY);
    invSize = invSize !== 0 ? 32767 / invSize : 0;
  }
  earcutLinked(outerNode, triangles, dim, minX, minY, invSize, 0);
  return triangles;
}
function linkedList(data, start, end, dim, clockwise) {
  let last;
  if (clockwise === signedArea(data, start, end, dim) > 0) {
    for (let i2 = start; i2 < end; i2 += dim) last = insertNode(i2 / dim | 0, data[i2], data[i2 + 1], last);
  } else {
    for (let i2 = end - dim; i2 >= start; i2 -= dim) last = insertNode(i2 / dim | 0, data[i2], data[i2 + 1], last);
  }
  if (last && equals(last, last.next)) {
    removeNode(last);
    last = last.next;
  }
  return last;
}
function filterPoints(start, end) {
  if (!start) return start;
  if (!end) end = start;
  let p2 = start, again;
  do {
    again = false;
    if (!p2.steiner && (equals(p2, p2.next) || area(p2.prev, p2, p2.next) === 0)) {
      removeNode(p2);
      p2 = end = p2.prev;
      if (p2 === p2.next) break;
      again = true;
    } else {
      p2 = p2.next;
    }
  } while (again || p2 !== end);
  return end;
}
function earcutLinked(ear, triangles, dim, minX, minY, invSize, pass) {
  if (!ear) return;
  if (!pass && invSize) indexCurve(ear, minX, minY, invSize);
  let stop = ear;
  while (ear.prev !== ear.next) {
    const prev = ear.prev;
    const next = ear.next;
    if (invSize ? isEarHashed(ear, minX, minY, invSize) : isEar(ear)) {
      triangles.push(prev.i, ear.i, next.i);
      removeNode(ear);
      ear = next.next;
      stop = next.next;
      continue;
    }
    ear = next;
    if (ear === stop) {
      if (!pass) {
        earcutLinked(filterPoints(ear), triangles, dim, minX, minY, invSize, 1);
      } else if (pass === 1) {
        ear = cureLocalIntersections(filterPoints(ear), triangles);
        earcutLinked(ear, triangles, dim, minX, minY, invSize, 2);
      } else if (pass === 2) {
        splitEarcut(ear, triangles, dim, minX, minY, invSize);
      }
      break;
    }
  }
}
function isEar(ear) {
  const a2 = ear.prev, b2 = ear, c2 = ear.next;
  if (area(a2, b2, c2) >= 0) return false;
  const ax = a2.x, bx = b2.x, cx = c2.x, ay = a2.y, by = b2.y, cy = c2.y;
  const x0 = Math.min(ax, bx, cx), y0 = Math.min(ay, by, cy), x1 = Math.max(ax, bx, cx), y1 = Math.max(ay, by, cy);
  let p2 = c2.next;
  while (p2 !== a2) {
    if (p2.x >= x0 && p2.x <= x1 && p2.y >= y0 && p2.y <= y1 && pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, p2.x, p2.y) && area(p2.prev, p2, p2.next) >= 0) return false;
    p2 = p2.next;
  }
  return true;
}
function isEarHashed(ear, minX, minY, invSize) {
  const a2 = ear.prev, b2 = ear, c2 = ear.next;
  if (area(a2, b2, c2) >= 0) return false;
  const ax = a2.x, bx = b2.x, cx = c2.x, ay = a2.y, by = b2.y, cy = c2.y;
  const x0 = Math.min(ax, bx, cx), y0 = Math.min(ay, by, cy), x1 = Math.max(ax, bx, cx), y1 = Math.max(ay, by, cy);
  const minZ = zOrder(x0, y0, minX, minY, invSize), maxZ = zOrder(x1, y1, minX, minY, invSize);
  let p2 = ear.prevZ, n2 = ear.nextZ;
  while (p2 && p2.z >= minZ && n2 && n2.z <= maxZ) {
    if (p2.x >= x0 && p2.x <= x1 && p2.y >= y0 && p2.y <= y1 && p2 !== a2 && p2 !== c2 && pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, p2.x, p2.y) && area(p2.prev, p2, p2.next) >= 0) return false;
    p2 = p2.prevZ;
    if (n2.x >= x0 && n2.x <= x1 && n2.y >= y0 && n2.y <= y1 && n2 !== a2 && n2 !== c2 && pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, n2.x, n2.y) && area(n2.prev, n2, n2.next) >= 0) return false;
    n2 = n2.nextZ;
  }
  while (p2 && p2.z >= minZ) {
    if (p2.x >= x0 && p2.x <= x1 && p2.y >= y0 && p2.y <= y1 && p2 !== a2 && p2 !== c2 && pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, p2.x, p2.y) && area(p2.prev, p2, p2.next) >= 0) return false;
    p2 = p2.prevZ;
  }
  while (n2 && n2.z <= maxZ) {
    if (n2.x >= x0 && n2.x <= x1 && n2.y >= y0 && n2.y <= y1 && n2 !== a2 && n2 !== c2 && pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, n2.x, n2.y) && area(n2.prev, n2, n2.next) >= 0) return false;
    n2 = n2.nextZ;
  }
  return true;
}
function cureLocalIntersections(start, triangles) {
  let p2 = start;
  do {
    const a2 = p2.prev, b2 = p2.next.next;
    if (!equals(a2, b2) && intersects(a2, p2, p2.next, b2) && locallyInside(a2, b2) && locallyInside(b2, a2)) {
      triangles.push(a2.i, p2.i, b2.i);
      removeNode(p2);
      removeNode(p2.next);
      p2 = start = b2;
    }
    p2 = p2.next;
  } while (p2 !== start);
  return filterPoints(p2);
}
function splitEarcut(start, triangles, dim, minX, minY, invSize) {
  let a2 = start;
  do {
    let b2 = a2.next.next;
    while (b2 !== a2.prev) {
      if (a2.i !== b2.i && isValidDiagonal(a2, b2)) {
        let c2 = splitPolygon(a2, b2);
        a2 = filterPoints(a2, a2.next);
        c2 = filterPoints(c2, c2.next);
        earcutLinked(a2, triangles, dim, minX, minY, invSize, 0);
        earcutLinked(c2, triangles, dim, minX, minY, invSize, 0);
        return;
      }
      b2 = b2.next;
    }
    a2 = a2.next;
  } while (a2 !== start);
}
function eliminateHoles(data, holeIndices, outerNode, dim) {
  const queue = [];
  for (let i2 = 0, len = holeIndices.length; i2 < len; i2++) {
    const start = holeIndices[i2] * dim;
    const end = i2 < len - 1 ? holeIndices[i2 + 1] * dim : data.length;
    const list = linkedList(data, start, end, dim, false);
    if (list === list.next) list.steiner = true;
    queue.push(getLeftmost(list));
  }
  queue.sort(compareXYSlope);
  for (let i2 = 0; i2 < queue.length; i2++) {
    outerNode = eliminateHole(queue[i2], outerNode);
  }
  return outerNode;
}
function compareXYSlope(a2, b2) {
  let result = a2.x - b2.x;
  if (result === 0) {
    result = a2.y - b2.y;
    if (result === 0) {
      const aSlope = (a2.next.y - a2.y) / (a2.next.x - a2.x);
      const bSlope = (b2.next.y - b2.y) / (b2.next.x - b2.x);
      result = aSlope - bSlope;
    }
  }
  return result;
}
function eliminateHole(hole, outerNode) {
  const bridge = findHoleBridge(hole, outerNode);
  if (!bridge) {
    return outerNode;
  }
  const bridgeReverse = splitPolygon(bridge, hole);
  filterPoints(bridgeReverse, bridgeReverse.next);
  return filterPoints(bridge, bridge.next);
}
function findHoleBridge(hole, outerNode) {
  let p2 = outerNode;
  const hx = hole.x;
  const hy = hole.y;
  let qx = -Infinity;
  let m2;
  if (equals(hole, p2)) return p2;
  do {
    if (equals(hole, p2.next)) return p2.next;
    else if (hy <= p2.y && hy >= p2.next.y && p2.next.y !== p2.y) {
      const x2 = p2.x + (hy - p2.y) * (p2.next.x - p2.x) / (p2.next.y - p2.y);
      if (x2 <= hx && x2 > qx) {
        qx = x2;
        m2 = p2.x < p2.next.x ? p2 : p2.next;
        if (x2 === hx) return m2;
      }
    }
    p2 = p2.next;
  } while (p2 !== outerNode);
  if (!m2) return null;
  const stop = m2;
  const mx = m2.x;
  const my = m2.y;
  let tanMin = Infinity;
  p2 = m2;
  do {
    if (hx >= p2.x && p2.x >= mx && hx !== p2.x && pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p2.x, p2.y)) {
      const tan = Math.abs(hy - p2.y) / (hx - p2.x);
      if (locallyInside(p2, hole) && (tan < tanMin || tan === tanMin && (p2.x > m2.x || p2.x === m2.x && sectorContainsSector(m2, p2)))) {
        m2 = p2;
        tanMin = tan;
      }
    }
    p2 = p2.next;
  } while (p2 !== stop);
  return m2;
}
function sectorContainsSector(m2, p2) {
  return area(m2.prev, m2, p2.prev) < 0 && area(p2.next, m2, m2.next) < 0;
}
function indexCurve(start, minX, minY, invSize) {
  let p2 = start;
  do {
    if (p2.z === 0) p2.z = zOrder(p2.x, p2.y, minX, minY, invSize);
    p2.prevZ = p2.prev;
    p2.nextZ = p2.next;
    p2 = p2.next;
  } while (p2 !== start);
  p2.prevZ.nextZ = null;
  p2.prevZ = null;
  sortLinked(p2);
}
function sortLinked(list) {
  let numMerges;
  let inSize = 1;
  do {
    let p2 = list;
    let e2;
    list = null;
    let tail = null;
    numMerges = 0;
    while (p2) {
      numMerges++;
      let q = p2;
      let pSize = 0;
      for (let i2 = 0; i2 < inSize; i2++) {
        pSize++;
        q = q.nextZ;
        if (!q) break;
      }
      let qSize = inSize;
      while (pSize > 0 || qSize > 0 && q) {
        if (pSize !== 0 && (qSize === 0 || !q || p2.z <= q.z)) {
          e2 = p2;
          p2 = p2.nextZ;
          pSize--;
        } else {
          e2 = q;
          q = q.nextZ;
          qSize--;
        }
        if (tail) tail.nextZ = e2;
        else list = e2;
        e2.prevZ = tail;
        tail = e2;
      }
      p2 = q;
    }
    tail.nextZ = null;
    inSize *= 2;
  } while (numMerges > 1);
  return list;
}
function zOrder(x2, y2, minX, minY, invSize) {
  x2 = (x2 - minX) * invSize | 0;
  y2 = (y2 - minY) * invSize | 0;
  x2 = (x2 | x2 << 8) & 16711935;
  x2 = (x2 | x2 << 4) & 252645135;
  x2 = (x2 | x2 << 2) & 858993459;
  x2 = (x2 | x2 << 1) & 1431655765;
  y2 = (y2 | y2 << 8) & 16711935;
  y2 = (y2 | y2 << 4) & 252645135;
  y2 = (y2 | y2 << 2) & 858993459;
  y2 = (y2 | y2 << 1) & 1431655765;
  return x2 | y2 << 1;
}
function getLeftmost(start) {
  let p2 = start, leftmost = start;
  do {
    if (p2.x < leftmost.x || p2.x === leftmost.x && p2.y < leftmost.y) leftmost = p2;
    p2 = p2.next;
  } while (p2 !== start);
  return leftmost;
}
function pointInTriangle(ax, ay, bx, by, cx, cy, px, py) {
  return (cx - px) * (ay - py) >= (ax - px) * (cy - py) && (ax - px) * (by - py) >= (bx - px) * (ay - py) && (bx - px) * (cy - py) >= (cx - px) * (by - py);
}
function pointInTriangleExceptFirst(ax, ay, bx, by, cx, cy, px, py) {
  return !(ax === px && ay === py) && pointInTriangle(ax, ay, bx, by, cx, cy, px, py);
}
function isValidDiagonal(a2, b2) {
  return a2.next.i !== b2.i && a2.prev.i !== b2.i && !intersectsPolygon(a2, b2) && // doesn't intersect other edges
  (locallyInside(a2, b2) && locallyInside(b2, a2) && middleInside(a2, b2) && // locally visible
  (area(a2.prev, a2, b2.prev) || area(a2, b2.prev, b2)) || // does not create opposite-facing sectors
  equals(a2, b2) && area(a2.prev, a2, a2.next) > 0 && area(b2.prev, b2, b2.next) > 0);
}
function area(p2, q, r2) {
  return (q.y - p2.y) * (r2.x - q.x) - (q.x - p2.x) * (r2.y - q.y);
}
function equals(p1, p2) {
  return p1.x === p2.x && p1.y === p2.y;
}
function intersects(p1, q1, p2, q2) {
  const o1 = sign(area(p1, q1, p2));
  const o2 = sign(area(p1, q1, q2));
  const o3 = sign(area(p2, q2, p1));
  const o4 = sign(area(p2, q2, q1));
  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(p1, p2, q1)) return true;
  if (o2 === 0 && onSegment(p1, q2, q1)) return true;
  if (o3 === 0 && onSegment(p2, p1, q2)) return true;
  if (o4 === 0 && onSegment(p2, q1, q2)) return true;
  return false;
}
function onSegment(p2, q, r2) {
  return q.x <= Math.max(p2.x, r2.x) && q.x >= Math.min(p2.x, r2.x) && q.y <= Math.max(p2.y, r2.y) && q.y >= Math.min(p2.y, r2.y);
}
function sign(num) {
  return num > 0 ? 1 : num < 0 ? -1 : 0;
}
function intersectsPolygon(a2, b2) {
  let p2 = a2;
  do {
    if (p2.i !== a2.i && p2.next.i !== a2.i && p2.i !== b2.i && p2.next.i !== b2.i && intersects(p2, p2.next, a2, b2)) return true;
    p2 = p2.next;
  } while (p2 !== a2);
  return false;
}
function locallyInside(a2, b2) {
  return area(a2.prev, a2, a2.next) < 0 ? area(a2, b2, a2.next) >= 0 && area(a2, a2.prev, b2) >= 0 : area(a2, b2, a2.prev) < 0 || area(a2, a2.next, b2) < 0;
}
function middleInside(a2, b2) {
  let p2 = a2;
  let inside = false;
  const px = (a2.x + b2.x) / 2;
  const py = (a2.y + b2.y) / 2;
  do {
    if (p2.y > py !== p2.next.y > py && p2.next.y !== p2.y && px < (p2.next.x - p2.x) * (py - p2.y) / (p2.next.y - p2.y) + p2.x)
      inside = !inside;
    p2 = p2.next;
  } while (p2 !== a2);
  return inside;
}
function splitPolygon(a2, b2) {
  const a22 = createNode(a2.i, a2.x, a2.y), b22 = createNode(b2.i, b2.x, b2.y), an = a2.next, bp = b2.prev;
  a2.next = b2;
  b2.prev = a2;
  a22.next = an;
  an.prev = a22;
  b22.next = a22;
  a22.prev = b22;
  bp.next = b22;
  b22.prev = bp;
  return b22;
}
function insertNode(i2, x2, y2, last) {
  const p2 = createNode(i2, x2, y2);
  if (!last) {
    p2.prev = p2;
    p2.next = p2;
  } else {
    p2.next = last.next;
    p2.prev = last;
    last.next.prev = p2;
    last.next = p2;
  }
  return p2;
}
function removeNode(p2) {
  p2.next.prev = p2.prev;
  p2.prev.next = p2.next;
  if (p2.prevZ) p2.prevZ.nextZ = p2.nextZ;
  if (p2.nextZ) p2.nextZ.prevZ = p2.prevZ;
}
function createNode(i2, x2, y2) {
  return {
    i: i2,
    // vertex index in coordinates array
    x: x2,
    y: y2,
    // vertex coordinates
    prev: null,
    // previous and next vertex nodes in a polygon ring
    next: null,
    z: 0,
    // z-order curve value
    prevZ: null,
    // previous and next nodes in z-order
    nextZ: null,
    steiner: false
    // indicates whether this is a steiner point
  };
}
function signedArea(data, start, end, dim) {
  let sum = 0;
  for (let i2 = start, j2 = end - dim; i2 < end; i2 += dim) {
    sum += (data[j2] - data[i2]) * (data[i2 + 1] + data[j2 + 1]);
    j2 = i2;
  }
  return sum;
}
const earcut = earcut$1.default || earcut$1;
var CLEAR = /* @__PURE__ */ ((CLEAR2) => {
  CLEAR2[CLEAR2["NONE"] = 0] = "NONE";
  CLEAR2[CLEAR2["COLOR"] = 16384] = "COLOR";
  CLEAR2[CLEAR2["STENCIL"] = 1024] = "STENCIL";
  CLEAR2[CLEAR2["DEPTH"] = 256] = "DEPTH";
  CLEAR2[CLEAR2["COLOR_DEPTH"] = 16640] = "COLOR_DEPTH";
  CLEAR2[CLEAR2["COLOR_STENCIL"] = 17408] = "COLOR_STENCIL";
  CLEAR2[CLEAR2["DEPTH_STENCIL"] = 1280] = "DEPTH_STENCIL";
  CLEAR2[CLEAR2["ALL"] = 17664] = "ALL";
  return CLEAR2;
})(CLEAR || {});
class SystemRunner {
  /**
   * @param name - The function name that will be executed on the listeners added to this Runner.
   */
  constructor(name) {
    this.items = [];
    this._name = name;
  }
  /* jsdoc/check-param-names */
  /**
   * Dispatch/Broadcast Runner to all listeners added to the queue.
   * @param {...any} params - (optional) parameters to pass to each listener
   */
  /* jsdoc/check-param-names */
  emit(a0, a1, a2, a3, a4, a5, a6, a7) {
    const { name, items } = this;
    for (let i2 = 0, len = items.length; i2 < len; i2++) {
      items[i2][name](a0, a1, a2, a3, a4, a5, a6, a7);
    }
    return this;
  }
  /**
   * Add a listener to the Runner
   *
   * Runners do not need to have scope or functions passed to them.
   * All that is required is to pass the listening object and ensure that it has contains a function that has the same name
   * as the name provided to the Runner when it was created.
   *
   * Eg A listener passed to this Runner will require a 'complete' function.
   *
   * ```ts
   * import { Runner } from 'pixi.js';
   *
   * const complete = new Runner('complete');
   * ```
   *
   * The scope used will be the object itself.
   * @param {any} item - The object that will be listening.
   */
  add(item) {
    if (item[this._name]) {
      this.remove(item);
      this.items.push(item);
    }
    return this;
  }
  /**
   * Remove a single listener from the dispatch queue.
   * @param {any} item - The listener that you would like to remove.
   */
  remove(item) {
    const index = this.items.indexOf(item);
    if (index !== -1) {
      this.items.splice(index, 1);
    }
    return this;
  }
  /**
   * Check to see if the listener is already in the Runner
   * @param {any} item - The listener that you would like to check.
   */
  contains(item) {
    return this.items.indexOf(item) !== -1;
  }
  /** Remove all listeners from the Runner */
  removeAll() {
    this.items.length = 0;
    return this;
  }
  /** Remove all references, don't use after this. */
  destroy() {
    this.removeAll();
    this.items = null;
    this._name = null;
  }
  /**
   * `true` if there are no this Runner contains no listeners
   * @readonly
   */
  get empty() {
    return this.items.length === 0;
  }
  /**
   * The name of the runner.
   * @readonly
   */
  get name() {
    return this._name;
  }
}
const defaultRunners = [
  "init",
  "destroy",
  "contextChange",
  "resolutionChange",
  "resetState",
  "renderEnd",
  "renderStart",
  "render",
  "update",
  "postrender",
  "prerender"
];
const _AbstractRenderer = class _AbstractRenderer2 extends EventEmitter {
  /**
   * Set up a system with a collection of SystemClasses and runners.
   * Systems are attached dynamically to this class when added.
   * @param config - the config for the system manager
   */
  constructor(config) {
    super();
    this.tick = 0;
    this.uid = uid("renderer");
    this.runners = /* @__PURE__ */ Object.create(null);
    this.renderPipes = /* @__PURE__ */ Object.create(null);
    this._initOptions = {};
    this._systemsHash = /* @__PURE__ */ Object.create(null);
    this.type = config.type;
    this.name = config.name;
    this.config = config;
    const combinedRunners = [...defaultRunners, ...this.config.runners ?? []];
    this._addRunners(...combinedRunners);
    this._unsafeEvalCheck();
  }
  /**
   * Initialize the renderer.
   * @param options - The options to use to create the renderer.
   */
  async init(options = {}) {
    const skip = options.skipExtensionImports === true ? true : options.manageImports === false;
    await loadEnvironmentExtensions(skip);
    this._addSystems(this.config.systems);
    this._addPipes(this.config.renderPipes, this.config.renderPipeAdaptors);
    for (const systemName in this._systemsHash) {
      const system = this._systemsHash[systemName];
      const defaultSystemOptions = system.constructor.defaultOptions;
      options = { ...defaultSystemOptions, ...options };
    }
    options = { ..._AbstractRenderer2.defaultOptions, ...options };
    this._roundPixels = options.roundPixels ? 1 : 0;
    for (let i2 = 0; i2 < this.runners.init.items.length; i2++) {
      await this.runners.init.items[i2].init(options);
    }
    this._initOptions = options;
  }
  render(args, deprecated) {
    this.tick++;
    let options = args;
    if (options instanceof Container) {
      options = { container: options };
      if (deprecated) {
        deprecation(v8_0_0, "passing a second argument is deprecated, please use render options instead");
        options.target = deprecated.renderTexture;
      }
    }
    options.target || (options.target = this.view.renderTarget);
    if (options.target === this.view.renderTarget) {
      this._lastObjectRendered = options.container;
      options.clearColor ?? (options.clearColor = this.background.colorRgba);
      options.clear ?? (options.clear = this.background.clearBeforeRender);
    }
    if (options.clearColor) {
      const isRGBAArray = Array.isArray(options.clearColor) && options.clearColor.length === 4;
      options.clearColor = isRGBAArray ? options.clearColor : Color.shared.setValue(options.clearColor).toArray();
    }
    if (!options.transform) {
      options.container.updateLocalTransform();
      options.transform = options.container.localTransform;
    }
    if (!options.container.visible) {
      return;
    }
    options.container.enableRenderGroup();
    this.runners.prerender.emit(options);
    this.runners.renderStart.emit(options);
    this.runners.render.emit(options);
    this.runners.renderEnd.emit(options);
    this.runners.postrender.emit(options);
  }
  /**
   * Resizes the WebGL view to the specified width and height.
   * @param desiredScreenWidth - The desired width of the screen.
   * @param desiredScreenHeight - The desired height of the screen.
   * @param resolution - The resolution / device pixel ratio of the renderer.
   */
  resize(desiredScreenWidth, desiredScreenHeight, resolution) {
    const previousResolution = this.view.resolution;
    this.view.resize(desiredScreenWidth, desiredScreenHeight, resolution);
    this.emit("resize", this.view.screen.width, this.view.screen.height, this.view.resolution);
    if (resolution !== void 0 && resolution !== previousResolution) {
      this.runners.resolutionChange.emit(resolution);
    }
  }
  /**
   * Clears the render target.
   * @param options - The options to use when clearing the render target.
   * @param options.target - The render target to clear.
   * @param options.clearColor - The color to clear with.
   * @param options.clear - The clear mode to use.
   * @advanced
   */
  clear(options = {}) {
    const renderer = this;
    options.target || (options.target = renderer.renderTarget.renderTarget);
    options.clearColor || (options.clearColor = this.background.colorRgba);
    options.clear ?? (options.clear = CLEAR.ALL);
    const { clear, clearColor, target } = options;
    Color.shared.setValue(clearColor ?? this.background.colorRgba);
    renderer.renderTarget.clear(target, clear, Color.shared.toArray());
  }
  /** The resolution / device pixel ratio of the renderer. */
  get resolution() {
    return this.view.resolution;
  }
  set resolution(value) {
    this.view.resolution = value;
    this.runners.resolutionChange.emit(value);
  }
  /**
   * Same as view.width, actual number of pixels in the canvas by horizontal.
   * @type {number}
   * @readonly
   * @default 800
   */
  get width() {
    return this.view.texture.frame.width;
  }
  /**
   * Same as view.height, actual number of pixels in the canvas by vertical.
   * @default 600
   */
  get height() {
    return this.view.texture.frame.height;
  }
  // NOTE: this was `view` in v7
  /**
   * The canvas element that everything is drawn to.
   * @type {environment.ICanvas}
   */
  get canvas() {
    return this.view.canvas;
  }
  /**
   * the last object rendered by the renderer. Useful for other plugins like interaction managers
   * @readonly
   */
  get lastObjectRendered() {
    return this._lastObjectRendered;
  }
  /**
   * Flag if we are rendering to the screen vs renderTexture
   * @readonly
   * @default true
   */
  get renderingToScreen() {
    const renderer = this;
    return renderer.renderTarget.renderingToScreen;
  }
  /**
   * Measurements of the screen. (0, 0, screenWidth, screenHeight).
   *
   * Its safe to use as filterArea or hitArea for the whole stage.
   */
  get screen() {
    return this.view.screen;
  }
  /**
   * Create a bunch of runners based of a collection of ids
   * @param runnerIds - the runner ids to add
   */
  _addRunners(...runnerIds) {
    runnerIds.forEach((runnerId) => {
      this.runners[runnerId] = new SystemRunner(runnerId);
    });
  }
  _addSystems(systems) {
    let i2;
    for (i2 in systems) {
      const val = systems[i2];
      this._addSystem(val.value, val.name);
    }
  }
  /**
   * Add a new system to the renderer.
   * @param ClassRef - Class reference
   * @param name - Property name for system, if not specified
   *        will use a static `name` property on the class itself. This
   *        name will be assigned as s property on the Renderer so make
   *        sure it doesn't collide with properties on Renderer.
   * @returns Return instance of renderer
   */
  _addSystem(ClassRef, name) {
    const system = new ClassRef(this);
    if (this[name]) {
      throw new Error(`Whoops! The name "${name}" is already in use`);
    }
    this[name] = system;
    this._systemsHash[name] = system;
    for (const i2 in this.runners) {
      this.runners[i2].add(system);
    }
    return this;
  }
  _addPipes(pipes, pipeAdaptors) {
    const adaptors = pipeAdaptors.reduce((acc, adaptor) => {
      acc[adaptor.name] = adaptor.value;
      return acc;
    }, {});
    pipes.forEach((pipe) => {
      const PipeClass = pipe.value;
      const name = pipe.name;
      const Adaptor = adaptors[name];
      this.renderPipes[name] = new PipeClass(
        this,
        Adaptor ? new Adaptor() : null
      );
      this.runners.destroy.add(this.renderPipes[name]);
    });
  }
  destroy(options = false) {
    this.runners.destroy.items.reverse();
    this.runners.destroy.emit(options);
    if (options === true || typeof options === "object" && options.releaseGlobalResources) {
      GlobalResourceRegistry.release();
    }
    Object.values(this.runners).forEach((runner) => {
      runner.destroy();
    });
    this._systemsHash = null;
    this.renderPipes = null;
  }
  /**
   * Generate a texture from a container.
   * @param options - options or container target to use when generating the texture
   * @returns a texture
   */
  generateTexture(options) {
    return this.textureGenerator.generateTexture(options);
  }
  /**
   * Whether the renderer will round coordinates to whole pixels when rendering.
   * Can be overridden on a per scene item basis.
   */
  get roundPixels() {
    return !!this._roundPixels;
  }
  /**
   * Overridable function by `pixi.js/unsafe-eval` to silence
   * throwing an error if platform doesn't support unsafe-evals.
   * @private
   * @ignore
   */
  _unsafeEvalCheck() {
    if (!unsafeEvalSupported()) {
      throw new Error("Current environment does not allow unsafe-eval, please use pixi.js/unsafe-eval module to enable support.");
    }
  }
  /**
   * Resets the rendering state of the renderer.
   * This is useful when you want to use the WebGL context directly and need to ensure PixiJS's internal state
   * stays synchronized. When modifying the WebGL context state externally, calling this method before the next Pixi
   * render will reset all internal caches and ensure it executes correctly.
   *
   * This is particularly useful when combining PixiJS with other rendering engines like Three.js:
   * ```js
   * // Reset Three.js state
   * threeRenderer.resetState();
   *
   * // Render a Three.js scene
   * threeRenderer.render(threeScene, threeCamera);
   *
   * // Reset PixiJS state since Three.js modified the WebGL context
   * pixiRenderer.resetState();
   *
   * // Now render Pixi content
   * pixiRenderer.render(pixiScene);
   * ```
   * @advanced
   */
  resetState() {
    this.runners.resetState.emit();
  }
};
_AbstractRenderer.defaultOptions = {
  /**
   * Default resolution / device pixel ratio of the renderer.
   * @default 1
   */
  resolution: 1,
  /**
   * Should the `failIfMajorPerformanceCaveat` flag be enabled as a context option used in the `isWebGLSupported`
   * function. If set to true, a WebGL renderer can fail to be created if the browser thinks there could be
   * performance issues when using WebGL.
   *
   * In PixiJS v6 this has changed from true to false by default, to allow WebGL to work in as many
   * scenarios as possible. However, some users may have a poor experience, for example, if a user has a gpu or
   * driver version blacklisted by the
   * browser.
   *
   * If your application requires high performance rendering, you may wish to set this to false.
   * We recommend one of two options if you decide to set this flag to false:
   *
   * 1: Use the Canvas renderer as a fallback in case high performance WebGL is
   *    not supported.
   *
   * 2: Call `isWebGLSupported` (which if found in the utils package) in your code before attempting to create a
   *    PixiJS renderer, and show an error message to the user if the function returns false, explaining that their
   *    device & browser combination does not support high performance WebGL.
   *    This is a much better strategy than trying to create a PixiJS renderer and finding it then fails.
   * @default false
   */
  failIfMajorPerformanceCaveat: false,
  /**
   * Should round pixels be forced when rendering?
   * @default false
   */
  roundPixels: false
};
let AbstractRenderer = _AbstractRenderer;
let _isWebGLSupported;
function isWebGLSupported(failIfMajorPerformanceCaveat) {
  if (_isWebGLSupported !== void 0)
    return _isWebGLSupported;
  _isWebGLSupported = (() => {
    const contextOptions = {
      stencil: true,
      failIfMajorPerformanceCaveat: failIfMajorPerformanceCaveat ?? AbstractRenderer.defaultOptions.failIfMajorPerformanceCaveat
    };
    try {
      if (!DOMAdapter.get().getWebGLRenderingContext()) {
        return false;
      }
      const canvas = DOMAdapter.get().createCanvas();
      let gl = canvas.getContext("webgl", contextOptions);
      const success = !!gl?.getContextAttributes()?.stencil;
      if (gl) {
        const loseContext = gl.getExtension("WEBGL_lose_context");
        if (loseContext) {
          loseContext.loseContext();
        }
      }
      gl = null;
      return success;
    } catch (_e) {
      return false;
    }
  })();
  return _isWebGLSupported;
}
let _isWebGPUSupported;
async function isWebGPUSupported(options = {}) {
  if (_isWebGPUSupported !== void 0)
    return _isWebGPUSupported;
  _isWebGPUSupported = await (async () => {
    const gpu = DOMAdapter.get().getNavigator().gpu;
    if (!gpu) {
      return false;
    }
    try {
      const adapter = await gpu.requestAdapter(options);
      await adapter.requestDevice();
      return true;
    } catch (_e) {
      return false;
    }
  })();
  return _isWebGPUSupported;
}
const renderPriority = ["webgl", "webgpu", "canvas"];
async function autoDetectRenderer(options) {
  let preferredOrder = [];
  if (options.preference) {
    preferredOrder.push(options.preference);
    renderPriority.forEach((item) => {
      if (item !== options.preference) {
        preferredOrder.push(item);
      }
    });
  } else {
    preferredOrder = renderPriority.slice();
  }
  let RendererClass;
  let finalOptions = {};
  for (let i2 = 0; i2 < preferredOrder.length; i2++) {
    const rendererType = preferredOrder[i2];
    if (rendererType === "webgpu" && await isWebGPUSupported()) {
      const { WebGPURenderer } = await import('./WebGPURenderer-DsoZyEPc.js');
      RendererClass = WebGPURenderer;
      finalOptions = { ...options, ...options.webgpu };
      break;
    } else if (rendererType === "webgl" && isWebGLSupported(
      options.failIfMajorPerformanceCaveat ?? AbstractRenderer.defaultOptions.failIfMajorPerformanceCaveat
    )) {
      const { WebGLRenderer } = await import('./WebGLRenderer-BvMNpEN0.js');
      RendererClass = WebGLRenderer;
      finalOptions = { ...options, ...options.webgl };
      break;
    } else if (rendererType === "canvas") {
      finalOptions = { ...options };
      throw new Error("CanvasRenderer is not yet implemented");
    }
  }
  delete finalOptions.webgpu;
  delete finalOptions.webgl;
  if (!RendererClass) {
    throw new Error("No available renderer for the current environment");
  }
  const renderer = new RendererClass();
  await renderer.init(finalOptions);
  return renderer;
}
const VERSION = "8.15.0";
class ApplicationInitHook {
  static init() {
    globalThis.__PIXI_APP_INIT__?.(this, VERSION);
  }
  static destroy() {
  }
}
ApplicationInitHook.extension = ExtensionType.Application;
class RendererInitHook {
  constructor(renderer) {
    this._renderer = renderer;
  }
  init() {
    globalThis.__PIXI_RENDERER_INIT__?.(this._renderer, VERSION);
  }
  destroy() {
    this._renderer = null;
  }
}
RendererInitHook.extension = {
  type: [
    ExtensionType.WebGLSystem,
    ExtensionType.WebGPUSystem
  ],
  name: "initHook",
  priority: -10
};
const _Application = class _Application2 {
  constructor(...args) {
    this.stage = new Container();
    if (args[0] !== void 0) {
      deprecation(v8_0_0, "Application constructor options are deprecated, please use Application.init() instead.");
    }
  }
  /**
   * Initializes the PixiJS application with the specified options.
   *
   * This method must be called after creating a new Application instance.
   * @param options - Configuration options for the application and renderer
   * @returns A promise that resolves when initialization is complete
   * @example
   * ```js
   * const app = new Application();
   *
   * // Initialize with custom options
   * await app.init({
   *     width: 800,
   *     height: 600,
   *     backgroundColor: 0x1099bb,
   *     preference: 'webgl', // or 'webgpu'
   * });
   * ```
   */
  async init(options) {
    options = { ...options };
    this.stage || (this.stage = new Container());
    this.renderer = await autoDetectRenderer(options);
    _Application2._plugins.forEach((plugin) => {
      plugin.init.call(this, options);
    });
  }
  /**
   * Renders the current stage to the screen.
   *
   * When using the default setup with {@link TickerPlugin} (enabled by default), you typically don't need to call
   * this method directly as rendering is handled automatically.
   *
   * Only use this method if you've disabled the {@link TickerPlugin} or need custom
   * render timing control.
   * @example
   * ```js
   * // Example 1: Default setup (TickerPlugin handles rendering)
   * const app = new Application();
   * await app.init();
   * // No need to call render() - TickerPlugin handles it
   *
   * // Example 2: Custom rendering loop (if TickerPlugin is disabled)
   * const app = new Application();
   * await app.init({ autoStart: false }); // Disable automatic rendering
   *
   * function animate() {
   *     app.render();
   *     requestAnimationFrame(animate);
   * }
   * animate();
   * ```
   */
  render() {
    this.renderer.render({ container: this.stage });
  }
  /**
   * Reference to the renderer's canvas element. This is the HTML element
   * that displays your application's graphics.
   * @readonly
   * @type {HTMLCanvasElement}
   * @example
   * ```js
   * // Create a new application
   * const app = new Application();
   * // Initialize the application
   * await app.init({...});
   * // Add canvas to the page
   * document.body.appendChild(app.canvas);
   *
   * // Access the canvas directly
   * console.log(app.canvas); // HTMLCanvasElement
   * ```
   */
  get canvas() {
    return this.renderer.canvas;
  }
  /**
   * Reference to the renderer's canvas element.
   * @type {HTMLCanvasElement}
   * @deprecated since 8.0.0
   * @see {@link Application#canvas}
   */
  get view() {
    deprecation(v8_0_0, "Application.view is deprecated, please use Application.canvas instead.");
    return this.renderer.canvas;
  }
  /**
   * Reference to the renderer's screen rectangle. This represents the visible area of your application.
   *
   * It's commonly used for:
   * - Setting filter areas for full-screen effects
   * - Defining hit areas for screen-wide interaction
   * - Determining the visible bounds of your application
   * @readonly
   * @example
   * ```js
   * // Use as filter area for a full-screen effect
   * const blurFilter = new BlurFilter();
   * sprite.filterArea = app.screen;
   *
   * // Use as hit area for screen-wide interaction
   * const screenSprite = new Sprite();
   * screenSprite.hitArea = app.screen;
   *
   * // Get screen dimensions
   * console.log(app.screen.width, app.screen.height);
   * ```
   * @see {@link Rectangle} For all available properties and methods
   */
  get screen() {
    return this.renderer.screen;
  }
  /**
   * Destroys the application and all of its resources.
   *
   * This method should be called when you want to completely
   * clean up the application and free all associated memory.
   * @param rendererDestroyOptions - Options for destroying the renderer:
   *  - `false` or `undefined`: Preserves the canvas element (default)
   *  - `true`: Removes the canvas element
   *  - `{ removeView: boolean }`: Object with removeView property to control canvas removal
   * @param options - Options for destroying the application:
   *  - `false` or `undefined`: Basic cleanup (default)
   *  - `true`: Complete cleanup including children
   *  - Detailed options object:
   *    - `children`: Remove children
   *    - `texture`: Destroy textures
   *    - `textureSource`: Destroy texture sources
   *    - `context`: Destroy WebGL context
   * @example
   * ```js
   * // Basic cleanup
   * app.destroy();
   *
   * // Remove canvas and do complete cleanup
   * app.destroy(true, true);
   *
   * // Remove canvas with explicit options
   * app.destroy({ removeView: true }, true);
   *
   * // Detailed cleanup with specific options
   * app.destroy(
   *     { removeView: true },
   *     {
   *         children: true,
   *         texture: true,
   *         textureSource: true,
   *         context: true
   *     }
   * );
   * ```
   * > [!WARNING] After calling destroy, the application instance should no longer be used.
   * > All properties will be null and further operations will throw errors.
   */
  destroy(rendererDestroyOptions = false, options = false) {
    const plugins = _Application2._plugins.slice(0);
    plugins.reverse();
    plugins.forEach((plugin) => {
      plugin.destroy.call(this);
    });
    this.stage.destroy(options);
    this.stage = null;
    this.renderer.destroy(rendererDestroyOptions);
    this.renderer = null;
  }
};
_Application._plugins = [];
let Application = _Application;
extensions.handleByList(ExtensionType.Application, Application._plugins);
extensions.add(ApplicationInitHook);
class AbstractBitmapFont extends EventEmitter {
  constructor() {
    super(...arguments);
    this.chars = /* @__PURE__ */ Object.create(null);
    this.lineHeight = 0;
    this.fontFamily = "";
    this.fontMetrics = { fontSize: 0, ascent: 0, descent: 0 };
    this.baseLineOffset = 0;
    this.distanceField = { type: "none", range: 0 };
    this.pages = [];
    this.applyFillAsTint = true;
    this.baseMeasurementFontSize = 100;
    this.baseRenderedFontSize = 100;
  }
  /**
   * The name of the font face.
   * @deprecated since 8.0.0 Use `fontFamily` instead.
   */
  get font() {
    deprecation(v8_0_0, "BitmapFont.font is deprecated, please use BitmapFont.fontFamily instead.");
    return this.fontFamily;
  }
  /**
   * The map of base page textures (i.e., sheets of glyphs).
   * @deprecated since 8.0.0 Use `pages` instead.
   */
  get pageTextures() {
    deprecation(v8_0_0, "BitmapFont.pageTextures is deprecated, please use BitmapFont.pages instead.");
    return this.pages;
  }
  /**
   * The size of the font face in pixels.
   * @deprecated since 8.0.0 Use `fontMetrics.fontSize` instead.
   */
  get size() {
    deprecation(v8_0_0, "BitmapFont.size is deprecated, please use BitmapFont.fontMetrics.fontSize instead.");
    return this.fontMetrics.fontSize;
  }
  /**
   * The kind of distance field for this font or "none".
   * @deprecated since 8.0.0 Use `distanceField.type` instead.
   */
  get distanceFieldRange() {
    deprecation(v8_0_0, "BitmapFont.distanceFieldRange is deprecated, please use BitmapFont.distanceField.range instead.");
    return this.distanceField.range;
  }
  /**
   * The range of the distance field in pixels.
   * @deprecated since 8.0.0 Use `distanceField.range` instead.
   */
  get distanceFieldType() {
    deprecation(v8_0_0, "BitmapFont.distanceFieldType is deprecated, please use BitmapFont.distanceField.type instead.");
    return this.distanceField.type;
  }
  destroy(destroyTextures = false) {
    this.emit("destroy", this);
    this.removeAllListeners();
    for (const i2 in this.chars) {
      this.chars[i2].texture?.destroy();
    }
    this.chars = null;
    if (destroyTextures) {
      this.pages.forEach((page) => page.texture.destroy(true));
      this.pages = null;
    }
  }
}
class LRU {
  /**
   * Creates a new LRU cache instance.
   * Note: Constructor does not validate parameters. Use lru() factory function for parameter validation.
   *
   * @constructor
   * @param {number} [max=0] - Maximum number of items to store. 0 means unlimited.
   * @param {number} [ttl=0] - Time to live in milliseconds. 0 means no expiration.
   * @param {boolean} [resetTtl=false] - Whether to reset TTL when accessing existing items via get().
   * @example
   * const cache = new LRU(1000, 60000, true); // 1000 items, 1 minute TTL, reset on access
   * @see {@link lru} For parameter validation
   * @since 1.0.0
   */
  constructor(max = 0, ttl = 0, resetTtl = false) {
    this.first = null;
    this.items = /* @__PURE__ */ Object.create(null);
    this.last = null;
    this.max = max;
    this.resetTtl = resetTtl;
    this.size = 0;
    this.ttl = ttl;
  }
  /**
   * Removes all items from the cache.
   *
   * @method clear
   * @memberof LRU
   * @returns {LRU} The LRU instance for method chaining.
   * @example
   * cache.clear();
   * console.log(cache.size); // 0
   * @since 1.0.0
   */
  clear() {
    this.first = null;
    this.items = /* @__PURE__ */ Object.create(null);
    this.last = null;
    this.size = 0;
    return this;
  }
  /**
   * Removes an item from the cache by key.
   *
   * @method delete
   * @memberof LRU
   * @param {string} key - The key of the item to delete.
   * @returns {LRU} The LRU instance for method chaining.
   * @example
   * cache.set('key1', 'value1');
   * cache.delete('key1');
   * console.log(cache.has('key1')); // false
   * @see {@link LRU#has}
   * @see {@link LRU#clear}
   * @since 1.0.0
   */
  delete(key) {
    if (this.has(key)) {
      const item = this.items[key];
      delete this.items[key];
      this.size--;
      if (item.prev !== null) {
        item.prev.next = item.next;
      }
      if (item.next !== null) {
        item.next.prev = item.prev;
      }
      if (this.first === item) {
        this.first = item.next;
      }
      if (this.last === item) {
        this.last = item.prev;
      }
    }
    return this;
  }
  /**
   * Returns an array of [key, value] pairs for the specified keys.
   * Order follows LRU order (least to most recently used).
   *
   * @method entries
   * @memberof LRU
   * @param {string[]} [keys=this.keys()] - Array of keys to get entries for. Defaults to all keys.
   * @returns {Array<Array<*>>} Array of [key, value] pairs in LRU order.
   * @example
   * cache.set('a', 1).set('b', 2);
   * console.log(cache.entries()); // [['a', 1], ['b', 2]]
   * console.log(cache.entries(['a'])); // [['a', 1]]
   * @see {@link LRU#keys}
   * @see {@link LRU#values}
   * @since 11.1.0
   */
  entries(keys = this.keys()) {
    const result = new Array(keys.length);
    for (let i2 = 0; i2 < keys.length; i2++) {
      const key = keys[i2];
      result[i2] = [key, this.get(key)];
    }
    return result;
  }
  /**
   * Removes the least recently used item from the cache.
   *
   * @method evict
   * @memberof LRU
   * @param {boolean} [bypass=false] - Whether to force eviction even when cache is empty.
   * @returns {LRU} The LRU instance for method chaining.
   * @example
   * cache.set('old', 'value').set('new', 'value');
   * cache.evict(); // Removes 'old' item
   * @see {@link LRU#setWithEvicted}
   * @since 1.0.0
   */
  evict(bypass = false) {
    if (bypass || this.size > 0) {
      const item = this.first;
      delete this.items[item.key];
      if (--this.size === 0) {
        this.first = null;
        this.last = null;
      } else {
        this.first = item.next;
        this.first.prev = null;
      }
    }
    return this;
  }
  /**
   * Returns the expiration timestamp for a given key.
   *
   * @method expiresAt
   * @memberof LRU
   * @param {string} key - The key to check expiration for.
   * @returns {number|undefined} The expiration timestamp in milliseconds, or undefined if key doesn't exist.
   * @example
   * const cache = new LRU(100, 5000); // 5 second TTL
   * cache.set('key1', 'value1');
   * console.log(cache.expiresAt('key1')); // timestamp 5 seconds from now
   * @see {@link LRU#get}
   * @see {@link LRU#has}
   * @since 1.0.0
   */
  expiresAt(key) {
    let result;
    if (this.has(key)) {
      result = this.items[key].expiry;
    }
    return result;
  }
  /**
   * Retrieves a value from the cache by key. Updates the item's position to most recently used.
   *
   * @method get
   * @memberof LRU
   * @param {string} key - The key to retrieve.
   * @returns {*} The value associated with the key, or undefined if not found or expired.
   * @example
   * cache.set('key1', 'value1');
   * console.log(cache.get('key1')); // 'value1'
   * console.log(cache.get('nonexistent')); // undefined
   * @see {@link LRU#set}
   * @see {@link LRU#has}
   * @since 1.0.0
   */
  get(key) {
    const item = this.items[key];
    if (item !== void 0) {
      if (this.ttl > 0) {
        if (item.expiry <= Date.now()) {
          this.delete(key);
          return void 0;
        }
      }
      this.moveToEnd(item);
      return item.value;
    }
    return void 0;
  }
  /**
   * Checks if a key exists in the cache.
   *
   * @method has
   * @memberof LRU
   * @param {string} key - The key to check for.
   * @returns {boolean} True if the key exists, false otherwise.
   * @example
   * cache.set('key1', 'value1');
   * console.log(cache.has('key1')); // true
   * console.log(cache.has('nonexistent')); // false
   * @see {@link LRU#get}
   * @see {@link LRU#delete}
   * @since 9.0.0
   */
  has(key) {
    return key in this.items;
  }
  /**
   * Efficiently moves an item to the end of the LRU list (most recently used position).
   * This is an internal optimization method that avoids the overhead of the full set() operation
   * when only LRU position needs to be updated.
   *
   * @method moveToEnd
   * @memberof LRU
   * @param {Object} item - The cache item with prev/next pointers to reposition.
   * @private
   * @since 11.3.5
   */
  moveToEnd(item) {
    if (this.last === item) {
      return;
    }
    if (item.prev !== null) {
      item.prev.next = item.next;
    }
    if (item.next !== null) {
      item.next.prev = item.prev;
    }
    if (this.first === item) {
      this.first = item.next;
    }
    item.prev = this.last;
    item.next = null;
    if (this.last !== null) {
      this.last.next = item;
    }
    this.last = item;
    if (this.first === null) {
      this.first = item;
    }
  }
  /**
   * Returns an array of all keys in the cache, ordered from least to most recently used.
   *
   * @method keys
   * @memberof LRU
   * @returns {string[]} Array of keys in LRU order.
   * @example
   * cache.set('a', 1).set('b', 2);
   * cache.get('a'); // Move 'a' to most recent
   * console.log(cache.keys()); // ['b', 'a']
   * @see {@link LRU#values}
   * @see {@link LRU#entries}
   * @since 9.0.0
   */
  keys() {
    const result = new Array(this.size);
    let x2 = this.first;
    let i2 = 0;
    while (x2 !== null) {
      result[i2++] = x2.key;
      x2 = x2.next;
    }
    return result;
  }
  /**
   * Sets a value in the cache and returns any evicted item.
   *
   * @method setWithEvicted
   * @memberof LRU
   * @param {string} key - The key to set.
   * @param {*} value - The value to store.
   * @param {boolean} [resetTtl=this.resetTtl] - Whether to reset the TTL for this operation.
   * @returns {Object|null} The evicted item (if any) with shape {key, value, expiry, prev, next}, or null.
   * @example
   * const cache = new LRU(2);
   * cache.set('a', 1).set('b', 2);
   * const evicted = cache.setWithEvicted('c', 3); // evicted = {key: 'a', value: 1, ...}
   * @see {@link LRU#set}
   * @see {@link LRU#evict}
   * @since 11.3.0
   */
  setWithEvicted(key, value, resetTtl = this.resetTtl) {
    let evicted = null;
    if (this.has(key)) {
      this.set(key, value, true, resetTtl);
    } else {
      if (this.max > 0 && this.size === this.max) {
        evicted = { ...this.first };
        this.evict(true);
      }
      let item = this.items[key] = {
        expiry: this.ttl > 0 ? Date.now() + this.ttl : this.ttl,
        key,
        prev: this.last,
        next: null,
        value
      };
      if (++this.size === 1) {
        this.first = item;
      } else {
        this.last.next = item;
      }
      this.last = item;
    }
    return evicted;
  }
  /**
   * Sets a value in the cache. Updates the item's position to most recently used.
   *
   * @method set
   * @memberof LRU
   * @param {string} key - The key to set.
   * @param {*} value - The value to store.
   * @param {boolean} [bypass=false] - Internal parameter for setWithEvicted method.
   * @param {boolean} [resetTtl=this.resetTtl] - Whether to reset the TTL for this operation.
   * @returns {LRU} The LRU instance for method chaining.
   * @example
   * cache.set('key1', 'value1')
   *      .set('key2', 'value2')
   *      .set('key3', 'value3');
   * @see {@link LRU#get}
   * @see {@link LRU#setWithEvicted}
   * @since 1.0.0
   */
  set(key, value, bypass = false, resetTtl = this.resetTtl) {
    let item = this.items[key];
    if (bypass || item !== void 0) {
      item.value = value;
      if (bypass === false && resetTtl) {
        item.expiry = this.ttl > 0 ? Date.now() + this.ttl : this.ttl;
      }
      this.moveToEnd(item);
    } else {
      if (this.max > 0 && this.size === this.max) {
        this.evict(true);
      }
      item = this.items[key] = {
        expiry: this.ttl > 0 ? Date.now() + this.ttl : this.ttl,
        key,
        prev: this.last,
        next: null,
        value
      };
      if (++this.size === 1) {
        this.first = item;
      } else {
        this.last.next = item;
      }
      this.last = item;
    }
    return this;
  }
  /**
   * Returns an array of all values in the cache for the specified keys.
   * Order follows LRU order (least to most recently used).
   *
   * @method values
   * @memberof LRU
   * @param {string[]} [keys=this.keys()] - Array of keys to get values for. Defaults to all keys.
   * @returns {Array<*>} Array of values corresponding to the keys in LRU order.
   * @example
   * cache.set('a', 1).set('b', 2);
   * console.log(cache.values()); // [1, 2]
   * console.log(cache.values(['a'])); // [1]
   * @see {@link LRU#keys}
   * @see {@link LRU#entries}
   * @since 11.1.0
   */
  values(keys = this.keys()) {
    const result = new Array(keys.length);
    for (let i2 = 0; i2 < keys.length; i2++) {
      result[i2] = this.get(keys[i2]);
    }
    return result;
  }
}
function lru(max = 1e3, ttl = 0, resetTtl = false) {
  if (isNaN(max) || max < 0) {
    throw new TypeError("Invalid max value");
  }
  if (isNaN(ttl) || ttl < 0) {
    throw new TypeError("Invalid ttl value");
  }
  if (typeof resetTtl !== "boolean") {
    throw new TypeError("Invalid resetTtl value");
  }
  return new LRU(max, ttl, resetTtl);
}
const genericFontFamilies = [
  "serif",
  "sans-serif",
  "monospace",
  "cursive",
  "fantasy",
  "system-ui"
];
function fontStringFromTextStyle(style) {
  const fontSizeString = typeof style.fontSize === "number" ? `${style.fontSize}px` : style.fontSize;
  let fontFamilies = style.fontFamily;
  if (!Array.isArray(style.fontFamily)) {
    fontFamilies = style.fontFamily.split(",");
  }
  for (let i2 = fontFamilies.length - 1; i2 >= 0; i2--) {
    let fontFamily = fontFamilies[i2].trim();
    if (!/([\"\'])[^\'\"]+\1/.test(fontFamily) && !genericFontFamilies.includes(fontFamily)) {
      fontFamily = `"${fontFamily}"`;
    }
    fontFamilies[i2] = fontFamily;
  }
  return `${style.fontStyle} ${style.fontVariant} ${style.fontWeight} ${fontSizeString} ${fontFamilies.join(",")}`;
}
const contextSettings = {
  // TextMetrics requires getImageData readback for measuring fonts.
  willReadFrequently: true
};
const _CanvasTextMetrics = class _CanvasTextMetrics2 {
  /**
   * Checking that we can use modern canvas 2D API.
   *
   * Note: This is an unstable API, Chrome < 94 use `textLetterSpacing`, later versions use `letterSpacing`.
   * @see TextMetrics.experimentalLetterSpacing
   * @see https://developer.mozilla.org/en-US/docs/Web/API/ICanvasRenderingContext2D/letterSpacing
   * @see https://developer.chrome.com/origintrials/#/view_trial/3585991203293757441
   */
  static get experimentalLetterSpacingSupported() {
    let result = _CanvasTextMetrics2._experimentalLetterSpacingSupported;
    if (result === void 0) {
      const proto = DOMAdapter.get().getCanvasRenderingContext2D().prototype;
      result = _CanvasTextMetrics2._experimentalLetterSpacingSupported = "letterSpacing" in proto || "textLetterSpacing" in proto;
    }
    return result;
  }
  /**
   * @param text - the text that was measured
   * @param style - the style that was measured
   * @param width - the measured width of the text
   * @param height - the measured height of the text
   * @param lines - an array of the lines of text broken by new lines and wrapping if specified in style
   * @param lineWidths - an array of the line widths for each line matched to `lines`
   * @param lineHeight - the measured line height for this style
   * @param maxLineWidth - the maximum line width for all measured lines
   * @param {FontMetrics} fontProperties - the font properties object from TextMetrics.measureFont
   */
  constructor(text, style, width, height, lines, lineWidths, lineHeight, maxLineWidth, fontProperties) {
    this.text = text;
    this.style = style;
    this.width = width;
    this.height = height;
    this.lines = lines;
    this.lineWidths = lineWidths;
    this.lineHeight = lineHeight;
    this.maxLineWidth = maxLineWidth;
    this.fontProperties = fontProperties;
  }
  /**
   * Measures the supplied string of text and returns a Rectangle.
   * @param text - The text to measure.
   * @param style - The text style to use for measuring
   * @param canvas - optional specification of the canvas to use for measuring.
   * @param wordWrap
   * @returns Measured width and height of the text.
   */
  static measureText(text = " ", style, canvas = _CanvasTextMetrics2._canvas, wordWrap = style.wordWrap) {
    const textKey = `${text}-${style.styleKey}-wordWrap-${wordWrap}`;
    if (_CanvasTextMetrics2._measurementCache.has(textKey)) {
      return _CanvasTextMetrics2._measurementCache.get(textKey);
    }
    const font = fontStringFromTextStyle(style);
    const fontProperties = _CanvasTextMetrics2.measureFont(font);
    if (fontProperties.fontSize === 0) {
      fontProperties.fontSize = style.fontSize;
      fontProperties.ascent = style.fontSize;
    }
    const context2 = _CanvasTextMetrics2.__context;
    context2.font = font;
    const outputText = wordWrap ? _CanvasTextMetrics2._wordWrap(text, style, canvas) : text;
    const lines = outputText.split(/(?:\r\n|\r|\n)/);
    const lineWidths = new Array(lines.length);
    let maxLineWidth = 0;
    for (let i2 = 0; i2 < lines.length; i2++) {
      const lineWidth = _CanvasTextMetrics2._measureText(lines[i2], style.letterSpacing, context2);
      lineWidths[i2] = lineWidth;
      maxLineWidth = Math.max(maxLineWidth, lineWidth);
    }
    const strokeWidth = style._stroke?.width || 0;
    let width = maxLineWidth + strokeWidth;
    if (style.dropShadow) {
      width += style.dropShadow.distance;
    }
    const lineHeight = style.lineHeight || fontProperties.fontSize;
    let height = Math.max(lineHeight, fontProperties.fontSize + strokeWidth) + (lines.length - 1) * (lineHeight + style.leading);
    if (style.dropShadow) {
      height += style.dropShadow.distance;
    }
    const measurements = new _CanvasTextMetrics2(
      text,
      style,
      width,
      height,
      lines,
      lineWidths,
      lineHeight + style.leading,
      maxLineWidth,
      fontProperties
    );
    _CanvasTextMetrics2._measurementCache.set(textKey, measurements);
    return measurements;
  }
  static _measureText(text, letterSpacing, context2) {
    let useExperimentalLetterSpacing = false;
    if (_CanvasTextMetrics2.experimentalLetterSpacingSupported) {
      if (_CanvasTextMetrics2.experimentalLetterSpacing) {
        context2.letterSpacing = `${letterSpacing}px`;
        context2.textLetterSpacing = `${letterSpacing}px`;
        useExperimentalLetterSpacing = true;
      } else {
        context2.letterSpacing = "0px";
        context2.textLetterSpacing = "0px";
      }
    }
    const metrics = context2.measureText(text);
    let metricWidth = metrics.width;
    const actualBoundingBoxLeft = -metrics.actualBoundingBoxLeft;
    const actualBoundingBoxRight = metrics.actualBoundingBoxRight;
    let boundsWidth = actualBoundingBoxRight - actualBoundingBoxLeft;
    if (metricWidth > 0) {
      if (useExperimentalLetterSpacing) {
        metricWidth -= letterSpacing;
        boundsWidth -= letterSpacing;
      } else {
        const val = (_CanvasTextMetrics2.graphemeSegmenter(text).length - 1) * letterSpacing;
        metricWidth += val;
        boundsWidth += val;
      }
    }
    return Math.max(metricWidth, boundsWidth);
  }
  /**
   * Applies newlines to a string to have it optimally fit into the horizontal
   * bounds set by the Text object's wordWrapWidth property.
   * @param text - String to apply word wrapping to
   * @param style - the style to use when wrapping
   * @param canvas - optional specification of the canvas to use for measuring.
   * @returns New string with new lines applied where required
   */
  static _wordWrap(text, style, canvas = _CanvasTextMetrics2._canvas) {
    const context2 = canvas.getContext("2d", contextSettings);
    let width = 0;
    let line = "";
    let lines = "";
    const cache = /* @__PURE__ */ Object.create(null);
    const { letterSpacing, whiteSpace } = style;
    const collapseSpaces = _CanvasTextMetrics2._collapseSpaces(whiteSpace);
    const collapseNewlines = _CanvasTextMetrics2._collapseNewlines(whiteSpace);
    let canPrependSpaces = !collapseSpaces;
    const wordWrapWidth = style.wordWrapWidth + letterSpacing;
    const tokens = _CanvasTextMetrics2._tokenize(text);
    for (let i2 = 0; i2 < tokens.length; i2++) {
      let token = tokens[i2];
      if (_CanvasTextMetrics2._isNewline(token)) {
        if (!collapseNewlines) {
          lines += _CanvasTextMetrics2._addLine(line);
          canPrependSpaces = !collapseSpaces;
          line = "";
          width = 0;
          continue;
        }
        token = " ";
      }
      if (collapseSpaces) {
        const currIsBreakingSpace = _CanvasTextMetrics2.isBreakingSpace(token);
        const lastIsBreakingSpace = _CanvasTextMetrics2.isBreakingSpace(line[line.length - 1]);
        if (currIsBreakingSpace && lastIsBreakingSpace) {
          continue;
        }
      }
      const tokenWidth = _CanvasTextMetrics2._getFromCache(token, letterSpacing, cache, context2);
      if (tokenWidth > wordWrapWidth) {
        if (line !== "") {
          lines += _CanvasTextMetrics2._addLine(line);
          line = "";
          width = 0;
        }
        if (_CanvasTextMetrics2.canBreakWords(token, style.breakWords)) {
          const characters = _CanvasTextMetrics2.wordWrapSplit(token);
          for (let j2 = 0; j2 < characters.length; j2++) {
            let char = characters[j2];
            let lastChar = char;
            let k2 = 1;
            while (characters[j2 + k2]) {
              const nextChar = characters[j2 + k2];
              if (!_CanvasTextMetrics2.canBreakChars(lastChar, nextChar, token, j2, style.breakWords)) {
                char += nextChar;
              } else {
                break;
              }
              lastChar = nextChar;
              k2++;
            }
            j2 += k2 - 1;
            const characterWidth = _CanvasTextMetrics2._getFromCache(char, letterSpacing, cache, context2);
            if (characterWidth + width > wordWrapWidth) {
              lines += _CanvasTextMetrics2._addLine(line);
              canPrependSpaces = false;
              line = "";
              width = 0;
            }
            line += char;
            width += characterWidth;
          }
        } else {
          if (line.length > 0) {
            lines += _CanvasTextMetrics2._addLine(line);
            line = "";
            width = 0;
          }
          const isLastToken = i2 === tokens.length - 1;
          lines += _CanvasTextMetrics2._addLine(token, !isLastToken);
          canPrependSpaces = false;
          line = "";
          width = 0;
        }
      } else {
        if (tokenWidth + width > wordWrapWidth) {
          canPrependSpaces = false;
          lines += _CanvasTextMetrics2._addLine(line);
          line = "";
          width = 0;
        }
        if (line.length > 0 || !_CanvasTextMetrics2.isBreakingSpace(token) || canPrependSpaces) {
          line += token;
          width += tokenWidth;
        }
      }
    }
    lines += _CanvasTextMetrics2._addLine(line, false);
    return lines;
  }
  /**
   * Convenience function for logging each line added during the wordWrap method.
   * @param line    - The line of text to add
   * @param newLine - Add new line character to end
   * @returns A formatted line
   */
  static _addLine(line, newLine = true) {
    line = _CanvasTextMetrics2._trimRight(line);
    line = newLine ? `${line}
` : line;
    return line;
  }
  /**
   * Gets & sets the widths of calculated characters in a cache object
   * @param key            - The key
   * @param letterSpacing  - The letter spacing
   * @param cache          - The cache
   * @param context        - The canvas context
   * @returns The from cache.
   */
  static _getFromCache(key, letterSpacing, cache, context2) {
    let width = cache[key];
    if (typeof width !== "number") {
      width = _CanvasTextMetrics2._measureText(key, letterSpacing, context2) + letterSpacing;
      cache[key] = width;
    }
    return width;
  }
  /**
   * Determines whether we should collapse breaking spaces.
   * @param whiteSpace - The TextStyle property whiteSpace
   * @returns Should collapse
   */
  static _collapseSpaces(whiteSpace) {
    return whiteSpace === "normal" || whiteSpace === "pre-line";
  }
  /**
   * Determines whether we should collapse newLine chars.
   * @param whiteSpace - The white space
   * @returns should collapse
   */
  static _collapseNewlines(whiteSpace) {
    return whiteSpace === "normal";
  }
  /**
   * Trims breaking whitespaces from string.
   * @param text - The text
   * @returns Trimmed string
   */
  static _trimRight(text) {
    if (typeof text !== "string") {
      return "";
    }
    for (let i2 = text.length - 1; i2 >= 0; i2--) {
      const char = text[i2];
      if (!_CanvasTextMetrics2.isBreakingSpace(char)) {
        break;
      }
      text = text.slice(0, -1);
    }
    return text;
  }
  /**
   * Determines if char is a newline.
   * @param char - The character
   * @returns True if newline, False otherwise.
   */
  static _isNewline(char) {
    if (typeof char !== "string") {
      return false;
    }
    return _CanvasTextMetrics2._newlines.includes(char.charCodeAt(0));
  }
  /**
   * Determines if char is a breaking whitespace.
   *
   * It allows one to determine whether char should be a breaking whitespace
   * For example certain characters in CJK langs or numbers.
   * It must return a boolean.
   * @param char - The character
   * @param [_nextChar] - The next character
   * @returns True if whitespace, False otherwise.
   */
  static isBreakingSpace(char, _nextChar) {
    if (typeof char !== "string") {
      return false;
    }
    return _CanvasTextMetrics2._breakingSpaces.includes(char.charCodeAt(0));
  }
  /**
   * Splits a string into words, breaking-spaces and newLine characters
   * @param text - The text
   * @returns A tokenized array
   */
  static _tokenize(text) {
    const tokens = [];
    let token = "";
    if (typeof text !== "string") {
      return tokens;
    }
    for (let i2 = 0; i2 < text.length; i2++) {
      const char = text[i2];
      const nextChar = text[i2 + 1];
      if (_CanvasTextMetrics2.isBreakingSpace(char, nextChar) || _CanvasTextMetrics2._isNewline(char)) {
        if (token !== "") {
          tokens.push(token);
          token = "";
        }
        if (char === "\r" && nextChar === "\n") {
          tokens.push("\r\n");
          i2++;
        } else {
          tokens.push(char);
        }
        continue;
      }
      token += char;
    }
    if (token !== "") {
      tokens.push(token);
    }
    return tokens;
  }
  /**
   * Overridable helper method used internally by TextMetrics, exposed to allow customizing the class's behavior.
   *
   * It allows one to customise which words should break
   * Examples are if the token is CJK or numbers.
   * It must return a boolean.
   * @param _token - The token
   * @param breakWords - The style attr break words
   * @returns Whether to break word or not
   */
  static canBreakWords(_token, breakWords) {
    return breakWords;
  }
  /**
   * Overridable helper method used internally by TextMetrics, exposed to allow customizing the class's behavior.
   *
   * It allows one to determine whether a pair of characters
   * should be broken by newlines
   * For example certain characters in CJK langs or numbers.
   * It must return a boolean.
   * @param _char - The character
   * @param _nextChar - The next character
   * @param _token - The token/word the characters are from
   * @param _index - The index in the token of the char
   * @param _breakWords - The style attr break words
   * @returns whether to break word or not
   */
  static canBreakChars(_char, _nextChar, _token, _index, _breakWords) {
    return true;
  }
  /**
   * Overridable helper method used internally by TextMetrics, exposed to allow customizing the class's behavior.
   *
   * It is called when a token (usually a word) has to be split into separate pieces
   * in order to determine the point to break a word.
   * It must return an array of characters.
   * @param token - The token to split
   * @returns The characters of the token
   * @see CanvasTextMetrics.graphemeSegmenter
   */
  static wordWrapSplit(token) {
    return _CanvasTextMetrics2.graphemeSegmenter(token);
  }
  /**
   * Calculates the ascent, descent and fontSize of a given font-style
   * @param font - String representing the style of the font
   * @returns Font properties object
   */
  static measureFont(font) {
    if (_CanvasTextMetrics2._fonts[font]) {
      return _CanvasTextMetrics2._fonts[font];
    }
    const context2 = _CanvasTextMetrics2._context;
    context2.font = font;
    const metrics = context2.measureText(_CanvasTextMetrics2.METRICS_STRING + _CanvasTextMetrics2.BASELINE_SYMBOL);
    const properties = {
      ascent: metrics.actualBoundingBoxAscent,
      descent: metrics.actualBoundingBoxDescent,
      fontSize: metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent
    };
    _CanvasTextMetrics2._fonts[font] = properties;
    return properties;
  }
  /**
   * Clear font metrics in metrics cache.
   * @param {string} [font] - font name. If font name not set then clear cache for all fonts.
   */
  static clearMetrics(font = "") {
    if (font) {
      delete _CanvasTextMetrics2._fonts[font];
    } else {
      _CanvasTextMetrics2._fonts = {};
    }
  }
  /**
   * Cached canvas element for measuring text
   * TODO: this should be private, but isn't because of backward compat, will fix later.
   * @ignore
   */
  static get _canvas() {
    if (!_CanvasTextMetrics2.__canvas) {
      let canvas;
      try {
        const c2 = new OffscreenCanvas(0, 0);
        const context2 = c2.getContext("2d", contextSettings);
        if (context2?.measureText) {
          _CanvasTextMetrics2.__canvas = c2;
          return c2;
        }
        canvas = DOMAdapter.get().createCanvas();
      } catch (_cx) {
        canvas = DOMAdapter.get().createCanvas();
      }
      canvas.width = canvas.height = 10;
      _CanvasTextMetrics2.__canvas = canvas;
    }
    return _CanvasTextMetrics2.__canvas;
  }
  /**
   * TODO: this should be private, but isn't because of backward compat, will fix later.
   * @ignore
   */
  static get _context() {
    if (!_CanvasTextMetrics2.__context) {
      _CanvasTextMetrics2.__context = _CanvasTextMetrics2._canvas.getContext("2d", contextSettings);
    }
    return _CanvasTextMetrics2.__context;
  }
};
_CanvasTextMetrics.METRICS_STRING = "|ÉqÅ";
_CanvasTextMetrics.BASELINE_SYMBOL = "M";
_CanvasTextMetrics.BASELINE_MULTIPLIER = 1.4;
_CanvasTextMetrics.HEIGHT_MULTIPLIER = 2;
_CanvasTextMetrics.graphemeSegmenter = (() => {
  if (typeof Intl?.Segmenter === "function") {
    const segmenter = new Intl.Segmenter();
    return (s2) => {
      const segments = segmenter.segment(s2);
      const result = [];
      let i2 = 0;
      for (const segment of segments) {
        result[i2++] = segment.segment;
      }
      return result;
    };
  }
  return (s2) => [...s2];
})();
_CanvasTextMetrics.experimentalLetterSpacing = false;
_CanvasTextMetrics._fonts = {};
_CanvasTextMetrics._newlines = [
  10,
  // line feed
  13
  // carriage return
];
_CanvasTextMetrics._breakingSpaces = [
  9,
  // character tabulation
  32,
  // space
  8192,
  // en quad
  8193,
  // em quad
  8194,
  // en space
  8195,
  // em space
  8196,
  // three-per-em space
  8197,
  // four-per-em space
  8198,
  // six-per-em space
  8200,
  // punctuation space
  8201,
  // thin space
  8202,
  // hair space
  8287,
  // medium mathematical space
  12288
  // ideographic space
];
_CanvasTextMetrics._measurementCache = lru(1e3);
let CanvasTextMetrics = _CanvasTextMetrics;
const emptyColorStops = [{ offset: 0, color: "white" }, { offset: 1, color: "black" }];
const _FillGradient = class _FillGradient2 {
  constructor(...args) {
    this.uid = uid("fillGradient");
    this._tick = 0;
    this.type = "linear";
    this.colorStops = [];
    let options = ensureGradientOptions(args);
    const defaults = options.type === "radial" ? _FillGradient2.defaultRadialOptions : _FillGradient2.defaultLinearOptions;
    options = { ...defaults, ...definedProps(options) };
    this._textureSize = options.textureSize;
    this._wrapMode = options.wrapMode;
    if (options.type === "radial") {
      this.center = options.center;
      this.outerCenter = options.outerCenter ?? this.center;
      this.innerRadius = options.innerRadius;
      this.outerRadius = options.outerRadius;
      this.scale = options.scale;
      this.rotation = options.rotation;
    } else {
      this.start = options.start;
      this.end = options.end;
    }
    this.textureSpace = options.textureSpace;
    this.type = options.type;
    options.colorStops.forEach((stop) => {
      this.addColorStop(stop.offset, stop.color);
    });
  }
  /**
   * Adds a color stop to the gradient
   * @param offset - Position of the stop (0-1)
   * @param color - Color of the stop
   * @returns This gradient instance for chaining
   */
  addColorStop(offset, color) {
    this.colorStops.push({ offset, color: Color.shared.setValue(color).toHexa() });
    return this;
  }
  /**
   * Builds the internal texture and transform for the gradient.
   * Called automatically when the gradient is first used.
   * @internal
   */
  buildLinearGradient() {
    if (this.texture)
      return;
    let { x: x0, y: y0 } = this.start;
    let { x: x1, y: y1 } = this.end;
    let dx = x1 - x0;
    let dy = y1 - y0;
    const flip = dx < 0 || dy < 0;
    if (this._wrapMode === "clamp-to-edge") {
      if (dx < 0) {
        const temp = x0;
        x0 = x1;
        x1 = temp;
        dx *= -1;
      }
      if (dy < 0) {
        const temp = y0;
        y0 = y1;
        y1 = temp;
        dy *= -1;
      }
    }
    const colorStops = this.colorStops.length ? this.colorStops : emptyColorStops;
    const defaultSize = this._textureSize;
    const { canvas, context: context2 } = getCanvas(defaultSize, 1);
    const gradient = !flip ? context2.createLinearGradient(0, 0, this._textureSize, 0) : context2.createLinearGradient(this._textureSize, 0, 0, 0);
    addColorStops(gradient, colorStops);
    context2.fillStyle = gradient;
    context2.fillRect(0, 0, defaultSize, 1);
    this.texture = new Texture({
      source: new ImageSource({
        resource: canvas,
        addressMode: this._wrapMode
      })
    });
    const dist = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);
    const m2 = new Matrix();
    m2.scale(dist / defaultSize, 1);
    m2.rotate(angle);
    m2.translate(x0, y0);
    if (this.textureSpace === "local") {
      m2.scale(defaultSize, defaultSize);
    }
    this.transform = m2;
  }
  /**
   * Builds the internal texture and transform for the gradient.
   * Called automatically when the gradient is first used.
   * @internal
   */
  buildGradient() {
    if (!this.texture)
      this._tick++;
    if (this.type === "linear") {
      this.buildLinearGradient();
    } else {
      this.buildRadialGradient();
    }
  }
  /**
   * Builds the internal texture and transform for the radial gradient.
   * Called automatically when the gradient is first used.
   * @internal
   */
  buildRadialGradient() {
    if (this.texture)
      return;
    const colorStops = this.colorStops.length ? this.colorStops : emptyColorStops;
    const defaultSize = this._textureSize;
    const { canvas, context: context2 } = getCanvas(defaultSize, defaultSize);
    const { x: x0, y: y0 } = this.center;
    const { x: x1, y: y1 } = this.outerCenter;
    const r0 = this.innerRadius;
    const r1 = this.outerRadius;
    const ox = x1 - r1;
    const oy = y1 - r1;
    const scale = defaultSize / (r1 * 2);
    const cx = (x0 - ox) * scale;
    const cy = (y0 - oy) * scale;
    const gradient = context2.createRadialGradient(
      cx,
      cy,
      r0 * scale,
      (x1 - ox) * scale,
      (y1 - oy) * scale,
      r1 * scale
    );
    addColorStops(gradient, colorStops);
    context2.fillStyle = colorStops[colorStops.length - 1].color;
    context2.fillRect(0, 0, defaultSize, defaultSize);
    context2.fillStyle = gradient;
    context2.translate(cx, cy);
    context2.rotate(this.rotation);
    context2.scale(1, this.scale);
    context2.translate(-cx, -cy);
    context2.fillRect(0, 0, defaultSize, defaultSize);
    this.texture = new Texture({
      source: new ImageSource({
        resource: canvas,
        addressMode: this._wrapMode
      })
    });
    const m2 = new Matrix();
    m2.scale(1 / scale, 1 / scale);
    m2.translate(ox, oy);
    if (this.textureSpace === "local") {
      m2.scale(defaultSize, defaultSize);
    }
    this.transform = m2;
  }
  /** Destroys the gradient, releasing resources. This will also destroy the internal texture. */
  destroy() {
    this.texture?.destroy(true);
    this.texture = null;
    this.transform = null;
    this.colorStops = [];
    this.start = null;
    this.end = null;
    this.center = null;
    this.outerCenter = null;
  }
  /**
   * Returns a unique key for this gradient instance.
   * This key is used for caching and texture management.
   * @returns {string} Unique key for the gradient
   */
  get styleKey() {
    return `fill-gradient-${this.uid}-${this._tick}`;
  }
};
_FillGradient.defaultLinearOptions = {
  start: { x: 0, y: 0 },
  end: { x: 0, y: 1 },
  colorStops: [],
  textureSpace: "local",
  type: "linear",
  textureSize: 256,
  wrapMode: "clamp-to-edge"
};
_FillGradient.defaultRadialOptions = {
  center: { x: 0.5, y: 0.5 },
  innerRadius: 0,
  outerRadius: 0.5,
  colorStops: [],
  scale: 1,
  textureSpace: "local",
  type: "radial",
  textureSize: 256,
  wrapMode: "clamp-to-edge"
};
let FillGradient = _FillGradient;
function addColorStops(gradient, colorStops) {
  for (let i2 = 0; i2 < colorStops.length; i2++) {
    const stop = colorStops[i2];
    gradient.addColorStop(stop.offset, stop.color);
  }
}
function getCanvas(width, height) {
  const canvas = DOMAdapter.get().createCanvas(width, height);
  const context2 = canvas.getContext("2d");
  return { canvas, context: context2 };
}
function ensureGradientOptions(args) {
  let options = args[0] ?? {};
  if (typeof options === "number" || args[1]) {
    deprecation("8.5.2", `use options object instead`);
    options = {
      type: "linear",
      start: { x: args[0], y: args[1] },
      end: { x: args[2], y: args[3] },
      textureSpace: args[4],
      textureSize: args[5] ?? FillGradient.defaultLinearOptions.textureSize
    };
  }
  return options;
}
const repetitionMap = {
  repeat: {
    addressModeU: "repeat",
    addressModeV: "repeat"
  },
  "repeat-x": {
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge"
  },
  "repeat-y": {
    addressModeU: "clamp-to-edge",
    addressModeV: "repeat"
  },
  "no-repeat": {
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge"
  }
};
class FillPattern {
  constructor(texture, repetition) {
    this.uid = uid("fillPattern");
    this._tick = 0;
    this.transform = new Matrix();
    this.texture = texture;
    this.transform.scale(
      1 / texture.frame.width,
      1 / texture.frame.height
    );
    if (repetition) {
      texture.source.style.addressModeU = repetitionMap[repetition].addressModeU;
      texture.source.style.addressModeV = repetitionMap[repetition].addressModeV;
    }
  }
  /**
   * Sets the transform for the pattern
   * @param transform - The transform matrix to apply to the pattern.
   * If not provided, the pattern will use the default transform.
   */
  setTransform(transform) {
    const texture = this.texture;
    this.transform.copyFrom(transform);
    this.transform.invert();
    this.transform.scale(
      1 / texture.frame.width,
      1 / texture.frame.height
    );
    this._tick++;
  }
  /** Internal texture used to render the gradient */
  get texture() {
    return this._texture;
  }
  set texture(value) {
    if (this._texture === value)
      return;
    this._texture = value;
    this._tick++;
  }
  /**
   * Returns a unique key for this instance.
   * This key is used for caching.
   * @returns {string} Unique key for the instance
   */
  get styleKey() {
    return `fill-pattern-${this.uid}-${this._tick}`;
  }
  /** Destroys the fill pattern, releasing resources. This will also destroy the internal texture. */
  destroy() {
    this.texture.destroy(true);
    this.texture = null;
  }
}
var parseSvgPath;
var hasRequiredParseSvgPath;
function requireParseSvgPath() {
  if (hasRequiredParseSvgPath) return parseSvgPath;
  hasRequiredParseSvgPath = 1;
  parseSvgPath = parse2;
  var length = { a: 7, c: 6, h: 1, l: 2, m: 2, q: 4, s: 4, t: 2, v: 1, z: 0 };
  var segment = /([astvzqmhlc])([^astvzqmhlc]*)/ig;
  function parse2(path2) {
    var data = [];
    path2.replace(segment, function(_, command, args) {
      var type = command.toLowerCase();
      args = parseValues(args);
      if (type == "m" && args.length > 2) {
        data.push([command].concat(args.splice(0, 2)));
        type = "l";
        command = command == "m" ? "l" : "L";
      }
      while (true) {
        if (args.length == length[type]) {
          args.unshift(command);
          return data.push(args);
        }
        if (args.length < length[type]) throw new Error("malformed path data");
        data.push([command].concat(args.splice(0, length[type])));
      }
    });
    return data;
  }
  var number = /-?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?/ig;
  function parseValues(args) {
    var numbers = args.match(number);
    return numbers ? numbers.map(Number) : [];
  }
  return parseSvgPath;
}
var parseSvgPathExports = requireParseSvgPath();
const parse = /* @__PURE__ */ getDefaultExportFromCjs(parseSvgPathExports);
function parseSVGPath(svgPath, path2) {
  const commands = parse(svgPath);
  const subpaths = [];
  let currentSubPath = null;
  let lastX = 0;
  let lastY = 0;
  for (let i2 = 0; i2 < commands.length; i2++) {
    const command = commands[i2];
    const type = command[0];
    const data = command;
    switch (type) {
      case "M":
        lastX = data[1];
        lastY = data[2];
        path2.moveTo(lastX, lastY);
        break;
      case "m":
        lastX += data[1];
        lastY += data[2];
        path2.moveTo(lastX, lastY);
        break;
      case "H":
        lastX = data[1];
        path2.lineTo(lastX, lastY);
        break;
      case "h":
        lastX += data[1];
        path2.lineTo(lastX, lastY);
        break;
      case "V":
        lastY = data[1];
        path2.lineTo(lastX, lastY);
        break;
      case "v":
        lastY += data[1];
        path2.lineTo(lastX, lastY);
        break;
      case "L":
        lastX = data[1];
        lastY = data[2];
        path2.lineTo(lastX, lastY);
        break;
      case "l":
        lastX += data[1];
        lastY += data[2];
        path2.lineTo(lastX, lastY);
        break;
      case "C":
        lastX = data[5];
        lastY = data[6];
        path2.bezierCurveTo(
          data[1],
          data[2],
          // First control point
          data[3],
          data[4],
          // Second control point
          lastX,
          lastY
          // End point
        );
        break;
      case "c":
        path2.bezierCurveTo(
          lastX + data[1],
          lastY + data[2],
          // First control point
          lastX + data[3],
          lastY + data[4],
          // Second control point
          lastX + data[5],
          lastY + data[6]
          // End point
        );
        lastX += data[5];
        lastY += data[6];
        break;
      case "S":
        lastX = data[3];
        lastY = data[4];
        path2.bezierCurveToShort(
          data[1],
          data[2],
          // Control point
          lastX,
          lastY
          // End point
        );
        break;
      case "s":
        path2.bezierCurveToShort(
          lastX + data[1],
          lastY + data[2],
          // Control point
          lastX + data[3],
          lastY + data[4]
          // End point
        );
        lastX += data[3];
        lastY += data[4];
        break;
      case "Q":
        lastX = data[3];
        lastY = data[4];
        path2.quadraticCurveTo(
          data[1],
          data[2],
          // Control point
          lastX,
          lastY
          // End point
        );
        break;
      case "q":
        path2.quadraticCurveTo(
          lastX + data[1],
          lastY + data[2],
          // Control point
          lastX + data[3],
          lastY + data[4]
          // End point
        );
        lastX += data[3];
        lastY += data[4];
        break;
      case "T":
        lastX = data[1];
        lastY = data[2];
        path2.quadraticCurveToShort(
          lastX,
          lastY
          // End point
        );
        break;
      case "t":
        lastX += data[1];
        lastY += data[2];
        path2.quadraticCurveToShort(
          lastX,
          lastY
          // End point
        );
        break;
      case "A":
        lastX = data[6];
        lastY = data[7];
        path2.arcToSvg(
          data[1],
          // rx
          data[2],
          // ry
          data[3],
          // x-axis-rotation
          data[4],
          // large-arc-flag
          data[5],
          // sweep-flag
          lastX,
          lastY
          // End point
        );
        break;
      case "a":
        lastX += data[6];
        lastY += data[7];
        path2.arcToSvg(
          data[1],
          // rx
          data[2],
          // ry
          data[3],
          // x-axis-rotation
          data[4],
          // large-arc-flag
          data[5],
          // sweep-flag
          lastX,
          lastY
          // End point
        );
        break;
      case "Z":
      case "z":
        path2.closePath();
        if (subpaths.length > 0) {
          currentSubPath = subpaths.pop();
          if (currentSubPath) {
            lastX = currentSubPath.startX;
            lastY = currentSubPath.startY;
          } else {
            lastX = 0;
            lastY = 0;
          }
        }
        currentSubPath = null;
        break;
      default:
        warn(`Unknown SVG path command: ${type}`);
    }
    if (type !== "Z" && type !== "z") {
      if (currentSubPath === null) {
        currentSubPath = { startX: lastX, startY: lastY };
        subpaths.push(currentSubPath);
      }
    }
  }
  return path2;
}
class Circle {
  /**
   * @param x - The X coordinate of the center of this circle
   * @param y - The Y coordinate of the center of this circle
   * @param radius - The radius of the circle
   */
  constructor(x2 = 0, y2 = 0, radius = 0) {
    this.type = "circle";
    this.x = x2;
    this.y = y2;
    this.radius = radius;
  }
  /**
   * Creates a clone of this Circle instance.
   * @example
   * ```ts
   * // Basic circle cloning
   * const original = new Circle(100, 100, 50);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.radius = 75;
   *
   * // Verify independence
   * console.log(original.radius); // 50
   * console.log(modified.radius); // 75
   * ```
   * @returns A copy of the Circle
   * @see {@link Circle.copyFrom} For copying into existing circle
   * @see {@link Circle.copyTo} For copying to another circle
   */
  clone() {
    return new Circle(this.x, this.y, this.radius);
  }
  /**
   * Checks whether the x and y coordinates given are contained within this circle.
   *
   * Uses the distance formula to determine if a point is inside the circle's radius.
   *
   * Commonly used for hit testing in PixiJS events and graphics.
   * @example
   * ```ts
   * // Basic containment check
   * const circle = new Circle(100, 100, 50);
   * const isInside = circle.contains(120, 120);
   *
   * // Check mouse position
   * const circle = new Circle(0, 0, 100);
   * container.hitArea = circle;
   * container.on('pointermove', (e) => {
   *     // only called if pointer is within circle
   * });
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @returns Whether the x/y coordinates are within this Circle
   * @see {@link Circle.strokeContains} For checking stroke intersection
   * @see {@link Circle.getBounds} For getting bounding box
   */
  contains(x2, y2) {
    if (this.radius <= 0)
      return false;
    const r2 = this.radius * this.radius;
    let dx = this.x - x2;
    let dy = this.y - y2;
    dx *= dx;
    dy *= dy;
    return dx + dy <= r2;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this circle including the stroke.
   * @example
   * ```ts
   * // Basic stroke check
   * const circle = new Circle(100, 100, 50);
   * const isOnStroke = circle.strokeContains(150, 100, 4); // 4px line width
   *
   * // Check with different alignments
   * const innerStroke = circle.strokeContains(150, 100, 4, 1);   // Inside
   * const centerStroke = circle.strokeContains(150, 100, 4, 0.5); // Centered
   * const outerStroke = circle.strokeContains(150, 100, 4, 0);   // Outside
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @param width - The width of the line to check
   * @param alignment - The alignment of the stroke, 0.5 by default
   * @returns Whether the x/y coordinates are within this Circle's stroke
   * @see {@link Circle.contains} For checking fill containment
   * @see {@link Circle.getBounds} For getting stroke bounds
   */
  strokeContains(x2, y2, width, alignment = 0.5) {
    if (this.radius === 0)
      return false;
    const dx = this.x - x2;
    const dy = this.y - y2;
    const radius = this.radius;
    const outerWidth = (1 - alignment) * width;
    const distance = Math.sqrt(dx * dx + dy * dy);
    return distance <= radius + outerWidth && distance > radius - (width - outerWidth);
  }
  /**
   * Returns the framing rectangle of the circle as a Rectangle object.
   * @example
   * ```ts
   * // Basic bounds calculation
   * const circle = new Circle(100, 100, 50);
   * const bounds = circle.getBounds();
   * // bounds: x=50, y=50, width=100, height=100
   *
   * // Reuse existing rectangle
   * const rect = new Rectangle();
   * circle.getBounds(rect);
   * ```
   * @param out - Optional Rectangle object to store the result
   * @returns The framing rectangle
   * @see {@link Rectangle} For rectangle properties
   * @see {@link Circle.contains} For point containment
   */
  getBounds(out2) {
    out2 || (out2 = new Rectangle());
    out2.x = this.x - this.radius;
    out2.y = this.y - this.radius;
    out2.width = this.radius * 2;
    out2.height = this.radius * 2;
    return out2;
  }
  /**
   * Copies another circle to this one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Circle(100, 100, 50);
   * const target = new Circle();
   * target.copyFrom(source);
   * ```
   * @param circle - The circle to copy from
   * @returns Returns itself
   * @see {@link Circle.copyTo} For copying to another circle
   * @see {@link Circle.clone} For creating new circle copy
   */
  copyFrom(circle) {
    this.x = circle.x;
    this.y = circle.y;
    this.radius = circle.radius;
    return this;
  }
  /**
   * Copies this circle to another one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Circle(100, 100, 50);
   * const target = new Circle();
   * source.copyTo(target);
   * ```
   * @param circle - The circle to copy to
   * @returns Returns given parameter
   * @see {@link Circle.copyFrom} For copying from another circle
   * @see {@link Circle.clone} For creating new circle copy
   */
  copyTo(circle) {
    circle.copyFrom(this);
    return circle;
  }
  toString() {
    return `[pixi.js/math:Circle x=${this.x} y=${this.y} radius=${this.radius}]`;
  }
}
class Ellipse {
  /**
   * @param x - The X coordinate of the center of this ellipse
   * @param y - The Y coordinate of the center of this ellipse
   * @param halfWidth - The half width of this ellipse
   * @param halfHeight - The half height of this ellipse
   */
  constructor(x2 = 0, y2 = 0, halfWidth = 0, halfHeight = 0) {
    this.type = "ellipse";
    this.x = x2;
    this.y = y2;
    this.halfWidth = halfWidth;
    this.halfHeight = halfHeight;
  }
  /**
   * Creates a clone of this Ellipse instance.
   * @example
   * ```ts
   * // Basic cloning
   * const original = new Ellipse(100, 100, 50, 25);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.halfWidth *= 2;
   * modified.halfHeight *= 2;
   *
   * // Verify independence
   * console.log(original.halfWidth);  // 50
   * console.log(modified.halfWidth);  // 100
   * ```
   * @returns A copy of the ellipse
   * @see {@link Ellipse.copyFrom} For copying into existing ellipse
   * @see {@link Ellipse.copyTo} For copying to another ellipse
   */
  clone() {
    return new Ellipse(this.x, this.y, this.halfWidth, this.halfHeight);
  }
  /**
   * Checks whether the x and y coordinates given are contained within this ellipse.
   * Uses normalized coordinates and the ellipse equation to determine containment.
   * @example
   * ```ts
   * // Basic containment check
   * const ellipse = new Ellipse(100, 100, 50, 25);
   * const isInside = ellipse.contains(120, 110);
   * ```
   * @remarks
   * - Uses ellipse equation (x²/a² + y²/b² ≤ 1)
   * - Returns false if dimensions are 0 or negative
   * - Normalized to center (0,0) for calculation
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @returns Whether the x/y coords are within this ellipse
   * @see {@link Ellipse.strokeContains} For checking stroke intersection
   * @see {@link Ellipse.getBounds} For getting containing rectangle
   */
  contains(x2, y2) {
    if (this.halfWidth <= 0 || this.halfHeight <= 0) {
      return false;
    }
    let normx = (x2 - this.x) / this.halfWidth;
    let normy = (y2 - this.y) / this.halfHeight;
    normx *= normx;
    normy *= normy;
    return normx + normy <= 1;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this ellipse including stroke.
   * @example
   * ```ts
   * // Basic stroke check
   * const ellipse = new Ellipse(100, 100, 50, 25);
   * const isOnStroke = ellipse.strokeContains(150, 100, 4); // 4px line width
   *
   * // Check with different alignments
   * const innerStroke = ellipse.strokeContains(150, 100, 4, 1);   // Inside
   * const centerStroke = ellipse.strokeContains(150, 100, 4, 0.5); // Centered
   * const outerStroke = ellipse.strokeContains(150, 100, 4, 0);   // Outside
   * ```
   * @remarks
   * - Uses normalized ellipse equations
   * - Considers stroke alignment
   * - Returns false if dimensions are 0
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @param strokeWidth - The width of the line to check
   * @param alignment - The alignment of the stroke (1 = inner, 0.5 = centered, 0 = outer)
   * @returns Whether the x/y coords are within this ellipse's stroke
   * @see {@link Ellipse.contains} For checking fill containment
   * @see {@link Ellipse.getBounds} For getting stroke bounds
   */
  strokeContains(x2, y2, strokeWidth, alignment = 0.5) {
    const { halfWidth, halfHeight } = this;
    if (halfWidth <= 0 || halfHeight <= 0) {
      return false;
    }
    const strokeOuterWidth = strokeWidth * (1 - alignment);
    const strokeInnerWidth = strokeWidth - strokeOuterWidth;
    const innerHorizontal = halfWidth - strokeInnerWidth;
    const innerVertical = halfHeight - strokeInnerWidth;
    const outerHorizontal = halfWidth + strokeOuterWidth;
    const outerVertical = halfHeight + strokeOuterWidth;
    const normalizedX = x2 - this.x;
    const normalizedY = y2 - this.y;
    const innerEllipse = normalizedX * normalizedX / (innerHorizontal * innerHorizontal) + normalizedY * normalizedY / (innerVertical * innerVertical);
    const outerEllipse = normalizedX * normalizedX / (outerHorizontal * outerHorizontal) + normalizedY * normalizedY / (outerVertical * outerVertical);
    return innerEllipse > 1 && outerEllipse <= 1;
  }
  /**
   * Returns the framing rectangle of the ellipse as a Rectangle object.
   * @example
   * ```ts
   * // Basic bounds calculation
   * const ellipse = new Ellipse(100, 100, 50, 25);
   * const bounds = ellipse.getBounds();
   * // bounds: x=50, y=75, width=100, height=50
   *
   * // Reuse existing rectangle
   * const rect = new Rectangle();
   * ellipse.getBounds(rect);
   * ```
   * @remarks
   * - Creates Rectangle if none provided
   * - Top-left is (x-halfWidth, y-halfHeight)
   * - Width is halfWidth * 2
   * - Height is halfHeight * 2
   * @param out - Optional Rectangle object to store the result
   * @returns The framing rectangle
   * @see {@link Rectangle} For rectangle properties
   * @see {@link Ellipse.contains} For checking if a point is inside
   */
  getBounds(out2) {
    out2 || (out2 = new Rectangle());
    out2.x = this.x - this.halfWidth;
    out2.y = this.y - this.halfHeight;
    out2.width = this.halfWidth * 2;
    out2.height = this.halfHeight * 2;
    return out2;
  }
  /**
   * Copies another ellipse to this one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Ellipse(100, 100, 50, 25);
   * const target = new Ellipse();
   * target.copyFrom(source);
   * ```
   * @param ellipse - The ellipse to copy from
   * @returns Returns itself
   * @see {@link Ellipse.copyTo} For copying to another ellipse
   * @see {@link Ellipse.clone} For creating new ellipse copy
   */
  copyFrom(ellipse) {
    this.x = ellipse.x;
    this.y = ellipse.y;
    this.halfWidth = ellipse.halfWidth;
    this.halfHeight = ellipse.halfHeight;
    return this;
  }
  /**
   * Copies this ellipse to another one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Ellipse(100, 100, 50, 25);
   * const target = new Ellipse();
   * source.copyTo(target);
   * ```
   * @param ellipse - The ellipse to copy to
   * @returns Returns given parameter
   * @see {@link Ellipse.copyFrom} For copying from another ellipse
   * @see {@link Ellipse.clone} For creating new ellipse copy
   */
  copyTo(ellipse) {
    ellipse.copyFrom(this);
    return ellipse;
  }
  toString() {
    return `[pixi.js/math:Ellipse x=${this.x} y=${this.y} halfWidth=${this.halfWidth} halfHeight=${this.halfHeight}]`;
  }
}
function squaredDistanceToLineSegment(x2, y2, x1, y1, x22, y22) {
  const a2 = x2 - x1;
  const b2 = y2 - y1;
  const c2 = x22 - x1;
  const d2 = y22 - y1;
  const dot = a2 * c2 + b2 * d2;
  const lenSq = c2 * c2 + d2 * d2;
  let param = -1;
  if (lenSq !== 0) {
    param = dot / lenSq;
  }
  let xx;
  let yy;
  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x22;
    yy = y22;
  } else {
    xx = x1 + param * c2;
    yy = y1 + param * d2;
  }
  const dx = x2 - xx;
  const dy = y2 - yy;
  return dx * dx + dy * dy;
}
let tempRect$2;
let tempRect2;
class Polygon {
  /**
   * @param points - This can be an array of Points
   *  that form the polygon, a flat array of numbers that will be interpreted as [x,y, x,y, ...], or
   *  the arguments passed can be all the points of the polygon e.g.
   *  `new Polygon(new Point(), new Point(), ...)`, or the arguments passed can be flat
   *  x,y values e.g. `new Polygon(x,y, x,y, x,y, ...)` where `x` and `y` are Numbers.
   */
  constructor(...points) {
    this.type = "polygon";
    let flat = Array.isArray(points[0]) ? points[0] : points;
    if (typeof flat[0] !== "number") {
      const p2 = [];
      for (let i2 = 0, il = flat.length; i2 < il; i2++) {
        p2.push(flat[i2].x, flat[i2].y);
      }
      flat = p2;
    }
    this.points = flat;
    this.closePath = true;
  }
  /**
   * Determines whether the polygon's points are arranged in a clockwise direction.
   * Uses the shoelace formula (surveyor's formula) to calculate the signed area.
   *
   * A positive area indicates clockwise winding, while negative indicates counter-clockwise.
   *
   * The formula sums up the cross products of adjacent vertices:
   * For each pair of adjacent points (x1,y1) and (x2,y2), we calculate (x1*y2 - x2*y1)
   * The final sum divided by 2 gives the signed area - positive for clockwise.
   * @example
   * ```ts
   * // Check polygon winding
   * const polygon = new Polygon([0, 0, 100, 0, 50, 100]);
   * console.log(polygon.isClockwise()); // Check direction
   *
   * // Use in path construction
   * const hole = new Polygon([25, 25, 75, 25, 75, 75, 25, 75]);
   * if (hole.isClockwise() === shape.isClockwise()) {
   *     hole.points.reverse(); // Reverse for proper hole winding
   * }
   * ```
   * @returns `true` if the polygon's points are arranged clockwise, `false` if counter-clockwise
   */
  isClockwise() {
    let area2 = 0;
    const points = this.points;
    const length = points.length;
    for (let i2 = 0; i2 < length; i2 += 2) {
      const x1 = points[i2];
      const y1 = points[i2 + 1];
      const x2 = points[(i2 + 2) % length];
      const y2 = points[(i2 + 3) % length];
      area2 += (x2 - x1) * (y2 + y1);
    }
    return area2 < 0;
  }
  /**
   * Checks if this polygon completely contains another polygon.
   * Used for detecting holes in shapes, like when parsing SVG paths.
   * @example
   * ```ts
   * // Basic containment check
   * const outerSquare = new Polygon([0,0, 100,0, 100,100, 0,100]); // A square
   * const innerSquare = new Polygon([25,25, 75,25, 75,75, 25,75]); // A smaller square inside
   *
   * outerSquare.containsPolygon(innerSquare); // Returns true
   * innerSquare.containsPolygon(outerSquare); // Returns false
   * ```
   * @remarks
   * - Uses bounds check for quick rejection
   * - Tests all points for containment
   * @param polygon - The polygon to test for containment
   * @returns True if this polygon completely contains the other polygon
   * @see {@link Polygon.contains} For single point testing
   * @see {@link Polygon.getBounds} For bounds calculation
   */
  containsPolygon(polygon) {
    const thisBounds = this.getBounds(tempRect$2);
    const otherBounds = polygon.getBounds(tempRect2);
    if (!thisBounds.containsRect(otherBounds)) {
      return false;
    }
    const points = polygon.points;
    for (let i2 = 0; i2 < points.length; i2 += 2) {
      const x2 = points[i2];
      const y2 = points[i2 + 1];
      if (!this.contains(x2, y2)) {
        return false;
      }
    }
    return true;
  }
  /**
   * Creates a clone of this polygon.
   * @example
   * ```ts
   * // Basic cloning
   * const original = new Polygon([0, 0, 100, 0, 50, 100]);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.points[0] = 10; // Modify first x coordinate
   * ```
   * @returns A copy of the polygon
   * @see {@link Polygon.copyFrom} For copying into existing polygon
   * @see {@link Polygon.copyTo} For copying to another polygon
   */
  clone() {
    const points = this.points.slice();
    const polygon = new Polygon(points);
    polygon.closePath = this.closePath;
    return polygon;
  }
  /**
   * Checks whether the x and y coordinates passed to this function are contained within this polygon.
   * Uses raycasting algorithm for point-in-polygon testing.
   * @example
   * ```ts
   * // Basic containment check
   * const polygon = new Polygon([0, 0, 100, 0, 50, 100]);
   * const isInside = polygon.contains(25, 25); // true
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @returns Whether the x/y coordinates are within this polygon
   * @see {@link Polygon.strokeContains} For checking stroke intersection
   * @see {@link Polygon.containsPolygon} For polygon-in-polygon testing
   */
  contains(x2, y2) {
    let inside = false;
    const length = this.points.length / 2;
    for (let i2 = 0, j2 = length - 1; i2 < length; j2 = i2++) {
      const xi = this.points[i2 * 2];
      const yi = this.points[i2 * 2 + 1];
      const xj = this.points[j2 * 2];
      const yj = this.points[j2 * 2 + 1];
      const intersect = yi > y2 !== yj > y2 && x2 < (xj - xi) * ((y2 - yi) / (yj - yi)) + xi;
      if (intersect) {
        inside = !inside;
      }
    }
    return inside;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this polygon including the stroke.
   * @example
   * ```ts
   * // Basic stroke check
   * const polygon = new Polygon([0, 0, 100, 0, 50, 100]);
   * const isOnStroke = polygon.strokeContains(25, 25, 4); // 4px line width
   *
   * // Check with different alignments
   * const innerStroke = polygon.strokeContains(25, 25, 4, 1);   // Inside
   * const centerStroke = polygon.strokeContains(25, 25, 4, 0.5); // Centered
   * const outerStroke = polygon.strokeContains(25, 25, 4, 0);   // Outside
   * ```
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @param strokeWidth - The width of the line to check
   * @param alignment - The alignment of the stroke (1 = inner, 0.5 = centered, 0 = outer)
   * @returns Whether the x/y coordinates are within this polygon's stroke
   * @see {@link Polygon.contains} For checking fill containment
   * @see {@link Polygon.getBounds} For getting stroke bounds
   */
  strokeContains(x2, y2, strokeWidth, alignment = 0.5) {
    const strokeWidthSquared = strokeWidth * strokeWidth;
    const rightWidthSquared = strokeWidthSquared * (1 - alignment);
    const leftWidthSquared = strokeWidthSquared - rightWidthSquared;
    const { points } = this;
    const iterationLength = points.length - (this.closePath ? 0 : 2);
    for (let i2 = 0; i2 < iterationLength; i2 += 2) {
      const x1 = points[i2];
      const y1 = points[i2 + 1];
      const x22 = points[(i2 + 2) % points.length];
      const y22 = points[(i2 + 3) % points.length];
      const distanceSquared = squaredDistanceToLineSegment(x2, y2, x1, y1, x22, y22);
      const sign2 = Math.sign((x22 - x1) * (y2 - y1) - (y22 - y1) * (x2 - x1));
      if (distanceSquared <= (sign2 < 0 ? leftWidthSquared : rightWidthSquared)) {
        return true;
      }
    }
    return false;
  }
  /**
   * Returns the framing rectangle of the polygon as a Rectangle object.
   * @example
   * ```ts
   * // Basic bounds calculation
   * const polygon = new Polygon([0, 0, 100, 0, 50, 100]);
   * const bounds = polygon.getBounds();
   * // bounds: x=0, y=0, width=100, height=100
   *
   * // Reuse existing rectangle
   * const rect = new Rectangle();
   * polygon.getBounds(rect);
   * ```
   * @param out - Optional rectangle to store the result
   * @returns The framing rectangle
   * @see {@link Rectangle} For rectangle properties
   * @see {@link Polygon.contains} For checking if a point is inside
   */
  getBounds(out2) {
    out2 || (out2 = new Rectangle());
    const points = this.points;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i2 = 0, n2 = points.length; i2 < n2; i2 += 2) {
      const x2 = points[i2];
      const y2 = points[i2 + 1];
      minX = x2 < minX ? x2 : minX;
      maxX = x2 > maxX ? x2 : maxX;
      minY = y2 < minY ? y2 : minY;
      maxY = y2 > maxY ? y2 : maxY;
    }
    out2.x = minX;
    out2.width = maxX - minX;
    out2.y = minY;
    out2.height = maxY - minY;
    return out2;
  }
  /**
   * Copies another polygon to this one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Polygon([0, 0, 100, 0, 50, 100]);
   * const target = new Polygon();
   * target.copyFrom(source);
   * ```
   * @param polygon - The polygon to copy from
   * @returns Returns itself
   * @see {@link Polygon.copyTo} For copying to another polygon
   * @see {@link Polygon.clone} For creating new polygon copy
   */
  copyFrom(polygon) {
    this.points = polygon.points.slice();
    this.closePath = polygon.closePath;
    return this;
  }
  /**
   * Copies this polygon to another one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new Polygon([0, 0, 100, 0, 50, 100]);
   * const target = new Polygon();
   * source.copyTo(target);
   * ```
   * @param polygon - The polygon to copy to
   * @returns Returns given parameter
   * @see {@link Polygon.copyFrom} For copying from another polygon
   * @see {@link Polygon.clone} For creating new polygon copy
   */
  copyTo(polygon) {
    polygon.copyFrom(this);
    return polygon;
  }
  toString() {
    return `[pixi.js/math:PolygoncloseStroke=${this.closePath}points=${this.points.reduce((pointsDesc, currentPoint) => `${pointsDesc}, ${currentPoint}`, "")}]`;
  }
  /**
   * Get the last X coordinate of the polygon.
   * @example
   * ```ts
   * // Basic coordinate access
   * const polygon = new Polygon([0, 0, 100, 200, 300, 400]);
   * console.log(polygon.lastX); // 300
   * ```
   * @readonly
   * @returns The x-coordinate of the last vertex
   * @see {@link Polygon.lastY} For last Y coordinate
   * @see {@link Polygon.points} For raw points array
   */
  get lastX() {
    return this.points[this.points.length - 2];
  }
  /**
   * Get the last Y coordinate of the polygon.
   * @example
   * ```ts
   * // Basic coordinate access
   * const polygon = new Polygon([0, 0, 100, 200, 300, 400]);
   * console.log(polygon.lastY); // 400
   * ```
   * @readonly
   * @returns The y-coordinate of the last vertex
   * @see {@link Polygon.lastX} For last X coordinate
   * @see {@link Polygon.points} For raw points array
   */
  get lastY() {
    return this.points[this.points.length - 1];
  }
  /**
   * Get the last X coordinate of the polygon.
   * @readonly
   * @deprecated since 8.11.0, use {@link Polygon.lastX} instead.
   */
  get x() {
    deprecation("8.11.0", "Polygon.lastX is deprecated, please use Polygon.lastX instead.");
    return this.points[this.points.length - 2];
  }
  /**
   * Get the last Y coordinate of the polygon.
   * @readonly
   * @deprecated since 8.11.0, use {@link Polygon.lastY} instead.
   */
  get y() {
    deprecation("8.11.0", "Polygon.y is deprecated, please use Polygon.lastY instead.");
    return this.points[this.points.length - 1];
  }
  /**
   * Get the first X coordinate of the polygon.
   * @example
   * ```ts
   * // Basic coordinate access
   * const polygon = new Polygon([0, 0, 100, 200, 300, 400]);
   * console.log(polygon.x); // 0
   * ```
   * @readonly
   * @returns The x-coordinate of the first vertex
   * @see {@link Polygon.startY} For first Y coordinate
   * @see {@link Polygon.points} For raw points array
   */
  get startX() {
    return this.points[0];
  }
  /**
   * Get the first Y coordinate of the polygon.
   * @example
   * ```ts
   * // Basic coordinate access
   * const polygon = new Polygon([0, 0, 100, 200, 300, 400]);
   * console.log(polygon.y); // 0
   * ```
   * @readonly
   * @returns The y-coordinate of the first vertex
   * @see {@link Polygon.startX} For first X coordinate
   * @see {@link Polygon.points} For raw points array
   */
  get startY() {
    return this.points[1];
  }
}
const isCornerWithinStroke = (pX, pY, cornerX, cornerY, radius, strokeWidthInner, strokeWidthOuter) => {
  const dx = pX - cornerX;
  const dy = pY - cornerY;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return distance >= radius - strokeWidthInner && distance <= radius + strokeWidthOuter;
};
class RoundedRectangle {
  /**
   * @param x - The X coordinate of the upper-left corner of the rounded rectangle
   * @param y - The Y coordinate of the upper-left corner of the rounded rectangle
   * @param width - The overall width of this rounded rectangle
   * @param height - The overall height of this rounded rectangle
   * @param radius - Controls the radius of the rounded corners
   */
  constructor(x2 = 0, y2 = 0, width = 0, height = 0, radius = 20) {
    this.type = "roundedRectangle";
    this.x = x2;
    this.y = y2;
    this.width = width;
    this.height = height;
    this.radius = radius;
  }
  /**
   * Returns the framing rectangle of the rounded rectangle as a Rectangle object
   * @example
   * ```ts
   * // Basic bounds calculation
   * const rect = new RoundedRectangle(100, 100, 200, 150, 20);
   * const bounds = rect.getBounds();
   * // bounds: x=100, y=100, width=200, height=150
   *
   * // Reuse existing rectangle
   * const out = new Rectangle();
   * rect.getBounds(out);
   * ```
   * @remarks
   * - Rectangle matches outer dimensions
   * - Ignores corner radius
   * @param out - Optional rectangle to store the result
   * @returns The framing rectangle
   * @see {@link Rectangle} For rectangle properties
   * @see {@link RoundedRectangle.contains} For checking if a point is inside
   */
  getBounds(out2) {
    out2 || (out2 = new Rectangle());
    out2.x = this.x;
    out2.y = this.y;
    out2.width = this.width;
    out2.height = this.height;
    return out2;
  }
  /**
   * Creates a clone of this Rounded Rectangle.
   * @example
   * ```ts
   * // Basic cloning
   * const original = new RoundedRectangle(100, 100, 200, 150, 20);
   * const copy = original.clone();
   *
   * // Clone and modify
   * const modified = original.clone();
   * modified.radius = 30;
   * modified.width *= 2;
   *
   * // Verify independence
   * console.log(original.radius);  // 20
   * console.log(modified.radius);  // 30
   * ```
   * @returns A copy of the rounded rectangle
   * @see {@link RoundedRectangle.copyFrom} For copying into existing rectangle
   * @see {@link RoundedRectangle.copyTo} For copying to another rectangle
   */
  clone() {
    return new RoundedRectangle(this.x, this.y, this.width, this.height, this.radius);
  }
  /**
   * Copies another rectangle to this one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new RoundedRectangle(100, 100, 200, 150, 20);
   * const target = new RoundedRectangle();
   * target.copyFrom(source);
   *
   * // Chain with other operations
   * const rect = new RoundedRectangle()
   *     .copyFrom(source)
   *     .getBounds(rect);
   * ```
   * @param rectangle - The rectangle to copy from
   * @returns Returns itself
   * @see {@link RoundedRectangle.copyTo} For copying to another rectangle
   * @see {@link RoundedRectangle.clone} For creating new rectangle copy
   */
  copyFrom(rectangle) {
    this.x = rectangle.x;
    this.y = rectangle.y;
    this.width = rectangle.width;
    this.height = rectangle.height;
    return this;
  }
  /**
   * Copies this rectangle to another one.
   * @example
   * ```ts
   * // Basic copying
   * const source = new RoundedRectangle(100, 100, 200, 150, 20);
   * const target = new RoundedRectangle();
   * source.copyTo(target);
   *
   * // Chain with other operations
   * const result = source
   *     .copyTo(new RoundedRectangle())
   *     .getBounds();
   * ```
   * @param rectangle - The rectangle to copy to
   * @returns Returns given parameter
   * @see {@link RoundedRectangle.copyFrom} For copying from another rectangle
   * @see {@link RoundedRectangle.clone} For creating new rectangle copy
   */
  copyTo(rectangle) {
    rectangle.copyFrom(this);
    return rectangle;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this Rounded Rectangle
   * @example
   * ```ts
   * // Basic containment check
   * const rect = new RoundedRectangle(100, 100, 200, 150, 20);
   * const isInside = rect.contains(150, 125); // true
   * // Check corner radius
   * const corner = rect.contains(100, 100); // false if within corner curve
   * ```
   * @remarks
   * - Returns false if width/height is 0 or negative
   * - Handles rounded corners with radius check
   * @param x - The X coordinate of the point to test
   * @param y - The Y coordinate of the point to test
   * @returns Whether the x/y coordinates are within this Rounded Rectangle
   * @see {@link RoundedRectangle.strokeContains} For checking stroke intersection
   * @see {@link RoundedRectangle.getBounds} For getting containing rectangle
   */
  contains(x2, y2) {
    if (this.width <= 0 || this.height <= 0) {
      return false;
    }
    if (x2 >= this.x && x2 <= this.x + this.width) {
      if (y2 >= this.y && y2 <= this.y + this.height) {
        const radius = Math.max(0, Math.min(this.radius, Math.min(this.width, this.height) / 2));
        if (y2 >= this.y + radius && y2 <= this.y + this.height - radius || x2 >= this.x + radius && x2 <= this.x + this.width - radius) {
          return true;
        }
        let dx = x2 - (this.x + radius);
        let dy = y2 - (this.y + radius);
        const radius2 = radius * radius;
        if (dx * dx + dy * dy <= radius2) {
          return true;
        }
        dx = x2 - (this.x + this.width - radius);
        if (dx * dx + dy * dy <= radius2) {
          return true;
        }
        dy = y2 - (this.y + this.height - radius);
        if (dx * dx + dy * dy <= radius2) {
          return true;
        }
        dx = x2 - (this.x + radius);
        if (dx * dx + dy * dy <= radius2) {
          return true;
        }
      }
    }
    return false;
  }
  /**
   * Checks whether the x and y coordinates given are contained within this rectangle including the stroke.
   * @example
   * ```ts
   * // Basic stroke check
   * const rect = new RoundedRectangle(100, 100, 200, 150, 20);
   * const isOnStroke = rect.strokeContains(150, 100, 4); // 4px line width
   *
   * // Check with different alignments
   * const innerStroke = rect.strokeContains(150, 100, 4, 1);   // Inside
   * const centerStroke = rect.strokeContains(150, 100, 4, 0.5); // Centered
   * const outerStroke = rect.strokeContains(150, 100, 4, 0);   // Outside
   * ```
   * @param pX - The X coordinate of the point to test
   * @param pY - The Y coordinate of the point to test
   * @param strokeWidth - The width of the line to check
   * @param alignment - The alignment of the stroke (1 = inner, 0.5 = centered, 0 = outer)
   * @returns Whether the x/y coordinates are within this rectangle's stroke
   * @see {@link RoundedRectangle.contains} For checking fill containment
   * @see {@link RoundedRectangle.getBounds} For getting stroke bounds
   */
  strokeContains(pX, pY, strokeWidth, alignment = 0.5) {
    const { x: x2, y: y2, width, height, radius } = this;
    const strokeWidthOuter = strokeWidth * (1 - alignment);
    const strokeWidthInner = strokeWidth - strokeWidthOuter;
    const innerX = x2 + radius;
    const innerY = y2 + radius;
    const innerWidth = width - radius * 2;
    const innerHeight = height - radius * 2;
    const rightBound = x2 + width;
    const bottomBound = y2 + height;
    if ((pX >= x2 - strokeWidthOuter && pX <= x2 + strokeWidthInner || pX >= rightBound - strokeWidthInner && pX <= rightBound + strokeWidthOuter) && pY >= innerY && pY <= innerY + innerHeight) {
      return true;
    }
    if ((pY >= y2 - strokeWidthOuter && pY <= y2 + strokeWidthInner || pY >= bottomBound - strokeWidthInner && pY <= bottomBound + strokeWidthOuter) && pX >= innerX && pX <= innerX + innerWidth) {
      return true;
    }
    return (
      // Top-left
      pX < innerX && pY < innerY && isCornerWithinStroke(
        pX,
        pY,
        innerX,
        innerY,
        radius,
        strokeWidthInner,
        strokeWidthOuter
      ) || pX > rightBound - radius && pY < innerY && isCornerWithinStroke(
        pX,
        pY,
        rightBound - radius,
        innerY,
        radius,
        strokeWidthInner,
        strokeWidthOuter
      ) || pX > rightBound - radius && pY > bottomBound - radius && isCornerWithinStroke(
        pX,
        pY,
        rightBound - radius,
        bottomBound - radius,
        radius,
        strokeWidthInner,
        strokeWidthOuter
      ) || pX < innerX && pY > bottomBound - radius && isCornerWithinStroke(
        pX,
        pY,
        innerX,
        bottomBound - radius,
        radius,
        strokeWidthInner,
        strokeWidthOuter
      )
    );
  }
  toString() {
    return `[pixi.js/math:RoundedRectangle x=${this.x} y=${this.y}width=${this.width} height=${this.height} radius=${this.radius}]`;
  }
}
const cachedGroups = {};
function getTextureBatchBindGroup(textures, size, maxTextures) {
  let uid2 = 2166136261;
  for (let i2 = 0; i2 < size; i2++) {
    uid2 ^= textures[i2].uid;
    uid2 = Math.imul(uid2, 16777619);
    uid2 >>>= 0;
  }
  return cachedGroups[uid2] || generateTextureBatchBindGroup(textures, size, uid2, maxTextures);
}
function generateTextureBatchBindGroup(textures, size, key, maxTextures) {
  const bindGroupResources = {};
  let bindIndex = 0;
  for (let i2 = 0; i2 < maxTextures; i2++) {
    const texture = i2 < size ? textures[i2] : Texture.EMPTY.source;
    bindGroupResources[bindIndex++] = texture.source;
    bindGroupResources[bindIndex++] = texture.style;
  }
  const bindGroup = new BindGroup(bindGroupResources);
  cachedGroups[key] = bindGroup;
  return bindGroup;
}
class ViewableBuffer {
  constructor(sizeOrBuffer) {
    if (typeof sizeOrBuffer === "number") {
      this.rawBinaryData = new ArrayBuffer(sizeOrBuffer);
    } else if (sizeOrBuffer instanceof Uint8Array) {
      this.rawBinaryData = sizeOrBuffer.buffer;
    } else {
      this.rawBinaryData = sizeOrBuffer;
    }
    this.uint32View = new Uint32Array(this.rawBinaryData);
    this.float32View = new Float32Array(this.rawBinaryData);
    this.size = this.rawBinaryData.byteLength;
  }
  /** View on the raw binary data as a `Int8Array`. */
  get int8View() {
    if (!this._int8View) {
      this._int8View = new Int8Array(this.rawBinaryData);
    }
    return this._int8View;
  }
  /** View on the raw binary data as a `Uint8Array`. */
  get uint8View() {
    if (!this._uint8View) {
      this._uint8View = new Uint8Array(this.rawBinaryData);
    }
    return this._uint8View;
  }
  /**  View on the raw binary data as a `Int16Array`. */
  get int16View() {
    if (!this._int16View) {
      this._int16View = new Int16Array(this.rawBinaryData);
    }
    return this._int16View;
  }
  /** View on the raw binary data as a `Int32Array`. */
  get int32View() {
    if (!this._int32View) {
      this._int32View = new Int32Array(this.rawBinaryData);
    }
    return this._int32View;
  }
  /** View on the raw binary data as a `Float64Array`. */
  get float64View() {
    if (!this._float64Array) {
      this._float64Array = new Float64Array(this.rawBinaryData);
    }
    return this._float64Array;
  }
  /** View on the raw binary data as a `BigUint64Array`. */
  get bigUint64View() {
    if (!this._bigUint64Array) {
      this._bigUint64Array = new BigUint64Array(this.rawBinaryData);
    }
    return this._bigUint64Array;
  }
  /**
   * Returns the view of the given type.
   * @param type - One of `int8`, `uint8`, `int16`,
   *    `uint16`, `int32`, `uint32`, and `float32`.
   * @returns - typed array of given type
   */
  view(type) {
    return this[`${type}View`];
  }
  /** Destroys all buffer references. Do not use after calling this. */
  destroy() {
    this.rawBinaryData = null;
    this.uint32View = null;
    this.float32View = null;
    this.uint16View = null;
    this._int8View = null;
    this._uint8View = null;
    this._int16View = null;
    this._int32View = null;
    this._float64Array = null;
    this._bigUint64Array = null;
  }
  /**
   * Returns the size of the given type in bytes.
   * @param type - One of `int8`, `uint8`, `int16`,
   *   `uint16`, `int32`, `uint32`, and `float32`.
   * @returns - size of the type in bytes
   */
  static sizeOf(type) {
    switch (type) {
      case "int8":
      case "uint8":
        return 1;
      case "int16":
      case "uint16":
        return 2;
      case "int32":
      case "uint32":
      case "float32":
        return 4;
      default:
        throw new Error(`${type} isn't a valid view type`);
    }
  }
}
function fastCopy(sourceBuffer, destinationBuffer) {
  const lengthDouble = sourceBuffer.byteLength / 8 | 0;
  const sourceFloat64View = new Float64Array(sourceBuffer, 0, lengthDouble);
  const destinationFloat64View = new Float64Array(destinationBuffer, 0, lengthDouble);
  destinationFloat64View.set(sourceFloat64View);
  const remainingBytes = sourceBuffer.byteLength - lengthDouble * 8;
  if (remainingBytes > 0) {
    const sourceUint8View = new Uint8Array(sourceBuffer, lengthDouble * 8, remainingBytes);
    const destinationUint8View = new Uint8Array(destinationBuffer, lengthDouble * 8, remainingBytes);
    destinationUint8View.set(sourceUint8View);
  }
}
const BLEND_TO_NPM = {
  normal: "normal-npm",
  add: "add-npm",
  screen: "screen-npm"
};
var STENCIL_MODES = /* @__PURE__ */ ((STENCIL_MODES2) => {
  STENCIL_MODES2[STENCIL_MODES2["DISABLED"] = 0] = "DISABLED";
  STENCIL_MODES2[STENCIL_MODES2["RENDERING_MASK_ADD"] = 1] = "RENDERING_MASK_ADD";
  STENCIL_MODES2[STENCIL_MODES2["MASK_ACTIVE"] = 2] = "MASK_ACTIVE";
  STENCIL_MODES2[STENCIL_MODES2["INVERSE_MASK_ACTIVE"] = 3] = "INVERSE_MASK_ACTIVE";
  STENCIL_MODES2[STENCIL_MODES2["RENDERING_MASK_REMOVE"] = 4] = "RENDERING_MASK_REMOVE";
  STENCIL_MODES2[STENCIL_MODES2["NONE"] = 5] = "NONE";
  return STENCIL_MODES2;
})(STENCIL_MODES || {});
function getAdjustedBlendModeBlend(blendMode, textureSource) {
  if (textureSource.alphaMode === "no-premultiply-alpha") {
    return BLEND_TO_NPM[blendMode] || blendMode;
  }
  return blendMode;
}
const fragTemplate = [
  "precision mediump float;",
  "void main(void){",
  "float test = 0.1;",
  "%forloop%",
  "gl_FragColor = vec4(0.0);",
  "}"
].join("\n");
function generateIfTestSrc(maxIfs) {
  let src = "";
  for (let i2 = 0; i2 < maxIfs; ++i2) {
    if (i2 > 0) {
      src += "\nelse ";
    }
    if (i2 < maxIfs - 1) {
      src += `if(test == ${i2}.0){}`;
    }
  }
  return src;
}
function checkMaxIfStatementsInShader(maxIfs, gl) {
  if (maxIfs === 0) {
    throw new Error("Invalid value of `0` passed to `checkMaxIfStatementsInShader`");
  }
  const shader = gl.createShader(gl.FRAGMENT_SHADER);
  try {
    while (true) {
      const fragmentSrc = fragTemplate.replace(/%forloop%/gi, generateIfTestSrc(maxIfs));
      gl.shaderSource(shader, fragmentSrc);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        maxIfs = maxIfs / 2 | 0;
      } else {
        break;
      }
    }
  } finally {
    gl.deleteShader(shader);
  }
  return maxIfs;
}
let maxTexturesPerBatchCache = null;
function getMaxTexturesPerBatch() {
  if (maxTexturesPerBatchCache)
    return maxTexturesPerBatchCache;
  const gl = getTestContext();
  maxTexturesPerBatchCache = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
  maxTexturesPerBatchCache = checkMaxIfStatementsInShader(
    maxTexturesPerBatchCache,
    gl
  );
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  return maxTexturesPerBatchCache;
}
class BatchTextureArray {
  constructor() {
    this.ids = /* @__PURE__ */ Object.create(null);
    this.textures = [];
    this.count = 0;
  }
  /** Clear the textures and their locations. */
  clear() {
    for (let i2 = 0; i2 < this.count; i2++) {
      const t2 = this.textures[i2];
      this.textures[i2] = null;
      this.ids[t2.uid] = null;
    }
    this.count = 0;
  }
}
class Batch {
  constructor() {
    this.renderPipeId = "batch";
    this.action = "startBatch";
    this.start = 0;
    this.size = 0;
    this.textures = new BatchTextureArray();
    this.blendMode = "normal";
    this.topology = "triangle-strip";
    this.canBundle = true;
  }
  destroy() {
    this.textures = null;
    this.gpuBindGroup = null;
    this.bindGroup = null;
    this.batcher = null;
  }
}
const batchPool = [];
let batchPoolIndex = 0;
GlobalResourceRegistry.register({
  clear: () => {
    if (batchPool.length > 0) {
      for (const item of batchPool) {
        if (item)
          item.destroy();
      }
    }
    batchPool.length = 0;
    batchPoolIndex = 0;
  }
});
function getBatchFromPool() {
  return batchPoolIndex > 0 ? batchPool[--batchPoolIndex] : new Batch();
}
function returnBatchToPool(batch) {
  batchPool[batchPoolIndex++] = batch;
}
let BATCH_TICK = 0;
const _Batcher = class _Batcher2 {
  constructor(options) {
    this.uid = uid("batcher");
    this.dirty = true;
    this.batchIndex = 0;
    this.batches = [];
    this._elements = [];
    options = { ..._Batcher2.defaultOptions, ...options };
    if (!options.maxTextures) {
      deprecation("v8.8.0", "maxTextures is a required option for Batcher now, please pass it in the options");
      options.maxTextures = getMaxTexturesPerBatch();
    }
    const { maxTextures, attributesInitialSize, indicesInitialSize } = options;
    this.attributeBuffer = new ViewableBuffer(attributesInitialSize * 4);
    this.indexBuffer = new Uint16Array(indicesInitialSize);
    this.maxTextures = maxTextures;
  }
  begin() {
    this.elementSize = 0;
    this.elementStart = 0;
    this.indexSize = 0;
    this.attributeSize = 0;
    for (let i2 = 0; i2 < this.batchIndex; i2++) {
      returnBatchToPool(this.batches[i2]);
    }
    this.batchIndex = 0;
    this._batchIndexStart = 0;
    this._batchIndexSize = 0;
    this.dirty = true;
  }
  add(batchableObject) {
    this._elements[this.elementSize++] = batchableObject;
    batchableObject._indexStart = this.indexSize;
    batchableObject._attributeStart = this.attributeSize;
    batchableObject._batcher = this;
    this.indexSize += batchableObject.indexSize;
    this.attributeSize += batchableObject.attributeSize * this.vertexSize;
  }
  checkAndUpdateTexture(batchableObject, texture) {
    const textureId = batchableObject._batch.textures.ids[texture._source.uid];
    if (!textureId && textureId !== 0)
      return false;
    batchableObject._textureId = textureId;
    batchableObject.texture = texture;
    return true;
  }
  updateElement(batchableObject) {
    this.dirty = true;
    const attributeBuffer = this.attributeBuffer;
    if (batchableObject.packAsQuad) {
      this.packQuadAttributes(
        batchableObject,
        attributeBuffer.float32View,
        attributeBuffer.uint32View,
        batchableObject._attributeStart,
        batchableObject._textureId
      );
    } else {
      this.packAttributes(
        batchableObject,
        attributeBuffer.float32View,
        attributeBuffer.uint32View,
        batchableObject._attributeStart,
        batchableObject._textureId
      );
    }
  }
  /**
   * breaks the batcher. This happens when a batch gets too big,
   * or we need to switch to a different type of rendering (a filter for example)
   * @param instructionSet
   */
  break(instructionSet) {
    const elements = this._elements;
    if (!elements[this.elementStart])
      return;
    let batch = getBatchFromPool();
    let textureBatch = batch.textures;
    textureBatch.clear();
    const firstElement = elements[this.elementStart];
    let blendMode = getAdjustedBlendModeBlend(firstElement.blendMode, firstElement.texture._source);
    let topology = firstElement.topology;
    if (this.attributeSize * 4 > this.attributeBuffer.size) {
      this._resizeAttributeBuffer(this.attributeSize * 4);
    }
    if (this.indexSize > this.indexBuffer.length) {
      this._resizeIndexBuffer(this.indexSize);
    }
    const f32 = this.attributeBuffer.float32View;
    const u32 = this.attributeBuffer.uint32View;
    const indexBuffer = this.indexBuffer;
    let size = this._batchIndexSize;
    let start = this._batchIndexStart;
    let action = "startBatch";
    const maxTextures = this.maxTextures;
    for (let i2 = this.elementStart; i2 < this.elementSize; ++i2) {
      const element = elements[i2];
      elements[i2] = null;
      const texture = element.texture;
      const source2 = texture._source;
      const adjustedBlendMode = getAdjustedBlendModeBlend(element.blendMode, source2);
      const breakRequired = blendMode !== adjustedBlendMode || topology !== element.topology;
      if (source2._batchTick === BATCH_TICK && !breakRequired) {
        element._textureId = source2._textureBindLocation;
        size += element.indexSize;
        if (element.packAsQuad) {
          this.packQuadAttributes(
            element,
            f32,
            u32,
            element._attributeStart,
            element._textureId
          );
          this.packQuadIndex(
            indexBuffer,
            element._indexStart,
            element._attributeStart / this.vertexSize
          );
        } else {
          this.packAttributes(
            element,
            f32,
            u32,
            element._attributeStart,
            element._textureId
          );
          this.packIndex(
            element,
            indexBuffer,
            element._indexStart,
            element._attributeStart / this.vertexSize
          );
        }
        element._batch = batch;
        continue;
      }
      source2._batchTick = BATCH_TICK;
      if (textureBatch.count >= maxTextures || breakRequired) {
        this._finishBatch(
          batch,
          start,
          size - start,
          textureBatch,
          blendMode,
          topology,
          instructionSet,
          action
        );
        action = "renderBatch";
        start = size;
        blendMode = adjustedBlendMode;
        topology = element.topology;
        batch = getBatchFromPool();
        textureBatch = batch.textures;
        textureBatch.clear();
        ++BATCH_TICK;
      }
      element._textureId = source2._textureBindLocation = textureBatch.count;
      textureBatch.ids[source2.uid] = textureBatch.count;
      textureBatch.textures[textureBatch.count++] = source2;
      element._batch = batch;
      size += element.indexSize;
      if (element.packAsQuad) {
        this.packQuadAttributes(
          element,
          f32,
          u32,
          element._attributeStart,
          element._textureId
        );
        this.packQuadIndex(
          indexBuffer,
          element._indexStart,
          element._attributeStart / this.vertexSize
        );
      } else {
        this.packAttributes(
          element,
          f32,
          u32,
          element._attributeStart,
          element._textureId
        );
        this.packIndex(
          element,
          indexBuffer,
          element._indexStart,
          element._attributeStart / this.vertexSize
        );
      }
    }
    if (textureBatch.count > 0) {
      this._finishBatch(
        batch,
        start,
        size - start,
        textureBatch,
        blendMode,
        topology,
        instructionSet,
        action
      );
      start = size;
      ++BATCH_TICK;
    }
    this.elementStart = this.elementSize;
    this._batchIndexStart = start;
    this._batchIndexSize = size;
  }
  _finishBatch(batch, indexStart, indexSize, textureBatch, blendMode, topology, instructionSet, action) {
    batch.gpuBindGroup = null;
    batch.bindGroup = null;
    batch.action = action;
    batch.batcher = this;
    batch.textures = textureBatch;
    batch.blendMode = blendMode;
    batch.topology = topology;
    batch.start = indexStart;
    batch.size = indexSize;
    ++BATCH_TICK;
    this.batches[this.batchIndex++] = batch;
    instructionSet.add(batch);
  }
  finish(instructionSet) {
    this.break(instructionSet);
  }
  /**
   * Resizes the attribute buffer to the given size (1 = 1 float32)
   * @param size - the size in vertices to ensure (not bytes!)
   */
  ensureAttributeBuffer(size) {
    if (size * 4 <= this.attributeBuffer.size)
      return;
    this._resizeAttributeBuffer(size * 4);
  }
  /**
   * Resizes the index buffer to the given size (1 = 1 float32)
   * @param size - the size in vertices to ensure (not bytes!)
   */
  ensureIndexBuffer(size) {
    if (size <= this.indexBuffer.length)
      return;
    this._resizeIndexBuffer(size);
  }
  _resizeAttributeBuffer(size) {
    const newSize = Math.max(size, this.attributeBuffer.size * 2);
    const newArrayBuffer = new ViewableBuffer(newSize);
    fastCopy(this.attributeBuffer.rawBinaryData, newArrayBuffer.rawBinaryData);
    this.attributeBuffer = newArrayBuffer;
  }
  _resizeIndexBuffer(size) {
    const indexBuffer = this.indexBuffer;
    let newSize = Math.max(size, indexBuffer.length * 1.5);
    newSize += newSize % 2;
    const newIndexBuffer = newSize > 65535 ? new Uint32Array(newSize) : new Uint16Array(newSize);
    if (newIndexBuffer.BYTES_PER_ELEMENT !== indexBuffer.BYTES_PER_ELEMENT) {
      for (let i2 = 0; i2 < indexBuffer.length; i2++) {
        newIndexBuffer[i2] = indexBuffer[i2];
      }
    } else {
      fastCopy(indexBuffer.buffer, newIndexBuffer.buffer);
    }
    this.indexBuffer = newIndexBuffer;
  }
  packQuadIndex(indexBuffer, index, indicesOffset) {
    indexBuffer[index] = indicesOffset + 0;
    indexBuffer[index + 1] = indicesOffset + 1;
    indexBuffer[index + 2] = indicesOffset + 2;
    indexBuffer[index + 3] = indicesOffset + 0;
    indexBuffer[index + 4] = indicesOffset + 2;
    indexBuffer[index + 5] = indicesOffset + 3;
  }
  packIndex(element, indexBuffer, index, indicesOffset) {
    const indices = element.indices;
    const size = element.indexSize;
    const indexOffset = element.indexOffset;
    const attributeOffset = element.attributeOffset;
    for (let i2 = 0; i2 < size; i2++) {
      indexBuffer[index++] = indicesOffset + indices[i2 + indexOffset] - attributeOffset;
    }
  }
  /**
   * Destroys the batch and its resources.
   * @param options - destruction options
   * @param options.shader - whether to destroy the associated shader
   */
  destroy(options = {}) {
    if (this.batches === null)
      return;
    for (let i2 = 0; i2 < this.batchIndex; i2++) {
      returnBatchToPool(this.batches[i2]);
    }
    this.batches = null;
    this.geometry.destroy(true);
    this.geometry = null;
    if (options.shader) {
      this.shader?.destroy();
      this.shader = null;
    }
    for (let i2 = 0; i2 < this._elements.length; i2++) {
      if (this._elements[i2])
        this._elements[i2]._batch = null;
    }
    this._elements = null;
    this.indexBuffer = null;
    this.attributeBuffer.destroy();
    this.attributeBuffer = null;
  }
};
_Batcher.defaultOptions = {
  maxTextures: null,
  attributesInitialSize: 4,
  indicesInitialSize: 6
};
let Batcher = _Batcher;
var BufferUsage = /* @__PURE__ */ ((BufferUsage2) => {
  BufferUsage2[BufferUsage2["MAP_READ"] = 1] = "MAP_READ";
  BufferUsage2[BufferUsage2["MAP_WRITE"] = 2] = "MAP_WRITE";
  BufferUsage2[BufferUsage2["COPY_SRC"] = 4] = "COPY_SRC";
  BufferUsage2[BufferUsage2["COPY_DST"] = 8] = "COPY_DST";
  BufferUsage2[BufferUsage2["INDEX"] = 16] = "INDEX";
  BufferUsage2[BufferUsage2["VERTEX"] = 32] = "VERTEX";
  BufferUsage2[BufferUsage2["UNIFORM"] = 64] = "UNIFORM";
  BufferUsage2[BufferUsage2["STORAGE"] = 128] = "STORAGE";
  BufferUsage2[BufferUsage2["INDIRECT"] = 256] = "INDIRECT";
  BufferUsage2[BufferUsage2["QUERY_RESOLVE"] = 512] = "QUERY_RESOLVE";
  BufferUsage2[BufferUsage2["STATIC"] = 1024] = "STATIC";
  return BufferUsage2;
})(BufferUsage || {});
class Buffer extends EventEmitter {
  /**
   * Creates a new Buffer with the given options
   * @param options - the options for the buffer
   */
  constructor(options) {
    let { data, size } = options;
    const { usage, label, shrinkToFit } = options;
    super();
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this._gcLastUsed = -1;
    this.autoGarbageCollect = true;
    this.uid = uid("buffer");
    this._resourceType = "buffer";
    this._resourceId = uid("resource");
    this._touched = 0;
    this._updateID = 1;
    this._dataInt32 = null;
    this.shrinkToFit = true;
    this.destroyed = false;
    if (data instanceof Array) {
      data = new Float32Array(data);
    }
    this._data = data;
    size ?? (size = data?.byteLength);
    const mappedAtCreation = !!data;
    this.descriptor = {
      size,
      usage,
      mappedAtCreation,
      label
    };
    this.shrinkToFit = shrinkToFit ?? true;
  }
  /** the data in the buffer */
  get data() {
    return this._data;
  }
  set data(value) {
    this.setDataWithSize(value, value.length, true);
  }
  get dataInt32() {
    if (!this._dataInt32) {
      this._dataInt32 = new Int32Array(this.data.buffer);
    }
    return this._dataInt32;
  }
  /** whether the buffer is static or not */
  get static() {
    return !!(this.descriptor.usage & BufferUsage.STATIC);
  }
  set static(value) {
    if (value) {
      this.descriptor.usage |= BufferUsage.STATIC;
    } else {
      this.descriptor.usage &= ~BufferUsage.STATIC;
    }
  }
  /**
   * Sets the data in the buffer to the given value. This will immediately update the buffer on the GPU.
   * If you only want to update a subset of the buffer, you can pass in the size of the data.
   * @param value - the data to set
   * @param size - the size of the data in bytes
   * @param syncGPU - should the buffer be updated on the GPU immediately?
   */
  setDataWithSize(value, size, syncGPU) {
    this._updateID++;
    this._updateSize = size * value.BYTES_PER_ELEMENT;
    if (this._data === value) {
      if (syncGPU)
        this.emit("update", this);
      return;
    }
    const oldData = this._data;
    this._data = value;
    this._dataInt32 = null;
    if (!oldData || oldData.length !== value.length) {
      if (!this.shrinkToFit && oldData && value.byteLength < oldData.byteLength) {
        if (syncGPU)
          this.emit("update", this);
      } else {
        this.descriptor.size = value.byteLength;
        this._resourceId = uid("resource");
        this.emit("change", this);
      }
      return;
    }
    if (syncGPU)
      this.emit("update", this);
  }
  /**
   * updates the buffer on the GPU to reflect the data in the buffer.
   * By default it will update the entire buffer. If you only want to update a subset of the buffer,
   * you can pass in the size of the buffer to update.
   * @param sizeInBytes - the new size of the buffer in bytes
   */
  update(sizeInBytes) {
    this._updateSize = sizeInBytes ?? this._updateSize;
    this._updateID++;
    this.emit("update", this);
  }
  /** Unloads the buffer from the GPU */
  unload() {
    this.emit("unload", this);
    for (const key in this._gpuData) {
      this._gpuData[key]?.destroy();
    }
    this._gpuData = /* @__PURE__ */ Object.create(null);
  }
  /** Destroys the buffer */
  destroy() {
    this.destroyed = true;
    this.unload();
    this.emit("destroy", this);
    this.emit("change", this);
    this._data = null;
    this.descriptor = null;
    this.removeAllListeners();
  }
}
function ensureIsBuffer(buffer, index) {
  if (!(buffer instanceof Buffer)) {
    let usage = index ? BufferUsage.INDEX : BufferUsage.VERTEX;
    if (buffer instanceof Array) {
      if (index) {
        buffer = new Uint32Array(buffer);
        usage = BufferUsage.INDEX | BufferUsage.COPY_DST;
      } else {
        buffer = new Float32Array(buffer);
        usage = BufferUsage.VERTEX | BufferUsage.COPY_DST;
      }
    }
    buffer = new Buffer({
      data: buffer,
      label: index ? "index-mesh-buffer" : "vertex-mesh-buffer",
      usage
    });
  }
  return buffer;
}
function getGeometryBounds(geometry, attributeId, bounds) {
  const attribute = geometry.getAttribute(attributeId);
  if (!attribute) {
    bounds.minX = 0;
    bounds.minY = 0;
    bounds.maxX = 0;
    bounds.maxY = 0;
    return bounds;
  }
  const data = attribute.buffer.data;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const byteSize = data.BYTES_PER_ELEMENT;
  const offset = (attribute.offset || 0) / byteSize;
  const stride = (attribute.stride || 2 * 4) / byteSize;
  for (let i2 = offset; i2 < data.length; i2 += stride) {
    const x2 = data[i2];
    const y2 = data[i2 + 1];
    if (x2 > maxX)
      maxX = x2;
    if (y2 > maxY)
      maxY = y2;
    if (x2 < minX)
      minX = x2;
    if (y2 < minY)
      minY = y2;
  }
  bounds.minX = minX;
  bounds.minY = minY;
  bounds.maxX = maxX;
  bounds.maxY = maxY;
  return bounds;
}
function ensureIsAttribute(attribute) {
  if (attribute instanceof Buffer || Array.isArray(attribute) || attribute.BYTES_PER_ELEMENT) {
    attribute = {
      buffer: attribute
    };
  }
  attribute.buffer = ensureIsBuffer(attribute.buffer, false);
  return attribute;
}
class Geometry extends EventEmitter {
  /**
   * Create a new instance of a geometry
   * @param options - The options for the geometry.
   */
  constructor(options = {}) {
    super();
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this.autoGarbageCollect = true;
    this._gcLastUsed = -1;
    this.uid = uid("geometry");
    this._layoutKey = 0;
    this.instanceCount = 1;
    this._bounds = new Bounds();
    this._boundsDirty = true;
    const { attributes, indexBuffer, topology } = options;
    this.buffers = [];
    this.attributes = {};
    if (attributes) {
      for (const i2 in attributes) {
        this.addAttribute(i2, attributes[i2]);
      }
    }
    this.instanceCount = options.instanceCount ?? 1;
    if (indexBuffer) {
      this.addIndex(indexBuffer);
    }
    this.topology = topology || "triangle-list";
  }
  onBufferUpdate() {
    this._boundsDirty = true;
    this.emit("update", this);
  }
  /**
   * Returns the requested attribute.
   * @param id - The name of the attribute required
   * @returns - The attribute requested.
   */
  getAttribute(id) {
    return this.attributes[id];
  }
  /**
   * Returns the index buffer
   * @returns - The index buffer.
   */
  getIndex() {
    return this.indexBuffer;
  }
  /**
   * Returns the requested buffer.
   * @param id - The name of the buffer required.
   * @returns - The buffer requested.
   */
  getBuffer(id) {
    return this.getAttribute(id).buffer;
  }
  /**
   * Used to figure out how many vertices there are in this geometry
   * @returns the number of vertices in the geometry
   */
  getSize() {
    for (const i2 in this.attributes) {
      const attribute = this.attributes[i2];
      const buffer = attribute.buffer;
      return buffer.data.length / (attribute.stride / 4 || attribute.size);
    }
    return 0;
  }
  /**
   * Adds an attribute to the geometry.
   * @param name - The name of the attribute to add.
   * @param attributeOption - The attribute option to add.
   */
  addAttribute(name, attributeOption) {
    const attribute = ensureIsAttribute(attributeOption);
    const bufferIndex = this.buffers.indexOf(attribute.buffer);
    if (bufferIndex === -1) {
      this.buffers.push(attribute.buffer);
      attribute.buffer.on("update", this.onBufferUpdate, this);
      attribute.buffer.on("change", this.onBufferUpdate, this);
    }
    this.attributes[name] = attribute;
  }
  /**
   * Adds an index buffer to the geometry.
   * @param indexBuffer - The index buffer to add. Can be a Buffer, TypedArray, or an array of numbers.
   */
  addIndex(indexBuffer) {
    this.indexBuffer = ensureIsBuffer(indexBuffer, true);
    this.buffers.push(this.indexBuffer);
  }
  /** Returns the bounds of the geometry. */
  get bounds() {
    if (!this._boundsDirty)
      return this._bounds;
    this._boundsDirty = false;
    return getGeometryBounds(this, "aPosition", this._bounds);
  }
  /** Unloads the geometry from the GPU. */
  unload() {
    this.emit("unload", this);
    for (const key in this._gpuData) {
      this._gpuData[key]?.destroy();
    }
    this._gpuData = /* @__PURE__ */ Object.create(null);
  }
  /**
   * destroys the geometry.
   * @param destroyBuffers - destroy the buffers associated with this geometry
   */
  destroy(destroyBuffers = false) {
    this.emit("destroy", this);
    this.removeAllListeners();
    if (destroyBuffers) {
      this.buffers.forEach((buffer) => buffer.destroy());
    }
    this.unload();
    this.indexBuffer?.destroy();
    this.attributes = null;
    this.buffers = null;
    this.indexBuffer = null;
    this._bounds = null;
  }
}
const placeHolderBufferData = new Float32Array(1);
const placeHolderIndexData = new Uint32Array(1);
class BatchGeometry extends Geometry {
  constructor() {
    const vertexSize = 6;
    const attributeBuffer = new Buffer({
      data: placeHolderBufferData,
      label: "attribute-batch-buffer",
      usage: BufferUsage.VERTEX | BufferUsage.COPY_DST,
      shrinkToFit: false
    });
    const indexBuffer = new Buffer({
      data: placeHolderIndexData,
      label: "index-batch-buffer",
      usage: BufferUsage.INDEX | BufferUsage.COPY_DST,
      // | BufferUsage.STATIC,
      shrinkToFit: false
    });
    const stride = vertexSize * 4;
    super({
      attributes: {
        aPosition: {
          buffer: attributeBuffer,
          format: "float32x2",
          stride,
          offset: 0
        },
        aUV: {
          buffer: attributeBuffer,
          format: "float32x2",
          stride,
          offset: 2 * 4
        },
        aColor: {
          buffer: attributeBuffer,
          format: "unorm8x4",
          stride,
          offset: 4 * 4
        },
        aTextureIdAndRound: {
          buffer: attributeBuffer,
          format: "uint16x2",
          stride,
          offset: 5 * 4
        }
      },
      indexBuffer
    });
  }
}
function addBits(srcParts, parts, name) {
  if (srcParts) {
    for (const i2 in srcParts) {
      const id = i2.toLocaleLowerCase();
      const part = parts[id];
      if (part) {
        let sanitisedPart = srcParts[i2];
        if (i2 === "header") {
          sanitisedPart = sanitisedPart.replace(/@in\s+[^;]+;\s*/g, "").replace(/@out\s+[^;]+;\s*/g, "");
        }
        if (name) {
          part.push(`//----${name}----//`);
        }
        part.push(sanitisedPart);
      } else {
        warn(`${i2} placement hook does not exist in shader`);
      }
    }
  }
}
const findHooksRx = /\{\{(.*?)\}\}/g;
function compileHooks(programSrc) {
  const parts = {};
  const partMatches = programSrc.match(findHooksRx)?.map((hook) => hook.replace(/[{()}]/g, "")) ?? [];
  partMatches.forEach((hook) => {
    parts[hook] = [];
  });
  return parts;
}
function extractInputs(fragmentSource, out2) {
  let match;
  const regex = /@in\s+([^;]+);/g;
  while ((match = regex.exec(fragmentSource)) !== null) {
    out2.push(match[1]);
  }
}
function compileInputs(fragments, template, sort = false) {
  const results = [];
  extractInputs(template, results);
  fragments.forEach((fragment2) => {
    if (fragment2.header) {
      extractInputs(fragment2.header, results);
    }
  });
  const mainInput = results;
  if (sort) {
    mainInput.sort();
  }
  const finalString = mainInput.map((inValue, i2) => `       @location(${i2}) ${inValue},`).join("\n");
  let cleanedString = template.replace(/@in\s+[^;]+;\s*/g, "");
  cleanedString = cleanedString.replace("{{in}}", `
${finalString}
`);
  return cleanedString;
}
function extractOutputs(fragmentSource, out2) {
  let match;
  const regex = /@out\s+([^;]+);/g;
  while ((match = regex.exec(fragmentSource)) !== null) {
    out2.push(match[1]);
  }
}
function extractVariableName(value) {
  const regex = /\b(\w+)\s*:/g;
  const match = regex.exec(value);
  return match ? match[1] : "";
}
function stripVariable(value) {
  const regex = /@.*?\s+/g;
  return value.replace(regex, "");
}
function compileOutputs(fragments, template) {
  const results = [];
  extractOutputs(template, results);
  fragments.forEach((fragment2) => {
    if (fragment2.header) {
      extractOutputs(fragment2.header, results);
    }
  });
  let index = 0;
  const mainStruct = results.sort().map((inValue) => {
    if (inValue.indexOf("builtin") > -1) {
      return inValue;
    }
    return `@location(${index++}) ${inValue}`;
  }).join(",\n");
  const mainStart = results.sort().map((inValue) => `       var ${stripVariable(inValue)};`).join("\n");
  const mainEnd = `return VSOutput(
            ${results.sort().map((inValue) => ` ${extractVariableName(inValue)}`).join(",\n")});`;
  let compiledCode = template.replace(/@out\s+[^;]+;\s*/g, "");
  compiledCode = compiledCode.replace("{{struct}}", `
${mainStruct}
`);
  compiledCode = compiledCode.replace("{{start}}", `
${mainStart}
`);
  compiledCode = compiledCode.replace("{{return}}", `
${mainEnd}
`);
  return compiledCode;
}
function injectBits(templateSrc, fragmentParts) {
  let out2 = templateSrc;
  for (const i2 in fragmentParts) {
    const parts = fragmentParts[i2];
    const toInject = parts.join("\n");
    if (toInject.length) {
      out2 = out2.replace(`{{${i2}}}`, `//-----${i2} START-----//
${parts.join("\n")}
//----${i2} FINISH----//`);
    } else {
      out2 = out2.replace(`{{${i2}}}`, "");
    }
  }
  return out2;
}
const cacheMap = /* @__PURE__ */ Object.create(null);
const bitCacheMap = /* @__PURE__ */ new Map();
let CACHE_UID = 0;
function compileHighShader({
  template,
  bits
}) {
  const cacheId = generateCacheId(template, bits);
  if (cacheMap[cacheId])
    return cacheMap[cacheId];
  const { vertex: vertex2, fragment: fragment2 } = compileInputsAndOutputs(template, bits);
  cacheMap[cacheId] = compileBits(vertex2, fragment2, bits);
  return cacheMap[cacheId];
}
function compileHighShaderGl({
  template,
  bits
}) {
  const cacheId = generateCacheId(template, bits);
  if (cacheMap[cacheId])
    return cacheMap[cacheId];
  cacheMap[cacheId] = compileBits(template.vertex, template.fragment, bits);
  return cacheMap[cacheId];
}
function compileInputsAndOutputs(template, bits) {
  const vertexFragments = bits.map((shaderBit) => shaderBit.vertex).filter((v2) => !!v2);
  const fragmentFragments = bits.map((shaderBit) => shaderBit.fragment).filter((v2) => !!v2);
  let compiledVertex = compileInputs(vertexFragments, template.vertex, true);
  compiledVertex = compileOutputs(vertexFragments, compiledVertex);
  const compiledFragment = compileInputs(fragmentFragments, template.fragment, true);
  return {
    vertex: compiledVertex,
    fragment: compiledFragment
  };
}
function generateCacheId(template, bits) {
  return bits.map((highFragment) => {
    if (!bitCacheMap.has(highFragment)) {
      bitCacheMap.set(highFragment, CACHE_UID++);
    }
    return bitCacheMap.get(highFragment);
  }).sort((a2, b2) => a2 - b2).join("-") + template.vertex + template.fragment;
}
function compileBits(vertex2, fragment2, bits) {
  const vertexParts = compileHooks(vertex2);
  const fragmentParts = compileHooks(fragment2);
  bits.forEach((shaderBit) => {
    addBits(shaderBit.vertex, vertexParts, shaderBit.name);
    addBits(shaderBit.fragment, fragmentParts, shaderBit.name);
  });
  return {
    vertex: injectBits(vertex2, vertexParts),
    fragment: injectBits(fragment2, fragmentParts)
  };
}
const vertexGPUTemplate = (
  /* wgsl */
  `
    @in aPosition: vec2<f32>;
    @in aUV: vec2<f32>;

    @out @builtin(position) vPosition: vec4<f32>;
    @out vUV : vec2<f32>;
    @out vColor : vec4<f32>;

    {{header}}

    struct VSOutput {
        {{struct}}
    };

    @vertex
    fn main( {{in}} ) -> VSOutput {

        var worldTransformMatrix = globalUniforms.uWorldTransformMatrix;
        var modelMatrix = mat3x3<f32>(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
          );
        var position = aPosition;
        var uv = aUV;

        {{start}}

        vColor = vec4<f32>(1., 1., 1., 1.);

        {{main}}

        vUV = uv;

        var modelViewProjectionMatrix = globalUniforms.uProjectionMatrix * worldTransformMatrix * modelMatrix;

        vPosition =  vec4<f32>((modelViewProjectionMatrix *  vec3<f32>(position, 1.0)).xy, 0.0, 1.0);

        vColor *= globalUniforms.uWorldColorAlpha;

        {{end}}

        {{return}}
    };
`
);
const fragmentGPUTemplate = (
  /* wgsl */
  `
    @in vUV : vec2<f32>;
    @in vColor : vec4<f32>;

    {{header}}

    @fragment
    fn main(
        {{in}}
      ) -> @location(0) vec4<f32> {

        {{start}}

        var outColor:vec4<f32>;

        {{main}}

        var finalColor:vec4<f32> = outColor * vColor;

        {{end}}

        return finalColor;
      };
`
);
const vertexGlTemplate = (
  /* glsl */
  `
    in vec2 aPosition;
    in vec2 aUV;

    out vec4 vColor;
    out vec2 vUV;

    {{header}}

    void main(void){

        mat3 worldTransformMatrix = uWorldTransformMatrix;
        mat3 modelMatrix = mat3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
          );
        vec2 position = aPosition;
        vec2 uv = aUV;

        {{start}}

        vColor = vec4(1.);

        {{main}}

        vUV = uv;

        mat3 modelViewProjectionMatrix = uProjectionMatrix * worldTransformMatrix * modelMatrix;

        gl_Position = vec4((modelViewProjectionMatrix * vec3(position, 1.0)).xy, 0.0, 1.0);

        vColor *= uWorldColorAlpha;

        {{end}}
    }
`
);
const fragmentGlTemplate = (
  /* glsl */
  `

    in vec4 vColor;
    in vec2 vUV;

    out vec4 finalColor;

    {{header}}

    void main(void) {

        {{start}}

        vec4 outColor;

        {{main}}

        finalColor = outColor * vColor;

        {{end}}
    }
`
);
const globalUniformsBit = {
  name: "global-uniforms-bit",
  vertex: {
    header: (
      /* wgsl */
      `
        struct GlobalUniforms {
            uProjectionMatrix:mat3x3<f32>,
            uWorldTransformMatrix:mat3x3<f32>,
            uWorldColorAlpha: vec4<f32>,
            uResolution: vec2<f32>,
        }

        @group(0) @binding(0) var<uniform> globalUniforms : GlobalUniforms;
        `
    )
  }
};
const globalUniformsBitGl = {
  name: "global-uniforms-bit",
  vertex: {
    header: (
      /* glsl */
      `
          uniform mat3 uProjectionMatrix;
          uniform mat3 uWorldTransformMatrix;
          uniform vec4 uWorldColorAlpha;
          uniform vec2 uResolution;
        `
    )
  }
};
function compileHighShaderGpuProgram({ bits, name }) {
  const source2 = compileHighShader({
    template: {
      fragment: fragmentGPUTemplate,
      vertex: vertexGPUTemplate
    },
    bits: [
      globalUniformsBit,
      ...bits
    ]
  });
  return GpuProgram.from({
    name,
    vertex: {
      source: source2.vertex,
      entryPoint: "main"
    },
    fragment: {
      source: source2.fragment,
      entryPoint: "main"
    }
  });
}
function compileHighShaderGlProgram({ bits, name }) {
  return new GlProgram({
    name,
    ...compileHighShaderGl({
      template: {
        vertex: vertexGlTemplate,
        fragment: fragmentGlTemplate
      },
      bits: [
        globalUniformsBitGl,
        ...bits
      ]
    })
  });
}
const colorBit = {
  name: "color-bit",
  vertex: {
    header: (
      /* wgsl */
      `
            @in aColor: vec4<f32>;
        `
    ),
    main: (
      /* wgsl */
      `
            vColor *= vec4<f32>(aColor.rgb * aColor.a, aColor.a);
        `
    )
  }
};
const colorBitGl = {
  name: "color-bit",
  vertex: {
    header: (
      /* glsl */
      `
            in vec4 aColor;
        `
    ),
    main: (
      /* glsl */
      `
            vColor *= vec4(aColor.rgb * aColor.a, aColor.a);
        `
    )
  }
};
const textureBatchBitGpuCache = {};
function generateBindingSrc(maxTextures) {
  const src = [];
  if (maxTextures === 1) {
    src.push("@group(1) @binding(0) var textureSource1: texture_2d<f32>;");
    src.push("@group(1) @binding(1) var textureSampler1: sampler;");
  } else {
    let bindingIndex = 0;
    for (let i2 = 0; i2 < maxTextures; i2++) {
      src.push(`@group(1) @binding(${bindingIndex++}) var textureSource${i2 + 1}: texture_2d<f32>;`);
      src.push(`@group(1) @binding(${bindingIndex++}) var textureSampler${i2 + 1}: sampler;`);
    }
  }
  return src.join("\n");
}
function generateSampleSrc(maxTextures) {
  const src = [];
  if (maxTextures === 1) {
    src.push("outColor = textureSampleGrad(textureSource1, textureSampler1, vUV, uvDx, uvDy);");
  } else {
    src.push("switch vTextureId {");
    for (let i2 = 0; i2 < maxTextures; i2++) {
      if (i2 === maxTextures - 1) {
        src.push(`  default:{`);
      } else {
        src.push(`  case ${i2}:{`);
      }
      src.push(`      outColor = textureSampleGrad(textureSource${i2 + 1}, textureSampler${i2 + 1}, vUV, uvDx, uvDy);`);
      src.push(`      break;}`);
    }
    src.push(`}`);
  }
  return src.join("\n");
}
function generateTextureBatchBit(maxTextures) {
  if (!textureBatchBitGpuCache[maxTextures]) {
    textureBatchBitGpuCache[maxTextures] = {
      name: "texture-batch-bit",
      vertex: {
        header: `
                @in aTextureIdAndRound: vec2<u32>;
                @out @interpolate(flat) vTextureId : u32;
            `,
        main: `
                vTextureId = aTextureIdAndRound.y;
            `,
        end: `
                if(aTextureIdAndRound.x == 1)
                {
                    vPosition = vec4<f32>(roundPixels(vPosition.xy, globalUniforms.uResolution), vPosition.zw);
                }
            `
      },
      fragment: {
        header: `
                @in @interpolate(flat) vTextureId: u32;

                ${generateBindingSrc(maxTextures)}
            `,
        main: `
                var uvDx = dpdx(vUV);
                var uvDy = dpdy(vUV);

                ${generateSampleSrc(maxTextures)}
            `
      }
    };
  }
  return textureBatchBitGpuCache[maxTextures];
}
const textureBatchBitGlCache = {};
function generateSampleGlSrc(maxTextures) {
  const src = [];
  for (let i2 = 0; i2 < maxTextures; i2++) {
    if (i2 > 0) {
      src.push("else");
    }
    if (i2 < maxTextures - 1) {
      src.push(`if(vTextureId < ${i2}.5)`);
    }
    src.push("{");
    src.push(`	outColor = texture(uTextures[${i2}], vUV);`);
    src.push("}");
  }
  return src.join("\n");
}
function generateTextureBatchBitGl(maxTextures) {
  if (!textureBatchBitGlCache[maxTextures]) {
    textureBatchBitGlCache[maxTextures] = {
      name: "texture-batch-bit",
      vertex: {
        header: `
                in vec2 aTextureIdAndRound;
                out float vTextureId;

            `,
        main: `
                vTextureId = aTextureIdAndRound.y;
            `,
        end: `
                if(aTextureIdAndRound.x == 1.)
                {
                    gl_Position.xy = roundPixels(gl_Position.xy, uResolution);
                }
            `
      },
      fragment: {
        header: `
                in float vTextureId;

                uniform sampler2D uTextures[${maxTextures}];

            `,
        main: `

                ${generateSampleGlSrc(maxTextures)}
            `
      }
    };
  }
  return textureBatchBitGlCache[maxTextures];
}
const roundPixelsBit = {
  name: "round-pixels-bit",
  vertex: {
    header: (
      /* wgsl */
      `
            fn roundPixels(position: vec2<f32>, targetSize: vec2<f32>) -> vec2<f32>
            {
                return (floor(((position * 0.5 + 0.5) * targetSize) + 0.5) / targetSize) * 2.0 - 1.0;
            }
        `
    )
  }
};
const roundPixelsBitGl = {
  name: "round-pixels-bit",
  vertex: {
    header: (
      /* glsl */
      `
            vec2 roundPixels(vec2 position, vec2 targetSize)
            {
                return (floor(((position * 0.5 + 0.5) * targetSize) + 0.5) / targetSize) * 2.0 - 1.0;
            }
        `
    )
  }
};
const batchSamplersUniformGroupHash = {};
function getBatchSamplersUniformGroup(maxTextures) {
  let batchSamplersUniformGroup = batchSamplersUniformGroupHash[maxTextures];
  if (batchSamplersUniformGroup)
    return batchSamplersUniformGroup;
  const sampleValues = new Int32Array(maxTextures);
  for (let i2 = 0; i2 < maxTextures; i2++) {
    sampleValues[i2] = i2;
  }
  batchSamplersUniformGroup = batchSamplersUniformGroupHash[maxTextures] = new UniformGroup({
    uTextures: { value: sampleValues, type: `i32`, size: maxTextures }
  }, { isStatic: true });
  return batchSamplersUniformGroup;
}
class DefaultShader extends Shader {
  constructor(maxTextures) {
    const glProgram = compileHighShaderGlProgram({
      name: "batch",
      bits: [
        colorBitGl,
        generateTextureBatchBitGl(maxTextures),
        roundPixelsBitGl
      ]
    });
    const gpuProgram = compileHighShaderGpuProgram({
      name: "batch",
      bits: [
        colorBit,
        generateTextureBatchBit(maxTextures),
        roundPixelsBit
      ]
    });
    super({
      glProgram,
      gpuProgram,
      resources: {
        batchSamplers: getBatchSamplersUniformGroup(maxTextures)
      }
    });
    this.maxTextures = maxTextures;
  }
}
let defaultShader = null;
const _DefaultBatcher = class _DefaultBatcher2 extends Batcher {
  constructor(options) {
    super(options);
    this.geometry = new BatchGeometry();
    this.name = _DefaultBatcher2.extension.name;
    this.vertexSize = 6;
    defaultShader ?? (defaultShader = new DefaultShader(options.maxTextures));
    this.shader = defaultShader;
  }
  /**
   * Packs the attributes of a DefaultBatchableMeshElement into the provided views.
   * @param element - The DefaultBatchableMeshElement to pack.
   * @param float32View - The Float32Array view to pack into.
   * @param uint32View - The Uint32Array view to pack into.
   * @param index - The starting index in the views.
   * @param textureId - The texture ID to use.
   */
  packAttributes(element, float32View, uint32View, index, textureId) {
    const textureIdAndRound = textureId << 16 | element.roundPixels & 65535;
    const wt = element.transform;
    const a2 = wt.a;
    const b2 = wt.b;
    const c2 = wt.c;
    const d2 = wt.d;
    const tx = wt.tx;
    const ty = wt.ty;
    const { positions, uvs } = element;
    const argb = element.color;
    const offset = element.attributeOffset;
    const end = offset + element.attributeSize;
    for (let i2 = offset; i2 < end; i2++) {
      const i22 = i2 * 2;
      const x2 = positions[i22];
      const y2 = positions[i22 + 1];
      float32View[index++] = a2 * x2 + c2 * y2 + tx;
      float32View[index++] = d2 * y2 + b2 * x2 + ty;
      float32View[index++] = uvs[i22];
      float32View[index++] = uvs[i22 + 1];
      uint32View[index++] = argb;
      uint32View[index++] = textureIdAndRound;
    }
  }
  /**
   * Packs the attributes of a DefaultBatchableQuadElement into the provided views.
   * @param element - The DefaultBatchableQuadElement to pack.
   * @param float32View - The Float32Array view to pack into.
   * @param uint32View - The Uint32Array view to pack into.
   * @param index - The starting index in the views.
   * @param textureId - The texture ID to use.
   */
  packQuadAttributes(element, float32View, uint32View, index, textureId) {
    const texture = element.texture;
    const wt = element.transform;
    const a2 = wt.a;
    const b2 = wt.b;
    const c2 = wt.c;
    const d2 = wt.d;
    const tx = wt.tx;
    const ty = wt.ty;
    const bounds = element.bounds;
    const w0 = bounds.maxX;
    const w1 = bounds.minX;
    const h0 = bounds.maxY;
    const h1 = bounds.minY;
    const uvs = texture.uvs;
    const argb = element.color;
    const textureIdAndRound = textureId << 16 | element.roundPixels & 65535;
    float32View[index + 0] = a2 * w1 + c2 * h1 + tx;
    float32View[index + 1] = d2 * h1 + b2 * w1 + ty;
    float32View[index + 2] = uvs.x0;
    float32View[index + 3] = uvs.y0;
    uint32View[index + 4] = argb;
    uint32View[index + 5] = textureIdAndRound;
    float32View[index + 6] = a2 * w0 + c2 * h1 + tx;
    float32View[index + 7] = d2 * h1 + b2 * w0 + ty;
    float32View[index + 8] = uvs.x1;
    float32View[index + 9] = uvs.y1;
    uint32View[index + 10] = argb;
    uint32View[index + 11] = textureIdAndRound;
    float32View[index + 12] = a2 * w0 + c2 * h0 + tx;
    float32View[index + 13] = d2 * h0 + b2 * w0 + ty;
    float32View[index + 14] = uvs.x2;
    float32View[index + 15] = uvs.y2;
    uint32View[index + 16] = argb;
    uint32View[index + 17] = textureIdAndRound;
    float32View[index + 18] = a2 * w1 + c2 * h0 + tx;
    float32View[index + 19] = d2 * h0 + b2 * w1 + ty;
    float32View[index + 20] = uvs.x3;
    float32View[index + 21] = uvs.y3;
    uint32View[index + 22] = argb;
    uint32View[index + 23] = textureIdAndRound;
  }
  /**
   * Updates the maximum number of textures that can be used in the shader.
   * @param maxTextures - The maximum number of textures that can be used in the shader.
   * @internal
   */
  _updateMaxTextures(maxTextures) {
    if (this.shader.maxTextures === maxTextures)
      return;
    defaultShader = new DefaultShader(maxTextures);
    this.shader = defaultShader;
  }
  destroy() {
    this.shader = null;
    super.destroy();
  }
};
_DefaultBatcher.extension = {
  type: [
    ExtensionType.Batcher
  ],
  name: "default"
};
let DefaultBatcher = _DefaultBatcher;
class GCManagedHash {
  constructor(options) {
    this.items = /* @__PURE__ */ Object.create(null);
    const { renderer, type, onUnload, priority, name } = options;
    this._renderer = renderer;
    renderer.gc.addResourceHash(this, "items", type, priority ?? 0);
    this._onUnload = onUnload;
    this.name = name;
  }
  /**
   * Add an item to the hash. No-op if already added.
   * @param item
   * @returns true if the item was added, false if it was already in the hash
   */
  add(item) {
    if (this.items[item.uid])
      return false;
    this.items[item.uid] = item;
    item.once("unload", this.remove, this);
    item._gcLastUsed = this._renderer.gc.now;
    return true;
  }
  remove(item, ...args) {
    if (!this.items[item.uid])
      return;
    const gpuData = item._gpuData[this._renderer.uid];
    if (!gpuData)
      return;
    this._onUnload?.(item, ...args);
    gpuData.destroy();
    item._gpuData[this._renderer.uid] = null;
    this.items[item.uid] = null;
  }
  removeAll(...args) {
    Object.values(this.items).forEach((item) => item && this.remove(item, ...args));
  }
  destroy(...args) {
    this.removeAll(...args);
    this.items = /* @__PURE__ */ Object.create(null);
    this._renderer = null;
    this._onUnload = null;
  }
}
function buildUvs(vertices, verticesStride, verticesOffset, uvs, uvsOffset, uvsStride, size, matrix = null) {
  let index = 0;
  verticesOffset *= verticesStride;
  uvsOffset *= uvsStride;
  const a2 = matrix.a;
  const b2 = matrix.b;
  const c2 = matrix.c;
  const d2 = matrix.d;
  const tx = matrix.tx;
  const ty = matrix.ty;
  while (index < size) {
    const x2 = vertices[verticesOffset];
    const y2 = vertices[verticesOffset + 1];
    uvs[uvsOffset] = a2 * x2 + c2 * y2 + tx;
    uvs[uvsOffset + 1] = b2 * x2 + d2 * y2 + ty;
    uvsOffset += uvsStride;
    verticesOffset += verticesStride;
    index++;
  }
}
function buildSimpleUvs(uvs, uvsOffset, uvsStride, size) {
  let index = 0;
  uvsOffset *= uvsStride;
  while (index < size) {
    uvs[uvsOffset] = 0;
    uvs[uvsOffset + 1] = 0;
    uvsOffset += uvsStride;
    index++;
  }
}
function transformVertices(vertices, m2, offset, stride, size) {
  const a2 = m2.a;
  const b2 = m2.b;
  const c2 = m2.c;
  const d2 = m2.d;
  const tx = m2.tx;
  const ty = m2.ty;
  offset || (offset = 0);
  stride || (stride = 2);
  size || (size = vertices.length / stride - offset);
  let index = offset * stride;
  for (let i2 = 0; i2 < size; i2++) {
    const x2 = vertices[index];
    const y2 = vertices[index + 1];
    vertices[index] = a2 * x2 + c2 * y2 + tx;
    vertices[index + 1] = b2 * x2 + d2 * y2 + ty;
    index += stride;
  }
}
const identityMatrix = new Matrix();
class BatchableGraphics {
  constructor() {
    this.packAsQuad = false;
    this.batcherName = "default";
    this.topology = "triangle-list";
    this.applyTransform = true;
    this.roundPixels = 0;
    this._batcher = null;
    this._batch = null;
  }
  get uvs() {
    return this.geometryData.uvs;
  }
  get positions() {
    return this.geometryData.vertices;
  }
  get indices() {
    return this.geometryData.indices;
  }
  get blendMode() {
    if (this.renderable && this.applyTransform) {
      return this.renderable.groupBlendMode;
    }
    return "normal";
  }
  get color() {
    const rgb = this.baseColor;
    const bgr = rgb >> 16 | rgb & 65280 | (rgb & 255) << 16;
    const renderable = this.renderable;
    if (renderable) {
      return multiplyHexColors(bgr, renderable.groupColor) + (this.alpha * renderable.groupAlpha * 255 << 24);
    }
    return bgr + (this.alpha * 255 << 24);
  }
  get transform() {
    return this.renderable?.groupTransform || identityMatrix;
  }
  copyTo(gpuBuffer) {
    gpuBuffer.indexOffset = this.indexOffset;
    gpuBuffer.indexSize = this.indexSize;
    gpuBuffer.attributeOffset = this.attributeOffset;
    gpuBuffer.attributeSize = this.attributeSize;
    gpuBuffer.baseColor = this.baseColor;
    gpuBuffer.alpha = this.alpha;
    gpuBuffer.texture = this.texture;
    gpuBuffer.geometryData = this.geometryData;
    gpuBuffer.topology = this.topology;
  }
  reset() {
    this.applyTransform = true;
    this.renderable = null;
    this.topology = "triangle-list";
  }
  destroy() {
    this.renderable = null;
    this.texture = null;
    this.geometryData = null;
    this._batcher = null;
    this._batch = null;
  }
}
const buildCircle = {
  extension: {
    type: ExtensionType.ShapeBuilder,
    name: "circle"
  },
  build(shape, points) {
    let x2;
    let y2;
    let dx;
    let dy;
    let rx;
    let ry;
    if (shape.type === "circle") {
      const circle = shape;
      rx = ry = circle.radius;
      if (rx <= 0) {
        return false;
      }
      x2 = circle.x;
      y2 = circle.y;
      dx = dy = 0;
    } else if (shape.type === "ellipse") {
      const ellipse = shape;
      rx = ellipse.halfWidth;
      ry = ellipse.halfHeight;
      if (rx <= 0 || ry <= 0) {
        return false;
      }
      x2 = ellipse.x;
      y2 = ellipse.y;
      dx = dy = 0;
    } else {
      const roundedRect = shape;
      const halfWidth = roundedRect.width / 2;
      const halfHeight = roundedRect.height / 2;
      x2 = roundedRect.x + halfWidth;
      y2 = roundedRect.y + halfHeight;
      rx = ry = Math.max(0, Math.min(roundedRect.radius, Math.min(halfWidth, halfHeight)));
      dx = halfWidth - rx;
      dy = halfHeight - ry;
    }
    if (dx < 0 || dy < 0) {
      return false;
    }
    const n2 = Math.ceil(2.3 * Math.sqrt(rx + ry));
    const m2 = n2 * 8 + (dx ? 4 : 0) + (dy ? 4 : 0);
    if (m2 === 0) {
      return false;
    }
    if (n2 === 0) {
      points[0] = points[6] = x2 + dx;
      points[1] = points[3] = y2 + dy;
      points[2] = points[4] = x2 - dx;
      points[5] = points[7] = y2 - dy;
      return true;
    }
    let j1 = 0;
    let j2 = n2 * 4 + (dx ? 2 : 0) + 2;
    let j3 = j2;
    let j4 = m2;
    let x0 = dx + rx;
    let y0 = dy;
    let x1 = x2 + x0;
    let x22 = x2 - x0;
    let y1 = y2 + y0;
    points[j1++] = x1;
    points[j1++] = y1;
    points[--j2] = y1;
    points[--j2] = x22;
    if (dy) {
      const y222 = y2 - y0;
      points[j3++] = x22;
      points[j3++] = y222;
      points[--j4] = y222;
      points[--j4] = x1;
    }
    for (let i2 = 1; i2 < n2; i2++) {
      const a2 = Math.PI / 2 * (i2 / n2);
      const x02 = dx + Math.cos(a2) * rx;
      const y02 = dy + Math.sin(a2) * ry;
      const x12 = x2 + x02;
      const x222 = x2 - x02;
      const y12 = y2 + y02;
      const y222 = y2 - y02;
      points[j1++] = x12;
      points[j1++] = y12;
      points[--j2] = y12;
      points[--j2] = x222;
      points[j3++] = x222;
      points[j3++] = y222;
      points[--j4] = y222;
      points[--j4] = x12;
    }
    x0 = dx;
    y0 = dy + ry;
    x1 = x2 + x0;
    x22 = x2 - x0;
    y1 = y2 + y0;
    const y22 = y2 - y0;
    points[j1++] = x1;
    points[j1++] = y1;
    points[--j4] = y22;
    points[--j4] = x1;
    if (dx) {
      points[j1++] = x22;
      points[j1++] = y1;
      points[--j4] = y22;
      points[--j4] = x22;
    }
    return true;
  },
  triangulate(points, vertices, verticesStride, verticesOffset, indices, indicesOffset) {
    if (points.length === 0) {
      return;
    }
    let centerX = 0;
    let centerY = 0;
    for (let i2 = 0; i2 < points.length; i2 += 2) {
      centerX += points[i2];
      centerY += points[i2 + 1];
    }
    centerX /= points.length / 2;
    centerY /= points.length / 2;
    let count2 = verticesOffset;
    vertices[count2 * verticesStride] = centerX;
    vertices[count2 * verticesStride + 1] = centerY;
    const centerIndex = count2++;
    for (let i2 = 0; i2 < points.length; i2 += 2) {
      vertices[count2 * verticesStride] = points[i2];
      vertices[count2 * verticesStride + 1] = points[i2 + 1];
      if (i2 > 0) {
        indices[indicesOffset++] = count2;
        indices[indicesOffset++] = centerIndex;
        indices[indicesOffset++] = count2 - 1;
      }
      count2++;
    }
    indices[indicesOffset++] = centerIndex + 1;
    indices[indicesOffset++] = centerIndex;
    indices[indicesOffset++] = count2 - 1;
  }
};
const buildEllipse = { ...buildCircle, extension: { ...buildCircle.extension, name: "ellipse" } };
const buildRoundedRectangle = { ...buildCircle, extension: { ...buildCircle.extension, name: "roundedRectangle" } };
const closePointEps = 1e-4;
const curveEps = 1e-4;
function getOrientationOfPoints(points) {
  const m2 = points.length;
  if (m2 < 6) {
    return 1;
  }
  let area2 = 0;
  for (let i2 = 0, x1 = points[m2 - 2], y1 = points[m2 - 1]; i2 < m2; i2 += 2) {
    const x2 = points[i2];
    const y2 = points[i2 + 1];
    area2 += (x2 - x1) * (y2 + y1);
    x1 = x2;
    y1 = y2;
  }
  if (area2 < 0) {
    return -1;
  }
  return 1;
}
function square(x2, y2, nx, ny, innerWeight, outerWeight, clockwise, verts) {
  const ix = x2 - nx * innerWeight;
  const iy = y2 - ny * innerWeight;
  const ox = x2 + nx * outerWeight;
  const oy = y2 + ny * outerWeight;
  let exx;
  let eyy;
  if (clockwise) {
    exx = ny;
    eyy = -nx;
  } else {
    exx = -ny;
    eyy = nx;
  }
  const eix = ix + exx;
  const eiy = iy + eyy;
  const eox = ox + exx;
  const eoy = oy + eyy;
  verts.push(eix, eiy);
  verts.push(eox, eoy);
  return 2;
}
function round(cx, cy, sx, sy, ex, ey, verts, clockwise) {
  const cx2p0x = sx - cx;
  const cy2p0y = sy - cy;
  let angle0 = Math.atan2(cx2p0x, cy2p0y);
  let angle1 = Math.atan2(ex - cx, ey - cy);
  if (clockwise && angle0 < angle1) {
    angle0 += Math.PI * 2;
  } else if (!clockwise && angle0 > angle1) {
    angle1 += Math.PI * 2;
  }
  let startAngle = angle0;
  const angleDiff = angle1 - angle0;
  const absAngleDiff = Math.abs(angleDiff);
  const radius = Math.sqrt(cx2p0x * cx2p0x + cy2p0y * cy2p0y);
  const segCount = (15 * absAngleDiff * Math.sqrt(radius) / Math.PI >> 0) + 1;
  const angleInc = angleDiff / segCount;
  startAngle += angleInc;
  if (clockwise) {
    verts.push(cx, cy);
    verts.push(sx, sy);
    for (let i2 = 1, angle = startAngle; i2 < segCount; i2++, angle += angleInc) {
      verts.push(cx, cy);
      verts.push(
        cx + Math.sin(angle) * radius,
        cy + Math.cos(angle) * radius
      );
    }
    verts.push(cx, cy);
    verts.push(ex, ey);
  } else {
    verts.push(sx, sy);
    verts.push(cx, cy);
    for (let i2 = 1, angle = startAngle; i2 < segCount; i2++, angle += angleInc) {
      verts.push(
        cx + Math.sin(angle) * radius,
        cy + Math.cos(angle) * radius
      );
      verts.push(cx, cy);
    }
    verts.push(ex, ey);
    verts.push(cx, cy);
  }
  return segCount * 2;
}
function buildLine(points, lineStyle, flipAlignment, closed, vertices, indices) {
  const eps = closePointEps;
  if (points.length === 0) {
    return;
  }
  const style = lineStyle;
  let alignment = style.alignment;
  if (lineStyle.alignment !== 0.5) {
    let orientation = getOrientationOfPoints(points);
    alignment = (alignment - 0.5) * orientation + 0.5;
  }
  const firstPoint = new Point(points[0], points[1]);
  const lastPoint = new Point(points[points.length - 2], points[points.length - 1]);
  const closedShape = closed;
  const closedPath = Math.abs(firstPoint.x - lastPoint.x) < eps && Math.abs(firstPoint.y - lastPoint.y) < eps;
  if (closedShape) {
    points = points.slice();
    if (closedPath) {
      points.pop();
      points.pop();
      lastPoint.set(points[points.length - 2], points[points.length - 1]);
    }
    const midPointX = (firstPoint.x + lastPoint.x) * 0.5;
    const midPointY = (lastPoint.y + firstPoint.y) * 0.5;
    points.unshift(midPointX, midPointY);
    points.push(midPointX, midPointY);
  }
  const verts = vertices;
  const length = points.length / 2;
  let indexCount = points.length;
  const indexStart = verts.length / 2;
  const width = style.width / 2;
  const widthSquared = width * width;
  const miterLimitSquared = style.miterLimit * style.miterLimit;
  let x0 = points[0];
  let y0 = points[1];
  let x1 = points[2];
  let y1 = points[3];
  let x2 = 0;
  let y2 = 0;
  let perpX = -(y0 - y1);
  let perpY = x0 - x1;
  let perp1x = 0;
  let perp1y = 0;
  let dist = Math.sqrt(perpX * perpX + perpY * perpY);
  perpX /= dist;
  perpY /= dist;
  perpX *= width;
  perpY *= width;
  const ratio = alignment;
  const innerWeight = (1 - ratio) * 2;
  const outerWeight = ratio * 2;
  if (!closedShape) {
    if (style.cap === "round") {
      indexCount += round(
        x0 - perpX * (innerWeight - outerWeight) * 0.5,
        y0 - perpY * (innerWeight - outerWeight) * 0.5,
        x0 - perpX * innerWeight,
        y0 - perpY * innerWeight,
        x0 + perpX * outerWeight,
        y0 + perpY * outerWeight,
        verts,
        true
      ) + 2;
    } else if (style.cap === "square") {
      indexCount += square(x0, y0, perpX, perpY, innerWeight, outerWeight, true, verts);
    }
  }
  verts.push(
    x0 - perpX * innerWeight,
    y0 - perpY * innerWeight
  );
  verts.push(
    x0 + perpX * outerWeight,
    y0 + perpY * outerWeight
  );
  for (let i2 = 1; i2 < length - 1; ++i2) {
    x0 = points[(i2 - 1) * 2];
    y0 = points[(i2 - 1) * 2 + 1];
    x1 = points[i2 * 2];
    y1 = points[i2 * 2 + 1];
    x2 = points[(i2 + 1) * 2];
    y2 = points[(i2 + 1) * 2 + 1];
    perpX = -(y0 - y1);
    perpY = x0 - x1;
    dist = Math.sqrt(perpX * perpX + perpY * perpY);
    perpX /= dist;
    perpY /= dist;
    perpX *= width;
    perpY *= width;
    perp1x = -(y1 - y2);
    perp1y = x1 - x2;
    dist = Math.sqrt(perp1x * perp1x + perp1y * perp1y);
    perp1x /= dist;
    perp1y /= dist;
    perp1x *= width;
    perp1y *= width;
    const dx0 = x1 - x0;
    const dy0 = y0 - y1;
    const dx1 = x1 - x2;
    const dy1 = y2 - y1;
    const dot = dx0 * dx1 + dy0 * dy1;
    const cross = dy0 * dx1 - dy1 * dx0;
    const clockwise = cross < 0;
    if (Math.abs(cross) < 1e-3 * Math.abs(dot)) {
      verts.push(
        x1 - perpX * innerWeight,
        y1 - perpY * innerWeight
      );
      verts.push(
        x1 + perpX * outerWeight,
        y1 + perpY * outerWeight
      );
      if (dot >= 0) {
        if (style.join === "round") {
          indexCount += round(
            x1,
            y1,
            x1 - perpX * innerWeight,
            y1 - perpY * innerWeight,
            x1 - perp1x * innerWeight,
            y1 - perp1y * innerWeight,
            verts,
            false
          ) + 4;
        } else {
          indexCount += 2;
        }
        verts.push(
          x1 - perp1x * outerWeight,
          y1 - perp1y * outerWeight
        );
        verts.push(
          x1 + perp1x * innerWeight,
          y1 + perp1y * innerWeight
        );
      }
      continue;
    }
    const c1 = (-perpX + x0) * (-perpY + y1) - (-perpX + x1) * (-perpY + y0);
    const c2 = (-perp1x + x2) * (-perp1y + y1) - (-perp1x + x1) * (-perp1y + y2);
    const px = (dx0 * c2 - dx1 * c1) / cross;
    const py = (dy1 * c1 - dy0 * c2) / cross;
    const pDist = (px - x1) * (px - x1) + (py - y1) * (py - y1);
    const imx = x1 + (px - x1) * innerWeight;
    const imy = y1 + (py - y1) * innerWeight;
    const omx = x1 - (px - x1) * outerWeight;
    const omy = y1 - (py - y1) * outerWeight;
    const smallerInsideSegmentSq = Math.min(dx0 * dx0 + dy0 * dy0, dx1 * dx1 + dy1 * dy1);
    const insideWeight = clockwise ? innerWeight : outerWeight;
    const smallerInsideDiagonalSq = smallerInsideSegmentSq + insideWeight * insideWeight * widthSquared;
    const insideMiterOk = pDist <= smallerInsideDiagonalSq;
    if (insideMiterOk) {
      if (style.join === "bevel" || pDist / widthSquared > miterLimitSquared) {
        if (clockwise) {
          verts.push(imx, imy);
          verts.push(x1 + perpX * outerWeight, y1 + perpY * outerWeight);
          verts.push(imx, imy);
          verts.push(x1 + perp1x * outerWeight, y1 + perp1y * outerWeight);
        } else {
          verts.push(x1 - perpX * innerWeight, y1 - perpY * innerWeight);
          verts.push(omx, omy);
          verts.push(x1 - perp1x * innerWeight, y1 - perp1y * innerWeight);
          verts.push(omx, omy);
        }
        indexCount += 2;
      } else if (style.join === "round") {
        if (clockwise) {
          verts.push(imx, imy);
          verts.push(x1 + perpX * outerWeight, y1 + perpY * outerWeight);
          indexCount += round(
            x1,
            y1,
            x1 + perpX * outerWeight,
            y1 + perpY * outerWeight,
            x1 + perp1x * outerWeight,
            y1 + perp1y * outerWeight,
            verts,
            true
          ) + 4;
          verts.push(imx, imy);
          verts.push(x1 + perp1x * outerWeight, y1 + perp1y * outerWeight);
        } else {
          verts.push(x1 - perpX * innerWeight, y1 - perpY * innerWeight);
          verts.push(omx, omy);
          indexCount += round(
            x1,
            y1,
            x1 - perpX * innerWeight,
            y1 - perpY * innerWeight,
            x1 - perp1x * innerWeight,
            y1 - perp1y * innerWeight,
            verts,
            false
          ) + 4;
          verts.push(x1 - perp1x * innerWeight, y1 - perp1y * innerWeight);
          verts.push(omx, omy);
        }
      } else {
        verts.push(imx, imy);
        verts.push(omx, omy);
      }
    } else {
      verts.push(x1 - perpX * innerWeight, y1 - perpY * innerWeight);
      verts.push(x1 + perpX * outerWeight, y1 + perpY * outerWeight);
      if (style.join === "round") {
        if (clockwise) {
          indexCount += round(
            x1,
            y1,
            x1 + perpX * outerWeight,
            y1 + perpY * outerWeight,
            x1 + perp1x * outerWeight,
            y1 + perp1y * outerWeight,
            verts,
            true
          ) + 2;
        } else {
          indexCount += round(
            x1,
            y1,
            x1 - perpX * innerWeight,
            y1 - perpY * innerWeight,
            x1 - perp1x * innerWeight,
            y1 - perp1y * innerWeight,
            verts,
            false
          ) + 2;
        }
      } else if (style.join === "miter" && pDist / widthSquared <= miterLimitSquared) {
        if (clockwise) {
          verts.push(omx, omy);
          verts.push(omx, omy);
        } else {
          verts.push(imx, imy);
          verts.push(imx, imy);
        }
        indexCount += 2;
      }
      verts.push(x1 - perp1x * innerWeight, y1 - perp1y * innerWeight);
      verts.push(x1 + perp1x * outerWeight, y1 + perp1y * outerWeight);
      indexCount += 2;
    }
  }
  x0 = points[(length - 2) * 2];
  y0 = points[(length - 2) * 2 + 1];
  x1 = points[(length - 1) * 2];
  y1 = points[(length - 1) * 2 + 1];
  perpX = -(y0 - y1);
  perpY = x0 - x1;
  dist = Math.sqrt(perpX * perpX + perpY * perpY);
  perpX /= dist;
  perpY /= dist;
  perpX *= width;
  perpY *= width;
  verts.push(x1 - perpX * innerWeight, y1 - perpY * innerWeight);
  verts.push(x1 + perpX * outerWeight, y1 + perpY * outerWeight);
  if (!closedShape) {
    if (style.cap === "round") {
      indexCount += round(
        x1 - perpX * (innerWeight - outerWeight) * 0.5,
        y1 - perpY * (innerWeight - outerWeight) * 0.5,
        x1 - perpX * innerWeight,
        y1 - perpY * innerWeight,
        x1 + perpX * outerWeight,
        y1 + perpY * outerWeight,
        verts,
        false
      ) + 2;
    } else if (style.cap === "square") {
      indexCount += square(x1, y1, perpX, perpY, innerWeight, outerWeight, false, verts);
    }
  }
  const eps2 = curveEps * curveEps;
  for (let i2 = indexStart; i2 < indexCount + indexStart - 2; ++i2) {
    x0 = verts[i2 * 2];
    y0 = verts[i2 * 2 + 1];
    x1 = verts[(i2 + 1) * 2];
    y1 = verts[(i2 + 1) * 2 + 1];
    x2 = verts[(i2 + 2) * 2];
    y2 = verts[(i2 + 2) * 2 + 1];
    if (Math.abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) < eps2) {
      continue;
    }
    indices.push(i2, i2 + 1, i2 + 2);
  }
}
function buildPixelLine(points, closed, vertices, indices) {
  const eps = closePointEps;
  if (points.length === 0) {
    return;
  }
  const fx = points[0];
  const fy = points[1];
  const lx = points[points.length - 2];
  const ly = points[points.length - 1];
  const closePath = closed || Math.abs(fx - lx) < eps && Math.abs(fy - ly) < eps;
  const verts = vertices;
  const length = points.length / 2;
  const indexStart = verts.length / 2;
  for (let i2 = 0; i2 < length; i2++) {
    verts.push(points[i2 * 2]);
    verts.push(points[i2 * 2 + 1]);
  }
  for (let i2 = 0; i2 < length - 1; i2++) {
    indices.push(indexStart + i2, indexStart + i2 + 1);
  }
  if (closePath) {
    indices.push(indexStart + length - 1, indexStart);
  }
}
function triangulateWithHoles(points, holes, vertices, verticesStride, verticesOffset, indices, indicesOffset) {
  const triangles = earcut(points, holes, 2);
  if (!triangles) {
    return;
  }
  for (let i2 = 0; i2 < triangles.length; i2 += 3) {
    indices[indicesOffset++] = triangles[i2] + verticesOffset;
    indices[indicesOffset++] = triangles[i2 + 1] + verticesOffset;
    indices[indicesOffset++] = triangles[i2 + 2] + verticesOffset;
  }
  let index = verticesOffset * verticesStride;
  for (let i2 = 0; i2 < points.length; i2 += 2) {
    vertices[index] = points[i2];
    vertices[index + 1] = points[i2 + 1];
    index += verticesStride;
  }
}
const emptyArray = [];
const buildPolygon = {
  extension: {
    type: ExtensionType.ShapeBuilder,
    name: "polygon"
  },
  build(shape, points) {
    for (let i2 = 0; i2 < shape.points.length; i2++) {
      points[i2] = shape.points[i2];
    }
    return true;
  },
  triangulate(points, vertices, verticesStride, verticesOffset, indices, indicesOffset) {
    triangulateWithHoles(points, emptyArray, vertices, verticesStride, verticesOffset, indices, indicesOffset);
  }
};
const buildRectangle = {
  extension: {
    type: ExtensionType.ShapeBuilder,
    name: "rectangle"
  },
  build(shape, points) {
    const rectData = shape;
    const x2 = rectData.x;
    const y2 = rectData.y;
    const width = rectData.width;
    const height = rectData.height;
    if (!(width > 0 && height > 0)) {
      return false;
    }
    points[0] = x2;
    points[1] = y2;
    points[2] = x2 + width;
    points[3] = y2;
    points[4] = x2 + width;
    points[5] = y2 + height;
    points[6] = x2;
    points[7] = y2 + height;
    return true;
  },
  triangulate(points, vertices, verticesStride, verticesOffset, indices, indicesOffset) {
    let count2 = 0;
    verticesOffset *= verticesStride;
    vertices[verticesOffset + count2] = points[0];
    vertices[verticesOffset + count2 + 1] = points[1];
    count2 += verticesStride;
    vertices[verticesOffset + count2] = points[2];
    vertices[verticesOffset + count2 + 1] = points[3];
    count2 += verticesStride;
    vertices[verticesOffset + count2] = points[6];
    vertices[verticesOffset + count2 + 1] = points[7];
    count2 += verticesStride;
    vertices[verticesOffset + count2] = points[4];
    vertices[verticesOffset + count2 + 1] = points[5];
    count2 += verticesStride;
    const verticesIndex = verticesOffset / verticesStride;
    indices[indicesOffset++] = verticesIndex;
    indices[indicesOffset++] = verticesIndex + 1;
    indices[indicesOffset++] = verticesIndex + 2;
    indices[indicesOffset++] = verticesIndex + 1;
    indices[indicesOffset++] = verticesIndex + 3;
    indices[indicesOffset++] = verticesIndex + 2;
  }
};
const buildTriangle = {
  extension: {
    type: ExtensionType.ShapeBuilder,
    name: "triangle"
  },
  build(shape, points) {
    points[0] = shape.x;
    points[1] = shape.y;
    points[2] = shape.x2;
    points[3] = shape.y2;
    points[4] = shape.x3;
    points[5] = shape.y3;
    return true;
  },
  triangulate(points, vertices, verticesStride, verticesOffset, indices, indicesOffset) {
    let count2 = 0;
    verticesOffset *= verticesStride;
    vertices[verticesOffset + count2] = points[0];
    vertices[verticesOffset + count2 + 1] = points[1];
    count2 += verticesStride;
    vertices[verticesOffset + count2] = points[2];
    vertices[verticesOffset + count2 + 1] = points[3];
    count2 += verticesStride;
    vertices[verticesOffset + count2] = points[4];
    vertices[verticesOffset + count2 + 1] = points[5];
    const verticesIndex = verticesOffset / verticesStride;
    indices[indicesOffset++] = verticesIndex;
    indices[indicesOffset++] = verticesIndex + 1;
    indices[indicesOffset++] = verticesIndex + 2;
  }
};
const tempTextureMatrix$1 = new Matrix();
const tempRect$1 = new Rectangle();
function generateTextureMatrix(out2, style, shape, matrix) {
  const textureMatrix = style.matrix ? out2.copyFrom(style.matrix).invert() : out2.identity();
  if (style.textureSpace === "local") {
    const bounds = shape.getBounds(tempRect$1);
    if (style.width) {
      bounds.pad(style.width);
    }
    const { x: tx, y: ty } = bounds;
    const sx = 1 / bounds.width;
    const sy = 1 / bounds.height;
    const mTx = -tx * sx;
    const mTy = -ty * sy;
    const a1 = textureMatrix.a;
    const b1 = textureMatrix.b;
    const c1 = textureMatrix.c;
    const d1 = textureMatrix.d;
    textureMatrix.a *= sx;
    textureMatrix.b *= sx;
    textureMatrix.c *= sy;
    textureMatrix.d *= sy;
    textureMatrix.tx = mTx * a1 + mTy * c1 + textureMatrix.tx;
    textureMatrix.ty = mTx * b1 + mTy * d1 + textureMatrix.ty;
  } else {
    textureMatrix.translate(style.texture.frame.x, style.texture.frame.y);
    textureMatrix.scale(1 / style.texture.source.width, 1 / style.texture.source.height);
  }
  const sourceStyle = style.texture.source.style;
  if (!(style.fill instanceof FillGradient) && sourceStyle.addressMode === "clamp-to-edge") {
    sourceStyle.addressMode = "repeat";
    sourceStyle.update();
  }
  if (matrix) {
    textureMatrix.append(tempTextureMatrix$1.copyFrom(matrix).invert());
  }
  return textureMatrix;
}
const shapeBuilders = {};
extensions.handleByMap(ExtensionType.ShapeBuilder, shapeBuilders);
extensions.add(buildRectangle, buildPolygon, buildTriangle, buildCircle, buildEllipse, buildRoundedRectangle);
const tempRect = new Rectangle();
const tempTextureMatrix = new Matrix();
function buildContextBatches(context2, gpuContext) {
  const { geometryData, batches } = gpuContext;
  batches.length = 0;
  geometryData.indices.length = 0;
  geometryData.vertices.length = 0;
  geometryData.uvs.length = 0;
  for (let i2 = 0; i2 < context2.instructions.length; i2++) {
    const instruction = context2.instructions[i2];
    if (instruction.action === "texture") {
      addTextureToGeometryData(instruction.data, batches, geometryData);
    } else if (instruction.action === "fill" || instruction.action === "stroke") {
      const isStroke = instruction.action === "stroke";
      const shapePath = instruction.data.path.shapePath;
      const style = instruction.data.style;
      const hole = instruction.data.hole;
      if (isStroke && hole) {
        addShapePathToGeometryData(hole.shapePath, style, true, batches, geometryData);
      }
      if (hole) {
        shapePath.shapePrimitives[shapePath.shapePrimitives.length - 1].holes = hole.shapePath.shapePrimitives;
      }
      addShapePathToGeometryData(shapePath, style, isStroke, batches, geometryData);
    }
  }
}
function addTextureToGeometryData(data, batches, geometryData) {
  const points = [];
  const build = shapeBuilders.rectangle;
  const rect = tempRect;
  rect.x = data.dx;
  rect.y = data.dy;
  rect.width = data.dw;
  rect.height = data.dh;
  const matrix = data.transform;
  if (!build.build(rect, points)) {
    return;
  }
  const { vertices, uvs, indices } = geometryData;
  const indexOffset = indices.length;
  const vertOffset = vertices.length / 2;
  if (matrix) {
    transformVertices(points, matrix);
  }
  build.triangulate(points, vertices, 2, vertOffset, indices, indexOffset);
  const texture = data.image;
  const textureUvs = texture.uvs;
  uvs.push(
    textureUvs.x0,
    textureUvs.y0,
    textureUvs.x1,
    textureUvs.y1,
    textureUvs.x3,
    textureUvs.y3,
    textureUvs.x2,
    textureUvs.y2
  );
  const graphicsBatch = BigPool.get(BatchableGraphics);
  graphicsBatch.indexOffset = indexOffset;
  graphicsBatch.indexSize = indices.length - indexOffset;
  graphicsBatch.attributeOffset = vertOffset;
  graphicsBatch.attributeSize = vertices.length / 2 - vertOffset;
  graphicsBatch.baseColor = data.style;
  graphicsBatch.alpha = data.alpha;
  graphicsBatch.texture = texture;
  graphicsBatch.geometryData = geometryData;
  batches.push(graphicsBatch);
}
function addShapePathToGeometryData(shapePath, style, isStroke, batches, geometryData) {
  const { vertices, uvs, indices } = geometryData;
  shapePath.shapePrimitives.forEach(({ shape, transform: matrix, holes }) => {
    const points = [];
    const build = shapeBuilders[shape.type];
    if (!build.build(shape, points)) {
      return;
    }
    const indexOffset = indices.length;
    const vertOffset = vertices.length / 2;
    let topology = "triangle-list";
    if (matrix) {
      transformVertices(points, matrix);
    }
    if (!isStroke) {
      if (holes) {
        const holeIndices = [];
        const otherPoints = points.slice();
        const holeArrays = getHoleArrays(holes);
        holeArrays.forEach((holePoints) => {
          holeIndices.push(otherPoints.length / 2);
          otherPoints.push(...holePoints);
        });
        triangulateWithHoles(otherPoints, holeIndices, vertices, 2, vertOffset, indices, indexOffset);
      } else {
        build.triangulate(points, vertices, 2, vertOffset, indices, indexOffset);
      }
    } else {
      const close = shape.closePath ?? true;
      const lineStyle = style;
      if (!lineStyle.pixelLine) {
        buildLine(points, lineStyle, false, close, vertices, indices);
      } else {
        buildPixelLine(points, close, vertices, indices);
        topology = "line-list";
      }
    }
    const uvsOffset = uvs.length / 2;
    const texture = style.texture;
    if (texture !== Texture.WHITE) {
      const textureMatrix = generateTextureMatrix(tempTextureMatrix, style, shape, matrix);
      buildUvs(vertices, 2, vertOffset, uvs, uvsOffset, 2, vertices.length / 2 - vertOffset, textureMatrix);
    } else {
      buildSimpleUvs(uvs, uvsOffset, 2, vertices.length / 2 - vertOffset);
    }
    const graphicsBatch = BigPool.get(BatchableGraphics);
    graphicsBatch.indexOffset = indexOffset;
    graphicsBatch.indexSize = indices.length - indexOffset;
    graphicsBatch.attributeOffset = vertOffset;
    graphicsBatch.attributeSize = vertices.length / 2 - vertOffset;
    graphicsBatch.baseColor = style.color;
    graphicsBatch.alpha = style.alpha;
    graphicsBatch.texture = texture;
    graphicsBatch.geometryData = geometryData;
    graphicsBatch.topology = topology;
    batches.push(graphicsBatch);
  });
}
function getHoleArrays(holePrimitives) {
  const holeArrays = [];
  for (let k2 = 0; k2 < holePrimitives.length; k2++) {
    const holePrimitive = holePrimitives[k2].shape;
    const holePoints = [];
    const holeBuilder = shapeBuilders[holePrimitive.type];
    if (holeBuilder.build(holePrimitive, holePoints)) {
      holeArrays.push(holePoints);
    }
  }
  return holeArrays;
}
class GpuGraphicsContext {
  constructor() {
    this.batches = [];
    this.geometryData = {
      vertices: [],
      uvs: [],
      indices: []
    };
  }
  reset() {
    if (this.batches) {
      this.batches.forEach((batch) => {
        BigPool.return(batch);
      });
    }
    if (this.graphicsData) {
      BigPool.return(this.graphicsData);
    }
    this.isBatchable = false;
    this.context = null;
    this.batches.length = 0;
    this.geometryData.indices.length = 0;
    this.geometryData.vertices.length = 0;
    this.geometryData.uvs.length = 0;
    this.graphicsData = null;
  }
  destroy() {
    this.reset();
    this.batches = null;
    this.geometryData = null;
  }
}
class GraphicsContextRenderData {
  constructor() {
    this.instructions = new InstructionSet();
  }
  init(options) {
    const maxTextures = options.maxTextures;
    this.batcher ? this.batcher._updateMaxTextures(maxTextures) : this.batcher = new DefaultBatcher({ maxTextures });
    this.instructions.reset();
  }
  /**
   * @deprecated since version 8.0.0
   * Use `batcher.geometry` instead.
   * @see {Batcher#geometry}
   */
  get geometry() {
    deprecation(v8_3_4, "GraphicsContextRenderData#geometry is deprecated, please use batcher.geometry instead.");
    return this.batcher.geometry;
  }
  destroy() {
    this.batcher.destroy();
    this.instructions.destroy();
    this.batcher = null;
    this.instructions = null;
  }
}
const _GraphicsContextSystem = class _GraphicsContextSystem2 {
  constructor(renderer) {
    this._renderer = renderer;
    this._managedContexts = new GCManagedHash({ renderer, type: "resource", name: "graphicsContext" });
  }
  /**
   * Runner init called, update the default options
   * @ignore
   */
  init(options) {
    _GraphicsContextSystem2.defaultOptions.bezierSmoothness = options?.bezierSmoothness ?? _GraphicsContextSystem2.defaultOptions.bezierSmoothness;
  }
  /**
   * Returns the render data for a given GraphicsContext.
   * @param context - The GraphicsContext to get the render data for.
   * @internal
   */
  getContextRenderData(context2) {
    return context2._gpuData[this._renderer.uid].graphicsData || this._initContextRenderData(context2);
  }
  /**
   * Updates the GPU context for a given GraphicsContext.
   * If the context is dirty, it will rebuild the batches and geometry data.
   * @param context - The GraphicsContext to update.
   * @returns The updated GpuGraphicsContext.
   * @internal
   */
  updateGpuContext(context2) {
    const hasContext = !!context2._gpuData[this._renderer.uid];
    const gpuContext = context2._gpuData[this._renderer.uid] || this._initContext(context2);
    if (context2.dirty || !hasContext) {
      if (hasContext) {
        gpuContext.reset();
      }
      buildContextBatches(context2, gpuContext);
      const batchMode = context2.batchMode;
      if (context2.customShader || batchMode === "no-batch") {
        gpuContext.isBatchable = false;
      } else if (batchMode === "auto") {
        gpuContext.isBatchable = gpuContext.geometryData.vertices.length < 400;
      } else {
        gpuContext.isBatchable = true;
      }
      context2.dirty = false;
    }
    return gpuContext;
  }
  /**
   * Returns the GpuGraphicsContext for a given GraphicsContext.
   * If it does not exist, it will initialize a new one.
   * @param context - The GraphicsContext to get the GpuGraphicsContext for.
   * @returns The GpuGraphicsContext for the given GraphicsContext.
   * @internal
   */
  getGpuContext(context2) {
    return context2._gpuData[this._renderer.uid] || this._initContext(context2);
  }
  _initContextRenderData(context2) {
    const graphicsData = BigPool.get(GraphicsContextRenderData, {
      maxTextures: this._renderer.limits.maxBatchableTextures
    });
    const gpuContext = context2._gpuData[this._renderer.uid];
    const { batches, geometryData } = gpuContext;
    gpuContext.graphicsData = graphicsData;
    const vertexSize = geometryData.vertices.length;
    const indexSize = geometryData.indices.length;
    for (let i2 = 0; i2 < batches.length; i2++) {
      batches[i2].applyTransform = false;
    }
    const batcher = graphicsData.batcher;
    batcher.ensureAttributeBuffer(vertexSize);
    batcher.ensureIndexBuffer(indexSize);
    batcher.begin();
    for (let i2 = 0; i2 < batches.length; i2++) {
      const batch = batches[i2];
      batcher.add(batch);
    }
    batcher.finish(graphicsData.instructions);
    const geometry = batcher.geometry;
    geometry.indexBuffer.setDataWithSize(batcher.indexBuffer, batcher.indexSize, true);
    geometry.buffers[0].setDataWithSize(batcher.attributeBuffer.float32View, batcher.attributeSize, true);
    const drawBatches = batcher.batches;
    for (let i2 = 0; i2 < drawBatches.length; i2++) {
      const batch = drawBatches[i2];
      batch.bindGroup = getTextureBatchBindGroup(
        batch.textures.textures,
        batch.textures.count,
        this._renderer.limits.maxBatchableTextures
      );
    }
    return graphicsData;
  }
  _initContext(context2) {
    const gpuContext = new GpuGraphicsContext();
    gpuContext.context = context2;
    context2._gpuData[this._renderer.uid] = gpuContext;
    this._managedContexts.add(context2);
    return gpuContext;
  }
  destroy() {
    this._managedContexts.destroy();
    this._renderer = null;
  }
};
_GraphicsContextSystem.extension = {
  type: [
    ExtensionType.WebGLSystem,
    ExtensionType.WebGPUSystem,
    ExtensionType.CanvasSystem
  ],
  name: "graphicsContext"
};
_GraphicsContextSystem.defaultOptions = {
  /**
   * A value from 0 to 1 that controls the smoothness of bezier curves (the higher the smoother)
   * @default 0.5
   */
  bezierSmoothness: 0.5
};
let GraphicsContextSystem = _GraphicsContextSystem;
const RECURSION_LIMIT$1 = 8;
const FLT_EPSILON$1 = 11920929e-14;
const PATH_DISTANCE_EPSILON$1 = 1;
function buildAdaptiveBezier(points, sX, sY, cp1x, cp1y, cp2x, cp2y, eX, eY, smoothness) {
  const scale = 1;
  const smoothing = Math.min(
    0.99,
    // a value of 1.0 actually inverts smoothing, so we cap it at 0.99
    Math.max(0, smoothness ?? GraphicsContextSystem.defaultOptions.bezierSmoothness)
  );
  let distanceTolerance = (PATH_DISTANCE_EPSILON$1 - smoothing) / scale;
  distanceTolerance *= distanceTolerance;
  begin$1(sX, sY, cp1x, cp1y, cp2x, cp2y, eX, eY, points, distanceTolerance);
  return points;
}
function begin$1(sX, sY, cp1x, cp1y, cp2x, cp2y, eX, eY, points, distanceTolerance) {
  recursive$1(sX, sY, cp1x, cp1y, cp2x, cp2y, eX, eY, points, distanceTolerance, 0);
  points.push(eX, eY);
}
function recursive$1(x1, y1, x2, y2, x3, y3, x4, y4, points, distanceTolerance, level) {
  if (level > RECURSION_LIMIT$1) {
    return;
  }
  const x12 = (x1 + x2) / 2;
  const y12 = (y1 + y2) / 2;
  const x23 = (x2 + x3) / 2;
  const y23 = (y2 + y3) / 2;
  const x34 = (x3 + x4) / 2;
  const y34 = (y3 + y4) / 2;
  const x123 = (x12 + x23) / 2;
  const y123 = (y12 + y23) / 2;
  const x234 = (x23 + x34) / 2;
  const y234 = (y23 + y34) / 2;
  const x1234 = (x123 + x234) / 2;
  const y1234 = (y123 + y234) / 2;
  if (level > 0) {
    let dx = x4 - x1;
    let dy = y4 - y1;
    const d2 = Math.abs((x2 - x4) * dy - (y2 - y4) * dx);
    const d3 = Math.abs((x3 - x4) * dy - (y3 - y4) * dx);
    if (d2 > FLT_EPSILON$1 && d3 > FLT_EPSILON$1) {
      if ((d2 + d3) * (d2 + d3) <= distanceTolerance * (dx * dx + dy * dy)) {
        {
          points.push(x1234, y1234);
          return;
        }
      }
    } else if (d2 > FLT_EPSILON$1) {
      if (d2 * d2 <= distanceTolerance * (dx * dx + dy * dy)) {
        {
          points.push(x1234, y1234);
          return;
        }
      }
    } else if (d3 > FLT_EPSILON$1) {
      if (d3 * d3 <= distanceTolerance * (dx * dx + dy * dy)) {
        {
          points.push(x1234, y1234);
          return;
        }
      }
    } else {
      dx = x1234 - (x1 + x4) / 2;
      dy = y1234 - (y1 + y4) / 2;
      if (dx * dx + dy * dy <= distanceTolerance) {
        points.push(x1234, y1234);
        return;
      }
    }
  }
  recursive$1(x1, y1, x12, y12, x123, y123, x1234, y1234, points, distanceTolerance, level + 1);
  recursive$1(x1234, y1234, x234, y234, x34, y34, x4, y4, points, distanceTolerance, level + 1);
}
const RECURSION_LIMIT = 8;
const FLT_EPSILON = 11920929e-14;
const PATH_DISTANCE_EPSILON = 1;
function buildAdaptiveQuadratic(points, sX, sY, cp1x, cp1y, eX, eY, smoothness) {
  const scale = 1;
  const smoothing = Math.min(
    0.99,
    // a value of 1.0 actually inverts smoothing, so we cap it at 0.99
    Math.max(0, smoothness ?? GraphicsContextSystem.defaultOptions.bezierSmoothness)
  );
  let distanceTolerance = (PATH_DISTANCE_EPSILON - smoothing) / scale;
  distanceTolerance *= distanceTolerance;
  begin(sX, sY, cp1x, cp1y, eX, eY, points, distanceTolerance);
  return points;
}
function begin(sX, sY, cp1x, cp1y, eX, eY, points, distanceTolerance) {
  recursive(points, sX, sY, cp1x, cp1y, eX, eY, distanceTolerance, 0);
  points.push(eX, eY);
}
function recursive(points, x1, y1, x2, y2, x3, y3, distanceTolerance, level) {
  if (level > RECURSION_LIMIT) {
    return;
  }
  const x12 = (x1 + x2) / 2;
  const y12 = (y1 + y2) / 2;
  const x23 = (x2 + x3) / 2;
  const y23 = (y2 + y3) / 2;
  const x123 = (x12 + x23) / 2;
  const y123 = (y12 + y23) / 2;
  let dx = x3 - x1;
  let dy = y3 - y1;
  const d2 = Math.abs((x2 - x3) * dy - (y2 - y3) * dx);
  if (d2 > FLT_EPSILON) {
    if (d2 * d2 <= distanceTolerance * (dx * dx + dy * dy)) {
      {
        points.push(x123, y123);
        return;
      }
    }
  } else {
    dx = x123 - (x1 + x3) / 2;
    dy = y123 - (y1 + y3) / 2;
    if (dx * dx + dy * dy <= distanceTolerance) {
      points.push(x123, y123);
      return;
    }
  }
  recursive(points, x1, y1, x12, y12, x123, y123, distanceTolerance, level + 1);
  recursive(points, x123, y123, x23, y23, x3, y3, distanceTolerance, level + 1);
}
function buildArc(points, x2, y2, radius, start, end, clockwise, steps) {
  let dist = Math.abs(start - end);
  if (!clockwise && start > end) {
    dist = 2 * Math.PI - dist;
  } else if (clockwise && end > start) {
    dist = 2 * Math.PI - dist;
  }
  steps || (steps = Math.max(6, Math.floor(6 * Math.pow(radius, 1 / 3) * (dist / Math.PI))));
  steps = Math.max(steps, 3);
  let f2 = dist / steps;
  let t2 = start;
  f2 *= clockwise ? -1 : 1;
  for (let i2 = 0; i2 < steps + 1; i2++) {
    const cs = Math.cos(t2);
    const sn = Math.sin(t2);
    const nx = x2 + cs * radius;
    const ny = y2 + sn * radius;
    points.push(nx, ny);
    t2 += f2;
  }
}
function buildArcTo(points, x1, y1, x2, y2, radius) {
  const fromX = points[points.length - 2];
  const fromY = points[points.length - 1];
  const a1 = fromY - y1;
  const b1 = fromX - x1;
  const a2 = y2 - y1;
  const b2 = x2 - x1;
  const mm = Math.abs(a1 * b2 - b1 * a2);
  if (mm < 1e-8 || radius === 0) {
    if (points[points.length - 2] !== x1 || points[points.length - 1] !== y1) {
      points.push(x1, y1);
    }
    return;
  }
  const dd = a1 * a1 + b1 * b1;
  const cc = a2 * a2 + b2 * b2;
  const tt = a1 * a2 + b1 * b2;
  const k1 = radius * Math.sqrt(dd) / mm;
  const k2 = radius * Math.sqrt(cc) / mm;
  const j1 = k1 * tt / dd;
  const j2 = k2 * tt / cc;
  const cx = k1 * b2 + k2 * b1;
  const cy = k1 * a2 + k2 * a1;
  const px = b1 * (k2 + j1);
  const py = a1 * (k2 + j1);
  const qx = b2 * (k1 + j2);
  const qy = a2 * (k1 + j2);
  const startAngle = Math.atan2(py - cy, px - cx);
  const endAngle = Math.atan2(qy - cy, qx - cx);
  buildArc(
    points,
    cx + x1,
    cy + y1,
    radius,
    startAngle,
    endAngle,
    b1 * a2 > b2 * a1
  );
}
const TAU = Math.PI * 2;
const out = {
  centerX: 0,
  centerY: 0,
  ang1: 0,
  ang2: 0
};
const mapToEllipse = ({ x: x2, y: y2 }, rx, ry, cosPhi, sinPhi, centerX, centerY, out2) => {
  x2 *= rx;
  y2 *= ry;
  const xp = cosPhi * x2 - sinPhi * y2;
  const yp = sinPhi * x2 + cosPhi * y2;
  out2.x = xp + centerX;
  out2.y = yp + centerY;
  return out2;
};
function approxUnitArc(ang1, ang2) {
  const a1 = ang2 === -1.5707963267948966 ? -0.551915024494 : 4 / 3 * Math.tan(ang2 / 4);
  const a2 = ang2 === 1.5707963267948966 ? 0.551915024494 : a1;
  const x1 = Math.cos(ang1);
  const y1 = Math.sin(ang1);
  const x2 = Math.cos(ang1 + ang2);
  const y2 = Math.sin(ang1 + ang2);
  return [
    {
      x: x1 - y1 * a2,
      y: y1 + x1 * a2
    },
    {
      x: x2 + y2 * a2,
      y: y2 - x2 * a2
    },
    {
      x: x2,
      y: y2
    }
  ];
}
const vectorAngle = (ux2, uy2, vx2, vy2) => {
  const sign2 = ux2 * vy2 - uy2 * vx2 < 0 ? -1 : 1;
  let dot = ux2 * vx2 + uy2 * vy2;
  if (dot > 1) {
    dot = 1;
  }
  if (dot < -1) {
    dot = -1;
  }
  return sign2 * Math.acos(dot);
};
const getArcCenter = (px, py, cx, cy, rx, ry, largeArcFlag, sweepFlag, sinPhi, cosPhi, pxp, pyp, out2) => {
  const rxSq = Math.pow(rx, 2);
  const rySq = Math.pow(ry, 2);
  const pxpSq = Math.pow(pxp, 2);
  const pypSq = Math.pow(pyp, 2);
  let radicant = rxSq * rySq - rxSq * pypSq - rySq * pxpSq;
  if (radicant < 0) {
    radicant = 0;
  }
  radicant /= rxSq * pypSq + rySq * pxpSq;
  radicant = Math.sqrt(radicant) * (largeArcFlag === sweepFlag ? -1 : 1);
  const centerXp = radicant * rx / ry * pyp;
  const centerYp = radicant * -ry / rx * pxp;
  const centerX = cosPhi * centerXp - sinPhi * centerYp + (px + cx) / 2;
  const centerY = sinPhi * centerXp + cosPhi * centerYp + (py + cy) / 2;
  const vx1 = (pxp - centerXp) / rx;
  const vy1 = (pyp - centerYp) / ry;
  const vx2 = (-pxp - centerXp) / rx;
  const vy2 = (-pyp - centerYp) / ry;
  const ang1 = vectorAngle(1, 0, vx1, vy1);
  let ang2 = vectorAngle(vx1, vy1, vx2, vy2);
  if (sweepFlag === 0 && ang2 > 0) {
    ang2 -= TAU;
  }
  if (sweepFlag === 1 && ang2 < 0) {
    ang2 += TAU;
  }
  out2.centerX = centerX;
  out2.centerY = centerY;
  out2.ang1 = ang1;
  out2.ang2 = ang2;
};
function buildArcToSvg(points, px, py, cx, cy, rx, ry, xAxisRotation = 0, largeArcFlag = 0, sweepFlag = 0) {
  if (rx === 0 || ry === 0) {
    return;
  }
  const sinPhi = Math.sin(xAxisRotation * TAU / 360);
  const cosPhi = Math.cos(xAxisRotation * TAU / 360);
  const pxp = cosPhi * (px - cx) / 2 + sinPhi * (py - cy) / 2;
  const pyp = -sinPhi * (px - cx) / 2 + cosPhi * (py - cy) / 2;
  if (pxp === 0 && pyp === 0) {
    return;
  }
  rx = Math.abs(rx);
  ry = Math.abs(ry);
  const lambda = Math.pow(pxp, 2) / Math.pow(rx, 2) + Math.pow(pyp, 2) / Math.pow(ry, 2);
  if (lambda > 1) {
    rx *= Math.sqrt(lambda);
    ry *= Math.sqrt(lambda);
  }
  getArcCenter(
    px,
    py,
    cx,
    cy,
    rx,
    ry,
    largeArcFlag,
    sweepFlag,
    sinPhi,
    cosPhi,
    pxp,
    pyp,
    out
  );
  let { ang1, ang2 } = out;
  const { centerX, centerY } = out;
  let ratio = Math.abs(ang2) / (TAU / 4);
  if (Math.abs(1 - ratio) < 1e-7) {
    ratio = 1;
  }
  const segments = Math.max(Math.ceil(ratio), 1);
  ang2 /= segments;
  let lastX = points[points.length - 2];
  let lastY = points[points.length - 1];
  const outCurvePoint = { x: 0, y: 0 };
  for (let i2 = 0; i2 < segments; i2++) {
    const curve = approxUnitArc(ang1, ang2);
    const { x: x1, y: y1 } = mapToEllipse(curve[0], rx, ry, cosPhi, sinPhi, centerX, centerY, outCurvePoint);
    const { x: x2, y: y2 } = mapToEllipse(curve[1], rx, ry, cosPhi, sinPhi, centerX, centerY, outCurvePoint);
    const { x: x3, y: y3 } = mapToEllipse(curve[2], rx, ry, cosPhi, sinPhi, centerX, centerY, outCurvePoint);
    buildAdaptiveBezier(
      points,
      lastX,
      lastY,
      x1,
      y1,
      x2,
      y2,
      x3,
      y3
    );
    lastX = x3;
    lastY = y3;
    ang1 += ang2;
  }
}
function roundedShapeArc(g2, points, radius) {
  const vecFrom = (p2, pp) => {
    const x2 = pp.x - p2.x;
    const y2 = pp.y - p2.y;
    const len = Math.sqrt(x2 * x2 + y2 * y2);
    const nx = x2 / len;
    const ny = y2 / len;
    return { len, nx, ny };
  };
  const sharpCorner = (i2, p2) => {
    if (i2 === 0) {
      g2.moveTo(p2.x, p2.y);
    } else {
      g2.lineTo(p2.x, p2.y);
    }
  };
  let p1 = points[points.length - 1];
  for (let i2 = 0; i2 < points.length; i2++) {
    const p2 = points[i2 % points.length];
    const pRadius = p2.radius ?? radius;
    if (pRadius <= 0) {
      sharpCorner(i2, p2);
      p1 = p2;
      continue;
    }
    const p3 = points[(i2 + 1) % points.length];
    const v1 = vecFrom(p2, p1);
    const v2 = vecFrom(p2, p3);
    if (v1.len < 1e-4 || v2.len < 1e-4) {
      sharpCorner(i2, p2);
      p1 = p2;
      continue;
    }
    let angle = Math.asin(v1.nx * v2.ny - v1.ny * v2.nx);
    let radDirection = 1;
    let drawDirection = false;
    if (v1.nx * v2.nx - v1.ny * -v2.ny < 0) {
      if (angle < 0) {
        angle = Math.PI + angle;
      } else {
        angle = Math.PI - angle;
        radDirection = -1;
        drawDirection = true;
      }
    } else if (angle > 0) {
      radDirection = -1;
      drawDirection = true;
    }
    const halfAngle = angle / 2;
    let cRadius;
    let lenOut = Math.abs(
      Math.cos(halfAngle) * pRadius / Math.sin(halfAngle)
    );
    if (lenOut > Math.min(v1.len / 2, v2.len / 2)) {
      lenOut = Math.min(v1.len / 2, v2.len / 2);
      cRadius = Math.abs(lenOut * Math.sin(halfAngle) / Math.cos(halfAngle));
    } else {
      cRadius = pRadius;
    }
    const cX = p2.x + v2.nx * lenOut + -v2.ny * cRadius * radDirection;
    const cY = p2.y + v2.ny * lenOut + v2.nx * cRadius * radDirection;
    const startAngle = Math.atan2(v1.ny, v1.nx) + Math.PI / 2 * radDirection;
    const endAngle = Math.atan2(v2.ny, v2.nx) - Math.PI / 2 * radDirection;
    if (i2 === 0) {
      g2.moveTo(
        cX + Math.cos(startAngle) * cRadius,
        cY + Math.sin(startAngle) * cRadius
      );
    }
    g2.arc(cX, cY, cRadius, startAngle, endAngle, drawDirection);
    p1 = p2;
  }
}
function roundedShapeQuadraticCurve(g2, points, radius, smoothness) {
  const distance = (p1, p2) => Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
  const pointLerp = (p1, p2, t2) => ({
    x: p1.x + (p2.x - p1.x) * t2,
    y: p1.y + (p2.y - p1.y) * t2
  });
  const numPoints = points.length;
  for (let i2 = 0; i2 < numPoints; i2++) {
    const thisPoint = points[(i2 + 1) % numPoints];
    const pRadius = thisPoint.radius ?? radius;
    if (pRadius <= 0) {
      if (i2 === 0) {
        g2.moveTo(thisPoint.x, thisPoint.y);
      } else {
        g2.lineTo(thisPoint.x, thisPoint.y);
      }
      continue;
    }
    const lastPoint = points[i2];
    const nextPoint = points[(i2 + 2) % numPoints];
    const lastEdgeLength = distance(lastPoint, thisPoint);
    let start;
    if (lastEdgeLength < 1e-4) {
      start = thisPoint;
    } else {
      const lastOffsetDistance = Math.min(lastEdgeLength / 2, pRadius);
      start = pointLerp(
        thisPoint,
        lastPoint,
        lastOffsetDistance / lastEdgeLength
      );
    }
    const nextEdgeLength = distance(nextPoint, thisPoint);
    let end;
    if (nextEdgeLength < 1e-4) {
      end = thisPoint;
    } else {
      const nextOffsetDistance = Math.min(nextEdgeLength / 2, pRadius);
      end = pointLerp(
        thisPoint,
        nextPoint,
        nextOffsetDistance / nextEdgeLength
      );
    }
    if (i2 === 0) {
      g2.moveTo(start.x, start.y);
    } else {
      g2.lineTo(start.x, start.y);
    }
    g2.quadraticCurveTo(thisPoint.x, thisPoint.y, end.x, end.y, smoothness);
  }
}
const tempRectangle = new Rectangle();
class ShapePath {
  constructor(graphicsPath2D) {
    this.shapePrimitives = [];
    this._currentPoly = null;
    this._bounds = new Bounds();
    this._graphicsPath2D = graphicsPath2D;
    this.signed = graphicsPath2D.checkForHoles;
  }
  /**
   * Sets the starting point for a new sub-path. Any subsequent drawing commands are considered part of this path.
   * @param x - The x-coordinate for the starting point.
   * @param y - The y-coordinate for the starting point.
   * @returns The instance of the current object for chaining.
   */
  moveTo(x2, y2) {
    this.startPoly(x2, y2);
    return this;
  }
  /**
   * Connects the current point to a new point with a straight line. This method updates the current path.
   * @param x - The x-coordinate of the new point to connect to.
   * @param y - The y-coordinate of the new point to connect to.
   * @returns The instance of the current object for chaining.
   */
  lineTo(x2, y2) {
    this._ensurePoly();
    const points = this._currentPoly.points;
    const fromX = points[points.length - 2];
    const fromY = points[points.length - 1];
    if (fromX !== x2 || fromY !== y2) {
      points.push(x2, y2);
    }
    return this;
  }
  /**
   * Adds an arc to the path. The arc is centered at (x, y)
   *  position with radius `radius` starting at `startAngle` and ending at `endAngle`.
   * @param x - The x-coordinate of the arc's center.
   * @param y - The y-coordinate of the arc's center.
   * @param radius - The radius of the arc.
   * @param startAngle - The starting angle of the arc, in radians.
   * @param endAngle - The ending angle of the arc, in radians.
   * @param counterclockwise - Specifies whether the arc should be drawn in the anticlockwise direction. False by default.
   * @returns The instance of the current object for chaining.
   */
  arc(x2, y2, radius, startAngle, endAngle, counterclockwise) {
    this._ensurePoly(false);
    const points = this._currentPoly.points;
    buildArc(points, x2, y2, radius, startAngle, endAngle, counterclockwise);
    return this;
  }
  /**
   * Adds an arc to the path with the arc tangent to the line joining two specified points.
   * The arc radius is specified by `radius`.
   * @param x1 - The x-coordinate of the first point.
   * @param y1 - The y-coordinate of the first point.
   * @param x2 - The x-coordinate of the second point.
   * @param y2 - The y-coordinate of the second point.
   * @param radius - The radius of the arc.
   * @returns The instance of the current object for chaining.
   */
  arcTo(x1, y1, x2, y2, radius) {
    this._ensurePoly();
    const points = this._currentPoly.points;
    buildArcTo(points, x1, y1, x2, y2, radius);
    return this;
  }
  /**
   * Adds an SVG-style arc to the path, allowing for elliptical arcs based on the SVG spec.
   * @param rx - The x-radius of the ellipse.
   * @param ry - The y-radius of the ellipse.
   * @param xAxisRotation - The rotation of the ellipse's x-axis relative
   * to the x-axis of the coordinate system, in degrees.
   * @param largeArcFlag - Determines if the arc should be greater than or less than 180 degrees.
   * @param sweepFlag - Determines if the arc should be swept in a positive angle direction.
   * @param x - The x-coordinate of the arc's end point.
   * @param y - The y-coordinate of the arc's end point.
   * @returns The instance of the current object for chaining.
   */
  arcToSvg(rx, ry, xAxisRotation, largeArcFlag, sweepFlag, x2, y2) {
    const points = this._currentPoly.points;
    buildArcToSvg(
      points,
      this._currentPoly.lastX,
      this._currentPoly.lastY,
      x2,
      y2,
      rx,
      ry,
      xAxisRotation,
      largeArcFlag,
      sweepFlag
    );
    return this;
  }
  /**
   * Adds a cubic Bezier curve to the path.
   * It requires three points: the first two are control points and the third one is the end point.
   * The starting point is the last point in the current path.
   * @param cp1x - The x-coordinate of the first control point.
   * @param cp1y - The y-coordinate of the first control point.
   * @param cp2x - The x-coordinate of the second control point.
   * @param cp2y - The y-coordinate of the second control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothness - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2, smoothness) {
    this._ensurePoly();
    const currentPoly = this._currentPoly;
    buildAdaptiveBezier(
      this._currentPoly.points,
      currentPoly.lastX,
      currentPoly.lastY,
      cp1x,
      cp1y,
      cp2x,
      cp2y,
      x2,
      y2,
      smoothness
    );
    return this;
  }
  /**
   * Adds a quadratic curve to the path. It requires two points: the control point and the end point.
   * The starting point is the last point in the current path.
   * @param cp1x - The x-coordinate of the control point.
   * @param cp1y - The y-coordinate of the control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothing - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  quadraticCurveTo(cp1x, cp1y, x2, y2, smoothing) {
    this._ensurePoly();
    const currentPoly = this._currentPoly;
    buildAdaptiveQuadratic(
      this._currentPoly.points,
      currentPoly.lastX,
      currentPoly.lastY,
      cp1x,
      cp1y,
      x2,
      y2,
      smoothing
    );
    return this;
  }
  /**
   * Closes the current path by drawing a straight line back to the start.
   * If the shape is already closed or there are no points in the path, this method does nothing.
   * @returns The instance of the current object for chaining.
   */
  closePath() {
    this.endPoly(true);
    return this;
  }
  /**
   * Adds another path to the current path. This method allows for the combination of multiple paths into one.
   * @param path - The `GraphicsPath` object representing the path to add.
   * @param transform - An optional `Matrix` object to apply a transformation to the path before adding it.
   * @returns The instance of the current object for chaining.
   */
  addPath(path2, transform) {
    this.endPoly();
    if (transform && !transform.isIdentity()) {
      path2 = path2.clone(true);
      path2.transform(transform);
    }
    const shapePrimitives = this.shapePrimitives;
    const start = shapePrimitives.length;
    for (let i2 = 0; i2 < path2.instructions.length; i2++) {
      const instruction = path2.instructions[i2];
      this[instruction.action](...instruction.data);
    }
    if (path2.checkForHoles && shapePrimitives.length - start > 1) {
      let mainShape = null;
      for (let i2 = start; i2 < shapePrimitives.length; i2++) {
        const shapePrimitive = shapePrimitives[i2];
        if (shapePrimitive.shape.type === "polygon") {
          const polygon = shapePrimitive.shape;
          const mainPolygon = mainShape?.shape;
          if (mainPolygon && mainPolygon.containsPolygon(polygon)) {
            mainShape.holes || (mainShape.holes = []);
            mainShape.holes.push(shapePrimitive);
            shapePrimitives.copyWithin(i2, i2 + 1);
            shapePrimitives.length--;
            i2--;
          } else {
            mainShape = shapePrimitive;
          }
        }
      }
    }
    return this;
  }
  /**
   * Finalizes the drawing of the current path. Optionally, it can close the path.
   * @param closePath - A boolean indicating whether to close the path after finishing. False by default.
   */
  finish(closePath = false) {
    this.endPoly(closePath);
  }
  /**
   * Draws a rectangle shape. This method adds a new rectangle path to the current drawing.
   * @param x - The x-coordinate of the top-left corner of the rectangle.
   * @param y - The y-coordinate of the top-left corner of the rectangle.
   * @param w - The width of the rectangle.
   * @param h - The height of the rectangle.
   * @param transform - An optional `Matrix` object to apply a transformation to the rectangle.
   * @returns The instance of the current object for chaining.
   */
  rect(x2, y2, w2, h2, transform) {
    this.drawShape(new Rectangle(x2, y2, w2, h2), transform);
    return this;
  }
  /**
   * Draws a circle shape. This method adds a new circle path to the current drawing.
   * @param x - The x-coordinate of the center of the circle.
   * @param y - The y-coordinate of the center of the circle.
   * @param radius - The radius of the circle.
   * @param transform - An optional `Matrix` object to apply a transformation to the circle.
   * @returns The instance of the current object for chaining.
   */
  circle(x2, y2, radius, transform) {
    this.drawShape(new Circle(x2, y2, radius), transform);
    return this;
  }
  /**
   * Draws a polygon shape. This method allows for the creation of complex polygons by specifying a sequence of points.
   * @param points - An array of numbers, or or an array of PointData objects eg [{x,y}, {x,y}, {x,y}]
   * representing the x and y coordinates of the polygon's vertices, in sequence.
   * @param close - A boolean indicating whether to close the polygon path. True by default.
   * @param transform - An optional `Matrix` object to apply a transformation to the polygon.
   * @returns The instance of the current object for chaining.
   */
  poly(points, close, transform) {
    const polygon = new Polygon(points);
    polygon.closePath = close;
    this.drawShape(polygon, transform);
    return this;
  }
  /**
   * Draws a regular polygon with a specified number of sides. All sides and angles are equal.
   * @param x - The x-coordinate of the center of the polygon.
   * @param y - The y-coordinate of the center of the polygon.
   * @param radius - The radius of the circumscribed circle of the polygon.
   * @param sides - The number of sides of the polygon. Must be 3 or more.
   * @param rotation - The rotation angle of the polygon, in radians. Zero by default.
   * @param transform - An optional `Matrix` object to apply a transformation to the polygon.
   * @returns The instance of the current object for chaining.
   */
  regularPoly(x2, y2, radius, sides, rotation = 0, transform) {
    sides = Math.max(sides | 0, 3);
    const startAngle = -1 * Math.PI / 2 + rotation;
    const delta = Math.PI * 2 / sides;
    const polygon = [];
    for (let i2 = 0; i2 < sides; i2++) {
      const angle = startAngle - i2 * delta;
      polygon.push(
        x2 + radius * Math.cos(angle),
        y2 + radius * Math.sin(angle)
      );
    }
    this.poly(polygon, true, transform);
    return this;
  }
  /**
   * Draws a polygon with rounded corners.
   * Similar to `regularPoly` but with the ability to round the corners of the polygon.
   * @param x - The x-coordinate of the center of the polygon.
   * @param y - The y-coordinate of the center of the polygon.
   * @param radius - The radius of the circumscribed circle of the polygon.
   * @param sides - The number of sides of the polygon. Must be 3 or more.
   * @param corner - The radius of the rounding of the corners.
   * @param rotation - The rotation angle of the polygon, in radians. Zero by default.
   * @param smoothness - Optional parameter to adjust the smoothness of the rounding.
   * @returns The instance of the current object for chaining.
   */
  roundPoly(x2, y2, radius, sides, corner, rotation = 0, smoothness) {
    sides = Math.max(sides | 0, 3);
    if (corner <= 0) {
      return this.regularPoly(x2, y2, radius, sides, rotation);
    }
    const sideLength = radius * Math.sin(Math.PI / sides) - 1e-3;
    corner = Math.min(corner, sideLength);
    const startAngle = -1 * Math.PI / 2 + rotation;
    const delta = Math.PI * 2 / sides;
    const internalAngle = (sides - 2) * Math.PI / sides / 2;
    for (let i2 = 0; i2 < sides; i2++) {
      const angle = i2 * delta + startAngle;
      const x0 = x2 + radius * Math.cos(angle);
      const y0 = y2 + radius * Math.sin(angle);
      const a1 = angle + Math.PI + internalAngle;
      const a2 = angle - Math.PI - internalAngle;
      const x1 = x0 + corner * Math.cos(a1);
      const y1 = y0 + corner * Math.sin(a1);
      const x3 = x0 + corner * Math.cos(a2);
      const y3 = y0 + corner * Math.sin(a2);
      if (i2 === 0) {
        this.moveTo(x1, y1);
      } else {
        this.lineTo(x1, y1);
      }
      this.quadraticCurveTo(x0, y0, x3, y3, smoothness);
    }
    return this.closePath();
  }
  /**
   * Draws a shape with rounded corners. This function supports custom radius for each corner of the shape.
   * Optionally, corners can be rounded using a quadratic curve instead of an arc, providing a different aesthetic.
   * @param points - An array of `RoundedPoint` representing the corners of the shape to draw.
   * A minimum of 3 points is required.
   * @param radius - The default radius for the corners.
   * This radius is applied to all corners unless overridden in `points`.
   * @param useQuadratic - If set to true, rounded corners are drawn using a quadraticCurve
   *  method instead of an arc method. Defaults to false.
   * @param smoothness - Specifies the smoothness of the curve when `useQuadratic` is true.
   * Higher values make the curve smoother.
   * @returns The instance of the current object for chaining.
   */
  roundShape(points, radius, useQuadratic = false, smoothness) {
    if (points.length < 3) {
      return this;
    }
    if (useQuadratic) {
      roundedShapeQuadraticCurve(this, points, radius, smoothness);
    } else {
      roundedShapeArc(this, points, radius);
    }
    return this.closePath();
  }
  /**
   * Draw Rectangle with fillet corners. This is much like rounded rectangle
   * however it support negative numbers as well for the corner radius.
   * @param x - Upper left corner of rect
   * @param y - Upper right corner of rect
   * @param width - Width of rect
   * @param height - Height of rect
   * @param fillet - accept negative or positive values
   */
  filletRect(x2, y2, width, height, fillet) {
    if (fillet === 0) {
      return this.rect(x2, y2, width, height);
    }
    const maxFillet = Math.min(width, height) / 2;
    const inset = Math.min(maxFillet, Math.max(-maxFillet, fillet));
    const right = x2 + width;
    const bottom = y2 + height;
    const dir = inset < 0 ? -inset : 0;
    const size = Math.abs(inset);
    return this.moveTo(x2, y2 + size).arcTo(x2 + dir, y2 + dir, x2 + size, y2, size).lineTo(right - size, y2).arcTo(right - dir, y2 + dir, right, y2 + size, size).lineTo(right, bottom - size).arcTo(right - dir, bottom - dir, x2 + width - size, bottom, size).lineTo(x2 + size, bottom).arcTo(x2 + dir, bottom - dir, x2, bottom - size, size).closePath();
  }
  /**
   * Draw Rectangle with chamfer corners. These are angled corners.
   * @param x - Upper left corner of rect
   * @param y - Upper right corner of rect
   * @param width - Width of rect
   * @param height - Height of rect
   * @param chamfer - non-zero real number, size of corner cutout
   * @param transform
   */
  chamferRect(x2, y2, width, height, chamfer, transform) {
    if (chamfer <= 0) {
      return this.rect(x2, y2, width, height);
    }
    const inset = Math.min(chamfer, Math.min(width, height) / 2);
    const right = x2 + width;
    const bottom = y2 + height;
    const points = [
      x2 + inset,
      y2,
      right - inset,
      y2,
      right,
      y2 + inset,
      right,
      bottom - inset,
      right - inset,
      bottom,
      x2 + inset,
      bottom,
      x2,
      bottom - inset,
      x2,
      y2 + inset
    ];
    for (let i2 = points.length - 1; i2 >= 2; i2 -= 2) {
      if (points[i2] === points[i2 - 2] && points[i2 - 1] === points[i2 - 3]) {
        points.splice(i2 - 1, 2);
      }
    }
    return this.poly(points, true, transform);
  }
  /**
   * Draws an ellipse at the specified location and with the given x and y radii.
   * An optional transformation can be applied, allowing for rotation, scaling, and translation.
   * @param x - The x-coordinate of the center of the ellipse.
   * @param y - The y-coordinate of the center of the ellipse.
   * @param radiusX - The horizontal radius of the ellipse.
   * @param radiusY - The vertical radius of the ellipse.
   * @param transform - An optional `Matrix` object to apply a transformation to the ellipse. This can include rotations.
   * @returns The instance of the current object for chaining.
   */
  ellipse(x2, y2, radiusX, radiusY, transform) {
    this.drawShape(new Ellipse(x2, y2, radiusX, radiusY), transform);
    return this;
  }
  /**
   * Draws a rectangle with rounded corners.
   * The corner radius can be specified to determine how rounded the corners should be.
   * An optional transformation can be applied, which allows for rotation, scaling, and translation of the rectangle.
   * @param x - The x-coordinate of the top-left corner of the rectangle.
   * @param y - The y-coordinate of the top-left corner of the rectangle.
   * @param w - The width of the rectangle.
   * @param h - The height of the rectangle.
   * @param radius - The radius of the rectangle's corners. If not specified, corners will be sharp.
   * @param transform - An optional `Matrix` object to apply a transformation to the rectangle.
   * @returns The instance of the current object for chaining.
   */
  roundRect(x2, y2, w2, h2, radius, transform) {
    this.drawShape(new RoundedRectangle(x2, y2, w2, h2, radius), transform);
    return this;
  }
  /**
   * Draws a given shape on the canvas.
   * This is a generic method that can draw any type of shape specified by the `ShapePrimitive` parameter.
   * An optional transformation matrix can be applied to the shape, allowing for complex transformations.
   * @param shape - The shape to draw, defined as a `ShapePrimitive` object.
   * @param matrix - An optional `Matrix` for transforming the shape. This can include rotations,
   * scaling, and translations.
   * @returns The instance of the current object for chaining.
   */
  drawShape(shape, matrix) {
    this.endPoly();
    this.shapePrimitives.push({ shape, transform: matrix });
    return this;
  }
  /**
   * Starts a new polygon path from the specified starting point.
   * This method initializes a new polygon or ends the current one if it exists.
   * @param x - The x-coordinate of the starting point of the new polygon.
   * @param y - The y-coordinate of the starting point of the new polygon.
   * @returns The instance of the current object for chaining.
   */
  startPoly(x2, y2) {
    let currentPoly = this._currentPoly;
    if (currentPoly) {
      this.endPoly();
    }
    currentPoly = new Polygon();
    currentPoly.points.push(x2, y2);
    this._currentPoly = currentPoly;
    return this;
  }
  /**
   * Ends the current polygon path. If `closePath` is set to true,
   * the path is closed by connecting the last point to the first one.
   * This method finalizes the current polygon and prepares it for drawing or adding to the shape primitives.
   * @param closePath - A boolean indicating whether to close the polygon by connecting the last point
   *  back to the starting point. False by default.
   * @returns The instance of the current object for chaining.
   */
  endPoly(closePath = false) {
    const shape = this._currentPoly;
    if (shape && shape.points.length > 2) {
      shape.closePath = closePath;
      this.shapePrimitives.push({ shape });
    }
    this._currentPoly = null;
    return this;
  }
  _ensurePoly(start = true) {
    if (this._currentPoly)
      return;
    this._currentPoly = new Polygon();
    if (start) {
      const lastShape = this.shapePrimitives[this.shapePrimitives.length - 1];
      if (lastShape) {
        let lx = lastShape.shape.x;
        let ly = lastShape.shape.y;
        if (lastShape.transform && !lastShape.transform.isIdentity()) {
          const t2 = lastShape.transform;
          const tempX = lx;
          lx = t2.a * lx + t2.c * ly + t2.tx;
          ly = t2.b * tempX + t2.d * ly + t2.ty;
        }
        this._currentPoly.points.push(lx, ly);
      } else {
        this._currentPoly.points.push(0, 0);
      }
    }
  }
  /** Builds the path. */
  buildPath() {
    const path2 = this._graphicsPath2D;
    this.shapePrimitives.length = 0;
    this._currentPoly = null;
    for (let i2 = 0; i2 < path2.instructions.length; i2++) {
      const instruction = path2.instructions[i2];
      this[instruction.action](...instruction.data);
    }
    this.finish();
  }
  /** Gets the bounds of the path. */
  get bounds() {
    const bounds = this._bounds;
    bounds.clear();
    const shapePrimitives = this.shapePrimitives;
    for (let i2 = 0; i2 < shapePrimitives.length; i2++) {
      const shapePrimitive = shapePrimitives[i2];
      const boundsRect = shapePrimitive.shape.getBounds(tempRectangle);
      if (shapePrimitive.transform) {
        bounds.addRect(boundsRect, shapePrimitive.transform);
      } else {
        bounds.addRect(boundsRect);
      }
    }
    return bounds;
  }
}
class GraphicsPath {
  /**
   * Creates a `GraphicsPath` instance optionally from an SVG path string or an array of `PathInstruction`.
   * @param instructions - An SVG path string or an array of `PathInstruction` objects.
   * @param signed
   */
  constructor(instructions, signed = false) {
    this.instructions = [];
    this.uid = uid("graphicsPath");
    this._dirty = true;
    this.checkForHoles = signed;
    if (typeof instructions === "string") {
      parseSVGPath(instructions, this);
    } else {
      this.instructions = instructions?.slice() ?? [];
    }
  }
  /**
   * Provides access to the internal shape path, ensuring it is up-to-date with the current instructions.
   * @returns The `ShapePath` instance associated with this `GraphicsPath`.
   */
  get shapePath() {
    if (!this._shapePath) {
      this._shapePath = new ShapePath(this);
    }
    if (this._dirty) {
      this._dirty = false;
      this._shapePath.buildPath();
    }
    return this._shapePath;
  }
  /**
   * Adds another `GraphicsPath` to this path, optionally applying a transformation.
   * @param path - The `GraphicsPath` to add.
   * @param transform - An optional transformation to apply to the added path.
   * @returns The instance of the current object for chaining.
   */
  addPath(path2, transform) {
    path2 = path2.clone();
    this.instructions.push({ action: "addPath", data: [path2, transform] });
    this._dirty = true;
    return this;
  }
  arc(...args) {
    this.instructions.push({ action: "arc", data: args });
    this._dirty = true;
    return this;
  }
  arcTo(...args) {
    this.instructions.push({ action: "arcTo", data: args });
    this._dirty = true;
    return this;
  }
  arcToSvg(...args) {
    this.instructions.push({ action: "arcToSvg", data: args });
    this._dirty = true;
    return this;
  }
  bezierCurveTo(...args) {
    this.instructions.push({ action: "bezierCurveTo", data: args });
    this._dirty = true;
    return this;
  }
  /**
   * Adds a cubic Bezier curve to the path.
   * It requires two points: the second control point and the end point. The first control point is assumed to be
   * The starting point is the last point in the current path.
   * @param cp2x - The x-coordinate of the second control point.
   * @param cp2y - The y-coordinate of the second control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothness - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  bezierCurveToShort(cp2x, cp2y, x2, y2, smoothness) {
    const last = this.instructions[this.instructions.length - 1];
    const lastPoint = this.getLastPoint(Point.shared);
    let cp1x = 0;
    let cp1y = 0;
    if (!last || last.action !== "bezierCurveTo") {
      cp1x = lastPoint.x;
      cp1y = lastPoint.y;
    } else {
      cp1x = last.data[2];
      cp1y = last.data[3];
      const currentX = lastPoint.x;
      const currentY = lastPoint.y;
      cp1x = currentX + (currentX - cp1x);
      cp1y = currentY + (currentY - cp1y);
    }
    this.instructions.push({ action: "bezierCurveTo", data: [cp1x, cp1y, cp2x, cp2y, x2, y2, smoothness] });
    this._dirty = true;
    return this;
  }
  /**
   * Closes the current path by drawing a straight line back to the start.
   * If the shape is already closed or there are no points in the path, this method does nothing.
   * @returns The instance of the current object for chaining.
   */
  closePath() {
    this.instructions.push({ action: "closePath", data: [] });
    this._dirty = true;
    return this;
  }
  ellipse(...args) {
    this.instructions.push({ action: "ellipse", data: args });
    this._dirty = true;
    return this;
  }
  lineTo(...args) {
    this.instructions.push({ action: "lineTo", data: args });
    this._dirty = true;
    return this;
  }
  moveTo(...args) {
    this.instructions.push({ action: "moveTo", data: args });
    return this;
  }
  quadraticCurveTo(...args) {
    this.instructions.push({ action: "quadraticCurveTo", data: args });
    this._dirty = true;
    return this;
  }
  /**
   * Adds a quadratic curve to the path. It uses the previous point as the control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothness - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  quadraticCurveToShort(x2, y2, smoothness) {
    const last = this.instructions[this.instructions.length - 1];
    const lastPoint = this.getLastPoint(Point.shared);
    let cpx1 = 0;
    let cpy1 = 0;
    if (!last || last.action !== "quadraticCurveTo") {
      cpx1 = lastPoint.x;
      cpy1 = lastPoint.y;
    } else {
      cpx1 = last.data[0];
      cpy1 = last.data[1];
      const currentX = lastPoint.x;
      const currentY = lastPoint.y;
      cpx1 = currentX + (currentX - cpx1);
      cpy1 = currentY + (currentY - cpy1);
    }
    this.instructions.push({ action: "quadraticCurveTo", data: [cpx1, cpy1, x2, y2, smoothness] });
    this._dirty = true;
    return this;
  }
  /**
   * Draws a rectangle shape. This method adds a new rectangle path to the current drawing.
   * @param x - The x-coordinate of the top-left corner of the rectangle.
   * @param y - The y-coordinate of the top-left corner of the rectangle.
   * @param w - The width of the rectangle.
   * @param h - The height of the rectangle.
   * @param transform - An optional `Matrix` object to apply a transformation to the rectangle.
   * @returns The instance of the current object for chaining.
   */
  rect(x2, y2, w2, h2, transform) {
    this.instructions.push({ action: "rect", data: [x2, y2, w2, h2, transform] });
    this._dirty = true;
    return this;
  }
  /**
   * Draws a circle shape. This method adds a new circle path to the current drawing.
   * @param x - The x-coordinate of the center of the circle.
   * @param y - The y-coordinate of the center of the circle.
   * @param radius - The radius of the circle.
   * @param transform - An optional `Matrix` object to apply a transformation to the circle.
   * @returns The instance of the current object for chaining.
   */
  circle(x2, y2, radius, transform) {
    this.instructions.push({ action: "circle", data: [x2, y2, radius, transform] });
    this._dirty = true;
    return this;
  }
  roundRect(...args) {
    this.instructions.push({ action: "roundRect", data: args });
    this._dirty = true;
    return this;
  }
  poly(...args) {
    this.instructions.push({ action: "poly", data: args });
    this._dirty = true;
    return this;
  }
  regularPoly(...args) {
    this.instructions.push({ action: "regularPoly", data: args });
    this._dirty = true;
    return this;
  }
  roundPoly(...args) {
    this.instructions.push({ action: "roundPoly", data: args });
    this._dirty = true;
    return this;
  }
  roundShape(...args) {
    this.instructions.push({ action: "roundShape", data: args });
    this._dirty = true;
    return this;
  }
  filletRect(...args) {
    this.instructions.push({ action: "filletRect", data: args });
    this._dirty = true;
    return this;
  }
  chamferRect(...args) {
    this.instructions.push({ action: "chamferRect", data: args });
    this._dirty = true;
    return this;
  }
  /**
   * Draws a star shape centered at a specified location. This method allows for the creation
   *  of stars with a variable number of points, outer radius, optional inner radius, and rotation.
   * The star is drawn as a closed polygon with alternating outer and inner vertices to create the star's points.
   * An optional transformation can be applied to scale, rotate, or translate the star as needed.
   * @param x - The x-coordinate of the center of the star.
   * @param y - The y-coordinate of the center of the star.
   * @param points - The number of points of the star.
   * @param radius - The outer radius of the star (distance from the center to the outer points).
   * @param innerRadius - Optional. The inner radius of the star
   * (distance from the center to the inner points between the outer points).
   * If not provided, defaults to half of the `radius`.
   * @param rotation - Optional. The rotation of the star in radians, where 0 is aligned with the y-axis.
   * Defaults to 0, meaning one point is directly upward.
   * @param transform - An optional `Matrix` object to apply a transformation to the star.
   * This can include rotations, scaling, and translations.
   * @returns The instance of the current object for chaining further drawing commands.
   */
  // eslint-disable-next-line max-len
  star(x2, y2, points, radius, innerRadius, rotation, transform) {
    innerRadius || (innerRadius = radius / 2);
    const startAngle = -1 * Math.PI / 2 + rotation;
    const len = points * 2;
    const delta = Math.PI * 2 / len;
    const polygon = [];
    for (let i2 = 0; i2 < len; i2++) {
      const r2 = i2 % 2 ? innerRadius : radius;
      const angle = i2 * delta + startAngle;
      polygon.push(
        x2 + r2 * Math.cos(angle),
        y2 + r2 * Math.sin(angle)
      );
    }
    this.poly(polygon, true, transform);
    return this;
  }
  /**
   * Creates a copy of the current `GraphicsPath` instance. This method supports both shallow and deep cloning.
   * A shallow clone copies the reference of the instructions array, while a deep clone creates a new array and
   * copies each instruction individually, ensuring that modifications to the instructions of the cloned `GraphicsPath`
   * do not affect the original `GraphicsPath` and vice versa.
   * @param deep - A boolean flag indicating whether the clone should be deep.
   * @returns A new `GraphicsPath` instance that is a clone of the current instance.
   */
  clone(deep = false) {
    const newGraphicsPath2D = new GraphicsPath();
    newGraphicsPath2D.checkForHoles = this.checkForHoles;
    if (!deep) {
      newGraphicsPath2D.instructions = this.instructions.slice();
    } else {
      for (let i2 = 0; i2 < this.instructions.length; i2++) {
        const instruction = this.instructions[i2];
        newGraphicsPath2D.instructions.push({ action: instruction.action, data: instruction.data.slice() });
      }
    }
    return newGraphicsPath2D;
  }
  clear() {
    this.instructions.length = 0;
    this._dirty = true;
    return this;
  }
  /**
   * Applies a transformation matrix to all drawing instructions within the `GraphicsPath`.
   * This method enables the modification of the path's geometry according to the provided
   * transformation matrix, which can include translations, rotations, scaling, and skewing.
   *
   * Each drawing instruction in the path is updated to reflect the transformation,
   * ensuring the visual representation of the path is consistent with the applied matrix.
   *
   * Note: The transformation is applied directly to the coordinates and control points of the drawing instructions,
   * not to the path as a whole. This means the transformation's effects are baked into the individual instructions,
   * allowing for fine-grained control over the path's appearance.
   * @param matrix - A `Matrix` object representing the transformation to apply.
   * @returns The instance of the current object for chaining further operations.
   */
  transform(matrix) {
    if (matrix.isIdentity())
      return this;
    const a2 = matrix.a;
    const b2 = matrix.b;
    const c2 = matrix.c;
    const d2 = matrix.d;
    const tx = matrix.tx;
    const ty = matrix.ty;
    let x2 = 0;
    let y2 = 0;
    let cpx1 = 0;
    let cpy1 = 0;
    let cpx2 = 0;
    let cpy2 = 0;
    let rx = 0;
    let ry = 0;
    for (let i2 = 0; i2 < this.instructions.length; i2++) {
      const instruction = this.instructions[i2];
      const data = instruction.data;
      switch (instruction.action) {
        case "moveTo":
        case "lineTo":
          x2 = data[0];
          y2 = data[1];
          data[0] = a2 * x2 + c2 * y2 + tx;
          data[1] = b2 * x2 + d2 * y2 + ty;
          break;
        case "bezierCurveTo":
          cpx1 = data[0];
          cpy1 = data[1];
          cpx2 = data[2];
          cpy2 = data[3];
          x2 = data[4];
          y2 = data[5];
          data[0] = a2 * cpx1 + c2 * cpy1 + tx;
          data[1] = b2 * cpx1 + d2 * cpy1 + ty;
          data[2] = a2 * cpx2 + c2 * cpy2 + tx;
          data[3] = b2 * cpx2 + d2 * cpy2 + ty;
          data[4] = a2 * x2 + c2 * y2 + tx;
          data[5] = b2 * x2 + d2 * y2 + ty;
          break;
        case "quadraticCurveTo":
          cpx1 = data[0];
          cpy1 = data[1];
          x2 = data[2];
          y2 = data[3];
          data[0] = a2 * cpx1 + c2 * cpy1 + tx;
          data[1] = b2 * cpx1 + d2 * cpy1 + ty;
          data[2] = a2 * x2 + c2 * y2 + tx;
          data[3] = b2 * x2 + d2 * y2 + ty;
          break;
        case "arcToSvg":
          x2 = data[5];
          y2 = data[6];
          rx = data[0];
          ry = data[1];
          data[0] = a2 * rx + c2 * ry;
          data[1] = b2 * rx + d2 * ry;
          data[5] = a2 * x2 + c2 * y2 + tx;
          data[6] = b2 * x2 + d2 * y2 + ty;
          break;
        case "circle":
          data[4] = adjustTransform(data[3], matrix);
          break;
        case "rect":
          data[4] = adjustTransform(data[4], matrix);
          break;
        case "ellipse":
          data[8] = adjustTransform(data[8], matrix);
          break;
        case "roundRect":
          data[5] = adjustTransform(data[5], matrix);
          break;
        case "addPath":
          data[0].transform(matrix);
          break;
        case "poly":
          data[2] = adjustTransform(data[2], matrix);
          break;
        default:
          warn("unknown transform action", instruction.action);
          break;
      }
    }
    this._dirty = true;
    return this;
  }
  get bounds() {
    return this.shapePath.bounds;
  }
  /**
   * Retrieves the last point from the current drawing instructions in the `GraphicsPath`.
   * This method is useful for operations that depend on the path's current endpoint,
   * such as connecting subsequent shapes or paths. It supports various drawing instructions,
   * ensuring the last point's position is accurately determined regardless of the path's complexity.
   *
   * If the last instruction is a `closePath`, the method iterates backward through the instructions
   *  until it finds an actionable instruction that defines a point (e.g., `moveTo`, `lineTo`,
   * `quadraticCurveTo`, etc.). For compound paths added via `addPath`, it recursively retrieves
   * the last point from the nested path.
   * @param out - A `Point` object where the last point's coordinates will be stored.
   * This object is modified directly to contain the result.
   * @returns The `Point` object containing the last point's coordinates.
   */
  getLastPoint(out2) {
    let index = this.instructions.length - 1;
    let lastInstruction = this.instructions[index];
    if (!lastInstruction) {
      out2.x = 0;
      out2.y = 0;
      return out2;
    }
    while (lastInstruction.action === "closePath") {
      index--;
      if (index < 0) {
        out2.x = 0;
        out2.y = 0;
        return out2;
      }
      lastInstruction = this.instructions[index];
    }
    switch (lastInstruction.action) {
      case "moveTo":
      case "lineTo":
        out2.x = lastInstruction.data[0];
        out2.y = lastInstruction.data[1];
        break;
      case "quadraticCurveTo":
        out2.x = lastInstruction.data[2];
        out2.y = lastInstruction.data[3];
        break;
      case "bezierCurveTo":
        out2.x = lastInstruction.data[4];
        out2.y = lastInstruction.data[5];
        break;
      case "arc":
      case "arcToSvg":
        out2.x = lastInstruction.data[5];
        out2.y = lastInstruction.data[6];
        break;
      case "addPath":
        lastInstruction.data[0].getLastPoint(out2);
        break;
    }
    return out2;
  }
}
function adjustTransform(currentMatrix, transform) {
  if (currentMatrix) {
    return currentMatrix.prepend(transform);
  }
  return transform.clone();
}
function parseSVGFloatAttribute(svg, id, defaultValue) {
  const value = svg.getAttribute(id);
  return value ? Number(value) : defaultValue;
}
function parseSVGDefinitions(svg, session) {
  const definitions = svg.querySelectorAll("defs");
  for (let i2 = 0; i2 < definitions.length; i2++) {
    const definition = definitions[i2];
    for (let j2 = 0; j2 < definition.children.length; j2++) {
      const child = definition.children[j2];
      switch (child.nodeName.toLowerCase()) {
        case "lineargradient":
          session.defs[child.id] = parseLinearGradient(child);
          break;
        case "radialgradient":
          session.defs[child.id] = parseRadialGradient();
          break;
      }
    }
  }
}
function parseLinearGradient(child) {
  const x0 = parseSVGFloatAttribute(child, "x1", 0);
  const y0 = parseSVGFloatAttribute(child, "y1", 0);
  const x1 = parseSVGFloatAttribute(child, "x2", 1);
  const y1 = parseSVGFloatAttribute(child, "y2", 0);
  const gradientUnit = child.getAttribute("gradientUnits") || "objectBoundingBox";
  const gradient = new FillGradient(
    x0,
    y0,
    x1,
    y1,
    gradientUnit === "objectBoundingBox" ? "local" : "global"
  );
  for (let k2 = 0; k2 < child.children.length; k2++) {
    const stop = child.children[k2];
    const offset = parseSVGFloatAttribute(stop, "offset", 0);
    const color = Color.shared.setValue(stop.getAttribute("stop-color")).toNumber();
    gradient.addColorStop(offset, color);
  }
  return gradient;
}
function parseRadialGradient(_child) {
  warn("[SVG Parser] Radial gradients are not yet supported");
  return new FillGradient(0, 0, 1, 0);
}
function extractSvgUrlId(url) {
  const match = url.match(/url\s*\(\s*['"]?\s*#([^'"\s)]+)\s*['"]?\s*\)/i);
  return match ? match[1] : "";
}
const styleAttributes = {
  // Fill properties
  fill: { type: "paint", default: 0 },
  // Fill color/gradient
  "fill-opacity": { type: "number", default: 1 },
  // Fill transparency
  // Stroke properties
  stroke: { type: "paint", default: 0 },
  // Stroke color/gradient
  "stroke-width": { type: "number", default: 1 },
  // Width of stroke
  "stroke-opacity": { type: "number", default: 1 },
  // Stroke transparency
  "stroke-linecap": { type: "string", default: "butt" },
  // End cap style: butt, round, square
  "stroke-linejoin": { type: "string", default: "miter" },
  // Join style: miter, round, bevel
  "stroke-miterlimit": { type: "number", default: 10 },
  // Limit on miter join sharpness
  "stroke-dasharray": { type: "string", default: "none" },
  // Dash pattern
  "stroke-dashoffset": { type: "number", default: 0 },
  // Offset for dash pattern
  // Global properties
  opacity: { type: "number", default: 1 }
  // Overall opacity
};
function parseSVGStyle(svg, session) {
  const style = svg.getAttribute("style");
  const strokeStyle = {};
  const fillStyle = {};
  const result = {
    strokeStyle,
    fillStyle,
    useFill: false,
    useStroke: false
  };
  for (const key in styleAttributes) {
    const attribute = svg.getAttribute(key);
    if (attribute) {
      parseAttribute(session, result, key, attribute.trim());
    }
  }
  if (style) {
    const styleParts = style.split(";");
    for (let i2 = 0; i2 < styleParts.length; i2++) {
      const stylePart = styleParts[i2].trim();
      const [key, value] = stylePart.split(":");
      if (styleAttributes[key]) {
        parseAttribute(session, result, key, value.trim());
      }
    }
  }
  return {
    strokeStyle: result.useStroke ? strokeStyle : null,
    fillStyle: result.useFill ? fillStyle : null,
    useFill: result.useFill,
    useStroke: result.useStroke
  };
}
function parseAttribute(session, result, id, value) {
  switch (id) {
    case "stroke":
      if (value !== "none") {
        if (value.startsWith("url(")) {
          const id2 = extractSvgUrlId(value);
          result.strokeStyle.fill = session.defs[id2];
        } else {
          result.strokeStyle.color = Color.shared.setValue(value).toNumber();
        }
        result.useStroke = true;
      }
      break;
    case "stroke-width":
      result.strokeStyle.width = Number(value);
      break;
    case "fill":
      if (value !== "none") {
        if (value.startsWith("url(")) {
          const id2 = extractSvgUrlId(value);
          result.fillStyle.fill = session.defs[id2];
        } else {
          result.fillStyle.color = Color.shared.setValue(value).toNumber();
        }
        result.useFill = true;
      }
      break;
    case "fill-opacity":
      result.fillStyle.alpha = Number(value);
      break;
    case "stroke-opacity":
      result.strokeStyle.alpha = Number(value);
      break;
    case "opacity":
      result.fillStyle.alpha = Number(value);
      result.strokeStyle.alpha = Number(value);
      break;
  }
}
function checkForNestedPattern(subpathsWithArea) {
  if (subpathsWithArea.length <= 2) {
    return true;
  }
  const areas = subpathsWithArea.map((s2) => s2.area).sort((a2, b2) => b2 - a2);
  const [largestArea, secondArea] = areas;
  const smallestArea = areas[areas.length - 1];
  const largestToSecondRatio = largestArea / secondArea;
  const secondToSmallestRatio = secondArea / smallestArea;
  if (largestToSecondRatio > 3 && secondToSmallestRatio < 2) {
    return false;
  }
  return true;
}
function extractSubpaths(pathData) {
  const parts = pathData.split(/(?=[Mm])/);
  const subpaths = parts.filter((part) => part.trim().length > 0);
  return subpaths;
}
function calculatePathArea(pathData) {
  const coords = pathData.match(/[-+]?[0-9]*\.?[0-9]+/g);
  if (!coords || coords.length < 4)
    return 0;
  const numbers = coords.map(Number);
  const xs = [];
  const ys = [];
  for (let i2 = 0; i2 < numbers.length; i2 += 2) {
    if (i2 + 1 < numbers.length) {
      xs.push(numbers[i2]);
      ys.push(numbers[i2 + 1]);
    }
  }
  if (xs.length === 0 || ys.length === 0)
    return 0;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const area2 = (maxX - minX) * (maxY - minY);
  return area2;
}
function appendSVGPath(pathData, graphicsPath) {
  const tempPath = new GraphicsPath(pathData, false);
  for (const instruction of tempPath.instructions) {
    graphicsPath.instructions.push(instruction);
  }
}
function SVGParser(svg, graphicsContext) {
  if (typeof svg === "string") {
    const div = document.createElement("div");
    div.innerHTML = svg.trim();
    svg = div.querySelector("svg");
  }
  const session = {
    context: graphicsContext,
    defs: {},
    path: new GraphicsPath()
  };
  parseSVGDefinitions(svg, session);
  const children = svg.children;
  const { fillStyle, strokeStyle } = parseSVGStyle(svg, session);
  for (let i2 = 0; i2 < children.length; i2++) {
    const child = children[i2];
    if (child.nodeName.toLowerCase() === "defs")
      continue;
    renderChildren(child, session, fillStyle, strokeStyle);
  }
  return graphicsContext;
}
function renderChildren(svg, session, fillStyle, strokeStyle) {
  const children = svg.children;
  const { fillStyle: f1, strokeStyle: s1 } = parseSVGStyle(svg, session);
  if (f1 && fillStyle) {
    fillStyle = { ...fillStyle, ...f1 };
  } else if (f1) {
    fillStyle = f1;
  }
  if (s1 && strokeStyle) {
    strokeStyle = { ...strokeStyle, ...s1 };
  } else if (s1) {
    strokeStyle = s1;
  }
  const noStyle = !fillStyle && !strokeStyle;
  if (noStyle) {
    fillStyle = { color: 0 };
  }
  let x2;
  let y2;
  let x1;
  let y1;
  let x22;
  let y22;
  let cx;
  let cy;
  let r2;
  let rx;
  let ry;
  let points;
  let pointsString;
  let d2;
  let graphicsPath;
  let width;
  let height;
  switch (svg.nodeName.toLowerCase()) {
    case "path": {
      d2 = svg.getAttribute("d");
      const fillRule = svg.getAttribute("fill-rule");
      const subpaths = extractSubpaths(d2);
      const hasExplicitEvenodd = fillRule === "evenodd";
      const hasMultipleSubpaths = subpaths.length > 1;
      const shouldProcessHoles = hasExplicitEvenodd && hasMultipleSubpaths;
      if (shouldProcessHoles) {
        const subpathsWithArea = subpaths.map((subpath) => ({
          path: subpath,
          area: calculatePathArea(subpath)
        }));
        subpathsWithArea.sort((a2, b2) => b2.area - a2.area);
        const useMultipleHolesApproach = subpaths.length > 3 || !checkForNestedPattern(subpathsWithArea);
        if (useMultipleHolesApproach) {
          for (let i2 = 0; i2 < subpathsWithArea.length; i2++) {
            const subpath = subpathsWithArea[i2];
            const isMainShape = i2 === 0;
            session.context.beginPath();
            const newPath = new GraphicsPath(void 0, true);
            appendSVGPath(subpath.path, newPath);
            session.context.path(newPath);
            if (isMainShape) {
              if (fillStyle)
                session.context.fill(fillStyle);
              if (strokeStyle)
                session.context.stroke(strokeStyle);
            } else {
              session.context.cut();
            }
          }
        } else {
          for (let i2 = 0; i2 < subpathsWithArea.length; i2++) {
            const subpath = subpathsWithArea[i2];
            const isHole = i2 % 2 === 1;
            session.context.beginPath();
            const newPath = new GraphicsPath(void 0, true);
            appendSVGPath(subpath.path, newPath);
            session.context.path(newPath);
            if (isHole) {
              session.context.cut();
            } else {
              if (fillStyle)
                session.context.fill(fillStyle);
              if (strokeStyle)
                session.context.stroke(strokeStyle);
            }
          }
        }
      } else {
        const useEvenoddForGraphicsPath = fillRule ? fillRule === "evenodd" : true;
        graphicsPath = new GraphicsPath(d2, useEvenoddForGraphicsPath);
        session.context.path(graphicsPath);
        if (fillStyle)
          session.context.fill(fillStyle);
        if (strokeStyle)
          session.context.stroke(strokeStyle);
      }
      break;
    }
    case "circle":
      cx = parseSVGFloatAttribute(svg, "cx", 0);
      cy = parseSVGFloatAttribute(svg, "cy", 0);
      r2 = parseSVGFloatAttribute(svg, "r", 0);
      session.context.ellipse(cx, cy, r2, r2);
      if (fillStyle)
        session.context.fill(fillStyle);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "rect":
      x2 = parseSVGFloatAttribute(svg, "x", 0);
      y2 = parseSVGFloatAttribute(svg, "y", 0);
      width = parseSVGFloatAttribute(svg, "width", 0);
      height = parseSVGFloatAttribute(svg, "height", 0);
      rx = parseSVGFloatAttribute(svg, "rx", 0);
      ry = parseSVGFloatAttribute(svg, "ry", 0);
      if (rx || ry) {
        session.context.roundRect(x2, y2, width, height, rx || ry);
      } else {
        session.context.rect(x2, y2, width, height);
      }
      if (fillStyle)
        session.context.fill(fillStyle);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "ellipse":
      cx = parseSVGFloatAttribute(svg, "cx", 0);
      cy = parseSVGFloatAttribute(svg, "cy", 0);
      rx = parseSVGFloatAttribute(svg, "rx", 0);
      ry = parseSVGFloatAttribute(svg, "ry", 0);
      session.context.beginPath();
      session.context.ellipse(cx, cy, rx, ry);
      if (fillStyle)
        session.context.fill(fillStyle);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "line":
      x1 = parseSVGFloatAttribute(svg, "x1", 0);
      y1 = parseSVGFloatAttribute(svg, "y1", 0);
      x22 = parseSVGFloatAttribute(svg, "x2", 0);
      y22 = parseSVGFloatAttribute(svg, "y2", 0);
      session.context.beginPath();
      session.context.moveTo(x1, y1);
      session.context.lineTo(x22, y22);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "polygon":
      pointsString = svg.getAttribute("points");
      points = pointsString.match(/-?\d+/g).map((n2) => parseInt(n2, 10));
      session.context.poly(points, true);
      if (fillStyle)
        session.context.fill(fillStyle);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "polyline":
      pointsString = svg.getAttribute("points");
      points = pointsString.match(/-?\d+/g).map((n2) => parseInt(n2, 10));
      session.context.poly(points, false);
      if (strokeStyle)
        session.context.stroke(strokeStyle);
      break;
    case "g":
    case "svg":
      break;
    default: {
      warn(`[SVG parser] <${svg.nodeName}> elements unsupported`);
      break;
    }
  }
  if (noStyle) {
    fillStyle = null;
  }
  for (let i2 = 0; i2 < children.length; i2++) {
    renderChildren(children[i2], session, fillStyle, strokeStyle);
  }
}
function isColorLike(value) {
  return Color.isColorLike(value);
}
function isFillPattern(value) {
  return value instanceof FillPattern;
}
function isFillGradient(value) {
  return value instanceof FillGradient;
}
function isTexture(value) {
  return value instanceof Texture;
}
function handleColorLike(fill, value, defaultStyle) {
  const temp = Color.shared.setValue(value ?? 0);
  fill.color = temp.toNumber();
  fill.alpha = temp.alpha === 1 ? defaultStyle.alpha : temp.alpha;
  fill.texture = Texture.WHITE;
  return { ...defaultStyle, ...fill };
}
function handleTexture(fill, value, defaultStyle) {
  fill.texture = value;
  return { ...defaultStyle, ...fill };
}
function handleFillPattern(fill, value, defaultStyle) {
  fill.fill = value;
  fill.color = 16777215;
  fill.texture = value.texture;
  fill.matrix = value.transform;
  return { ...defaultStyle, ...fill };
}
function handleFillGradient(fill, value, defaultStyle) {
  value.buildGradient();
  fill.fill = value;
  fill.color = 16777215;
  fill.texture = value.texture;
  fill.matrix = value.transform;
  fill.textureSpace = value.textureSpace;
  return { ...defaultStyle, ...fill };
}
function handleFillObject(value, defaultStyle) {
  const style = { ...defaultStyle, ...value };
  const color = Color.shared.setValue(style.color);
  style.alpha *= color.alpha;
  style.color = color.toNumber();
  return style;
}
function toFillStyle(value, defaultStyle) {
  if (value === void 0 || value === null) {
    return null;
  }
  const fill = {};
  const objectStyle = value;
  if (isColorLike(value)) {
    return handleColorLike(fill, value, defaultStyle);
  } else if (isTexture(value)) {
    return handleTexture(fill, value, defaultStyle);
  } else if (isFillPattern(value)) {
    return handleFillPattern(fill, value, defaultStyle);
  } else if (isFillGradient(value)) {
    return handleFillGradient(fill, value, defaultStyle);
  } else if (objectStyle.fill && isFillPattern(objectStyle.fill)) {
    return handleFillPattern(objectStyle, objectStyle.fill, defaultStyle);
  } else if (objectStyle.fill && isFillGradient(objectStyle.fill)) {
    return handleFillGradient(objectStyle, objectStyle.fill, defaultStyle);
  }
  return handleFillObject(objectStyle, defaultStyle);
}
function toStrokeStyle(value, defaultStyle) {
  const { width, alignment, miterLimit, cap, join, pixelLine, ...rest } = defaultStyle;
  const fill = toFillStyle(value, rest);
  if (!fill) {
    return null;
  }
  return {
    width,
    alignment,
    miterLimit,
    cap,
    join,
    pixelLine,
    ...fill
  };
}
const tmpPoint = new Point();
const tempMatrix = new Matrix();
const _GraphicsContext = class _GraphicsContext2 extends EventEmitter {
  constructor() {
    super(...arguments);
    this._gpuData = /* @__PURE__ */ Object.create(null);
    this.autoGarbageCollect = true;
    this._gcLastUsed = -1;
    this.uid = uid("graphicsContext");
    this.dirty = true;
    this.batchMode = "auto";
    this.instructions = [];
    this.destroyed = false;
    this._activePath = new GraphicsPath();
    this._transform = new Matrix();
    this._fillStyle = { ..._GraphicsContext2.defaultFillStyle };
    this._strokeStyle = { ..._GraphicsContext2.defaultStrokeStyle };
    this._stateStack = [];
    this._tick = 0;
    this._bounds = new Bounds();
    this._boundsDirty = true;
  }
  /**
   * Creates a new GraphicsContext object that is a clone of this instance, copying all properties,
   * including the current drawing state, transformations, styles, and instructions.
   * @returns A new GraphicsContext instance with the same properties and state as this one.
   */
  clone() {
    const clone = new _GraphicsContext2();
    clone.batchMode = this.batchMode;
    clone.instructions = this.instructions.slice();
    clone._activePath = this._activePath.clone();
    clone._transform = this._transform.clone();
    clone._fillStyle = { ...this._fillStyle };
    clone._strokeStyle = { ...this._strokeStyle };
    clone._stateStack = this._stateStack.slice();
    clone._bounds = this._bounds.clone();
    clone._boundsDirty = true;
    return clone;
  }
  /**
   * The current fill style of the graphics context. This can be a color, gradient, pattern, or a more complex style defined by a FillStyle object.
   */
  get fillStyle() {
    return this._fillStyle;
  }
  set fillStyle(value) {
    this._fillStyle = toFillStyle(value, _GraphicsContext2.defaultFillStyle);
  }
  /**
   * The current stroke style of the graphics context. Similar to fill styles, stroke styles can encompass colors, gradients, patterns, or more detailed configurations via a StrokeStyle object.
   */
  get strokeStyle() {
    return this._strokeStyle;
  }
  set strokeStyle(value) {
    this._strokeStyle = toStrokeStyle(value, _GraphicsContext2.defaultStrokeStyle);
  }
  /**
   * Sets the current fill style of the graphics context. The fill style can be a color, gradient,
   * pattern, or a more complex style defined by a FillStyle object.
   * @param style - The fill style to apply. This can be a simple color, a gradient or pattern object,
   *                or a FillStyle or ConvertedFillStyle object.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  setFillStyle(style) {
    this._fillStyle = toFillStyle(style, _GraphicsContext2.defaultFillStyle);
    return this;
  }
  /**
   * Sets the current stroke style of the graphics context. Similar to fill styles, stroke styles can
   * encompass colors, gradients, patterns, or more detailed configurations via a StrokeStyle object.
   * @param style - The stroke style to apply. Can be defined as a color, a gradient or pattern,
   *                or a StrokeStyle or ConvertedStrokeStyle object.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  setStrokeStyle(style) {
    this._strokeStyle = toFillStyle(style, _GraphicsContext2.defaultStrokeStyle);
    return this;
  }
  texture(texture, tint, dx, dy, dw, dh) {
    this.instructions.push({
      action: "texture",
      data: {
        image: texture,
        dx: dx || 0,
        dy: dy || 0,
        dw: dw || texture.frame.width,
        dh: dh || texture.frame.height,
        transform: this._transform.clone(),
        alpha: this._fillStyle.alpha,
        style: tint ? Color.shared.setValue(tint).toNumber() : 16777215
      }
    });
    this.onUpdate();
    return this;
  }
  /**
   * Resets the current path. Any previous path and its commands are discarded and a new path is
   * started. This is typically called before beginning a new shape or series of drawing commands.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  beginPath() {
    this._activePath = new GraphicsPath();
    return this;
  }
  fill(style, alpha) {
    let path2;
    const lastInstruction = this.instructions[this.instructions.length - 1];
    if (this._tick === 0 && lastInstruction?.action === "stroke") {
      path2 = lastInstruction.data.path;
    } else {
      path2 = this._activePath.clone();
    }
    if (!path2)
      return this;
    if (style != null) {
      if (alpha !== void 0 && typeof style === "number") {
        deprecation(v8_0_0, "GraphicsContext.fill(color, alpha) is deprecated, use GraphicsContext.fill({ color, alpha }) instead");
        style = { color: style, alpha };
      }
      this._fillStyle = toFillStyle(style, _GraphicsContext2.defaultFillStyle);
    }
    this.instructions.push({
      action: "fill",
      // TODO copy fill style!
      data: { style: this.fillStyle, path: path2 }
    });
    this.onUpdate();
    this._initNextPathLocation();
    this._tick = 0;
    return this;
  }
  _initNextPathLocation() {
    const { x: x2, y: y2 } = this._activePath.getLastPoint(Point.shared);
    this._activePath.clear();
    this._activePath.moveTo(x2, y2);
  }
  /**
   * Strokes the current path with the current stroke style. This method can take an optional
   * FillInput parameter to define the stroke's appearance, including its color, width, and other properties.
   * @param style - (Optional) The stroke style to apply. Can be defined as a simple color or a more complex style object. If omitted, uses the current stroke style.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  stroke(style) {
    let path2;
    const lastInstruction = this.instructions[this.instructions.length - 1];
    if (this._tick === 0 && lastInstruction?.action === "fill") {
      path2 = lastInstruction.data.path;
    } else {
      path2 = this._activePath.clone();
    }
    if (!path2)
      return this;
    if (style != null) {
      this._strokeStyle = toStrokeStyle(style, _GraphicsContext2.defaultStrokeStyle);
    }
    this.instructions.push({
      action: "stroke",
      // TODO copy fill style!
      data: { style: this.strokeStyle, path: path2 }
    });
    this.onUpdate();
    this._initNextPathLocation();
    this._tick = 0;
    return this;
  }
  /**
   * Applies a cutout to the last drawn shape. This is used to create holes or complex shapes by
   * subtracting a path from the previously drawn path. If a hole is not completely in a shape, it will
   * fail to cut correctly!
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  cut() {
    for (let i2 = 0; i2 < 2; i2++) {
      const lastInstruction = this.instructions[this.instructions.length - 1 - i2];
      const holePath = this._activePath.clone();
      if (lastInstruction) {
        if (lastInstruction.action === "stroke" || lastInstruction.action === "fill") {
          if (lastInstruction.data.hole) {
            lastInstruction.data.hole.addPath(holePath);
          } else {
            lastInstruction.data.hole = holePath;
            break;
          }
        }
      }
    }
    this._initNextPathLocation();
    return this;
  }
  /**
   * Adds an arc to the current path, which is centered at (x, y) with the specified radius,
   * starting and ending angles, and direction.
   * @param x - The x-coordinate of the arc's center.
   * @param y - The y-coordinate of the arc's center.
   * @param radius - The arc's radius.
   * @param startAngle - The starting angle, in radians.
   * @param endAngle - The ending angle, in radians.
   * @param counterclockwise - (Optional) Specifies whether the arc is drawn counterclockwise (true) or clockwise (false). Defaults to false.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  arc(x2, y2, radius, startAngle, endAngle, counterclockwise) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.arc(
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty,
      radius,
      startAngle,
      endAngle,
      counterclockwise
    );
    return this;
  }
  /**
   * Adds an arc to the current path with the given control points and radius, connected to the previous point
   * by a straight line if necessary.
   * @param x1 - The x-coordinate of the first control point.
   * @param y1 - The y-coordinate of the first control point.
   * @param x2 - The x-coordinate of the second control point.
   * @param y2 - The y-coordinate of the second control point.
   * @param radius - The arc's radius.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  arcTo(x1, y1, x2, y2, radius) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.arcTo(
      t2.a * x1 + t2.c * y1 + t2.tx,
      t2.b * x1 + t2.d * y1 + t2.ty,
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty,
      radius
    );
    return this;
  }
  /**
   * Adds an SVG-style arc to the path, allowing for elliptical arcs based on the SVG spec.
   * @param rx - The x-radius of the ellipse.
   * @param ry - The y-radius of the ellipse.
   * @param xAxisRotation - The rotation of the ellipse's x-axis relative
   * to the x-axis of the coordinate system, in degrees.
   * @param largeArcFlag - Determines if the arc should be greater than or less than 180 degrees.
   * @param sweepFlag - Determines if the arc should be swept in a positive angle direction.
   * @param x - The x-coordinate of the arc's end point.
   * @param y - The y-coordinate of the arc's end point.
   * @returns The instance of the current object for chaining.
   */
  arcToSvg(rx, ry, xAxisRotation, largeArcFlag, sweepFlag, x2, y2) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.arcToSvg(
      rx,
      ry,
      xAxisRotation,
      // should we rotate this with transform??
      largeArcFlag,
      sweepFlag,
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty
    );
    return this;
  }
  /**
   * Adds a cubic Bezier curve to the path.
   * It requires three points: the first two are control points and the third one is the end point.
   * The starting point is the last point in the current path.
   * @param cp1x - The x-coordinate of the first control point.
   * @param cp1y - The y-coordinate of the first control point.
   * @param cp2x - The x-coordinate of the second control point.
   * @param cp2y - The y-coordinate of the second control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothness - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2, smoothness) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.bezierCurveTo(
      t2.a * cp1x + t2.c * cp1y + t2.tx,
      t2.b * cp1x + t2.d * cp1y + t2.ty,
      t2.a * cp2x + t2.c * cp2y + t2.tx,
      t2.b * cp2x + t2.d * cp2y + t2.ty,
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty,
      smoothness
    );
    return this;
  }
  /**
   * Closes the current path by drawing a straight line back to the start.
   * If the shape is already closed or there are no points in the path, this method does nothing.
   * @returns The instance of the current object for chaining.
   */
  closePath() {
    this._tick++;
    this._activePath?.closePath();
    return this;
  }
  /**
   * Draws an ellipse at the specified location and with the given x and y radii.
   * An optional transformation can be applied, allowing for rotation, scaling, and translation.
   * @param x - The x-coordinate of the center of the ellipse.
   * @param y - The y-coordinate of the center of the ellipse.
   * @param radiusX - The horizontal radius of the ellipse.
   * @param radiusY - The vertical radius of the ellipse.
   * @returns The instance of the current object for chaining.
   */
  ellipse(x2, y2, radiusX, radiusY) {
    this._tick++;
    this._activePath.ellipse(x2, y2, radiusX, radiusY, this._transform.clone());
    return this;
  }
  /**
   * Draws a circle shape. This method adds a new circle path to the current drawing.
   * @param x - The x-coordinate of the center of the circle.
   * @param y - The y-coordinate of the center of the circle.
   * @param radius - The radius of the circle.
   * @returns The instance of the current object for chaining.
   */
  circle(x2, y2, radius) {
    this._tick++;
    this._activePath.circle(x2, y2, radius, this._transform.clone());
    return this;
  }
  /**
   * Adds another `GraphicsPath` to this path, optionally applying a transformation.
   * @param path - The `GraphicsPath` to add.
   * @returns The instance of the current object for chaining.
   */
  path(path2) {
    this._tick++;
    this._activePath.addPath(path2, this._transform.clone());
    return this;
  }
  /**
   * Connects the current point to a new point with a straight line. This method updates the current path.
   * @param x - The x-coordinate of the new point to connect to.
   * @param y - The y-coordinate of the new point to connect to.
   * @returns The instance of the current object for chaining.
   */
  lineTo(x2, y2) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.lineTo(
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty
    );
    return this;
  }
  /**
   * Sets the starting point for a new sub-path. Any subsequent drawing commands are considered part of this path.
   * @param x - The x-coordinate for the starting point.
   * @param y - The y-coordinate for the starting point.
   * @returns The instance of the current object for chaining.
   */
  moveTo(x2, y2) {
    this._tick++;
    const t2 = this._transform;
    const instructions = this._activePath.instructions;
    const transformedX = t2.a * x2 + t2.c * y2 + t2.tx;
    const transformedY = t2.b * x2 + t2.d * y2 + t2.ty;
    if (instructions.length === 1 && instructions[0].action === "moveTo") {
      instructions[0].data[0] = transformedX;
      instructions[0].data[1] = transformedY;
      return this;
    }
    this._activePath.moveTo(
      transformedX,
      transformedY
    );
    return this;
  }
  /**
   * Adds a quadratic curve to the path. It requires two points: the control point and the end point.
   * The starting point is the last point in the current path.
   * @param cpx - The x-coordinate of the control point.
   * @param cpy - The y-coordinate of the control point.
   * @param x - The x-coordinate of the end point.
   * @param y - The y-coordinate of the end point.
   * @param smoothness - Optional parameter to adjust the smoothness of the curve.
   * @returns The instance of the current object for chaining.
   */
  quadraticCurveTo(cpx, cpy, x2, y2, smoothness) {
    this._tick++;
    const t2 = this._transform;
    this._activePath.quadraticCurveTo(
      t2.a * cpx + t2.c * cpy + t2.tx,
      t2.b * cpx + t2.d * cpy + t2.ty,
      t2.a * x2 + t2.c * y2 + t2.tx,
      t2.b * x2 + t2.d * y2 + t2.ty,
      smoothness
    );
    return this;
  }
  /**
   * Draws a rectangle shape. This method adds a new rectangle path to the current drawing.
   * @param x - The x-coordinate of the top-left corner of the rectangle.
   * @param y - The y-coordinate of the top-left corner of the rectangle.
   * @param w - The width of the rectangle.
   * @param h - The height of the rectangle.
   * @returns The instance of the current object for chaining.
   */
  rect(x2, y2, w2, h2) {
    this._tick++;
    this._activePath.rect(x2, y2, w2, h2, this._transform.clone());
    return this;
  }
  /**
   * Draws a rectangle with rounded corners.
   * The corner radius can be specified to determine how rounded the corners should be.
   * An optional transformation can be applied, which allows for rotation, scaling, and translation of the rectangle.
   * @param x - The x-coordinate of the top-left corner of the rectangle.
   * @param y - The y-coordinate of the top-left corner of the rectangle.
   * @param w - The width of the rectangle.
   * @param h - The height of the rectangle.
   * @param radius - The radius of the rectangle's corners. If not specified, corners will be sharp.
   * @returns The instance of the current object for chaining.
   */
  roundRect(x2, y2, w2, h2, radius) {
    this._tick++;
    this._activePath.roundRect(x2, y2, w2, h2, radius, this._transform.clone());
    return this;
  }
  /**
   * Draws a polygon shape by specifying a sequence of points. This method allows for the creation of complex polygons,
   * which can be both open and closed. An optional transformation can be applied, enabling the polygon to be scaled,
   * rotated, or translated as needed.
   * @param points - An array of numbers, or an array of PointData objects eg [{x,y}, {x,y}, {x,y}]
   * representing the x and y coordinates, of the polygon's vertices, in sequence.
   * @param close - A boolean indicating whether to close the polygon path. True by default.
   */
  poly(points, close) {
    this._tick++;
    this._activePath.poly(points, close, this._transform.clone());
    return this;
  }
  /**
   * Draws a regular polygon with a specified number of sides. All sides and angles are equal.
   * @param x - The x-coordinate of the center of the polygon.
   * @param y - The y-coordinate of the center of the polygon.
   * @param radius - The radius of the circumscribed circle of the polygon.
   * @param sides - The number of sides of the polygon. Must be 3 or more.
   * @param rotation - The rotation angle of the polygon, in radians. Zero by default.
   * @param transform - An optional `Matrix` object to apply a transformation to the polygon.
   * @returns The instance of the current object for chaining.
   */
  regularPoly(x2, y2, radius, sides, rotation = 0, transform) {
    this._tick++;
    this._activePath.regularPoly(x2, y2, radius, sides, rotation, transform);
    return this;
  }
  /**
   * Draws a polygon with rounded corners.
   * Similar to `regularPoly` but with the ability to round the corners of the polygon.
   * @param x - The x-coordinate of the center of the polygon.
   * @param y - The y-coordinate of the center of the polygon.
   * @param radius - The radius of the circumscribed circle of the polygon.
   * @param sides - The number of sides of the polygon. Must be 3 or more.
   * @param corner - The radius of the rounding of the corners.
   * @param rotation - The rotation angle of the polygon, in radians. Zero by default.
   * @returns The instance of the current object for chaining.
   */
  roundPoly(x2, y2, radius, sides, corner, rotation) {
    this._tick++;
    this._activePath.roundPoly(x2, y2, radius, sides, corner, rotation);
    return this;
  }
  /**
   * Draws a shape with rounded corners. This function supports custom radius for each corner of the shape.
   * Optionally, corners can be rounded using a quadratic curve instead of an arc, providing a different aesthetic.
   * @param points - An array of `RoundedPoint` representing the corners of the shape to draw.
   * A minimum of 3 points is required.
   * @param radius - The default radius for the corners.
   * This radius is applied to all corners unless overridden in `points`.
   * @param useQuadratic - If set to true, rounded corners are drawn using a quadraticCurve
   *  method instead of an arc method. Defaults to false.
   * @param smoothness - Specifies the smoothness of the curve when `useQuadratic` is true.
   * Higher values make the curve smoother.
   * @returns The instance of the current object for chaining.
   */
  roundShape(points, radius, useQuadratic, smoothness) {
    this._tick++;
    this._activePath.roundShape(points, radius, useQuadratic, smoothness);
    return this;
  }
  /**
   * Draw Rectangle with fillet corners. This is much like rounded rectangle
   * however it support negative numbers as well for the corner radius.
   * @param x - Upper left corner of rect
   * @param y - Upper right corner of rect
   * @param width - Width of rect
   * @param height - Height of rect
   * @param fillet - accept negative or positive values
   */
  filletRect(x2, y2, width, height, fillet) {
    this._tick++;
    this._activePath.filletRect(x2, y2, width, height, fillet);
    return this;
  }
  /**
   * Draw Rectangle with chamfer corners. These are angled corners.
   * @param x - Upper left corner of rect
   * @param y - Upper right corner of rect
   * @param width - Width of rect
   * @param height - Height of rect
   * @param chamfer - non-zero real number, size of corner cutout
   * @param transform
   */
  chamferRect(x2, y2, width, height, chamfer, transform) {
    this._tick++;
    this._activePath.chamferRect(x2, y2, width, height, chamfer, transform);
    return this;
  }
  /**
   * Draws a star shape centered at a specified location. This method allows for the creation
   *  of stars with a variable number of points, outer radius, optional inner radius, and rotation.
   * The star is drawn as a closed polygon with alternating outer and inner vertices to create the star's points.
   * An optional transformation can be applied to scale, rotate, or translate the star as needed.
   * @param x - The x-coordinate of the center of the star.
   * @param y - The y-coordinate of the center of the star.
   * @param points - The number of points of the star.
   * @param radius - The outer radius of the star (distance from the center to the outer points).
   * @param innerRadius - Optional. The inner radius of the star
   * (distance from the center to the inner points between the outer points).
   * If not provided, defaults to half of the `radius`.
   * @param rotation - Optional. The rotation of the star in radians, where 0 is aligned with the y-axis.
   * Defaults to 0, meaning one point is directly upward.
   * @returns The instance of the current object for chaining further drawing commands.
   */
  star(x2, y2, points, radius, innerRadius = 0, rotation = 0) {
    this._tick++;
    this._activePath.star(x2, y2, points, radius, innerRadius, rotation, this._transform.clone());
    return this;
  }
  /**
   * Parses and renders an SVG string into the graphics context. This allows for complex shapes and paths
   * defined in SVG format to be drawn within the graphics context.
   * @param svg - The SVG string to be parsed and rendered.
   */
  svg(svg) {
    this._tick++;
    SVGParser(svg, this);
    return this;
  }
  /**
   * Restores the most recently saved graphics state by popping the top of the graphics state stack.
   * This includes transformations, fill styles, and stroke styles.
   */
  restore() {
    const state = this._stateStack.pop();
    if (state) {
      this._transform = state.transform;
      this._fillStyle = state.fillStyle;
      this._strokeStyle = state.strokeStyle;
    }
    return this;
  }
  /** Saves the current graphics state, including transformations, fill styles, and stroke styles, onto a stack. */
  save() {
    this._stateStack.push({
      transform: this._transform.clone(),
      fillStyle: { ...this._fillStyle },
      strokeStyle: { ...this._strokeStyle }
    });
    return this;
  }
  /**
   * Returns the current transformation matrix of the graphics context.
   * @returns The current transformation matrix.
   */
  getTransform() {
    return this._transform;
  }
  /**
   * Resets the current transformation matrix to the identity matrix, effectively removing any transformations (rotation, scaling, translation) previously applied.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  resetTransform() {
    this._transform.identity();
    return this;
  }
  /**
   * Applies a rotation transformation to the graphics context around the current origin.
   * @param angle - The angle of rotation in radians.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  rotate(angle) {
    this._transform.rotate(angle);
    return this;
  }
  /**
   * Applies a scaling transformation to the graphics context, scaling drawings by x horizontally and by y vertically.
   * @param x - The scale factor in the horizontal direction.
   * @param y - (Optional) The scale factor in the vertical direction. If not specified, the x value is used for both directions.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  scale(x2, y2 = x2) {
    this._transform.scale(x2, y2);
    return this;
  }
  setTransform(a2, b2, c2, d2, dx, dy) {
    if (a2 instanceof Matrix) {
      this._transform.set(a2.a, a2.b, a2.c, a2.d, a2.tx, a2.ty);
      return this;
    }
    this._transform.set(a2, b2, c2, d2, dx, dy);
    return this;
  }
  transform(a2, b2, c2, d2, dx, dy) {
    if (a2 instanceof Matrix) {
      this._transform.append(a2);
      return this;
    }
    tempMatrix.set(a2, b2, c2, d2, dx, dy);
    this._transform.append(tempMatrix);
    return this;
  }
  /**
   * Applies a translation transformation to the graphics context, moving the origin by the specified amounts.
   * @param x - The amount to translate in the horizontal direction.
   * @param y - (Optional) The amount to translate in the vertical direction. If not specified, the x value is used for both directions.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  translate(x2, y2 = x2) {
    this._transform.translate(x2, y2);
    return this;
  }
  /**
   * Clears all drawing commands from the graphics context, effectively resetting it. This includes clearing the path,
   * and optionally resetting transformations to the identity matrix.
   * @returns The instance of the current GraphicsContext for method chaining.
   */
  clear() {
    this._activePath.clear();
    this.instructions.length = 0;
    this.resetTransform();
    this.onUpdate();
    return this;
  }
  onUpdate() {
    this._boundsDirty = true;
    if (this.dirty)
      return;
    this.emit("update", this, 16);
    this.dirty = true;
  }
  /** The bounds of the graphic shape. */
  get bounds() {
    if (!this._boundsDirty)
      return this._bounds;
    this._boundsDirty = false;
    const bounds = this._bounds;
    bounds.clear();
    for (let i2 = 0; i2 < this.instructions.length; i2++) {
      const instruction = this.instructions[i2];
      const action = instruction.action;
      if (action === "fill") {
        const data = instruction.data;
        bounds.addBounds(data.path.bounds);
      } else if (action === "texture") {
        const data = instruction.data;
        bounds.addFrame(data.dx, data.dy, data.dx + data.dw, data.dy + data.dh, data.transform);
      }
      if (action === "stroke") {
        const data = instruction.data;
        const alignment = data.style.alignment;
        const outerPadding = data.style.width * (1 - alignment);
        const _bounds = data.path.bounds;
        bounds.addFrame(
          _bounds.minX - outerPadding,
          _bounds.minY - outerPadding,
          _bounds.maxX + outerPadding,
          _bounds.maxY + outerPadding
        );
      }
    }
    return bounds;
  }
  /**
   * Check to see if a point is contained within this geometry.
   * @param point - Point to check if it's contained.
   * @returns {boolean} `true` if the point is contained within geometry.
   */
  containsPoint(point) {
    if (!this.bounds.containsPoint(point.x, point.y))
      return false;
    const instructions = this.instructions;
    let hasHit = false;
    for (let k2 = 0; k2 < instructions.length; k2++) {
      const instruction = instructions[k2];
      const data = instruction.data;
      const path2 = data.path;
      if (!instruction.action || !path2)
        continue;
      const style = data.style;
      const shapes = path2.shapePath.shapePrimitives;
      for (let i2 = 0; i2 < shapes.length; i2++) {
        const shape = shapes[i2].shape;
        if (!style || !shape)
          continue;
        const transform = shapes[i2].transform;
        const transformedPoint = transform ? transform.applyInverse(point, tmpPoint) : point;
        if (instruction.action === "fill") {
          hasHit = shape.contains(transformedPoint.x, transformedPoint.y);
        } else {
          const strokeStyle = style;
          hasHit = shape.strokeContains(transformedPoint.x, transformedPoint.y, strokeStyle.width, strokeStyle.alignment);
        }
        const holes = data.hole;
        if (holes) {
          const holeShapes = holes.shapePath?.shapePrimitives;
          if (holeShapes) {
            for (let j2 = 0; j2 < holeShapes.length; j2++) {
              if (holeShapes[j2].shape.contains(transformedPoint.x, transformedPoint.y)) {
                hasHit = false;
              }
            }
          }
        }
        if (hasHit) {
          return true;
        }
      }
    }
    return hasHit;
  }
  /** Unloads the GPU data from the graphics context. */
  unload() {
    this.emit("unload", this);
    for (const key in this._gpuData) {
      this._gpuData[key]?.destroy();
    }
    this._gpuData = /* @__PURE__ */ Object.create(null);
  }
  /**
   * Destroys the GraphicsData object.
   * @param options - Options parameter. A boolean will act as if all options
   *  have been set to that value
   * @example
   * context.destroy();
   * context.destroy(true);
   * context.destroy({ texture: true, textureSource: true });
   */
  destroy(options = false) {
    if (this.destroyed)
      return;
    this.destroyed = true;
    this._stateStack.length = 0;
    this._transform = null;
    this.unload();
    this.emit("destroy", this);
    this.removeAllListeners();
    const destroyTexture = typeof options === "boolean" ? options : options?.texture;
    if (destroyTexture) {
      const destroyTextureSource = typeof options === "boolean" ? options : options?.textureSource;
      if (this._fillStyle.texture) {
        this._fillStyle.fill && "uid" in this._fillStyle.fill ? this._fillStyle.fill.destroy() : this._fillStyle.texture.destroy(destroyTextureSource);
      }
      if (this._strokeStyle.texture) {
        this._strokeStyle.fill && "uid" in this._strokeStyle.fill ? this._strokeStyle.fill.destroy() : this._strokeStyle.texture.destroy(destroyTextureSource);
      }
    }
    this._fillStyle = null;
    this._strokeStyle = null;
    this.instructions = null;
    this._activePath = null;
    this._bounds = null;
    this._stateStack = null;
    this.customShader = null;
    this._transform = null;
  }
};
_GraphicsContext.defaultFillStyle = {
  /** The color to use for the fill. */
  color: 16777215,
  /** The alpha value to use for the fill. */
  alpha: 1,
  /** The texture to use for the fill. */
  texture: Texture.WHITE,
  /** The matrix to apply. */
  matrix: null,
  /** The fill pattern to use. */
  fill: null,
  /** Whether coordinates are 'global' or 'local' */
  textureSpace: "local"
};
_GraphicsContext.defaultStrokeStyle = {
  /** The width of the stroke. */
  width: 1,
  /** The color to use for the stroke. */
  color: 16777215,
  /** The alpha value to use for the stroke. */
  alpha: 1,
  /** The alignment of the stroke. */
  alignment: 0.5,
  /** The miter limit to use. */
  miterLimit: 10,
  /** The line cap style to use. */
  cap: "butt",
  /** The line join style to use. */
  join: "miter",
  /** The texture to use for the fill. */
  texture: Texture.WHITE,
  /** The matrix to apply. */
  matrix: null,
  /** The fill pattern to use. */
  fill: null,
  /** Whether coordinates are 'global' or 'local' */
  textureSpace: "local",
  /** If the stroke is a pixel line. */
  pixelLine: false
};
let GraphicsContext = _GraphicsContext;
const _TextStyle = class _TextStyle2 extends EventEmitter {
  constructor(style = {}) {
    super();
    this.uid = uid("textStyle");
    this._tick = 0;
    convertV7Tov8Style(style);
    const fullStyle = { ..._TextStyle2.defaultTextStyle, ...style };
    for (const key in fullStyle) {
      const thisKey = key;
      this[thisKey] = fullStyle[key];
    }
    this.update();
    this._tick = 0;
  }
  /**
   * Alignment for multiline text, does not affect single line text.
   * @type {'left'|'center'|'right'|'justify'}
   */
  get align() {
    return this._align;
  }
  set align(value) {
    if (this._align === value)
      return;
    this._align = value;
    this.update();
  }
  /** Indicates if lines can be wrapped within words, it needs wordWrap to be set to true. */
  get breakWords() {
    return this._breakWords;
  }
  set breakWords(value) {
    if (this._breakWords === value)
      return;
    this._breakWords = value;
    this.update();
  }
  /** Set a drop shadow for the text. */
  get dropShadow() {
    return this._dropShadow;
  }
  set dropShadow(value) {
    if (this._dropShadow === value)
      return;
    if (value !== null && typeof value === "object") {
      this._dropShadow = this._createProxy({ ..._TextStyle2.defaultDropShadow, ...value });
    } else {
      this._dropShadow = value ? this._createProxy({ ..._TextStyle2.defaultDropShadow }) : null;
    }
    this.update();
  }
  /** The font family, can be a single font name, or a list of names where the first is the preferred font. */
  get fontFamily() {
    return this._fontFamily;
  }
  set fontFamily(value) {
    if (this._fontFamily === value)
      return;
    this._fontFamily = value;
    this.update();
  }
  /** The font size (as a number it converts to px, but as a string, equivalents are '26px','20pt','160%' or '1.6em') */
  get fontSize() {
    return this._fontSize;
  }
  set fontSize(value) {
    if (this._fontSize === value)
      return;
    if (typeof value === "string") {
      this._fontSize = parseInt(value, 10);
    } else {
      this._fontSize = value;
    }
    this.update();
  }
  /**
   * The font style.
   * @type {'normal'|'italic'|'oblique'}
   */
  get fontStyle() {
    return this._fontStyle;
  }
  set fontStyle(value) {
    if (this._fontStyle === value)
      return;
    this._fontStyle = value.toLowerCase();
    this.update();
  }
  /**
   * The font variant.
   * @type {'normal'|'small-caps'}
   */
  get fontVariant() {
    return this._fontVariant;
  }
  set fontVariant(value) {
    if (this._fontVariant === value)
      return;
    this._fontVariant = value;
    this.update();
  }
  /**
   * The font weight.
   * @type {'normal'|'bold'|'bolder'|'lighter'|'100'|'200'|'300'|'400'|'500'|'600'|'700'|'800'|'900'}
   */
  get fontWeight() {
    return this._fontWeight;
  }
  set fontWeight(value) {
    if (this._fontWeight === value)
      return;
    this._fontWeight = value;
    this.update();
  }
  /** The space between lines. */
  get leading() {
    return this._leading;
  }
  set leading(value) {
    if (this._leading === value)
      return;
    this._leading = value;
    this.update();
  }
  /** The amount of spacing between letters, default is 0. */
  get letterSpacing() {
    return this._letterSpacing;
  }
  set letterSpacing(value) {
    if (this._letterSpacing === value)
      return;
    this._letterSpacing = value;
    this.update();
  }
  /** The line height, a number that represents the vertical space that a letter uses. */
  get lineHeight() {
    return this._lineHeight;
  }
  set lineHeight(value) {
    if (this._lineHeight === value)
      return;
    this._lineHeight = value;
    this.update();
  }
  /**
   * Occasionally some fonts are cropped. Adding some padding will prevent this from happening
   * by adding padding to all sides of the text.
   * > [!NOTE] This will NOT affect the positioning or bounds of the text.
   */
  get padding() {
    return this._padding;
  }
  set padding(value) {
    if (this._padding === value)
      return;
    this._padding = value;
    this.update();
  }
  /**
   * An optional filter or array of filters to apply to the text, allowing for advanced visual effects.
   * These filters will be applied to the text as it is created, resulting in faster rendering for static text
   * compared to applying the filter directly to the text object (which would be applied at run time).
   * @default null
   */
  get filters() {
    return this._filters;
  }
  set filters(value) {
    if (this._filters === value)
      return;
    this._filters = Object.freeze(value);
    this.update();
  }
  /**
   * Trim transparent borders from the text texture.
   * > [!IMPORTANT] PERFORMANCE WARNING:
   * > This is a costly operation as it requires scanning pixel alpha values.
   * > Avoid using `trim: true` for dynamic text, as it could significantly impact performance.
   */
  get trim() {
    return this._trim;
  }
  set trim(value) {
    if (this._trim === value)
      return;
    this._trim = value;
    this.update();
  }
  /**
   * The baseline of the text that is rendered.
   * @type {'alphabetic'|'top'|'hanging'|'middle'|'ideographic'|'bottom'}
   */
  get textBaseline() {
    return this._textBaseline;
  }
  set textBaseline(value) {
    if (this._textBaseline === value)
      return;
    this._textBaseline = value;
    this.update();
  }
  /**
   * How newlines and spaces should be handled.
   * Default is 'pre' (preserve, preserve).
   *
   *  value       | New lines     |   Spaces
   *  ---         | ---           |   ---
   * 'normal'     | Collapse      |   Collapse
   * 'pre'        | Preserve      |   Preserve
   * 'pre-line'   | Preserve      |   Collapse
   * @type {'normal'|'pre'|'pre-line'}
   */
  get whiteSpace() {
    return this._whiteSpace;
  }
  set whiteSpace(value) {
    if (this._whiteSpace === value)
      return;
    this._whiteSpace = value;
    this.update();
  }
  /** Indicates if word wrap should be used. */
  get wordWrap() {
    return this._wordWrap;
  }
  set wordWrap(value) {
    if (this._wordWrap === value)
      return;
    this._wordWrap = value;
    this.update();
  }
  /** The width at which text will wrap, it needs wordWrap to be set to true. */
  get wordWrapWidth() {
    return this._wordWrapWidth;
  }
  set wordWrapWidth(value) {
    if (this._wordWrapWidth === value)
      return;
    this._wordWrapWidth = value;
    this.update();
  }
  /**
   * The fill style that will be used to color the text.
   * This can be:
   * - A color string like 'red', '#00FF00', or 'rgba(255,0,0,0.5)'
   * - A hex number like 0xff0000 for red
   * - A FillStyle object with properties like { color: 0xff0000, alpha: 0.5 }
   * - A FillGradient for gradient fills
   * - A FillPattern for pattern/texture fills
   *
   * When using a FillGradient, vertical gradients (angle of 90 degrees) are applied per line of text,
   * while gradients at any other angle are spread across the entire text body as a whole.
   * @example
   * // Vertical gradient applied per line
   * const verticalGradient = new FillGradient(0, 0, 0, 1)
   *     .addColorStop(0, 0xff0000)
   *     .addColorStop(1, 0x0000ff);
   *
   * const text = new Text({
   *     text: 'Line 1\nLine 2',
   *     style: { fill: verticalGradient }
   * });
   *
   * To manage the gradient in a global scope, set the textureSpace property of the FillGradient to 'global'.
   * @type {string|number|FillStyle|FillGradient|FillPattern}
   */
  get fill() {
    return this._originalFill;
  }
  set fill(value) {
    if (value === this._originalFill)
      return;
    this._originalFill = value;
    if (this._isFillStyle(value)) {
      this._originalFill = this._createProxy({ ...GraphicsContext.defaultFillStyle, ...value }, () => {
        this._fill = toFillStyle(
          { ...this._originalFill },
          GraphicsContext.defaultFillStyle
        );
      });
    }
    this._fill = toFillStyle(
      value === 0 ? "black" : value,
      GraphicsContext.defaultFillStyle
    );
    this.update();
  }
  /** A fillstyle that will be used on the text stroke, e.g., 'blue', '#FCFF00'. */
  get stroke() {
    return this._originalStroke;
  }
  set stroke(value) {
    if (value === this._originalStroke)
      return;
    this._originalStroke = value;
    if (this._isFillStyle(value)) {
      this._originalStroke = this._createProxy({ ...GraphicsContext.defaultStrokeStyle, ...value }, () => {
        this._stroke = toStrokeStyle(
          { ...this._originalStroke },
          GraphicsContext.defaultStrokeStyle
        );
      });
    }
    this._stroke = toStrokeStyle(value, GraphicsContext.defaultStrokeStyle);
    this.update();
  }
  update() {
    this._tick++;
    this.emit("update", this);
  }
  /** Resets all properties to the default values */
  reset() {
    const defaultStyle = _TextStyle2.defaultTextStyle;
    for (const key in defaultStyle) {
      this[key] = defaultStyle[key];
    }
  }
  /**
   * Returns a unique key for this instance.
   * This key is used for caching.
   * @returns {string} Unique key for the instance
   */
  get styleKey() {
    return `${this.uid}-${this._tick}`;
  }
  /**
   * Creates a new TextStyle object with the same values as this one.
   * @returns New cloned TextStyle object
   */
  clone() {
    return new _TextStyle2({
      align: this.align,
      breakWords: this.breakWords,
      dropShadow: this._dropShadow ? { ...this._dropShadow } : null,
      fill: this._fill,
      fontFamily: this.fontFamily,
      fontSize: this.fontSize,
      fontStyle: this.fontStyle,
      fontVariant: this.fontVariant,
      fontWeight: this.fontWeight,
      leading: this.leading,
      letterSpacing: this.letterSpacing,
      lineHeight: this.lineHeight,
      padding: this.padding,
      stroke: this._stroke,
      textBaseline: this.textBaseline,
      whiteSpace: this.whiteSpace,
      wordWrap: this.wordWrap,
      wordWrapWidth: this.wordWrapWidth,
      filters: this._filters ? [...this._filters] : void 0
    });
  }
  /**
   * Returns the final padding for the text style, taking into account any filters applied.
   * Used internally for correct measurements
   * @internal
   * @returns {number} The final padding for the text style.
   */
  _getFinalPadding() {
    let filterPadding = 0;
    if (this._filters) {
      for (let i2 = 0; i2 < this._filters.length; i2++) {
        filterPadding += this._filters[i2].padding;
      }
    }
    return Math.max(this._padding, filterPadding);
  }
  /**
   * Destroys this text style.
   * @param options - Options parameter. A boolean will act as if all options
   *  have been set to that value
   * @example
   * // Destroy the text style and its textures
   * textStyle.destroy({ texture: true, textureSource: true });
   * textStyle.destroy(true);
   */
  destroy(options = false) {
    this.removeAllListeners();
    const destroyTexture = typeof options === "boolean" ? options : options?.texture;
    if (destroyTexture) {
      const destroyTextureSource = typeof options === "boolean" ? options : options?.textureSource;
      if (this._fill?.texture) {
        this._fill.texture.destroy(destroyTextureSource);
      }
      if (this._originalFill?.texture) {
        this._originalFill.texture.destroy(destroyTextureSource);
      }
      if (this._stroke?.texture) {
        this._stroke.texture.destroy(destroyTextureSource);
      }
      if (this._originalStroke?.texture) {
        this._originalStroke.texture.destroy(destroyTextureSource);
      }
    }
    this._fill = null;
    this._stroke = null;
    this.dropShadow = null;
    this._originalStroke = null;
    this._originalFill = null;
  }
  _createProxy(value, cb) {
    return new Proxy(value, {
      set: (target, property, newValue) => {
        if (target[property] === newValue)
          return true;
        target[property] = newValue;
        cb?.(property, newValue);
        this.update();
        return true;
      }
    });
  }
  _isFillStyle(value) {
    return (value ?? null) !== null && !(Color.isColorLike(value) || value instanceof FillGradient || value instanceof FillPattern);
  }
};
_TextStyle.defaultDropShadow = {
  alpha: 1,
  angle: Math.PI / 6,
  blur: 0,
  color: "black",
  distance: 5
};
_TextStyle.defaultTextStyle = {
  align: "left",
  breakWords: false,
  dropShadow: null,
  fill: "black",
  fontFamily: "Arial",
  fontSize: 26,
  fontStyle: "normal",
  fontVariant: "normal",
  fontWeight: "normal",
  leading: 0,
  letterSpacing: 0,
  lineHeight: 0,
  padding: 0,
  stroke: null,
  textBaseline: "alphabetic",
  trim: false,
  whiteSpace: "pre",
  wordWrap: false,
  wordWrapWidth: 100
};
let TextStyle = _TextStyle;
function convertV7Tov8Style(style) {
  const oldStyle = style;
  if (typeof oldStyle.dropShadow === "boolean" && oldStyle.dropShadow) {
    const defaults = TextStyle.defaultDropShadow;
    style.dropShadow = {
      alpha: oldStyle.dropShadowAlpha ?? defaults.alpha,
      angle: oldStyle.dropShadowAngle ?? defaults.angle,
      blur: oldStyle.dropShadowBlur ?? defaults.blur,
      color: oldStyle.dropShadowColor ?? defaults.color,
      distance: oldStyle.dropShadowDistance ?? defaults.distance
    };
  }
  if (oldStyle.strokeThickness !== void 0) {
    deprecation(v8_0_0, "strokeThickness is now a part of stroke");
    const color = oldStyle.stroke;
    let obj = {};
    if (Color.isColorLike(color)) {
      obj.color = color;
    } else if (color instanceof FillGradient || color instanceof FillPattern) {
      obj.fill = color;
    } else if (Object.hasOwnProperty.call(color, "color") || Object.hasOwnProperty.call(color, "fill")) {
      obj = color;
    } else {
      throw new Error("Invalid stroke value.");
    }
    style.stroke = {
      ...obj,
      width: oldStyle.strokeThickness
    };
  }
  if (Array.isArray(oldStyle.fillGradientStops)) {
    deprecation(v8_0_0, "gradient fill is now a fill pattern: `new FillGradient(...)`");
    if (!Array.isArray(oldStyle.fill) || oldStyle.fill.length === 0) {
      throw new Error("Invalid fill value. Expected an array of colors for gradient fill.");
    }
    if (oldStyle.fill.length !== oldStyle.fillGradientStops.length) {
      warn("The number of fill colors must match the number of fill gradient stops.");
    }
    const gradientFill = new FillGradient({
      start: { x: 0, y: 0 },
      end: { x: 0, y: 1 },
      textureSpace: "local"
    });
    const fillGradientStops = oldStyle.fillGradientStops.slice();
    const fills = oldStyle.fill.map((color) => Color.shared.setValue(color).toNumber());
    fillGradientStops.forEach((stop, index) => {
      gradientFill.addColorStop(stop, fills[index]);
    });
    style.fill = {
      fill: gradientFill
    };
  }
}
class CanvasPoolClass {
  constructor(canvasOptions) {
    this._canvasPool = /* @__PURE__ */ Object.create(null);
    this.canvasOptions = canvasOptions || {};
    this.enableFullScreen = false;
  }
  /**
   * Creates texture with params that were specified in pool constructor.
   * @param pixelWidth - Width of texture in pixels.
   * @param pixelHeight - Height of texture in pixels.
   */
  _createCanvasAndContext(pixelWidth, pixelHeight) {
    const canvas = DOMAdapter.get().createCanvas();
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    const context2 = canvas.getContext("2d");
    return { canvas, context: context2 };
  }
  /**
   * Gets a Power-of-Two render texture or fullScreen texture
   * @param minWidth - The minimum width of the render texture.
   * @param minHeight - The minimum height of the render texture.
   * @param resolution - The resolution of the render texture.
   * @returns The new render texture.
   */
  getOptimalCanvasAndContext(minWidth, minHeight, resolution = 1) {
    minWidth = Math.ceil(minWidth * resolution - 1e-6);
    minHeight = Math.ceil(minHeight * resolution - 1e-6);
    minWidth = nextPow2(minWidth);
    minHeight = nextPow2(minHeight);
    const key = (minWidth << 17) + (minHeight << 1);
    if (!this._canvasPool[key]) {
      this._canvasPool[key] = [];
    }
    let canvasAndContext = this._canvasPool[key].pop();
    if (!canvasAndContext) {
      canvasAndContext = this._createCanvasAndContext(minWidth, minHeight);
    }
    return canvasAndContext;
  }
  /**
   * Place a render texture back into the pool.
   * @param canvasAndContext
   */
  returnCanvasAndContext(canvasAndContext) {
    const canvas = canvasAndContext.canvas;
    const { width, height } = canvas;
    const key = (width << 17) + (height << 1);
    canvasAndContext.context.resetTransform();
    canvasAndContext.context.clearRect(0, 0, width, height);
    this._canvasPool[key].push(canvasAndContext);
  }
  clear() {
    this._canvasPool = {};
  }
}
const CanvasPool = new CanvasPoolClass();
GlobalResourceRegistry.register(CanvasPool);
const PRECISION = 1e5;
function getCanvasFillStyle(fillStyle, context2, textMetrics, padding = 0) {
  if (fillStyle.texture === Texture.WHITE && !fillStyle.fill) {
    return Color.shared.setValue(fillStyle.color).setAlpha(fillStyle.alpha ?? 1).toHexa();
  } else if (!fillStyle.fill) {
    const pattern = context2.createPattern(fillStyle.texture.source.resource, "repeat");
    const tempMatrix2 = fillStyle.matrix.copyTo(Matrix.shared);
    tempMatrix2.scale(fillStyle.texture.frame.width, fillStyle.texture.frame.height);
    pattern.setTransform(tempMatrix2);
    return pattern;
  } else if (fillStyle.fill instanceof FillPattern) {
    const fillPattern = fillStyle.fill;
    const pattern = context2.createPattern(fillPattern.texture.source.resource, "repeat");
    const tempMatrix2 = fillPattern.transform.copyTo(Matrix.shared);
    tempMatrix2.scale(
      fillPattern.texture.frame.width,
      fillPattern.texture.frame.height
    );
    pattern.setTransform(tempMatrix2);
    return pattern;
  } else if (fillStyle.fill instanceof FillGradient) {
    const fillGradient = fillStyle.fill;
    const isLinear = fillGradient.type === "linear";
    const isLocal = fillGradient.textureSpace === "local";
    let width = 1;
    let height = 1;
    if (isLocal && textMetrics) {
      width = textMetrics.width + padding;
      height = textMetrics.height + padding;
    }
    let gradient;
    let isNearlyVertical = false;
    if (isLinear) {
      const { start, end } = fillGradient;
      gradient = context2.createLinearGradient(
        start.x * width,
        start.y * height,
        end.x * width,
        end.y * height
      );
      isNearlyVertical = Math.abs(end.x - start.x) < Math.abs((end.y - start.y) * 0.1);
    } else {
      const { center, innerRadius, outerCenter, outerRadius } = fillGradient;
      gradient = context2.createRadialGradient(
        center.x * width,
        center.y * height,
        innerRadius * width,
        outerCenter.x * width,
        outerCenter.y * height,
        outerRadius * width
      );
    }
    if (isNearlyVertical && isLocal && textMetrics) {
      const ratio = textMetrics.lineHeight / height;
      for (let i2 = 0; i2 < textMetrics.lines.length; i2++) {
        const start = (i2 * textMetrics.lineHeight + padding / 2) / height;
        fillGradient.colorStops.forEach((stop) => {
          const globalStop = start + stop.offset * ratio;
          gradient.addColorStop(
            // fix to 5 decimal places to avoid floating point precision issues
            Math.floor(globalStop * PRECISION) / PRECISION,
            Color.shared.setValue(stop.color).toHex()
          );
        });
      }
    } else {
      fillGradient.colorStops.forEach((stop) => {
        gradient.addColorStop(stop.offset, Color.shared.setValue(stop.color).toHex());
      });
    }
    return gradient;
  }
  warn("FillStyle not recognised", fillStyle);
  return "red";
}
const _DynamicBitmapFont = class _DynamicBitmapFont2 extends AbstractBitmapFont {
  /**
   * @param options - The options for the dynamic bitmap font.
   */
  constructor(options) {
    super();
    this.resolution = 1;
    this.pages = [];
    this._padding = 0;
    this._measureCache = /* @__PURE__ */ Object.create(null);
    this._currentChars = [];
    this._currentX = 0;
    this._currentY = 0;
    this._currentMaxCharHeight = 0;
    this._currentPageIndex = -1;
    this._skipKerning = false;
    const dynamicOptions = { ..._DynamicBitmapFont2.defaultOptions, ...options };
    this._textureSize = dynamicOptions.textureSize;
    this._mipmap = dynamicOptions.mipmap;
    const style = dynamicOptions.style.clone();
    if (dynamicOptions.overrideFill) {
      style._fill.color = 16777215;
      style._fill.alpha = 1;
      style._fill.texture = Texture.WHITE;
      style._fill.fill = null;
    }
    this.applyFillAsTint = dynamicOptions.overrideFill;
    const requestedFontSize = style.fontSize;
    style.fontSize = this.baseMeasurementFontSize;
    const font = fontStringFromTextStyle(style);
    if (dynamicOptions.overrideSize) {
      if (style._stroke) {
        style._stroke.width *= this.baseRenderedFontSize / requestedFontSize;
      }
    } else {
      style.fontSize = this.baseRenderedFontSize = requestedFontSize;
    }
    this._style = style;
    this._skipKerning = dynamicOptions.skipKerning ?? false;
    this.resolution = dynamicOptions.resolution ?? 1;
    this._padding = dynamicOptions.padding ?? 4;
    if (dynamicOptions.textureStyle) {
      this._textureStyle = dynamicOptions.textureStyle instanceof TextureStyle ? dynamicOptions.textureStyle : new TextureStyle(dynamicOptions.textureStyle);
    }
    this.fontMetrics = CanvasTextMetrics.measureFont(font);
    this.lineHeight = style.lineHeight || this.fontMetrics.fontSize || style.fontSize;
  }
  ensureCharacters(chars) {
    const charList = CanvasTextMetrics.graphemeSegmenter(chars).filter((char) => !this._currentChars.includes(char)).filter((char, index, self2) => self2.indexOf(char) === index);
    if (!charList.length)
      return;
    this._currentChars = [...this._currentChars, ...charList];
    let pageData;
    if (this._currentPageIndex === -1) {
      pageData = this._nextPage();
    } else {
      pageData = this.pages[this._currentPageIndex];
    }
    let { canvas, context: context2 } = pageData.canvasAndContext;
    let textureSource = pageData.texture.source;
    const style = this._style;
    let currentX = this._currentX;
    let currentY = this._currentY;
    let currentMaxCharHeight = this._currentMaxCharHeight;
    const fontScale = this.baseRenderedFontSize / this.baseMeasurementFontSize;
    const padding = this._padding * fontScale;
    let skipTexture = false;
    const maxTextureWidth = canvas.width / this.resolution;
    const maxTextureHeight = canvas.height / this.resolution;
    for (let i2 = 0; i2 < charList.length; i2++) {
      const char = charList[i2];
      const metrics = CanvasTextMetrics.measureText(char, style, canvas, false);
      metrics.lineHeight = metrics.height;
      const width = metrics.width * fontScale;
      const textureGlyphWidth = Math.ceil((style.fontStyle === "italic" ? 2 : 1) * width);
      const height = metrics.height * fontScale;
      const paddedWidth = textureGlyphWidth + padding * 2;
      const paddedHeight = height + padding * 2;
      skipTexture = false;
      if (char !== "\n" && char !== "\r" && char !== "	" && char !== " ") {
        skipTexture = true;
        currentMaxCharHeight = Math.ceil(Math.max(paddedHeight, currentMaxCharHeight));
      }
      if (currentX + paddedWidth > maxTextureWidth) {
        currentY += currentMaxCharHeight;
        currentMaxCharHeight = paddedHeight;
        currentX = 0;
        if (currentY + currentMaxCharHeight > maxTextureHeight) {
          textureSource.update();
          const pageData2 = this._nextPage();
          canvas = pageData2.canvasAndContext.canvas;
          context2 = pageData2.canvasAndContext.context;
          textureSource = pageData2.texture.source;
          currentX = 0;
          currentY = 0;
          currentMaxCharHeight = 0;
        }
      }
      const xAdvance = width / fontScale - (style.dropShadow?.distance ?? 0) - (style._stroke?.width ?? 0);
      this.chars[char] = {
        id: char.codePointAt(0),
        xOffset: -this._padding,
        yOffset: -this._padding,
        xAdvance,
        kerning: {}
      };
      if (skipTexture) {
        this._drawGlyph(
          context2,
          metrics,
          currentX + padding,
          currentY + padding,
          fontScale,
          style
        );
        const px = textureSource.width * fontScale;
        const py = textureSource.height * fontScale;
        const frame = new Rectangle(
          currentX / px * textureSource.width,
          currentY / py * textureSource.height,
          paddedWidth / px * textureSource.width,
          paddedHeight / py * textureSource.height
        );
        this.chars[char].texture = new Texture({
          source: textureSource,
          frame
        });
        currentX += Math.ceil(paddedWidth);
      }
    }
    textureSource.update();
    this._currentX = currentX;
    this._currentY = currentY;
    this._currentMaxCharHeight = currentMaxCharHeight;
    this._skipKerning && this._applyKerning(charList, context2);
  }
  /**
   * @deprecated since 8.0.0
   * The map of base page textures (i.e., sheets of glyphs).
   */
  get pageTextures() {
    deprecation(v8_0_0, "BitmapFont.pageTextures is deprecated, please use BitmapFont.pages instead.");
    return this.pages;
  }
  _applyKerning(newChars, context2) {
    const measureCache = this._measureCache;
    for (let i2 = 0; i2 < newChars.length; i2++) {
      const first = newChars[i2];
      for (let j2 = 0; j2 < this._currentChars.length; j2++) {
        const second = this._currentChars[j2];
        let c1 = measureCache[first];
        if (!c1)
          c1 = measureCache[first] = context2.measureText(first).width;
        let c2 = measureCache[second];
        if (!c2)
          c2 = measureCache[second] = context2.measureText(second).width;
        let total = context2.measureText(first + second).width;
        let amount = total - (c1 + c2);
        if (amount) {
          this.chars[first].kerning[second] = amount;
        }
        total = context2.measureText(first + second).width;
        amount = total - (c1 + c2);
        if (amount) {
          this.chars[second].kerning[first] = amount;
        }
      }
    }
  }
  _nextPage() {
    this._currentPageIndex++;
    const textureResolution = this.resolution;
    const canvasAndContext = CanvasPool.getOptimalCanvasAndContext(
      this._textureSize,
      this._textureSize,
      textureResolution
    );
    this._setupContext(canvasAndContext.context, this._style, textureResolution);
    const resolution = textureResolution * (this.baseRenderedFontSize / this.baseMeasurementFontSize);
    const texture = new Texture({
      source: new ImageSource({
        resource: canvasAndContext.canvas,
        resolution,
        alphaMode: "premultiply-alpha-on-upload",
        autoGenerateMipmaps: this._mipmap
      })
    });
    if (this._textureStyle) {
      texture.source.style = this._textureStyle;
    }
    const pageData = {
      canvasAndContext,
      texture
    };
    this.pages[this._currentPageIndex] = pageData;
    return pageData;
  }
  // canvas style!
  _setupContext(context2, style, resolution) {
    style.fontSize = this.baseRenderedFontSize;
    context2.scale(resolution, resolution);
    context2.font = fontStringFromTextStyle(style);
    style.fontSize = this.baseMeasurementFontSize;
    context2.textBaseline = style.textBaseline;
    const stroke = style._stroke;
    const strokeThickness = stroke?.width ?? 0;
    if (stroke) {
      context2.lineWidth = strokeThickness;
      context2.lineJoin = stroke.join;
      context2.miterLimit = stroke.miterLimit;
      context2.strokeStyle = getCanvasFillStyle(stroke, context2);
    }
    if (style._fill) {
      context2.fillStyle = getCanvasFillStyle(style._fill, context2);
    }
    if (style.dropShadow) {
      const shadowOptions = style.dropShadow;
      const rgb = Color.shared.setValue(shadowOptions.color).toArray();
      const dropShadowBlur = shadowOptions.blur * resolution;
      const dropShadowDistance = shadowOptions.distance * resolution;
      context2.shadowColor = `rgba(${rgb[0] * 255},${rgb[1] * 255},${rgb[2] * 255},${shadowOptions.alpha})`;
      context2.shadowBlur = dropShadowBlur;
      context2.shadowOffsetX = Math.cos(shadowOptions.angle) * dropShadowDistance;
      context2.shadowOffsetY = Math.sin(shadowOptions.angle) * dropShadowDistance;
    } else {
      context2.shadowColor = "black";
      context2.shadowBlur = 0;
      context2.shadowOffsetX = 0;
      context2.shadowOffsetY = 0;
    }
  }
  _drawGlyph(context2, metrics, x2, y2, fontScale, style) {
    const char = metrics.text;
    const fontProperties = metrics.fontProperties;
    const stroke = style._stroke;
    const strokeThickness = (stroke?.width ?? 0) * fontScale;
    const tx = x2 + strokeThickness / 2;
    const ty = y2 - strokeThickness / 2;
    const descent = fontProperties.descent * fontScale;
    const lineHeight = metrics.lineHeight * fontScale;
    let removeShadow = false;
    if (style.stroke && strokeThickness) {
      removeShadow = true;
      context2.strokeText(char, tx, ty + lineHeight - descent);
    }
    const { shadowBlur, shadowOffsetX, shadowOffsetY } = context2;
    if (style._fill) {
      if (removeShadow) {
        context2.shadowBlur = 0;
        context2.shadowOffsetX = 0;
        context2.shadowOffsetY = 0;
      }
      context2.fillText(char, tx, ty + lineHeight - descent);
    }
    if (removeShadow) {
      context2.shadowBlur = shadowBlur;
      context2.shadowOffsetX = shadowOffsetX;
      context2.shadowOffsetY = shadowOffsetY;
    }
  }
  destroy() {
    super.destroy();
    for (let i2 = 0; i2 < this.pages.length; i2++) {
      const { canvasAndContext, texture } = this.pages[i2];
      CanvasPool.returnCanvasAndContext(canvasAndContext);
      texture.destroy(true);
    }
    this.pages = null;
  }
};
_DynamicBitmapFont.defaultOptions = {
  textureSize: 512,
  style: new TextStyle(),
  mipmap: true
};
let DynamicBitmapFont = _DynamicBitmapFont;
function getBitmapTextLayout(chars, style, font, trimEnd) {
  const layoutData = {
    width: 0,
    height: 0,
    offsetY: 0,
    scale: style.fontSize / font.baseMeasurementFontSize,
    lines: [{
      width: 0,
      charPositions: [],
      spaceWidth: 0,
      spacesIndex: [],
      chars: []
    }]
  };
  layoutData.offsetY = font.baseLineOffset;
  let currentLine = layoutData.lines[0];
  let previousChar = null;
  let firstWord = true;
  const currentWord = {
    width: 0,
    start: 0,
    index: 0,
    // use index to not modify the array as we use it a lot!
    positions: [],
    chars: []
  };
  const scale = font.baseMeasurementFontSize / style.fontSize;
  const adjustedLetterSpacing = style.letterSpacing * scale;
  const adjustedWordWrapWidth = style.wordWrapWidth * scale;
  const adjustedLineHeight = style.lineHeight ? style.lineHeight * scale : font.lineHeight;
  const breakWords = style.wordWrap && style.breakWords;
  const nextWord = (word) => {
    const start = currentLine.width;
    for (let j2 = 0; j2 < currentWord.index; j2++) {
      const position = word.positions[j2];
      currentLine.chars.push(word.chars[j2]);
      currentLine.charPositions.push(position + start);
    }
    currentLine.width += word.width;
    firstWord = false;
    currentWord.width = 0;
    currentWord.index = 0;
    currentWord.chars.length = 0;
  };
  const nextLine = () => {
    let index = currentLine.chars.length - 1;
    if (trimEnd) {
      let lastChar = currentLine.chars[index];
      while (lastChar === " ") {
        currentLine.width -= font.chars[lastChar].xAdvance;
        lastChar = currentLine.chars[--index];
      }
    }
    layoutData.width = Math.max(layoutData.width, currentLine.width);
    currentLine = {
      width: 0,
      charPositions: [],
      chars: [],
      spaceWidth: 0,
      spacesIndex: []
    };
    firstWord = true;
    layoutData.lines.push(currentLine);
    layoutData.height += adjustedLineHeight;
  };
  const checkIsOverflow = (lineWidth) => lineWidth - adjustedLetterSpacing > adjustedWordWrapWidth;
  for (let i2 = 0; i2 < chars.length + 1; i2++) {
    let char;
    const isEnd = i2 === chars.length;
    if (!isEnd) {
      char = chars[i2];
    }
    const charData = font.chars[char] || font.chars[" "];
    const isSpace = /(?:\s)/.test(char);
    const isWordBreak = isSpace || char === "\r" || char === "\n" || isEnd;
    if (isWordBreak) {
      const addWordToNextLine = !firstWord && style.wordWrap && checkIsOverflow(currentLine.width + currentWord.width);
      if (addWordToNextLine) {
        nextLine();
        nextWord(currentWord);
        if (!isEnd) {
          currentLine.charPositions.push(0);
        }
      } else {
        currentWord.start = currentLine.width;
        nextWord(currentWord);
        if (!isEnd) {
          currentLine.charPositions.push(0);
        }
      }
      if (char === "\r" || char === "\n") {
        nextLine();
      } else if (!isEnd) {
        const spaceWidth = charData.xAdvance + (charData.kerning[previousChar] || 0) + adjustedLetterSpacing;
        currentLine.width += spaceWidth;
        currentLine.spaceWidth = spaceWidth;
        currentLine.spacesIndex.push(currentLine.charPositions.length);
        currentLine.chars.push(char);
      }
    } else {
      const kerning = charData.kerning[previousChar] || 0;
      const nextCharWidth = charData.xAdvance + kerning + adjustedLetterSpacing;
      const addWordToNextLine = breakWords && checkIsOverflow(currentLine.width + currentWord.width + nextCharWidth);
      if (addWordToNextLine) {
        nextWord(currentWord);
        nextLine();
      }
      currentWord.positions[currentWord.index++] = currentWord.width + kerning;
      currentWord.chars.push(char);
      currentWord.width += nextCharWidth;
    }
    previousChar = char;
  }
  nextLine();
  if (style.align === "center") {
    alignCenter(layoutData);
  } else if (style.align === "right") {
    alignRight(layoutData);
  } else if (style.align === "justify") {
    alignJustify(layoutData);
  }
  return layoutData;
}
function alignCenter(measurementData) {
  for (let i2 = 0; i2 < measurementData.lines.length; i2++) {
    const line = measurementData.lines[i2];
    const offset = measurementData.width / 2 - line.width / 2;
    for (let j2 = 0; j2 < line.charPositions.length; j2++) {
      line.charPositions[j2] += offset;
    }
  }
}
function alignRight(measurementData) {
  for (let i2 = 0; i2 < measurementData.lines.length; i2++) {
    const line = measurementData.lines[i2];
    const offset = measurementData.width - line.width;
    for (let j2 = 0; j2 < line.charPositions.length; j2++) {
      line.charPositions[j2] += offset;
    }
  }
}
function alignJustify(measurementData) {
  const width = measurementData.width;
  for (let i2 = 0; i2 < measurementData.lines.length; i2++) {
    const line = measurementData.lines[i2];
    let indy = 0;
    let spaceIndex = line.spacesIndex[indy++];
    let offset = 0;
    const totalSpaces = line.spacesIndex.length;
    const newSpaceWidth = (width - line.width) / totalSpaces;
    const spaceWidth = newSpaceWidth;
    for (let j2 = 0; j2 < line.charPositions.length; j2++) {
      if (j2 === spaceIndex) {
        spaceIndex = line.spacesIndex[indy++];
        offset += spaceWidth;
      }
      line.charPositions[j2] += offset;
    }
  }
}
function resolveCharacters(chars) {
  if (chars === "") {
    return [];
  }
  if (typeof chars === "string") {
    chars = [chars];
  }
  const result = [];
  for (let i2 = 0, j2 = chars.length; i2 < j2; i2++) {
    const item = chars[i2];
    if (Array.isArray(item)) {
      if (item.length !== 2) {
        throw new Error(`[BitmapFont]: Invalid character range length, expecting 2 got ${item.length}.`);
      }
      if (item[0].length === 0 || item[1].length === 0) {
        throw new Error("[BitmapFont]: Invalid character delimiter.");
      }
      const startCode = item[0].charCodeAt(0);
      const endCode = item[1].charCodeAt(0);
      if (endCode < startCode) {
        throw new Error("[BitmapFont]: Invalid character range.");
      }
      for (let i22 = startCode, j22 = endCode; i22 <= j22; i22++) {
        result.push(String.fromCharCode(i22));
      }
    } else {
      result.push(...Array.from(item));
    }
  }
  if (result.length === 0) {
    throw new Error("[BitmapFont]: Empty set when resolving characters.");
  }
  return result;
}
let fontCount = 0;
class BitmapFontManagerClass {
  constructor() {
    this.ALPHA = [["a", "z"], ["A", "Z"], " "];
    this.NUMERIC = [["0", "9"]];
    this.ALPHANUMERIC = [["a", "z"], ["A", "Z"], ["0", "9"], " "];
    this.ASCII = [[" ", "~"]];
    this.defaultOptions = {
      chars: this.ALPHANUMERIC,
      resolution: 1,
      padding: 4,
      skipKerning: false,
      textureStyle: null
    };
    this.measureCache = lru(1e3);
  }
  /**
   * Get a font for the specified text and style.
   * @param text - The text to get the font for
   * @param style - The style to use
   */
  getFont(text, style) {
    let fontFamilyKey = `${style.fontFamily}-bitmap`;
    let overrideFill = true;
    if (style._fill.fill && !style._stroke) {
      fontFamilyKey += style._fill.fill.styleKey;
      overrideFill = false;
    } else if (style._stroke || style.dropShadow) {
      fontFamilyKey = `${style.styleKey}-bitmap`;
      overrideFill = false;
    }
    if (!Cache.has(fontFamilyKey)) {
      const styleCopy = Object.create(style);
      styleCopy.lineHeight = 0;
      const fnt = new DynamicBitmapFont({
        style: styleCopy,
        overrideFill,
        overrideSize: true,
        ...this.defaultOptions
      });
      fontCount++;
      if (fontCount > 50) {
        warn("BitmapText", `You have dynamically created ${fontCount} bitmap fonts, this can be inefficient. Try pre installing your font styles using \`BitmapFont.install({name:"style1", style})\``);
      }
      fnt.once("destroy", () => {
        fontCount--;
        Cache.remove(fontFamilyKey);
      });
      Cache.set(
        fontFamilyKey,
        fnt
      );
    }
    const dynamicFont = Cache.get(fontFamilyKey);
    dynamicFont.ensureCharacters?.(text);
    return dynamicFont;
  }
  /**
   * Get the layout of a text for the specified style.
   * @param text - The text to get the layout for
   * @param style - The style to use
   * @param trimEnd - Whether to ignore whitespaces at the end of each line
   */
  getLayout(text, style, trimEnd = true) {
    const bitmapFont = this.getFont(text, style);
    const id = `${text}-${style.styleKey}-${trimEnd}`;
    if (this.measureCache.has(id)) {
      return this.measureCache.get(id);
    }
    const segments = CanvasTextMetrics.graphemeSegmenter(text);
    const layoutData = getBitmapTextLayout(segments, style, bitmapFont, trimEnd);
    this.measureCache.set(id, layoutData);
    return layoutData;
  }
  /**
   * Measure the text using the specified style.
   * @param text - The text to measure
   * @param style - The style to use
   * @param trimEnd - Whether to ignore whitespaces at the end of each line
   */
  measureText(text, style, trimEnd = true) {
    return this.getLayout(text, style, trimEnd);
  }
  // eslint-disable-next-line max-len
  install(...args) {
    let options = args[0];
    if (typeof options === "string") {
      options = {
        name: options,
        style: args[1],
        chars: args[2]?.chars,
        resolution: args[2]?.resolution,
        padding: args[2]?.padding,
        skipKerning: args[2]?.skipKerning
      };
      deprecation(v8_0_0, "BitmapFontManager.install(name, style, options) is deprecated, use BitmapFontManager.install({name, style, ...options})");
    }
    const name = options?.name;
    if (!name) {
      throw new Error("[BitmapFontManager] Property `name` is required.");
    }
    options = { ...this.defaultOptions, ...options };
    const textStyle = options.style;
    const style = textStyle instanceof TextStyle ? textStyle : new TextStyle(textStyle);
    const overrideFill = options.dynamicFill ?? this._canUseTintForStyle(style);
    const font = new DynamicBitmapFont({
      style,
      overrideFill,
      skipKerning: options.skipKerning,
      padding: options.padding,
      resolution: options.resolution,
      overrideSize: false,
      textureStyle: options.textureStyle
    });
    const flatChars = resolveCharacters(options.chars);
    font.ensureCharacters(flatChars.join(""));
    Cache.set(`${name}-bitmap`, font);
    font.once("destroy", () => Cache.remove(`${name}-bitmap`));
    return font;
  }
  /**
   * Uninstalls a bitmap font from the cache.
   * @param {string} name - The name of the bitmap font to uninstall.
   */
  uninstall(name) {
    const cacheKey = `${name}-bitmap`;
    const font = Cache.get(cacheKey);
    if (font) {
      font.destroy();
    }
  }
  /**
   * Determines if a style can use tinting instead of baking colors into the bitmap.
   * Tinting is more efficient as it allows reusing the same bitmap with different colors.
   * @param style - The text style to evaluate
   * @returns true if the style can use tinting, false if colors must be baked in
   * @private
   */
  _canUseTintForStyle(style) {
    return !style._stroke && (!style.dropShadow || style.dropShadow.color === 0) && !style._fill.fill && style._fill.color === 16777215;
  }
}
const BitmapFontManager = new BitmapFontManagerClass();
class BitmapFont extends AbstractBitmapFont {
  constructor(options, url) {
    super();
    const { textures, data } = options;
    Object.keys(data.pages).forEach((key) => {
      const pageData = data.pages[parseInt(key, 10)];
      const texture = textures[pageData.id];
      this.pages.push({ texture });
    });
    Object.keys(data.chars).forEach((key) => {
      const charData = data.chars[key];
      const {
        frame: textureFrame,
        source: textureSource,
        rotate: textureRotate
      } = textures[charData.page];
      const frame = groupD8.transformRectCoords(
        charData,
        textureFrame,
        textureRotate,
        new Rectangle()
      );
      const texture = new Texture({
        frame,
        orig: new Rectangle(0, 0, charData.width, charData.height),
        source: textureSource,
        rotate: textureRotate
      });
      this.chars[key] = {
        id: key.codePointAt(0),
        xOffset: charData.xOffset,
        yOffset: charData.yOffset,
        xAdvance: charData.xAdvance,
        kerning: charData.kerning ?? {},
        texture
      };
    });
    this.baseRenderedFontSize = data.fontSize;
    this.baseMeasurementFontSize = data.fontSize;
    this.fontMetrics = {
      ascent: 0,
      descent: 0,
      fontSize: data.fontSize
    };
    this.baseLineOffset = data.baseLineOffset;
    this.lineHeight = data.lineHeight;
    this.fontFamily = data.fontFamily;
    this.distanceField = data.distanceField ?? {
      type: "none",
      range: 0
    };
    this.url = url;
  }
  /** Destroys the BitmapFont object. */
  destroy() {
    super.destroy();
    for (let i2 = 0; i2 < this.pages.length; i2++) {
      const { texture } = this.pages[i2];
      texture.destroy(true);
    }
    this.pages = null;
  }
  /**
   * Generates and installs a bitmap font with the specified options.
   * The font will be cached and available for use in BitmapText objects.
   * @param options - Setup options for font generation
   * @returns Installed font instance
   * @example
   * ```ts
   * // Install a basic font
   * BitmapFont.install({
   *     name: 'Title',
   *     style: {
   *         fontFamily: 'Arial',
   *         fontSize: 32,
   *         fill: '#ffffff'
   *     }
   * });
   *
   * // Install with advanced options
   * BitmapFont.install({
   *     name: 'Custom',
   *     style: {
   *         fontFamily: 'Arial',
   *         fontSize: 24,
   *         fill: '#00ff00',
   *         stroke: { color: '#000000', width: 2 }
   *     },
   *     chars: [['a', 'z'], ['A', 'Z'], ['0', '9']],
   *     resolution: 2,
   *     padding: 4,
   *     textureStyle: {
   *         scaleMode: 'nearest'
   *     }
   * });
   * ```
   */
  static install(options) {
    BitmapFontManager.install(options);
  }
  /**
   * Uninstalls a bitmap font from the cache.
   * This frees up memory and resources associated with the font.
   * @param name - The name of the bitmap font to uninstall
   * @example
   * ```ts
   * // Remove a font when it's no longer needed
   * BitmapFont.uninstall('MyCustomFont');
   *
   * // Clear multiple fonts
   * ['Title', 'Heading', 'Body'].forEach(BitmapFont.uninstall);
   * ```
   */
  static uninstall(name) {
    BitmapFontManager.uninstall(name);
  }
}
const bitmapFontTextParser = {
  test(data) {
    return typeof data === "string" && data.startsWith("info face=");
  },
  parse(txt) {
    const items = txt.match(/^[a-z]+\s+.+$/gm);
    const rawData = {
      info: [],
      common: [],
      page: [],
      char: [],
      chars: [],
      kerning: [],
      kernings: [],
      distanceField: []
    };
    for (const i2 in items) {
      const name = items[i2].match(/^[a-z]+/gm)[0];
      const attributeList = items[i2].match(/[a-zA-Z]+=([^\s"']+|"([^"]*)")/gm);
      const itemData = {};
      for (const i22 in attributeList) {
        const split = attributeList[i22].split("=");
        const key = split[0];
        const strValue = split[1].replace(/"/gm, "");
        const floatValue = parseFloat(strValue);
        const value = isNaN(floatValue) ? strValue : floatValue;
        itemData[key] = value;
      }
      rawData[name].push(itemData);
    }
    const font = {
      chars: {},
      pages: [],
      lineHeight: 0,
      fontSize: 0,
      fontFamily: "",
      distanceField: null,
      baseLineOffset: 0
    };
    const [info] = rawData.info;
    const [common] = rawData.common;
    const [distanceField] = rawData.distanceField ?? [];
    if (distanceField) {
      font.distanceField = {
        range: parseInt(distanceField.distanceRange, 10),
        type: distanceField.fieldType
      };
    }
    font.fontSize = parseInt(info.size, 10);
    font.fontFamily = info.face;
    font.lineHeight = parseInt(common.lineHeight, 10);
    const page = rawData.page;
    for (let i2 = 0; i2 < page.length; i2++) {
      font.pages.push({
        id: parseInt(page[i2].id, 10) || 0,
        file: page[i2].file
      });
    }
    const map = {};
    font.baseLineOffset = font.lineHeight - parseInt(common.base, 10);
    const char = rawData.char;
    for (let i2 = 0; i2 < char.length; i2++) {
      const charNode = char[i2];
      const id = parseInt(charNode.id, 10);
      let letter = charNode.letter ?? charNode.char ?? String.fromCharCode(id);
      if (letter === "space")
        letter = " ";
      map[id] = letter;
      font.chars[letter] = {
        id,
        // texture deets..
        page: parseInt(charNode.page, 10) || 0,
        x: parseInt(charNode.x, 10),
        y: parseInt(charNode.y, 10),
        width: parseInt(charNode.width, 10),
        height: parseInt(charNode.height, 10),
        xOffset: parseInt(charNode.xoffset, 10),
        yOffset: parseInt(charNode.yoffset, 10),
        xAdvance: parseInt(charNode.xadvance, 10),
        kerning: {}
      };
    }
    const kerning = rawData.kerning || [];
    for (let i2 = 0; i2 < kerning.length; i2++) {
      const first = parseInt(kerning[i2].first, 10);
      const second = parseInt(kerning[i2].second, 10);
      const amount = parseInt(kerning[i2].amount, 10);
      font.chars[map[second]].kerning[map[first]] = amount;
    }
    return font;
  }
};
const bitmapFontXMLParser = {
  test(data) {
    const xml = data;
    return typeof xml !== "string" && "getElementsByTagName" in xml && xml.getElementsByTagName("page").length && xml.getElementsByTagName("info")[0].getAttribute("face") !== null;
  },
  parse(xml) {
    const data = {
      chars: {},
      pages: [],
      lineHeight: 0,
      fontSize: 0,
      fontFamily: "",
      distanceField: null,
      baseLineOffset: 0
    };
    const info = xml.getElementsByTagName("info")[0];
    const common = xml.getElementsByTagName("common")[0];
    const distanceField = xml.getElementsByTagName("distanceField")[0];
    if (distanceField) {
      data.distanceField = {
        type: distanceField.getAttribute("fieldType"),
        range: parseInt(distanceField.getAttribute("distanceRange"), 10)
      };
    }
    const page = xml.getElementsByTagName("page");
    const char = xml.getElementsByTagName("char");
    const kerning = xml.getElementsByTagName("kerning");
    data.fontSize = parseInt(info.getAttribute("size"), 10);
    data.fontFamily = info.getAttribute("face");
    data.lineHeight = parseInt(common.getAttribute("lineHeight"), 10);
    for (let i2 = 0; i2 < page.length; i2++) {
      data.pages.push({
        id: parseInt(page[i2].getAttribute("id"), 10) || 0,
        file: page[i2].getAttribute("file")
      });
    }
    const map = {};
    data.baseLineOffset = data.lineHeight - parseInt(common.getAttribute("base"), 10);
    for (let i2 = 0; i2 < char.length; i2++) {
      const charNode = char[i2];
      const id = parseInt(charNode.getAttribute("id"), 10);
      let letter = charNode.getAttribute("letter") ?? charNode.getAttribute("char") ?? String.fromCharCode(id);
      if (letter === "space")
        letter = " ";
      map[id] = letter;
      data.chars[letter] = {
        id,
        // texture deets..
        page: parseInt(charNode.getAttribute("page"), 10) || 0,
        x: parseInt(charNode.getAttribute("x"), 10),
        y: parseInt(charNode.getAttribute("y"), 10),
        width: parseInt(charNode.getAttribute("width"), 10),
        height: parseInt(charNode.getAttribute("height"), 10),
        // render deets..
        xOffset: parseInt(charNode.getAttribute("xoffset"), 10),
        yOffset: parseInt(charNode.getAttribute("yoffset"), 10),
        // + baseLineOffset,
        xAdvance: parseInt(charNode.getAttribute("xadvance"), 10),
        kerning: {}
      };
    }
    for (let i2 = 0; i2 < kerning.length; i2++) {
      const first = parseInt(kerning[i2].getAttribute("first"), 10);
      const second = parseInt(kerning[i2].getAttribute("second"), 10);
      const amount = parseInt(kerning[i2].getAttribute("amount"), 10);
      data.chars[map[second]].kerning[map[first]] = amount;
    }
    return data;
  }
};
const bitmapFontXMLStringParser = {
  test(data) {
    if (typeof data === "string" && data.match(/<font(\s|>)/)) {
      return bitmapFontXMLParser.test(DOMAdapter.get().parseXML(data));
    }
    return false;
  },
  parse(data) {
    return bitmapFontXMLParser.parse(DOMAdapter.get().parseXML(data));
  }
};
const validExtensions = [".xml", ".fnt"];
const bitmapFontCachePlugin = {
  extension: {
    type: ExtensionType.CacheParser,
    name: "cacheBitmapFont"
  },
  test: (asset) => asset instanceof BitmapFont,
  getCacheableAssets(keys, asset) {
    const out2 = {};
    keys.forEach((key) => {
      out2[key] = asset;
      out2[`${key}-bitmap`] = asset;
    });
    out2[`${asset.fontFamily}-bitmap`] = asset;
    return out2;
  }
};
const loadBitmapFont = {
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.Normal
  },
  /** used for deprecation purposes */
  name: "loadBitmapFont",
  id: "bitmap-font",
  test(url) {
    return validExtensions.includes(path.extname(url).toLowerCase());
  },
  async testParse(data) {
    return bitmapFontTextParser.test(data) || bitmapFontXMLStringParser.test(data);
  },
  async parse(asset, data, loader) {
    const bitmapFontData = bitmapFontTextParser.test(asset) ? bitmapFontTextParser.parse(asset) : bitmapFontXMLStringParser.parse(asset);
    const { src } = data;
    const { pages } = bitmapFontData;
    const textureUrls = [];
    const textureOptions = bitmapFontData.distanceField ? {
      scaleMode: "linear",
      alphaMode: "premultiply-alpha-on-upload",
      autoGenerateMipmaps: false,
      resolution: 1
    } : {};
    for (let i2 = 0; i2 < pages.length; ++i2) {
      const pageFile = pages[i2].file;
      let imagePath = path.join(path.dirname(src), pageFile);
      imagePath = copySearchParams(imagePath, src);
      textureUrls.push({
        src: imagePath,
        data: textureOptions
      });
    }
    const loadedTextures = await loader.load(textureUrls);
    const textures = textureUrls.map((url) => loadedTextures[url.src]);
    const bitmapFont = new BitmapFont({
      data: bitmapFontData,
      textures
    }, src);
    return bitmapFont;
  },
  async load(url, _options) {
    const response = await DOMAdapter.get().fetch(url);
    return await response.text();
  },
  async unload(bitmapFont, _resolvedAsset, loader) {
    await Promise.all(bitmapFont.pages.map((page) => loader.unload(page.texture.source._sourceOrigin)));
    bitmapFont.destroy();
  }
};
class BackgroundLoader {
  /**
   * @param loader
   * @param verbose - should the loader log to the console
   */
  constructor(loader, verbose = false) {
    this._loader = loader;
    this._assetList = [];
    this._isLoading = false;
    this._maxConcurrent = 1;
    this.verbose = verbose;
  }
  /**
   * Adds assets to the background loading queue. Assets are loaded one at a time to minimize
   * performance impact.
   * @param assetUrls - Array of resolved assets to load in the background
   * @example
   * ```ts
   * // Add assets to background load queue
   * backgroundLoader.add([
   *     { src: 'images/level1/bg.png' },
   *     { src: 'images/level1/characters.json' }
   * ]);
   *
   * // Assets will load sequentially in the background
   * // The loader automatically pauses when high-priority loads occur
   * // e.g. Assets.load() is called
   * ```
   * @remarks
   * - Assets are loaded one at a time to minimize performance impact
   * - Loading automatically pauses when Assets.load() is called
   * - No progress tracking is available for background loading
   * - Assets are cached as they complete loading
   * @internal
   */
  add(assetUrls) {
    assetUrls.forEach((a2) => {
      this._assetList.push(a2);
    });
    if (this.verbose) {
      console.log("[BackgroundLoader] assets: ", this._assetList);
    }
    if (this._isActive && !this._isLoading) {
      void this._next();
    }
  }
  /**
   * Loads the next set of assets. Will try to load as many assets as it can at the same time.
   *
   * The max assets it will try to load at one time will be 4.
   */
  async _next() {
    if (this._assetList.length && this._isActive) {
      this._isLoading = true;
      const toLoad = [];
      const toLoadAmount = Math.min(this._assetList.length, this._maxConcurrent);
      for (let i2 = 0; i2 < toLoadAmount; i2++) {
        toLoad.push(this._assetList.pop());
      }
      await this._loader.load(toLoad);
      this._isLoading = false;
      void this._next();
    }
  }
  /**
   * Controls the active state of the background loader. When active, the loader will
   * continue processing its queue. When inactive, loading is paused.
   * @returns Whether the background loader is currently active
   * @example
   * ```ts
   * // Pause background loading
   * backgroundLoader.active = false;
   *
   * // Resume background loading
   * backgroundLoader.active = true;
   *
   * // Check current state
   * console.log(backgroundLoader.active); // true/false
   *
   * // Common use case: Pause during intensive operations
   * backgroundLoader.active = false;  // Pause background loading
   * ... // Perform high-priority tasks
   * backgroundLoader.active = true;   // Resume background loading
   * ```
   * @remarks
   * - Setting to true resumes loading immediately
   * - Setting to false pauses after current asset completes
   * - Background loading is automatically paused during `Assets.load()`
   * - Assets already being loaded will complete even when set to false
   */
  get active() {
    return this._isActive;
  }
  set active(value) {
    if (this._isActive === value)
      return;
    this._isActive = value;
    if (value && !this._isLoading) {
      void this._next();
    }
  }
}
const cacheTextureArray = {
  extension: {
    type: ExtensionType.CacheParser,
    name: "cacheTextureArray"
  },
  test: (asset) => Array.isArray(asset) && asset.every((t2) => t2 instanceof Texture),
  getCacheableAssets: (keys, asset) => {
    const out2 = {};
    keys.forEach((key) => {
      asset.forEach((item, i2) => {
        out2[key + (i2 === 0 ? "" : i2 + 1)] = item;
      });
    });
    return out2;
  }
};
async function testImageFormat(imageData) {
  if ("Image" in globalThis) {
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        resolve(true);
      };
      image.onerror = () => {
        resolve(false);
      };
      image.src = imageData;
    });
  }
  if ("createImageBitmap" in globalThis && "fetch" in globalThis) {
    try {
      const blob = await (await fetch(imageData)).blob();
      await createImageBitmap(blob);
    } catch (_e) {
      return false;
    }
    return true;
  }
  return false;
}
const detectAvif = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: 1
  },
  test: async () => testImageFormat(
    // eslint-disable-next-line max-len
    "data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgANogQEAwgMg8f8D///8WfhwB8+ErK42A="
  ),
  add: async (formats) => [...formats, "avif"],
  remove: async (formats) => formats.filter((f2) => f2 !== "avif")
};
const imageFormats = ["png", "jpg", "jpeg"];
const detectDefaults = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: -1
  },
  test: () => Promise.resolve(true),
  add: async (formats) => [...formats, ...imageFormats],
  remove: async (formats) => formats.filter((f2) => !imageFormats.includes(f2))
};
const inWorker = "WorkerGlobalScope" in globalThis && globalThis instanceof globalThis.WorkerGlobalScope;
function testVideoFormat(mimeType) {
  if (inWorker) {
    return false;
  }
  const video = document.createElement("video");
  return video.canPlayType(mimeType) !== "";
}
const detectMp4 = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: 0
  },
  test: async () => testVideoFormat("video/mp4"),
  add: async (formats) => [...formats, "mp4", "m4v"],
  remove: async (formats) => formats.filter((f2) => f2 !== "mp4" && f2 !== "m4v")
};
const detectOgv = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: 0
  },
  test: async () => testVideoFormat("video/ogg"),
  add: async (formats) => [...formats, "ogv"],
  remove: async (formats) => formats.filter((f2) => f2 !== "ogv")
};
const detectWebm = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: 0
  },
  test: async () => testVideoFormat("video/webm"),
  add: async (formats) => [...formats, "webm"],
  remove: async (formats) => formats.filter((f2) => f2 !== "webm")
};
const detectWebp = {
  extension: {
    type: ExtensionType.DetectionParser,
    priority: 0
  },
  test: async () => testImageFormat(
    "data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA="
  ),
  add: async (formats) => [...formats, "webp"],
  remove: async (formats) => formats.filter((f2) => f2 !== "webp")
};
const _Loader = class _Loader2 {
  constructor() {
    this.loadOptions = { ..._Loader2.defaultOptions };
    this._parsers = [];
    this._parsersValidated = false;
    this.parsers = new Proxy(this._parsers, {
      set: (target, key, value) => {
        this._parsersValidated = false;
        target[key] = value;
        return true;
      }
    });
    this.promiseCache = {};
  }
  /** function used for testing */
  reset() {
    this._parsersValidated = false;
    this.promiseCache = {};
  }
  /**
   * Used internally to generate a promise for the asset to be loaded.
   * @param url - The URL to be loaded
   * @param data - any custom additional information relevant to the asset being loaded
   * @returns - a promise that will resolve to an Asset for example a Texture of a JSON object
   */
  _getLoadPromiseAndParser(url, data) {
    const result = {
      promise: null,
      parser: null
    };
    result.promise = (async () => {
      let asset = null;
      let parser = null;
      if (data.parser || data.loadParser) {
        parser = this._parserHash[data.parser || data.loadParser];
        if (data.loadParser) {
          warn(
            `[Assets] "loadParser" is deprecated, use "parser" instead for ${url}`
          );
        }
        if (!parser) {
          warn(
            `[Assets] specified load parser "${data.parser || data.loadParser}" not found while loading ${url}`
          );
        }
      }
      if (!parser) {
        for (let i2 = 0; i2 < this.parsers.length; i2++) {
          const parserX = this.parsers[i2];
          if (parserX.load && parserX.test?.(url, data, this)) {
            parser = parserX;
            break;
          }
        }
        if (!parser) {
          warn(`[Assets] ${url} could not be loaded as we don't know how to parse it, ensure the correct parser has been added`);
          return null;
        }
      }
      asset = await parser.load(url, data, this);
      result.parser = parser;
      for (let i2 = 0; i2 < this.parsers.length; i2++) {
        const parser2 = this.parsers[i2];
        if (parser2.parse) {
          if (parser2.parse && await parser2.testParse?.(asset, data, this)) {
            asset = await parser2.parse(asset, data, this) || asset;
            result.parser = parser2;
          }
        }
      }
      return asset;
    })();
    return result;
  }
  async load(assetsToLoadIn, onProgressOrOptions) {
    if (!this._parsersValidated) {
      this._validateParsers();
    }
    const options = typeof onProgressOrOptions === "function" ? { ..._Loader2.defaultOptions, ...this.loadOptions, onProgress: onProgressOrOptions } : { ..._Loader2.defaultOptions, ...this.loadOptions, ...onProgressOrOptions || {} };
    const { onProgress, onError, strategy, retryCount, retryDelay } = options;
    let count2 = 0;
    const assets = {};
    const singleAsset = isSingleItem(assetsToLoadIn);
    const assetsToLoad = convertToList(assetsToLoadIn, (item) => ({
      alias: [item],
      src: item,
      data: {}
    }));
    const total = assetsToLoad.reduce((sum, asset) => sum + (asset.progressSize || 1), 0);
    const promises = assetsToLoad.map(async (asset) => {
      const url = path.toAbsolute(asset.src);
      if (assets[asset.src])
        return;
      await this._loadAssetWithRetry(url, asset, { onProgress, onError, strategy, retryCount, retryDelay }, assets);
      count2 += asset.progressSize || 1;
      if (onProgress)
        onProgress(count2 / total);
    });
    await Promise.all(promises);
    return singleAsset ? assets[assetsToLoad[0].src] : assets;
  }
  /**
   * Unloads one or more assets. Any unloaded assets will be destroyed, freeing up memory for your app.
   * The parser that created the asset, will be the one that unloads it.
   * @example
   * // Single asset:
   * const asset = await Loader.load('cool.png');
   *
   * await Loader.unload('cool.png');
   *
   * console.log(asset.destroyed); // true
   * @param assetsToUnloadIn - urls that you want to unload, or a single one!
   */
  async unload(assetsToUnloadIn) {
    const assetsToUnload = convertToList(assetsToUnloadIn, (item) => ({
      alias: [item],
      src: item
    }));
    const promises = assetsToUnload.map(async (asset) => {
      const url = path.toAbsolute(asset.src);
      const loadPromise = this.promiseCache[url];
      if (loadPromise) {
        const loadedAsset = await loadPromise.promise;
        delete this.promiseCache[url];
        await loadPromise.parser?.unload?.(loadedAsset, asset, this);
      }
    });
    await Promise.all(promises);
  }
  /** validates our parsers, right now it only checks for name conflicts but we can add more here as required! */
  _validateParsers() {
    this._parsersValidated = true;
    this._parserHash = this._parsers.filter((parser) => parser.name || parser.id).reduce((hash, parser) => {
      if (!parser.name && !parser.id) {
        warn(`[Assets] parser should have an id`);
      } else if (hash[parser.name] || hash[parser.id]) {
        warn(`[Assets] parser id conflict "${parser.id}"`);
      }
      hash[parser.name] = parser;
      if (parser.id)
        hash[parser.id] = parser;
      return hash;
    }, {});
  }
  async _loadAssetWithRetry(url, asset, options, assets) {
    let attempt = 0;
    const { onError, strategy, retryCount, retryDelay } = options;
    const wait = (ms) => new Promise((r2) => setTimeout(r2, ms));
    while (true) {
      try {
        if (!this.promiseCache[url]) {
          this.promiseCache[url] = this._getLoadPromiseAndParser(url, asset);
        }
        assets[asset.src] = await this.promiseCache[url].promise;
        return;
      } catch (e2) {
        delete this.promiseCache[url];
        delete assets[asset.src];
        attempt++;
        const isLast = strategy !== "retry" || attempt > retryCount;
        if (strategy === "retry" && !isLast) {
          if (onError)
            onError(e2, asset);
          await wait(retryDelay);
          continue;
        }
        if (strategy === "skip") {
          if (onError)
            onError(e2, asset);
          return;
        }
        if (onError)
          onError(e2, asset);
        const error = new Error(`[Loader.load] Failed to load ${url}.
${e2}`);
        if (e2 instanceof Error && e2.stack) {
          error.stack = e2.stack;
        }
        throw error;
      }
    }
  }
};
_Loader.defaultOptions = {
  onProgress: void 0,
  onError: void 0,
  strategy: "throw",
  retryCount: 3,
  retryDelay: 250
};
let Loader = _Loader;
function checkDataUrl(url, mimes) {
  if (Array.isArray(mimes)) {
    for (const mime of mimes) {
      if (url.startsWith(`data:${mime}`))
        return true;
    }
    return false;
  }
  return url.startsWith(`data:${mimes}`);
}
function checkExtension(url, extension) {
  const tempURL = url.split("?")[0];
  const ext = path.extname(tempURL).toLowerCase();
  if (Array.isArray(extension)) {
    return extension.includes(ext);
  }
  return ext === extension;
}
const validJSONExtension = ".json";
const validJSONMIME = "application/json";
const loadJson = {
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.Low
  },
  /** used for deprecation purposes */
  name: "loadJson",
  id: "json",
  test(url) {
    return checkDataUrl(url, validJSONMIME) || checkExtension(url, validJSONExtension);
  },
  async load(url) {
    const response = await DOMAdapter.get().fetch(url);
    const json = await response.json();
    return json;
  }
};
const validTXTExtension = ".txt";
const validTXTMIME = "text/plain";
const loadTxt = {
  /** used for deprecation purposes */
  name: "loadTxt",
  id: "text",
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.Low,
    name: "loadTxt"
  },
  test(url) {
    return checkDataUrl(url, validTXTMIME) || checkExtension(url, validTXTExtension);
  },
  async load(url) {
    const response = await DOMAdapter.get().fetch(url);
    const txt = await response.text();
    return txt;
  }
};
const validWeights = [
  "normal",
  "bold",
  "100",
  "200",
  "300",
  "400",
  "500",
  "600",
  "700",
  "800",
  "900"
];
const validFontExtensions = [".ttf", ".otf", ".woff", ".woff2"];
const validFontMIMEs = [
  "font/ttf",
  "font/otf",
  "font/woff",
  "font/woff2"
];
const CSS_IDENT_TOKEN_REGEX = /^(--|-?[A-Z_])[0-9A-Z_-]*$/i;
function getFontFamilyName(url) {
  const ext = path.extname(url);
  const name = path.basename(url, ext);
  const nameWithSpaces = name.replace(/(-|_)/g, " ");
  const nameTokens = nameWithSpaces.toLowerCase().split(" ").map((word) => word.charAt(0).toUpperCase() + word.slice(1));
  let valid = nameTokens.length > 0;
  for (const token of nameTokens) {
    if (!token.match(CSS_IDENT_TOKEN_REGEX)) {
      valid = false;
      break;
    }
  }
  let fontFamilyName = nameTokens.join(" ");
  if (!valid) {
    fontFamilyName = `"${fontFamilyName.replace(/[\\"]/g, "\\$&")}"`;
  }
  return fontFamilyName;
}
const validURICharactersRegex = /^[0-9A-Za-z%:/?#\[\]@!\$&'()\*\+,;=\-._~]*$/;
function encodeURIWhenNeeded(uri) {
  if (validURICharactersRegex.test(uri)) {
    return uri;
  }
  return encodeURI(uri);
}
const loadWebFont = {
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.Low
  },
  /** used for deprecation purposes */
  name: "loadWebFont",
  id: "web-font",
  test(url) {
    return checkDataUrl(url, validFontMIMEs) || checkExtension(url, validFontExtensions);
  },
  async load(url, options) {
    const fonts = DOMAdapter.get().getFontFaceSet();
    if (fonts) {
      const fontFaces = [];
      const name = options.data?.family ?? getFontFamilyName(url);
      const weights = options.data?.weights?.filter((weight) => validWeights.includes(weight)) ?? ["normal"];
      const data = options.data ?? {};
      for (let i2 = 0; i2 < weights.length; i2++) {
        const weight = weights[i2];
        const font = new FontFace(name, `url(${encodeURIWhenNeeded(url)})`, {
          ...data,
          weight
        });
        await font.load();
        fonts.add(font);
        fontFaces.push(font);
      }
      if (Cache.has(`${name}-and-url`)) {
        const cached = Cache.get(`${name}-and-url`);
        cached.entries.push({ url, faces: fontFaces });
      } else {
        Cache.set(`${name}-and-url`, {
          entries: [{ url, faces: fontFaces }]
        });
      }
      return fontFaces.length === 1 ? fontFaces[0] : fontFaces;
    }
    warn("[loadWebFont] FontFace API is not supported. Skipping loading font");
    return null;
  },
  unload(font) {
    const fonts = Array.isArray(font) ? font : [font];
    const fontFamily = fonts[0].family;
    const cached = Cache.get(`${fontFamily}-and-url`);
    const entry = cached.entries.find((f2) => f2.faces.some((t2) => fonts.indexOf(t2) !== -1));
    entry.faces = entry.faces.filter((f2) => fonts.indexOf(f2) === -1);
    if (entry.faces.length === 0) {
      cached.entries = cached.entries.filter((f2) => f2 !== entry);
    }
    fonts.forEach((t2) => {
      DOMAdapter.get().getFontFaceSet().delete(t2);
    });
    if (cached.entries.length === 0) {
      Cache.remove(`${fontFamily}-and-url`);
    }
  }
};
function getResolutionOfUrl(url, defaultValue = 1) {
  const resolution = Resolver.RETINA_PREFIX?.exec(url);
  if (resolution) {
    return parseFloat(resolution[1]);
  }
  return defaultValue;
}
function createTexture(source2, loader, url) {
  source2.label = url;
  source2._sourceOrigin = url;
  const texture = new Texture({
    source: source2,
    label: url
  });
  const unload = () => {
    delete loader.promiseCache[url];
    if (Cache.has(url)) {
      Cache.remove(url);
    }
  };
  texture.source.once("destroy", () => {
    if (loader.promiseCache[url]) {
      warn("[Assets] A TextureSource managed by Assets was destroyed instead of unloaded! Use Assets.unload() instead of destroying the TextureSource.");
      unload();
    }
  });
  texture.once("destroy", () => {
    if (!source2.destroyed) {
      warn("[Assets] A Texture managed by Assets was destroyed instead of unloaded! Use Assets.unload() instead of destroying the Texture.");
      unload();
    }
  });
  return texture;
}
const validSVGExtension = ".svg";
const validSVGMIME = "image/svg+xml";
const loadSvg = {
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.Low,
    name: "loadSVG"
  },
  /** used for deprecation purposes */
  name: "loadSVG",
  id: "svg",
  config: {
    crossOrigin: "anonymous",
    parseAsGraphicsContext: false
  },
  test(url) {
    return checkDataUrl(url, validSVGMIME) || checkExtension(url, validSVGExtension);
  },
  async load(url, asset, loader) {
    if (asset.data?.parseAsGraphicsContext ?? this.config.parseAsGraphicsContext) {
      return loadAsGraphics(url);
    }
    return loadAsTexture(url, asset, loader, this.config.crossOrigin);
  },
  unload(asset) {
    asset.destroy(true);
  }
};
async function loadAsTexture(url, asset, loader, crossOrigin2) {
  const response = await DOMAdapter.get().fetch(url);
  const image = DOMAdapter.get().createImage();
  image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(await response.text())}`;
  image.crossOrigin = crossOrigin2;
  await image.decode();
  const width = asset.data?.width ?? image.width;
  const height = asset.data?.height ?? image.height;
  const resolution = asset.data?.resolution || getResolutionOfUrl(url);
  const canvasWidth = Math.ceil(width * resolution);
  const canvasHeight = Math.ceil(height * resolution);
  const canvas = DOMAdapter.get().createCanvas(canvasWidth, canvasHeight);
  const context2 = canvas.getContext("2d");
  context2.imageSmoothingEnabled = true;
  context2.imageSmoothingQuality = "high";
  context2.drawImage(image, 0, 0, width * resolution, height * resolution);
  const { parseAsGraphicsContext: _p, ...rest } = asset.data ?? {};
  const base = new ImageSource({
    resource: canvas,
    alphaMode: "premultiply-alpha-on-upload",
    resolution,
    ...rest
  });
  return createTexture(base, loader, url);
}
async function loadAsGraphics(url) {
  const response = await DOMAdapter.get().fetch(url);
  const svgSource = await response.text();
  const context2 = new GraphicsContext();
  context2.svg(svgSource);
  return context2;
}
const WORKER_CODE$1 = `(function () {
    'use strict';

    const WHITE_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=";
    async function checkImageBitmap() {
      try {
        if (typeof createImageBitmap !== "function")
          return false;
        const response = await fetch(WHITE_PNG);
        const imageBlob = await response.blob();
        const imageBitmap = await createImageBitmap(imageBlob);
        return imageBitmap.width === 1 && imageBitmap.height === 1;
      } catch (_e) {
        return false;
      }
    }
    void checkImageBitmap().then((result) => {
      self.postMessage(result);
    });

})();
`;
let WORKER_URL$1 = null;
let WorkerInstance$1 = class WorkerInstance {
  constructor() {
    if (!WORKER_URL$1) {
      WORKER_URL$1 = URL.createObjectURL(new Blob([WORKER_CODE$1], { type: "application/javascript" }));
    }
    this.worker = new Worker(WORKER_URL$1);
  }
};
WorkerInstance$1.revokeObjectURL = function revokeObjectURL() {
  if (WORKER_URL$1) {
    URL.revokeObjectURL(WORKER_URL$1);
    WORKER_URL$1 = null;
  }
};
const WORKER_CODE = '(function () {\n    \'use strict\';\n\n    async function loadImageBitmap(url, alphaMode) {\n      const response = await fetch(url);\n      if (!response.ok) {\n        throw new Error(`[WorkerManager.loadImageBitmap] Failed to fetch ${url}: ${response.status} ${response.statusText}`);\n      }\n      const imageBlob = await response.blob();\n      return alphaMode === "premultiplied-alpha" ? createImageBitmap(imageBlob, { premultiplyAlpha: "none" }) : createImageBitmap(imageBlob);\n    }\n    self.onmessage = async (event) => {\n      try {\n        const imageBitmap = await loadImageBitmap(event.data.data[0], event.data.data[1]);\n        self.postMessage({\n          data: imageBitmap,\n          uuid: event.data.uuid,\n          id: event.data.id\n        }, [imageBitmap]);\n      } catch (e) {\n        self.postMessage({\n          error: e,\n          uuid: event.data.uuid,\n          id: event.data.id\n        });\n      }\n    };\n\n})();\n';
let WORKER_URL = null;
class WorkerInstance2 {
  constructor() {
    if (!WORKER_URL) {
      WORKER_URL = URL.createObjectURL(new Blob([WORKER_CODE], { type: "application/javascript" }));
    }
    this.worker = new Worker(WORKER_URL);
  }
}
WorkerInstance2.revokeObjectURL = function revokeObjectURL2() {
  if (WORKER_URL) {
    URL.revokeObjectURL(WORKER_URL);
    WORKER_URL = null;
  }
};
let UUID = 0;
let MAX_WORKERS;
class WorkerManagerClass {
  constructor() {
    this._initialized = false;
    this._createdWorkers = 0;
    this._workerPool = [];
    this._queue = [];
    this._resolveHash = {};
  }
  /**
   * Checks if ImageBitmap is supported in the current environment.
   *
   * This method uses a dedicated worker to test ImageBitmap support
   * and caches the result for subsequent calls.
   * @returns Promise that resolves to true if ImageBitmap is supported, false otherwise
   */
  isImageBitmapSupported() {
    if (this._isImageBitmapSupported !== void 0)
      return this._isImageBitmapSupported;
    this._isImageBitmapSupported = new Promise((resolve) => {
      const { worker } = new WorkerInstance$1();
      worker.addEventListener("message", (event) => {
        worker.terminate();
        WorkerInstance$1.revokeObjectURL();
        resolve(event.data);
      });
    });
    return this._isImageBitmapSupported;
  }
  /**
   * Loads an image as an ImageBitmap using a web worker.
   * @param src - The source URL or path of the image to load
   * @param asset - Optional resolved asset containing additional texture source options
   * @returns Promise that resolves to the loaded ImageBitmap
   * @example
   * ```typescript
   * const bitmap = await WorkerManager.loadImageBitmap('image.png');
   * const bitmapWithOptions = await WorkerManager.loadImageBitmap('image.png', asset);
   * ```
   */
  loadImageBitmap(src, asset) {
    return this._run("loadImageBitmap", [src, asset?.data?.alphaMode]);
  }
  /**
   * Initializes the worker pool if not already initialized.
   * Currently a no-op but reserved for future initialization logic.
   */
  async _initWorkers() {
    if (this._initialized)
      return;
    this._initialized = true;
  }
  /**
   * Gets an available worker from the pool or creates a new one if needed.
   *
   * Workers are created up to the MAX_WORKERS limit (based on navigator.hardwareConcurrency).
   * Each worker is configured with a message handler for processing results.
   * @returns Available worker or undefined if pool is at capacity and no workers are free
   */
  _getWorker() {
    if (MAX_WORKERS === void 0) {
      MAX_WORKERS = navigator.hardwareConcurrency || 4;
    }
    let worker = this._workerPool.pop();
    if (!worker && this._createdWorkers < MAX_WORKERS) {
      this._createdWorkers++;
      worker = new WorkerInstance2().worker;
      worker.addEventListener("message", (event) => {
        this._complete(event.data);
        this._returnWorker(event.target);
        this._next();
      });
    }
    return worker;
  }
  /**
   * Returns a worker to the pool after completing a task.
   * @param worker - The worker to return to the pool
   */
  _returnWorker(worker) {
    this._workerPool.push(worker);
  }
  /**
   * Handles completion of a worker task by resolving or rejecting the corresponding promise.
   * @param data - Result data from the worker containing uuid, data, and optional error
   */
  _complete(data) {
    if (!this._resolveHash[data.uuid]) {
      return;
    }
    if (data.error !== void 0) {
      this._resolveHash[data.uuid].reject(data.error);
    } else {
      this._resolveHash[data.uuid].resolve(data.data);
    }
    delete this._resolveHash[data.uuid];
  }
  /**
   * Executes a task using the worker pool system.
   *
   * Queues the task and processes it when a worker becomes available.
   * @param id - Identifier for the type of task to run
   * @param args - Arguments to pass to the worker
   * @returns Promise that resolves with the worker's result
   */
  async _run(id, args) {
    await this._initWorkers();
    const promise2 = new Promise((resolve, reject) => {
      this._queue.push({ id, arguments: args, resolve, reject });
    });
    this._next();
    return promise2;
  }
  /**
   * Processes the next item in the queue if workers are available.
   *
   * This method is called after worker initialization and when workers
   * complete tasks to continue processing the queue.
   */
  _next() {
    if (!this._queue.length)
      return;
    const worker = this._getWorker();
    if (!worker) {
      return;
    }
    const toDo = this._queue.pop();
    const id = toDo.id;
    this._resolveHash[UUID] = { resolve: toDo.resolve, reject: toDo.reject };
    worker.postMessage({
      data: toDo.arguments,
      uuid: UUID++,
      id
    });
  }
  /**
   * Resets the worker manager, terminating all workers and clearing the queue.
   *
   * This method:
   * - Terminates all active workers
   * - Rejects all pending promises with an error
   * - Clears all internal state
   * - Resets initialization flags
   *
   * This should be called when the worker manager is no longer needed
   * to prevent memory leaks and ensure proper cleanup.
   * @example
   * ```typescript
   * // Clean up when shutting down
   * WorkerManager.reset();
   * ```
   */
  reset() {
    this._workerPool.forEach((worker) => worker.terminate());
    this._workerPool.length = 0;
    Object.values(this._resolveHash).forEach(({ reject }) => {
      reject?.(new Error("WorkerManager has been reset before completion"));
    });
    this._resolveHash = {};
    this._queue.length = 0;
    this._initialized = false;
    this._createdWorkers = 0;
  }
}
const WorkerManager = new WorkerManagerClass();
const validImageExtensions = [".jpeg", ".jpg", ".png", ".webp", ".avif"];
const validImageMIMEs = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/avif"
];
async function loadImageBitmap(url, asset) {
  const response = await DOMAdapter.get().fetch(url);
  if (!response.ok) {
    throw new Error(`[loadImageBitmap] Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const imageBlob = await response.blob();
  return asset?.data?.alphaMode === "premultiplied-alpha" ? createImageBitmap(imageBlob, { premultiplyAlpha: "none" }) : createImageBitmap(imageBlob);
}
const loadTextures = {
  /** used for deprecation purposes */
  name: "loadTextures",
  id: "texture",
  extension: {
    type: ExtensionType.LoadParser,
    priority: LoaderParserPriority.High,
    name: "loadTextures"
  },
  config: {
    preferWorkers: true,
    preferCreateImageBitmap: true,
    crossOrigin: "anonymous"
  },
  test(url) {
    return checkDataUrl(url, validImageMIMEs) || checkExtension(url, validImageExtensions);
  },
  async load(url, asset, loader) {
    let src = null;
    if (globalThis.createImageBitmap && this.config.preferCreateImageBitmap) {
      if (this.config.preferWorkers && await WorkerManager.isImageBitmapSupported()) {
        src = await WorkerManager.loadImageBitmap(url, asset);
      } else {
        src = await loadImageBitmap(url, asset);
      }
    } else {
      src = await new Promise((resolve, reject) => {
        src = DOMAdapter.get().createImage();
        src.crossOrigin = this.config.crossOrigin;
        src.src = url;
        if (src.complete) {
          resolve(src);
        } else {
          src.onload = () => {
            resolve(src);
          };
          src.onerror = reject;
        }
      });
    }
    const base = new ImageSource({
      resource: src,
      alphaMode: "premultiply-alpha-on-upload",
      resolution: asset.data?.resolution || getResolutionOfUrl(url),
      ...asset.data
    });
    return createTexture(base, loader, url);
  },
  unload(texture) {
    texture.destroy(true);
  }
};
const potentialVideoExtensions = [".mp4", ".m4v", ".webm", ".ogg", ".ogv", ".h264", ".avi", ".mov"];
let validVideoExtensions;
let validVideoMIMEs;
function crossOrigin(element, url, crossorigin) {
  if (crossorigin === void 0 && !url.startsWith("data:")) {
    element.crossOrigin = determineCrossOrigin(url);
  } else if (crossorigin !== false) {
    element.crossOrigin = typeof crossorigin === "string" ? crossorigin : "anonymous";
  }
}
function preloadVideo(element) {
  return new Promise((resolve, reject) => {
    element.addEventListener("canplaythrough", loaded);
    element.addEventListener("error", error);
    element.load();
    function loaded() {
      cleanup();
      resolve();
    }
    function error(err) {
      cleanup();
      reject(err);
    }
    function cleanup() {
      element.removeEventListener("canplaythrough", loaded);
      element.removeEventListener("error", error);
    }
  });
}
function determineCrossOrigin(url, loc = globalThis.location) {
  if (url.startsWith("data:")) {
    return "";
  }
  loc || (loc = globalThis.location);
  const parsedUrl = new URL(url, document.baseURI);
  if (parsedUrl.hostname !== loc.hostname || parsedUrl.port !== loc.port || parsedUrl.protocol !== loc.protocol) {
    return "anonymous";
  }
  return "";
}
function getBrowserSupportedVideoExtensions() {
  const supportedExtensions = [];
  const supportedMimes = [];
  for (const ext of potentialVideoExtensions) {
    const mimeType = VideoSource.MIME_TYPES[ext.substring(1)] || `video/${ext.substring(1)}`;
    if (testVideoFormat(mimeType)) {
      supportedExtensions.push(ext);
      if (!supportedMimes.includes(mimeType)) {
        supportedMimes.push(mimeType);
      }
    }
  }
  return {
    validVideoExtensions: supportedExtensions,
    validVideoMime: supportedMimes
  };
}
const loadVideoTextures = {
  /** used for deprecation purposes */
  name: "loadVideo",
  id: "video",
  extension: {
    type: ExtensionType.LoadParser,
    name: "loadVideo"
  },
  test(url) {
    if (!validVideoExtensions || !validVideoMIMEs) {
      const { validVideoExtensions: ve, validVideoMime: vm } = getBrowserSupportedVideoExtensions();
      validVideoExtensions = ve;
      validVideoMIMEs = vm;
    }
    const isValidDataUrl = checkDataUrl(url, validVideoMIMEs);
    const isValidExtension = checkExtension(url, validVideoExtensions);
    return isValidDataUrl || isValidExtension;
  },
  async load(url, asset, loader) {
    const options = {
      ...VideoSource.defaultOptions,
      resolution: asset.data?.resolution || getResolutionOfUrl(url),
      alphaMode: asset.data?.alphaMode || await detectVideoAlphaMode(),
      ...asset.data
    };
    const videoElement = document.createElement("video");
    const attributeMap = {
      preload: options.autoLoad !== false ? "auto" : void 0,
      "webkit-playsinline": options.playsinline !== false ? "" : void 0,
      playsinline: options.playsinline !== false ? "" : void 0,
      muted: options.muted === true ? "" : void 0,
      loop: options.loop === true ? "" : void 0,
      autoplay: options.autoPlay !== false ? "" : void 0
    };
    Object.keys(attributeMap).forEach((key) => {
      const value = attributeMap[key];
      if (value !== void 0)
        videoElement.setAttribute(key, value);
    });
    if (options.muted === true) {
      videoElement.muted = true;
    }
    crossOrigin(videoElement, url, options.crossorigin);
    const sourceElement = document.createElement("source");
    let mime;
    if (options.mime) {
      mime = options.mime;
    } else if (url.startsWith("data:")) {
      mime = url.slice(5, url.indexOf(";"));
    } else if (!url.startsWith("blob:")) {
      const ext = url.split("?")[0].slice(url.lastIndexOf(".") + 1).toLowerCase();
      mime = VideoSource.MIME_TYPES[ext] || `video/${ext}`;
    }
    sourceElement.src = url;
    if (mime) {
      sourceElement.type = mime;
    }
    return new Promise((resolve) => {
      const onCanPlay = async () => {
        const base = new VideoSource({ ...options, resource: videoElement });
        videoElement.removeEventListener("canplay", onCanPlay);
        if (asset.data.preload) {
          await preloadVideo(videoElement);
        }
        resolve(createTexture(base, loader, url));
      };
      if (options.preload && !options.autoPlay) {
        videoElement.load();
      }
      videoElement.addEventListener("canplay", onCanPlay);
      videoElement.appendChild(sourceElement);
    });
  },
  unload(texture) {
    texture.destroy(true);
  }
};
const resolveTextureUrl = {
  extension: {
    type: ExtensionType.ResolveParser,
    name: "resolveTexture"
  },
  test: loadTextures.test,
  parse: (value) => ({
    resolution: parseFloat(Resolver.RETINA_PREFIX.exec(value)?.[1] ?? "1"),
    format: value.split(".").pop(),
    src: value
  })
};
const resolveJsonUrl = {
  extension: {
    type: ExtensionType.ResolveParser,
    priority: -2,
    name: "resolveJson"
  },
  test: (value) => Resolver.RETINA_PREFIX.test(value) && value.endsWith(".json"),
  parse: resolveTextureUrl.parse
};
class AssetsClass {
  constructor() {
    this._detections = [];
    this._initialized = false;
    this.resolver = new Resolver();
    this.loader = new Loader();
    this.cache = Cache;
    this._backgroundLoader = new BackgroundLoader(this.loader);
    this._backgroundLoader.active = true;
    this.reset();
  }
  /**
   * Initializes the Assets class with configuration options. While not required,
   * calling this before loading assets is recommended to set up default behaviors.
   * @param options - Configuration options for the Assets system
   * @example
   * ```ts
   * // Basic initialization (optional as Assets.load will call this automatically)
   * await Assets.init();
   *
   * // With CDN configuration
   * await Assets.init({
   *     basePath: 'https://my-cdn.com/assets/',
   *     defaultSearchParams: { version: '1.0.0' }
   * });
   *
   * // With manifest and preferences
   * await Assets.init({
   *     manifest: {
   *         bundles: [{
   *             name: 'game-screen',
   *             assets: [
   *                 {
   *                     alias: 'hero',
   *                     src: 'hero.{png,webp}',
   *                     data: { scaleMode: SCALE_MODES.NEAREST }
   *                 },
   *                 {
   *                     alias: 'map',
   *                     src: 'map.json'
   *                 }
   *             ]
   *         }]
   *     },
   *     // Optimize for device capabilities
   *     texturePreference: {
   *         resolution: window.devicePixelRatio,
   *         format: ['webp', 'png']
   *     },
   *     // Set global preferences
   *     preferences: {
   *         crossOrigin: 'anonymous',
   *     }
   * });
   *
   * // Load assets after initialization
   * const heroTexture = await Assets.load('hero');
   * ```
   * @remarks
   * - Can be called only once; subsequent calls will be ignored with a warning
   * - Format detection runs automatically unless `skipDetections` is true
   * - The manifest can be a URL to a JSON file or an inline object
   * @see {@link AssetInitOptions} For all available initialization options
   * @see {@link AssetsManifest} For manifest format details
   */
  async init(options = {}) {
    if (this._initialized) {
      warn("[Assets]AssetManager already initialized, did you load before calling this Assets.init()?");
      return;
    }
    this._initialized = true;
    if (options.defaultSearchParams) {
      this.resolver.setDefaultSearchParams(options.defaultSearchParams);
    }
    if (options.basePath) {
      this.resolver.basePath = options.basePath;
    }
    if (options.bundleIdentifier) {
      this.resolver.setBundleIdentifier(options.bundleIdentifier);
    }
    if (options.manifest) {
      let manifest = options.manifest;
      if (typeof manifest === "string") {
        manifest = await this.load(manifest);
      }
      this.resolver.addManifest(manifest);
    }
    const resolutionPref = options.texturePreference?.resolution ?? 1;
    const resolution = typeof resolutionPref === "number" ? [resolutionPref] : resolutionPref;
    const formats = await this._detectFormats({
      preferredFormats: options.texturePreference?.format,
      skipDetections: options.skipDetections,
      detections: this._detections
    });
    this.resolver.prefer({
      params: {
        format: formats,
        resolution
      }
    });
    if (options.preferences) {
      this.setPreferences(options.preferences);
    }
    if (options.loadOptions) {
      this.loader.loadOptions = {
        ...this.loader.loadOptions,
        ...options.loadOptions
      };
    }
  }
  /**
   * Registers assets with the Assets resolver. This method maps keys (aliases) to asset sources,
   * allowing you to load assets using friendly names instead of direct URLs.
   * @param assets - The unresolved assets to add to the resolver
   * @example
   * ```ts
   * // Basic usage - single asset
   * Assets.add({
   *     alias: 'myTexture',
   *     src: 'assets/texture.png'
   * });
   * const texture = await Assets.load('myTexture');
   *
   * // Multiple aliases for the same asset
   * Assets.add({
   *     alias: ['hero', 'player'],
   *     src: 'hero.png'
   * });
   * const hero1 = await Assets.load('hero');
   * const hero2 = await Assets.load('player'); // Same texture
   *
   * // Multiple format support
   * Assets.add({
   *     alias: 'character',
   *     src: 'character.{webp,png}' // Will choose best format
   * });
   * Assets.add({
   *     alias: 'character',
   *     src: ['character.webp', 'character.png'], // Explicitly specify formats
   * });
   *
   * // With texture options
   * Assets.add({
   *     alias: 'sprite',
   *     src: 'sprite.png',
   *     data: { scaleMode: 'nearest' }
   * });
   *
   * // Multiple assets at once
   * Assets.add([
   *     { alias: 'bg', src: 'background.png' },
   *     { alias: 'music', src: 'music.mp3' },
   *     { alias: 'spritesheet', src: 'sheet.json', data: { ignoreMultiPack: false } }
   * ]);
   * ```
   * @remarks
   * - Assets are resolved when loaded, not when added
   * - Multiple formats use the best available format for the browser
   * - Adding with same alias overwrites previous definition
   * - The `data` property is passed to the asset loader
   * @see {@link Resolver} For details on asset resolution
   * @see {@link LoaderParser} For asset-specific data options
   * @advanced
   */
  add(assets) {
    this.resolver.add(assets);
  }
  async load(urls, onProgress) {
    if (!this._initialized) {
      await this.init();
    }
    const singleAsset = isSingleItem(urls);
    const urlArray = convertToList(urls).map((url) => {
      if (typeof url !== "string") {
        const aliases = this.resolver.getAlias(url);
        if (aliases.some((alias) => !this.resolver.hasKey(alias))) {
          this.add(url);
        }
        return Array.isArray(aliases) ? aliases[0] : aliases;
      }
      if (!this.resolver.hasKey(url))
        this.add({ alias: url, src: url });
      return url;
    });
    const resolveResults = this.resolver.resolve(urlArray);
    const out2 = await this._mapLoadToResolve(resolveResults, onProgress);
    return singleAsset ? out2[urlArray[0]] : out2;
  }
  /**
   * Registers a bundle of assets that can be loaded as a group. Bundles are useful for organizing
   * assets into logical groups, such as game levels or UI screens.
   * @param bundleId - Unique identifier for the bundle
   * @param assets - Assets to include in the bundle
   * @example
   * ```ts
   * // Add a bundle using array format
   * Assets.addBundle('animals', [
   *     { alias: 'bunny', src: 'bunny.png' },
   *     { alias: 'chicken', src: 'chicken.png' },
   *     { alias: 'thumper', src: 'thumper.png' },
   * ]);
   *
   * // Add a bundle using object format
   * Assets.addBundle('animals', {
   *     bunny: 'bunny.png',
   *     chicken: 'chicken.png',
   *     thumper: 'thumper.png',
   * });
   *
   * // Add a bundle with advanced options
   * Assets.addBundle('ui', [
   *     {
   *         alias: 'button',
   *         src: 'button.{webp,png}',
   *         data: { scaleMode: 'nearest' }
   *     },
   *     {
   *         alias: ['logo', 'brand'],  // Multiple aliases
   *         src: 'logo.svg',
   *         data: { resolution: 2 }
   *     }
   * ]);
   *
   * // Load the bundle
   * await Assets.loadBundle('animals');
   *
   * // Use the loaded assets
   * const bunny = Sprite.from('bunny');
   * const chicken = Sprite.from('chicken');
   * ```
   * @remarks
   * - Bundle IDs must be unique
   * - Assets in bundles are not loaded until `loadBundle` is called
   * - Bundles can be background loaded using `backgroundLoadBundle`
   * - Assets in bundles can be loaded individually using their aliases
   * @see {@link Assets.loadBundle} For loading bundles
   * @see {@link Assets.backgroundLoadBundle} For background loading bundles
   * @see {@link Assets.unloadBundle} For unloading bundles
   * @see {@link AssetsManifest} For manifest format details
   */
  addBundle(bundleId, assets) {
    this.resolver.addBundle(bundleId, assets);
  }
  /**
   * Loads a bundle or multiple bundles of assets. Bundles are collections of related assets
   * that can be loaded together.
   * @param bundleIds - Single bundle ID or array of bundle IDs to load
   * @param onProgress - Optional callback for load progress (0.0 to 1.0)
   * @returns Promise that resolves with the loaded bundle assets
   * @example
   * ```ts
   * // Define bundles in your manifest
   * const manifest = {
   *     bundles: [
   *         {
   *             name: 'load-screen',
   *             assets: [
   *                 {
   *                     alias: 'background',
   *                     src: 'sunset.png',
   *                 },
   *                 {
   *                     alias: 'bar',
   *                     src: 'load-bar.{png,webp}', // use an array of individual assets
   *                 },
   *             ],
   *         },
   *         {
   *             name: 'game-screen',
   *             assets: [
   *                 {
   *                     alias: 'character',
   *                     src: 'robot.png',
   *                 },
   *                 {
   *                     alias: 'enemy',
   *                     src: 'bad-guy.png',
   *                 },
   *             ],
   *         },
   *     ]
   * };
   *
   * // Initialize with manifest
   * await Assets.init({ manifest });
   *
   * // Or add bundles programmatically
   * Assets.addBundle('load-screen', [...]);
   * Assets.loadBundle('load-screen');
   *
   * // Load a single bundle
   * await Assets.loadBundle('load-screen');
   * const bg = Sprite.from('background'); // Uses alias from bundle
   *
   * // Load multiple bundles
   * await Assets.loadBundle([
   *     'load-screen',
   *     'game-screen'
   * ]);
   *
   * // Load with progress tracking
   * await Assets.loadBundle('game-screen', (progress) => {
   *     console.log(`Loading: ${Math.round(progress * 100)}%`);
   * });
   * ```
   * @remarks
   * - Bundle assets are cached automatically
   * - Bundles can be pre-loaded using `backgroundLoadBundle`
   * - Assets in bundles can be accessed by their aliases
   * - Progress callback receives values from 0.0 to 1.0
   * @throws {Error} If the bundle ID doesn't exist in the manifest
   * @see {@link Assets.addBundle} For adding bundles programmatically
   * @see {@link Assets.backgroundLoadBundle} For background loading bundles
   * @see {@link Assets.unloadBundle} For unloading bundles
   * @see {@link AssetsManifest} For manifest format details
   */
  async loadBundle(bundleIds, onProgress) {
    if (!this._initialized) {
      await this.init();
    }
    let singleAsset = false;
    if (typeof bundleIds === "string") {
      singleAsset = true;
      bundleIds = [bundleIds];
    }
    const resolveResults = this.resolver.resolveBundle(bundleIds);
    const out2 = {};
    const keys = Object.keys(resolveResults);
    let total = 0;
    const counts = [];
    const _onProgress = () => {
      onProgress?.(counts.reduce((a2, b2) => a2 + b2, 0) / total);
    };
    const promises = keys.map((bundleId, i2) => {
      const resolveResult = resolveResults[bundleId];
      const values = Object.values(resolveResult);
      const totalAssetsToLoad = [...new Set(values.flat())];
      const progressSize = totalAssetsToLoad.reduce((sum, asset) => sum + (asset.progressSize || 1), 0);
      counts.push(0);
      total += progressSize;
      return this._mapLoadToResolve(resolveResult, (e2) => {
        counts[i2] = e2 * progressSize;
        _onProgress();
      }).then((resolveResult2) => {
        out2[bundleId] = resolveResult2;
      });
    });
    await Promise.all(promises);
    return singleAsset ? out2[bundleIds[0]] : out2;
  }
  /**
   * Initiates background loading of assets. This allows assets to be loaded passively while other operations
   * continue, making them instantly available when needed later.
   *
   * Background loading is useful for:
   * - Preloading game levels while in a menu
   * - Loading non-critical assets during gameplay
   * - Reducing visible loading screens
   * @param urls - Single URL/alias or array of URLs/aliases to load in the background
   * @example
   * ```ts
   * // Basic background loading
   * Assets.backgroundLoad('images/level2-assets.png');
   *
   * // Background load multiple assets
   * Assets.backgroundLoad([
   *     'images/sprite1.png',
   *     'images/sprite2.png',
   *     'images/background.png'
   * ]);
   *
   * // Later, when you need the assets
   * const textures = await Assets.load([
   *     'images/sprite1.png',
   *     'images/sprite2.png'
   * ]); // Resolves immediately if background loading completed
   * ```
   * @remarks
   * - Background loading happens one asset at a time to avoid blocking the main thread
   * - Loading can be interrupted safely by calling `Assets.load()`
   * - Assets are cached as they complete loading
   * - No progress tracking is available for background loading
   */
  async backgroundLoad(urls) {
    if (!this._initialized) {
      await this.init();
    }
    if (typeof urls === "string") {
      urls = [urls];
    }
    const resolveResults = this.resolver.resolve(urls);
    this._backgroundLoader.add(Object.values(resolveResults));
  }
  /**
   * Initiates background loading of asset bundles. Similar to backgroundLoad but works with
   * predefined bundles of assets.
   *
   * Perfect for:
   * - Preloading level bundles during gameplay
   * - Loading UI assets during splash screens
   * - Preparing assets for upcoming game states
   * @param bundleIds - Single bundle ID or array of bundle IDs to load in the background
   * @example
   * ```ts
   * // Define bundles in your manifest
   * await Assets.init({
   *     manifest: {
   *         bundles: [
   *             {
   *               name: 'home',
   *               assets: [
   *                 {
   *                     alias: 'background',
   *                     src: 'images/home-bg.png',
   *                 },
   *                 {
   *                     alias: 'logo',
   *                     src: 'images/logo.png',
   *                 }
   *              ]
   *            },
   *            {
   *             name: 'level-1',
   *             assets: [
   *                 {
   *                     alias: 'background',
   *                     src: 'images/level1/bg.png',
   *                 },
   *                 {
   *                     alias: 'sprites',
   *                     src: 'images/level1/sprites.json'
   *                 }
   *             ]
   *         }]
   *     }
   * });
   *
   * // Load the home screen assets right away
   * await Assets.loadBundle('home');
   * showHomeScreen();
   *
   * // Start background loading while showing home screen
   * Assets.backgroundLoadBundle('level-1');
   *
   * // When player starts level, load completes faster
   * await Assets.loadBundle('level-1');
   * hideHomeScreen();
   * startLevel();
   * ```
   * @remarks
   * - Bundle assets are loaded one at a time
   * - Loading can be interrupted safely by calling `Assets.loadBundle()`
   * - Assets are cached as they complete loading
   * - Requires bundles to be registered via manifest or `addBundle`
   * @see {@link Assets.addBundle} For adding bundles programmatically
   * @see {@link Assets.loadBundle} For immediate bundle loading
   * @see {@link AssetsManifest} For manifest format details
   */
  async backgroundLoadBundle(bundleIds) {
    if (!this._initialized) {
      await this.init();
    }
    if (typeof bundleIds === "string") {
      bundleIds = [bundleIds];
    }
    const resolveResults = this.resolver.resolveBundle(bundleIds);
    Object.values(resolveResults).forEach((resolveResult) => {
      this._backgroundLoader.add(Object.values(resolveResult));
    });
  }
  /**
   * Only intended for development purposes.
   * This will wipe the resolver and caches.
   * You will need to reinitialize the Asset
   * @internal
   */
  reset() {
    this.resolver.reset();
    this.loader.reset();
    this.cache.reset();
    this._initialized = false;
  }
  get(keys) {
    if (typeof keys === "string") {
      return Cache.get(keys);
    }
    const assets = {};
    for (let i2 = 0; i2 < keys.length; i2++) {
      assets[i2] = Cache.get(keys[i2]);
    }
    return assets;
  }
  /**
   * helper function to map resolved assets back to loaded assets
   * @param resolveResults - the resolve results from the resolver
   * @param progressOrLoadOptions - the progress callback or load options
   */
  async _mapLoadToResolve(resolveResults, progressOrLoadOptions) {
    const resolveArray = [...new Set(Object.values(resolveResults))];
    this._backgroundLoader.active = false;
    const loadedAssets = await this.loader.load(resolveArray, progressOrLoadOptions);
    this._backgroundLoader.active = true;
    const out2 = {};
    resolveArray.forEach((resolveResult) => {
      const asset = loadedAssets[resolveResult.src];
      const keys = [resolveResult.src];
      if (resolveResult.alias) {
        keys.push(...resolveResult.alias);
      }
      keys.forEach((key) => {
        out2[key] = asset;
      });
      Cache.set(keys, asset);
    });
    return out2;
  }
  /**
   * Unloads assets and releases them from memory. This method ensures proper cleanup of
   * loaded assets when they're no longer needed.
   * @param urls - Single URL/alias or array of URLs/aliases to unload
   * @example
   * ```ts
   * // Unload a single asset
   * await Assets.unload('images/sprite.png');
   *
   * // Unload using an alias
   * await Assets.unload('hero'); // Unloads the asset registered with 'hero' alias
   *
   * // Unload multiple assets
   * await Assets.unload([
   *     'images/background.png',
   *     'images/character.png',
   *     'hero'
   * ]);
   *
   * // Unload and handle creation of new instances
   * await Assets.unload('hero');
   * const newHero = await Assets.load('hero'); // Will load fresh from source
   * ```
   * @remarks
   * > [!WARNING]
   * > Make sure assets aren't being used before unloading:
   * > - Remove sprites using the texture
   * > - Clear any references to the asset
   * > - Textures will be destroyed and can't be used after unloading
   * @throws {Error} If the asset is not found in cache
   */
  async unload(urls) {
    if (!this._initialized) {
      await this.init();
    }
    const urlArray = convertToList(urls).map((url) => typeof url !== "string" ? url.src : url);
    const resolveResults = this.resolver.resolve(urlArray);
    await this._unloadFromResolved(resolveResults);
  }
  /**
   * Unloads all assets in a bundle. Use this to free memory when a bundle's assets
   * are no longer needed, such as when switching game levels.
   * @param bundleIds - Single bundle ID or array of bundle IDs to unload
   * @example
   * ```ts
   * // Define and load a bundle
   * Assets.addBundle('level-1', {
   *     background: 'level1/bg.png',
   *     sprites: 'level1/sprites.json',
   *     music: 'level1/music.mp3'
   * });
   *
   * // Load the bundle
   * const level1 = await Assets.loadBundle('level-1');
   *
   * // Use the assets
   * const background = Sprite.from(level1.background);
   *
   * // When done with the level, unload everything
   * await Assets.unloadBundle('level-1');
   * // background sprite is now invalid!
   *
   * // Unload multiple bundles
   * await Assets.unloadBundle([
   *     'level-1',
   *     'level-2',
   *     'ui-elements'
   * ]);
   * ```
   * @remarks
   * > [!WARNING]
   * > - All assets in the bundle will be destroyed
   * > - Bundle needs to be reloaded to use assets again
   * > - Make sure no sprites or other objects are using the assets
   * @throws {Error} If the bundle is not found
   * @see {@link Assets.addBundle} For adding bundles
   * @see {@link Assets.loadBundle} For loading bundles
   */
  async unloadBundle(bundleIds) {
    if (!this._initialized) {
      await this.init();
    }
    bundleIds = convertToList(bundleIds);
    const resolveResults = this.resolver.resolveBundle(bundleIds);
    const promises = Object.keys(resolveResults).map((bundleId) => this._unloadFromResolved(resolveResults[bundleId]));
    await Promise.all(promises);
  }
  async _unloadFromResolved(resolveResult) {
    const resolveArray = Object.values(resolveResult);
    resolveArray.forEach((resolveResult2) => {
      Cache.remove(resolveResult2.src);
    });
    await this.loader.unload(resolveArray);
  }
  /**
   * Detects the supported formats for the browser, and returns an array of supported formats, respecting
   * the users preferred formats order.
   * @param options - the options to use when detecting formats
   * @param options.preferredFormats - the preferred formats to use
   * @param options.skipDetections - if we should skip the detections altogether
   * @param options.detections - the detections to use
   * @returns - the detected formats
   */
  async _detectFormats(options) {
    let formats = [];
    if (options.preferredFormats) {
      formats = Array.isArray(options.preferredFormats) ? options.preferredFormats : [options.preferredFormats];
    }
    for (const detection of options.detections) {
      if (options.skipDetections || await detection.test()) {
        formats = await detection.add(formats);
      } else if (!options.skipDetections) {
        formats = await detection.remove(formats);
      }
    }
    formats = formats.filter((format, index) => formats.indexOf(format) === index);
    return formats;
  }
  /**
   * All the detection parsers currently added to the Assets class.
   * @advanced
   */
  get detections() {
    return this._detections;
  }
  /**
   * Sets global preferences for asset loading behavior. This method configures how assets
   * are loaded and processed across all parsers.
   * @param preferences - Asset loading preferences
   * @example
   * ```ts
   * // Basic preferences
   * Assets.setPreferences({
   *     crossOrigin: 'anonymous',
   *     parseAsGraphicsContext: false
   * });
   * ```
   * @remarks
   * Preferences are applied to all compatible parsers and affect future asset loading.
   * Common preferences include:
   * - `crossOrigin`: CORS setting for loaded assets
   * - `preferWorkers`: Whether to use web workers for loading textures
   * - `preferCreateImageBitmap`: Use `createImageBitmap` for texture creation. Turning this off will use the `Image` constructor instead.
   * @see {@link AssetsPreferences} For all available preferences
   */
  setPreferences(preferences) {
    this.loader.parsers.forEach((parser) => {
      if (!parser.config)
        return;
      Object.keys(parser.config).filter((key) => key in preferences).forEach((key) => {
        parser.config[key] = preferences[key];
      });
    });
  }
}
const Assets = new AssetsClass();
extensions.handleByList(ExtensionType.LoadParser, Assets.loader.parsers).handleByList(ExtensionType.ResolveParser, Assets.resolver.parsers).handleByList(ExtensionType.CacheParser, Assets.cache.parsers).handleByList(ExtensionType.DetectionParser, Assets.detections);
extensions.add(
  cacheTextureArray,
  detectDefaults,
  detectAvif,
  detectWebp,
  detectMp4,
  detectOgv,
  detectWebm,
  loadJson,
  loadTxt,
  loadWebFont,
  loadSvg,
  loadTextures,
  loadVideoTextures,
  loadBitmapFont,
  bitmapFontCachePlugin,
  resolveTextureUrl,
  resolveJsonUrl
);
const assetKeyMap = {
  loader: ExtensionType.LoadParser,
  resolver: ExtensionType.ResolveParser,
  cache: ExtensionType.CacheParser,
  detection: ExtensionType.DetectionParser
};
extensions.handle(ExtensionType.Asset, (extension) => {
  const ref = extension.ref;
  Object.entries(assetKeyMap).filter(([key]) => !!ref[key]).forEach(([key, type]) => extensions.add(Object.assign(
    ref[key],
    // Allow the function to optionally define it's own
    // ExtensionMetadata, the use cases here is priority for LoaderParsers
    { extension: ref[key].extension ?? type }
  )));
}, (extension) => {
  const ref = extension.ref;
  Object.keys(assetKeyMap).filter((key) => !!ref[key]).forEach((key) => extensions.remove(ref[key]));
});
class Graphics extends ViewContainer {
  /**
   * Creates a new Graphics object.
   * @param options - Options for the Graphics.
   */
  constructor(options) {
    if (options instanceof GraphicsContext) {
      options = { context: options };
    }
    const { context: context2, roundPixels, ...rest } = options || {};
    super({
      label: "Graphics",
      ...rest
    });
    this.renderPipeId = "graphics";
    if (!context2) {
      this.context = this._ownedContext = new GraphicsContext();
      this.context.autoGarbageCollect = this.autoGarbageCollect;
    } else {
      this.context = context2;
    }
    this.didViewUpdate = true;
    this.allowChildren = false;
    this.roundPixels = roundPixels ?? false;
  }
  set context(context2) {
    if (context2 === this._context)
      return;
    if (this._context) {
      this._context.off("update", this.onViewUpdate, this);
      this._context.off("unload", this.unload, this);
    }
    this._context = context2;
    this._context.on("update", this.onViewUpdate, this);
    this._context.on("unload", this.unload, this);
    this.onViewUpdate();
  }
  /**
   * The underlying graphics context used for drawing operations.
   * Controls how shapes and paths are rendered.
   * @example
   * ```ts
   * // Create a shared context
   * const sharedContext = new GraphicsContext();
   *
   * // Create graphics objects sharing the same context
   * const graphics1 = new Graphics();
   * const graphics2 = new Graphics();
   *
   * // Assign shared context
   * graphics1.context = sharedContext;
   * graphics2.context = sharedContext;
   *
   * // Both graphics will show the same shapes
   * sharedContext
   *     .rect(0, 0, 100, 100)
   *     .fill({ color: 0xff0000 });
   * ```
   * @see {@link GraphicsContext} For drawing operations
   * @see {@link GraphicsOptions} For context configuration
   */
  get context() {
    return this._context;
  }
  /**
   * The local bounds of the graphics object.
   * Returns the boundaries after all graphical operations but before any transforms.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Draw a shape
   * graphics
   *     .rect(0, 0, 100, 100)
   *     .fill({ color: 0xff0000 });
   *
   * // Get bounds information
   * const bounds = graphics.bounds;
   * console.log(bounds.width);  // 100
   * console.log(bounds.height); // 100
   * ```
   * @readonly
   * @see {@link Bounds} For bounds operations
   * @see {@link Container#getBounds} For transformed bounds
   */
  get bounds() {
    return this._context.bounds;
  }
  /**
   * Graphics objects do not need to update their bounds as the context handles this.
   * @private
   */
  updateBounds() {
  }
  /**
   * Checks if the object contains the given point.
   * Returns true if the point lies within the Graphics object's rendered area.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Draw a shape
   * graphics
   *     .rect(0, 0, 100, 100)
   *     .fill({ color: 0xff0000 });
   *
   * // Check point intersection
   * if (graphics.containsPoint({ x: 50, y: 50 })) {
   *     console.log('Point is inside rectangle!');
   * }
   * ```
   * @param point - The point to check in local coordinates
   * @returns True if the point is inside the Graphics object
   * @see {@link Graphics#bounds} For bounding box checks
   * @see {@link PointData} For point data structure
   */
  containsPoint(point) {
    return this._context.containsPoint(point);
  }
  /**
   * Destroys this graphics renderable and optionally its context.
   * @param options - Options parameter. A boolean will act as if all options
   *
   * If the context was created by this graphics and `destroy(false)` or `destroy()` is called
   * then the context will still be destroyed.
   *
   * If you want to explicitly not destroy this context that this graphics created,
   * then you should pass destroy({ context: false })
   *
   * If the context was passed in as an argument to the constructor then it will not be destroyed
   * @example
   * ```ts
   * // Destroy the graphics and its context
   * graphics.destroy();
   * graphics.destroy(true);
   * graphics.destroy({ context: true, texture: true, textureSource: true });
   * ```
   */
  destroy(options) {
    if (this._ownedContext && !options) {
      this._ownedContext.destroy(options);
    } else if (options === true || options?.context === true) {
      this._context.destroy(options);
    }
    this._ownedContext = null;
    this._context = null;
    super.destroy(options);
  }
  /**
   * @param now - The current time in milliseconds.
   * @internal
   */
  _onTouch(now) {
    this._gcLastUsed = now;
    this._context._gcLastUsed = now;
  }
  _callContextMethod(method, args) {
    this.context[method](...args);
    return this;
  }
  // --------------------------------------- GraphicsContext methods ---------------------------------------
  /**
   * Sets the current fill style of the graphics context.
   * The fill style can be a color, gradient, pattern, or a complex style object.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Basic color fill
   * graphics
   *     .setFillStyle({ color: 0xff0000 }) // Red fill
   *     .rect(0, 0, 100, 100)
   *     .fill();
   *
   * // Gradient fill
   * const gradient = new FillGradient({
   *    end: { x: 1, y: 0 },
   *    colorStops: [
   *         { offset: 0, color: 0xff0000 }, // Red at start
   *         { offset: 0.5, color: 0x00ff00 }, // Green at middle
   *         { offset: 1, color: 0x0000ff }, // Blue at end
   *    ],
   * });
   *
   * graphics
   *     .setFillStyle(gradient)
   *     .circle(100, 100, 50)
   *     .fill();
   *
   * // Pattern fill
   * const pattern = new FillPattern(texture);
   * graphics
   *     .setFillStyle({
   *         fill: pattern,
   *         alpha: 0.5
   *     })
   *     .rect(0, 0, 200, 200)
   *     .fill();
   * ```
   * @param {FillInput} args - The fill style to apply
   * @returns The Graphics instance for chaining
   * @see {@link FillStyle} For fill style options
   * @see {@link FillGradient} For gradient fills
   * @see {@link FillPattern} For pattern fills
   */
  setFillStyle(...args) {
    return this._callContextMethod("setFillStyle", args);
  }
  /**
   * Sets the current stroke style of the graphics context.
   * Similar to fill styles, stroke styles can encompass colors, gradients, patterns, or more detailed configurations.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Basic color stroke
   * graphics
   *     .setStrokeStyle({
   *         width: 2,
   *         color: 0x000000
   *     })
   *     .rect(0, 0, 100, 100)
   *     .stroke();
   *
   * // Complex stroke style
   * graphics
   *     .setStrokeStyle({
   *         width: 4,
   *         color: 0xff0000,
   *         alpha: 0.5,
   *         join: 'round',
   *         cap: 'round',
   *         alignment: 0.5
   *     })
   *     .circle(100, 100, 50)
   *     .stroke();
   *
   * // Gradient stroke
   * const gradient = new FillGradient({
   *    end: { x: 1, y: 0 },
   *    colorStops: [
   *         { offset: 0, color: 0xff0000 }, // Red at start
   *         { offset: 0.5, color: 0x00ff00 }, // Green at middle
   *         { offset: 1, color: 0x0000ff }, // Blue at end
   *    ],
   * });
   *
   * graphics
   *     .setStrokeStyle({
   *         width: 10,
   *         fill: gradient
   *     })
   *     .poly([0,0, 100,50, 0,100])
   *     .stroke();
   * ```
   * @param {StrokeInput} args - The stroke style to apply
   * @returns The Graphics instance for chaining
   * @see {@link StrokeStyle} For stroke style options
   * @see {@link FillGradient} For gradient strokes
   * @see {@link FillPattern} For pattern strokes
   */
  setStrokeStyle(...args) {
    return this._callContextMethod("setStrokeStyle", args);
  }
  fill(...args) {
    return this._callContextMethod("fill", args);
  }
  /**
   * Strokes the current path with the current stroke style or specified style.
   * Outlines the shape using the stroke settings.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Stroke with direct color
   * graphics
   *     .circle(50, 50, 25)
   *     .stroke({
   *         width: 2,
   *         color: 0xff0000
   *     }); // 2px red stroke
   *
   * // Fill with texture
   * graphics
   *    .rect(0, 0, 100, 100)
   *    .stroke(myTexture); // Fill with texture
   *
   * // Stroke with gradient
   * const gradient = new FillGradient({
   *     end: { x: 1, y: 0 },
   *     colorStops: [
   *         { offset: 0, color: 0xff0000 },
   *         { offset: 0.5, color: 0x00ff00 },
   *         { offset: 1, color: 0x0000ff },
   *     ],
   * });
   *
   * graphics
   *     .rect(0, 0, 100, 100)
   *     .stroke({
   *         width: 4,
   *         fill: gradient,
   *         alignment: 0.5,
   *         join: 'round'
   *     });
   * ```
   * @param {StrokeStyle} args - Optional stroke style to apply. Can be:
   * - A stroke style object with width, color, etc.
   * - A gradient
   * - A pattern
   * If omitted, uses current stroke style.
   * @returns The Graphics instance for chaining
   * @see {@link StrokeStyle} For stroke style options
   * @see {@link FillGradient} For gradient strokes
   * @see {@link setStrokeStyle} For setting default stroke style
   */
  stroke(...args) {
    return this._callContextMethod("stroke", args);
  }
  texture(...args) {
    return this._callContextMethod("texture", args);
  }
  /**
   * Resets the current path. Any previous path and its commands are discarded and a new path is
   * started. This is typically called before beginning a new shape or series of drawing commands.
   * @example
   * ```ts
   * const graphics = new Graphics();
   * graphics
   *     .circle(150, 150, 50)
   *     .fill({ color: 0x00ff00 })
   *     .beginPath() // Starts a new path
   *     .circle(250, 150, 50)
   *     .fill({ color: 0x0000ff });
   * ```
   * @returns The Graphics instance for chaining
   * @see {@link Graphics#moveTo} For starting a new subpath
   * @see {@link Graphics#closePath} For closing the current path
   */
  beginPath() {
    return this._callContextMethod("beginPath", []);
  }
  /**
   * Applies a cutout to the last drawn shape. This is used to create holes or complex shapes by
   * subtracting a path from the previously drawn path.
   *
   * If a hole is not completely in a shape, it will fail to cut correctly.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Draw outer circle
   * graphics
   *     .circle(100, 100, 50)
   *     .fill({ color: 0xff0000 });
   *     .circle(100, 100, 25) // Inner circle
   *     .cut() // Cuts out the inner circle from the outer circle
   * ```
   */
  cut() {
    return this._callContextMethod("cut", []);
  }
  arc(...args) {
    return this._callContextMethod("arc", args);
  }
  arcTo(...args) {
    return this._callContextMethod("arcTo", args);
  }
  arcToSvg(...args) {
    return this._callContextMethod("arcToSvg", args);
  }
  bezierCurveTo(...args) {
    return this._callContextMethod("bezierCurveTo", args);
  }
  /**
   * Closes the current path by drawing a straight line back to the start point.
   *
   * This is useful for completing shapes and ensuring they are properly closed for fills.
   * @example
   * ```ts
   * // Create a triangle with closed path
   * const graphics = new Graphics();
   * graphics
   *     .moveTo(50, 50)
   *     .lineTo(100, 100)
   *     .lineTo(0, 100)
   *     .closePath()
   * ```
   * @returns The Graphics instance for method chaining
   * @see {@link Graphics#beginPath} For starting a new path
   * @see {@link Graphics#fill} For filling closed paths
   * @see {@link Graphics#stroke} For stroking paths
   */
  closePath() {
    return this._callContextMethod("closePath", []);
  }
  ellipse(...args) {
    return this._callContextMethod("ellipse", args);
  }
  circle(...args) {
    return this._callContextMethod("circle", args);
  }
  path(...args) {
    return this._callContextMethod("path", args);
  }
  lineTo(...args) {
    return this._callContextMethod("lineTo", args);
  }
  moveTo(...args) {
    return this._callContextMethod("moveTo", args);
  }
  quadraticCurveTo(...args) {
    return this._callContextMethod("quadraticCurveTo", args);
  }
  rect(...args) {
    return this._callContextMethod("rect", args);
  }
  roundRect(...args) {
    return this._callContextMethod("roundRect", args);
  }
  poly(...args) {
    return this._callContextMethod("poly", args);
  }
  regularPoly(...args) {
    return this._callContextMethod("regularPoly", args);
  }
  roundPoly(...args) {
    return this._callContextMethod("roundPoly", args);
  }
  roundShape(...args) {
    return this._callContextMethod("roundShape", args);
  }
  filletRect(...args) {
    return this._callContextMethod("filletRect", args);
  }
  chamferRect(...args) {
    return this._callContextMethod("chamferRect", args);
  }
  star(...args) {
    return this._callContextMethod("star", args);
  }
  svg(...args) {
    return this._callContextMethod("svg", args);
  }
  restore(...args) {
    return this._callContextMethod("restore", args);
  }
  /**
   * Saves the current graphics state onto a stack. The state includes:
   * - Current transformation matrix
   * - Current fill style
   * - Current stroke style
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Save state before complex operations
   * graphics.save();
   *
   * // Create transformed and styled shape
   * graphics
   *     .translateTransform(100, 100)
   *     .rotateTransform(Math.PI / 4)
   *     .setFillStyle({
   *         color: 0xff0000,
   *         alpha: 0.5
   *     })
   *     .rect(-25, -25, 50, 50)
   *     .fill();
   *
   * // Restore to original state
   * graphics.restore();
   *
   * // Continue drawing with previous state
   * graphics
   *     .circle(50, 50, 25)
   *     .fill();
   * ```
   * @returns The Graphics instance for method chaining
   * @see {@link Graphics#restore} For restoring the saved state
   * @see {@link Graphics#setTransform} For setting transformations
   */
  save() {
    return this._callContextMethod("save", []);
  }
  /**
   * Returns the current transformation matrix of the graphics context.
   * This matrix represents all accumulated transformations including translate, scale, and rotate.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Apply some transformations
   * graphics
   *     .translateTransform(100, 100)
   *     .rotateTransform(Math.PI / 4);
   *
   * // Get the current transform matrix
   * const matrix = graphics.getTransform();
   * console.log(matrix.tx, matrix.ty); // 100, 100
   *
   * // Use the matrix for other operations
   * graphics
   *     .setTransform(matrix)
   *     .circle(0, 0, 50)
   *     .fill({ color: 0xff0000 });
   * ```
   * @returns The current transformation matrix.
   * @see {@link Graphics#setTransform} For setting the transform matrix
   * @see {@link Matrix} For matrix operations
   */
  getTransform() {
    return this.context.getTransform();
  }
  /**
   * Resets the current transformation matrix to the identity matrix, effectively removing
   * any transformations (rotation, scaling, translation) previously applied.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Apply transformations
   * graphics
   *     .translateTransform(100, 100)
   *     .scaleTransform(2, 2)
   *     .circle(0, 0, 25)
   *     .fill({ color: 0xff0000 });
   * // Reset transform to default state
   * graphics
   *     .resetTransform()
   *     .circle(50, 50, 25) // Will draw at actual coordinates
   *     .fill({ color: 0x00ff00 });
   * ```
   * @returns The Graphics instance for method chaining
   * @see {@link Graphics#getTransform} For getting the current transform
   * @see {@link Graphics#setTransform} For setting a specific transform
   * @see {@link Graphics#save} For saving the current transform state
   * @see {@link Graphics#restore} For restoring a previous transform state
   */
  resetTransform() {
    return this._callContextMethod("resetTransform", []);
  }
  rotateTransform(...args) {
    return this._callContextMethod("rotate", args);
  }
  scaleTransform(...args) {
    return this._callContextMethod("scale", args);
  }
  setTransform(...args) {
    return this._callContextMethod("setTransform", args);
  }
  transform(...args) {
    return this._callContextMethod("transform", args);
  }
  translateTransform(...args) {
    return this._callContextMethod("translate", args);
  }
  /**
   * Clears all drawing commands from the graphics context, effectively resetting it.
   * This includes clearing the current path, fill style, stroke style, and transformations.
   *
   * > [!NOTE] Graphics objects are not designed to be continuously cleared and redrawn.
   * > Instead, they are intended to be used for static or semi-static graphics that
   * > can be redrawn as needed. Frequent clearing and redrawing may lead to performance issues.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Draw some shapes
   * graphics
   *     .circle(100, 100, 50)
   *     .fill({ color: 0xff0000 })
   *     .rect(200, 100, 100, 50)
   *     .fill({ color: 0x00ff00 });
   *
   * // Clear all graphics
   * graphics.clear();
   *
   * // Start fresh with new shapes
   * graphics
   *     .circle(150, 150, 30)
   *     .fill({ color: 0x0000ff });
   * ```
   * @returns The Graphics instance for method chaining
   * @see {@link Graphics#beginPath} For starting a new path without clearing styles
   * @see {@link Graphics#save} For saving the current state
   * @see {@link Graphics#restore} For restoring a previous state
   */
  clear() {
    return this._callContextMethod("clear", []);
  }
  /**
   * Gets or sets the current fill style for the graphics context. The fill style determines
   * how shapes are filled when using the fill() method.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Basic color fill
   * graphics.fillStyle = {
   *     color: 0xff0000,  // Red
   *     alpha: 1
   * };
   *
   * // Using gradients
   * const gradient = new FillGradient({
   *     end: { x: 0, y: 1 }, // Vertical gradient
   *     stops: [
   *         { offset: 0, color: 0xff0000, alpha: 1 }, // Start color
   *         { offset: 1, color: 0x0000ff, alpha: 1 }  // End color
   *     ]
   * });
   *
   * graphics.fillStyle = {
   *     fill: gradient,
   *     alpha: 0.8
   * };
   *
   * // Using patterns
   * graphics.fillStyle = {
   *     texture: myTexture,
   *     alpha: 1,
   *     matrix: new Matrix()
   *         .scale(0.5, 0.5)
   *         .rotate(Math.PI / 4)
   * };
   * ```
   * @type {ConvertedFillStyle}
   * @see {@link FillStyle} For all available fill style options
   * @see {@link FillGradient} For creating gradient fills
   * @see {@link Graphics#fill} For applying the fill to paths
   */
  get fillStyle() {
    return this._context.fillStyle;
  }
  set fillStyle(value) {
    this._context.fillStyle = value;
  }
  /**
   * Gets or sets the current stroke style for the graphics context. The stroke style determines
   * how paths are outlined when using the stroke() method.
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Basic stroke style
   * graphics.strokeStyle = {
   *     width: 2,
   *     color: 0xff0000,
   *     alpha: 1
   * };
   *
   * // Using with gradients
   * const gradient = new FillGradient({
   *   end: { x: 0, y: 1 },
   *   stops: [
   *       { offset: 0, color: 0xff0000, alpha: 1 },
   *       { offset: 1, color: 0x0000ff, alpha: 1 }
   *   ]
   * });
   *
   * graphics.strokeStyle = {
   *     width: 4,
   *     fill: gradient,
   *     alignment: 0.5,
   *     join: 'round',
   *     cap: 'round'
   * };
   *
   * // Complex stroke settings
   * graphics.strokeStyle = {
   *     width: 6,
   *     color: 0x00ff00,
   *     alpha: 0.5,
   *     join: 'miter',
   *     miterLimit: 10,
   * };
   * ```
   * @see {@link StrokeStyle} For all available stroke style options
   * @see {@link Graphics#stroke} For applying the stroke to paths
   */
  get strokeStyle() {
    return this._context.strokeStyle;
  }
  set strokeStyle(value) {
    this._context.strokeStyle = value;
  }
  /**
   * Creates a new Graphics object that copies the current graphics content.
   * The clone can either share the same context (shallow clone) or have its own independent
   * context (deep clone).
   * @example
   * ```ts
   * const graphics = new Graphics();
   *
   * // Create original graphics content
   * graphics
   *     .circle(100, 100, 50)
   *     .fill({ color: 0xff0000 });
   *
   * // Create a shallow clone (shared context)
   * const shallowClone = graphics.clone();
   *
   * // Changes to original affect the clone
   * graphics
   *     .circle(200, 100, 30)
   *     .fill({ color: 0x00ff00 });
   *
   * // Create a deep clone (independent context)
   * const deepClone = graphics.clone(true);
   *
   * // Modify deep clone independently
   * deepClone
   *     .translateTransform(100, 100)
   *     .circle(0, 0, 40)
   *     .fill({ color: 0x0000ff });
   * ```
   * @param deep - Whether to create a deep clone of the graphics object.
   *              If false (default), the context will be shared between objects.
   *              If true, creates an independent copy of the context.
   * @returns A new Graphics instance with either shared or copied context
   * @see {@link Graphics#context} For accessing the underlying graphics context
   * @see {@link GraphicsContext} For understanding the shared context behavior
   */
  clone(deep = false) {
    if (deep) {
      return new Graphics(this._context.clone());
    }
    this._ownedContext = null;
    const clone = new Graphics(this._context);
    return clone;
  }
  // -------- v7 deprecations ---------
  /**
   * @param width
   * @param color
   * @param alpha
   * @deprecated since 8.0.0 Use {@link Graphics#setStrokeStyle} instead
   */
  lineStyle(width, color, alpha) {
    deprecation(v8_0_0, "Graphics#lineStyle is no longer needed. Use Graphics#setStrokeStyle to set the stroke style.");
    const strokeStyle = {};
    width && (strokeStyle.width = width);
    color && (strokeStyle.color = color);
    alpha && (strokeStyle.alpha = alpha);
    this.context.strokeStyle = strokeStyle;
    return this;
  }
  /**
   * @param color
   * @param alpha
   * @deprecated since 8.0.0 Use {@link Graphics#fill} instead
   */
  beginFill(color, alpha) {
    deprecation(v8_0_0, "Graphics#beginFill is no longer needed. Use Graphics#fill to fill the shape with the desired style.");
    const fillStyle = {};
    if (color !== void 0)
      fillStyle.color = color;
    if (alpha !== void 0)
      fillStyle.alpha = alpha;
    this.context.fillStyle = fillStyle;
    return this;
  }
  /**
   * @deprecated since 8.0.0 Use {@link Graphics#fill} instead
   */
  endFill() {
    deprecation(v8_0_0, "Graphics#endFill is no longer needed. Use Graphics#fill to fill the shape with the desired style.");
    this.context.fill();
    const strokeStyle = this.context.strokeStyle;
    if (strokeStyle.width !== GraphicsContext.defaultStrokeStyle.width || strokeStyle.color !== GraphicsContext.defaultStrokeStyle.color || strokeStyle.alpha !== GraphicsContext.defaultStrokeStyle.alpha) {
      this.context.stroke();
    }
    return this;
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#circle} instead
   */
  drawCircle(...args) {
    deprecation(v8_0_0, "Graphics#drawCircle has been renamed to Graphics#circle");
    return this._callContextMethod("circle", args);
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#ellipse} instead
   */
  drawEllipse(...args) {
    deprecation(v8_0_0, "Graphics#drawEllipse has been renamed to Graphics#ellipse");
    return this._callContextMethod("ellipse", args);
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#poly} instead
   */
  drawPolygon(...args) {
    deprecation(v8_0_0, "Graphics#drawPolygon has been renamed to Graphics#poly");
    return this._callContextMethod("poly", args);
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#rect} instead
   */
  drawRect(...args) {
    deprecation(v8_0_0, "Graphics#drawRect has been renamed to Graphics#rect");
    return this._callContextMethod("rect", args);
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#roundRect} instead
   */
  drawRoundedRect(...args) {
    deprecation(v8_0_0, "Graphics#drawRoundedRect has been renamed to Graphics#roundRect");
    return this._callContextMethod("roundRect", args);
  }
  /**
   * @param {...any} args
   * @deprecated since 8.0.0 Use {@link Graphics#star} instead
   */
  drawStar(...args) {
    deprecation(v8_0_0, "Graphics#drawStar has been renamed to Graphics#star");
    return this._callContextMethod("star", args);
  }
}
class RenderTexture extends Texture {
  /**
   * Creates a RenderTexture. Pass `dynamic: true` in options to allow resizing after creation.
   * @param options - Options for the RenderTexture, including width, height, and dynamic.
   * @returns A new RenderTexture instance.
   * @example
   * const rt = RenderTexture.create({ width: 100, height: 100, dynamic: true });
   * rt.resize(500, 500);
   */
  static create(options) {
    const { dynamic, ...rest } = options;
    return new RenderTexture({
      source: new TextureSource(rest),
      dynamic: dynamic ?? false
    });
  }
  /**
   * Resizes the render texture.
   * @param width - The new width of the render texture.
   * @param height - The new height of the render texture.
   * @param resolution - The new resolution of the render texture.
   * @returns This texture.
   */
  resize(width, height, resolution) {
    this.source.resize(width, height, resolution);
    return this;
  }
}
var DEPRECATED_WRAP_MODES = /* @__PURE__ */ ((DEPRECATED_WRAP_MODES2) => {
  DEPRECATED_WRAP_MODES2["CLAMP"] = "clamp-to-edge";
  DEPRECATED_WRAP_MODES2["REPEAT"] = "repeat";
  DEPRECATED_WRAP_MODES2["MIRRORED_REPEAT"] = "mirror-repeat";
  return DEPRECATED_WRAP_MODES2;
})(DEPRECATED_WRAP_MODES || {});
new Proxy(DEPRECATED_WRAP_MODES, {
  get(target, prop) {
    deprecation(v8_0_0, `DRAW_MODES.${prop} is deprecated, use '${DEPRECATED_WRAP_MODES[prop]}' instead`);
    return target[prop];
  }
});
var DEPRECATED_SCALE_MODES = /* @__PURE__ */ ((DEPRECATED_SCALE_MODES2) => {
  DEPRECATED_SCALE_MODES2["NEAREST"] = "nearest";
  DEPRECATED_SCALE_MODES2["LINEAR"] = "linear";
  return DEPRECATED_SCALE_MODES2;
})(DEPRECATED_SCALE_MODES || {});
new Proxy(DEPRECATED_SCALE_MODES, {
  get(target, prop) {
    deprecation(v8_0_0, `DRAW_MODES.${prop} is deprecated, use '${DEPRECATED_SCALE_MODES[prop]}' instead`);
    return target[prop];
  }
});
extensions.add(browserExt, webworkerExt);
var vertex = "in vec2 aPosition;\nout vec2 vTextureCoord;\n\nuniform vec4 uInputSize;\nuniform vec4 uOutputFrame;\nuniform vec4 uOutputTexture;\n\nvec4 filterVertexPosition( void )\n{\n    vec2 position = aPosition * uOutputFrame.zw + uOutputFrame.xy;\n    \n    position.x = position.x * (2.0 / uOutputTexture.x) - 1.0;\n    position.y = position.y * (2.0*uOutputTexture.z / uOutputTexture.y) - uOutputTexture.z;\n\n    return vec4(position, 0.0, 1.0);\n}\n\nvec2 filterTextureCoord( void )\n{\n    return aPosition * (uOutputFrame.zw * uInputSize.zw);\n}\n\nvoid main(void)\n{\n    gl_Position = filterVertexPosition();\n    vTextureCoord = filterTextureCoord();\n}\n";
var wgslVertex = "struct GlobalFilterUniforms {\n  uInputSize:vec4<f32>,\n  uInputPixel:vec4<f32>,\n  uInputClamp:vec4<f32>,\n  uOutputFrame:vec4<f32>,\n  uGlobalFrame:vec4<f32>,\n  uOutputTexture:vec4<f32>,\n};\n\n@group(0) @binding(0) var<uniform> gfu: GlobalFilterUniforms;\n\nstruct VSOutput {\n    @builtin(position) position: vec4<f32>,\n    @location(0) uv : vec2<f32>\n  };\n\nfn filterVertexPosition(aPosition:vec2<f32>) -> vec4<f32>\n{\n    var position = aPosition * gfu.uOutputFrame.zw + gfu.uOutputFrame.xy;\n\n    position.x = position.x * (2.0 / gfu.uOutputTexture.x) - 1.0;\n    position.y = position.y * (2.0*gfu.uOutputTexture.z / gfu.uOutputTexture.y) - gfu.uOutputTexture.z;\n\n    return vec4(position, 0.0, 1.0);\n}\n\nfn filterTextureCoord( aPosition:vec2<f32> ) -> vec2<f32>\n{\n    return aPosition * (gfu.uOutputFrame.zw * gfu.uInputSize.zw);\n}\n\nfn globalTextureCoord( aPosition:vec2<f32> ) -> vec2<f32>\n{\n  return  (aPosition.xy / gfu.uGlobalFrame.zw) + (gfu.uGlobalFrame.xy / gfu.uGlobalFrame.zw);  \n}\n\nfn getSize() -> vec2<f32>\n{\n  return gfu.uGlobalFrame.zw;\n}\n  \n@vertex\nfn mainVertex(\n  @location(0) aPosition : vec2<f32>, \n) -> VSOutput {\n  return VSOutput(\n   filterVertexPosition(aPosition),\n   filterTextureCoord(aPosition)\n  );\n}";
var fragment$1 = "\nin vec2 vTextureCoord;\nout vec4 finalColor;\n\nuniform sampler2D uTexture;\nuniform vec2 uOffset;\n\nvoid main(void)\n{\n    vec4 color = vec4(0.0);\n\n    // Sample top left pixel\n    color += texture(uTexture, vec2(vTextureCoord.x - uOffset.x, vTextureCoord.y + uOffset.y));\n\n    // Sample top right pixel\n    color += texture(uTexture, vec2(vTextureCoord.x + uOffset.x, vTextureCoord.y + uOffset.y));\n\n    // Sample bottom right pixel\n    color += texture(uTexture, vec2(vTextureCoord.x + uOffset.x, vTextureCoord.y - uOffset.y));\n\n    // Sample bottom left pixel\n    color += texture(uTexture, vec2(vTextureCoord.x - uOffset.x, vTextureCoord.y - uOffset.y));\n\n    // Average\n    color *= 0.25;\n\n    finalColor = color;\n}";
var source$1 = "struct KawaseBlurUniforms {\n  uOffset:vec2<f32>,\n};\n\n@group(0) @binding(1) var uTexture: texture_2d<f32>; \n@group(0) @binding(2) var uSampler: sampler;\n@group(1) @binding(0) var<uniform> kawaseBlurUniforms : KawaseBlurUniforms;\n\n@fragment\nfn mainFragment(\n  @builtin(position) position: vec4<f32>,\n  @location(0) uv : vec2<f32>\n) -> @location(0) vec4<f32> {\n  let uOffset = kawaseBlurUniforms.uOffset;\n  var color: vec4<f32> = vec4<f32>(0.0);\n\n  // Sample top left pixel\n  color += textureSample(uTexture, uSampler, vec2<f32>(uv.x - uOffset.x, uv.y + uOffset.y));\n  // Sample top right pixel\n  color += textureSample(uTexture, uSampler, vec2<f32>(uv.x + uOffset.x, uv.y + uOffset.y));\n  // Sample bottom right pixel\n  color += textureSample(uTexture, uSampler, vec2<f32>(uv.x + uOffset.x, uv.y - uOffset.y));\n  // Sample bottom left pixel\n  color += textureSample(uTexture, uSampler, vec2<f32>(uv.x - uOffset.x, uv.y - uOffset.y));\n  // Average\n  color *= 0.25;\n\n  return color;\n}";
var fragmentClamp = "\nprecision highp float;\nin vec2 vTextureCoord;\nout vec4 finalColor;\n\nuniform sampler2D uTexture;\nuniform vec2 uOffset;\n\nuniform vec4 uInputClamp;\n\nvoid main(void)\n{\n    vec4 color = vec4(0.0);\n\n    // Sample top left pixel\n    color += texture(uTexture, clamp(vec2(vTextureCoord.x - uOffset.x, vTextureCoord.y + uOffset.y), uInputClamp.xy, uInputClamp.zw));\n\n    // Sample top right pixel\n    color += texture(uTexture, clamp(vec2(vTextureCoord.x + uOffset.x, vTextureCoord.y + uOffset.y), uInputClamp.xy, uInputClamp.zw));\n\n    // Sample bottom right pixel\n    color += texture(uTexture, clamp(vec2(vTextureCoord.x + uOffset.x, vTextureCoord.y - uOffset.y), uInputClamp.xy, uInputClamp.zw));\n\n    // Sample bottom left pixel\n    color += texture(uTexture, clamp(vec2(vTextureCoord.x - uOffset.x, vTextureCoord.y - uOffset.y), uInputClamp.xy, uInputClamp.zw));\n\n    // Average\n    color *= 0.25;\n\n    finalColor = color;\n}\n";
var sourceClamp = "struct KawaseBlurUniforms {\n  uOffset:vec2<f32>,\n};\n\nstruct GlobalFilterUniforms {\n  uInputSize:vec4<f32>,\n  uInputPixel:vec4<f32>,\n  uInputClamp:vec4<f32>,\n  uOutputFrame:vec4<f32>,\n  uGlobalFrame:vec4<f32>,\n  uOutputTexture:vec4<f32>,\n};\n\n@group(0) @binding(0) var<uniform> gfu: GlobalFilterUniforms;\n\n@group(0) @binding(1) var uTexture: texture_2d<f32>; \n@group(0) @binding(2) var uSampler: sampler;\n@group(1) @binding(0) var<uniform> kawaseBlurUniforms : KawaseBlurUniforms;\n\n@fragment\nfn mainFragment(\n  @builtin(position) position: vec4<f32>,\n  @location(0) uv : vec2<f32>\n) -> @location(0) vec4<f32> {\n  let uOffset = kawaseBlurUniforms.uOffset;\n  var color: vec4<f32> = vec4(0.0);\n\n  // Sample top left pixel\n  color += textureSample(uTexture, uSampler, clamp(vec2<f32>(uv.x - uOffset.x, uv.y + uOffset.y), gfu.uInputClamp.xy, gfu.uInputClamp.zw));\n  // Sample top right pixel\n  color += textureSample(uTexture, uSampler, clamp(vec2<f32>(uv.x + uOffset.x, uv.y + uOffset.y), gfu.uInputClamp.xy, gfu.uInputClamp.zw));\n  // Sample bottom right pixel\n  color += textureSample(uTexture, uSampler, clamp(vec2<f32>(uv.x + uOffset.x, uv.y - uOffset.y), gfu.uInputClamp.xy, gfu.uInputClamp.zw));\n  // Sample bottom left pixel\n  color += textureSample(uTexture, uSampler, clamp(vec2<f32>(uv.x - uOffset.x, uv.y - uOffset.y), gfu.uInputClamp.xy, gfu.uInputClamp.zw));\n  // Average\n  color *= 0.25;\n    \n  return color;\n}";
var __defProp$1 = Object.defineProperty;
var __defNormalProp$1 = (obj, key, value) => key in obj ? __defProp$1(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField$1 = (obj, key, value) => {
  __defNormalProp$1(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
const _KawaseBlurFilter = class _KawaseBlurFilter2 extends Filter {
  /** @ignore */
  constructor(...args) {
    let options = args[0] ?? {};
    if (typeof options === "number" || Array.isArray(options)) {
      deprecation("6.0.0", "KawaseBlurFilter constructor params are now options object. See params: { strength, quality, clamp, pixelSize }");
      options = { strength: options };
      if (args[1] !== void 0)
        options.quality = args[1];
      if (args[2] !== void 0)
        options.clamp = args[2];
    }
    options = { ..._KawaseBlurFilter2.DEFAULT_OPTIONS, ...options };
    const gpuProgram = GpuProgram.from({
      vertex: {
        source: wgslVertex,
        entryPoint: "mainVertex"
      },
      fragment: {
        source: options?.clamp ? sourceClamp : source$1,
        entryPoint: "mainFragment"
      }
    });
    const glProgram = GlProgram.from({
      vertex,
      fragment: options?.clamp ? fragmentClamp : fragment$1,
      name: "kawase-blur-filter"
    });
    super({
      gpuProgram,
      glProgram,
      resources: {
        kawaseBlurUniforms: {
          uOffset: { value: new Float32Array(2), type: "vec2<f32>" }
        }
      }
    });
    __publicField$1(this, "uniforms");
    __publicField$1(this, "_pixelSize", { x: 0, y: 0 });
    __publicField$1(this, "_clamp");
    __publicField$1(this, "_kernels", []);
    __publicField$1(this, "_blur");
    __publicField$1(this, "_quality");
    this.uniforms = this.resources.kawaseBlurUniforms.uniforms;
    this.pixelSize = options.pixelSize ?? { x: 1, y: 1 };
    if (Array.isArray(options.strength)) {
      this.kernels = options.strength;
    } else if (typeof options.strength === "number") {
      this._blur = options.strength;
      this.quality = options.quality ?? 3;
    }
    this._clamp = !!options.clamp;
  }
  /**
   * Override existing apply method in `Filter`
   * @override
   * @ignore
   */
  apply(filterManager, input, output, clearMode) {
    const uvX = this.pixelSizeX / input.source.width;
    const uvY = this.pixelSizeY / input.source.height;
    let offset;
    if (this._quality === 1 || this._blur === 0) {
      offset = this._kernels[0] + 0.5;
      this.uniforms.uOffset[0] = offset * uvX;
      this.uniforms.uOffset[1] = offset * uvY;
      filterManager.applyFilter(this, input, output, clearMode);
    } else {
      const renderTarget = TexturePool.getSameSizeTexture(input);
      let source2 = input;
      let target = renderTarget;
      let tmp;
      const last = this._quality - 1;
      for (let i2 = 0; i2 < last; i2++) {
        offset = this._kernels[i2] + 0.5;
        this.uniforms.uOffset[0] = offset * uvX;
        this.uniforms.uOffset[1] = offset * uvY;
        filterManager.applyFilter(this, source2, target, true);
        tmp = source2;
        source2 = target;
        target = tmp;
      }
      offset = this._kernels[last] + 0.5;
      this.uniforms.uOffset[0] = offset * uvX;
      this.uniforms.uOffset[1] = offset * uvY;
      filterManager.applyFilter(this, source2, output, clearMode);
      TexturePool.returnTexture(renderTarget);
    }
  }
  /**
    * The amount of blur, value greater than `0`.
    * @default 4
    */
  get strength() {
    return this._blur;
  }
  set strength(value) {
    this._blur = value;
    this._generateKernels();
  }
  /**
    * The quality of the filter, integer greater than `1`.
    * @default 3
    */
  get quality() {
    return this._quality;
  }
  set quality(value) {
    this._quality = Math.max(1, Math.round(value));
    this._generateKernels();
  }
  /**
    * The kernel size of the blur filter, for advanced usage
    * @default [0]
    */
  get kernels() {
    return this._kernels;
  }
  set kernels(value) {
    if (Array.isArray(value) && value.length > 0) {
      this._kernels = value;
      this._quality = value.length;
      this._blur = Math.max(...value);
    } else {
      this._kernels = [0];
      this._quality = 1;
    }
  }
  /**
    * The size of the pixels. Large size is blurrier. For advanced usage.
    * @default {x:1,y:1}
    */
  get pixelSize() {
    return this._pixelSize;
  }
  set pixelSize(value) {
    if (typeof value === "number") {
      this.pixelSizeX = this.pixelSizeY = value;
      return;
    }
    if (Array.isArray(value)) {
      this.pixelSizeX = value[0];
      this.pixelSizeY = value[1];
      return;
    }
    this._pixelSize = value;
  }
  /**
    * The size of the pixels on the `x` axis. Large size is blurrier. For advanced usage.
    * @default 1
    */
  get pixelSizeX() {
    return this.pixelSize.x;
  }
  set pixelSizeX(value) {
    this.pixelSize.x = value;
  }
  /**
    * The size of the pixels on the `y` axis. Large size is blurrier. For advanced usage.
    * @default 1
    */
  get pixelSizeY() {
    return this.pixelSize.y;
  }
  set pixelSizeY(value) {
    this.pixelSize.y = value;
  }
  /**
    * Get the if the filter is clamped
    * @default false
    */
  get clamp() {
    return this._clamp;
  }
  /** Update padding based on kernel data */
  _updatePadding() {
    this.padding = Math.ceil(this._kernels.reduce((acc, v2) => acc + v2 + 0.5, 0));
  }
  /** Auto generate kernels by blur & quality */
  _generateKernels() {
    const blur = this._blur;
    const quality = this._quality;
    const kernels = [blur];
    if (blur > 0) {
      let k2 = blur;
      const step = blur / quality;
      for (let i2 = 1; i2 < quality; i2++) {
        k2 -= step;
        kernels.push(k2);
      }
    }
    this._kernels = kernels;
    this._updatePadding();
  }
};
__publicField$1(_KawaseBlurFilter, "DEFAULT_OPTIONS", {
  strength: 4,
  quality: 3,
  clamp: false,
  pixelSize: { x: 1, y: 1 }
});
let KawaseBlurFilter = _KawaseBlurFilter;
var fragment = "precision highp float;\nin vec2 vTextureCoord;\nout vec4 finalColor;\n\nuniform sampler2D uTexture;\nuniform float uAlpha;\nuniform vec3 uColor;\nuniform vec2 uOffset;\n\nuniform vec4 uInputSize;\n\nvoid main(void){\n    vec4 sample = texture(uTexture, vTextureCoord - uOffset * uInputSize.zw);\n\n    // Premultiply alpha\n    sample.rgb = uColor.rgb * sample.a;\n\n    // alpha user alpha\n    sample *= uAlpha;\n\n    finalColor = sample;\n}";
var source = "struct DropShadowUniforms {\n  uAlpha: f32,\n  uColor: vec3<f32>,\n  uOffset: vec2<f32>,\n};\n\nstruct GlobalFilterUniforms {\n  uInputSize:vec4<f32>,\n  uInputPixel:vec4<f32>,\n  uInputClamp:vec4<f32>,\n  uOutputFrame:vec4<f32>,\n  uGlobalFrame:vec4<f32>,\n  uOutputTexture:vec4<f32>,\n};\n\n@group(0) @binding(0) var<uniform> gfu: GlobalFilterUniforms;\n\n@group(0) @binding(1) var uTexture: texture_2d<f32>; \n@group(0) @binding(2) var uSampler: sampler;\n@group(1) @binding(0) var<uniform> dropShadowUniforms : DropShadowUniforms;\n\n@fragment\nfn mainFragment(\n  @builtin(position) position: vec4<f32>,\n  @location(0) uv : vec2<f32>\n) -> @location(0) vec4<f32> {\n  var color: vec4<f32> = textureSample(uTexture, uSampler, uv - dropShadowUniforms.uOffset * gfu.uInputSize.zw);\n\n  // Premultiply alpha\n  color = vec4<f32>(vec3<f32>(dropShadowUniforms.uColor.rgb * color.a), color.a);\n  // alpha user alpha\n  color *= dropShadowUniforms.uAlpha;\n\n  return color;\n}";
var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
const _DropShadowFilter = class _DropShadowFilter2 extends Filter {
  /**
   * @param options - Options for the DropShadowFilter constructor.
   */
  constructor(options) {
    options = { ..._DropShadowFilter2.DEFAULT_OPTIONS, ...options };
    const gpuProgram = GpuProgram.from({
      vertex: {
        source: wgslVertex,
        entryPoint: "mainVertex"
      },
      fragment: {
        source,
        entryPoint: "mainFragment"
      }
    });
    const glProgram = GlProgram.from({
      vertex,
      fragment,
      name: "drop-shadow-filter"
    });
    super({
      gpuProgram,
      glProgram,
      resources: {
        dropShadowUniforms: {
          uAlpha: { value: options.alpha, type: "f32" },
          uColor: { value: new Float32Array(3), type: "vec3<f32>" },
          uOffset: { value: options.offset, type: "vec2<f32>" }
        }
      },
      resolution: options.resolution
    });
    __publicField(this, "uniforms");
    __publicField(this, "shadowOnly", false);
    __publicField(this, "_color");
    __publicField(this, "_blurFilter");
    __publicField(this, "_basePass");
    this.uniforms = this.resources.dropShadowUniforms.uniforms;
    this._color = new Color();
    this.color = options.color ?? 0;
    this._blurFilter = new KawaseBlurFilter({
      strength: options.kernels ?? options.blur,
      quality: options.kernels ? void 0 : options.quality
    });
    this._basePass = new Filter({
      gpuProgram: GpuProgram.from({
        vertex: {
          source: wgslVertex,
          entryPoint: "mainVertex"
        },
        fragment: {
          source: `
                    @group(0) @binding(1) var uTexture: texture_2d<f32>; 
                    @group(0) @binding(2) var uSampler: sampler;
                    @fragment
                    fn mainFragment(
                        @builtin(position) position: vec4<f32>,
                        @location(0) uv : vec2<f32>
                    ) -> @location(0) vec4<f32> {
                        return textureSample(uTexture, uSampler, uv);
                    }
                    `,
          entryPoint: "mainFragment"
        }
      }),
      glProgram: GlProgram.from({
        vertex,
        fragment: `
                in vec2 vTextureCoord;
                out vec4 finalColor;
                uniform sampler2D uTexture;

                void main(void){
                    finalColor = texture(uTexture, vTextureCoord);
                }
                `,
        name: "drop-shadow-filter"
      }),
      resources: {}
    });
    Object.assign(this, options);
  }
  /**
   * Override existing apply method in `Filter`
   * @override
   * @ignore
   */
  apply(filterManager, input, output, clearMode) {
    const renderTarget = TexturePool.getSameSizeTexture(input);
    filterManager.applyFilter(this, input, renderTarget, true);
    this._blurFilter.apply(filterManager, renderTarget, output, clearMode);
    if (!this.shadowOnly) {
      filterManager.applyFilter(this._basePass, input, output, false);
    }
    TexturePool.returnTexture(renderTarget);
  }
  /**
   * Set the offset position of the drop-shadow relative to the original image.
   * @default [4,4]
   */
  get offset() {
    return this.uniforms.uOffset;
  }
  set offset(value) {
    this.uniforms.uOffset = value;
    this._updatePadding();
  }
  /**
   * Set the offset position of the drop-shadow relative to the original image on the `x` axis
   * @default 4
   */
  get offsetX() {
    return this.offset.x;
  }
  set offsetX(value) {
    this.offset.x = value;
    this._updatePadding();
  }
  /**
   * Set the offset position of the drop-shadow relative to the original image on the `y` axis
   * @default 4
   */
  get offsetY() {
    return this.offset.y;
  }
  set offsetY(value) {
    this.offset.y = value;
    this._updatePadding();
  }
  /**
   * The color value of shadow.
   * @example [0.0, 0.0, 0.0] = 0x000000
   * @default 0x000000
   */
  get color() {
    return this._color.value;
  }
  set color(value) {
    this._color.setValue(value);
    const [r2, g2, b2] = this._color.toArray();
    this.uniforms.uColor[0] = r2;
    this.uniforms.uColor[1] = g2;
    this.uniforms.uColor[2] = b2;
  }
  /**
   * Coefficient for alpha multiplication
   * @default 1
   */
  get alpha() {
    return this.uniforms.uAlpha;
  }
  set alpha(value) {
    this.uniforms.uAlpha = value;
  }
  /**
   * The strength of the shadow's blur.
   * @default 2
   */
  get blur() {
    return this._blurFilter.strength;
  }
  set blur(value) {
    this._blurFilter.strength = value;
    this._updatePadding();
  }
  /**
   * Sets the quality of the Blur Filter
   * @default 4
   */
  get quality() {
    return this._blurFilter.quality;
  }
  set quality(value) {
    this._blurFilter.quality = value;
    this._updatePadding();
  }
  /** Sets the kernels of the Blur Filter */
  get kernels() {
    return this._blurFilter.kernels;
  }
  set kernels(value) {
    this._blurFilter.kernels = value;
  }
  /**
   * Sets the pixelSize of the Kawase Blur filter
   * @default [1,1]
   */
  get pixelSize() {
    return this._blurFilter.pixelSize;
  }
  set pixelSize(value) {
    if (typeof value === "number") {
      value = { x: value, y: value };
    }
    if (Array.isArray(value)) {
      value = { x: value[0], y: value[1] };
    }
    this._blurFilter.pixelSize = value;
  }
  /**
   * Sets the pixelSize of the Kawase Blur filter on the `x` axis
   * @default 1
   */
  get pixelSizeX() {
    return this._blurFilter.pixelSizeX;
  }
  set pixelSizeX(value) {
    this._blurFilter.pixelSizeX = value;
  }
  /**
   * Sets the pixelSize of the Kawase Blur filter on the `y` axis
   * @default 1
   */
  get pixelSizeY() {
    return this._blurFilter.pixelSizeY;
  }
  set pixelSizeY(value) {
    this._blurFilter.pixelSizeY = value;
  }
  /**
   * Recalculate the proper padding amount.
   * @private
   */
  _updatePadding() {
    const offsetPadding = Math.max(
      Math.abs(this.offsetX),
      Math.abs(this.offsetY)
    );
    this.padding = offsetPadding + this.blur * 2 + this.quality * 4;
  }
};
__publicField(_DropShadowFilter, "DEFAULT_OPTIONS", {
  offset: { x: 4, y: 4 },
  color: 0,
  alpha: 0.5,
  shadowOnly: false,
  kernels: void 0,
  blur: 2,
  quality: 3,
  pixelSize: { x: 1, y: 1 },
  resolution: 1
});
function ImageEditor_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const antialias = true;
    let changeable = fallback($$props["changeable"], false);
    let sources2 = fallback($$props["sources"], () => ["upload", "webcam", "clipboard"], true);
    let transforms = fallback($$props["transforms"], () => ["crop", "resize"], true);
    let canvas_size = $$props["canvas_size"];
    let is_dragging = fallback($$props["is_dragging"], false);
    let background_image = fallback($$props["background_image"], false);
    let brush_options = $$props["brush_options"];
    let eraser_options = $$props["eraser_options"];
    let fixed_canvas = fallback($$props["fixed_canvas"], false);
    let root = $$props["root"];
    let i18n = $$props["i18n"];
    let upload = $$props["upload"];
    let composite = $$props["composite"];
    let layers = $$props["layers"];
    let background = $$props["background"];
    let border_region = fallback($$props["border_region"], 0);
    let layer_options = $$props["layer_options"];
    let current_tool = $$props["current_tool"];
    let webcam_options = $$props["webcam_options"];
    let show_download_button = fallback($$props["show_download_button"], false);
    let theme_mode = $$props["theme_mode"];
    let full_history = fallback($$props["full_history"], null);
    let has_drawn = fallback($$props["has_drawn"], false);
    async function get_blobs() {
      return { background: null, layers: [], composite: null };
    }
    let editor;
    function add_image(image) {
      editor.add_image({ image });
    }
    async function add_image_from_url(source2) {
      return;
    }
    async function add_layers_from_url(source2) {
      return;
    }
    let crop;
    let crop_zoom;
    let can_undo = fallback($$props["can_undo"], false);
    function handle_tool_change({ tool }) {
      editor.set_tool(tool);
      current_tool = tool;
    }
    function handle_subtool_change({ tool, subtool }) {
      editor.set_subtool(subtool);
      current_subtool = subtool;
      {
        return;
      }
    }
    function update_brush_options() {
      const default_color = brush_options.default_color === "auto" ? brush_options.colors[0] : brush_options.default_color;
      if (Array.isArray(default_color)) {
        default_color[0];
        default_color[1];
      } else {
        const color = tinycolor(default_color);
        if (color.getAlpha() < 1) {
          color.getAlpha();
        }
      }
      typeof brush_options.default_size === "number" ? brush_options.default_size : 25;
    }
    function update_eraser_options() {
      eraser_options.default_size === "auto" ? 25 : eraser_options.default_size;
    }
    let current_subtool = null;
    async function handle_crop_confirm() {
      const { image } = await crop.get_crop_bounds();
      if (!image) return;
      await editor.add_image({ image, resize: false });
      handle_subtool_change({ tool: "image", subtool: null });
    }
    let old_background = background;
    if (background == null && layers.length == 0 && composite == null && editor) ;
    if (old_background !== background) {
      handle_tool_change({ tool: "draw" });
      handle_subtool_change({ tool: "draw", subtool: null });
      current_tool = "draw";
      old_background = background;
    }
    current_tool === "image" && current_subtool === "crop" && crop_zoom.set_zoom("fit");
    background_image = can_undo && editor.command_manager.contains("AddImage");
    {
      if (brush_options) {
        update_brush_options();
      }
      if (eraser_options) {
        update_eraser_options();
      }
    }
    current_tool !== "image" || current_tool === "image" && background_image || current_tool === "image" && current_subtool === "webcam" || !sources2.includes("upload");
    add_image_from_url();
    add_layers_from_url();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div data-testid="image"${attr_class("image-container svelte-6fnl1z", void 0, { "dark-bg": current_subtool === "crop" })} aria-label="Click to upload or drop files" aria-dropeffect="copy">`);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div${attr_class("pixi-target svelte-6fnl1z", void 0, { "visible": current_subtool !== "crop" })}></div> <div${attr_class("pixi-target-crop svelte-6fnl1z", void 0, { "visible": current_subtool === "crop" })}></div> `);
      if (current_subtool === "crop") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="crop-confirm-button svelte-6fnl1z">`);
        IconButton($$renderer3, {
          Icon: Check,
          label: "Confirm crop",
          show_label: true,
          size: "large",
          padded: true,
          color: "white",
          background: "var(--color-green-500)",
          label_position: "right",
          onclick: handle_crop_confirm
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <!--[-->`);
      slot($$renderer3, $$props, "default", {});
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      changeable,
      sources: sources2,
      transforms,
      canvas_size,
      is_dragging,
      background_image,
      brush_options,
      eraser_options,
      fixed_canvas,
      root,
      i18n,
      upload,
      composite,
      layers,
      background,
      border_region,
      layer_options,
      current_tool,
      webcam_options,
      show_download_button,
      theme_mode,
      full_history,
      has_drawn,
      can_undo,
      antialias,
      get_blobs,
      add_image,
      add_image_from_url,
      add_layers_from_url
    });
  });
}
const RE_HEADING = /^(#\s*)(.+)$/m;
function inject(text) {
  const trimmed_text = text.trim();
  const heading_match = trimmed_text.match(RE_HEADING);
  if (!heading_match) {
    return [false, trimmed_text || false];
  }
  const [full_match, , heading_content] = heading_match;
  const _heading = heading_content.trim();
  if (trimmed_text === full_match) {
    return [_heading, false];
  }
  const heading_end_index = heading_match.index !== void 0 ? heading_match.index + full_match.length : 0;
  const remaining_text = trimmed_text.substring(heading_end_index).trim();
  const _paragraph = remaining_text || false;
  return [_heading, _paragraph];
}
function InteractiveImageEditor($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let heading, paragraph;
    let brush = $$props["brush"];
    let eraser = $$props["eraser"];
    let sources2 = $$props["sources"];
    let i18n = $$props["i18n"];
    let root = $$props["root"];
    let label = fallback($$props["label"], void 0);
    let show_label = $$props["show_label"];
    let changeable = fallback($$props["changeable"], false);
    let theme_mode = $$props["theme_mode"];
    let layers = $$props["layers"];
    let composite = $$props["composite"];
    let background = $$props["background"];
    let layer_options = $$props["layer_options"];
    let transforms = $$props["transforms"];
    let accept_blobs = $$props["accept_blobs"];
    let canvas_size = $$props["canvas_size"];
    let fixed_canvas = fallback($$props["fixed_canvas"], false);
    let realtime = $$props["realtime"];
    let upload = $$props["upload"];
    let is_dragging = $$props["is_dragging"];
    let placeholder = fallback($$props["placeholder"], void 0);
    let border_region = $$props["border_region"];
    let full_history = fallback($$props["full_history"], null);
    let webcam_options = $$props["webcam_options"];
    let show_download_button = fallback($$props["show_download_button"], false);
    let editor;
    let has_drawn = false;
    function is_not_null(o2) {
      return !!o2;
    }
    function is_file_data(o2) {
      return !!o2;
    }
    async function get_data() {
      let blobs;
      try {
        blobs = await editor.get_blobs();
      } catch (e2) {
        return { background: null, layers: [], composite: null };
      }
      const bg = blobs.background ? upload(await prepare_files([new File([blobs.background], "background.png")]), root) : Promise.resolve(null);
      const layers2 = blobs.layers.filter(is_not_null).map(async (blob, i2) => upload(await prepare_files([new File([blob], `layer_${i2}.png`)]), root));
      const composite2 = blobs.composite ? upload(await prepare_files([new File([blobs.composite], "composite.png")]), root) : Promise.resolve(null);
      const [background2, composite_, ...layers_] = await Promise.all([bg, composite2, ...layers2]);
      return {
        background: Array.isArray(background2) ? background2[0] : background2,
        layers: layers_.flatMap((layer) => Array.isArray(layer) ? layer : [layer]).filter(is_file_data),
        composite: Array.isArray(composite_) ? composite_[0] : composite_
      };
    }
    let background_image = false;
    let can_undo;
    let image_id = fallback($$props["image_id"], null);
    let current_tool;
    [heading, paragraph] = placeholder ? inject(placeholder) : [false, false];
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, {
        show_label,
        Icon: Image$1,
        label: label || i18n("image.image")
      });
      $$renderer3.push(`<!----> `);
      ImageEditor_1($$renderer3, {
        transforms,
        composite,
        layers,
        background,
        canvas_size,
        changeable,
        sources: sources2,
        brush_options: brush,
        eraser_options: eraser,
        fixed_canvas,
        border_region,
        layer_options,
        i18n,
        root,
        upload,
        webcam_options,
        show_download_button,
        theme_mode,
        get background_image() {
          return background_image;
        },
        set background_image($$value) {
          background_image = $$value;
          $$settled = false;
        },
        get current_tool() {
          return current_tool;
        },
        set current_tool($$value) {
          current_tool = $$value;
          $$settled = false;
        },
        get is_dragging() {
          return is_dragging;
        },
        set is_dragging($$value) {
          is_dragging = $$value;
          $$settled = false;
        },
        get has_drawn() {
          return has_drawn;
        },
        set has_drawn($$value) {
          has_drawn = $$value;
          $$settled = false;
        },
        get can_undo() {
          return can_undo;
        },
        set can_undo($$value) {
          can_undo = $$value;
          $$settled = false;
        },
        get full_history() {
          return full_history;
        },
        set full_history($$value) {
          full_history = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          if (current_tool === "image" && !can_undo) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<div class="empty wrap svelte-b4yllx">`);
            if (sources2 && sources2.length) {
              $$renderer4.push("<!--[-->");
              if (heading || paragraph) {
                $$renderer4.push("<!--[-->");
                if (heading) {
                  $$renderer4.push("<!--[-->");
                  $$renderer4.push(`<h2 class="svelte-b4yllx">${escape_html(heading)}</h2>`);
                } else {
                  $$renderer4.push("<!--[!-->");
                }
                $$renderer4.push(`<!--]--> `);
                if (paragraph) {
                  $$renderer4.push("<!--[-->");
                  $$renderer4.push(`<p class="svelte-b4yllx">${escape_html(paragraph)}</p>`);
                } else {
                  $$renderer4.push("<!--[!-->");
                }
                $$renderer4.push(`<!--]-->`);
              } else {
                $$renderer4.push("<!--[!-->");
                $$renderer4.push(`<div>Upload an image</div>`);
              }
              $$renderer4.push(`<!--]-->`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            if (sources2 && sources2.length && brush && !placeholder) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<div class="or svelte-b4yllx">or</div>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            if (brush && !placeholder) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<div>select the draw tool to start</div>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--></div>`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!---->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      brush,
      eraser,
      sources: sources2,
      i18n,
      root,
      label,
      show_label,
      changeable,
      theme_mode,
      layers,
      composite,
      background,
      layer_options,
      transforms,
      accept_blobs,
      canvas_size,
      fixed_canvas,
      realtime,
      upload,
      is_dragging,
      placeholder,
      border_region,
      full_history,
      webcam_options,
      show_download_button,
      image_id,
      get_data
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    class ImageEditorGradio extends Gradio {
      async get_data() {
        const data = await super.get_data();
        const value = await _get_value();
        return { ...data, value };
      }
    }
    const gradio = new ImageEditorGradio(props, {
      server: { accept_blobs: () => {
      } },
      buttons: [],
      height: 350,
      border_region: 0
    });
    let editor_instance;
    let image_id = null;
    async function _get_value() {
      if (image_id) {
        const val = { id: image_id };
        image_id = null;
        return val;
      }
      const blobs = await editor_instance.get_data();
      return blobs;
    }
    let is_dragging;
    let has_value = gradio.props.value?.background || gradio.props.value?.layers?.length || gradio.props.value?.composite;
    let normalised_background = gradio.props.value?.background ? new FileData(gradio.props.value.background) : null;
    let normalised_composite = gradio.props.value?.composite ? new FileData(gradio.props.value.composite) : null;
    let normalised_layers = gradio.props.value?.layers?.map((layer) => new FileData(layer)) || [];
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: "solid",
          border_mode: is_dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height,
          width: gradio.props.width,
          allow_overflow: true,
          overflow_behavior: "visible",
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            ImagePreview($$renderer4, {
              value: gradio.props.value?.composite || null,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              buttons: gradio.props.buttons,
              selectable: gradio.props._selectable,
              i18n: gradio.i18n
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: has_value ? "solid" : "dashed",
          border_mode: is_dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height,
          width: gradio.props.width,
          allow_overflow: true,
          overflow_behavior: "visible",
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status,
              {
                on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
              }
            ]));
            $$renderer4.push(`<!----> `);
            InteractiveImageEditor($$renderer4, {
              border_region: gradio.props.border_region,
              canvas_size: gradio.props.canvas_size,
              layers: normalised_layers,
              composite: normalised_composite,
              background: normalised_background,
              root: gradio.shared.root,
              sources: gradio.props.sources,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              fixed_canvas: gradio.props.fixed_canvas,
              brush: gradio.props.brush,
              eraser: gradio.props.eraser,
              changeable: gradio.shared.attached_events.includes("apply"),
              realtime: gradio.shared.attached_events.includes("change") || gradio.shared.attached_events.includes("input"),
              i18n: gradio.i18n,
              transforms: gradio.props.transforms,
              accept_blobs: gradio.shared.server.accept_blobs,
              layer_options: gradio.props.layers,
              upload: (...args) => gradio.shared.client.upload(...args),
              placeholder: gradio.props.placeholder,
              webcam_options: gradio.props.webcam_options,
              show_download_button: gradio.props.buttons === null ? true : gradio.props.buttons.includes("download"),
              theme_mode: gradio.shared.theme_mode,
              get is_dragging() {
                return is_dragging;
              },
              set is_dragging($$value) {
                is_dragging = $$value;
                $$settled = false;
              },
              get image_id() {
                return image_id;
              },
              set image_id($$value) {
                image_id = $$value;
                $$settled = false;
              }
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
const Index$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: Index
}, Symbol.toStringTag, { value: "Module" }));

export { UPDATE_BLEND as $, AbstractRenderer as A, BufferUsage as B, Container as C, DOMAdapter as D, ExtensionType as E, Filter as F, GCManagedHash as G, BigPool as H, getGlobalBounds as I, Bounds as J, TexturePool as K, FilterEffect as L, Matrix as M, Sprite as N, getAttributeInfoFromFormat as O, Point as P, unsafeEvalSupported as Q, RendererType as R, State as S, Ticker as T, UPDATE_PRIORITY as U, GlobalResourceRegistry as V, Rectangle as W, SystemRunner as X, multiplyColors as Y, UPDATE_VISIBLE as Z, UPDATE_COLOR as _, EventEmitter as a, TextureStyle as a0, Color as a1, RenderGroup as a2, getLocalBounds as a3, RenderTexture as a4, VERSION as a5, deprecation as a6, v8_0_0 as a7, RendererInitHook as a8, Geometry as a9, checkMaxIfStatementsInShader as aa, compileHighShaderGlProgram as ab, colorBitGl as ac, roundPixelsBitGl as ad, generateTextureBatchBitGl as ae, getBatchSamplersUniformGroup as af, nextPow2 as ag, CanvasTextMetrics as ah, fontStringFromTextStyle as ai, getCanvasFillStyle as aj, BatchableGraphics as ak, getAdjustedBlendModeBlend as al, ViewableBuffer as am, BitmapFontManager as an, getBitmapTextLayout as ao, Cache as ap, Graphics as aq, updateQuadBounds as ar, TextStyle as as, GraphicsContextSystem as at, Index$1 as au, STENCIL_MODES as b, Buffer as c, BindGroup as d, extensions as e, fastCopy as f, getTextureBatchBindGroup as g, createIdFromString as h, CLEAR as i, CanvasSource as j, TextureSource as k, UniformGroup as l, CanvasPool as m, compileHighShaderGpuProgram as n, colorBit as o, roundPixelsBit as p, generateTextureBatchBit as q, removeItems as r, Shader as s, Texture as t, uid as u, TextureMatrix as v, warn as w, GpuProgram as x, GlProgram as y, DefaultBatcher as z };
//# sourceMappingURL=Index4-DjOJEvOl.js.map
