import { f as fallback } from './async-D55cHugf.js';
import { c as spread_props, b as store_get, u as unsubscribe_stores, d as bind_props, f as attr_class, s as slot, g as attr_style, i as stringify, n as sanitize_props, r as rest_props, m as attributes } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { D as Download } from './Download-ByiErn53.js';
import { I as Image } from './Image2-BLbvQKZw.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FullscreenButton } from './FullscreenButton-DVAVExlR.js';
import { w as writable } from './index-Cg-Pg6j3.js';
import { l as loop, r as raf, i as is_date, S as Static } from './index3-BSof0MuO.js';
import { U as Upload } from './Upload2-OQ1gRY0J.js';
import { B as Block } from './Block-JRVGROlG.js';
import { U as UploadText } from './UploadText-CnLI0lM1.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './clone-Yk88IHKV.js';
import './Maximize-B77VDSzq.js';
import './Upload-CIQ-D6yx.js';

/*
Adapted from https://github.com/mattdesl
Distributed under MIT License https://github.com/mattdesl/eases/blob/master/LICENSE.md
*/

/**
 * @param {number} t
 * @returns {number}
 */
function linear(t) {
	return t;
}

/** @import { Task } from '../internal/client/types' */
/** @import { Tweened } from './public' */
/** @import { TweenedOptions } from './private' */

/**
 * @template T
 * @param {T} a
 * @param {T} b
 * @returns {(t: number) => T}
 */
function get_interpolator(a, b) {
	if (a === b || a !== a) return () => a;

	const type = typeof a;
	if (type !== typeof b || Array.isArray(a) !== Array.isArray(b)) {
		throw new Error('Cannot interpolate values of different type');
	}

	if (Array.isArray(a)) {
		const arr = /** @type {Array<any>} */ (b).map((bi, i) => {
			return get_interpolator(/** @type {Array<any>} */ (a)[i], bi);
		});

		// @ts-ignore
		return (t) => arr.map((fn) => fn(t));
	}

	if (type === 'object') {
		if (!a || !b) {
			throw new Error('Object cannot be null');
		}

		if (is_date(a) && is_date(b)) {
			const an = a.getTime();
			const bn = b.getTime();
			const delta = bn - an;

			// @ts-ignore
			return (t) => new Date(an + t * delta);
		}

		const keys = Object.keys(b);

		/** @type {Record<string, (t: number) => T>} */
		const interpolators = {};
		keys.forEach((key) => {
			// @ts-ignore
			interpolators[key] = get_interpolator(a[key], b[key]);
		});

		// @ts-ignore
		return (t) => {
			/** @type {Record<string, any>} */
			const result = {};
			keys.forEach((key) => {
				result[key] = interpolators[key](t);
			});
			return result;
		};
	}

	if (type === 'number') {
		const delta = /** @type {number} */ (b) - /** @type {number} */ (a);
		// @ts-ignore
		return (t) => a + t * delta;
	}

	// for non-numeric values, snap to the final value immediately
	return () => b;
}

/**
 * A tweened store in Svelte is a special type of store that provides smooth transitions between state values over time.
 *
 * @deprecated Use [`Tween`](https://svelte.dev/docs/svelte/svelte-motion#Tween) instead
 * @template T
 * @param {T} [value]
 * @param {TweenedOptions<T>} [defaults]
 * @returns {Tweened<T>}
 */
function tweened(value, defaults = {}) {
	const store = writable(value);
	/** @type {Task} */
	let task;
	let target_value = value;
	/**
	 * @param {T} new_value
	 * @param {TweenedOptions<T>} [opts]
	 */
	function set(new_value, opts) {
		target_value = new_value;

		if (value == null) {
			store.set((value = new_value));
			return Promise.resolve();
		}

		/** @type {Task | null} */
		let previous_task = task;

		let started = false;
		let {
			delay = 0,
			duration = 400,
			easing = linear,
			interpolate = get_interpolator
		} = { ...defaults, ...opts };

		if (duration === 0) {
			if (previous_task) {
				previous_task.abort();
				previous_task = null;
			}
			store.set((value = target_value));
			return Promise.resolve();
		}

		const start = raf.now() + delay;

		/** @type {(t: number) => T} */
		let fn;
		task = loop((now) => {
			if (now < start) return true;
			if (!started) {
				fn = interpolate(/** @type {any} */ (value), new_value);
				if (typeof duration === 'function')
					duration = duration(/** @type {any} */ (value), new_value);
				started = true;
			}
			if (previous_task) {
				previous_task.abort();
				previous_task = null;
			}
			const elapsed = now - start;
			if (elapsed > /** @type {number} */ (duration)) {
				store.set((value = new_value));
				return false;
			}
			// @ts-ignore
			store.set((value = fn(easing(elapsed / duration))));
			return true;
		});
		return task.promise;
	}
	return {
		set,
		update: (fn, opts) =>
			set(fn(/** @type {any} */ (target_value), /** @type {any} */ (value)), opts),
		subscribe: store.subscribe
	};
}

/* empty css                                           */
var noop = { value: () => {
} };
function Dispatch(_) {
  this._ = _;
}
function parseTypenames$1(typenames, types) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name = "", i = t.indexOf(".");
    if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
    if (t && !types.hasOwnProperty(t)) throw new Error("unknown type: " + t);
    return { type: t, name };
  });
}
Dispatch.prototype = {
  constructor: Dispatch,
  on: function(typename, callback) {
    var _ = this._, T = parseTypenames$1(typename + "", _), t, i = -1, n = T.length;
    if (arguments.length < 2) {
      while (++i < n) if ((t = (typename = T[i]).type) && (t = get(_[t], typename.name))) return t;
      return;
    }
    if (callback != null && typeof callback !== "function") throw new Error("invalid callback: " + callback);
    while (++i < n) {
      if (t = (typename = T[i]).type) _[t] = set(_[t], typename.name, callback);
      else if (callback == null) for (t in _) _[t] = set(_[t], typename.name, null);
    }
    return this;
  },
  copy: function() {
    var copy = {}, _ = this._;
    for (var t in _) copy[t] = _[t].slice();
    return new Dispatch(copy);
  },
  call: function(type, that) {
    if ((n = arguments.length - 2) > 0) for (var args = new Array(n), i = 0, n, t; i < n; ++i) args[i] = arguments[i + 2];
    if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
    for (t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
  },
  apply: function(type, that, args) {
    if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
    for (var t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
  }
};
function get(type, name) {
  for (var i = 0, n = type.length, c; i < n; ++i) {
    if ((c = type[i]).name === name) {
      return c.value;
    }
  }
}
function set(type, name, callback) {
  for (var i = 0, n = type.length; i < n; ++i) {
    if (type[i].name === name) {
      type[i] = noop, type = type.slice(0, i).concat(type.slice(i + 1));
      break;
    }
  }
  if (callback != null) type.push({ name, value: callback });
  return type;
}
var xhtml = "http://www.w3.org/1999/xhtml";
const namespaces = {
  svg: "http://www.w3.org/2000/svg",
  xhtml,
  xlink: "http://www.w3.org/1999/xlink",
  xml: "http://www.w3.org/XML/1998/namespace",
  xmlns: "http://www.w3.org/2000/xmlns/"
};
function namespace(name) {
  var prefix = name += "", i = prefix.indexOf(":");
  if (i >= 0 && (prefix = name.slice(0, i)) !== "xmlns") name = name.slice(i + 1);
  return namespaces.hasOwnProperty(prefix) ? { space: namespaces[prefix], local: name } : name;
}
function creatorInherit(name) {
  return function() {
    var document2 = this.ownerDocument, uri = this.namespaceURI;
    return uri === xhtml && document2.documentElement.namespaceURI === xhtml ? document2.createElement(name) : document2.createElementNS(uri, name);
  };
}
function creatorFixed(fullname) {
  return function() {
    return this.ownerDocument.createElementNS(fullname.space, fullname.local);
  };
}
function creator(name) {
  var fullname = namespace(name);
  return (fullname.local ? creatorFixed : creatorInherit)(fullname);
}
function none() {
}
function selector(selector2) {
  return selector2 == null ? none : function() {
    return this.querySelector(selector2);
  };
}
function selection_select(select2) {
  if (typeof select2 !== "function") select2 = selector(select2);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = new Array(n), node, subnode, i = 0; i < n; ++i) {
      if ((node = group[i]) && (subnode = select2.call(node, node.__data__, i, group))) {
        if ("__data__" in node) subnode.__data__ = node.__data__;
        subgroup[i] = subnode;
      }
    }
  }
  return new Selection(subgroups, this._parents);
}
function array(x) {
  return x == null ? [] : Array.isArray(x) ? x : Array.from(x);
}
function empty() {
  return [];
}
function selectorAll(selector2) {
  return selector2 == null ? empty : function() {
    return this.querySelectorAll(selector2);
  };
}
function arrayAll(select2) {
  return function() {
    return array(select2.apply(this, arguments));
  };
}
function selection_selectAll(select2) {
  if (typeof select2 === "function") select2 = arrayAll(select2);
  else select2 = selectorAll(select2);
  for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        subgroups.push(select2.call(node, node.__data__, i, group));
        parents.push(node);
      }
    }
  }
  return new Selection(subgroups, parents);
}
function matcher(selector2) {
  return function() {
    return this.matches(selector2);
  };
}
function childMatcher(selector2) {
  return function(node) {
    return node.matches(selector2);
  };
}
var find = Array.prototype.find;
function childFind(match) {
  return function() {
    return find.call(this.children, match);
  };
}
function childFirst() {
  return this.firstElementChild;
}
function selection_selectChild(match) {
  return this.select(match == null ? childFirst : childFind(typeof match === "function" ? match : childMatcher(match)));
}
var filter = Array.prototype.filter;
function children() {
  return Array.from(this.children);
}
function childrenFilter(match) {
  return function() {
    return filter.call(this.children, match);
  };
}
function selection_selectChildren(match) {
  return this.selectAll(match == null ? children : childrenFilter(typeof match === "function" ? match : childMatcher(match)));
}
function selection_filter(match) {
  if (typeof match !== "function") match = matcher(match);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = [], node, i = 0; i < n; ++i) {
      if ((node = group[i]) && match.call(node, node.__data__, i, group)) {
        subgroup.push(node);
      }
    }
  }
  return new Selection(subgroups, this._parents);
}
function sparse(update) {
  return new Array(update.length);
}
function selection_enter() {
  return new Selection(this._enter || this._groups.map(sparse), this._parents);
}
function EnterNode(parent, datum2) {
  this.ownerDocument = parent.ownerDocument;
  this.namespaceURI = parent.namespaceURI;
  this._next = null;
  this._parent = parent;
  this.__data__ = datum2;
}
EnterNode.prototype = {
  constructor: EnterNode,
  appendChild: function(child) {
    return this._parent.insertBefore(child, this._next);
  },
  insertBefore: function(child, next) {
    return this._parent.insertBefore(child, next);
  },
  querySelector: function(selector2) {
    return this._parent.querySelector(selector2);
  },
  querySelectorAll: function(selector2) {
    return this._parent.querySelectorAll(selector2);
  }
};
function constant$1(x) {
  return function() {
    return x;
  };
}
function bindIndex(parent, group, enter, update, exit, data) {
  var i = 0, node, groupLength = group.length, dataLength = data.length;
  for (; i < dataLength; ++i) {
    if (node = group[i]) {
      node.__data__ = data[i];
      update[i] = node;
    } else {
      enter[i] = new EnterNode(parent, data[i]);
    }
  }
  for (; i < groupLength; ++i) {
    if (node = group[i]) {
      exit[i] = node;
    }
  }
}
function bindKey(parent, group, enter, update, exit, data, key) {
  var i, node, nodeByKeyValue = /* @__PURE__ */ new Map(), groupLength = group.length, dataLength = data.length, keyValues = new Array(groupLength), keyValue;
  for (i = 0; i < groupLength; ++i) {
    if (node = group[i]) {
      keyValues[i] = keyValue = key.call(node, node.__data__, i, group) + "";
      if (nodeByKeyValue.has(keyValue)) {
        exit[i] = node;
      } else {
        nodeByKeyValue.set(keyValue, node);
      }
    }
  }
  for (i = 0; i < dataLength; ++i) {
    keyValue = key.call(parent, data[i], i, data) + "";
    if (node = nodeByKeyValue.get(keyValue)) {
      update[i] = node;
      node.__data__ = data[i];
      nodeByKeyValue.delete(keyValue);
    } else {
      enter[i] = new EnterNode(parent, data[i]);
    }
  }
  for (i = 0; i < groupLength; ++i) {
    if ((node = group[i]) && nodeByKeyValue.get(keyValues[i]) === node) {
      exit[i] = node;
    }
  }
}
function datum(node) {
  return node.__data__;
}
function selection_data(value, key) {
  if (!arguments.length) return Array.from(this, datum);
  var bind = key ? bindKey : bindIndex, parents = this._parents, groups = this._groups;
  if (typeof value !== "function") value = constant$1(value);
  for (var m = groups.length, update = new Array(m), enter = new Array(m), exit = new Array(m), j = 0; j < m; ++j) {
    var parent = parents[j], group = groups[j], groupLength = group.length, data = arraylike(value.call(parent, parent && parent.__data__, j, parents)), dataLength = data.length, enterGroup = enter[j] = new Array(dataLength), updateGroup = update[j] = new Array(dataLength), exitGroup = exit[j] = new Array(groupLength);
    bind(parent, group, enterGroup, updateGroup, exitGroup, data, key);
    for (var i0 = 0, i1 = 0, previous, next; i0 < dataLength; ++i0) {
      if (previous = enterGroup[i0]) {
        if (i0 >= i1) i1 = i0 + 1;
        while (!(next = updateGroup[i1]) && ++i1 < dataLength) ;
        previous._next = next || null;
      }
    }
  }
  update = new Selection(update, parents);
  update._enter = enter;
  update._exit = exit;
  return update;
}
function arraylike(data) {
  return typeof data === "object" && "length" in data ? data : Array.from(data);
}
function selection_exit() {
  return new Selection(this._exit || this._groups.map(sparse), this._parents);
}
function selection_join(onenter, onupdate, onexit) {
  var enter = this.enter(), update = this, exit = this.exit();
  if (typeof onenter === "function") {
    enter = onenter(enter);
    if (enter) enter = enter.selection();
  } else {
    enter = enter.append(onenter + "");
  }
  if (onupdate != null) {
    update = onupdate(update);
    if (update) update = update.selection();
  }
  if (onexit == null) exit.remove();
  else onexit(exit);
  return enter && update ? enter.merge(update).order() : update;
}
function selection_merge(context) {
  var selection = context.selection ? context.selection() : context;
  for (var groups0 = this._groups, groups1 = selection._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
    for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i = 0; i < n; ++i) {
      if (node = group0[i] || group1[i]) {
        merge[i] = node;
      }
    }
  }
  for (; j < m0; ++j) {
    merges[j] = groups0[j];
  }
  return new Selection(merges, this._parents);
}
function selection_order() {
  for (var groups = this._groups, j = -1, m = groups.length; ++j < m; ) {
    for (var group = groups[j], i = group.length - 1, next = group[i], node; --i >= 0; ) {
      if (node = group[i]) {
        if (next && node.compareDocumentPosition(next) ^ 4) next.parentNode.insertBefore(node, next);
        next = node;
      }
    }
  }
  return this;
}
function selection_sort(compare) {
  if (!compare) compare = ascending;
  function compareNode(a, b) {
    return a && b ? compare(a.__data__, b.__data__) : !a - !b;
  }
  for (var groups = this._groups, m = groups.length, sortgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, sortgroup = sortgroups[j] = new Array(n), node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        sortgroup[i] = node;
      }
    }
    sortgroup.sort(compareNode);
  }
  return new Selection(sortgroups, this._parents).order();
}
function ascending(a, b) {
  return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
}
function selection_call() {
  var callback = arguments[0];
  arguments[0] = this;
  callback.apply(null, arguments);
  return this;
}
function selection_nodes() {
  return Array.from(this);
}
function selection_node() {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group = groups[j], i = 0, n = group.length; i < n; ++i) {
      var node = group[i];
      if (node) return node;
    }
  }
  return null;
}
function selection_size() {
  let size = 0;
  for (const node of this) ++size;
  return size;
}
function selection_empty() {
  return !this.node();
}
function selection_each(callback) {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group = groups[j], i = 0, n = group.length, node; i < n; ++i) {
      if (node = group[i]) callback.call(node, node.__data__, i, group);
    }
  }
  return this;
}
function attrRemove(name) {
  return function() {
    this.removeAttribute(name);
  };
}
function attrRemoveNS(fullname) {
  return function() {
    this.removeAttributeNS(fullname.space, fullname.local);
  };
}
function attrConstant(name, value) {
  return function() {
    this.setAttribute(name, value);
  };
}
function attrConstantNS(fullname, value) {
  return function() {
    this.setAttributeNS(fullname.space, fullname.local, value);
  };
}
function attrFunction(name, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.removeAttribute(name);
    else this.setAttribute(name, v);
  };
}
function attrFunctionNS(fullname, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.removeAttributeNS(fullname.space, fullname.local);
    else this.setAttributeNS(fullname.space, fullname.local, v);
  };
}
function selection_attr(name, value) {
  var fullname = namespace(name);
  if (arguments.length < 2) {
    var node = this.node();
    return fullname.local ? node.getAttributeNS(fullname.space, fullname.local) : node.getAttribute(fullname);
  }
  return this.each((value == null ? fullname.local ? attrRemoveNS : attrRemove : typeof value === "function" ? fullname.local ? attrFunctionNS : attrFunction : fullname.local ? attrConstantNS : attrConstant)(fullname, value));
}
function defaultView(node) {
  return node.ownerDocument && node.ownerDocument.defaultView || node.document && node || node.defaultView;
}
function styleRemove(name) {
  return function() {
    this.style.removeProperty(name);
  };
}
function styleConstant(name, value, priority) {
  return function() {
    this.style.setProperty(name, value, priority);
  };
}
function styleFunction(name, value, priority) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.style.removeProperty(name);
    else this.style.setProperty(name, v, priority);
  };
}
function selection_style(name, value, priority) {
  return arguments.length > 1 ? this.each((value == null ? styleRemove : typeof value === "function" ? styleFunction : styleConstant)(name, value, priority == null ? "" : priority)) : styleValue(this.node(), name);
}
function styleValue(node, name) {
  return node.style.getPropertyValue(name) || defaultView(node).getComputedStyle(node, null).getPropertyValue(name);
}
function propertyRemove(name) {
  return function() {
    delete this[name];
  };
}
function propertyConstant(name, value) {
  return function() {
    this[name] = value;
  };
}
function propertyFunction(name, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) delete this[name];
    else this[name] = v;
  };
}
function selection_property(name, value) {
  return arguments.length > 1 ? this.each((value == null ? propertyRemove : typeof value === "function" ? propertyFunction : propertyConstant)(name, value)) : this.node()[name];
}
function classArray(string) {
  return string.trim().split(/^|\s+/);
}
function classList(node) {
  return node.classList || new ClassList(node);
}
function ClassList(node) {
  this._node = node;
  this._names = classArray(node.getAttribute("class") || "");
}
ClassList.prototype = {
  add: function(name) {
    var i = this._names.indexOf(name);
    if (i < 0) {
      this._names.push(name);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  remove: function(name) {
    var i = this._names.indexOf(name);
    if (i >= 0) {
      this._names.splice(i, 1);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  contains: function(name) {
    return this._names.indexOf(name) >= 0;
  }
};
function classedAdd(node, names) {
  var list = classList(node), i = -1, n = names.length;
  while (++i < n) list.add(names[i]);
}
function classedRemove(node, names) {
  var list = classList(node), i = -1, n = names.length;
  while (++i < n) list.remove(names[i]);
}
function classedTrue(names) {
  return function() {
    classedAdd(this, names);
  };
}
function classedFalse(names) {
  return function() {
    classedRemove(this, names);
  };
}
function classedFunction(names, value) {
  return function() {
    (value.apply(this, arguments) ? classedAdd : classedRemove)(this, names);
  };
}
function selection_classed(name, value) {
  var names = classArray(name + "");
  if (arguments.length < 2) {
    var list = classList(this.node()), i = -1, n = names.length;
    while (++i < n) if (!list.contains(names[i])) return false;
    return true;
  }
  return this.each((typeof value === "function" ? classedFunction : value ? classedTrue : classedFalse)(names, value));
}
function textRemove() {
  this.textContent = "";
}
function textConstant(value) {
  return function() {
    this.textContent = value;
  };
}
function textFunction(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.textContent = v == null ? "" : v;
  };
}
function selection_text(value) {
  return arguments.length ? this.each(value == null ? textRemove : (typeof value === "function" ? textFunction : textConstant)(value)) : this.node().textContent;
}
function htmlRemove() {
  this.innerHTML = "";
}
function htmlConstant(value) {
  return function() {
    this.innerHTML = value;
  };
}
function htmlFunction(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.innerHTML = v == null ? "" : v;
  };
}
function selection_html(value) {
  return arguments.length ? this.each(value == null ? htmlRemove : (typeof value === "function" ? htmlFunction : htmlConstant)(value)) : this.node().innerHTML;
}
function raise() {
  if (this.nextSibling) this.parentNode.appendChild(this);
}
function selection_raise() {
  return this.each(raise);
}
function lower() {
  if (this.previousSibling) this.parentNode.insertBefore(this, this.parentNode.firstChild);
}
function selection_lower() {
  return this.each(lower);
}
function selection_append(name) {
  var create = typeof name === "function" ? name : creator(name);
  return this.select(function() {
    return this.appendChild(create.apply(this, arguments));
  });
}
function constantNull() {
  return null;
}
function selection_insert(name, before) {
  var create = typeof name === "function" ? name : creator(name), select2 = before == null ? constantNull : typeof before === "function" ? before : selector(before);
  return this.select(function() {
    return this.insertBefore(create.apply(this, arguments), select2.apply(this, arguments) || null);
  });
}
function remove() {
  var parent = this.parentNode;
  if (parent) parent.removeChild(this);
}
function selection_remove() {
  return this.each(remove);
}
function selection_cloneShallow() {
  var clone = this.cloneNode(false), parent = this.parentNode;
  return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
}
function selection_cloneDeep() {
  var clone = this.cloneNode(true), parent = this.parentNode;
  return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
}
function selection_clone(deep) {
  return this.select(deep ? selection_cloneDeep : selection_cloneShallow);
}
function selection_datum(value) {
  return arguments.length ? this.property("__data__", value) : this.node().__data__;
}
function contextListener(listener) {
  return function(event) {
    listener.call(this, event, this.__data__);
  };
}
function parseTypenames(typenames) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name = "", i = t.indexOf(".");
    if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
    return { type: t, name };
  });
}
function onRemove(typename) {
  return function() {
    var on = this.__on;
    if (!on) return;
    for (var j = 0, i = -1, m = on.length, o; j < m; ++j) {
      if (o = on[j], (!typename.type || o.type === typename.type) && o.name === typename.name) {
        this.removeEventListener(o.type, o.listener, o.options);
      } else {
        on[++i] = o;
      }
    }
    if (++i) on.length = i;
    else delete this.__on;
  };
}
function onAdd(typename, value, options) {
  return function() {
    var on = this.__on, o, listener = contextListener(value);
    if (on) for (var j = 0, m = on.length; j < m; ++j) {
      if ((o = on[j]).type === typename.type && o.name === typename.name) {
        this.removeEventListener(o.type, o.listener, o.options);
        this.addEventListener(o.type, o.listener = listener, o.options = options);
        o.value = value;
        return;
      }
    }
    this.addEventListener(typename.type, listener, options);
    o = { type: typename.type, name: typename.name, value, listener, options };
    if (!on) this.__on = [o];
    else on.push(o);
  };
}
function selection_on(typename, value, options) {
  var typenames = parseTypenames(typename + ""), i, n = typenames.length, t;
  if (arguments.length < 2) {
    var on = this.node().__on;
    if (on) for (var j = 0, m = on.length, o; j < m; ++j) {
      for (i = 0, o = on[j]; i < n; ++i) {
        if ((t = typenames[i]).type === o.type && t.name === o.name) {
          return o.value;
        }
      }
    }
    return;
  }
  on = value ? onAdd : onRemove;
  for (i = 0; i < n; ++i) this.each(on(typenames[i], value, options));
  return this;
}
function dispatchEvent(node, type, params) {
  var window = defaultView(node), event = window.CustomEvent;
  if (typeof event === "function") {
    event = new event(type, params);
  } else {
    event = window.document.createEvent("Event");
    if (params) event.initEvent(type, params.bubbles, params.cancelable), event.detail = params.detail;
    else event.initEvent(type, false, false);
  }
  node.dispatchEvent(event);
}
function dispatchConstant(type, params) {
  return function() {
    return dispatchEvent(this, type, params);
  };
}
function dispatchFunction(type, params) {
  return function() {
    return dispatchEvent(this, type, params.apply(this, arguments));
  };
}
function selection_dispatch(type, params) {
  return this.each((typeof params === "function" ? dispatchFunction : dispatchConstant)(type, params));
}
function* selection_iterator() {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group = groups[j], i = 0, n = group.length, node; i < n; ++i) {
      if (node = group[i]) yield node;
    }
  }
}
function Selection(groups, parents) {
  this._groups = groups;
  this._parents = parents;
}
function selection_selection() {
  return this;
}
Selection.prototype = {
  constructor: Selection,
  select: selection_select,
  selectAll: selection_selectAll,
  selectChild: selection_selectChild,
  selectChildren: selection_selectChildren,
  filter: selection_filter,
  data: selection_data,
  enter: selection_enter,
  exit: selection_exit,
  join: selection_join,
  merge: selection_merge,
  selection: selection_selection,
  order: selection_order,
  sort: selection_sort,
  call: selection_call,
  nodes: selection_nodes,
  node: selection_node,
  size: selection_size,
  empty: selection_empty,
  each: selection_each,
  attr: selection_attr,
  style: selection_style,
  property: selection_property,
  classed: selection_classed,
  text: selection_text,
  html: selection_html,
  raise: selection_raise,
  lower: selection_lower,
  append: selection_append,
  insert: selection_insert,
  remove: selection_remove,
  clone: selection_clone,
  datum: selection_datum,
  on: selection_on,
  dispatch: selection_dispatch,
  [Symbol.iterator]: selection_iterator
};
function Slider($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }
    let position = fallback($$props["position"], 0.5);
    let disabled = fallback($$props["disabled"], false);
    let slider_color = fallback($$props["slider_color"], "var(--border-color-primary)");
    let image_size = fallback($$props["image_size"], () => ({ top: 0, left: 0, width: 0, height: 0 }), true);
    let el = fallback($$props["el"], void 0);
    let parent_el = fallback($$props["parent_el"], void 0);
    let px = 0;
    let active = false;
    let container_width = 0;
    function set_position(width) {
      container_width = parent_el?.getBoundingClientRect().width || 0;
      if (width === 0) {
        image_size.width = el?.getBoundingClientRect().width || 0;
      }
      px = clamp(image_size.width * position + image_size.left, 0, container_width);
    }
    function update_position_from_pc(pc) {
      px = clamp(image_size.width * pc + image_size.left, 0, container_width);
    }
    set_position(image_size.width);
    update_position_from_pc(position);
    $$renderer2.push(`<div class="wrap svelte-b2bl92" role="none"><div class="content svelte-b2bl92"><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></div> <div${attr_class("outer svelte-b2bl92", void 0, { "disabled": disabled, "grab": active })} role="none"${attr_style(`transform: translateX(${stringify(px)}px)`)}><span${attr_class("icon-wrap svelte-b2bl92", void 0, { "active": active, "disabled": disabled })}><span class="icon left svelte-b2bl92">◢</span><span class="icon center svelte-b2bl92"${attr_style("", { "--color": slider_color })}></span><span class="icon right svelte-b2bl92">◢</span></span> <div class="inner svelte-b2bl92"${attr_style("", { "--color": slider_color })}></div></div></div>`);
    bind_props($$props, { position, disabled, slider_color, image_size, el, parent_el });
  });
}
function ImageEl($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "src",
    "fullscreen",
    "fixed",
    "transform",
    "img_el",
    "hidden",
    "variant",
    "max_height"
  ]);
  $$renderer.component(($$renderer2) => {
    let src = fallback($$props["src"], void 0);
    let fullscreen = fallback($$props["fullscreen"], false);
    let fixed = fallback($$props["fixed"], false);
    let transform = fallback($$props["transform"], "translate(0px, 0px) scale(1)");
    let img_el = fallback($$props["img_el"], null);
    let hidden = fallback($$props["hidden"], false);
    let variant = fallback($$props["variant"], "upload");
    let max_height = fallback($$props["max_height"], 500);
    $$renderer2.push(`<img${attributes(
      { src, ...$$restProps },
      "svelte-j3ek2n",
      {
        fixed,
        hidden,
        preview: variant === "preview",
        slider: variant === "upload",
        fullscreen,
        small: !fullscreen
      },
      {
        transform,
        "max-height": max_height && !fullscreen ? `${max_height}px` : null
      }
    )} onload="this.__e=event" onerror="this.__e=event"/>`);
    bind_props($$props, {
      src,
      fullscreen,
      fixed,
      transform,
      img_el,
      hidden,
      variant,
      max_height
    });
  });
}
class ZoomableImage {
  container;
  image;
  scale;
  offsetX;
  offsetY;
  isDragging;
  lastX;
  lastY;
  initial_left_padding;
  initial_top_padding;
  initial_width;
  initial_height;
  subscribers;
  handleImageLoad;
  real_image_size = { top: 0, left: 0, width: 0, height: 0 };
  last_touch_distance;
  constructor(container, image) {
    this.container = container;
    this.image = image;
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.isDragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.initial_left_padding = 0;
    this.initial_top_padding = 0;
    this.initial_width = 0;
    this.initial_height = 0;
    this.subscribers = [];
    this.last_touch_distance = 0;
    this.handleWheel = this.handleWheel.bind(this);
    this.handleMouseDown = this.handleMouseDown.bind(this);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.handleMouseUp = this.handleMouseUp.bind(this);
    this.handleImageLoad = this.init.bind(this);
    this.handleTouchStart = this.handleTouchStart.bind(this);
    this.handleTouchMove = this.handleTouchMove.bind(this);
    this.handleTouchEnd = this.handleTouchEnd.bind(this);
    this.image.addEventListener("load", this.handleImageLoad);
    this.container.addEventListener("wheel", this.handleWheel);
    this.container.addEventListener("mousedown", this.handleMouseDown);
    document.addEventListener("mousemove", this.handleMouseMove);
    document.addEventListener("mouseup", this.handleMouseUp);
    this.container.addEventListener("touchstart", this.handleTouchStart);
    document.addEventListener("touchmove", this.handleTouchMove);
    document.addEventListener("touchend", this.handleTouchEnd);
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === this.container) {
          this.handleResize();
          this.get_image_size(this.image);
        }
      }
    });
    observer.observe(this.container);
  }
  handleResize() {
    this.init();
  }
  init() {
    const containerRect = this.container.getBoundingClientRect();
    const imageRect = this.image.getBoundingClientRect();
    this.initial_left_padding = imageRect.left - containerRect.left;
    this.initial_top_padding = imageRect.top - containerRect.top;
    this.initial_width = imageRect.width;
    this.initial_height = imageRect.height;
    this.reset_zoom();
    this.updateTransform();
  }
  reset_zoom() {
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.updateTransform();
  }
  handleMouseDown(e) {
    const imageRect = this.image.getBoundingClientRect();
    if (e.clientX >= imageRect.left && e.clientX <= imageRect.right && e.clientY >= imageRect.top && e.clientY <= imageRect.bottom) {
      e.preventDefault();
      if (this.scale === 1) return;
      this.isDragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.image.style.cursor = "grabbing";
    }
  }
  handleMouseMove(e) {
    if (!this.isDragging) return;
    const deltaX = e.clientX - this.lastX;
    const deltaY = e.clientY - this.lastY;
    this.offsetX += deltaX;
    this.offsetY += deltaY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.updateTransform();
    this.updateTransform();
  }
  handleMouseUp() {
    if (this.isDragging) {
      this.constrain_to_bounds(true);
      this.updateTransform();
      this.isDragging = false;
      this.image.style.cursor = this.scale > 1 ? "grab" : "zoom-in";
    }
  }
  async handleWheel(e) {
    e.preventDefault();
    const containerRect = this.container.getBoundingClientRect();
    const imageRect = this.image.getBoundingClientRect();
    if (e.clientX < imageRect.left || e.clientX > imageRect.right || e.clientY < imageRect.top || e.clientY > imageRect.bottom) {
      return;
    }
    const zoomFactor = 1.05;
    const oldScale = this.scale;
    const newScale = -Math.sign(e.deltaY) > 0 ? Math.min(15, oldScale * zoomFactor) : Math.max(1, oldScale / zoomFactor);
    if (newScale === oldScale) return;
    const cursorX = e.clientX - containerRect.left - this.initial_left_padding;
    const cursorY = e.clientY - containerRect.top - this.initial_top_padding;
    this.scale = newScale;
    this.offsetX = this.compute_new_offset({
      cursor_position: cursorX,
      current_offset: this.offsetX,
      new_scale: newScale,
      old_scale: oldScale
    });
    this.offsetY = this.compute_new_offset({
      cursor_position: cursorY,
      current_offset: this.offsetY,
      new_scale: newScale,
      old_scale: oldScale
    });
    this.updateTransform();
    this.constrain_to_bounds();
    this.updateTransform();
    this.image.style.cursor = this.scale > 1 ? "grab" : "zoom-in";
  }
  // compute_offset_for_positions({ position: number, scale: number }) {
  // 	return position - (scale / this.scale) * (position - this.offset);
  // }
  compute_new_position({
    position,
    scale,
    anchor_position
  }) {
    return position - (position - anchor_position) * (scale / this.scale);
  }
  compute_new_offset({
    cursor_position,
    current_offset,
    new_scale,
    old_scale
  }) {
    return cursor_position - new_scale / old_scale * (cursor_position - current_offset);
  }
  constrain_to_bounds(pan = false) {
    if (this.scale === 1) {
      this.offsetX = 0;
      this.offsetY = 0;
      return;
    }
    const onscreen = {
      top: this.real_image_size.top * this.scale + this.offsetY,
      left: this.real_image_size.left * this.scale + this.offsetX,
      width: this.real_image_size.width * this.scale,
      height: this.real_image_size.height * this.scale,
      bottom: this.real_image_size.top * this.scale + this.offsetY + this.real_image_size.height * this.scale,
      right: this.real_image_size.left * this.scale + this.offsetX + this.real_image_size.width * this.scale
    };
    const real_image_size_right = this.real_image_size.left + this.real_image_size.width;
    const real_image_size_bottom = this.real_image_size.top + this.real_image_size.height;
    if (pan) {
      if (onscreen.top > this.real_image_size.top) {
        this.offsetY = this.calculate_position(
          this.real_image_size.top,
          0,
          "y"
        );
      } else if (onscreen.bottom < real_image_size_bottom) {
        this.offsetY = this.calculate_position(real_image_size_bottom, 1, "y");
      }
      if (onscreen.left > this.real_image_size.left) {
        this.offsetX = this.calculate_position(
          this.real_image_size.left,
          0,
          "x"
        );
      } else if (onscreen.right < real_image_size_right) {
        this.offsetX = this.calculate_position(real_image_size_right, 1, "x");
      }
    }
  }
  updateTransform() {
    this.notify({ x: this.offsetX, y: this.offsetY, scale: this.scale });
  }
  destroy() {
    this.container.removeEventListener("wheel", this.handleWheel);
    this.container.removeEventListener("mousedown", this.handleMouseDown);
    document.removeEventListener("mousemove", this.handleMouseMove);
    document.removeEventListener("mouseup", this.handleMouseUp);
    this.container.removeEventListener("touchstart", this.handleTouchStart);
    document.removeEventListener("touchmove", this.handleTouchMove);
    document.removeEventListener("touchend", this.handleTouchEnd);
    this.image.removeEventListener("load", this.handleImageLoad);
  }
  subscribe(cb) {
    this.subscribers.push(cb);
  }
  unsubscribe(cb) {
    this.subscribers = this.subscribers.filter(
      (subscriber) => subscriber !== cb
    );
  }
  notify({ x, y, scale }) {
    this.subscribers.forEach((subscriber) => subscriber({ x, y, scale }));
  }
  handleTouchStart(e) {
    e.preventDefault();
    const imageRect = this.image.getBoundingClientRect();
    const touch = e.touches[0];
    if (touch.clientX >= imageRect.left && touch.clientX <= imageRect.right && touch.clientY >= imageRect.top && touch.clientY <= imageRect.bottom) {
      if (e.touches.length === 1 && this.scale > 1) {
        this.isDragging = true;
        this.lastX = touch.clientX;
        this.lastY = touch.clientY;
      } else if (e.touches.length === 2) {
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        this.last_touch_distance = Math.hypot(
          touch2.clientX - touch1.clientX,
          touch2.clientY - touch1.clientY
        );
      }
    }
  }
  get_image_size(img) {
    if (!img) return;
    const container = img.parentElement?.getBoundingClientRect();
    if (!container) return;
    const naturalAspect = img.naturalWidth / img.naturalHeight;
    const containerAspect = container.width / container.height;
    let displayedWidth, displayedHeight;
    if (naturalAspect > containerAspect) {
      displayedWidth = container.width;
      displayedHeight = container.width / naturalAspect;
    } else {
      displayedHeight = container.height;
      displayedWidth = container.height * naturalAspect;
    }
    const offsetX = (container.width - displayedWidth) / 2;
    const offsetY = (container.height - displayedHeight) / 2;
    this.real_image_size = {
      top: offsetY,
      left: offsetX,
      width: displayedWidth,
      height: displayedHeight
    };
  }
  handleTouchMove(e) {
    if (e.touches.length === 1 && this.isDragging) {
      e.preventDefault();
      const touch = e.touches[0];
      const deltaX = touch.clientX - this.lastX;
      const deltaY = touch.clientY - this.lastY;
      this.offsetX += deltaX;
      this.offsetY += deltaY;
      this.lastX = touch.clientX;
      this.lastY = touch.clientY;
      this.updateTransform();
    } else if (e.touches.length === 2) {
      e.preventDefault();
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      const current_distance = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      );
      if (this.last_touch_distance === 0) {
        this.last_touch_distance = current_distance;
        return;
      }
      const zoomFactor = current_distance / this.last_touch_distance;
      const oldScale = this.scale;
      const newScale = Math.min(15, Math.max(1, oldScale * zoomFactor));
      if (newScale === oldScale) {
        this.last_touch_distance = current_distance;
        return;
      }
      const containerRect = this.container.getBoundingClientRect();
      const midX = (touch1.clientX + touch2.clientX) / 2 - containerRect.left - this.initial_left_padding;
      const midY = (touch1.clientY + touch2.clientY) / 2 - containerRect.top - this.initial_top_padding;
      this.scale = newScale;
      this.offsetX = this.compute_new_offset({
        cursor_position: midX,
        current_offset: this.offsetX,
        new_scale: newScale,
        old_scale: oldScale
      });
      this.offsetY = this.compute_new_offset({
        cursor_position: midY,
        current_offset: this.offsetY,
        new_scale: newScale,
        old_scale: oldScale
      });
      this.updateTransform();
      this.constrain_to_bounds();
      this.updateTransform();
      this.last_touch_distance = current_distance;
      this.image.style.cursor = this.scale > 1 ? "grab" : "zoom-in";
    }
  }
  handleTouchEnd(e) {
    if (this.isDragging) {
      this.constrain_to_bounds(true);
      this.updateTransform();
      this.isDragging = false;
    }
    if (e.touches.length === 0) {
      this.last_touch_distance = 0;
    }
  }
  calculate_position(screen_coord, image_anchor, axis) {
    this.container.getBoundingClientRect();
    if (axis === "x") {
      const relative_screen_x = screen_coord;
      const anchor_x = this.real_image_size.left + image_anchor * this.real_image_size.width;
      return relative_screen_x - anchor_x * this.scale;
    }
    if (axis === "y") {
      const relative_screen_y = screen_coord;
      const anchor_y = this.real_image_size.top + image_anchor * this.real_image_size.height;
      return relative_screen_y - anchor_y * this.scale;
    }
    return 0;
  }
}
function SliderPreview($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let coords_at_viewport, style;
    let value = fallback($$props["value"], () => [null, null], true);
    let label = fallback($$props["label"], void 0);
    let show_download_button = fallback($$props["show_download_button"], true);
    let show_label = $$props["show_label"];
    let i18n = $$props["i18n"];
    let position = $$props["position"];
    let layer_images = fallback($$props["layer_images"], true);
    let show_single = fallback($$props["show_single"], false);
    let slider_color = $$props["slider_color"];
    let show_fullscreen_button = fallback($$props["show_fullscreen_button"], true);
    let fullscreen = fallback($$props["fullscreen"], false);
    let buttons = fallback($$props["buttons"], null);
    let on_custom_button_click = fallback($$props["on_custom_button_click"], null);
    let el_width = fallback($$props["el_width"], 0);
    let max_height = $$props["max_height"];
    let interactive = fallback($$props["interactive"], true);
    let img;
    let slider_wrap;
    let transform = tweened({ x: 0, y: 0, z: 1 }, { duration: 75 });
    let parent_el;
    function get_coords_at_viewport(viewport_percent_x, viewportWidth, image_width, img_offset_x, tx, scale) {
      const px_relative_to_image = viewport_percent_x * image_width;
      const pixel_position = px_relative_to_image + img_offset_x;
      const normalised_position = (pixel_position - tx) / scale;
      const percent_position = normalised_position / viewportWidth;
      return percent_position;
    }
    let viewport_width = 0;
    let zoomable_image = null;
    let observer = null;
    function init_image(img2, slider_wrap2) {
      if (!img2 || !slider_wrap2) return;
      zoomable_image?.destroy();
      observer?.disconnect();
      img2?.getBoundingClientRect().width || 0;
      viewport_width = slider_wrap2?.getBoundingClientRect().width || 0;
      zoomable_image = new ZoomableImage(slider_wrap2, img2);
      zoomable_image.subscribe(({ x, y, scale }) => {
        transform.set({ x, y, z: scale });
      });
      observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.target === slider_wrap2) {
            viewport_width = entry.contentRect.width;
          }
          if (entry.target === img2) {
            entry.contentRect.width;
          }
        }
      });
      observer.observe(slider_wrap2);
      observer.observe(img2);
    }
    let image_size = { top: 0, left: 0, width: 0, height: 0 };
    coords_at_viewport = get_coords_at_viewport(position, viewport_width, image_size.width, image_size.left, store_get($$store_subs ??= {}, "$transform", transform).x, store_get($$store_subs ??= {}, "$transform", transform).z);
    style = layer_images ? `clip-path: inset(0 0 0 ${coords_at_viewport * 100}%)` : "";
    init_image(img, slider_wrap);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: Image, label: label || i18n("image.image") });
      $$renderer3.push(`<!----> `);
      if ((value === null || value[0] === null || value[1] === null) && !show_single) {
        $$renderer3.push("<!--[-->");
        Empty($$renderer3, {
          unpadded_box: true,
          size: "large",
          children: ($$renderer4) => {
            Image($$renderer4);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<div class="image-container svelte-1880bc6">`);
        IconButtonWrapper($$renderer3, {
          buttons,
          on_custom_button_click,
          children: ($$renderer4) => {
            IconButton($$renderer4, {
              Icon: Undo,
              label: i18n("common.undo"),
              disabled: store_get($$store_subs ??= {}, "$transform", transform).z === 1,
              onclick: () => zoomable_image?.reset_zoom()
            });
            $$renderer4.push(`<!----> `);
            if (show_fullscreen_button) {
              $$renderer4.push("<!--[-->");
              FullscreenButton($$renderer4, { fullscreen });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            if (show_download_button) {
              $$renderer4.push("<!--[-->");
              DownloadLink($$renderer4, {
                href: value[1]?.url,
                download: value[1]?.orig_name || "image",
                children: ($$renderer5) => {
                  IconButton($$renderer5, { Icon: Download, label: i18n("common.download") });
                },
                $$slots: { default: true }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            if (interactive) {
              $$renderer4.push("<!--[-->");
              IconButton($$renderer4, {
                Icon: Clear,
                label: "Remove Image",
                onclick: (event) => {
                  value = [null, null];
                  event.stopPropagation();
                }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
        });
        $$renderer3.push(`<!----> <div${attr_class("slider-wrap svelte-1880bc6", void 0, { "limit_height": !fullscreen })}>`);
        Slider($$renderer3, {
          slider_color,
          image_size,
          get position() {
            return position;
          },
          set position($$value) {
            position = $$value;
            $$settled = false;
          },
          get el() {
            return slider_wrap;
          },
          set el($$value) {
            slider_wrap = $$value;
            $$settled = false;
          },
          get parent_el() {
            return parent_el;
          },
          set parent_el($$value) {
            parent_el = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            ImageEl($$renderer4, {
              src: value?.[0]?.url,
              alt: "",
              loading: "lazy",
              variant: "preview",
              transform: `translate(${stringify(store_get($$store_subs ??= {}, "$transform", transform).x)}px, ${stringify(store_get($$store_subs ??= {}, "$transform", transform).y)}px) scale(${stringify(store_get($$store_subs ??= {}, "$transform", transform).z)})`,
              fullscreen,
              max_height,
              get img_el() {
                return img;
              },
              set img_el($$value) {
                img = $$value;
                $$settled = false;
              }
            });
            $$renderer4.push(`<!----> `);
            ImageEl($$renderer4, {
              variant: "preview",
              fixed: layer_images,
              hidden: !value?.[1]?.url,
              src: value?.[1]?.url,
              alt: "",
              loading: "lazy",
              style: `${stringify(style)}; background: var(--block-background-fill);`,
              transform: `translate(${stringify(store_get($$store_subs ??= {}, "$transform", transform).x)}px, ${stringify(store_get($$store_subs ??= {}, "$transform", transform).y)}px) scale(${stringify(store_get($$store_subs ??= {}, "$transform", transform).z)})`,
              fullscreen,
              max_height
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></div></div>`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, {
      value,
      label,
      show_download_button,
      show_label,
      i18n,
      position,
      layer_images,
      show_single,
      slider_color,
      show_fullscreen_button,
      fullscreen,
      buttons,
      on_custom_button_click,
      el_width,
      max_height,
      interactive
    });
  });
}
function ClearImage($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="svelte-2ufkjh">`);
    IconButton($$renderer2, {
      Icon: Clear,
      label: "Remove Image",
      onclick: (event) => {
        event.stopPropagation();
      }
    });
    $$renderer2.push(`<!----></div>`);
  });
}
function Image_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let label = fallback($$props["label"], void 0);
    let show_label = $$props["show_label"];
    let root2 = $$props["root"];
    let position = $$props["position"];
    let upload_count = fallback($$props["upload_count"], 2);
    let show_download_button = fallback($$props["show_download_button"], true);
    let slider_color = $$props["slider_color"];
    let upload = $$props["upload"];
    let stream_handler = $$props["stream_handler"];
    let max_file_size = fallback($$props["max_file_size"], null);
    let i18n = $$props["i18n"];
    let max_height = $$props["max_height"];
    let upload_promise = fallback($$props["upload_promise"], null);
    let value_ = value || [null, null];
    let img;
    let el_width;
    async function handle_upload(detail, n) {
      const new_value = [value[0], value[1]];
      if (detail.length > 1) {
        new_value[n] = detail[0];
      } else {
        new_value[n] = detail[n];
      }
      value = new_value;
      await tick();
    }
    let old_value = "";
    let dragging = fallback($$props["dragging"], false);
    if (JSON.stringify(value) !== old_value) {
      old_value = JSON.stringify(value);
      value_ = value;
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: Image, label: label || i18n("image.image") });
      $$renderer3.push(`<!----> <div data-testid="image" class="image-container svelte-1c8zs50">`);
      if (value?.[0]?.url || value?.[1]?.url) {
        $$renderer3.push("<!--[-->");
        ClearImage($$renderer3);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (value?.[1]?.url) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="icon-buttons svelte-1c8zs50">`);
        if (show_download_button) {
          $$renderer3.push("<!--[-->");
          DownloadLink($$renderer3, {
            href: value[1].url,
            download: value[1].orig_name || "image",
            children: ($$renderer4) => {
              IconButton($$renderer4, { Icon: Download });
            },
            $$slots: { default: true }
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      Slider($$renderer3, {
        disabled: upload_count == 2 || !value?.[0],
        slider_color,
        get position() {
          return position;
        },
        set position($$value) {
          position = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          $$renderer4.push(`<div${attr_class("upload-wrap svelte-1c8zs50", void 0, { "side-by-side": upload_count === 2 })}${attr_style("", { display: upload_count === 2 ? "flex" : "block" })}>`);
          if (!value_?.[0]) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<div${attr_class("wrap svelte-1c8zs50", void 0, { "half-wrap": upload_count === 1 })}>`);
            Upload($$renderer4, {
              filetype: "image/*",
              onload: (e) => handle_upload(e, 0),
              disable_click: !!value?.[0],
              root: root2,
              file_count: "multiple",
              upload,
              stream_handler,
              max_file_size,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get dragging() {
                return dragging;
              },
              set dragging($$value) {
                dragging = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                $$renderer5.push(`<!--[-->`);
                slot($$renderer5, $$props, "default", {});
                $$renderer5.push(`<!--]-->`);
              },
              $$slots: { default: true }
            });
            $$renderer4.push(`<!----></div>`);
          } else {
            $$renderer4.push("<!--[!-->");
            ImageEl($$renderer4, {
              variant: "upload",
              src: value_[0]?.url,
              alt: "",
              max_height,
              get img_el() {
                return img;
              },
              set img_el($$value) {
                img = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer4.push(`<!--]--> `);
          if (!value_?.[1] && upload_count === 2) {
            $$renderer4.push("<!--[-->");
            Upload($$renderer4, {
              filetype: "image/*",
              onload: (e) => handle_upload(e, 1),
              disable_click: !!value?.[1],
              root: root2,
              file_count: "multiple",
              upload,
              stream_handler,
              max_file_size,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get dragging() {
                return dragging;
              },
              set dragging($$value) {
                dragging = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                $$renderer5.push(`<!--[-->`);
                slot($$renderer5, $$props, "default", {});
                $$renderer5.push(`<!--]-->`);
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer4.push("<!--[!-->");
            if (!value_?.[1] && upload_count === 1) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<div${attr_class("empty-wrap fixed svelte-1c8zs50", void 0, { "white-icon": !value?.[0]?.url })}${attr_style("", {
                width: `${stringify(el_width * (1 - position))}px`,
                transform: `translateX(${stringify(el_width * position)}px)`
              })}>`);
              Empty($$renderer4, {
                unpadded_box: true,
                size: "large",
                children: ($$renderer5) => {
                  Image($$renderer5);
                },
                $$slots: { default: true }
              });
              $$renderer4.push(`<!----></div>`);
            } else {
              $$renderer4.push("<!--[!-->");
              if (value_?.[1]) {
                $$renderer4.push("<!--[-->");
                ImageEl($$renderer4, {
                  variant: "upload",
                  src: value_[1].url,
                  alt: "",
                  fixed: upload_count === 1,
                  transform: "translate(0px, 0px) scale(1)",
                  max_height
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]-->`);
            }
            $$renderer4.push(`<!--]-->`);
          }
          $$renderer4.push(`<!--]--></div>`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      value,
      label,
      show_label,
      root: root2,
      position,
      upload_count,
      show_download_button,
      slider_color,
      upload,
      stream_handler,
      max_file_size,
      i18n,
      max_height,
      upload_promise,
      dragging
    });
  });
}
function SliderUpload($$renderer, $$props) {
  let value = fallback($$props["value"], () => [null, null], true);
  let upload = $$props["upload"];
  let stream_handler = $$props["stream_handler"];
  let label = $$props["label"];
  let show_label = $$props["show_label"];
  let i18n = $$props["i18n"];
  let root2 = $$props["root"];
  let upload_count = fallback($$props["upload_count"], 1);
  let dragging = $$props["dragging"];
  let max_height = $$props["max_height"];
  let max_file_size = fallback($$props["max_file_size"], null);
  let upload_promise = fallback($$props["upload_promise"], null);
  let $$settled = true;
  let $$inner_renderer;
  function $$render_inner($$renderer2) {
    Image_1($$renderer2, {
      slider_color: "var(--border-color-primary)",
      position: 0.5,
      root: root2,
      label,
      show_label,
      upload_count,
      stream_handler,
      upload,
      max_file_size,
      max_height,
      i18n,
      get upload_promise() {
        return upload_promise;
      },
      set upload_promise($$value) {
        upload_promise = $$value;
        $$settled = false;
      },
      get value() {
        return value;
      },
      set value($$value) {
        value = $$value;
        $$settled = false;
      },
      get dragging() {
        return dragging;
      },
      set dragging($$value) {
        dragging = $$value;
        $$settled = false;
      },
      children: ($$renderer3) => {
        $$renderer3.push(`<!--[-->`);
        slot($$renderer3, $$props, "default", {});
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
  }
  do {
    $$settled = true;
    $$inner_renderer = $$renderer.copy();
    $$render_inner($$inner_renderer);
  } while (!$$settled);
  $$renderer.subsume($$inner_renderer);
  bind_props($$props, {
    value,
    upload,
    stream_handler,
    label,
    show_label,
    i18n,
    root: root2,
    upload_count,
    dragging,
    max_height,
    max_file_size,
    upload_promise
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let upload_promise = void 0;
    class ImageSliderGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new ImageSliderGradio(props);
    gradio.props.value;
    let fullscreen = false;
    let dragging = false;
    let normalised_slider_position = Math.max(0, Math.min(100, gradio.props.slider_position)) / 100;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!gradio.shared.interactive || gradio.props.value?.[1] && gradio.props.value?.[0]) {
        $$renderer3.push("<!--[-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          allow_overflow: false,
          container: gradio.shared.container,
          scale: gradio.shared.scale,
          min_width: gradio.shared.min_width,
          get fullscreen() {
            return fullscreen;
          },
          set fullscreen($$value) {
            fullscreen = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            Static($$renderer4, spread_props([
              { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
              gradio.shared.loading_status
            ]));
            $$renderer4.push(`<!----> `);
            SliderPreview($$renderer4, {
              fullscreen,
              interactive: gradio.shared.interactive,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              show_download_button: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "download"),
              i18n: gradio.i18n,
              show_fullscreen_button: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "fullscreen"),
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              position: normalised_slider_position,
              slider_color: gradio.props.slider_color,
              max_height: gradio.props.max_height,
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
            $$renderer4.push(`<!---->`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        Block($$renderer3, {
          visible: gradio.shared.visible,
          variant: gradio.props.value === null ? "dashed" : "solid",
          border_mode: dragging ? "focus" : "base",
          padding: false,
          elem_id: gradio.shared.elem_id,
          elem_classes: gradio.shared.elem_classes,
          height: gradio.props.height || void 0,
          width: gradio.props.width,
          allow_overflow: false,
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
            SliderUpload($$renderer4, {
              root: gradio.shared.root,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              upload_count: gradio.props.upload_count,
              max_file_size: gradio.shared.max_file_size,
              i18n: gradio.i18n,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: gradio.shared.client?.stream,
              max_height: gradio.props.max_height,
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              },
              get dragging() {
                return dragging;
              },
              set dragging($$value) {
                dragging = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                {
                  $$renderer5.push("<!--[-->");
                  UploadText($$renderer5, {
                    i18n: gradio.i18n,
                    type: "image",
                    placeholder: gradio.props.placeholder
                  });
                }
                $$renderer5.push(`<!--]-->`);
              },
              $$slots: { default: true }
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

export { Index as default };
//# sourceMappingURL=Index57-D9780Wsu.js.map
