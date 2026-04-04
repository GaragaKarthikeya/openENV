import { p as block, a9 as is_runes, h as hydrating, e as hydrate_next } from './index-BvBk1Iap.js';
import { B as BranchManager } from './i18n-CMo5lpzy.js';

/** @import { TemplateNode } from '#client' */

/**
 * @template V
 * @param {TemplateNode} node
 * @param {() => V} get_key
 * @param {(anchor: Node) => TemplateNode | void} render_fn
 * @returns {void}
 */
function key(node, get_key, render_fn) {
	if (hydrating) {
		hydrate_next();
	}

	var branches = new BranchManager(node);

	var legacy = !is_runes();

	block(() => {
		var key = get_key();

		// key blocks in Svelte <5 had stupid semantics
		if (legacy && key !== null && typeof key === 'object') {
			key = /** @type {V} */ ({});
		}

		branches.ensure(key, render_fn);
	});
}

export { key as k };
//# sourceMappingURL=key-BV46G5Fv.js.map
