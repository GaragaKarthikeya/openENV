import { p as block, E as EFFECT_TRANSPARENT, D as DEV, q as invalid_snippet } from './index-BvBk1Iap.js';
import { B as BranchManager } from './i18n-CMo5lpzy.js';

/** @import { Snippet } from 'svelte' */
/** @import { TemplateNode } from '#client' */
/** @import { Getters } from '#shared' */

/**
 * @template {(node: TemplateNode, ...args: any[]) => void} SnippetFn
 * @param {TemplateNode} node
 * @param {() => SnippetFn | null | undefined} get_snippet
 * @param {(() => any)[]} args
 * @returns {void}
 */
function snippet(node, get_snippet, ...args) {
	var branches = new BranchManager(node);

	block(() => {
		const snippet = get_snippet() ?? null;

		if (DEV && snippet == null) {
			invalid_snippet();
		}

		branches.ensure(snippet, snippet && ((anchor) => snippet(anchor, ...args)));
	}, EFFECT_TRANSPARENT);
}

export { snippet as s };
//# sourceMappingURL=snippet-Bl5YVoCg.js.map
