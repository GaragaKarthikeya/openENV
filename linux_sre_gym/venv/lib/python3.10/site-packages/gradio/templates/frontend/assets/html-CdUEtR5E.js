import { t as template_effect, b as active_effect, r as remove_effect_dom, c as create_fragment_from_html, d as assign_nodes, g as get_first_child, h as hydrating, e as hydrate_next, i as hydrate_node, C as COMMENT_NODE, j as get_next_sibling, k as hydration_mismatch, H as HYDRATION_ERROR, D as DEV, s as set_hydrate_node, l as hash, m as dev_current_component_function, F as FILENAME, n as hydration_html_changed, o as sanitize_location } from './index-BvBk1Iap.js';

/** @import { Effect, TemplateNode } from '#client' */

/**
 * @param {Element} element
 * @param {string | null} server_hash
 * @param {string} value
 */
function check_hash(element, server_hash, value) {
	if (!server_hash || server_hash === hash(String(value ?? ''))) return;

	let location;

	// @ts-expect-error
	const loc = element.__svelte_meta?.loc;
	if (loc) {
		location = `near ${loc.file}:${loc.line}:${loc.column}`;
	} else if (dev_current_component_function?.[FILENAME]) {
		location = `in ${dev_current_component_function[FILENAME]}`;
	}

	hydration_html_changed(sanitize_location(location));
}

/**
 * @param {Element | Text | Comment} node
 * @param {() => string} get_value
 * @param {boolean} [svg]
 * @param {boolean} [mathml]
 * @param {boolean} [skip_warning]
 * @returns {void}
 */
function html(node, get_value, svg = false, mathml = false, skip_warning = false) {
	var anchor = node;

	var value = '';

	template_effect(() => {
		var effect = /** @type {Effect} */ (active_effect);

		if (value === (value = get_value() ?? '')) {
			if (hydrating) hydrate_next();
			return;
		}

		if (effect.nodes !== null) {
			remove_effect_dom(effect.nodes.start, /** @type {TemplateNode} */ (effect.nodes.end));
			effect.nodes = null;
		}

		if (value === '') return;

		if (hydrating) {
			// We're deliberately not trying to repair mismatches between server and client,
			// as it's costly and error-prone (and it's an edge case to have a mismatch anyway)
			var hash = /** @type {Comment} */ (hydrate_node).data;

			/** @type {TemplateNode | null} */
			var next = hydrate_next();
			var last = next;

			while (
				next !== null &&
				(next.nodeType !== COMMENT_NODE || /** @type {Comment} */ (next).data !== '')
			) {
				last = next;
				next = get_next_sibling(next);
			}

			if (next === null) {
				hydration_mismatch();
				throw HYDRATION_ERROR;
			}

			if (DEV && !skip_warning) {
				check_hash(/** @type {Element} */ (next.parentNode), hash, value);
			}

			assign_nodes(hydrate_node, last);
			anchor = set_hydrate_node(next);
			return;
		}

		var html = value + '';
		if (svg) html = `<svg>${html}</svg>`;
		else if (mathml) html = `<math>${html}</math>`;

		// Don't use create_fragment_with_script_from_html here because that would mean script tags are executed.
		// @html is basically `.innerHTML = ...` and that doesn't execute scripts either due to security reasons.
		/** @type {DocumentFragment | Element} */
		var node = create_fragment_from_html(html);

		if (svg || mathml) {
			node = /** @type {Element} */ (get_first_child(node));
		}

		assign_nodes(
			/** @type {TemplateNode} */ (get_first_child(node)),
			/** @type {TemplateNode} */ (node.lastChild)
		);

		if (svg || mathml) {
			while (get_first_child(node)) {
				anchor.before(/** @type {TemplateNode} */ (get_first_child(node)));
			}
		} else {
			anchor.before(node);
		}
	});
}

export { html as h };
//# sourceMappingURL=html-CdUEtR5E.js.map
