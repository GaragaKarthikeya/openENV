import { i as if_block, s as slot, d as bind_this, a as set_attribute, b as set_class, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, t as template_effect, a as append, T as pop, X as sibling, W as from_html, V as child, M as tick, w as get, Y as reset, a5 as user_derived } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root = from_html(`<div role="region" aria-label="Draggable items container"><!> <!></div>`);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let container_el;
	let dragged_el = null;
	let items = [];
	let current_drop_target = null;

	const drag_event_handlers = [
		{ event: "dragstart", handler: handle_drag_start },
		{ event: "dragend", handler: handle_drag_end }
	];

	const drop_event_handlers = [
		{ event: "dragover", handler: handle_drag_over },
		{ event: "drop", handler: handle_drop },
		{ event: "dragenter", handler: handle_drag_enter }
	];

	function add_drag_listeners(element) {
		drag_event_handlers.forEach(({ event, handler }) => {
			element.addEventListener(event, handler);
		});
	}

	function remove_drag_listeners(element) {
		drag_event_handlers.forEach(({ event, handler }) => {
			element.removeEventListener(event, handler);
		});
	}

	function add_drop_listeners(element) {
		drop_event_handlers.forEach(({ event, handler }) => {
			element.addEventListener(event, handler);
		});
	}

	function remove_drop_listeners(element) {
		drop_event_handlers.forEach(({ event, handler }) => {
			element.removeEventListener(event, handler);
		});
	}

	function setup_draggable_item(element, index) {
		element.classList.add("draggable-item");
		element.setAttribute("data-index", index.toString());
		element.style.position = "relative";

		const handle = document.createElement("div");

		handle.className = "drag-handle";
		handle.setAttribute("draggable", "true");
		handle.setAttribute("aria-grabbed", "false");
		handle.setAttribute("data-index", index.toString());
		handle.innerHTML = "⋮⋮";
		element.appendChild(handle);
		add_drag_listeners(handle);
		add_drop_listeners(element);
	}

	function setup_drag_and_drop() {
		if (!container_el) return;

		items.forEach((item) => {
			const handle = item.querySelector(".drag-handle");

			if (handle) {
				remove_drag_listeners(handle);
				handle.remove();
			}

			remove_drop_listeners(item);
		});

		items = [];

		const children = Array.from(container_el.children);
		let item_index = 0;

		children.forEach((child) => {
			if (child.classList.contains("status-tracker")) return;

			if (child.classList.contains("form") && child.children.length > 0) {
				const form_children = Array.from(child.children);

				form_children.forEach((form_child) => {
					items.push(form_child);
					setup_draggable_item(form_child, item_index);
					item_index++;
				});
			} else {
				items.push(child);
				setup_draggable_item(child, item_index);
				item_index++;
			}
		});
	}

	function handle_drag_start(e) {
		e.stopPropagation?.();

		const handle = e.currentTarget;
		const element = handle.parentElement;

		dragged_el = element;
		parseInt(handle.dataset.index || "-1");
		handle.setAttribute("aria-grabbed", "true");
		element.classList.add("dragging");

		if (e.dataTransfer) {
			e.dataTransfer.effectAllowed = "move";
			e.dataTransfer.setData("text/html", element.outerHTML);
			e.dataTransfer.setDragImage(element, 10, 10);
		}
	}

	function handle_drag_end(e) {
		e.stopPropagation?.();

		const handle = e.currentTarget;
		const element = handle.parentElement;

		element.classList.remove("dragging");
		handle.setAttribute("aria-grabbed", "false");

		if (current_drop_target) {
			current_drop_target.classList.remove("drop-target");
			current_drop_target = null;
		}

		dragged_el = null;
	}

	function handle_drag_over(e) {
		e.preventDefault();
		e.dataTransfer && (e.dataTransfer.dropEffect = "move");

		return false;
	}

	function handle_drag_enter(e) {
		e.stopPropagation?.();

		const target = e.currentTarget;

		if (target === dragged_el) return;
		if (!dragged_el || !container_el) return;

		const dragged_parent = dragged_el.parentElement;
		const target_parent = target.parentElement;
		const is_dragged_in_form = dragged_parent?.classList.contains("form") && dragged_parent.parentElement === container_el;
		const is_target_in_form = target_parent?.classList.contains("form") && target_parent.parentElement === container_el;
		const is_dragged_direct = dragged_parent === container_el;
		const is_target_direct = target_parent === container_el;

		if (!((is_dragged_in_form || is_dragged_direct) && (is_target_in_form || is_target_direct))) {
			return;
		}

		if (current_drop_target && current_drop_target !== target) {
			current_drop_target.classList.remove("drop-target");
		}

		target.classList.add("drop-target");
		current_drop_target = target;
	}

	async function handle_drop(e) {
		e.stopPropagation?.();

		const target = e.currentTarget;

		target.classList.remove("drop-target");

		if (current_drop_target) {
			current_drop_target.classList.remove("drop-target");
			current_drop_target = null;
		}

		if (dragged_el && dragged_el !== target && container_el) {
			const dragged_parent = dragged_el.parentElement;
			const target_parent = target.parentElement;
			const is_dragged_in_form = dragged_parent?.classList.contains("form") && dragged_parent.parentElement === container_el;
			const is_target_in_form = target_parent?.classList.contains("form") && target_parent.parentElement === container_el;
			const is_dragged_direct = dragged_parent === container_el;
			const is_target_direct = target_parent === container_el;

			if (!((is_dragged_in_form || is_dragged_direct) && (is_target_in_form || is_target_direct))) {
				return false;
			}

			const placeholder = document.createElement("div");

			placeholder.style.display = "none";
			dragged_parent.insertBefore(placeholder, dragged_el);
			target_parent.insertBefore(dragged_el, target);
			dragged_parent.insertBefore(target, placeholder);
			placeholder.remove();
			dragged_el?.classList.remove("dragging");

			const dragged_handle = dragged_el?.querySelector(".drag-handle");

			if (dragged_handle) {
				dragged_handle.setAttribute("aria-grabbed", "false");
			}

			await tick();
			setup_drag_and_drop();
		}

		return false;
	}

	onMount(() => {
		setup_drag_and_drop();

		const observer = new MutationObserver(setup_drag_and_drop);

		if (container_el) {
			observer.observe(container_el, { childList: true, subtree: false });
		}

		return () => {
			observer.disconnect();

			items.forEach((item) => {
				const handle = item.querySelector(".drag-handle");

				if (handle) {
					remove_drag_listeners(handle);
				}

				remove_drop_listeners(item);
			});
		};
	});

	var div = root();
	let classes;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.shared.loading_status
					? gradio.shared.loading_status.status == "pending" ? "generating" : gradio.shared.loading_status.status
					: null);

				Static($$anchor, spread_props(
					{
						get autoscroll() {
							return gradio.shared.autoscroll;
						},

						get i18n() {
							return gradio.i18n;
						}
					},
					() => gradio.shared.loading_status,
					{
						get status() {
							return get($0);
						}
					}
				));
			}
		};

		if_block(node, ($$render) => {
			if (gradio.shared.loading_status && gradio.props.show_progress) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	slot(node_1, $$props, 'default', {});
	reset(div);
	bind_this(div, ($$value) => container_el = $$value, () => container_el);

	template_effect(
		($0) => {
			set_attribute(div, 'id', gradio.shared.elem_id);

			classes = set_class(div, 1, `draggable ${$0 ?? ''}`, 'svelte-1kr8imm', classes, {
				hide: !gradio.shared.visible,
				horizontal: gradio.props.orientation === "row",
				vertical: gradio.props.orientation === "column"
			});
		},
		[() => (gradio.shared.elem_classes || []).join(' ')]
	);

	append($$anchor, div);
	pop();
}

export { Index as default };
//# sourceMappingURL=Index-CKqdXlH-.js.map
