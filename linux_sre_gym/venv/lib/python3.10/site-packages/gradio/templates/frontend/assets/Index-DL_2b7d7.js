import { p as prop, s as slot, d as bind_this, b as set_class, a as set_attribute, f as set_style, g as spread_props, i as if_block, r as rest_props } from './i18n-CMo5lpzy.js';
import { aj as delegate, R as push, aa as onMount, x as set, w as get, y as user_effect, V as child, X as sibling, Y as reset, t as template_effect, a as append, T as pop, u as state, a5 as user_derived, W as from_html, S as first_child, a6 as comment } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-C97CbKPo.js';
import './clone-DaB-S2nH.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root$1 = from_html(`<div><button class="toggle-button svelte-1uruprb" aria-label="Toggle Sidebar"><div class="chevron svelte-1uruprb"><span class="chevron-left svelte-1uruprb"></span></div></button> <div class="sidebar-content svelte-1uruprb"><!></div></div>`);

function Sidebar($$anchor, $$props) {
	push($$props, true);

	let open = prop($$props, 'open', 15, true),
		position = prop($$props, 'position', 11, "left"),
		elem_classes = prop($$props, 'elem_classes', 19, () => []),
		elem_id = prop($$props, 'elem_id', 3, ""),
		onexpand = prop($$props, 'onexpand', 3, () => {}),
		oncollapse = prop($$props, 'oncollapse', 3, () => {});

	// Using a temporary variable to animate the sidebar opening at the start
	let mounted = state(false);

	let _open = state(false);
	let sidebar_div;
	let overlap_amount = state(0);
	let width_css = user_derived(() => typeof $$props.width === "number" ? `${$$props.width}px` : $$props.width);
	let prefersReducedMotion = state(false);

	// Check if the sidebar overlaps with the main content
	function check_overlap() {
		if (!sidebar_div.closest(".wrap")) return;

		const parent_rect = sidebar_div.closest(".wrap")?.getBoundingClientRect();

		if (!parent_rect) return;

		const sidebar_rect = sidebar_div.getBoundingClientRect();

		const available_space = position() === "left"
			? parent_rect.left
			: window.innerWidth - parent_rect.right;

		set(overlap_amount, Math.max(0, sidebar_rect.width - available_space + 30), true);
	}

	onMount(() => {
		sidebar_div.closest(".wrap")?.classList.add("sidebar-parent");
		check_overlap();
		window.addEventListener("resize", check_overlap);

		const update_parent_overlap = () => {
			document.documentElement.style.setProperty("--overlap-amount", `${get(overlap_amount)}px`);
		};

		update_parent_overlap();
		set(mounted, true);

		const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

		set(prefersReducedMotion, mediaQuery.matches, true);

		const updateMotionPreference = (e) => {
			set(prefersReducedMotion, e.matches, true);
		};

		mediaQuery.addEventListener("change", updateMotionPreference);

		return () => {
			window.removeEventListener("resize", check_overlap);
			mediaQuery.removeEventListener("change", updateMotionPreference);
		};
	});

	// We need to wait for the component to be mounted before we can set the open state
	// so that it animates correctly.
	user_effect(() => {
		if (get(mounted)) set(_open, open(), true);
	});

	let _elem_classes = user_derived(() => elem_classes()?.join(" ") || "");
	var div = root$1();
	let classes;
	var button = child(div);

	button.__click = () => {
		set(_open, !get(_open));
		open(get(_open));

		if (get(_open)) {
			onexpand()?.();
		} else {
			oncollapse()?.();
		}
	};

	var div_1 = sibling(button, 2);
	var node = child(div_1);

	slot(node, $$props, 'default', {});
	reset(div_1);
	reset(div);
	bind_this(div, ($$value) => sidebar_div = $$value, () => sidebar_div);

	template_effect(() => {
		classes = set_class(div, 1, `sidebar ${get(_elem_classes) ?? ''}`, 'svelte-1uruprb', classes, {
			open: get(_open),
			right: position() === "right",
			'reduce-motion': get(prefersReducedMotion)
		});

		set_attribute(div, 'id', elem_id());
		set_style(div, `width: ${get(width_css) ?? ''}; ${position() ?? ''}: calc(${get(width_css) ?? ''} * -1)`);
	});

	append($$anchor, div);
	pop();
}

delegate(['click']);

var root = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	var fragment = root();
	var node = first_child(fragment);

	Static(node, spread_props(
		{
			get autoscroll() {
				return gradio.shared.autoscroll;
			},

			get i18n() {
				return gradio.i18n;
			}
		},
		() => gradio.shared.loading_status
	));

	var node_1 = sibling(node, 2);

	{
		var consequent = ($$anchor) => {
			Sidebar($$anchor, {
				get width() {
					return gradio.props.width;
				},
				onexpand: () => gradio.dispatch("expand"),
				oncollapse: () => gradio.dispatch("collapse"),
				get elem_classes() {
					return gradio.shared.elem_classes;
				},

				get elem_id() {
					return gradio.shared.elem_id;
				},

				get open() {
					return gradio.props.open;
				},

				set open($$value) {
					gradio.props.open = $$value;
				},

				get position() {
					return gradio.props.position;
				},

				set position($$value) {
					gradio.props.position = $$value;
				},

				children: ($$anchor, $$slotProps) => {
					BaseColumn($$anchor, {
						children: ($$anchor, $$slotProps) => {
							var fragment_3 = comment();
							var node_2 = first_child(fragment_3);

							slot(node_2, $$props, 'default', {});
							append($$anchor, fragment_3);
						},
						$$slots: { default: true }
					});
				},
				$$slots: { default: true }
			});
		};

		if_block(node_1, ($$render) => {
			if (gradio.shared.visible) $$render(consequent);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { Index as default };
//# sourceMappingURL=Index-DL_2b7d7.js.map
