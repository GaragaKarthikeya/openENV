import { B as BranchManager, d as bind_this, r as rest_props, p as prop, i as if_block, k as each, u as index, b as set_class, g as spread_props, f as set_style, h as clsx } from './i18n-CMo5lpzy.js';
import { p as block, aA as is_promise, O as queue_micro_task, aB as internal_set, aC as Batch, aD as unset_context, aE as is_flushing_sync, U as flushSync, aF as source, aG as UNINITIALIZED, a9 as is_runes, a3 as mutable_source, aH as capture, h as hydrating, e as hydrate_next, aI as HYDRATION_START_ELSE, s as set_hydrate_node, aJ as skip_nodes, ar as set_hydrating, R as push, y as user_effect, w as get, aK as mount, aL as unmount, x as set, a as append, T as pop, u as state, W as from_html, a6 as comment, S as first_child, X as sibling, aj as delegate, a5 as user_derived, Z as event, V as child, t as template_effect, a0 as set_text, Y as reset } from './index-BvBk1Iap.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { k as key } from './key-BV46G5Fv.js';
import './Textbox-v-cNeNHQ.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import Example from './Example-a64hbcxE.js';
import './snippet-Bl5YVoCg.js';
import './clone-DaB-S2nH.js';
import './actions-Cit_UA5S.js';
import './input-C317uaXR.js';
import './BlockTitle-7XQ-8ENg.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './Check-BlnBgjLT.js';
import './Copy-DFFray7x.js';
import './Send-Ee555fP_.js';
import './Square-B6INQV79.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './index-C7e-J7CF.js';
import './size-Cu4PabCL.js';
/* empty css                                               */

/** @import { Source, TemplateNode } from '#client' */

const PENDING = 0;
const THEN = 1;

/** @typedef {typeof PENDING | typeof THEN | typeof CATCH} AwaitState */

/**
 * @template V
 * @param {TemplateNode} node
 * @param {(() => any)} get_input
 * @param {null | ((anchor: Node) => void)} pending_fn
 * @param {null | ((anchor: Node, value: Source<V>) => void)} then_fn
 * @param {null | ((anchor: Node, error: unknown) => void)} catch_fn
 * @returns {void}
 */
function await_block(node, get_input, pending_fn, then_fn, catch_fn) {
	if (hydrating) {
		hydrate_next();
	}

	var runes = is_runes();

	var v = /** @type {V} */ (UNINITIALIZED);
	var value = runes ? source(v) : mutable_source(v, false, false);
	var error = runes ? source(v) : mutable_source(v, false, false);

	var branches = new BranchManager(node);

	block(() => {
		var input = get_input();
		var destroyed = false;

		/** Whether or not there was a hydration mismatch. Needs to be a `let` or else it isn't treeshaken out */
		// @ts-ignore coercing `node` to a `Comment` causes TypeScript and Prettier to fight
		let mismatch = hydrating && is_promise(input) === (node.data === HYDRATION_START_ELSE);

		if (mismatch) {
			// Hydration mismatch: remove everything inside the anchor and start fresh
			set_hydrate_node(skip_nodes());
			set_hydrating(false);
		}

		if (is_promise(input)) {
			var restore = capture();
			var resolved = false;

			/**
			 * @param {() => void} fn
			 */
			const resolve = (fn) => {
				if (destroyed) return;

				resolved = true;
				// We don't want to restore the previous batch here; {#await} blocks don't follow the async logic
				// we have elsewhere, instead pending/resolve/fail states are each their own batch so to speak.
				restore(false);
				// Make sure we have a batch, since the branch manager expects one to exist
				Batch.ensure();

				if (hydrating) {
					// `restore()` could set `hydrating` to `true`, which we very much
					// don't want — we want to restore everything _except_ this
					set_hydrating(false);
				}

				try {
					fn();
				} finally {
					unset_context();

					// without this, the DOM does not update until two ticks after the promise
					// resolves, which is unexpected behaviour (and somewhat irksome to test)
					if (!is_flushing_sync) flushSync();
				}
			};

			input.then(
				(v) => {
					resolve(() => {
						internal_set(value, v);
						branches.ensure(THEN, then_fn && ((target) => then_fn(target, value)));
					});
				},
				(e) => {
					resolve(() => {
						internal_set(error, e);
						branches.ensure(THEN, catch_fn && ((target) => catch_fn(target, error)));

						if (!catch_fn) {
							// Rethrow the error if no catch block exists
							throw error.v;
						}
					});
				}
			);

			if (hydrating) {
				branches.ensure(PENDING, pending_fn);
			} else {
				// Wait a microtask before checking if we should show the pending state as
				// the promise might have resolved by then
				queue_micro_task(() => {
					if (!resolved) {
						resolve(() => {
							branches.ensure(PENDING, pending_fn);
						});
					}
				});
			}
		} else {
			internal_set(value, input);
			branches.ensure(THEN, then_fn && ((target) => then_fn(target, value)));
		}

		if (mismatch) {
			// continue in hydration mode
			set_hydrating(true);
		}

		return () => {
			destroyed = true;
		};
	});
}

var root = from_html(`<span></span>`);

function MountExample($$anchor, $$props) {
	push($$props, true);

	let rest = rest_props($$props, ['$$slots', '$$events', '$$legacy', 'component', 'runtime']);
	let el = state(null);

	user_effect(() => {
		if (!get(el) || !$$props.component) return;

		const _component = $$props.component;
		const _runtime = $$props.runtime;
		const _rest = rest;

		if (_runtime) {
			const mounted = _runtime.mount(_component.default, { target: get(el), props: _rest });

			return () => {
				_runtime.unmount(mounted);
			};
		} else {
			const mounted = mount(_component.default, { target: get(el), props: _rest });

			return () => {
				unmount(mounted);
			};
		}
	});

	var span = root();

	bind_this(span, ($$value) => set(el, $$value), () => get(el));
	append($$anchor, span);
	pop();
}

var root_5 = from_html(`<button class="gallery-item svelte-16f20a1"><!></button>`);
var root_3 = from_html(`<div class="gallery svelte-16f20a1"></div>`);
var root_13 = from_html(`<th class="svelte-16f20a1"> </th>`);
var root_16 = from_html(`<td><!></td>`);
var root_14 = from_html(`<tr class="tr-body svelte-16f20a1"></tr>`);
var root_12 = from_html(`<div class="table-wrap svelte-16f20a1"><table tabindex="0" role="grid" class="svelte-16f20a1"><thead><tr class="tr-head svelte-16f20a1"></tr></thead><tbody></tbody></table></div>`);
var root_20 = from_html(`<div>...</div>`);
var root_21 = from_html(`<button> </button>`);
var root_18 = from_html(`<div class="paginate svelte-16f20a1">Pages: <!></div>`);
var root_2$1 = from_html(`<!> <!>`, 1);

function Dataset($$anchor, $$props) {
	push($$props, true);

	let sample_labels = prop($$props, 'sample_labels', 3, null),
		value = prop($$props, 'value', 15, null),
		samples_per_page = prop($$props, 'samples_per_page', 3, 10),
		layout = prop($$props, 'layout', 3, null);

	// Although the `samples_dir` prop is not used in any of the core Gradio component, it is kept for backward compatibility
	// with any custom components created with gradio<=4.20.0
	let samples_dir = user_derived(() => $$props.proxy_url
		? `/proxy=${$$props.proxy_url}file=`
		: `${$$props.root}/file=`);

	let current_hover = state(-1);
	let gallery = user_derived(() => ($$props.components.length < 2 || sample_labels() !== null) && layout() !== "table");

	let effective_samples = user_derived(() => {
		if (sample_labels()) {
			return sample_labels().map((e) => [e]);
		}

		return $$props.samples ?? [];
	});

	// page resets to 0 whenever effective_samples changes,
	// but can still be overwritten by user clicks
	let page = user_derived(() => {
		get(effective_samples);

		return 0;
	});

	let paginate = user_derived(() => get(effective_samples).length > samples_per_page());

	let selected_samples = user_derived(() => {
		if (get(paginate)) {
			return get(effective_samples).slice(get(page) * samples_per_page(), (get(page) + 1) * samples_per_page());
		}

		return get(effective_samples).slice();
	});

	let page_count = user_derived(() => Math.ceil(get(effective_samples).length / samples_per_page()));

	let visible_pages = user_derived(() => {
		if (!get(paginate)) return [];

		let pages = [];

		[0, get(page), get(page_count) - 1].forEach((anchor) => {
			for (let i = anchor - 2; i <= anchor + 2; i++) {
				if (i >= 0 && i < get(page_count) && !pages.includes(i)) {
					if (pages.length > 0 && i - pages[pages.length - 1] > 1) {
						pages.push(-1);
					}

					pages.push(i);
				}
			}
		});

		return pages;
	});

	function handle_mouseenter(i) {
		set(current_hover, i, true);
	}

	function handle_mouseleave() {
		set(current_hover, -1);
	}

	let component_meta = [];

	async function get_component_meta(selected_samples_json) {
		const _selected_samples = JSON.parse(selected_samples_json);

		// @ts-ignore
		component_meta = await Promise.all(_selected_samples && _selected_samples.map(async (sample_row) => await Promise.all(sample_row.map(async (sample_cell, j) => {
			const loaded = $$props.load_component($$props.components[j].name, "example", $$props.components[j].class_id);

			return {
				value: sample_cell,
				component: loaded.component,
				runtime: loaded.runtime
			};
		}))));
	}

	// Need to stringify the samples otherwise get_component_meta will trigger infinitely
	// Saw this when rendering examples in a gr.render block
	let selected_samples_json = user_derived(() => JSON.stringify(get(selected_samples) || []));

	var fragment = comment();
	var node = first_child(fragment);

	await_block(node, () => get_component_meta(get(selected_samples_json)), null, ($$anchor, _) => {
		var fragment_1 = root_2$1();
		var node_1 = first_child(fragment_1);

		{
			var consequent_3 = ($$anchor) => {
				var div = root_3();

				each(div, 21, () => get(selected_samples), index, ($$anchor, sample_row, i) => {
					var fragment_2 = comment();
					var node_2 = first_child(fragment_2);

					{
						var consequent_2 = ($$anchor) => {
							var button = root_5();

							button.__click = () => {
								value(i + get(page) * samples_per_page());
								$$props.onclick({ index: value(), value: get(sample_row) });
								$$props.onselect({ index: value(), value: get(sample_row) });
							};

							var node_3 = child(button);

							{
								var consequent = ($$anchor) => {
									{
										let $0 = user_derived(() => get(current_hover) === i);

										Example($$anchor, {
											get value() {
												return get(sample_row)[0];
											},

											get selected() {
												return get($0);
											},
											type: 'gallery'
										});
									}
								};

								var alternate = ($$anchor) => {
									var fragment_4 = comment();
									var node_4 = first_child(fragment_4);

									{
										var consequent_1 = ($$anchor) => {
											var fragment_5 = comment();
											var node_5 = first_child(fragment_5);

											await_block(node_5, () => Promise.all([component_meta[0][0].component, component_meta[0][0].runtime]), null, ($$anchor, $$source) => {
												var $$value = user_derived(() => {
													var [component, runtime] = get($$source);

													return { component, runtime };
												});

												var component = user_derived(() => get($$value).component);
												var runtime = user_derived(() => get($$value).runtime);
												var fragment_6 = comment();
												var node_6 = first_child(fragment_6);

												key(node_6, () => get(sample_row)[0], ($$anchor) => {
													{
														let $0 = user_derived(() => get(current_hover) === i);

														MountExample($$anchor, spread_props(
															{
																get component() {
																	return get(component);
																},

																get runtime() {
																	return get(runtime);
																}
															},
															() => $$props.component_props[0],
															{
																get value() {
																	return get(sample_row)[0];
																},

																get samples_dir() {
																	return get(samples_dir);
																},
																type: 'gallery',
																get selected() {
																	return get($0);
																},
																index: i,
																get root() {
																	return $$props.root;
																}
															}
														));
													}
												});

												append($$anchor, fragment_6);
											});

											append($$anchor, fragment_5);
										};

										if_block(
											node_4,
											($$render) => {
												if (component_meta.length) $$render(consequent_1);
											},
											true
										);
									}

									append($$anchor, fragment_4);
								};

								if_block(node_3, ($$render) => {
									if (sample_labels()) $$render(consequent); else $$render(alternate, false);
								});
							}

							reset(button);
							event('mouseenter', button, () => handle_mouseenter(i));
							event('mouseleave', button, () => handle_mouseleave());
							append($$anchor, button);
						};

						if_block(node_2, ($$render) => {
							if (get(sample_row)[0] != null) $$render(consequent_2);
						});
					}

					append($$anchor, fragment_2);
				});

				reset(div);
				append($$anchor, div);
			};

			var alternate_1 = ($$anchor) => {
				var fragment_8 = comment();
				var node_7 = first_child(fragment_8);

				{
					var consequent_5 = ($$anchor) => {
						var div_1 = root_12();
						var table = child(div_1);
						var thead = child(table);
						var tr = child(thead);

						each(tr, 21, () => $$props.headers, index, ($$anchor, header) => {
							var th = root_13();
							var text = child(th, true);

							reset(th);
							template_effect(() => set_text(text, get(header)));
							append($$anchor, th);
						});

						reset(tr);
						reset(thead);

						var tbody = sibling(thead);

						each(tbody, 21, () => component_meta, index, ($$anchor, sample_row, i) => {
							var tr_1 = root_14();

							tr_1.__click = () => {
								value(i + get(page) * samples_per_page());
								$$props.onclick({ index: value(), value: get(sample_row) });
								$$props.onselect({ index: value(), value: get(selected_samples)[i] });
							};

							each(tr_1, 21, () => get(sample_row), index, ($$anchor, $$item, j, $$array) => {
								let value = () => get($$item).value;
								let component = () => get($$item).component;
								let runtime = () => get($$item).runtime;
								const component_name = user_derived(() => $$props.components[j]);
								var fragment_9 = comment();
								var node_8 = first_child(fragment_9);

								{
									var consequent_4 = ($$anchor) => {
										var td = root_16();
										var node_9 = child(td);

										await_block(node_9, () => Promise.all([component(), runtime()]), null, ($$anchor, $$source) => {
											var $$value = user_derived(() => {
												var [component, runtime] = get($$source);

												return { component, runtime };
											});

											var component = user_derived(() => get($$value).component);
											var runtime = user_derived(() => get($$value).runtime);

											{
												let $0 = user_derived(() => get(current_hover) === i);

												MountExample($$anchor, spread_props(
													{
														get component() {
															return get(component);
														},

														get runtime() {
															return get(runtime);
														}
													},
													() => $$props.component_props[j],
													{
														get value() {
															return value();
														},

														get samples_dir() {
															return get(samples_dir);
														},
														type: 'table',
														get selected() {
															return get($0);
														},
														index: i,
														get root() {
															return $$props.root;
														}
													}
												));
											}
										});

										reset(td);

										template_effect(() => {
											set_style(td, `max-width: ${get(component_name) === 'textbox' ? '35ch' : 'auto'}`);
											set_class(td, 1, clsx(get(component_name)), 'svelte-16f20a1');
										});

										append($$anchor, td);
									};

									if_block(node_8, ($$render) => {
										if (get(component_name) !== undefined) $$render(consequent_4);
									});
								}

								append($$anchor, fragment_9);
							});

							reset(tr_1);
							event('mouseenter', tr_1, () => handle_mouseenter(i));
							event('mouseleave', tr_1, () => handle_mouseleave());
							append($$anchor, tr_1);
						});

						reset(tbody);
						reset(table);
						reset(div_1);
						append($$anchor, div_1);
					};

					if_block(
						node_7,
						($$render) => {
							if (get(selected_samples).length > 0) $$render(consequent_5);
						},
						true
					);
				}

				append($$anchor, fragment_8);
			};

			if_block(node_1, ($$render) => {
				if (get(gallery)) $$render(consequent_3); else $$render(alternate_1, false);
			});
		}

		var node_10 = sibling(node_1, 2);

		{
			var consequent_7 = ($$anchor) => {
				var div_2 = root_18();
				var node_11 = sibling(child(div_2));

				each(node_11, 17, () => get(visible_pages), index, ($$anchor, visible_page) => {
					var fragment_11 = comment();
					var node_12 = first_child(fragment_11);

					{
						var consequent_6 = ($$anchor) => {
							var div_3 = root_20();

							append($$anchor, div_3);
						};

						var alternate_2 = ($$anchor) => {
							var button_1 = root_21();

							button_1.__click = () => set(page, get(visible_page));

							let classes;
							var text_1 = child(button_1, true);

							reset(button_1);

							template_effect(() => {
								classes = set_class(button_1, 1, 'svelte-16f20a1', null, classes, { 'current-page': get(page) === get(visible_page) });
								set_text(text_1, get(visible_page) + 1);
							});

							append($$anchor, button_1);
						};

						if_block(node_12, ($$render) => {
							if (get(visible_page) === -1) $$render(consequent_6); else $$render(alternate_2, false);
						});
					}

					append($$anchor, fragment_11);
				});

				reset(div_2);
				append($$anchor, div_2);
			};

			if_block(node_10, ($$render) => {
				if (get(paginate)) $$render(consequent_7);
			});
		}

		append($$anchor, fragment_1);
	});

	append($$anchor, fragment);
	pop();
}

delegate(['click']);

var root_2 = from_html(`<div class="label svelte-bnxc4d"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" class="svelte-bnxc4d"><path fill="currentColor" d="M10 6h18v2H10zm0 18h18v2H10zm0-9h18v2H10zm-6 0h2v2H4zm0-9h2v2H4zm0 18h2v2H4z"></path></svg> </div>`);
var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	// Need to mark samples as state, otherwise get_component_meta constantly triggers
	let samples = user_derived(() => gradio.props.samples ?? []);

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},
		padding: false,
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		allow_overflow: false,
		container: false,
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					var div = root_2();
					var text = sibling(child(div));

					reset(div);
					template_effect(() => set_text(text, ` ${(gradio.shared.label || "Examples") ?? ''}`));
					append($$anchor, div);
				};

				if_block(node, ($$render) => {
					if (gradio.shared.show_label) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			Dataset(node_1, spread_props(
				{
					onclick: (d) => (
						gradio.props.value = d.index,
						gradio.dispatch("click", gradio.props.value)
					),
					onselect: (data) => gradio.dispatch("select", data),
					get load_component() {
						return gradio.shared.load_component;
					},

					get samples() {
						return get(samples);
					}
				},
				() => gradio.props
			));

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-N-hoPZP9.js.map
