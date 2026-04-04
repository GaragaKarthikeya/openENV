const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./ApiDocs-9hYmXNno.js","./i18n-CMo5lpzy.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./i18n-CGFL4PZH.css","./misc-ByVveAau.js","./clear-D8gK36Dy.js","./Button-CH2w4_A4.js","./snippet-Bl5YVoCg.js","./Image-CDw5qb5O.js","./Image-DQfoGqtb.css","./ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js","./ScrollFade-BXTrLZiw.css","./Button-CoUFCYL9.css","./ImagePreview-CAbFrrhc.css","./index-Lz7fsHFt.js","./StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js","./html-CdUEtR5E.js","./index-C7e-J7CF.js","./StreamingBar-M0OJiclA.css","./Clear-Dgs5UhTO.js","./Block-Bx3BikCN.js","./input-C317uaXR.js","./size-Cu4PabCL.js","./state.svelte-CCPBiyik.js","./utils.svelte-CYLB_jIA.js","./clone-DaB-S2nH.js","./index-BJk8VOjP.js","./index-DnoGeqVF.js","./ApiDocs-CKt-ueZo.css","./ApiRecorder-BvE_t-D7.js","./ApiRecorder-CY6I4tgj.css","./Settings-D8txZmif.js","./Dropdown-DUCTuRQm.js","./window-BGXjFJtG.js","./BlockTitle-7XQ-8ENg.js","./Info-BrkJhX6_.js","./DropdownArrow-Dw1NZKCM.js","./IconButtonWrapper-Ba0GidgU.js","./Dropdown-_c0K7TyY.css","./Checkbox-BwK0lQJ8.js","./Checkbox-CJBAs2gn.css","./Settings-BwGmMS8d.css","./Example-ml6sVHU8.css","./Index-DrpT8pjh.js","./Code-C9UfsaaR.js","./Code-DSw5753G.css","./Index.svelte_svelte_type_style_lang-BiKot9ba.js","./actions-Cit_UA5S.js","./Check-BlnBgjLT.js","./Copy-DFFray7x.js","./MarkdownCode-Cg7CjFvQ.js","./prism-python-CbXMXbdR.js","./MarkdownCode-BFC2oRGU.css","./Index-BEAewkRY.css","./Index-Dm9eNXb9.css","./Example-ZgyKzkPU.css","./Example-CNix_cQb.css"])))=>i.map(i=>d[i]);
import { aO as get_boundary, L as current_batch, aP as flatten, h as hydrating, ar as set_hydrating, s as set_hydrate_node, i as hydrate_node, w as get, e as hydrate_next, aJ as skip_nodes, ao as create_text, p as block, aQ as HEAD_EFFECT, C as COMMENT_NODE, j as get_next_sibling, g as get_first_child, R as push, aR as run, aS as async_derived, u as state, y as user_effect, aT as run_after_blockers, a as append, T as pop, W as from_html, x as set, a6 as comment, S as first_child, v as proxy, z as untrack, aa as onMount, X as sibling, t as template_effect, a5 as user_derived, af as __vitePreload, V as child, a0 as set_text, Z as event, aU as deferred_template_effect, Y as reset, aV as $document, a7 as text } from './index-BvBk1Iap.js';
import { d as bind_this, k as each, u as index, i as if_block, g as spread_props, r as rest_props, O as derived, z as $format, j as get$1, P as AsyncFunction, p as prop, f as set_style, q as store_get, m as setup_stores, b as set_class, a as set_attribute, w as writable, A as prefix_css } from './i18n-CMo5lpzy.js';
import { h as html } from './html-CdUEtR5E.js';
import { c as component } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { b as bind_element_size } from './size-Cu4PabCL.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { L as LoadingStatus, T as Toast } from './state.svelte-CCPBiyik.js';
import { I as I18N_MARKER, t as translate_i18n_marker } from './utils.svelte-CYLB_jIA.js';
import { AppTree } from './index-BJk8VOjP.js';

/** @import { TemplateNode, Value } from '#client' */

/**
 * @param {TemplateNode} node
 * @param {Array<Promise<void>>} blockers
 * @param {Array<() => Promise<any>>} expressions
 * @param {(anchor: TemplateNode, ...deriveds: Value[]) => void} fn
 */
function async(node, blockers = [], expressions = [], fn) {
	var boundary = get_boundary();
	var batch = /** @type {Batch} */ (current_batch);
	var blocking = boundary.is_rendered();

	boundary.update_pending_count(1);
	batch.increment(blocking);

	var was_hydrating = hydrating;

	if (was_hydrating) {
		hydrate_next();

		var previous_hydrate_node = hydrate_node;
		var end = skip_nodes(false);
		set_hydrate_node(end);
	}

	flatten(blockers, [], expressions, (values) => {
		if (was_hydrating) {
			set_hydrating(true);
			set_hydrate_node(previous_hydrate_node);
		}

		try {
			// get values eagerly to avoid creating blocks if they reject
			for (const d of values) get(d);

			fn(node, ...values);
		} finally {
			if (was_hydrating) {
				set_hydrating(false);
			}

			boundary.update_pending_count(-1);
			batch.decrement(blocking);
		}
	});
}

/** @import { TemplateNode } from '#client' */

/**
 * @param {string} hash
 * @param {(anchor: Node) => void} render_fn
 * @returns {void}
 */
function head(hash, render_fn) {
	// The head function may be called after the first hydration pass and ssr comment nodes may still be present,
	// therefore we need to skip that when we detect that we're not in hydration mode.
	let previous_hydrate_node = null;
	let was_hydrating = hydrating;

	/** @type {Comment | Text} */
	var anchor;

	if (hydrating) {
		previous_hydrate_node = hydrate_node;

		var head_anchor = get_first_child(document.head);

		// There might be multiple head blocks in our app, and they could have been
		// rendered in an arbitrary order — find one corresponding to this component
		while (
			head_anchor !== null &&
			(head_anchor.nodeType !== COMMENT_NODE || /** @type {Comment} */ (head_anchor).data !== hash)
		) {
			head_anchor = get_next_sibling(head_anchor);
		}

		// If we can't find an opening hydration marker, skip hydration (this can happen
		// if a framework rendered body but not head content)
		if (head_anchor === null) {
			set_hydrating(false);
		} else {
			var start = /** @type {TemplateNode} */ (get_next_sibling(head_anchor));
			head_anchor.remove(); // in case this component is repeated

			set_hydrate_node(start);
		}
	}

	if (!hydrating) {
		anchor = document.head.appendChild(create_text());
	}

	try {
		block(() => render_fn(anchor), HEAD_EFFECT);
	} finally {
		if (was_hydrating) {
			set_hydrating(true);
			set_hydrate_node(/** @type {TemplateNode} */ (previous_hydrate_node));
		}
	}
}

var root$1 = from_html(`<span></span>`);

function MountCustomComponent($$anchor, $$props) {
	push($$props, true);
	var component, runtime, el;

	var $$promises = run([
		async () => component = await async_derived(() => $$props.node.component),
		async () => runtime = await async_derived(() => $$props.node.runtime),
		() => el = state(null),
		() => void user_effect(() => {
			if (!get(el) || !get(runtime) || !get(component)) return;

			// Read prop references so the effect re-runs when the node is
			// replaced during a dev reload (new objects are created by
			// app_tree.reload).
			const _shared_props = $$props.node.props.shared_props;

			const _props = $$props.node.props.props;
			const _runtime = get(runtime);

			const mounted = _runtime.mount(get(component).default, {
				target: get(el),
				props: {
					shared_props: _shared_props,
					props: _props,
					children: $$props.children
				}
			});

			return () => {
				_runtime.unmount(mounted);
			};
		})
	]);

	var span = root$1();

	run_after_blockers([$$promises[2]], () => {
		bind_this(span, ($$value) => set(el, $$value), () => get(el));
	});

	append($$anchor, span);
	pop();
}

function MountComponents($$anchor, $$props) {
	push($$props, true);

	let rest = rest_props($$props, ['$$slots', '$$events', '$$legacy', 'node']);
	var component$1;

	var $$promises = run([
		async () => component$1 = await async_derived(() => $$props.node.component)
	]);

	var fragment = comment();
	var node_1 = first_child(fragment);

	async(node_1, [$$promises[0]], [() => $$props.node && get(component$1)], (node_1, $$condition) => {
		var consequent_4 = ($$anchor) => {
			var fragment_1 = comment();
			var node_2 = first_child(fragment_1);

			async(
				node_2,
				[$$promises[0]],
				[
					() => $$props.node.props.shared_props.visible && !$$props.node.runtime
				],
				(node_2, $$condition) => {
					var consequent_1 = ($$anchor) => {
						var fragment_2 = comment();
						var node_3 = first_child(fragment_2);

						async(node_3, [$$promises[0]], void 0, ($$anchor) => {
							component(node_3, () => get(component$1).default, ($$anchor, $$component) => {
								$$component($$anchor, {
									get shared_props() {
										return $$props.node.props.shared_props;
									},

									get props() {
										return $$props.node.props.props;
									},

									children: ($$anchor, $$slotProps) => {
										var fragment_3 = comment();
										var node_4 = first_child(fragment_3);

										async(node_4, [$$promises[0]], [() => $$props.node.children && $$props.node.children.length], (node_4, $$condition) => {
											var consequent = ($$anchor) => {
												var fragment_4 = comment();
												var node_5 = first_child(fragment_4);

												async(node_5, [$$promises[0]], [() => $$props.node.children], (node_5, $$collection) => {
													each(node_5, 17, () => get($$collection), index, ($$anchor, _node) => {
														MountComponents($$anchor, {
															get node() {
																return get(_node);
															}
														});
													});
												});

												append($$anchor, fragment_4);
											};

											if_block(node_4, ($$render) => {
												if (get($$condition)) $$render(consequent);
											});
										});

										append($$anchor, fragment_3);
									},
									$$slots: { default: true }
								});
							});
						});

						append($$anchor, fragment_2);
					};

					var alternate = ($$anchor) => {
						var fragment_6 = comment();
						var node_6 = first_child(fragment_6);

						async(
							node_6,
							[$$promises[0]],
							[
								() => $$props.node.props.shared_props.visible && $$props.node.runtime
							],
							(node_6, $$condition) => {
								var consequent_3 = ($$anchor) => {
									async($$anchor, [$$promises[0]], void 0, ($$anchor) => {
										MountCustomComponent($$anchor, spread_props(() => rest, {
											get node() {
												return $$props.node;
											},

											children: ($$anchor, $$slotProps) => {
												var fragment_8 = comment();
												var node_7 = first_child(fragment_8);

												async(node_7, [$$promises[0]], [() => $$props.node.children && $$props.node.children.length], (node_7, $$condition) => {
													var consequent_2 = ($$anchor) => {
														var fragment_9 = comment();
														var node_8 = first_child(fragment_9);

														async(node_8, [$$promises[0]], [() => $$props.node.children], (node_8, $$collection) => {
															each(node_8, 17, () => get($$collection), index, ($$anchor, _node) => {
																MountComponents($$anchor, {
																	get node() {
																		return get(_node);
																	}
																});
															});
														});

														append($$anchor, fragment_9);
													};

													if_block(node_7, ($$render) => {
														if (get($$condition)) $$render(consequent_2);
													});
												});

												append($$anchor, fragment_8);
											},
											$$slots: { default: true }
										}));
									});
								};

								if_block(
									node_6,
									($$render) => {
										if (get($$condition)) $$render(consequent_3);
									},
									true
								);
							}
						);

						append($$anchor, fragment_6);
					};

					if_block(node_2, ($$render) => {
						if (get($$condition)) $$render(consequent_1); else $$render(alternate, false);
					});
				}
			);

			append($$anchor, fragment_1);
		};

		if_block(node_1, ($$render) => {
			if (get($$condition)) $$render(consequent_4);
		});
	});

	append($$anchor, fragment);
	pop();
}

function formatter(value) {
  if (value == null) {
    return "";
  }
  const string_value = String(value);
  const translate = get$1($format);
  if (string_value.includes(I18N_MARKER)) {
    return translate_i18n_marker(string_value, translate);
  }
  const direct_translation = translate(string_value);
  if (typeof direct_translation === "string" && direct_translation !== string_value) {
    return direct_translation;
  }
  return string_value;
}
const reactive_formatter = derived($format, () => formatter);

const logo = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='576'%20height='576'%20viewBox='0%200%20576%20576'%20fill='none'%3e%3cpath%20d='M287.5%20229L86%20344.5L287.5%20460L489%20344.5L287.5%20229Z'%20stroke='url(%23paint0_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='round'/%3e%3cpath%20d='M287.5%20116L86%20231.5L287.5%20347L489%20231.5L287.5%20116Z'%20stroke='url(%23paint1_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='round'/%3e%3cpath%20d='M86%20344L288%20229'%20stroke='url(%23paint2_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='bevel'/%3e%3cdefs%3e%3clinearGradient%20id='paint0_linear_102_7'%20x1='60'%20y1='341'%20x2='429.5'%20y2='344'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3clinearGradient%20id='paint1_linear_102_7'%20x1='513.5'%20y1='231'%20x2='143.5'%20y2='231'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3clinearGradient%20id='paint2_linear_102_7'%20x1='60'%20y1='344'%20x2='428.987'%20y2='341.811'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3c/defs%3e%3c/svg%3e";

const api_logo = "data:image/svg+xml,%3csvg%20width='28'%20height='28'%20viewBox='0%200%2028%2028'%20fill='none'%20xmlns='http://www.w3.org/2000/svg'%3e%3cpath%20fill-rule='evenodd'%20clip-rule='evenodd'%20d='M26.9425%202.94265C27.4632%202.42195%2027.4632%201.57773%2026.9425%201.05703C26.4218%200.536329%2025.5776%200.536329%2025.0569%201.05703L22.5713%203.54256C21.1213%202.59333%2019.5367%202.43378%2018.1753%202.64006C16.5495%202.88638%2015.1127%203.66838%2014.3905%204.39053L12.3905%206.39053C12.1405%206.64058%2012%206.97972%2012%207.33334C12%207.68697%2012.1405%208.0261%2012.3905%208.27615L19.7239%2015.6095C20.2446%2016.1302%2021.0888%2016.1302%2021.6095%2015.6095L23.6095%2013.6095C24.3316%2012.8873%2025.1136%2011.4505%2025.36%209.82475C25.5663%208.46312%2025.4066%206.87827%2024.4571%205.42807L26.9425%202.94265Z'%20fill='%233c4555'/%3e%3cpath%20fill-rule='evenodd'%20clip-rule='evenodd'%20d='M12.276%2012.9426C12.7967%2012.4219%2012.7967%2011.5777%2012.276%2011.057C11.7553%2010.5363%2010.9111%2010.5363%2010.3904%2011.057L8.66651%2012.7809L8.27615%2012.3905C8.0261%2012.1405%207.68697%2012%207.33334%2012C6.97972%2012%206.64058%2012.1405%206.39053%2012.3905L4.39053%2014.3905C3.66838%2015.1127%202.88638%2016.5495%202.64006%2018.1753C2.43377%2019.5367%202.59333%2021.1214%203.54262%2022.5714L1.05703%2025.057C0.536329%2025.5777%200.536329%2026.4219%201.05703%2026.9426C1.57773%2027.4633%202.42195%2027.4633%202.94265%2026.9426L5.42817%2024.4571C6.87835%2025.4066%208.46315%2025.5663%209.82475%2025.36C11.4505%2025.1136%2012.8873%2024.3316%2013.6095%2023.6095L15.6095%2021.6095C16.1302%2021.0888%2016.1302%2020.2446%2015.6095%2019.7239L15.2188%2019.3332L16.9426%2017.6093C17.4633%2017.0886%2017.4633%2016.2444%2016.9426%2015.7237C16.4219%2015.203%2015.5777%2015.203%2015.057%2015.7237L13.3332%2017.4475L10.5521%2014.6665L12.276%2012.9426Z'%20fill='%23FF7C00'/%3e%3c/svg%3e";

const settings_logo = "data:image/svg+xml,%3csvg%20width='24'%20height='24'%20viewBox='0%200%2024%2024'%20xmlns='http://www.w3.org/2000/svg'%3e%3c!--%20Outer%20gear%20teeth%20(gray)%20--%3e%3cpath%20d='M19.14%2012.94c.04-.3.06-.61.06-.94%200-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24%200-.43.17-.47.41l-.36%202.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47%200-.59.22L2.74%208.87c-.12.21-.08.47.12.61l2.03%201.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03%201.58c-.18.14-.23.41-.12.61l1.92%203.32c.12.22.37.29.59.22l2.39-.96c.5.38%201.03.7%201.62.94l.36%202.54c.05.24.24.41.48.41h3.84c.24%200%20.44-.17.47-.41l.36-2.54c.59-.24%201.13-.56%201.62-.94l2.39.96c.22.08.47%200%20.59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12%2015.6c-1.98%200-3.6-1.62-3.6-3.6s1.62-3.6%203.6-3.6%203.6%201.62%203.6%203.6-1.62%203.6-3.6%203.6z'%20fill='%23808080'/%3e%3c!--%20Inner%20circle%20(now%20gray)%20--%3e%3ccircle%20cx='12'%20cy='12'%20r='2.5'%20fill='%23808080'/%3e%3c/svg%3e";

const record_stop = "data:image/svg+xml,%3csvg%20viewBox='0%200%2020%2020'%20version='1.1'%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20fill='%23000000'%3e%3cg%20id='SVGRepo_bgCarrier'%20stroke-width='0'%3e%3c/g%3e%3cg%20id='SVGRepo_tracerCarrier'%20stroke-linecap='round'%20stroke-linejoin='round'%3e%3c/g%3e%3cg%20id='SVGRepo_iconCarrier'%3e%3ctitle%3erecord%20[%23982]%3c/title%3e%3cdesc%3eCreated%20with%20Sketch.%3c/desc%3e%3cdefs%3e%3c/defs%3e%3cg%20id='Page-1'%20stroke='none'%20stroke-width='1'%20fill='none'%20fill-rule='evenodd'%3e%3cg%20id='Dribbble-Light-Preview'%20transform='translate(-380.000000,%20-3839.000000)'%20fill='%23FF0000'%3e%3cg%20id='icons'%20transform='translate(56.000000,%20160.000000)'%3e%3cpath%20d='M338,3689%20C338,3691.209%20336.209,3693%20334,3693%20C331.791,3693%20330,3691.209%20330,3689%20C330,3686.791%20331.791,3685%20334,3685%20C336.209,3685%20338,3686.791%20338,3689%20M334,3697%20C329.589,3697%20326,3693.411%20326,3689%20C326,3684.589%20329.589,3681%20334,3681%20C338.411,3681%20342,3684.589%20342,3689%20C342,3693.411%20338.411,3697%20334,3697%20M334,3679%20C328.477,3679%20324,3683.477%20324,3689%20C324,3694.523%20328.477,3699%20334,3699%20C339.523,3699%20344,3694.523%20344,3689%20C344,3683.477%20339.523,3679%20334,3679'%20id='record-[%23982]'%3e%3c/path%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/svg%3e";

let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let removeSegment = {};
let root;
let add_message_callback;
let onRecordingStateChange = null;
let zoomEffects = [];
async function startRecording() {
  if (isRecording) {
    return;
  }
  try {
    const originalTitle = document.title;
    document.title = "[Sharing] Gradio Tab";
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 }
      },
      audio: true,
      selfBrowserSurface: "include"
    });
    document.title = originalTitle;
    const options = {
      videoBitsPerSecond: 5e6
    };
    mediaRecorder = new MediaRecorder(stream, options);
    recordedChunks = [];
    removeSegment = {};
    mediaRecorder.ondataavailable = handleDataAvailable;
    mediaRecorder.onstop = handleStop;
    mediaRecorder.start(1e3);
    isRecording = true;
    if (onRecordingStateChange) ;
    recordingStartTime = Date.now();
  } catch (error) {
    add_message_callback(
      "Recording Error",
      "Failed to start recording: " + error.message,
      "error"
    );
  }
}
function stopRecording() {
  if (!isRecording || !mediaRecorder) {
    return;
  }
  mediaRecorder.stop();
  isRecording = false;
}
function handleDataAvailable(event) {
  if (event.data.size > 0) {
    recordedChunks.push(event.data);
  }
}
function handleStop() {
  isRecording = false;
  const blob = new Blob(recordedChunks, {
    type: "video/mp4"
  });
  handleRecordingComplete(blob);
  const screenStream = mediaRecorder?.stream?.getTracks() || [];
  screenStream.forEach((track) => track.stop());
}
async function handleRecordingComplete(recordedBlob) {
  try {
    add_message_callback(
      "Processing video",
      "This may take a few seconds...",
      "info"
    );
    const formData = new FormData();
    formData.append("video", recordedBlob, "recording.mp4");
    if (removeSegment.start !== void 0 && removeSegment.end !== void 0) {
      formData.append("remove_segment_start", removeSegment.start.toString());
      formData.append("remove_segment_end", removeSegment.end.toString());
    }
    if (zoomEffects.length > 0) {
      formData.append("zoom_effects", JSON.stringify(zoomEffects));
    }
    const response = await fetch(root + "/gradio_api/process_recording", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(
        `Server returned ${response.status}: ${response.statusText}`
      );
    }
    const processedBlob = await response.blob();
    const defaultFilename = `gradio-screen-recording-${(/* @__PURE__ */ new Date()).toISOString().replace(/:/g, "-").replace(/\..+/, "")}.mp4`;
    saveWithDownloadAttribute(processedBlob, defaultFilename);
    zoomEffects = [];
  } catch (error) {
    add_message_callback(
      "Processing Error",
      "Failed to process recording. Saving original version.",
      "warning"
    );
    const defaultFilename = `gradio-screen-recording-${(/* @__PURE__ */ new Date()).toISOString().replace(/:/g, "-").replace(/\..+/, "")}.mp4`;
    saveWithDownloadAttribute(recordedBlob, defaultFilename);
  }
}
function saveWithDownloadAttribute(blob, suggestedName) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

const MESSAGE_QUOTE_RE = /^'([^]+)'$/;
const NOVALUE = /* @__PURE__ */ Symbol("NOVALUE");
class Dependency {
  id;
  inputs;
  outputs;
  cancels;
  pending = false;
  trigger_modes;
  event_args = {};
  targets = [];
  connection_type;
  // if this dependency has any then, success or failure triggers
  triggers = [];
  // the id of the original event_id that caused this dependency to run
  // in the case of chained events, it would be the id of the initial trigger
  original_trigger_id = null;
  show_progress_on = null;
  component_prop_inputs = [];
  show_progress;
  functions;
  constructor(dep_config) {
    this.id = dep_config.id;
    this.original_trigger_id = dep_config.id;
    this.inputs = dep_config.inputs;
    this.outputs = dep_config.outputs;
    this.connection_type = dep_config.connection;
    this.show_progress = dep_config.show_progress;
    this.functions = {
      frontend: dep_config.js ? process_frontend_fn(
        dep_config.js,
        dep_config.backend_fn,
        dep_config.inputs.length,
        dep_config.outputs.length
      ) : void 0,
      backend: dep_config.backend_fn,
      backend_js: dep_config.js_implementation ? new AsyncFunction(
        `let result = await (${dep_config.js_implementation})(...arguments);
						return (!Array.isArray(result)) ? [result] : result;`
      ) : void 0
    };
    this.targets = dep_config.targets;
    this.cancels = dep_config.cancels;
    this.trigger_modes = dep_config.trigger_mode;
    this.show_progress_on = dep_config.show_progress_on || null;
    this.component_prop_inputs = dep_config.component_prop_inputs || [];
    for (let i = 0; i < dep_config.event_specific_args?.length || 0; i++) {
      const key = dep_config.event_specific_args[i];
      this.event_args[key] = dep_config[key] ?? null;
    }
  }
  async run(client, data_payload, event_data, target_id) {
    let _data_payload = data_payload;
    if (this.functions.backend_js) {
      const data = await this.functions.backend_js(..._data_payload);
      return { type: "data", data };
    }
    if (this.functions.frontend) {
      _data_payload = await this.functions.frontend(data_payload);
    }
    if (this.functions.backend) {
      return {
        type: "submit",
        data: client.submit(
          this.id,
          _data_payload,
          event_data,
          target_id,
          void 0,
          { "x-gradio-user": "app" }
        )
      };
    } else if (this.functions.frontend) {
      return { type: "data", data: _data_payload };
    }
    return { type: "void", data: null };
  }
  add_trigger(dep_id, condition) {
    this.triggers.push([dep_id, condition]);
  }
  get_triggers() {
    return {
      success: this.triggers.filter(([, condition]) => condition === "success").map(([id]) => id),
      failure: this.triggers.filter(([, condition]) => condition === "failure").map(([id]) => id),
      all: this.triggers.filter(([, condition]) => condition === "all").map(([id]) => id)
    };
  }
}
class DependencyManager {
  dependencies_by_fn = /* @__PURE__ */ new Map();
  dependencies_by_event = /* @__PURE__ */ new Map();
  render_id_deps = /* @__PURE__ */ new Map();
  submissions = /* @__PURE__ */ new Map();
  client;
  queue = /* @__PURE__ */ new Set();
  add_to_api_calls;
  update_state_cb;
  get_state_cb;
  rerender_cb;
  log_cb;
  on_connection_lost_cb;
  loading_stati = new LoadingStatus();
  connection_lost = false;
  constructor(dependencies, client, update_state_cb, get_state_cb, rerender_cb, log_cb, add_to_api_calls, on_connection_lost_cb) {
    this.add_to_api_calls = add_to_api_calls;
    this.log_cb = log_cb;
    this.update_state_cb = update_state_cb;
    this.get_state_cb = get_state_cb;
    this.rerender_cb = rerender_cb;
    this.on_connection_lost_cb = on_connection_lost_cb;
    this.client = client;
    this.reload(
      dependencies,
      update_state_cb,
      get_state_cb,
      rerender_cb,
      client
    );
  }
  reload(dependencies, update_state, get_state, rerender, client) {
    const { by_id, by_event } = this.create(dependencies);
    this.dependencies_by_event = by_event;
    this.dependencies_by_fn = by_id;
    this.client = client;
    this.update_state_cb = update_state;
    this.get_state_cb = get_state;
    this.rerender_cb = rerender;
    for (const [dep_id, dep] of this.dependencies_by_fn) {
      for (const [output_id] of dep.targets) {
        this.set_event_args(output_id, dep.event_args);
      }
    }
    this.register_loading_stati(by_id);
  }
  register_loading_stati(deps) {
    for (const [_, dep] of deps) {
      this.loading_stati.register(
        dep.id,
        dep.show_progress_on || dep.outputs,
        dep.inputs,
        dep.show_progress
      );
    }
  }
  clear_loading_status(component_id) {
    this.loading_stati.clear(component_id);
  }
  async update_loading_stati_state() {
    for (const [component_id, loading_status] of Object.entries(
      this.loading_stati.current
    )) {
      this.update_state_cb(
        Number(component_id),
        {
          loading_status
        },
        false
      );
    }
  }
  dispatch_state_change_events(result) {
    if (result.changed_state_ids) {
      for (const changed_id of result.changed_state_ids) {
        const change_dep = this.dependencies_by_event.get(
          "change-" + changed_id
        );
        change_dep?.forEach((dep) => {
          this.dispatch({
            type: "fn",
            fn_index: dep.id,
            target_id: changed_id,
            event_data: null
          });
        });
      }
    }
  }
  /** Dispatches an event to the appropriate dependency
   * @param event_name the name of the event
   * @param target_id the id of the component that triggered the event
   * @param event_data any additional data to pass to the dependency
   * @returns a value if there is no backend fn, a 'submission' if there is a backend fn, or null if there is no dependency
   */
  async dispatch(event_meta) {
    if (this.connection_lost) return;
    let deps;
    if (event_meta.type === "fn") {
      const dep = this.dependencies_by_fn.get(event_meta.fn_index);
      if (dep) deps = [dep];
    } else {
      deps = this.dependencies_by_event.get(
        `${event_meta.event_name}-${event_meta.target_id}`
      );
    }
    for (let i = 0; i < (deps?.length || 0); i++) {
      const dep = deps ? deps[i] : void 0;
      if (dep) {
        this.cancel(dep.cancels);
        const dispatch_status = should_dispatch(
          dep.trigger_modes,
          this.submissions.has(dep.id)
        );
        if (dispatch_status === "skip") {
          continue;
        } else if (dispatch_status === "defer") {
          this.queue.add(dep.id);
          continue;
        }
        if (dep.functions.backend) {
          this.loading_stati.update({
            status: "pending",
            fn_index: dep.id,
            stream_state: null
          });
          this.update_loading_stati_state();
        }
        const data_payload = await this.gather_state(
          dep.inputs,
          dep.component_prop_inputs
        );
        const unset_args = await Promise.all(
          dep.targets.map(
            ([output_id]) => this.set_event_args(output_id, dep.event_args)
          )
        );
        const { success, failure, all } = dep.get_triggers();
        try {
          let target_id = null;
          if (event_meta.target_id !== void 0 || event_meta.type === "event") {
            target_id = event_meta.target_id || null;
          } else {
            target_id = dep.original_trigger_id;
          }
          if (dep.connection_type === "stream" && this.submissions.has(dep.id)) {
            const submission = this.submissions.get(dep.id);
            let payload = {
              fn_index: dep.id,
              data: data_payload,
              event_data: event_meta.event_data
            };
            submission.send_chunk(payload);
            unset_args.forEach((fn) => fn());
            continue;
          }
          this.add_to_api_calls({
            fn_index: dep.id,
            data: data_payload,
            event_data: event_meta.event_data,
            trigger_id: target_id
          });
          const dep_submission = await dep.run(
            this.client,
            data_payload,
            event_meta.event_data,
            target_id
          );
          if (dep_submission.type === "void") {
            unset_args.forEach((fn) => fn());
          } else if (dep_submission.type === "data") {
            await this.handle_data(dep.outputs, dep_submission.data);
            unset_args.forEach((fn) => fn());
          } else {
            let stream_state = null;
            if (dep.connection_type === "stream" && !this.submissions.has(dep.id)) {
              stream_state = "waiting";
            }
            this.submissions.set(dep.id, dep_submission.data);
            let index = 0;
            submit_loop: for await (const result of dep_submission.data) {
              if (index === 0) {
                dep.inputs.forEach((input_id) => {
                  this.update_state_cb(
                    input_id,
                    {
                      loading_status: {
                        validation_error: null
                      }
                    },
                    false
                  );
                });
              }
              index += 1;
              if (result === null) continue;
              if (result.type === "data") {
                await this.handle_data(dep.outputs, result.data);
              }
              if (result.type === "status") {
                if (result.original_msg === "process_starts" && dep.connection_type === "stream") {
                  stream_state = "open";
                }
                const { fn_index, ...status } = result;
                if (result.stage === "complete") {
                  stream_state = "closed";
                  success.forEach((dep_id) => {
                    this.dispatch({
                      type: "fn",
                      fn_index: dep_id,
                      event_data: null,
                      target_id
                    });
                  });
                  this.dispatch_state_change_events(result);
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                  break submit_loop;
                } else if (result.stage === "generating") {
                  this.dispatch_state_change_events(result);
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                } else if (result.stage === "error") {
                  if (result.broken || result.session_not_found) {
                    if (!this.connection_lost) {
                      this.connection_lost = true;
                      this.on_connection_lost_cb();
                    }
                    this.loading_stati.update({
                      status: "complete",
                      fn_index: dep.id,
                      stream_state: null
                    });
                    this.update_loading_stati_state();
                    break submit_loop;
                  }
                  if (Array.isArray(result?.message)) {
                    result.message.forEach((m, i2) => {
                      this.update_state_cb(
                        dep.inputs[i2],
                        {
                          loading_status: {
                            validation_error: !m.is_valid ? m.message : null,
                            show_validation_error: true
                          }
                        },
                        false
                      );
                    });
                    dep.outputs.forEach((output_id) => {
                      if (dep.inputs.includes(output_id)) return;
                      this.update_state_cb(
                        output_id,
                        {
                          loading_status: {
                            status: null
                          }
                        },
                        false
                      );
                    });
                    unset_args.forEach((fn) => fn());
                    this.submissions.delete(dep.id);
                    if (this.queue.has(dep.id)) {
                      this.queue.delete(dep.id);
                      this.dispatch(event_meta);
                    }
                    return;
                  }
                  const _message = result?.message?.replace(
                    MESSAGE_QUOTE_RE,
                    (_, b) => b
                  );
                  this.log_cb(
                    //@ts-ignore
                    result?._title ?? "Error",
                    _message || "",
                    fn_index,
                    "error",
                    status.duration,
                    status.visible
                  );
                  throw new Error("Dependency function failed");
                } else {
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                }
              }
              if (result.type === "render") {
                this.loading_stati.update({
                  status: "complete",
                  fn_index: dep.id,
                  stream_state: null
                });
                this.update_loading_stati_state();
                const { layout, components, render_id, dependencies } = result.data;
                this.rerender_cb(components, layout);
                const { by_id, by_event } = this.create(
                  dependencies
                );
                this.register_loading_stati(by_id);
                by_id.forEach(
                  (dep2) => this.dependencies_by_fn.set(dep2.id, dep2)
                );
                by_event.forEach(
                  (dep2, key) => this.dependencies_by_event.set(key, dep2)
                );
                const current_deps = this.render_id_deps.get(render_id);
                if (current_deps) {
                  current_deps.forEach((old_dep_id) => {
                    if (!by_id.has(old_dep_id)) {
                      this.dependencies_by_fn.delete(old_dep_id);
                    }
                  });
                }
                this.render_id_deps.set(
                  render_id,
                  new Set(Array.from(by_id.keys()))
                );
                this.register_loading_stati(by_id);
                break submit_loop;
              }
              if (result.type === "log") {
                this.handle_log(result);
              }
            }
            all.forEach((dep_id) => {
              this.dispatch({
                type: "fn",
                fn_index: dep_id,
                event_data: null,
                target_id
              });
            });
            unset_args.forEach((fn) => fn());
            this.submissions.delete(dep.id);
            if (this.queue.has(dep.id)) {
              this.queue.delete(dep.id);
              this.dispatch(event_meta);
            }
          }
        } catch (error) {
          this.loading_stati.update({
            status: "error",
            fn_index: dep.id,
            eta: 0,
            queue: false,
            stream_state: null
          });
          this.update_loading_stati_state();
          this.submissions.delete(dep.id);
          failure.forEach((dep_id) => {
            this.dispatch({
              type: "fn",
              fn_index: dep_id,
              event_data: null
            });
          });
        }
      }
    }
    return;
  }
  /**
   *  Creates a map of dependencies for easy lookup
   *
   * @param dependencies the list of dependencies from the backend
   * @returns a map of dependencies keyed by `${event_name}-${target_id}`
   * */
  create(dependencies) {
    const _deps_by_id = /* @__PURE__ */ new Map();
    const _deps_by_event = /* @__PURE__ */ new Map();
    const then_triggers = [];
    for (const dep_config of dependencies) {
      const dependency = new Dependency(dep_config);
      for (const [target_id, event_name] of dep_config.targets) {
        if (!_deps_by_event.has(`${event_name}-${target_id}`)) {
          _deps_by_event.set(`${event_name}-${target_id}`, []);
        }
        _deps_by_event.get(`${event_name}-${target_id}`)?.push(dependency);
      }
      _deps_by_id.set(dep_config.id, dependency);
      if (dep_config.trigger_after !== void 0) {
        const then_mode = dep_config.trigger_only_on_failure ? "failure" : dep_config.trigger_only_on_success ? "success" : "all";
        then_triggers.push([
          dep_config.id,
          dep_config.trigger_after,
          then_mode
        ]);
      }
    }
    for (const [dep_id, trigger_after, condition] of then_triggers) {
      const dependency = _deps_by_id.get(trigger_after);
      if (dependency) {
        dependency.add_trigger(dep_id, condition);
        dependency.original_trigger_id = walk_after_to_original(
          dependencies,
          trigger_after
        );
      }
    }
    return { by_id: _deps_by_id, by_event: _deps_by_event };
  }
  handle_log(msg) {
    const { title, log, fn_index, level, duration, visible } = msg;
    this.log_cb(title, log, fn_index, level, duration, visible);
  }
  /**
   *  Updates the state of the outputs based on the data received from the dependency
   *
   * @param outputs the ids of the output components
   * @param data the data to update the components with
   * */
  async handle_data(outputs, data) {
    await Promise.all(
      outputs.map(async (output_id, i) => {
        const _data = data[i] === void 0 ? NOVALUE : data[i];
        if (_data === NOVALUE) return;
        if (is_prop_update(_data)) {
          let pending_visibility_update = false;
          let pending_visibility_value = null;
          for (const [update_key, update_value] of Object.entries(_data)) {
            if (update_key === "__type__") continue;
            if (update_key === "visible") {
              pending_visibility_update = true;
              pending_visibility_value = update_value;
              continue;
            }
            await this.update_state_cb(
              outputs[i],
              {
                [update_key]: update_value
              },
              false
            );
          }
          if (pending_visibility_update) {
            await this.update_state_cb(
              outputs[i],
              {
                visible: pending_visibility_value
              },
              true
            );
          }
        } else {
          await this.update_state_cb(output_id, { value: _data }, false);
        }
      })
    );
  }
  /**
   * Gathers the current state of the inputs
   *
   * @param ids the ids of the components to gather state from
   * @param prop_indices the indices (relative to ids array) that should return all component props instead of just the value
   * @returns an array of the current state of the components, in the same order as the ids
   */
  async gather_state(ids, prop_indices = []) {
    return (await Promise.all(ids.map((id) => this.get_state_cb(id)))).map(
      (state, index) => {
        if (prop_indices.includes(index)) {
          return state ?? null;
        }
        return state?.value ?? null;
      }
    );
  }
  /** Sets the event arguments for a specific component
   *
   * @param id the id of the component to set the event arguments for
   * @param args the event arguments to set
   * @returns a function that can be called to reset the event arguments to their previous values
   */
  async set_event_args(id, args) {
    let current_args = {};
    const current_state = await this.get_state_cb?.(id);
    if (!current_state) return () => {
    };
    for (const [key] of Object.entries(args)) {
      current_args[key] = current_state?.[key] ?? null;
    }
    if (Object.keys(args).length === 0) {
      return () => {
      };
    }
    await this.update_state_cb(id, args, false);
    return () => {
      this.update_state_cb(id, current_args, false);
    };
  }
  async cancel(ids) {
    if (!ids) return;
    for (const id of ids) {
      const submission = this.submissions.get(id);
      if (submission) {
        await submission.cancel();
        this.loading_stati.update({
          status: "complete",
          fn_index: id,
          eta: 0,
          queue: false,
          stream_state: null
        });
        this.update_loading_stati_state();
        this.submissions.delete(id);
        const { failure, all } = this.dependencies_by_fn.get(id)?.get_triggers() || { failure: [], all: [] };
        failure.forEach((dep_id) => {
          this.dispatch({
            type: "fn",
            fn_index: dep_id,
            event_data: null,
            target_id: id
          });
        });
        all.forEach((dep_id) => {
          this.dispatch({
            type: "fn",
            fn_index: dep_id,
            event_data: null,
            target_id: id
          });
        });
      }
    }
  }
  dispatch_load_events() {
    this.dependencies_by_fn.forEach((dep) => {
      dep.targets.forEach(([target_id, event_name]) => {
        if (event_name === "load") {
          this.dispatch({
            type: "fn",
            fn_index: dep.id,
            event_data: null,
            target_id
          });
        }
      });
    });
  }
  get_fns_from_targets(target_id) {
    const fn_indices = [];
    this.dependencies_by_event.forEach((deps, key) => {
      const [, dep_target_id] = key.split("-");
      if (Number(dep_target_id) === target_id) {
        deps.forEach((dep) => {
          fn_indices.push(dep.id);
        });
      }
    });
    return fn_indices;
  }
  close_stream(id) {
    const fn_ids = this.get_fns_from_targets(id);
    for (const fn_id of fn_ids) {
      const submission = this.submissions.get(fn_id);
      if (submission) {
        submission.close_stream();
        this.submissions.delete(fn_id);
      }
      this.loading_stati.update({
        status: "complete",
        fn_index: fn_id,
        eta: 0,
        queue: false,
        stream_state: "closed"
      });
    }
    this.update_loading_stati_state();
  }
}
function is_prop_update(payload) {
  return typeof payload === "object" && payload !== null && "__type__" in payload && payload?.__type__ === "update";
}
function should_dispatch(mode, is_running) {
  if (!is_running) return "run";
  if (mode === "always_last") {
    return "defer";
  } else if (mode === "multiple") {
    return "run";
  } else if (mode === "once") {
    return "skip";
  }
  return "run";
}
function process_frontend_fn(source, backend_fn, input_length, output_length) {
  const wrap = backend_fn ? input_length === 1 : output_length === 1;
  try {
    return new AsyncFunction(
      "__fn_args",
      `  let result = await (${source})(...__fn_args);
  if (typeof result === "undefined") return [];
  return (${wrap} && !Array.isArray(result)) ? [result] : result;`
    );
  } catch (e) {
    throw e;
  }
}
function walk_after_to_original(dependency_map, dep_id) {
  let cache = /* @__PURE__ */ new Map();
  let current_id = dep_id;
  let safety_counter = 0;
  while (safety_counter < 100) {
    const dep = cache.get(current_id) || dependency_map.find((d) => d.id === current_id);
    if (!dep) break;
    cache.set(dep.id, dep);
    if (dep.trigger_after === null || dep.trigger_after === void 0) break;
    current_id = dep.trigger_after;
    safety_counter += 1;
  }
  return current_id;
}

var root_2 = from_html(`<!> <!>`, 1);
var root_6 = from_html(`<button class="show-api svelte-zxu34v"><!> <img class="svelte-zxu34v"/></button>`);
var root_9 = from_html(`<div class="divider show-api-divider svelte-zxu34v">·</div> <a href="https://gradio.app" class="built-with svelte-zxu34v" target="_blank" rel="noreferrer"> <img class="svelte-zxu34v"/></a>`, 1);
var root_10 = from_html(`<div>·</div> <button class="settings svelte-zxu34v"> <img class="svelte-zxu34v"/></button>`, 1);
var root_5 = from_html(`<footer aria-label="Gradio footer navigation" class="svelte-zxu34v"><!> <!> <button> <img class="svelte-zxu34v"/></button> <div class="divider svelte-zxu34v">·</div> <!></footer>`);
var root_11 = from_html(`<div id="api-recorder-container" class="svelte-zxu34v"><!></div>`);
var root_12 = from_html(`<div class="api-docs svelte-zxu34v" role="dialog" aria-modal="true"><div class="backdrop svelte-zxu34v"></div> <div class="api-docs-wrap svelte-zxu34v" role="document"><!></div></div>`);
var root_13 = from_html(`<div class="api-docs svelte-zxu34v" role="dialog" aria-modal="true"><div class="backdrop svelte-zxu34v"></div> <div class="api-docs-wrap svelte-zxu34v" role="document"><!></div></div>`);
var root_1 = from_html(`<div class="wrap svelte-zxu34v"><main class="contain svelte-zxu34v"><!></main> <!> <!> <!> <!> <!></div> <!>`, 1);

function Blocks($$anchor, $$props) {
	push($$props, true);

	const $reactive_formatter = () => store_get(reactive_formatter, '$reactive_formatter', $$stores);
	const $is_screen_recording = () => store_get(is_screen_recording, '$is_screen_recording', $$stores);
	const [$$stores, $$cleanup] = setup_stores();

	prop($$props, 'render_complete', 3, false);
		let ready = prop($$props, 'ready', 15, false),
		reload_count = prop($$props, 'reload_count', 11, 0),
		add_new_message = prop($$props, 'add_new_message', 15);

	$$props.components.forEach((comp) => {
		if (!comp.props.i18n) {
			comp.props.i18n = $reactive_formatter();
		}
	});

	let messages = state(proxy([]));
	let reconnect_interval = null;

	function gradio_event_dispatcher(id, event, data) {
		if (event === "share") {
			const { title, description } = data;

			// trigger_share(title, description);
			// TODO: lets combine all of the into a log type with levels
		} else if (event === "error") {
			new_message("Error", data, -1, event, 10, true);
		} else if (event === "warning") {
			new_message("Warning", data, -1, event, 10, true);
		} else if (event === "info") {
			new_message("Info", data, -1, event, 10, true);
		} else if (event === "gradio_expand" || event === "gradio_tab_select") {
			const id_ = event === "gradio_expand" ? id : data.component_id;

			app_tree.render_previously_invisible_children(id_);
		} else if (event == "clear_status") {
			app_tree.update_state(id, { loading_status: {} }, false);
			dep_manager.clear_loading_status(id);

			// TODO: the loading_status store should handle this via a method
			// update_status(id, "complete", data);
		} else if (event == "close_stream") {
			dep_manager.close_stream(id);
		} else if (event === "custom_button_click") {
			const button_id = data.id;

			dispatch_to_target(button_id, "click", null);
		} else {
			// Tabs are a bit weird. The Tabs component dispatches 'select' events
			// but the target id corresponds to the child Tab component that was selected.
			// So the id we get from the dispatcher belongs to the Tabs,
			// so we need to pull out the correct id here.
			if (event === "select" && id in app_tree.initial_tabs) {
				// this is the id of the selected tab
				id = data.id;
			}

			dep_manager.dispatch({
				type: "event",
				event_name: event,
				target_id: id,
				event_data: data
			});
		}
	}

	let app_tree = new AppTree(
		$$props.components,
		$$props.layout,
		$$props.dependencies,
		{
			root: $$props.root,
			theme: $$props.theme_mode,
			version: $$props.version,
			api_prefix: $$props.api_prefix,
			max_file_size: $$props.max_file_size,
			autoscroll: $$props.autoscroll,
			fill_height: $$props.fill_height
		},
		$$props.app,
		$reactive_formatter(),
		gradio_event_dispatcher
	);

	function dispatch_to_target(target_id, event, data) {
		dep_manager.dispatch({
			type: "event",
			event_name: event,
			target_id,
			event_data: data
		});
	}

	let api_calls = state(proxy([]));
	let last_api_call = state(null);

	// We need a callback to add to api_calls from the DependencyManager
	// We can't update a state variable from inside the DependencyManager because
	// svelte won't see it and won't update the UI.
	let add_to_api_calls = (payload) => {
		set(last_api_call, payload, true);

		if (!get(api_recorder_visible)) return;

		set(api_calls, [...get(api_calls), get(last_api_call)], true);
	};

	function handle_connection_lost() {
		set(messages, get(messages).filter((m) => m.type !== "error"), true);
		++_error_id;

		get(messages).push({
			title: "Connection Lost",
			message: LOST_CONNECTION_MESSAGE,
			fn_index: -1,
			type: "error",
			id: _error_id,
			duration: null,
			visible: true
		});

		reconnect_interval = setInterval(
			async () => {
				try {
					const status = await $$props.app.reconnect();

					if (status === "connected" || status === "changed") {
						clearInterval(reconnect_interval);
						reconnect_interval = null;
						window.location.reload();
					}
				} catch(e) {
					// server still unreachable
					console.debug(e);
				}
			},
			2000
		);
	}

	let dep_manager = new DependencyManager($$props.dependencies, $$props.app, app_tree.update_state.bind(app_tree), app_tree.get_state.bind(app_tree), app_tree.rerender.bind(app_tree), new_message, add_to_api_calls, handle_connection_lost);

	user_effect(() => {
		reload_count();

		untrack(() => {
			app_tree.reload($$props.components, $$props.layout, $$props.dependencies, {
				root: $$props.root,
				theme: $$props.theme_mode,
				version: $$props.version,
				api_prefix: $$props.api_prefix,
				max_file_size: $$props.max_file_size,
				autoscroll: $$props.autoscroll,
				fill_height: $$props.fill_height
			});

			dep_manager.reload($$props.dependencies, app_tree.update_state.bind(app_tree), app_tree.get_state.bind(app_tree), app_tree.rerender.bind(app_tree), $$props.app);
		});
	});

	let vibe_editor_width = 350;

	// export let
	let api_docs_visible = user_derived(() => $$props.search_params.get("view") === "api" && $$props.footer_links.includes("api"));

	let settings_visible = user_derived(() => $$props.search_params.get("view") === "settings");
	let api_recorder_visible = user_derived(() => $$props.search_params.get("view") === "api-recorder" && $$props.footer_links.includes("api"));
	let allow_zoom = true;
	let allow_video_trim = true;

	// Lazy component loading state
	let ApiDocs = null;

	let ApiRecorder = null;
	let Settings = null;
	let VibeEditor = state(null);

	async function loadApiDocs() {
		if (!ApiDocs || !ApiRecorder) {
			const api_docs_module = await __vitePreload(() => import('./ApiDocs-9hYmXNno.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]):void 0,import.meta.url);
			const api_recorder_module = await __vitePreload(() => import('./ApiRecorder-BvE_t-D7.js'),true              ?__vite__mapDeps([30,1,2,3,4,7,8,9,5,10,11,12,13,14,31]):void 0,import.meta.url);

			if (!ApiDocs) ApiDocs = api_docs_module?.default;
			if (!ApiRecorder) ApiRecorder = api_recorder_module?.default;
		}
	}

	async function loadApiRecorder() {
		if (!ApiRecorder) {
			const api_recorder_module = await __vitePreload(() => import('./ApiRecorder-BvE_t-D7.js'),true              ?__vite__mapDeps([30,1,2,3,4,7,8,9,5,10,11,12,13,14,31]):void 0,import.meta.url);

			ApiRecorder = api_recorder_module.default;
		}
	}

	async function loadSettings() {
		if (!Settings) {
			const settings_module = await __vitePreload(() => import('./Settings-D8txZmif.js'),true              ?__vite__mapDeps([32,1,2,3,4,6,11,8,12,33,22,16,17,18,19,34,35,36,37,38,39,40,26,41,23,24,15,20,25,27,28,42,43]):void 0,import.meta.url);

			Settings = settings_module.default;
		}
	}

	async function loadVibeEditor() {
		if (!get(VibeEditor)) {
			const vibe_editor_module = await __vitePreload(() => import('./Index-DrpT8pjh.js'),true              ?__vite__mapDeps([44,1,2,3,4,22,16,17,11,8,12,18,19,45,46,47,48,25,26,49,50,38,51,52,53,54,55,56,57]):void 0,import.meta.url);

			set(VibeEditor, vibe_editor_module.default, true);
		}
	}

	async function set_api_docs_visible(visible) {
		set(api_recorder_visible, false);

		if (visible) {
			await loadApiDocs();
		}

		set(api_docs_visible, visible);

		let params = new URLSearchParams(window.location.search);

		if (visible) {
			params.set("view", "api");
		} else {
			params.delete("view");
		}

		history.replaceState(null, "", "?" + params.toString());
	}

	async function set_settings_visible(visible) {
		if (visible) {
			await loadSettings();
		}

		let params = new URLSearchParams(window.location.search);

		if (visible) {
			params.set("view", "settings");
		} else {
			params.delete("view");
		}

		history.replaceState(null, "", "?" + params.toString());
		set(settings_visible, !get(settings_visible));
	}

	//
	function new_message(
		title,
		message,
		fn_index,
		type,
		duration = 10,
		visible = false
	) {
		if (!visible) return;

		get(messages).push({
			title,
			message,
			fn_index,
			type,
			id: ++_error_id,
			duration,
			visible
		});
	}

	add_new_message(new_message);

	let _error_id = -1;
	$reactive_formatter()("blocks.long_requests_queue");
	$reactive_formatter()("blocks.connection_can_break");
	const LOST_CONNECTION_MESSAGE = "Connection to the server was lost. Attempting reconnection...";
	$reactive_formatter()("blocks.waiting_for_inputs");

	// as state updates are not synchronous, we need to ensure updates are flushed before triggering any requests
	let is_screen_recording = writable(false);

	let footer_height = 0;
	let root_container;

	function handle_resize() {
		if ("parentIFrame" in window) {
			const box = root_container.children[0].getBoundingClientRect();

			if (!box) return;

			window.parentIFrame?.size(box.bottom + footer_height + 32);
		}
	}

	function screen_recording() {
		if ($is_screen_recording()) {
			stopRecording();
		} else {
			startRecording();
		}
	}

	onMount(() => {

		if ("parentIFrame" in window) {
			window.parentIFrame?.autoResize(false);
		}

		const mut = new MutationObserver(handle_resize);
		const res = new ResizeObserver(handle_resize);

		mut.observe(root_container, { childList: true, subtree: true, attributes: true });
		res.observe(root_container);

		app_tree.ready.then(() => {
			ready(true);
			dep_manager.dispatch_load_events();
		});

		if ($$props.vibe_mode) {
			void loadVibeEditor();
		}

		return () => {
			mut.disconnect();
			res.disconnect();

			if (reconnect_interval) clearInterval(reconnect_interval);
		};
	});

	function handle_close(id) {
		set(messages, get(messages).filter((m) => m.id !== id), true);
	}

	var fragment_2 = root_1();

	head('zxu34v', ($$anchor) => {
		var fragment = root_2();
		var node = first_child(fragment);

		{
			var consequent = ($$anchor) => {
				deferred_template_effect(() => {
					$document.title = $$props.title ?? '';
				});
			};

			if_block(node, ($$render) => {
				if ($$props.control_page_title) $$render(consequent);
			});
		}

		var node_1 = sibling(node, 2);

		{
			var consequent_1 = ($$anchor) => {
				var fragment_1 = comment();
				var node_2 = first_child(fragment_1);

				html(node_2, () => `\<style\>${prefix_css($$props.css, $$props.version)}</style>`);
				append($$anchor, fragment_1);
			};

			if_block(node_1, ($$render) => {
				if ($$props.css) $$render(consequent_1);
			});
		}

		append($$anchor, fragment);
	});

	var div = first_child(fragment_2);
	let styles;
	var main = child(div);
	let styles_1;
	var node_3 = child(main);

	MountComponents(node_3, {
		get node() {
			return app_tree.root;
		}
	});

	reset(main);
	bind_this(main, ($$value) => root_container = $$value, () => root_container);

	var node_4 = sibling(main, 2);

	{
		var consequent_6 = ($$anchor) => {
			var footer = root_5();
			var node_5 = child(footer);

			{
				var consequent_3 = ($$anchor) => {
					var button = root_6();
					var node_6 = child(button);

					{
						var consequent_2 = ($$anchor) => {
							var text$1 = text();

							template_effect(($0) => set_text(text$1, $0), [() => $reactive_formatter()("errors.use_via_api_or_mcp")]);
							append($$anchor, text$1);
						};

						var alternate = ($$anchor) => {
							var text_1 = text();

							template_effect(($0) => set_text(text_1, $0), [() => $reactive_formatter()("errors.use_via_api")]);
							append($$anchor, text_1);
						};

						if_block(node_6, ($$render) => {
							if ($$props.app.config?.mcp_server) $$render(consequent_2); else $$render(alternate, false);
						});
					}

					var img = sibling(node_6, 2);

					reset(button);

					template_effect(
						($0) => {
							set_attribute(img, 'src', api_logo);
							set_attribute(img, 'alt', $0);
						},
						[() => $reactive_formatter()("common.logo")]
					);

					event('click', button, () => {
						set_api_docs_visible(!get(api_docs_visible));
					});

					event('mouseenter', button, () => {
						loadApiDocs();
						loadApiRecorder();
					});

					append($$anchor, button);
				};

				if_block(node_5, ($$render) => {
					if ($$props.footer_links.includes("api")) $$render(consequent_3);
				});
			}

			var node_7 = sibling(node_5, 2);

			{
				var consequent_4 = ($$anchor) => {
					var fragment_5 = root_9();
					var a = sibling(first_child(fragment_5), 2);
					var text_2 = child(a);
					var img_1 = sibling(text_2);

					reset(a);

					template_effect(
						($0, $1) => {
							set_text(text_2, `${$0 ?? ''} `);
							set_attribute(img_1, 'src', logo);
							set_attribute(img_1, 'alt', $1);
						},
						[
							() => $reactive_formatter()("common.built_with_gradio"),
							() => $reactive_formatter()("common.logo")
						]
					);

					append($$anchor, fragment_5);
				};

				if_block(node_7, ($$render) => {
					if ($$props.footer_links.includes("gradio")) $$render(consequent_4);
				});
			}

			var button_1 = sibling(node_7, 2);
			let classes;
			var text_3 = child(button_1);
			var img_2 = sibling(text_3);

			reset(button_1);

			var node_8 = sibling(button_1, 4);

			{
				var consequent_5 = ($$anchor) => {
					var fragment_6 = root_10();
					var div_1 = first_child(fragment_6);
					let classes_1;
					var button_2 = sibling(div_1, 2);
					var text_4 = child(button_2);
					var img_3 = sibling(text_4);

					reset(button_2);

					template_effect(
						($0, $1) => {
							classes_1 = set_class(div_1, 1, 'divider svelte-zxu34v', null, classes_1, { hidden: !$is_screen_recording() });
							set_text(text_4, `${$0 ?? ''} `);
							set_attribute(img_3, 'src', settings_logo);
							set_attribute(img_3, 'alt', $1);
						},
						[
							() => $reactive_formatter()("common.settings"),
							() => $reactive_formatter()("common.settings")
						]
					);

					event('click', button_2, () => {
						set_settings_visible(!get(settings_visible));
					});

					event('mouseenter', button_2, () => {
						loadSettings();
					});

					append($$anchor, fragment_6);
				};

				if_block(node_8, ($$render) => {
					if ($$props.footer_links.includes("settings")) $$render(consequent_5);
				});
			}

			reset(footer);

			template_effect(
				($0, $1) => {
					classes = set_class(button_1, 1, 'record svelte-zxu34v', null, classes, { hidden: !$is_screen_recording() });
					set_text(text_3, `${$0 ?? ''} `);
					set_attribute(img_2, 'src', record_stop);
					set_attribute(img_2, 'alt', $1);
				},
				[
					() => $reactive_formatter()("common.stop_recording"),
					() => $reactive_formatter()("common.stop_recording")
				]
			);

			event('click', button_1, () => {
				screen_recording();
			});

			bind_element_size(footer, 'clientHeight', ($$value) => footer_height = $$value);
			append($$anchor, footer);
		};

		if_block(node_4, ($$render) => {
			if ($$props.footer_links.length > 0) $$render(consequent_6);
		});
	}

	var node_9 = sibling(node_4, 2);

	{
		var consequent_7 = ($$anchor) => {
			var div_2 = root_11();
			var node_10 = child(div_2);

			component(node_10, () => ApiRecorder, ($$anchor, $$component) => {
				$$component($$anchor, {
					get api_calls() {
						return get(api_calls);
					},

					get dependencies() {
						return $$props.dependencies;
					}
				});
			});

			reset(div_2);

			event('click', div_2, () => {
				set_api_docs_visible(true);
				set(api_recorder_visible, false);
			});

			append($$anchor, div_2);
		};

		if_block(node_9, ($$render) => {
			if (get(api_recorder_visible) && ApiRecorder) $$render(consequent_7);
		});
	}

	var node_11 = sibling(node_9, 2);

	{
		var consequent_8 = ($$anchor) => {
			var div_3 = root_12();
			var div_4 = child(div_3);
			var div_5 = sibling(div_4, 2);
			var node_12 = child(div_5);

			component(node_12, () => ApiDocs, ($$anchor, $$component) => {
				$$component($$anchor, {
					get root_node() {
						return app_tree.root;
					},

					get dependencies() {
						return $$props.dependencies;
					},

					get root() {
						return $$props.root;
					},

					get app() {
						return $$props.app;
					},

					get space_id() {
						return $$props.space_id;
					},

					get api_calls() {
						return get(api_calls);
					},

					get username() {
						return $$props.username;
					},

					get last_api_call() {
						return get(last_api_call);
					},

					$$events: {
						close: (event) => {
							set_api_docs_visible(false);
							set(api_calls, [], true);
							set(api_recorder_visible, set(api_recorder_visible, event.detail?.api_recorder_visible));
						}
					}
				});
			});

			reset(div_5);
			reset(div_3);
			template_effect(($0) => set_attribute(div_3, 'aria-label', $0), [() => $reactive_formatter()("errors.use_via_api")]);

			event('click', div_4, () => {
				set_api_docs_visible(false);
			});

			append($$anchor, div_3);
		};

		if_block(node_11, ($$render) => {
			if (get(api_docs_visible) && app_tree.root && ApiDocs) $$render(consequent_8);
		});
	}

	var node_13 = sibling(node_11, 2);

	{
		var consequent_9 = ($$anchor) => {
			var div_6 = root_13();
			var div_7 = child(div_6);
			var div_8 = sibling(div_7, 2);
			var node_14 = child(div_8);

			component(node_14, () => Settings, ($$anchor, $$component) => {
				$$component($$anchor, {
					onclose: () => {
						set_settings_visible(false);
					},

					start_recording: () => {
						screen_recording();
					},

					get pwa_enabled() {
						return $$props.app.config.pwa;
					},

					get root() {
						return $$props.root;
					},

					get space_id() {
						return $$props.space_id;
					},

					get i18n() {
						return $reactive_formatter();
					},

					get allow_zoom() {
						return allow_zoom;
					},

					set allow_zoom($$value) {
						allow_zoom = $$value;
					},

					get allow_video_trim() {
						return allow_video_trim;
					},

					set allow_video_trim($$value) {
						allow_video_trim = $$value;
					}
				});
			});

			reset(div_8);
			reset(div_6);
			template_effect(($0) => set_attribute(div_6, 'aria-label', $0), [() => $reactive_formatter()("common.settings")]);

			event('click', div_7, () => {
				set_settings_visible(false);
			});

			append($$anchor, div_6);
		};

		if_block(node_13, ($$render) => {
			if (get(settings_visible) && $$props.app.config && app_tree.root && Settings) $$render(consequent_9);
		});
	}

	var node_15 = sibling(node_13, 2);

	{
		var consequent_10 = ($$anchor) => {
			var fragment_7 = comment();
			var node_16 = first_child(fragment_7);

			component(node_16, () => get(VibeEditor), ($$anchor, $$component) => {
				$$component($$anchor, {
					get app() {
						return $$props.app;
					},

					get root() {
						return $$props.root;
					}
				});
			});

			append($$anchor, fragment_7);
		};

		if_block(node_15, ($$render) => {
			if ($$props.vibe_mode && get(VibeEditor)) $$render(consequent_10);
		});
	}

	reset(div);

	var node_17 = sibling(div, 2);

	{
		var consequent_11 = ($$anchor) => {
			Toast($$anchor, {
				get messages() {
					return get(messages);
				},
				on_close: handle_close
			});
		};

		if_block(node_17, ($$render) => {
			if (get(messages)) $$render(consequent_11);
		});
	}

	template_effect(() => {
		styles = set_style(div, '', styles, { 'min-height': $$props.app_mode ? "100%" : "auto" });

		styles_1 = set_style(main, '', styles_1, {
			'flex-grow': $$props.app_mode ? "1" : "auto",
			'margin-right': $$props.vibe_mode ? `${vibe_editor_width}px` : "0"
		});
	});

	append($$anchor, fragment_2);
	pop();
	$$cleanup();
}

const Blocks$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: Blocks
}, Symbol.toStringTag, { value: 'Module' }));

export { Blocks$1 as B, api_logo as a, settings_logo as s };
//# sourceMappingURL=Blocks-Wq-vWI14.js.map
