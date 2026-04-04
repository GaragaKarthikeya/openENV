import { p as prop, t as remove_input_defaults, b as set_class, k as each, u as index, i as if_block, a as set_attribute, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, aj as delegate, R as push, t as template_effect, T as pop, W as from_html, u as state, v as proxy, x as set, w as get, y as user_effect, X as sibling, a0 as set_text, a5 as user_derived, V as child, a6 as comment, S as first_child, Y as reset } from './index-BvBk1Iap.js';
import { k as key } from './key-BV46G5Fv.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { F as File } from './File-_cC4Z_uD.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { a as bind_checked } from './input-C317uaXR.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root$3 = from_svg(`<svg width="100%" height="100%" viewBox="0 0 14 17" version="1.1" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;"><g transform="matrix(1,0,0,1,-10.6667,-7.73588)"><path d="M12.7,24.033C12.256,24.322 11.806,24.339 11.351,24.084C10.896,23.829 10.668,23.434 10.667,22.9L10.667,9.1C10.667,8.567 10.895,8.172 11.351,7.916C11.807,7.66 12.256,7.677 12.7,7.967L23.567,14.867C23.967,15.133 24.167,15.511 24.167,16C24.167,16.489 23.967,16.867 23.567,17.133L12.7,24.033Z" style="fill:currentColor;fill-rule:nonzero;"></path></g></svg>`);

function ArrowIcon($$anchor) {
	var svg = root$3();

	append($$anchor, svg);
}

var root$2 = from_html(`<input type="checkbox"/>`);

function Checkbox($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15);
	var input = root$2();

	remove_input_defaults(input);
	input.__input = () => $$props.onchange?.(!value());

	let classes;

	template_effect(() => {
		input.disabled = $$props.disabled;
		classes = set_class(input, 1, 'svelte-3g0mcl', null, classes, { disabled: $$props.disabled && !value() });
	});

	bind_checked(input, value);
	append($$anchor, input);
	pop();
}

delegate(['input']);

const FileIcon = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='32'%20height='32'%20viewBox='0%200%2024%2024'%3e%3cpath%20fill='%23888888'%20d='M6%202c-1.1%200-1.99.9-1.99%202L4%2020c0%201.1.89%202%201.99%202H18c1.1%200%202-.9%202-2V8l-6-6H6zm7%207V3.5L18.5%209H13z'/%3e%3c/svg%3e";

const FolderIcon = "data:image/svg+xml,%3c?xml%20version='1.0'%20encoding='UTF-8'%20standalone='no'?%3e%3csvg%20viewBox='0%200%2032%2032'%20version='1.1'%20id='svg7'%20sodipodi:docname='light-folder-new.svg'%20inkscape:version='1.3.2%20(091e20e,%202023-11-25)'%20xmlns:inkscape='http://www.inkscape.org/namespaces/inkscape'%20xmlns:sodipodi='http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'%20xmlns='http://www.w3.org/2000/svg'%20xmlns:svg='http://www.w3.org/2000/svg'%3e%3csodipodi:namedview%20id='namedview7'%20pagecolor='%23ffffff'%20bordercolor='%23000000'%20borderopacity='0.25'%20inkscape:showpageshadow='2'%20inkscape:pageopacity='0.0'%20inkscape:pagecheckerboard='0'%20inkscape:deskcolor='%23d1d1d1'%20inkscape:zoom='7.375'%20inkscape:cx='15.932203'%20inkscape:cy='16'%20inkscape:window-width='1312'%20inkscape:window-height='529'%20inkscape:window-x='0'%20inkscape:window-y='38'%20inkscape:window-maximized='0'%20inkscape:current-layer='svg7'%20/%3e%3cdefs%20id='defs6'%3e%3cclipPath%20id='clipPath1'%3e%3cpath%20d='m69.63%2012.145h-.052c-22.727-.292-46.47%204.077-46.709%204.122-2.424.451-4.946%202.974-5.397%205.397-.044.237-4.414%2023.983-4.122%2046.71-.292%2022.777%204.078%2046.523%204.122%2046.761.451%202.423%202.974%204.945%205.398%205.398.237.044%2023.982%204.413%2046.709%204.121%2022.779.292%2046.524-4.077%2046.761-4.121%202.423-.452%204.946-2.976%205.398-5.399.044-.236%204.413-23.981%204.121-46.709.292-22.777-4.077-46.523-4.121-46.761-.453-2.423-2.976-4.946-5.398-5.397-.238-.045-23.984-4.414-46.71-4.122'%20id='path1'%20/%3e%3c/clipPath%3e%3clinearGradient%20gradientUnits='userSpaceOnUse'%20y2='352.98'%20x2='-601.15'%20y1='663.95'%20x1='-591.02'%20id='2'%3e%3cstop%20stop-color='%23a0a0a0'%20id='stop1'%20/%3e%3cstop%20offset='1'%20stop-color='%23aaa'%20id='stop2'%20/%3e%3c/linearGradient%3e%3clinearGradient%20gradientUnits='userSpaceOnUse'%20y2='354.29'%20x2='-704.05'%20y1='647.77'%20x1='-701.19'%20id='1'%3e%3cstop%20stop-color='%23acabab'%20id='stop3'%20/%3e%3cstop%20offset='1'%20stop-color='%23d4d4d4'%20id='stop4'%20/%3e%3c/linearGradient%3e%3clinearGradient%20id='0'%20x1='59.12'%20y1='-19.888'%20x2='59.15'%20y2='-37.783'%20gradientUnits='userSpaceOnUse'%20gradientTransform='matrix(4.17478%200%200%204.16765-1069.7%20447.73)'%3e%3cstop%20stop-color='%23a0a0a0'%20id='stop5'%20/%3e%3cstop%20offset='1'%20stop-color='%23bdbdbd'%20id='stop6'%20/%3e%3c/linearGradient%3e%3c/defs%3e%3cg%20transform='matrix(.07089%200%200%20.07017%2023.295-40.67)'%20fill='%2360aae5'%20id='g7'%20style='fill:%23888888;fill-opacity:1'%3e%3cpath%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20d='m-884.1%20294.78c-4.626%200-8.349%203.718-8.349%208.335v161.41l468.19%201v-121.2c0-4.618-3.724-8.335-8.35-8.335h-272.65c-8.51.751-9.607-.377-13.812-5.981-5.964-7.968-14.969-21.443-20.84-29.21-4.712-6.805-5.477-6.02-13.292-6.02z'%20fill='url(%230)'%20color='%23000'%20id='path6'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3crect%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20y='356.85'%20x='-890.28'%20height='295.13'%20width='463.85'%20fill='url(%231)'%20stroke='url(%231)'%20stroke-width='2.378'%20rx='9.63'%20id='rect6'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3crect%20width='463.85'%20height='295.13'%20x='-890.28'%20y='356.85'%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20fill='none'%20stroke='url(%232)'%20stroke-linejoin='round'%20stroke-linecap='round'%20stroke-width='5.376'%20rx='9.63'%20id='rect7'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3c/g%3e%3c/svg%3e";

var root_3 = from_html(`<span class="no-checkbox svelte-6im0c4" aria-hidden="true"></span>`);
var root_5 = from_html(`<span role="button" aria-label="expand directory" tabindex="0"><!></span>`);
var root_6 = from_html(`<span class="file-icon svelte-6im0c4"><img alt="file icon" class="svelte-6im0c4"/></span>`);
var root_1$1 = from_html(`<li class="svelte-6im0c4"><span><!> <!> <span class="item-name svelte-6im0c4"> </span></span> <!></li>`);
var root$1 = from_html(`<ul></ul>`);

function FileTree($$anchor, $$props) {
	push($$props, true);

	let path = prop($$props, 'path', 19, () => []),
		index_path = prop($$props, 'index_path', 19, () => []),
		selected_files = prop($$props, 'selected_files', 19, () => []),
		selected_folders = prop($$props, 'selected_folders', 19, () => []),
		is_selected_entirely = prop($$props, 'is_selected_entirely', 3, false),
		selectable = prop($$props, 'selectable', 3, false),
		file_count = prop($$props, 'file_count', 3, "multiple");

	let content = state(proxy([]));
	let opened_folders = state(proxy([]));

	const toggle_open_folder = (i) => {
		if (get(opened_folders).includes(i)) {
			set(opened_folders, get(opened_folders).filter((x) => x !== i), true);
		} else {
			set(opened_folders, [...get(opened_folders), i], true);
		}
	};

	const open_folder = (i) => {
		if (!get(opened_folders).includes(i)) {
			set(opened_folders, [...get(opened_folders), i], true);
		}
	};

	(async () => {
		set(content, await $$props.ls_fn(path()), true);

		if ($$props.valid_for_selection) {
			set(content, [{ name: ".", type: "file" }, ...get(content)], true);
		}

		set(opened_folders, get(content).map((x, i) => x.type === "folder" && (is_selected_entirely() || selected_files().some((y) => y[0] === x.name)) ? i : null).filter((x) => x !== null), true);
	})();

	user_effect(() => {
		if (is_selected_entirely()) {
			get(content).forEach((x) => {
				$$props.oncheck?.({ path: [...path(), x.name], checked: true, type: x.type });
			});
		}
	});

	function handle_select(full_index_path, item_path, _type) {
		$$props.onselect?.({
			index: full_index_path,
			value: item_path.join("/"),
			selected: true
		});
	}

	var ul = root$1();
	let classes;

	each(ul, 21, () => get(content), index, ($$anchor, $$item, i) => {
		let type = () => get($$item).type;
		let name = () => get($$item).name;
		let valid = () => get($$item).valid;
		const is_selected = user_derived(() => (type() === "file" ? selected_files() : selected_folders()).some((x) => x[0] === name() && x.length === 1));
		var li = root_1$1();
		var span = child(li);
		let classes_1;

		span.__click = (e) => {
			e.stopPropagation();

			if (selectable()) {
				handle_select([...index_path(), i], [...path(), name()], type());
			}
		};

		span.__keydown = (e) => {
			if (selectable() && (e.key === " " || e.key === "Enter")) {
				handle_select([...index_path(), i], [...path(), name()], type());
			}
		};

		var node = child(span);

		{
			var consequent_1 = ($$anchor) => {
				var fragment = comment();
				var node_1 = first_child(fragment);

				{
					var consequent = ($$anchor) => {
						var span_1 = root_3();

						append($$anchor, span_1);
					};

					var alternate = ($$anchor) => {
						Checkbox($$anchor, {
							disabled: false,
							get value() {
								return get(is_selected);
							},

							onchange: (checked) => {
								$$props.oncheck?.({ path: [...path(), name()], checked, type: type() });

								if (selectable()) {
									handle_select([...index_path(), i], [...path(), name()], type());
								}

								if (type() === "folder" && checked) {
									open_folder(i);
								}
							}
						});
					};

					if_block(node_1, ($$render) => {
						if (type() === "folder" && file_count() === "single") $$render(consequent); else $$render(alternate, false);
					});
				}

				append($$anchor, fragment);
			};

			if_block(node, ($$render) => {
				if ($$props.interactive) $$render(consequent_1);
			});
		}

		var node_2 = sibling(node, 2);

		{
			var consequent_2 = ($$anchor) => {
				var span_2 = root_5();
				let classes_2;

				span_2.__click = (e) => {
					e.stopPropagation();
					toggle_open_folder(i);
				};

				span_2.__keydown = (e) => {
					if (e.key === " " || e.key === "Enter") {
						toggle_open_folder(i);
					}
				};

				var node_3 = child(span_2);

				ArrowIcon(node_3);
				reset(span_2);
				template_effect(($0) => classes_2 = set_class(span_2, 1, 'icon svelte-6im0c4', null, classes_2, $0), [() => ({ hidden: !get(opened_folders).includes(i) })]);
				append($$anchor, span_2);
			};

			var alternate_1 = ($$anchor) => {
				var span_3 = root_6();
				var img = child(span_3);

				reset(span_3);
				template_effect(() => set_attribute(img, 'src', name() === "." ? FolderIcon : FileIcon));
				append($$anchor, span_3);
			};

			if_block(node_2, ($$render) => {
				if (type() === "folder") $$render(consequent_2); else $$render(alternate_1, false);
			});
		}

		var span_4 = sibling(node_2, 2);
		var text = child(span_4, true);

		reset(span_4);
		reset(span);

		var node_4 = sibling(span, 2);

		{
			var consequent_3 = ($$anchor) => {
				{
					let $0 = user_derived(() => [...path(), name()]);
					let $1 = user_derived(() => [...index_path(), i]);
					let $2 = user_derived(() => selected_files().filter((x) => x[0] === name()).map((x) => x.slice(1)));
					let $3 = user_derived(() => selected_folders().filter((x) => x[0] === name()).map((x) => x.slice(1)));
					let $4 = user_derived(() => selected_folders().some((x) => x[0] === name() && x.length === 1));
					let $5 = user_derived(() => valid() ?? false);

					FileTree($$anchor, {
						get path() {
							return get($0);
						},

						get index_path() {
							return get($1);
						},

						get selected_files() {
							return get($2);
						},

						get selected_folders() {
							return get($3);
						},

						get is_selected_entirely() {
							return get($4);
						},

						get interactive() {
							return $$props.interactive;
						},

						get selectable() {
							return selectable();
						},

						get ls_fn() {
							return $$props.ls_fn;
						},

						get file_count() {
							return file_count();
						},

						get valid_for_selection() {
							return get($5);
						},

						get oncheck() {
							return $$props.oncheck;
						},

						get onselect() {
							return $$props.onselect;
						}
					});
				}
			};

			if_block(node_4, ($$render) => {
				if (type() === "folder" && get(opened_folders).includes(i)) $$render(consequent_3);
			});
		}

		reset(li);

		template_effect(() => {
			classes_1 = set_class(span, 1, 'wrap svelte-6im0c4', null, classes_1, {
				selected: !$$props.interactive && get(is_selected),
				selectable: selectable()
			});

			set_attribute(span, 'role', selectable() ? "button" : undefined);
			set_attribute(span, 'tabindex', selectable() ? 0 : undefined);
			set_text(text, name());
		});

		append($$anchor, li);
	});

	reset(ul);

	template_effect(() => classes = set_class(ul, 1, 'svelte-6im0c4', null, classes, {
		'no-checkboxes': !$$props.interactive,
		root: path().length === 0
	}));

	append($$anchor, ul);
	pop();
}

delegate(['click', 'keydown']);

var root = from_html(`<div class="file-wrap svelte-1gs1iku"><!></div>`);

function DirectoryExplorer($$anchor, $$props) {
	push($$props, true);

	let file_count = prop($$props, 'file_count', 3, "multiple"),
		value = prop($$props, 'value', 15),
		selectable = prop($$props, 'selectable', 3, false);

	let selected_folders = state(proxy([]));

	const paths_equal = (path, path_2) => {
		return path.join("/") === path_2.join("/");
	};

	const path_in_set = (path, set) => {
		return set.some((x) => paths_equal(x, path));
	};

	const path_inside = (path, path_2) => {
		return path.join("/").startsWith(path_2.join("/"));
	};

	var div = root();
	var node = child(div);

	FileTree(node, {
		path: [],
		get selected_files() {
			return value();
		},

		get selected_folders() {
			return get(selected_folders);
		},

		get interactive() {
			return $$props.interactive;
		},

		get selectable() {
			return selectable();
		},

		get ls_fn() {
			return $$props.ls_fn;
		},

		get file_count() {
			return file_count();
		},
		valid_for_selection: false,
		oncheck: (detail) => {
			const { path, checked, type } = detail;

			if (checked) {
				if (file_count() === "single") {
					value([path]);
				} else if (type === "folder") {
					if (!path_in_set(path, get(selected_folders))) {
						set(selected_folders, [...get(selected_folders), path], true);
					}
				} else {
					if (!path_in_set(path, value())) {
						value([...value(), path]);
					}
				}
			} else {
				set(
					selected_folders,
					get(selected_folders).filter((folder) => !path_inside(path, folder)), // deselect all parent folders
					true
				);

				if (type === "folder") {
					set(
						selected_folders,
						get(selected_folders).filter((folder) => !path_inside(folder, path)), // deselect all children folders
						true
					);

					value(value().filter((file) => !path_inside(file, path)) // deselect all children files
					);
				} else {
					value(value().filter((x) => !paths_equal(x, path)));
				}
			}

			$$props.oninput?.();
		},

		get onselect() {
			return $$props.onselect;
		}
	});

	reset(div);
	append($$anchor, div);
	pop();
}

var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props, { value: [] });
	let old_value = state(proxy(gradio.props.value));

	let rerender_key = user_derived(() => [
		gradio.props.root_dir,
		gradio.props.glob,
		gradio.props.ignore_glob
	]);

	// Reset value when rerender_key changes
	// svelte-ignore state_referenced_locally
	let old_rerender_key = state(proxy(get(rerender_key)));

	user_effect(() => {
		if (JSON.stringify(get(old_rerender_key)) != JSON.stringify(get(rerender_key)) && get(old_value) == gradio.props.value) {
			set(old_rerender_key, get(rerender_key), true);
			gradio.props.value = [];
		}
	});

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	{
		let $0 = user_derived(() => gradio.props.value === null ? "dashed" : "solid");

		Block($$anchor, {
			get visible() {
				return gradio.shared.visible;
			},

			get variant() {
				return get($0);
			},
			border_mode: "base",
			padding: false,
			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return gradio.shared.elem_classes;
			},

			get container() {
				return gradio.shared.container;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get min_width() {
				return gradio.shared.min_width;
			},
			allow_overflow: true,
			overflow_behavior: 'auto',
			get height() {
				return gradio.props.height;
			},

			get max_height() {
				return gradio.props.max_height;
			},

			get min_height() {
				return gradio.props.min_height;
			},

			children: ($$anchor, $$slotProps) => {
				var fragment_1 = root_1();
				var node = first_child(fragment_1);

				Static(node, spread_props(() => gradio.shared.loading_status, {
					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get i18n() {
						return gradio.i18n;
					},
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}));

				var node_1 = sibling(node, 2);

				{
					var consequent = ($$anchor) => {
						IconButtonWrapper($$anchor, {
							get buttons() {
								return gradio.props.buttons;
							},

							on_custom_button_click: (id) => {
								gradio.dispatch("custom_button_click", { id });
							}
						});
					};

					if_block(node_1, ($$render) => {
						if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) $$render(consequent);
					});
				}

				var node_2 = sibling(node_1, 2);

				{
					let $0 = user_derived(() => gradio.shared.label || "FileExplorer");

					BlockLabel(node_2, {
						get show_label() {
							return gradio.shared.show_label;
						},

						get Icon() {
							return File;
						},

						get label() {
							return get($0);
						},
						float: false
					});
				}

				var node_3 = sibling(node_2, 2);

				key(node_3, () => get(rerender_key), ($$anchor) => {
					DirectoryExplorer($$anchor, {
						get file_count() {
							return gradio.props.file_count;
						},

						get interactive() {
							return gradio.shared.interactive;
						},

						get selectable() {
							return gradio.props._selectable;
						},

						get ls_fn() {
							return gradio.shared.server.ls;
						},
						oninput: () => gradio.dispatch("input"),
						onselect: (detail) => gradio.dispatch("select", detail),
						get value() {
							return gradio.props.value;
						},

						set value($$value) {
							gradio.props.value = $$value;
						}
					});
				});

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-DZsFFtsx.js.map
