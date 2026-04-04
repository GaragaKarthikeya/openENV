const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./embed-t6zjks9U.js","./mollweide-Ca6MRGcL.js","./ordinal-mJAyPK2n.js","./init-CT6u7j_0.js","./linear-BFftg_y2.js","./step-TZOpqHBK.js","./defaultLocale-DG5JsUjF.js","./time-uewm_MPu.js","./dispatch-tQmgj1It.js","./range-BRcAanmR.js","./index-OyOqWiks.js","./colors-CJG58WzC.js","./dsv-BhAd467f.js","./arc-BEcVRQAW.js"])))=>i.map(i=>d[i]);
import { R as push, u as state, v as proxy, y as user_effect, x as set, w as get, aa as onMount, z as untrack, S as first_child, a7 as text, t as template_effect, a as append, X as sibling, T as pop, a5 as user_derived, af as __vitePreload, W as from_html, a0 as set_text, a8 as next, V as child, Y as reset } from './index-BvBk1Iap.js';
import { i as if_block, r as rest_props, g as spread_props, d as bind_this } from './i18n-CMo5lpzy.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { L as LineChart } from './LineChart-BOORCIAk.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './clone-DaB-S2nH.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './Maximize-B_43ylTs.js';
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

var root_4 = from_html(`<!> <!>`, 1);
var root_9 = from_html(`<p class="caption svelte-19utvcn"> </p>`);
var root_8 = from_html(`<div class="svelte-19utvcn"></div> <!>`, 1);
var root_1 = from_html(`<!> <!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	let unique_colors = user_derived(() => {
		if (!gradio.props.color || !gradio.props.value || gradio.props.value.datatypes[gradio.props.color] !== "nominal") {
			return [];
		}

		const color_index = gradio.props.value.columns.indexOf(gradio.props.color);

		if (color_index === -1) return [];

		return Array.from(new Set(gradio.props.value.data.map((row) => row[color_index])));
	});

	let x_lim = user_derived(() => gradio.props.x_lim || null); // for some unknown reason, x_lim was getting set to undefined when used in re-render, so this line is needed
	let y_lim = user_derived(() => gradio.props.y_lim || null);
	let x_start = user_derived(() => get(x_lim)?.[0] !== null ? get(x_lim)?.[0] : undefined);
	let x_end = user_derived(() => get(x_lim)?.[1] !== null ? get(x_lim)?.[1] : undefined);
	let y_start = user_derived(() => get(y_lim)?.[0] !== null ? get(y_lim)?.[0] : undefined);
	let y_end = user_derived(() => get(y_lim)?.[1] !== null ? get(y_lim)?.[1] : undefined);
	let fullscreen = state(false);

	function reformat_sort(_sort) {
		if (_sort === "x") {
			return "ascending";
		} else if (_sort === "-x") {
			return "descending";
		} else if (_sort === "y") {
			return { field: gradio.props.y, order: "ascending" };
		} else if (_sort === "-y") {
			return { field: gradio.props.y, order: "descending" };
		} else if (_sort === null) {
			return null;
		} else if (Array.isArray(_sort)) {
			return _sort;
		}
	}

	let _sort = user_derived(() => reformat_sort(gradio.props.sort));
	let _data = state(proxy([]));

	function escape_field_name(fieldName) {
		// Escape special characters in field names according to Vega-Lite spec:
		// https://vega.github.io/vega-lite/docs/field.html
		return fieldName.replace(/\./g, "\\.").replace(/\[/g, "\\[").replace(/\]/g, "\\]");
	}

	let x_temporal = user_derived(() => gradio.props.value && gradio.props.value.datatypes[gradio.props.x] === "temporal");

	let _x_lim = user_derived(() => get(x_temporal)
		? [
			get(x_start) !== undefined ? get(x_start) * 1000 : null,
			get(x_end) !== undefined ? get(x_end) * 1000 : null
		]
		: get(x_lim));

	let mouse_down_on_chart = state(false);
	const SUFFIX_DURATION = { s: 1, m: 60, h: 60 * 60, d: 24 * 60 * 60 };

	let _x_bin = user_derived(() => gradio.props.x_bin
		? typeof gradio.props.x_bin === "string"
			? 1000 * parseInt(gradio.props.x_bin.substring(0, gradio.props.x_bin.length - 1)) * SUFFIX_DURATION[gradio.props.x_bin[gradio.props.x_bin.length - 1]]
			: gradio.props.x_bin
		: undefined);

	let _y_aggregate = user_derived(() => {
		if (gradio.props.value) {
			if (gradio.props.value.mark === "point") {
				const aggregating = get(_x_bin) !== undefined;

				return gradio.props.y_aggregate || aggregating ? "sum" : undefined;
			} else {
				return gradio.props.y_aggregate ? gradio.props.y_aggregate : "sum";
			}
		}

		return undefined;
	});

	let aggregating = user_derived(() => {
		if (gradio.props.value) {
			if (gradio.props.value.mark === "point") {
				return get(_x_bin) !== undefined;
			} else {
				return get(_x_bin) !== undefined || gradio.props.value.datatypes[gradio.props.x] === "nominal";
			}
		}

		return false;
	});

	function downsample(data, x_index, y_index, color_index, x_start, x_end) {
		if (data.length < 1000 || gradio.props.x_bin !== null || gradio.props.value?.mark !== "line" || gradio.props.value?.datatypes[gradio.props.x] === "nominal") {
			return data;
		}

		const bin_count = 250;
		let min_max_bins_per_color = {};

		if (x_start === undefined || x_end === undefined) {
			data.forEach((row) => {
				let x_value = row[x_index];

				if (x_start === undefined || x_value < x_start) {
					x_start = x_value;
				}

				if (x_end === undefined || x_value > x_end) {
					x_end = x_value;
				}
			});
		}

		if (x_start === undefined || x_end === undefined) {
			return data;
		}

		const x_range = x_end - x_start;
		const bin_size = x_range / bin_count;

		data.forEach((row, i) => {
			const x_value = row[x_index];
			const y_value = row[y_index];
			const color_value = color_index !== null ? row[color_index] : "any";
			const bin_index = Math.floor((x_value - x_start) / bin_size);

			if (min_max_bins_per_color[color_value] === undefined) {
				min_max_bins_per_color[color_value] = [];
			}

			min_max_bins_per_color[color_value][bin_index] = min_max_bins_per_color[color_value][bin_index] || [
				null,
				Number.POSITIVE_INFINITY,
				null,
				Number.NEGATIVE_INFINITY
			];

			if (y_value < min_max_bins_per_color[color_value][bin_index][1]) {
				min_max_bins_per_color[color_value][bin_index][0] = i;
				min_max_bins_per_color[color_value][bin_index][1] = y_value;
			}

			if (y_value > min_max_bins_per_color[color_value][bin_index][3]) {
				min_max_bins_per_color[color_value][bin_index][2] = i;
				min_max_bins_per_color[color_value][bin_index][3] = y_value;
			}
		});

		const downsampled_data = [];

		Object.values(min_max_bins_per_color).forEach((bins) => {
			bins.forEach(([min_index, _, max_index, __]) => {
				let indices = [];

				if (min_index !== null && max_index !== null) {
					indices = [
						Math.min(min_index, max_index),
						Math.max(min_index, max_index)
					];
				} else if (min_index !== null) {
					indices = [min_index];
				} else if (max_index !== null) {
					indices = [max_index];
				}

				indices.forEach((index) => {
					downsampled_data.push(data[index]);
				});
			});
		});

		return downsampled_data;
	}

	function reformat_data(data, x_start, x_end) {
		let x_index = data.columns.indexOf(gradio.props.x);
		let y_index = data.columns.indexOf(gradio.props.y);
		let color_index = gradio.props.color ? data.columns.indexOf(gradio.props.color) : null;
		let datatable = data.data;

		if (x_start !== undefined && x_end !== undefined) {
			const time_factor = data.datatypes[gradio.props.x] === "temporal" ? 1000 : 1;
			const _x_start = x_start * time_factor;
			const _x_end = x_end * time_factor;
			let largest_before_start = {};
			let smallest_after_end = {};

			const _datatable = datatable.filter((row, i) => {
				const x_value = row[x_index];
				const color_value = color_index !== null ? row[color_index] : "any";

				if (x_value < _x_start && (largest_before_start[color_value] === undefined || x_value > largest_before_start[color_value][1])) {
					largest_before_start[color_value] = [i, x_value];
				}

				if (x_value > _x_end && (smallest_after_end[color_value] === undefined || x_value < smallest_after_end[color_value][1])) {
					smallest_after_end[color_value] = [i, x_value];
				}

				return x_value >= _x_start && x_value <= _x_end;
			});

			datatable = [
				...Object.values(largest_before_start).map(([i, _]) => datatable[i]),
				...downsample(_datatable, x_index, y_index, color_index, _x_start, _x_end),
				...Object.values(smallest_after_end).map(([i, _]) => datatable[i])
			];
		} else {
			datatable = downsample(datatable, x_index, y_index, color_index, undefined, undefined);
		}

		if (gradio.props.tooltip == "all" || Array.isArray(gradio.props.tooltip)) {
			return datatable.map((row) => {
				const obj = {};

				data.columns.forEach((col, i) => {
					obj[col] = row[i];
				});

				return obj;
			});
		}

		return datatable.map((row) => {
			const obj = {
				[gradio.props.x]: row[x_index],
				[gradio.props.y]: row[y_index]
			};

			if (gradio.props.color && color_index !== null) {
				obj[gradio.props.color] = row[color_index];
			}

			return obj;
		});
	}

	user_effect(() => {
		set(
			_data,
			gradio.props.value
				? reformat_data(gradio.props.value, get(x_start), get(x_end))
				: [],
			true
		);
	});

	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value && get(view)) {
			set(old_value, gradio.props.value, true);
			get(view).data("data", get(_data)).runAsync();
		}
	});

	const is_browser = typeof window !== "undefined";
	let chart_element = state(void 0);
	let computed_style = user_derived(() => get(chart_element) ? window.getComputedStyle(get(chart_element)) : null);
	let view = state(void 0);
	let mounted = state(false);
	let old_width = state(0);
	let old_height = state(0);
	let resizeObserver = state(void 0);
	let vegaEmbed;

	async function load_chart() {
		if (get(mouse_down_on_chart)) {
			set(refresh_pending, true);

			return;
		}

		if (get(view)) {
			get(view).finalize();
		}

		if (!gradio.props.value || !get(chart_element)) return;

		set(old_width, get(chart_element).offsetWidth, true);
		set(old_height, get(chart_element).offsetHeight, true);

		const spec = create_vega_lite_spec();

		if (!spec) return;

		set(
			resizeObserver,
			new ResizeObserver((el) => {
				if (!el[0].target || !(el[0].target instanceof HTMLElement)) return;

				if (get(old_width) === 0 && get(chart_element).offsetWidth !== 0 && gradio.props.value.datatypes[gradio.props.x] === "nominal") {
					// a bug where when a nominal chart is first loaded, the width is 0, it doesn't resize
					load_chart();
				} else {
					const width_change = Math.abs(get(old_width) - el[0].target.offsetWidth);
					const height_change = Math.abs(get(old_height) - el[0].target.offsetHeight);

					if (width_change > 100 || height_change > 100) {
						set(old_width, el[0].target.offsetWidth, true);
						set(old_height, el[0].target.offsetHeight, true);
						load_chart();
					} else {
						get(view).signal("width", el[0].target.offsetWidth).run();

						if (get(fullscreen)) {
							get(view).signal("height", el[0].target.offsetHeight).run();
						}
					}
				}
			}),
			true
		);

		if (!vegaEmbed) {
			vegaEmbed = (await __vitePreload(async () => { const {default: __vite_default__} = await import('./embed-t6zjks9U.js');return { default: __vite_default__ }},true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13]):void 0,import.meta.url)).default;
		}

		vegaEmbed(get(chart_element), spec, { actions: false }).then(function (result) {
			set(view, result.view, true);
			get(resizeObserver).observe(get(chart_element));

			var debounceTimeout;
			var lastSelectTime = 0;

			get(view).addEventListener("dblclick", () => {
				gradio.dispatch("double_click");
			});

			// prevent double-clicks from highlighting text
			get(chart_element).addEventListener(
				"mousedown",
				function (e) {
					if (e.detail > 1) {
						e.preventDefault();
					}
				},
				false
			);

			if (gradio.props._selectable) {
				get(view).addSignalListener("brush", function (_, value) {
					if (Date.now() - lastSelectTime < 1000) return;

					set(mouse_down_on_chart, true);

					if (Object.keys(value).length === 0) return;

					clearTimeout(debounceTimeout);

					let range = value[Object.keys(value)[0]];

					if (get(x_temporal)) {
						range = [range[0] / 1000, range[1] / 1000];
					}

					debounceTimeout = setTimeout(
						function () {
							set(mouse_down_on_chart, false);
							lastSelectTime = Date.now();
							gradio.dispatch("select", { value: range, index: range, selected: true });

							if (get(refresh_pending)) {
								set(refresh_pending, false);
								load_chart();
							}
						},
						250
					);
				});
			}
		});
	}

	let refresh_pending = state(false);

	onMount(() => {
		set(mounted, true);

		return () => {
			set(mounted, false);

			if (get(view)) {
				get(view).finalize();
			}

			if (get(resizeObserver)) {
				get(resizeObserver).disconnect();
			}
		};
	});

	function export_chart() {
		if (!get(view) || !get(computed_style)) return;

		const block_background = get(computed_style).getPropertyValue("--block-background-fill");
		const export_background = block_background || "white";

		get(view).background(export_background).run();

		get(view).toImageURL("png", 2).then(function (url) {
			get(view).background("transparent").run();

			const link = document.createElement("a");

			link.setAttribute("href", url);
			link.setAttribute("download", "chart.png");
			link.style.display = "none";
			document.body.appendChild(link);
			link.click();
			document.body.removeChild(link);
		}).catch(function (err) {
			console.error("Export failed:", err);
			get(view).background("transparent").run();
		});
	}

	let _color_map = user_derived(() => JSON.stringify(gradio.props.color_map));

	user_effect(() => {
		// Track dependencies to trigger chart reload
		void gradio.props.title;

		void gradio.props.x_title;
		void gradio.props.y_title;
		void gradio.props.color_title;
		void gradio.props.x;
		void gradio.props.y;
		void gradio.props.color;
		void gradio.props.x_bin;
		void get(_y_aggregate);
		void get(_color_map);
		void gradio.props.colors_in_legend;
		void get(x_start);
		void get(x_end);
		void get(y_start);
		void get(y_end);
		void gradio.props.caption;
		void gradio.props.sort;
		void get(mounted);
		void get(chart_element);
		void get(fullscreen);
		void get(computed_style);

		if (get(mounted) && get(chart_element)) {
			untrack(() => {
				load_chart();
			});
		}
	});

	function create_vega_lite_spec() {
		if (!gradio.props.value || !get(computed_style)) return null;

		let accent_color = get(computed_style).getPropertyValue("--color-accent");
		let body_text_color = get(computed_style).getPropertyValue("--body-text-color");
		let borderColorPrimary = get(computed_style).getPropertyValue("--border-color-primary");
		let font_family = get(computed_style).fontFamily;
		let title_weight = get(computed_style).getPropertyValue("--block-title-text-weight");

		const font_to_px_val = (font) => {
			return font.endsWith("px") ? parseFloat(font.slice(0, -2)) : 12;
		};

		let text_size_md = font_to_px_val(get(computed_style).getPropertyValue("--text-md"));
		let text_size_sm = font_to_px_val(get(computed_style).getPropertyValue("--text-sm"));

		/* eslint-disable complexity */
		return {
			$schema: "https://vega.github.io/schema/vega-lite/v5.17.0.json",
			background: "transparent",
			config: {
				autosize: { type: "fit", contains: "padding" },
				axis: {
					labelFont: font_family,
					labelColor: body_text_color,
					titleFont: font_family,
					titleColor: body_text_color,
					titlePadding: 8,
					tickColor: borderColorPrimary,
					labelFontSize: text_size_sm,
					gridColor: borderColorPrimary,
					titleFontWeight: "normal",
					titleFontSize: text_size_sm,
					labelFontWeight: "normal",
					domain: false,
					labelAngle: 0,
					titleLimit: get(chart_element).offsetHeight * 0.8
				},
				legend: {
					labelColor: body_text_color,
					labelFont: font_family,
					titleColor: body_text_color,
					titleFont: font_family,
					titleFontWeight: "normal",
					titleFontSize: text_size_sm,
					labelFontWeight: "normal",
					offset: 2
				},
				title: {
					color: body_text_color,
					font: font_family,
					fontSize: text_size_md,
					fontWeight: title_weight,
					anchor: "middle"
				},
				view: { stroke: borderColorPrimary },
				mark: {
					stroke: gradio.props.value.mark !== "bar" ? accent_color : undefined,
					fill: gradio.props.value.mark === "bar" ? accent_color : undefined,
					cursor: "crosshair"
				}
			},
			data: { name: "data" },
			datasets: { data: get(_data) },
			layer: [
				"plot",
				...gradio.props.value.mark === "line" ? ["hover"] : []
			].map((mode) => {
				return {
					encoding: {
						size: gradio.props.value.mark === "line"
							? mode == "plot"
								? {
									condition: { empty: false, param: "hoverPlot", value: 3 },
									value: 2
								}
								: {
									condition: { empty: false, param: "hover", value: 100 },
									value: 0
								}
							: undefined,

						opacity: mode === "plot"
							? undefined
							: {
								condition: { empty: false, param: "hover", value: 1 },
								value: 0
							},

						x: {
							axis: {
								...gradio.props.x_label_angle !== null && { labelAngle: gradio.props.x_label_angle },
								labels: gradio.props.x_axis_labels_visible,
								ticks: gradio.props.x_axis_labels_visible,
								...gradio.props.x_axis_format !== null && { format: gradio.props.x_axis_format }
							},
							field: escape_field_name(gradio.props.x),
							title: gradio.props.x_title || gradio.props.x,
							type: gradio.props.value.datatypes[gradio.props.x],
							scale: {
								zero: false,
								domainMin: get(_x_lim)?.[0] !== null ? get(_x_lim)?.[0] : undefined,
								domainMax: get(_x_lim)?.[1] !== null ? get(_x_lim)?.[1] : undefined
							},
							bin: get(_x_bin) ? { step: get(_x_bin) } : undefined,
							sort: get(_sort)
						},
						y: {
							axis: {
								...gradio.props.y_label_angle && { labelAngle: gradio.props.y_label_angle },
								...gradio.props.y_axis_format !== null && { format: gradio.props.y_axis_format }
							},
							field: escape_field_name(gradio.props.y),
							title: gradio.props.y_title || gradio.props.y,
							type: gradio.props.value.datatypes[gradio.props.y],
							scale: {
								zero: false,
								domainMin: get(y_start) ?? undefined,
								domainMax: get(y_end) ?? undefined
							},
							aggregate: get(aggregating) ? get(_y_aggregate) : undefined
						},

						color: gradio.props.color
							? {
								field: escape_field_name(gradio.props.color),
								legend: {
									orient: "bottom",
									title: gradio.props.color_title,
									values: gradio.props.colors_in_legend?.length ? [...gradio.props.colors_in_legend] : undefined
								},

								scale: gradio.props.value.datatypes[gradio.props.color] === "nominal"
									? {
										domain: get(unique_colors),
										range: gradio.props.color_map
											? get(unique_colors).map((c) => gradio.props.color_map[c])
											: undefined
									}
									: {
										range: [100, 200, 300, 400, 500, 600, 700, 800, 900].map((n) => get(computed_style).getPropertyValue("--primary-" + n)),
										interpolate: "hsl"
									},
								type: gradio.props.value.datatypes[gradio.props.color]
							}
							: undefined,

						tooltip: gradio.props.tooltip == "none"
							? undefined
							: [
								{
									field: escape_field_name(gradio.props.y),
									type: gradio.props.value.datatypes[gradio.props.y],
									aggregate: get(aggregating) ? get(_y_aggregate) : undefined,
									title: gradio.props.y_title || gradio.props.y
								},

								{
									field: escape_field_name(gradio.props.x),
									type: gradio.props.value.datatypes[gradio.props.x],
									title: gradio.props.x_title || gradio.props.x,
									format: get(x_temporal) ? "%Y-%m-%d %H:%M:%S" : undefined,
									bin: get(_x_bin) ? { step: get(_x_bin) } : undefined
								},

								...gradio.props.color
									? [
										{
											field: gradio.props.color,
											type: gradio.props.value.datatypes[gradio.props.color]
										}
									]
									: [],

								...gradio.props.tooltip === "axis"
									? []
									: gradio.props.value?.columns.filter((col) => col !== gradio.props.x && col !== gradio.props.y && col !== gradio.props.color && (gradio.props.tooltip === "all" || gradio.props.tooltip.includes(col))).map((column) => ({ field: column, type: gradio.props.value.datatypes[column] }))
							]
					},
					strokeDash: {},
					mark: {
						clip: true,
						type: mode === "hover" ? "point" : gradio.props.value.mark
					},
					name: mode
				};
			}),

			// @ts-ignore
			params: [
				...gradio.props.value.mark === "line"
					? [
						{
							name: "hoverPlot",
							select: {
								clear: "mouseout",
								fields: gradio.props.color ? [gradio.props.color] : [],
								nearest: true,
								on: "mouseover",
								type: "point"
							},
							views: ["hover"]
						},

						{
							name: "hover",
							select: {
								clear: "mouseout",
								nearest: true,
								on: "mouseover",
								type: "point"
							},
							views: ["hover"]
						}
					]
					: [],

				...gradio.props._selectable
					? [
						{
							name: "brush",
							select: {
								encodings: ["x"],
								mark: { fill: "gray", fillOpacity: 0.3, stroke: "none" },
								type: "interval"
							},
							views: ["plot"]
						}
					]
					: []
			],
			width: get(chart_element).offsetWidth,
			height: gradio.props.height || get(fullscreen) ? "container" : undefined,
			title: gradio.props.title || undefined
		};
	}

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

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
		padding: true,
		get height() {
			return gradio.props.height;
		},

		get fullscreen() {
			return get(fullscreen);
		},

		set fullscreen($$value) {
			set(fullscreen, $$value, true);
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
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
							on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
						}
					));
				};

				if_block(node, ($$render) => {
					if (gradio.shared.loading_status) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				var consequent_3 = ($$anchor) => {
					IconButtonWrapper($$anchor, {
						get buttons() {
							return gradio.props.buttons;
						},

						on_custom_button_click: (id) => {
							gradio.dispatch("custom_button_click", { id });
						},

						children: ($$anchor, $$slotProps) => {
							var fragment_4 = root_4();
							var node_2 = first_child(fragment_4);

							{
								var consequent_1 = ($$anchor) => {
									IconButton($$anchor, {
										get Icon() {
											return Download;
										},
										label: 'Export',
										onclick: export_chart
									});
								};

								if_block(node_2, ($$render) => {
									if (gradio.props.buttons?.some((btn) => typeof btn === "string" && btn === "export")) $$render(consequent_1);
								});
							}

							var node_3 = sibling(node_2, 2);

							{
								var consequent_2 = ($$anchor) => {
									FullscreenButton($$anchor, {
										get fullscreen() {
											return get(fullscreen);
										},

										$$events: {
											fullscreen: ({ detail }) => {
												set(fullscreen, detail, true);
											}
										}
									});
								};

								if_block(node_3, ($$render) => {
									if (gradio.props.buttons?.some((btn) => typeof btn === "string" && btn === "fullscreen")) $$render(consequent_2);
								});
							}

							append($$anchor, fragment_4);
						},
						$$slots: { default: true }
					});
				};

				if_block(node_1, ($$render) => {
					if (gradio.props.buttons?.length) $$render(consequent_3);
				});
			}

			var node_4 = sibling(node_1, 2);

			BlockTitle(node_4, {
				get show_label() {
					return gradio.shared.show_label;
				},
				info: undefined,
				children: ($$anchor, $$slotProps) => {
					next();

					var text$1 = text();

					template_effect(() => set_text(text$1, gradio.shared.label));
					append($$anchor, text$1);
				},
				$$slots: { default: true }
			});

			var node_5 = sibling(node_4, 2);

			{
				var consequent_5 = ($$anchor) => {
					var fragment_8 = root_8();
					var div = first_child(fragment_8);

					bind_this(div, ($$value) => set(chart_element, $$value), () => get(chart_element));

					var node_6 = sibling(div, 2);

					{
						var consequent_4 = ($$anchor) => {
							var p = root_9();
							var text_1 = child(p, true);

							reset(p);
							template_effect(() => set_text(text_1, gradio.props.caption));
							append($$anchor, p);
						};

						if_block(node_6, ($$render) => {
							if (gradio.props.caption) $$render(consequent_4);
						});
					}

					append($$anchor, fragment_8);
				};

				var alternate = ($$anchor) => {
					Empty($$anchor, {
						unpadded_box: true,
						children: ($$anchor, $$slotProps) => {
							LineChart($$anchor);
						},
						$$slots: { default: true }
					});
				};

				if_block(node_5, ($$render) => {
					if (gradio.props.value && is_browser) $$render(consequent_5); else $$render(alternate, false);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-DgDgGRDA.js.map
