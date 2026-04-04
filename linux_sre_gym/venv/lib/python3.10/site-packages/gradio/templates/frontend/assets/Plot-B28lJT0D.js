const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./PlotlyPlot-CNehIspL.js","./i18n-CMo5lpzy.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./i18n-CGFL4PZH.css","./BokehPlot-CdY9Y2o7.js","./BokehPlot-6bmWV9d6.css","./MatplotlibPlot-DHrDLUuI.js","./MatplotlibPlot-CaRv3arg.css","./AltairPlot-BmjchJ6O.js","./color-Tpcfv14B.js","./mollweide-Ca6MRGcL.js","./ordinal-mJAyPK2n.js","./init-CT6u7j_0.js","./linear-BFftg_y2.js","./step-TZOpqHBK.js","./defaultLocale-DG5JsUjF.js","./time-uewm_MPu.js","./dispatch-tQmgj1It.js","./range-BRcAanmR.js","./index-OyOqWiks.js","./colors-CJG58WzC.js","./dsv-BhAd467f.js","./arc-BEcVRQAW.js","./AltairPlot-qXQqL0TB.css"])))=>i.map(i=>d[i]);
import { a as append, f as from_svg, R as push, y as user_effect, z as untrack, x as set, w as get, a6 as comment, S as first_child, T as pop, u as state, af as __vitePreload } from './index-BvBk1Iap.js';
import { i as if_block } from './i18n-CMo5lpzy.js';
import { k as key } from './key-BV46G5Fv.js';
import { c as component } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { b as bubble_event } from './misc-ByVveAau.js';
import { E as Empty } from './Empty-CHa-gxv7.js';

var root = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--carbon" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><circle cx="20" cy="4" r="2" fill="currentColor"></circle><circle cx="8" cy="16" r="2" fill="currentColor"></circle><circle cx="28" cy="12" r="2" fill="currentColor"></circle><circle cx="11" cy="7" r="2" fill="currentColor"></circle><circle cx="16" cy="24" r="2" fill="currentColor"></circle><path fill="currentColor" d="M30 3.413L28.586 2L4 26.585V2H2v26a2 2 0 0 0 2 2h26v-2H5.413Z"></path></svg>`);

function Plot$2($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

function Plot($$anchor, $$props) {
	push($$props, true);

	//@ts-nocheck
	let PlotComponent = state(null);

	let loaded_plotly_css = state(false);
	let key$1 = state(0);

	const plotTypeMapping = {
		plotly: () => __vitePreload(() => import('./PlotlyPlot-CNehIspL.js'),true              ?__vite__mapDeps([0,1,2,3,4]):void 0,import.meta.url),
		bokeh: () => __vitePreload(() => import('./BokehPlot-CdY9Y2o7.js'),true              ?__vite__mapDeps([5,1,2,3,4,6]):void 0,import.meta.url),
		matplotlib: () => __vitePreload(() => import('./MatplotlibPlot-DHrDLUuI.js'),true              ?__vite__mapDeps([7,1,2,3,4,8]):void 0,import.meta.url),
		altair: () => __vitePreload(() => import('./AltairPlot-BmjchJ6O.js'),true              ?__vite__mapDeps([9,1,2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]):void 0,import.meta.url)
	};

	let loadedPlotTypeMapping = {};
	const is_browser = typeof window !== "undefined";
	let _type = state(null);

	user_effect(() => {
		let type = $$props.value?.type;

		untrack(() => {
			set(key$1, get(key$1) + 1);

			if (type !== get(_type)) {
				set(PlotComponent, null);
			}

			if (type && type in plotTypeMapping && is_browser) {
				if (loadedPlotTypeMapping[type]) {
					set(PlotComponent, loadedPlotTypeMapping[type], true);
				} else {
					plotTypeMapping[type]().then((module) => {
						set(PlotComponent, module.default, true);
						loadedPlotTypeMapping[type] = get(PlotComponent);
					});
				}
			}

			set(_type, type, true);
		});

		$$props.on_change();
	});

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			var fragment_1 = comment();
			var node_1 = first_child(fragment_1);

			key(node_1, () => get(key$1), ($$anchor) => {
				var fragment_2 = comment();
				var node_2 = first_child(fragment_2);

				component(node_2, () => get(PlotComponent), ($$anchor, PlotComponent_1) => {
					PlotComponent_1($$anchor, {
						get value() {
							return $$props.value;
						},
						colors: [],
						get theme_mode() {
							return $$props.theme_mode;
						},

						get show_label() {
							return $$props.show_label;
						},

						get caption() {
							return $$props.caption;
						},

						get bokeh_version() {
							return $$props.bokeh_version;
						},

						get show_actions_button() {
							return $$props.show_actions_button;
						},

						get _selectable() {
							return $$props._selectable;
						},

						get x_lim() {
							return $$props.x_lim;
						},

						get loaded_plotly_css() {
							return get(loaded_plotly_css);
						},

						set loaded_plotly_css($$value) {
							set(loaded_plotly_css, $$value, true);
						},

						$$events: {
							select($$arg) {
								bubble_event.call(this, $$props, $$arg);
							}
						}
					});
				});

				append($$anchor, fragment_2);
			});

			append($$anchor, fragment_1);
		};

		var alternate = ($$anchor) => {
			Empty($$anchor, {
				unpadded_box: true,
				size: 'large',
				children: ($$anchor, $$slotProps) => {
					Plot$2($$anchor);
				},
				$$slots: { default: true }
			});
		};

		if_block(node, ($$render) => {
			if ($$props.value && get(PlotComponent)) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

const Plot$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: Plot
}, Symbol.toStringTag, { value: 'Module' }));

export { Plot$2 as P, Plot as a, Plot$1 as b };
//# sourceMappingURL=Plot-B28lJT0D.js.map
