import { p as prop, e as init, a as set_attribute } from './i18n-CMo5lpzy.js';
import { R as push, ae as onDestroy, a1 as legacy_pre_effect, G as deep_read_state, w as get, a2 as legacy_pre_effect_reset, t as template_effect, a as append, T as pop, a3 as mutable_source, U as flushSync, W as from_html, x as set } from './index-BvBk1Iap.js';

var root = from_html(`<div class="gradio-bokeh svelte-dz29a8"></div>`);

function BokehPlot($$anchor, $$props) {
	push($$props, false);

	const plot = mutable_source();

	//@ts-nocheck
	let value = prop($$props, 'value', 12);

	let bokeh_version = prop($$props, 'bokeh_version', 12);
	const div_id = `bokehDiv-${Math.random().toString(5).substring(2)}`;

	async function embed_bokeh(_plot) {
		if (document) {
			if (document.getElementById(div_id)) {
				document.getElementById(div_id).innerHTML = "";
			}
		}

		if (window.Bokeh) {
			load_bokeh();

			let plotObj = JSON.parse(_plot);

			await window.Bokeh.embed.embed_item(plotObj, div_id);
		}
	}

	const main_src = `https://cdn.bokeh.org/bokeh/release/bokeh-${bokeh_version()}.min.js`;

	const plugins_src = [
		`https://cdn.pydata.org/bokeh/release/bokeh-widgets-${bokeh_version()}.min.js`,
		`https://cdn.pydata.org/bokeh/release/bokeh-tables-${bokeh_version()}.min.js`,
		`https://cdn.pydata.org/bokeh/release/bokeh-gl-${bokeh_version()}.min.js`,
		`https://cdn.pydata.org/bokeh/release/bokeh-api-${bokeh_version()}.min.js`
	];

	let loaded = mutable_source(false);

	async function load_plugins() {
		await Promise.all(plugins_src.map((src, i) => {
			return new Promise((resolve) => {
				const script = document.createElement("script");

				script.onload = resolve;
				script.src = src;
				document.head.appendChild(script);

				return script;
			});
		}));

		set(loaded, true);
	}

	let plugin_scripts = [];

	function handle_bokeh_loaded() {
		plugin_scripts = load_plugins();
	}

	function load_bokeh() {
		const script = document.createElement("script");

		script.onload = handle_bokeh_loaded;
		script.src = main_src;

		const is_bokeh_script_present = document.head.querySelector(`script[src="${main_src}"]`);

		if (!is_bokeh_script_present) {
			document.head.appendChild(script);
		} else {
			handle_bokeh_loaded();
		}

		return script;
	}

	const main_script = bokeh_version() ? load_bokeh() : null;

	onDestroy(() => {
		if (main_script in document.children) {
			document.removeChild(main_script);
			plugin_scripts.forEach((child) => document.removeChild(child));
		}
	});

	legacy_pre_effect(() => (deep_read_state(value())), () => {
		set(plot, value()?.plot);
	});

	legacy_pre_effect(() => (get(loaded), get(plot)), () => {
		get(loaded) && embed_bokeh(get(plot));
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get bokeh_version() {
			return bokeh_version();
		},

		set bokeh_version($$value) {
			bokeh_version($$value);
			flushSync();
		}
	};

	init();

	var div = root();

	set_attribute(div, 'data-testid', "bokeh");
	template_effect(() => set_attribute(div, 'id', div_id));
	append($$anchor, div);

	return pop($$exports);
}

export { BokehPlot as default };
//# sourceMappingURL=BokehPlot-CdY9Y2o7.js.map
