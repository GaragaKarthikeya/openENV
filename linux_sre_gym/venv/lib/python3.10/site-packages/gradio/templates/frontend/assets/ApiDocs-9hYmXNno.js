import { p as prop, a as set_attribute, e as init, i as if_block, k as each, u as index, b as set_class, h as clsx, f as set_style, d as bind_this, t as remove_input_defaults, v as set_checked, F as set_value } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, f as from_svg, a4 as createEventDispatcher, S as first_child, X as sibling, V as child, Y as reset, a8 as next, a0 as set_text, Z as event, W as from_html, a7 as text, G as deep_read_state, z as untrack, w as get, a1 as legacy_pre_effect, a2 as legacy_pre_effect_reset, a3 as mutable_source, x as set, a6 as comment, ad as derived_safe_equal, aa as onMount, M as tick, b3 as invalidate_inner_signals, a5 as user_derived, at as to_array, as as mutate, aw as $window } from './index-BvBk1Iap.js';
import { b as bubble_event } from './misc-ByVveAau.js';
import { C as Clear } from './clear-D8gK36Dy.js';
import { a as api_logo } from './Blocks-Wq-vWI14.js';
import { B as Button } from './Button-CH2w4_A4.js';
import { L as Loader } from './index-Lz7fsHFt.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { h as html } from './html-CdUEtR5E.js';
import { a as bind_checked } from './input-C317uaXR.js';
import './size-Cu4PabCL.js';
import './state.svelte-CCPBiyik.js';
import './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';
import './index-BJk8VOjP.js';
import './index-DnoGeqVF.js';
import './snippet-Bl5YVoCg.js';
import './Image-CDw5qb5O.js';
/* empty css                                             */
/* empty css                                                    */
import './Clear-Dgs5UhTO.js';
import './index-C7e-J7CF.js';

var root$e = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path fill="currentColor" d="M12.1 2a9.8 9.8 0 0 0-5.4 1.6l6.4 6.4a2.1 2.1 0 0 1 .2 3a2.1 2.1 0 0 1-3-.2L3.7 6.4A9.84 9.84 0 0 0 2 12.1a10.14 10.14 0 0 0 10.1 10.1a11 11 0 0 0 2.6-.3l6.7 6.7a5 5 0 0 0 7.1-7.1l-6.7-6.7a11 11 0 0 0 .3-2.6A10 10 0 0 0 12.1 2m8 10.1a7.6 7.6 0 0 1-.3 2.1l-.3 1.1l.8.8l6.7 6.7a2.88 2.88 0 0 1 .9 2.1A2.72 2.72 0 0 1 27 27a2.9 2.9 0 0 1-4.2 0l-6.7-6.7l-.8-.8l-1.1.3a7.6 7.6 0 0 1-2.1.3a8.27 8.27 0 0 1-5.7-2.3A7.63 7.63 0 0 1 4 12.1a8.3 8.3 0 0 1 .3-2.2l4.4 4.4a4.14 4.14 0 0 0 5.9.2a4.14 4.14 0 0 0-.2-5.9L10 4.2a6.5 6.5 0 0 1 2-.3a8.27 8.27 0 0 1 5.7 2.3a8.5 8.5 0 0 1 2.4 5.9"></path></svg>`);

function Tool($$anchor, $$props) {
	push($$props, false);

	let icon_size = prop($$props, 'icon_size', 12, 16);

	var $$exports = {
		get icon_size() {
			return icon_size();
		},

		set icon_size($$value) {
			icon_size($$value);
			flushSync();
		}
	};

	var svg = root$e();

	template_effect(() => {
		set_attribute(svg, 'width', icon_size());
		set_attribute(svg, 'height', icon_size());
	});

	append($$anchor, svg);

	return pop($$exports);
}

var root$d = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path fill="currentColor" d="M17.74 30L16 29l4-7h6a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h9v2H6a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4h20a4 4 0 0 1 4 4v12a4 4 0 0 1-4 4h-4.84Z"></path><path fill="currentColor" d="M8 10h16v2H8zm0 6h10v2H8z"></path></svg>`);

function Prompt($$anchor, $$props) {
	push($$props, false);

	let icon_size = prop($$props, 'icon_size', 12, 16);

	var $$exports = {
		get icon_size() {
			return icon_size();
		},

		set icon_size($$value) {
			icon_size($$value);
			flushSync();
		}
	};

	var svg = root$d();

	template_effect(() => {
		set_attribute(svg, 'width', icon_size());
		set_attribute(svg, 'height', icon_size());
	});

	append($$anchor, svg);

	return pop($$exports);
}

var root$c = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path fill="currentColor" d="M19 10h7v2h-7zm0 5h7v2h-7zm0 5h7v2h-7z"></path><path fill="currentColor" d="M28 5H4a2 2 0 0 0-2 2v18a2 2 0 0 0 2 2h24a2.003 2.003 0 0 0 2-2V7a2 2 0 0 0-2-2M4 7h11v18H4Zm13 18V7h11l.002 18Z"></path></svg>`);

function Resource($$anchor, $$props) {
	push($$props, false);

	let icon_size = prop($$props, 'icon_size', 12, 16);

	var $$exports = {
		get icon_size() {
			return icon_size();
		},

		set icon_size($$value) {
			icon_size($$value);
			flushSync();
		}
	};

	var svg = root$c();

	template_effect(() => {
		set_attribute(svg, 'width', icon_size());
		set_attribute(svg, 'height', icon_size());
	});

	append($$anchor, svg);

	return pop($$exports);
}

var root_1$7 = from_html(`<div class="wrap prose svelte-1m8i4xt"><h1>API Docs</h1> <p class="attention svelte-1m8i4xt">No API Routes found for <code class="svelte-1m8i4xt"> </code></p> <p>To expose an API endpoint of your app in this page, set the <code>api_name</code> parameter of the event listener. <br/> For more information, visit the <a href="https://gradio.app/sharing_your_app/#api-page" target="_blank">API Page guide</a> . To hide the API documentation button and this page, set <code>footer_links=["gradio", "settings"]</code> in the <code>Blocks.launch()</code> method.</p></div> <button class="svelte-1m8i4xt"><!></button>`, 1);

function NoApi($$anchor, $$props) {
	push($$props, false);

	const dispatch = createEventDispatcher();
	let root = prop($$props, 'root', 12);

	var $$exports = {
		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		}
	};

	init();

	var fragment = root_1$7();
	var div = first_child(fragment);
	var p = sibling(child(div), 2);
	var code = sibling(child(p));
	var text = child(code, true);

	reset(code);
	reset(p);
	next(2);
	reset(div);

	var button = sibling(div, 2);
	var node = child(button);

	Clear(node);
	reset(button);
	template_effect(() => set_text(text, root()));
	event('click', button, () => dispatch("close"));
	append($$anchor, fragment);

	return pop($$exports);
}

var root_5$3 = from_html(`<div class="loading-dot self-baseline svelte-uv8a36"></div> <p class="self-baseline btn-text svelte-uv8a36">API Recorder</p>`, 1);
var root_1$6 = from_html(`<h2 class="svelte-uv8a36"><img alt="" class="svelte-uv8a36"/> <div class="title svelte-uv8a36"><!> documentation <div class="url svelte-uv8a36"> </div></div> <span class="counts svelte-uv8a36"><!> <p><span class="url svelte-uv8a36"> </span> <!><!><br/></p></span></h2> <button class="svelte-uv8a36"><!></button>`, 1);

function ApiBanner($$anchor, $$props) {
	push($$props, false);

	let root = prop($$props, 'root', 12);
	let api_count = prop($$props, 'api_count', 12);
	let current_language = prop($$props, 'current_language', 12, "python");
	const dispatch = createEventDispatcher();

	var $$exports = {
		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get api_count() {
			return api_count();
		},

		set api_count($$value) {
			api_count($$value);
			flushSync();
		},

		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		}
	};

	init();

	var fragment = root_1$6();
	var h2 = first_child(fragment);
	var img = child(h2);
	var div = sibling(img, 2);
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var text$1 = text('MCP');

			append($$anchor, text$1);
		};

		var alternate = ($$anchor) => {
			var text_1 = text('API');

			append($$anchor, text_1);
		};

		if_block(node, ($$render) => {
			if (current_language() === "mcp") $$render(consequent); else $$render(alternate, false);
		});
	}

	var div_1 = sibling(node, 2);
	var text_2 = child(div_1, true);

	reset(div_1);
	reset(div);

	var span = sibling(div, 2);
	var node_1 = child(span);

	{
		var consequent_1 = ($$anchor) => {
			Button($$anchor, {
				size: 'sm',
				variant: 'secondary',
				elem_id: 'start-api-recorder',
				onclick: () => dispatch("close", { api_recorder_visible: true }),
				children: ($$anchor, $$slotProps) => {
					var fragment_2 = root_5$3();

					next(2);
					append($$anchor, fragment_2);
				},
				$$slots: { default: true }
			});
		};

		if_block(node_1, ($$render) => {
			if (current_language() !== "mcp") $$render(consequent_1);
		});
	}

	var p = sibling(node_1, 2);
	var span_1 = child(p);
	var text_3 = child(span_1, true);

	reset(span_1);

	var node_2 = sibling(span_1, 2);

	{
		var consequent_2 = ($$anchor) => {
			var text_4 = text('API endpoint');

			append($$anchor, text_4);
		};

		var alternate_1 = ($$anchor) => {
			var text_5 = text('MCP Tool');

			append($$anchor, text_5);
		};

		if_block(node_2, ($$render) => {
			if (current_language() !== "mcp") $$render(consequent_2); else $$render(alternate_1, false);
		});
	}

	var node_3 = sibling(node_2);

	{
		var consequent_3 = ($$anchor) => {
			var text_6 = text('s');

			append($$anchor, text_6);
		};

		if_block(node_3, ($$render) => {
			if (api_count() > 1) $$render(consequent_3);
		});
	}

	next();
	reset(p);
	reset(span);
	reset(h2);

	var button = sibling(h2, 2);
	var node_4 = child(button);

	Clear(node_4);
	reset(button);

	template_effect(() => {
		set_attribute(img, 'src', api_logo);
		set_text(text_2, root());
		set_text(text_3, api_count());
	});

	event('click', button, () => dispatch("close"));
	append($$anchor, fragment);

	return pop($$exports);
}

function represent_value(value, type, lang = null) {
  if (type === void 0) {
    return lang === "py" ? "None" : null;
  }
  if (value === null && lang === "py") {
    return "None";
  }
  if (type === "string" || type === "str") {
    return lang === null ? value : '"' + value + '"';
  } else if (type === "number") {
    return lang === null ? parseFloat(value) : value;
  } else if (type === "boolean" || type == "bool") {
    if (lang === "py") {
      value = String(value);
      return value === "true" ? "True" : "False";
    } else if (lang === "js" || lang === "bash") {
      return value;
    }
    return value === "true";
  } else if (type === "List[str]") {
    value = JSON.stringify(value);
    return value;
  } else if (type.startsWith("Literal['")) {
    return '"' + value + '"';
  }
  if (lang === null) {
    return value === "" ? null : JSON.parse(value);
  } else if (typeof value === "string") {
    if (value === "") {
      return lang === "py" ? "None" : "null";
    }
    return value;
  }
  if (lang === "bash") {
    value = simplify_file_data(value);
  }
  if (lang === "py") {
    value = replace_file_data_with_file_function(value);
  }
  return stringify_except_file_function(value);
}
function simplify_file_data(obj) {
  if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
    if ("url" in obj && obj.url && "meta" in obj && obj.meta?._type === "gradio.FileData") {
      return { path: obj.url, meta: { _type: "gradio.FileData" } };
    }
  }
  if (Array.isArray(obj)) {
    obj.forEach((item, index) => {
      if (typeof item === "object" && item !== null) {
        obj[index] = simplify_file_data(item);
      }
    });
  } else if (typeof obj === "object" && obj !== null) {
    Object.keys(obj).forEach((key) => {
      obj[key] = simplify_file_data(obj[key]);
    });
  }
  return obj;
}
function replace_file_data_with_file_function(obj) {
  if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
    if ("url" in obj && obj.url && "meta" in obj && obj.meta?._type === "gradio.FileData") {
      return `handle_file('${obj.url}')`;
    }
  }
  if (Array.isArray(obj)) {
    obj.forEach((item, index) => {
      if (typeof item === "object" && item !== null) {
        obj[index] = replace_file_data_with_file_function(item);
      }
    });
  } else if (typeof obj === "object" && obj !== null) {
    Object.keys(obj).forEach((key) => {
      obj[key] = replace_file_data_with_file_function(obj[key]);
    });
  }
  return obj;
}
function stringify_except_file_function(obj) {
  let jsonString = JSON.stringify(obj, (key, value) => {
    if (value === null) {
      return "UNQUOTEDNone";
    }
    if (typeof value === "string" && value.startsWith("handle_file(") && value.endsWith(")")) {
      return `UNQUOTED${value}`;
    }
    return value;
  });
  const regex = /"UNQUOTEDhandle_file\(([^)]*)\)"/g;
  jsonString = jsonString.replace(regex, (match, p1) => `handle_file(${p1})`);
  const regexNone = /"UNQUOTEDNone"/g;
  return jsonString.replace(regexNone, "None");
}
function format_latency(val) {
  if (val < 1) return `${Math.round(val * 1e3)} ms`;
  return `${val.toFixed(2)} s`;
}

var root_3$6 = from_html(` <!>`, 1);
var root_6$3 = from_html(`<span style="font-weight:bold">Required</span>`);
var root_7$4 = from_html(`<span>Default:</span><span class="code svelte-1nfxyth" style="font-size: var(--text-sm);"> </span>`, 1);
var root_2$5 = from_html(`<hr class="hr svelte-1nfxyth"/> <div style="margin:10px;"><p style="white-space: nowrap; overflow-x: auto;"><span class="code svelte-1nfxyth" style="margin-right: 10px;"> </span> <span class="code highlight svelte-1nfxyth" style="margin-right: 10px;"><!></span> <!></p> <p class="desc svelte-1nfxyth"> </p></div>`, 1);
var root_8$2 = from_html(`<div class="load-wrap"><!></div>`);
var root$b = from_html(`<h4 class="svelte-1nfxyth"><div class="toggle-icon svelte-1nfxyth"><div class="toggle-dot svelte-1nfxyth"></div></div> <!>:</h4> <div></div> <!>`, 1);

function ParametersSnippet($$anchor, $$props) {
	push($$props, false);

	let is_running = prop($$props, 'is_running', 12);
	let endpoint_returns = prop($$props, 'endpoint_returns', 12);
	let js_returns = prop($$props, 'js_returns', 12);
	let current_language = prop($$props, 'current_language', 12);

	var $$exports = {
		get is_running() {
			return is_running();
		},

		set is_running($$value) {
			is_running($$value);
			flushSync();
		},

		get endpoint_returns() {
			return endpoint_returns();
		},

		set endpoint_returns($$value) {
			endpoint_returns($$value);
			flushSync();
		},

		get js_returns() {
			return js_returns();
		},

		set js_returns($$value) {
			js_returns($$value);
			flushSync();
		},

		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		}
	};

	init();

	var fragment = root$b();
	var h4 = first_child(fragment);
	var text$1 = sibling(child(h4));
	var node = sibling(text$1);

	{
		var consequent = ($$anchor) => {
			var text_1 = text('s');

			append($$anchor, text_1);
		};

		if_block(node, ($$render) => {
			if ((
				deep_read_state(endpoint_returns()),
				untrack(() => endpoint_returns().length != 1)
			)) $$render(consequent);
		});
	}

	next();
	reset(h4);

	var div = sibling(h4, 2);
	let classes;

	each(div, 5, endpoint_returns, index, ($$anchor, $$item, i) => {
		let label = () => get($$item).label;
		let python_type = () => get($$item).python_type;
		let component = () => get($$item).component;
		let parameter_name = () => get($$item).parameter_name;
		let parameter_has_default = () => get($$item).parameter_has_default;
		let parameter_default = () => get($$item).parameter_default;
		var fragment_1 = root_2$5();
		var div_1 = sibling(first_child(fragment_1), 2);
		var p = child(div_1);
		var span = child(p);
		var text_2 = child(span, true);

		reset(span);

		var span_1 = sibling(span, 2);
		var node_1 = child(span_1);

		{
			var consequent_2 = ($$anchor) => {
				var fragment_2 = root_3$6();
				var text_3 = first_child(fragment_2, true);
				var node_2 = sibling(text_3);

				{
					var consequent_1 = ($$anchor) => {
						var text_4 = text(' |\n							None');

						append($$anchor, text_4);
					};

					if_block(node_2, ($$render) => {
						if (parameter_has_default() && parameter_default() === null) $$render(consequent_1);
					});
				}

				template_effect(() => set_text(text_3, (python_type(), untrack(() => python_type().type))));
				append($$anchor, fragment_2);
			};

			var alternate = ($$anchor) => {
				var text_5 = text();

				template_effect(() => set_text(text_5, (
					deep_read_state(js_returns()),
					untrack(() => js_returns()[i].type || "any")
				)));

				append($$anchor, text_5);
			};

			if_block(node_1, ($$render) => {
				if (current_language() === "python") $$render(consequent_2); else $$render(alternate, false);
			});
		}

		reset(span_1);

		var node_3 = sibling(span_1, 2);

		{
			var consequent_3 = ($$anchor) => {
				var span_2 = root_6$3();

				append($$anchor, span_2);
			};

			var alternate_1 = ($$anchor) => {
				var fragment_4 = root_7$4();
				var span_3 = sibling(first_child(fragment_4));
				var text_6 = child(span_3, true);

				reset(span_3);

				template_effect(($0) => set_text(text_6, $0), [
					() => (
						deep_read_state(represent_value),
						parameter_default(),
						python_type(),
						untrack(() => represent_value(parameter_default(), python_type().type, "py"))
					)
				]);

				append($$anchor, fragment_4);
			};

			if_block(node_3, ($$render) => {
				if (!parameter_has_default() || current_language() == "bash") $$render(consequent_3); else $$render(alternate_1, false);
			});
		}

		reset(p);

		var p_1 = sibling(p, 2);
		var text_7 = child(p_1);

		reset(p_1);
		reset(div_1);

		template_effect(() => {
			set_text(text_2, current_language() !== "bash" && parameter_name() ? parameter_name() : "[" + i + "]");

			set_text(text_7, `The input value that is provided in the "${label() ?? ''}" ${component() ?? ''}
				component. ${(python_type(), untrack(() => python_type().description)) ?? ''}`);
		});

		append($$anchor, fragment_1);
	});

	reset(div);

	var node_4 = sibling(div, 2);

	{
		var consequent_4 = ($$anchor) => {
			var div_2 = root_8$2();
			var node_5 = child(div_2);

			Loader(node_5, { margin: false });
			reset(div_2);
			append($$anchor, div_2);
		};

		if_block(node_4, ($$render) => {
			if (is_running()) $$render(consequent_4);
		});
	}

	template_effect(() => {
		set_text(text$1, ` Accepts ${(
			deep_read_state(endpoint_returns()),
			untrack(() => endpoint_returns().length)
		) ?? ''} parameter`);

		classes = set_class(div, 1, '', null, classes, { hide: is_running() });
	});

	append($$anchor, fragment);

	return pop($$exports);
}

var root$a = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>`);

function IconCopy($$anchor, $$props) {
	push($$props, false);

	let classNames = prop($$props, 'classNames', 12, "");

	var $$exports = {
		get classNames() {
			return classNames();
		},

		set classNames($$value) {
			classNames($$value);
			flushSync();
		}
	};

	var svg = root$a();

	template_effect(() => set_class(svg, 0, clsx(classNames()), 'svelte-etvp4k'));
	append($$anchor, svg);

	return pop($$exports);
}

var root$9 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true"><path fill-rule="evenodd" fill="rgb(255, 124, 1)" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>`);

function IconCheck($$anchor, $$props) {
	push($$props, false);

	let classNames = prop($$props, 'classNames', 12, "");

	var $$exports = {
		get classNames() {
			return classNames();
		},

		set classNames($$value) {
			classNames($$value);
			flushSync();
		}
	};

	var svg = root$9();

	template_effect(() => set_class(svg, 0, clsx(classNames() || "icon-size"), 'svelte-5quqiz'));
	append($$anchor, svg);

	return pop($$exports);
}

var root$8 = from_html(`<button class="copy-button svelte-egghn7" aria-live="polite"><span class="inline-flex items-center justify-center rounded-md p-0.5 max-sm:p-0"><!></span></button>`);

function CopyButton($$anchor, $$props) {
	push($$props, false);

	let code = prop($$props, 'code', 12);
	let copied = mutable_source(false);

	function copy() {
		navigator.clipboard.writeText(code());
		set(copied, true);

		setTimeout(
			() => {
				set(copied, false);
			},
			1500
		);
	}

	legacy_pre_effect(() => (get(copied)), () => {
		get(copied);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get code() {
			return code();
		},

		set code($$value) {
			code($$value);
			flushSync();
		}
	};

	var button = root$8();
	var span = child(button);
	var node = child(span);

	{
		var consequent = ($$anchor) => {
			IconCheck($$anchor, { classNames: 'w-3 h-3 max-sm:w-2.5 max-sm:h-2.5' });
		};

		var alternate = ($$anchor) => {
			IconCopy($$anchor, { classNames: 'w-3 h-3 max-sm:w-2.5 max-sm:h-2.5' });
		};

		if_block(node, ($$render) => {
			if (get(copied)) $$render(consequent); else $$render(alternate, false);
		});
	}

	reset(span);
	reset(button);
	event('click', button, copy);
	append($$anchor, button);

	return pop($$exports);
}

var root_2$4 = from_html(`<div class="copy svelte-1ts4a2g"><!></div> <div><pre class="svelte-1ts4a2g"></pre></div>`, 1);
var root_4$4 = from_html(`<div class="copy svelte-1ts4a2g"><!></div> <div><pre class="svelte-1ts4a2g"></pre></div>`, 1);
var root_6$2 = from_html(`<div class="copy svelte-1ts4a2g"><!></div> <div><pre class="svelte-1ts4a2g"></pre></div>`, 1);
var root_1$5 = from_html(`<code class="svelte-1ts4a2g"><!></code>`);

function InstallSnippet($$anchor, $$props) {
	push($$props, false);

	let current_language = prop($$props, 'current_language', 12);
	let py_install = "pip install gradio_client";
	let js_install = "npm i -D @gradio/client";
	let bash_install = "curl --version";

	var $$exports = {
		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		}
	};

	Block($$anchor, {
		children: ($$anchor, $$slotProps) => {
			var code = root_1$5();
			var node = child(code);

			{
				var consequent = ($$anchor) => {
					var fragment_1 = root_2$4();
					var div = first_child(fragment_1);
					var node_1 = child(div);

					CopyButton(node_1, { code: py_install });
					reset(div);

					var div_1 = sibling(div, 2);
					var pre = child(div_1);

					pre.textContent = '$ pip install gradio_client';
					reset(div_1);
					append($$anchor, fragment_1);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_2 = comment();
					var node_2 = first_child(fragment_2);

					{
						var consequent_1 = ($$anchor) => {
							var fragment_3 = root_4$4();
							var div_2 = first_child(fragment_3);
							var node_3 = child(div_2);

							CopyButton(node_3, { code: js_install });
							reset(div_2);

							var div_3 = sibling(div_2, 2);
							var pre_1 = child(div_3);

							pre_1.textContent = '$ npm i -D @gradio/client';
							reset(div_3);
							append($$anchor, fragment_3);
						};

						var alternate = ($$anchor) => {
							var fragment_4 = comment();
							var node_4 = first_child(fragment_4);

							{
								var consequent_2 = ($$anchor) => {
									var fragment_5 = root_6$2();
									var div_4 = first_child(fragment_5);
									var node_5 = child(div_4);

									CopyButton(node_5, { code: bash_install });
									reset(div_4);

									var div_5 = sibling(div_4, 2);
									var pre_2 = child(div_5);

									pre_2.textContent = '$ curl --version';
									reset(div_5);
									append($$anchor, fragment_5);
								};

								if_block(
									node_4,
									($$render) => {
										if (current_language() === "bash") $$render(consequent_2);
									},
									true
								);
							}

							append($$anchor, fragment_4);
						};

						if_block(
							node_2,
							($$render) => {
								if (current_language() === "javascript") $$render(consequent_1); else $$render(alternate, false);
							},
							true
						);
					}

					append($$anchor, fragment_2);
				};

				if_block(node, ($$render) => {
					if (current_language() === "python") $$render(consequent); else $$render(alternate_1, false);
				});
			}

			reset(code);
			append($$anchor, code);
		},
		$$slots: { default: true }
	});

	return pop($$exports);
}

var root$7 = from_html(`<div class="tooltip-chart svelte-1q3jha5"><div class="tooltip-arrow svelte-1q3jha5"></div> <div class="chart-bars svelte-1q3jha5"><div class="chart-bar-container svelte-1q3jha5"><div class="chart-bar-label svelte-1q3jha5">p50</div> <div class="chart-bar-wrapper svelte-1q3jha5"><div class="chart-bar svelte-1q3jha5"></div></div> <div class="chart-bar-value svelte-1q3jha5"> </div></div> <div class="chart-bar-container svelte-1q3jha5"><div class="chart-bar-label svelte-1q3jha5">p90</div> <div class="chart-bar-wrapper svelte-1q3jha5"><div class="chart-bar svelte-1q3jha5"></div></div> <div class="chart-bar-value svelte-1q3jha5"> </div></div> <div class="chart-bar-container svelte-1q3jha5"><div class="chart-bar-label svelte-1q3jha5">p99</div> <div class="chart-bar-wrapper svelte-1q3jha5"><div class="chart-bar svelte-1q3jha5"></div></div> <div class="chart-bar-value svelte-1q3jha5"> </div></div></div></div>`);

function PercentileChart($$anchor, $$props) {
	push($$props, false);

	const max_latency = mutable_source();
	let p50 = prop($$props, 'p50', 12);
	let p90 = prop($$props, 'p90', 12);
	let p99 = prop($$props, 'p99', 12);

	legacy_pre_effect(
		() => (
			deep_read_state(p50()),
			deep_read_state(p90()),
			deep_read_state(p99())
		),
		() => {
			set(max_latency, Math.max(p50(), p90(), p99()));
		}
	);

	legacy_pre_effect_reset();

	var $$exports = {
		get p50() {
			return p50();
		},

		set p50($$value) {
			p50($$value);
			flushSync();
		},

		get p90() {
			return p90();
		},

		set p90($$value) {
			p90($$value);
			flushSync();
		},

		get p99() {
			return p99();
		},

		set p99($$value) {
			p99($$value);
			flushSync();
		}
	};

	init();

	var div = root$7();
	var div_1 = sibling(child(div), 2);
	var div_2 = child(div_1);
	var div_3 = sibling(child(div_2), 2);
	var div_4 = child(div_3);

	reset(div_3);

	var div_5 = sibling(div_3, 2);
	var text = child(div_5, true);

	reset(div_5);
	reset(div_2);

	var div_6 = sibling(div_2, 2);
	var div_7 = sibling(child(div_6), 2);
	var div_8 = child(div_7);

	reset(div_7);

	var div_9 = sibling(div_7, 2);
	var text_1 = child(div_9, true);

	reset(div_9);
	reset(div_6);

	var div_10 = sibling(div_6, 2);
	var div_11 = sibling(child(div_10), 2);
	var div_12 = child(div_11);

	reset(div_11);

	var div_13 = sibling(div_11, 2);
	var text_2 = child(div_13, true);

	reset(div_13);
	reset(div_10);
	reset(div_1);
	reset(div);

	template_effect(
		($0, $1, $2) => {
			set_style(div_4, `width: ${p50() / get(max_latency) * 100}%`);
			set_text(text, $0);
			set_style(div_8, `width: ${p90() / get(max_latency) * 100}%`);
			set_text(text_1, $1);
			set_style(div_12, `width: ${p99() / get(max_latency) * 100}%`);
			set_text(text_2, $2);
		},
		[
			() => (
				deep_read_state(format_latency),
				deep_read_state(p50()),
				untrack(() => format_latency(p50()))
			),

			() => (
				deep_read_state(format_latency),
				deep_read_state(p90()),
				untrack(() => format_latency(p90()))
			),

			() => (
				deep_read_state(format_latency),
				deep_read_state(p99()),
				untrack(() => format_latency(p99()))
			)
		]
	);

	append($$anchor, div);

	return pop($$exports);
}

var root_2$3 = from_html(`<span class="most-recently-used svelte-13swnto">Most recently used</span>`);
var root_1$4 = from_html(`<span class="analytics-inline svelte-13swnto"><span class="analytics-content svelte-13swnto"><span class="analytics-text-wrapper svelte-13swnto"> </span> <div class="analytics-tooltip svelte-13swnto"><!></div></span> <!></span>`);
var root$6 = from_html(`<h3 class="svelte-13swnto">API name: <span class="post svelte-13swnto"> </span> <span class="desc svelte-13swnto"> <!></span></h3>`);

function EndpointDetail($$anchor, $$props) {
	push($$props, false);

	const is_most_recently_used = mutable_source();
	let api_name = prop($$props, 'api_name', 12, null);
	let description = prop($$props, 'description', 12, null);
	let analytics = prop($$props, 'analytics', 12);
	let last_api_call = prop($$props, 'last_api_call', 12, null);
	let dependency_id = prop($$props, 'dependency_id', 12, null);
	const success_rate = api_name() ? analytics()[api_name()]?.success_rate : 0;

	legacy_pre_effect(
		() => (
			deep_read_state(last_api_call()),
			deep_read_state(dependency_id())
		),
		() => {
			set(is_most_recently_used, last_api_call()?.fn_index === dependency_id());
		}
	);

	legacy_pre_effect_reset();

	var $$exports = {
		get api_name() {
			return api_name();
		},

		set api_name($$value) {
			api_name($$value);
			flushSync();
		},

		get description() {
			return description();
		},

		set description($$value) {
			description($$value);
			flushSync();
		},

		get analytics() {
			return analytics();
		},

		set analytics($$value) {
			analytics($$value);
			flushSync();
		},

		get last_api_call() {
			return last_api_call();
		},

		set last_api_call($$value) {
			last_api_call($$value);
			flushSync();
		},

		get dependency_id() {
			return dependency_id();
		},

		set dependency_id($$value) {
			dependency_id($$value);
			flushSync();
		}
	};

	init();

	var h3 = root$6();
	var span = sibling(child(h3));
	var text = child(span, true);

	reset(span);

	var span_1 = sibling(span, 2);
	var text_1 = child(span_1);
	var node = sibling(text_1);

	{
		var consequent_1 = ($$anchor) => {
			const endpoint_analytics = derived_safe_equal(() => (
				deep_read_state(analytics()),
				deep_read_state(api_name()),
				untrack(() => analytics()[api_name()])
			));

			const p50 = derived_safe_equal(() => (
				deep_read_state(get(endpoint_analytics)),
				untrack(() => get(endpoint_analytics).process_time_percentiles["50th"])
			));

			const p90 = derived_safe_equal(() => (
				deep_read_state(get(endpoint_analytics)),
				untrack(() => get(endpoint_analytics).process_time_percentiles["90th"])
			));

			const p99 = derived_safe_equal(() => (
				deep_read_state(get(endpoint_analytics)),
				untrack(() => get(endpoint_analytics).process_time_percentiles["99th"])
			));

			var span_2 = root_1$4();
			var span_3 = child(span_2);
			var span_4 = child(span_3);
			var text_2 = child(span_4);

			reset(span_4);

			var div = sibling(span_4, 2);
			var node_1 = child(div);

			PercentileChart(node_1, {
				get p50() {
					return get(p50);
				},

				get p90() {
					return get(p90);
				},

				get p99() {
					return get(p99);
				}
			});

			reset(div);
			reset(span_3);

			var node_2 = sibling(span_3, 2);

			{
				var consequent = ($$anchor) => {
					var span_5 = root_2$3();

					append($$anchor, span_5);
				};

				if_block(node_2, ($$render) => {
					if (get(is_most_recently_used)) $$render(consequent);
				});
			}

			reset(span_2);

			template_effect(
				($0, $1) => set_text(text_2, `${(
					deep_read_state(get(endpoint_analytics)),
					untrack(() => get(endpoint_analytics).total_requests)
				) ?? ''} requests (${$0 ?? ''}% successful, p50: ${$1 ?? ''})`),
				[
					() => (untrack(() => Math.round(success_rate * 100))),
					() => (
						deep_read_state(format_latency),
						deep_read_state(get(p50)),
						untrack(() => format_latency(get(p50)))
					)
				]
			);

			append($$anchor, span_2);
		};

		if_block(node, ($$render) => {
			if ((
				deep_read_state(analytics()),
				deep_read_state(api_name()),
				untrack(() => analytics() && api_name() && analytics()[api_name()])
			)) $$render(consequent_1);
		});
	}

	reset(span_1);
	reset(h3);

	template_effect(() => {
		set_text(text, "/" + api_name());
		set_text(text_1, `${description() ?? ''} `);
	});

	append($$anchor, h3);

	return pop($$exports);
}

var root_1$3 = from_html(`<code class="svelte-1v9rhoq"><div class="copy svelte-1v9rhoq"><!></div> <pre class="svelte-1v9rhoq"><!></pre></code>`);
var root_2$2 = from_html(`<code class="svelte-1v9rhoq"><div class="copy svelte-1v9rhoq"><!></div> <pre class="svelte-1v9rhoq"><!></pre></code>`);
var root_3$5 = from_html(`<code class="svelte-1v9rhoq"><div class="copy svelte-1v9rhoq"><!></div> <pre class="svelte-1v9rhoq"><!></pre></code>`);
var root$5 = from_html(`<div class="container svelte-1v9rhoq"><!> <div><!></div> <div><!></div> <div><!></div></div>`);

function CodeSnippet($$anchor, $$props) {
	push($$props, false);

	const python_html = mutable_source();
	const js_html = mutable_source();
	const bash_html = mutable_source();
	let dependency = prop($$props, 'dependency', 12);
	let current_language = prop($$props, 'current_language', 12);
	let api_description = prop($$props, 'api_description', 12, null);
	let analytics = prop($$props, 'analytics', 12);
	let markdown_code_snippets = prop($$props, 'markdown_code_snippets', 12);
	let code_snippets = prop($$props, 'code_snippets', 12);
	let last_api_call = prop($$props, 'last_api_call', 12, null);

	function escape_html(text) {
		return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
	}

	function highlight_python(code) {
		let html = escape_html(code);

		html = html.replace(/\b(from|import|print)\b/g, '<span class="kw">$1</span>');
		html = html.replace(/(Client\()("[^"]*")(\))/g, '$1<span class="str">$2</span>$3');
		html = html.replace(/(api_name=)("[^"]*")/g, '$1<span class="api-name">$2</span>');

		return html;
	}

	function highlight_javascript(code) {
		let html = escape_html(code);

		html = html.replace(/\b(import|from|const|await)\b/g, '<span class="kw">$1</span>');
		html = html.replace(/(Client\.connect\()("[^"]*")(\))/g, '$1<span class="str">$2</span>$3');
		html = html.replace(/(\.predict\()("[^"]*")/g, '$1<span class="api-name">$2</span>');

		return html;
	}

	function highlight_bash(code) {
		return escape_html(code);
	}

	legacy_pre_effect(() => (deep_read_state(code_snippets())), () => {
		markdown_code_snippets(
			markdown_code_snippets()[dependency().api_name] = {
				python: code_snippets().python || "",
				javascript: code_snippets().javascript || "",
				bash: code_snippets().bash || ""
			},
			true
		);
	});

	legacy_pre_effect(() => (deep_read_state(code_snippets())), () => {
		set(python_html, highlight_python(code_snippets().python || ""));
	});

	legacy_pre_effect(() => (deep_read_state(code_snippets())), () => {
		set(js_html, highlight_javascript(code_snippets().javascript || ""));
	});

	legacy_pre_effect(() => (deep_read_state(code_snippets())), () => {
		set(bash_html, highlight_bash(code_snippets().bash || ""));
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get dependency() {
			return dependency();
		},

		set dependency($$value) {
			dependency($$value);
			flushSync();
		},

		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		},

		get api_description() {
			return api_description();
		},

		set api_description($$value) {
			api_description($$value);
			flushSync();
		},

		get analytics() {
			return analytics();
		},

		set analytics($$value) {
			analytics($$value);
			flushSync();
		},

		get markdown_code_snippets() {
			return markdown_code_snippets();
		},

		set markdown_code_snippets($$value) {
			markdown_code_snippets($$value);
			flushSync();
		},

		get code_snippets() {
			return code_snippets();
		},

		set code_snippets($$value) {
			code_snippets($$value);
			flushSync();
		},

		get last_api_call() {
			return last_api_call();
		},

		set last_api_call($$value) {
			last_api_call($$value);
			flushSync();
		}
	};

	init();

	var div = root$5();
	var node = child(div);

	EndpointDetail(node, {
		get api_name() {
			return (
				deep_read_state(dependency()),
				untrack(() => dependency().api_name)
			);
		},

		get description() {
			return api_description();
		},

		get analytics() {
			return analytics();
		},

		get last_api_call() {
			return last_api_call();
		},

		get dependency_id() {
			return (
				deep_read_state(dependency()),
				untrack(() => dependency().id)
			);
		}
	});

	var div_1 = sibling(node, 2);
	let classes;
	var node_1 = child(div_1);

	Block(node_1, {
		children: ($$anchor, $$slotProps) => {
			var code_1 = root_1$3();
			var div_2 = child(code_1);
			var node_2 = child(div_2);

			{
				let $0 = derived_safe_equal(() => (
					deep_read_state(code_snippets()),
					untrack(() => code_snippets().python || "")
				));

				CopyButton(node_2, {
					get code() {
						return get($0);
					}
				});
			}

			reset(div_2);

			var pre = sibling(div_2, 2);
			var node_3 = child(pre);

			html(node_3, () => get(python_html));
			reset(pre);
			reset(code_1);
			append($$anchor, code_1);
		},
		$$slots: { default: true }
	});

	reset(div_1);

	var div_3 = sibling(div_1, 2);
	let classes_1;
	var node_4 = child(div_3);

	Block(node_4, {
		children: ($$anchor, $$slotProps) => {
			var code_2 = root_2$2();
			var div_4 = child(code_2);
			var node_5 = child(div_4);

			{
				let $0 = derived_safe_equal(() => (
					deep_read_state(code_snippets()),
					untrack(() => code_snippets().javascript || "")
				));

				CopyButton(node_5, {
					get code() {
						return get($0);
					}
				});
			}

			reset(div_4);

			var pre_1 = sibling(div_4, 2);
			var node_6 = child(pre_1);

			html(node_6, () => get(js_html));
			reset(pre_1);
			reset(code_2);
			append($$anchor, code_2);
		},
		$$slots: { default: true }
	});

	reset(div_3);

	var div_5 = sibling(div_3, 2);
	let classes_2;
	var node_7 = child(div_5);

	Block(node_7, {
		children: ($$anchor, $$slotProps) => {
			var code_3 = root_3$5();
			var div_6 = child(code_3);
			var node_8 = child(div_6);

			{
				let $0 = derived_safe_equal(() => (
					deep_read_state(code_snippets()),
					untrack(() => code_snippets().bash || "")
				));

				CopyButton(node_8, {
					get code() {
						return get($0);
					}
				});
			}

			reset(div_6);

			var pre_2 = sibling(div_6, 2);
			var node_9 = child(pre_2);

			html(node_9, () => get(bash_html));
			reset(pre_2);
			reset(code_3);
			append($$anchor, code_3);
		},
		$$slots: { default: true }
	});

	reset(div_5);
	reset(div);

	template_effect(() => {
		classes = set_class(div_1, 1, 'svelte-1v9rhoq', null, classes, { hidden: current_language() !== "python" });
		classes_1 = set_class(div_3, 1, 'svelte-1v9rhoq', null, classes_1, { hidden: current_language() !== "javascript" });
		classes_2 = set_class(div_5, 1, 'svelte-1v9rhoq', null, classes_2, { hidden: current_language() !== "bash" });
	});

	append($$anchor, div);

	return pop($$exports);
}

var root_5$2 = from_html(
	`
client.<span class="highlight"> <span class="api-name svelte-154to14"> </span>
)
</span>`,
	1
);

var root_3$4 = from_html(`<code class="svelte-154to14"><div class="copy svelte-154to14"><!></div> <div><pre class="svelte-154to14"><span class="highlight">from</span> gradio_client <span class="highlight">import</span> Client, file

client = Client(<span class="token string svelte-154to14"> </span><!>)
<!></pre></div></code>`);

var root_9 = from_html(
	`
await client.predict(<span class="api-name svelte-154to14"> </span><!>);
						`,
	1
);

var root_7$3 = from_html(`<code class="svelte-154to14"><div class="copy svelte-154to14"><!></div> <div><pre class="svelte-154to14">import &lbrace; Client &rbrace; from "@gradio/client";

const app = await Client.connect(<span class="token string svelte-154to14"> </span><!>);
					<!></pre></div></code>`);

var root_13$2 = from_html(`<pre class="svelte-154to14"> </pre> <br/>`, 1);
var root_12$2 = from_html(`<code class="svelte-154to14"><div class="copy svelte-154to14"><!></div> <div></div></code>`);
var root_1$2 = from_html(`<div class="container svelte-154to14"><!></div>`);

function RecordingSnippet($$anchor, $$props) {
	push($$props, false);

	let dependencies = prop($$props, 'dependencies', 12);
	let short_root = prop($$props, 'short_root', 12);
	let root = prop($$props, 'root', 12);
	let api_prefix = prop($$props, 'api_prefix', 12, "");
	let current_language = prop($$props, 'current_language', 12);
	let username = prop($$props, 'username', 12);
	let python_code = mutable_source();
	let python_code_text = mutable_source();
	let js_code = mutable_source();
	let bash_code = mutable_source();
	let api_calls = prop($$props, 'api_calls', 28, () => []);

	async function get_info() {
		let response = await fetch(root().replace(/\/$/, "") + api_prefix() + "/info/?all_endpoints=true");
		let data = await response.json();

		return data;
	}

	let endpoints_info;
	let py_zipped = mutable_source([]);
	let js_zipped = mutable_source([]);
	let bash_zipped = mutable_source([]);

	function format_api_call(call, lang) {
		const api_name = `/${dependencies()[call.fn_index].api_name}`;

		// If an input is undefined (distinct from null) then it corresponds to a State component.
		let call_data_excluding_state = call.data.filter((d) => typeof d !== "undefined");

		const params = call_data_excluding_state.map((param, index) => {
			if (endpoints_info[api_name]) {
				const param_info = endpoints_info[api_name].parameters[index];

				if (!param_info) {
					return undefined;
				}

				const param_name = param_info.parameter_name;
				const python_type = param_info.python_type.type;

				if (lang === "py") {
					return `  ${param_name}=${represent_value(param, python_type, "py")}`;
				} else if (lang === "js") {
					return `    ${param_name}: ${represent_value(param, python_type, "js")}`;
				} else if (lang === "bash") {
					return `    ${represent_value(param, python_type, "bash")}`;
				}
			}

			return `  ${represent_value(param, undefined, lang)}`;
		}).filter((d) => typeof d !== "undefined").join(",\n");

		if (params) {
			if (lang === "py") {
				return `${params},\n`;
			} else if (lang === "js") {
				return `{\n${params},\n}`;
			} else if (lang === "bash") {
				return `\n${params}\n`;
			}
		}

		if (lang === "py") {
			return "";
		}

		return "\n";
	}

	onMount(async () => {
		const data = await get_info();

		endpoints_info = data["named_endpoints"];

		let py_api_calls = api_calls().map((call) => format_api_call(call, "py"));
		let js_api_calls = api_calls().map((call) => format_api_call(call, "js"));
		let bash_api_calls = api_calls().map((call) => format_api_call(call, "bash"));
		let api_names = api_calls().map((call) => dependencies()[call.fn_index].api_name || "");

		set(py_zipped, py_api_calls.map((call, index) => ({ call, api_name: api_names[index] })));
		set(js_zipped, js_api_calls.map((call, index) => ({ call, api_name: api_names[index] })));
		set(bash_zipped, bash_api_calls.map((call, index) => ({ call, api_name: api_names[index] })));
		await tick();
		set(python_code_text, get(python_code).innerText);
	});

	var $$exports = {
		get dependencies() {
			return dependencies();
		},

		set dependencies($$value) {
			dependencies($$value);
			flushSync();
		},

		get short_root() {
			return short_root();
		},

		set short_root($$value) {
			short_root($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get api_prefix() {
			return api_prefix();
		},

		set api_prefix($$value) {
			api_prefix($$value);
			flushSync();
		},

		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		},

		get username() {
			return username();
		},

		set username($$value) {
			username($$value);
			flushSync();
		},

		get api_calls() {
			return api_calls();
		},

		set api_calls($$value) {
			api_calls($$value);
			flushSync();
		}
	};

	init();

	var div = root_1$2();
	var node = child(div);

	Block(node, {
		border_mode: "focus",
		children: ($$anchor, $$slotProps) => {
			var fragment = comment();
			var node_1 = first_child(fragment);

			{
				var consequent_1 = ($$anchor) => {
					var code = root_3$4();
					var div_1 = child(code);
					var node_2 = child(div_1);

					CopyButton(node_2, {
						get code() {
							return get(python_code_text);
						}
					});

					reset(div_1);

					var div_2 = sibling(div_1, 2);
					var pre = child(div_2);
					var span = sibling(child(pre), 4);
					var text$1 = child(span);

					reset(span);

					var node_3 = sibling(span);

					{
						var consequent = ($$anchor) => {
							var text_1 = text();

							template_effect(() => set_text(text_1, `, auth=("${username() ?? ''}", **password**)`));
							append($$anchor, text_1);
						};

						if_block(node_3, ($$render) => {
							if (username() !== null) $$render(consequent);
						});
					}

					var node_4 = sibling(node_3, 2);

					each(node_4, 1, () => get(py_zipped), index, ($$anchor, $$item) => {
						let call = () => get($$item).call;
						let api_name = () => get($$item).api_name;

						next();

						var fragment_2 = root_5$2();
						var span_1 = sibling(first_child(fragment_2));
						var text_2 = child(span_1);
						var span_2 = sibling(text_2);
						var text_3 = child(span_2);

						reset(span_2);
						next();
						reset(span_1);

						template_effect(() => {
							set_text(text_2, `predict(
${call() ?? ''}  api_name=`);

							set_text(text_3, `"/${api_name() ?? ''}"`);
						});

						append($$anchor, fragment_2);
					});

					reset(pre);
					reset(div_2);
					bind_this(div_2, ($$value) => set(python_code, $$value), () => get(python_code));
					reset(code);
					template_effect(() => set_text(text$1, `"${short_root() ?? ''}"`));
					append($$anchor, code);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_3 = comment();
					var node_5 = first_child(fragment_3);

					{
						var consequent_4 = ($$anchor) => {
							var code_1 = root_7$3();
							var div_3 = child(code_1);
							var node_6 = child(div_3);

							{
								let $0 = derived_safe_equal(() => (get(js_code), untrack(() => get(js_code)?.innerText)));

								CopyButton(node_6, {
									get code() {
										return get($0);
									}
								});
							}

							reset(div_3);

							var div_4 = sibling(div_3, 2);
							var pre_1 = child(div_4);
							var span_3 = sibling(child(pre_1));
							var text_4 = child(span_3);

							reset(span_3);

							var node_7 = sibling(span_3);

							{
								var consequent_2 = ($$anchor) => {
									var text_5 = text();

									template_effect(() => set_text(text_5, `, {auth: ["${username() ?? ''}", **password**]}`));
									append($$anchor, text_5);
								};

								if_block(node_7, ($$render) => {
									if (username() !== null) $$render(consequent_2);
								});
							}

							var node_8 = sibling(node_7, 2);

							each(node_8, 1, () => get(js_zipped), index, ($$anchor, $$item) => {
								let call = () => get($$item).call;
								let api_name = () => get($$item).api_name;

								next();

								var fragment_5 = root_9();
								var span_4 = sibling(first_child(fragment_5));
								var text_6 = child(span_4);

								reset(span_4);

								var node_9 = sibling(span_4);

								{
									var consequent_3 = ($$anchor) => {
										var text_7 = text();

										template_effect(() => set_text(text_7, `, ${call() ?? ''}`));
										append($$anchor, text_7);
									};

									if_block(node_9, ($$render) => {
										if (call()) $$render(consequent_3);
									});
								}

								next();

								template_effect(() => set_text(text_6, `
  "/${api_name() ?? ''}"`));

								append($$anchor, fragment_5);
							});

							reset(pre_1);
							reset(div_4);
							bind_this(div_4, ($$value) => set(js_code, $$value), () => get(js_code));
							reset(code_1);
							template_effect(() => set_text(text_4, `"${short_root() ?? ''}"`));
							append($$anchor, code_1);
						};

						var alternate = ($$anchor) => {
							var fragment_7 = comment();
							var node_10 = first_child(fragment_7);

							{
								var consequent_5 = ($$anchor) => {
									var code_2 = root_12$2();
									var div_5 = child(code_2);
									var node_11 = child(div_5);

									{
										let $0 = derived_safe_equal(() => (
											get(bash_code),
											untrack(() => get(bash_code)?.innerText)
										));

										CopyButton(node_11, {
											get code() {
												return get($0);
											}
										});
									}

									reset(div_5);

									var div_6 = sibling(div_5, 2);

									each(div_6, 5, () => get(bash_zipped), index, ($$anchor, $$item) => {
										let call = () => get($$item).call;
										let api_name = () => get($$item).api_name;
										var fragment_8 = root_13$2();
										var pre_2 = first_child(fragment_8);
										var text_8 = child(pre_2);

										reset(pre_2);
										next(2);

										template_effect(() => set_text(text_8, `curl -X POST ${short_root() ?? ''}call/${api_name() ?? ''} -s -H "Content-Type: application/json" -d '{ 
	"data": [${call() ?? ''}]}' \\
  | awk -F'"' '{ print $4}' \\
  | read EVENT_ID; curl -N ${short_root() ?? ''}call/${api_name() ?? ''}/$EVENT_ID`));

										append($$anchor, fragment_8);
									});

									reset(div_6);
									bind_this(div_6, ($$value) => set(bash_code, $$value), () => get(bash_code));
									reset(code_2);
									append($$anchor, code_2);
								};

								if_block(
									node_10,
									($$render) => {
										if (current_language() === "bash") $$render(consequent_5);
									},
									true
								);
							}

							append($$anchor, fragment_7);
						};

						if_block(
							node_5,
							($$render) => {
								if (current_language() === "javascript") $$render(consequent_4); else $$render(alternate, false);
							},
							true
						);
					}

					append($$anchor, fragment_3);
				};

				if_block(node_1, ($$render) => {
					if (current_language() === "python") $$render(consequent_1); else $$render(alternate_1, false);
				});
			}

			append($$anchor, fragment);
		},
		$$slots: { default: true }
	});

	reset(div);
	append($$anchor, div);

	return pop($$exports);
}

const python = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20aria-hidden='true'%20focusable='false'%20role='img'%20width='1em'%20height='1em'%20preserveAspectRatio='xMidYMid%20meet'%20viewBox='0%200%2032%2032'%20%3e%3cpath%20d='M15.84.5a16.4,16.4,0,0,0-3.57.32C9.1,1.39,8.53,2.53,8.53,4.64V7.48H16v1H5.77a4.73,4.73,0,0,0-4.7,3.74,14.82,14.82,0,0,0,0,7.54c.57,2.28,1.86,3.82,4,3.82h2.6V20.14a4.73,4.73,0,0,1,4.63-4.63h7.38a3.72,3.72,0,0,0,3.73-3.73V4.64A4.16,4.16,0,0,0,19.65.82,20.49,20.49,0,0,0,15.84.5ZM11.78,2.77a1.39,1.39,0,0,1,1.38,1.46,1.37,1.37,0,0,1-1.38,1.38A1.42,1.42,0,0,1,10.4,4.23,1.44,1.44,0,0,1,11.78,2.77Z'%20fill='%235a9fd4'%20%3e%3c/path%3e%3cpath%20d='M16.16,31.5a16.4,16.4,0,0,0,3.57-.32c3.17-.57,3.74-1.71,3.74-3.82V24.52H16v-1H26.23a4.73,4.73,0,0,0,4.7-3.74,14.82,14.82,0,0,0,0-7.54c-.57-2.28-1.86-3.82-4-3.82h-2.6v3.41a4.73,4.73,0,0,1-4.63,4.63H12.35a3.72,3.72,0,0,0-3.73,3.73v7.14a4.16,4.16,0,0,0,3.73,3.82A20.49,20.49,0,0,0,16.16,31.5Zm4.06-2.27a1.39,1.39,0,0,1-1.38-1.46,1.37,1.37,0,0,1,1.38-1.38,1.42,1.42,0,0,1,1.38,1.38A1.44,1.44,0,0,1,20.22,29.23Z'%20fill='%23ffd43b'%20%3e%3c/path%3e%3c/svg%3e";

const javascript = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20aria-hidden='true'%20focusable='false'%20role='img'%20width='1em'%20height='1em'%20preserveAspectRatio='xMidYMid%20meet'%20viewBox='0%200%2032%2032'%20%3e%3crect%20width='32'%20height='32'%20fill='%23f7df1e'%3e%3c/rect%3e%3cpath%20d='M21.5,25a3.27,3.27,0,0,0,3,1.83c1.25,0,2-.63,2-1.49,0-1-.81-1.39-2.19-2L23.56,23C21.39,22.1,20,20.94,20,18.49c0-2.25,1.72-4,4.41-4a4.44,4.44,0,0,1,4.27,2.41l-2.34,1.5a2,2,0,0,0-1.93-1.29,1.31,1.31,0,0,0-1.44,1.29c0,.9.56,1.27,1.85,1.83l.75.32c2.55,1.1,4,2.21,4,4.72,0,2.71-2.12,4.19-5,4.19a5.78,5.78,0,0,1-5.48-3.07Zm-10.63.26c.48.84.91,1.55,1.94,1.55s1.61-.39,1.61-1.89V14.69h3V25c0,3.11-1.83,4.53-4.49,4.53a4.66,4.66,0,0,1-4.51-2.75Z'%20%3e%3c/path%3e%3c/svg%3e";

const bash = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20version='1.1'%20id='Layer_1'%20x='0px'%20y='0px'%20viewBox='0%200%20150%20150'%20style='enable-background:new%200%200%20150%20150;%20background-color:%20%2372a824;'%20xml:space='preserve'%3e%3cscript%20xmlns=''/%3e%3cstyle%20type='text/css'%3e%20.st0{fill:%23FFFFFF;}%20%3c/style%3e%3cg%3e%3cpath%20class='st0'%20d='M118.9,40.3L81.7,18.2c-2.2-1.3-4.7-2-7.2-2s-5,0.7-7.2,2L30.1,40.3c-4.4,2.6-7.2,7.5-7.2,12.8v44.2%20c0,5.3,2.7,10.1,7.2,12.8l37.2,22.1c2.2,1.3,4.7,2,7.2,2c2.5,0,5-0.7,7.2-2l37.2-22.1c4.4-2.6,7.2-7.5,7.2-12.8V53%20C126.1,47.8,123.4,42.9,118.9,40.3z%20M90.1,109.3l0.1,3.2c0,0.4-0.2,0.8-0.5,1l-1.9,1.1c-0.3,0.2-0.5,0-0.6-0.4l0-3.1%20c-1.6,0.7-3.2,0.8-4.3,0.4c-0.2-0.1-0.3-0.4-0.2-0.7l0.7-2.9c0.1-0.2,0.2-0.5,0.3-0.6c0.1-0.1,0.1-0.1,0.2-0.1%20c0.1-0.1,0.2-0.1,0.3,0c1.1,0.4,2.6,0.2,3.9-0.5c1.8-0.9,2.9-2.7,2.9-4.5c0-1.6-0.9-2.3-3-2.3c-2.7,0-5.2-0.5-5.3-4.5%20c0-3.3,1.7-6.7,4.4-8.8l0-3.2c0-0.4,0.2-0.8,0.5-1l1.8-1.2c0.3-0.2,0.5,0,0.6,0.4l0,3.2c1.3-0.5,2.5-0.7,3.6-0.4%20c0.2,0.1,0.3,0.4,0.2,0.7l-0.7,2.8c-0.1,0.2-0.2,0.4-0.3,0.6c-0.1,0.1-0.1,0.1-0.2,0.1c-0.1,0-0.2,0.1-0.3,0%20c-0.5-0.1-1.6-0.4-3.4,0.6c-1.9,1-2.6,2.6-2.5,3.8c0,1.5,0.8,1.9,3.3,1.9c3.4,0.1,4.9,1.6,5,5C94.7,103.4,92.9,107,90.1,109.3z%20M109.6,103.9c0,0.3,0,0.6-0.3,0.7l-9.4,5.7c-0.2,0.1-0.4,0-0.4-0.3v-2.4c0-0.3,0.2-0.5,0.4-0.6l9.3-5.5c0.2-0.1,0.4,0,0.4,0.3%20V103.9z%20M116.1,49.6L80.9,71.3c-4.4,2.6-7.6,5.4-7.6,10.7v43.4c0,3.2,1.3,5.2,3.2,5.8c-0.6,0.1-1.3,0.2-2,0.2%20c-2.1,0-4.1-0.6-5.9-1.6l-37.2-22.1c-3.6-2.2-5.9-6.2-5.9-10.5V53c0-4.3,2.3-8.4,5.9-10.5l37.2-22.1c1.8-1.1,3.8-1.6,5.9-1.6%20s4.1,0.6,5.9,1.6l37.2,22.1c3.1,1.8,5.1,5,5.7,8.5C122.1,48.4,119.3,47.7,116.1,49.6z'/%3e%3c/g%3e%3c/svg%3e";

var root_4$3 = from_html(`<span class="code svelte-17pn6bc"></span>`);
var root_3$3 = from_html(`<hr class="hr svelte-17pn6bc"/> <div style="margin:10px;"><p><!> <span class="code highlight svelte-17pn6bc"><!></span></p> <p class="desc svelte-17pn6bc"> </p></div>`, 1);
var root_7$2 = from_html(`<div class="load-wrap"><!></div>`);
var root$4 = from_html(`<h4 class="svelte-17pn6bc"><div class="toggle-icon svelte-17pn6bc"><div class="toggle-dot toggle-right svelte-17pn6bc"></div></div> Returns <!></h4> <div></div> <!>`, 1);

function ResponseSnippet($$anchor, $$props) {
	push($$props, false);

	let is_running = prop($$props, 'is_running', 12);
	let endpoint_returns = prop($$props, 'endpoint_returns', 12);
	let js_returns = prop($$props, 'js_returns', 12);
	let current_language = prop($$props, 'current_language', 12);

	var $$exports = {
		get is_running() {
			return is_running();
		},

		set is_running($$value) {
			is_running($$value);
			flushSync();
		},

		get endpoint_returns() {
			return endpoint_returns();
		},

		set endpoint_returns($$value) {
			endpoint_returns($$value);
			flushSync();
		},

		get js_returns() {
			return js_returns();
		},

		set js_returns($$value) {
			js_returns($$value);
			flushSync();
		},

		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		}
	};

	init();

	var fragment = root$4();
	var h4 = first_child(fragment);
	var node = sibling(child(h4), 2);

	{
		var consequent = ($$anchor) => {
			var text$1 = text();

			template_effect(() => set_text(text$1, `${current_language() == "python" ? "tuple" : "list"} of ${(
				deep_read_state(endpoint_returns()),
				untrack(() => endpoint_returns().length)
			) ?? ''}
		elements`));

			append($$anchor, text$1);
		};

		var alternate = ($$anchor) => {
			var text_1 = text('1 element');

			append($$anchor, text_1);
		};

		if_block(node, ($$render) => {
			if ((
				deep_read_state(endpoint_returns()),
				untrack(() => endpoint_returns().length > 1)
			)) $$render(consequent); else $$render(alternate, false);
		});
	}

	reset(h4);

	var div = sibling(h4, 2);
	let classes;

	each(div, 5, endpoint_returns, index, ($$anchor, $$item, i) => {
		let label = () => get($$item).label;
		let python_type = () => get($$item).python_type;
		let component = () => get($$item).component;
		var fragment_2 = root_3$3();
		var div_1 = sibling(first_child(fragment_2), 2);
		var p = child(div_1);
		var node_1 = child(p);

		{
			var consequent_1 = ($$anchor) => {
				var span = root_4$3();

				span.textContent = `[${i}]`;
				append($$anchor, span);
			};

			if_block(node_1, ($$render) => {
				if ((
					deep_read_state(endpoint_returns()),
					untrack(() => endpoint_returns().length > 1)
				)) $$render(consequent_1);
			});
		}

		var span_1 = sibling(node_1, 2);
		var node_2 = child(span_1);

		{
			var consequent_2 = ($$anchor) => {
				var text_2 = text();

				template_effect(() => set_text(text_2, (python_type(), untrack(() => python_type().type))));
				append($$anchor, text_2);
			};

			var alternate_1 = ($$anchor) => {
				var text_3 = text();

				template_effect(() => set_text(text_3, (
					deep_read_state(js_returns()),
					untrack(() => js_returns()[i].type)
				)));

				append($$anchor, text_3);
			};

			if_block(node_2, ($$render) => {
				if (current_language() === "python") $$render(consequent_2); else $$render(alternate_1, false);
			});
		}

		reset(span_1);
		reset(p);

		var p_1 = sibling(p, 2);
		var text_4 = child(p_1);

		reset(p_1);
		reset(div_1);

		template_effect(() => set_text(text_4, `The output value that appears in the "${label() ?? ''}" ${component() ?? ''}
				component.`));

		append($$anchor, fragment_2);
	});

	reset(div);

	var node_3 = sibling(div, 2);

	{
		var consequent_3 = ($$anchor) => {
			var div_2 = root_7$2();
			var node_4 = child(div_2);

			Loader(node_4, { margin: false });
			reset(div_2);
			append($$anchor, div_2);
		};

		if_block(node_3, ($$render) => {
			if (is_running()) $$render(consequent_3);
		});
	}

	template_effect(() => classes = set_class(div, 1, '', null, classes, { hide: is_running() }));
	append($$anchor, fragment);

	return pop($$exports);
}

const skill = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjE0IDIyIDEyMiAxMDYiIHN0eWxlPSJiYWNrZ3JvdW5kLWNvbG9yOiAjZmZmZmZmOyI+CjxyZWN0IHg9IjIyIiB5PSIyOCIgd2lkdGg9IjEwNiIgaGVpZ2h0PSI5NCIgcng9IjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSI2Ii8+CjxyZWN0IHg9IjIyIiB5PSIyOCIgd2lkdGg9IjEwNiIgaGVpZ2h0PSIyMCIgcng9IjgiIGZpbGw9IiMzNzQxNTEiIGZpbGwtb3BhY2l0eT0iMC4wOCIvPgo8Y2lyY2xlIGN4PSIzOCIgY3k9IjM4IiByPSI0IiBmaWxsPSIjZmY1ZjU3Ii8+CjxjaXJjbGUgY3g9IjUyIiBjeT0iMzgiIHI9IjQiIGZpbGw9IiNmZWJjMmUiLz4KPGNpcmNsZSBjeD0iNjYiIGN5PSIzOCIgcj0iNCIgZmlsbD0iIzI4Yzg0MCIvPgo8dGV4dCB4PSIzOCIgeT0iNzYiIGZpbGw9IiMzNzQxNTEiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMjIiIGZvbnQtd2VpZ2h0PSJib2xkIj4mZ3Q7XzwvdGV4dD4KPHJlY3QgeD0iMzgiIHk9Ijg4IiB3aWR0aD0iNTAiIGhlaWdodD0iNCIgcng9IjIiIGZpbGw9IiMzNzQxNTEiIGZpbGwtb3BhY2l0eT0iMC4zNSIvPgo8cmVjdCB4PSIzOCIgeT0iMTAwIiB3aWR0aD0iMzUiIGhlaWdodD0iNCIgcng9IjIiIGZpbGw9IiMzNzQxNTEiIGZpbGwtb3BhY2l0eT0iMC4yIi8+Cjwvc3ZnPgo=";

var root_1$1 = from_html(`<code class="svelte-1cyp2hi"><div class="copy svelte-1cyp2hi"><!></div> <div><pre class="svelte-1cyp2hi"> </pre></div></code>`);
var root_2$1 = from_html(`<code class="svelte-1cyp2hi"><div class="copy svelte-1cyp2hi"><!></div> <div><pre class="svelte-1cyp2hi"> </pre></div></code>`);
var root_3$2 = from_html(`<code class="svelte-1cyp2hi"><div class="copy svelte-1cyp2hi"><!></div> <div><pre class="svelte-1cyp2hi"> </pre></div></code>`);

var root$3 = from_html(`<div class="skill-content svelte-1cyp2hi"><p class="padded svelte-1cyp2hi">1. Make sure you are using the latest version of Gradio:</p> <!> <p class="padded svelte-1cyp2hi">2. Install the usage of this Space as a Skill for your coding agent by
		running this in your terminal:</p> <!> <p class="hint svelte-1cyp2hi">Instead of <span class="inline-code svelte-1cyp2hi">--claude</span>, you can use <span class="inline-code svelte-1cyp2hi">--cursor</span>, <span class="inline-code svelte-1cyp2hi">--codex</span>, or <span class="inline-code svelte-1cyp2hi">--opencode</span>. Combine flags to install for
		multiple agents. Use <span class="inline-code svelte-1cyp2hi">--global</span> to install user-level
		instead of per-project.</p> <p class="padded svelte-1cyp2hi">3. This will add a skill to your coding agent that describes how to use this
		Space via Python, JavaScript, and cURL API. The skill will look like this:</p> <!></div>`);

function SkillSnippet($$anchor, $$props) {
	push($$props, false);

	const effective_space_id = mutable_source();
	const skill_id = mutable_source();
	const install_gradio = mutable_source();
	const install_cmd_claude = mutable_source();
	const install_cmd_cursor = mutable_source();
	const install_cmd_codex = mutable_source();
	const skill_preview = mutable_source();
	let space_id = prop($$props, 'space_id', 12);

	legacy_pre_effect(() => (deep_read_state(space_id())), () => {
		set(effective_space_id, space_id() || "");
	});

	legacy_pre_effect(() => (get(effective_space_id)), () => {
		set(skill_id, get(effective_space_id).replace("/", "-"));
	});

	legacy_pre_effect(() => {}, () => {
		set(install_gradio, "pip install --upgrade gradio");
	});

	legacy_pre_effect(() => (get(effective_space_id)), () => {
		set(install_cmd_claude, `gradio skills add ${get(effective_space_id)} --claude`);
	});

	legacy_pre_effect(() => (get(effective_space_id)), () => {
		set(install_cmd_cursor, `gradio skills add ${get(effective_space_id)} --cursor`);
	});

	legacy_pre_effect(() => (get(effective_space_id)), () => {
		set(install_cmd_codex, `gradio skills add ${get(effective_space_id)} --codex`);
	});

	legacy_pre_effect(() => (get(skill_id), get(effective_space_id)), () => {
		set(skill_preview, `---
name: ${get(skill_id)}
description: Use the ${get(effective_space_id)} Gradio Space via API. Provides Python, JavaScript, and cURL usage examples.
---

# ${get(effective_space_id)}

This skill describes how to use the ${get(effective_space_id)} Gradio Space programmatically.

## API Endpoints
...`);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get space_id() {
			return space_id();
		},

		set space_id($$value) {
			space_id($$value);
			flushSync();
		}
	};

	var div = root$3();
	var node = sibling(child(div), 2);

	Block(node, {
		children: ($$anchor, $$slotProps) => {
			var code = root_1$1();
			var div_1 = child(code);
			var node_1 = child(div_1);

			CopyButton(node_1, {
				get code() {
					return get(install_gradio);
				}
			});

			reset(div_1);

			var div_2 = sibling(div_1, 2);
			var pre = child(div_2);
			var text = child(pre);

			reset(pre);
			reset(div_2);
			reset(code);
			template_effect(() => set_text(text, `$ ${get(install_gradio) ?? ''}`));
			append($$anchor, code);
		},
		$$slots: { default: true }
	});

	var node_2 = sibling(node, 4);

	Block(node_2, {
		children: ($$anchor, $$slotProps) => {
			var code_1 = root_2$1();
			var div_3 = child(code_1);
			var node_3 = child(div_3);

			CopyButton(node_3, {
				get code() {
					return get(install_cmd_claude);
				}
			});

			reset(div_3);

			var div_4 = sibling(div_3, 2);
			var pre_1 = child(div_4);
			var text_1 = child(pre_1);

			reset(pre_1);
			reset(div_4);
			reset(code_1);
			template_effect(() => set_text(text_1, `$ ${get(install_cmd_claude) ?? ''}`));
			append($$anchor, code_1);
		},
		$$slots: { default: true }
	});

	var node_4 = sibling(node_2, 6);

	Block(node_4, {
		children: ($$anchor, $$slotProps) => {
			var code_2 = root_3$2();
			var div_5 = child(code_2);
			var node_5 = child(div_5);

			CopyButton(node_5, {
				get code() {
					return get(skill_preview);
				}
			});

			reset(div_5);

			var div_6 = sibling(div_5, 2);
			var pre_2 = child(div_6);
			var text_2 = child(pre_2, true);

			reset(pre_2);
			reset(div_6);
			reset(code_2);
			template_effect(() => set_text(text_2, get(skill_preview)));
			append($$anchor, code_2);
		},
		$$slots: { default: true }
	});

	reset(div);
	append($$anchor, div);

	return pop($$exports);
}

const mcp = ""+new URL('mcp-DNm9doVd.svg', import.meta.url).href+"";

var root_3$1 = from_html(`<button type="button"> </button>`);

var root_5$1 = from_html(`<div class="mcp-url svelte-1uo4thf"><label for="mcp-server-url" class="svelte-1uo4thf"><span class="status-indicator active svelte-1uo4thf">●</span>MCP Server URL
					(Streamable HTTP)</label> <div class="textbox svelte-1uo4thf"><input id="mcp-server-url" type="text" readonly="" class="svelte-1uo4thf"/> <!></div></div>`);

var root_4$2 = from_html(`<!> <p class="svelte-1uo4thf">&nbsp;</p>`, 1);
var root_6$1 = from_html(`<div class="tool-selection-controls svelte-1uo4thf"><button class="select-all-btn svelte-1uo4thf">Select All</button> <button class="select-none-btn svelte-1uo4thf">Select None</button></div>`);
var root_8$1 = from_html(`<input type="checkbox" class="tool-checkbox svelte-1uo4thf"/>`);
var root_10 = from_html(`<div class="tool-analytics-wrapper svelte-1uo4thf" style="margin-left: 1em;"><span class="tool-analytics svelte-1uo4thf" style="color: var(--body-text-color-subdued);"> </span></div>`);
var root_13$1 = from_html(`<div class="parameter svelte-1uo4thf"><code class="svelte-1uo4thf"> </code> <span class="parameter-type svelte-1uo4thf"> </span> <p class="parameter-description svelte-1uo4thf"> </p></div>`);
var root_12$1 = from_html(`<div class="tool-parameters svelte-1uo4thf"></div>`);
var root_14$1 = from_html(`<p class="svelte-1uo4thf">Takes no input parameters</p>`);
var root_11$1 = from_html(`<div class="tool-content svelte-1uo4thf"><!></div>`);
var root_7$1 = from_html(`<div class="tool-item svelte-1uo4thf"><div class="tool-header-wrapper svelte-1uo4thf"><!> <button class="tool-header svelte-1uo4thf"><span style="display: inline-block" class="svelte-1uo4thf"><span style="display: inline-block; padding-right: 6px; vertical-align: sub" class="svelte-1uo4thf"><!></span> <span class="tool-name svelte-1uo4thf"> </span> &nbsp; <span class="tool-description svelte-1uo4thf"> </span> <!></span> <span class="tool-arrow svelte-1uo4thf"> </span></button></div> <!></div>`);
var root_15 = from_html(`<code class="svelte-1uo4thf"><div class="copy svelte-1uo4thf"><!></div> <div class="svelte-1uo4thf"><pre class="svelte-1uo4thf"> </pre></div></code>`);
var root_16 = from_html(`<code class="svelte-1uo4thf"><div class="copy svelte-1uo4thf"><!></div> <div class="svelte-1uo4thf"><pre class="svelte-1uo4thf"> </pre></div></code>`);

var root_17$1 = from_html(`<div class="file-upload-section svelte-1uo4thf"><label class="checkbox-label svelte-1uo4thf"><input type="checkbox" class="checkbox svelte-1uo4thf"/> Include Gradio file upload tool</label> <p class="file-upload-explanation svelte-1uo4thf">The <code class="svelte-1uo4thf">upload_files_to_gradio</code> tool uploads files from your
				local <code class="svelte-1uo4thf">UPLOAD_DIRECTORY</code> (or any of its subdirectories) to
				the Gradio app. This is needed because MCP servers require files to be
				provided as URLs. You can omit this tool if you prefer to upload files
				manually. This tool requires <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank" class="svelte-1uo4thf">uv</a> to be installed.</p></div>`);

var root_2 = from_html(
	`<div class="transport-selection svelte-1uo4thf"><div class="snippets svelte-1uo4thf"><span class="transport-label svelte-1uo4thf">Transport:</span> <!></div></div> <!> <div class="tool-selection svelte-1uo4thf"><strong class="svelte-1uo4thf"> <span style="display: inline-block; vertical-align: sub;" class="svelte-1uo4thf"><!></span>), Resources (<span style="display: inline-block; vertical-align: sub;" class="svelte-1uo4thf"><!></span>), and Prompts (<span style="display: inline-block; vertical-align: sub;" class="svelte-1uo4thf"><!></span>)</strong> <!></div> <div class="mcp-tools svelte-1uo4thf"></div> <p class="svelte-1uo4thf">&nbsp;</p> <div><strong class="svelte-1uo4thf">Streamable HTTP Transport</strong>: To add this MCP to clients that
		support Streamable HTTP, simply add the following configuration to your MCP
		config. <p class="svelte-1uo4thf">&nbsp;</p> <!></div> <div><strong class="svelte-1uo4thf">STDIO Transport</strong>: For clients that only support stdio (e.g.
		Claude Desktop), first <a href="https://nodejs.org/en/download/" target="_blank" class="svelte-1uo4thf">install Node.js</a>. Then, you can use the following command: <p class="svelte-1uo4thf">&nbsp;</p> <!></div> <!> <p class="svelte-1uo4thf">&nbsp;</p> <p class="svelte-1uo4thf"><a target="_blank" class="svelte-1uo4thf">Read more about MCP in the Gradio docs</a></p>`,
	1
);

var root_18 = from_html(
	`This Gradio app can also serve as an MCP server, with an MCP tool
	corresponding to each API endpoint. To enable this, launch this Gradio app
	with <code class="svelte-1uo4thf">.launch(mcp_server=True)</code> or set the <code class="svelte-1uo4thf">GRADIO_MCP_SERVER</code> env variable to <code class="svelte-1uo4thf">"True"</code>.`,
	1
);

function MCPSnippet($$anchor, $$props) {
	push($$props, false);

	const display_url = mutable_source();
	const mcp_json_streamable_http = mutable_source();
	const mcp_json_stdio_updated = mutable_source();
	let mcp_server_active = prop($$props, 'mcp_server_active', 12);
	let mcp_server_url_streamable = prop($$props, 'mcp_server_url_streamable', 12);
	let root = prop($$props, 'root', 12);
	let tools = prop($$props, 'tools', 12);
	let all_tools = prop($$props, 'all_tools', 28, () => []);
	let selected_tools = prop($$props, 'selected_tools', 28, () => new Set());
	let mcp_json_stdio = prop($$props, 'mcp_json_stdio', 12);
	let file_data_present = prop($$props, 'file_data_present', 12);
	let mcp_docs = prop($$props, 'mcp_docs', 12);
	let analytics = prop($$props, 'analytics', 12);
	let config_snippets = prop($$props, 'config_snippets', 12);
	let current_transport = mutable_source("streamable_http");
	let include_file_upload = mutable_source(true);
	const transports = [["streamable_http", "Streamable HTTP"], ["stdio", "STDIO"]];
	const tool_type_icons = { tool: Tool, resource: Resource, prompt: Prompt };

	// Helper function to add/remove file upload tool from config
	function update_config_with_file_upload(base_config, include_upload) {
		if (!base_config) return null;

		const config = JSON.parse(JSON.stringify(base_config));

		if (include_upload && file_data_present()) {
			const upload_file_mcp_server = {
				command: "uvx",
				args: [
					"--from",
					"gradio[mcp]",
					"gradio",
					"upload-mcp",
					root(),
					"<UPLOAD_DIRECTORY>"
				]
			};

			config.mcpServers.upload_files_to_gradio = upload_file_mcp_server;
		} else {
			delete config.mcpServers?.upload_files_to_gradio;
		}

		return config;
	}

	legacy_pre_effect(() => (deep_read_state(mcp_server_url_streamable())), () => {
		set(display_url, mcp_server_url_streamable());
	});

	legacy_pre_effect(
		() => (
			deep_read_state(mcp_json_stdio()),
			deep_read_state(mcp_server_url_streamable()),
			get(include_file_upload)
		),
		() => {
			set(mcp_json_streamable_http, update_config_with_file_upload(
				mcp_json_stdio()
					? { mcpServers: { gradio: { url: mcp_server_url_streamable() } } }
					: null,
				get(include_file_upload)
			));
		}
	);

	legacy_pre_effect(
		() => (
			deep_read_state(mcp_json_stdio()),
			get(include_file_upload)
		),
		() => {
			set(mcp_json_stdio_updated, update_config_with_file_upload(mcp_json_stdio(), get(include_file_upload)));
		}
	);

	legacy_pre_effect(
		() => (
			get(mcp_json_streamable_http),
			get(mcp_json_stdio_updated)
		),
		() => {
			config_snippets({
				streamable_http: JSON.stringify(get(mcp_json_streamable_http), null, 2),
				stdio: JSON.stringify(get(mcp_json_stdio_updated), null, 2)
			});
		}
	);

	legacy_pre_effect_reset();

	var $$exports = {
		get mcp_server_active() {
			return mcp_server_active();
		},

		set mcp_server_active($$value) {
			mcp_server_active($$value);
			flushSync();
		},

		get mcp_server_url_streamable() {
			return mcp_server_url_streamable();
		},

		set mcp_server_url_streamable($$value) {
			mcp_server_url_streamable($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get tools() {
			return tools();
		},

		set tools($$value) {
			tools($$value);
			flushSync();
		},

		get all_tools() {
			return all_tools();
		},

		set all_tools($$value) {
			all_tools($$value);
			flushSync();
		},

		get selected_tools() {
			return selected_tools();
		},

		set selected_tools($$value) {
			selected_tools($$value);
			flushSync();
		},

		get mcp_json_stdio() {
			return mcp_json_stdio();
		},

		set mcp_json_stdio($$value) {
			mcp_json_stdio($$value);
			flushSync();
		},

		get file_data_present() {
			return file_data_present();
		},

		set file_data_present($$value) {
			file_data_present($$value);
			flushSync();
		},

		get mcp_docs() {
			return mcp_docs();
		},

		set mcp_docs($$value) {
			mcp_docs($$value);
			flushSync();
		},

		get analytics() {
			return analytics();
		},

		set analytics($$value) {
			analytics($$value);
			flushSync();
		},

		get config_snippets() {
			return config_snippets();
		},

		set config_snippets($$value) {
			config_snippets($$value);
			flushSync();
		}
	};

	init();

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_8 = ($$anchor) => {
			var fragment_1 = root_2();
			var div = first_child(fragment_1);
			var div_1 = child(div);
			var node_1 = sibling(child(div_1), 2);

			each(node_1, 1, () => transports, index, ($$anchor, $$item) => {
				var $$array = user_derived(() => to_array(get($$item), 2));
				let transport = () => get($$array)[0];
				let display_name = () => get($$array)[1];
				var button = root_3$1();
				var text = child(button, true);

				reset(button);

				template_effect(() => {
					set_class(button, 1, `snippet ${get(current_transport) === transport() ? 'current-lang' : 'inactive-lang'}`, 'svelte-1uo4thf');
					set_text(text, display_name());
				});

				event('click', button, () => set(current_transport, transport()));
				append($$anchor, button);
			});

			reset(div_1);
			reset(div);

			var node_2 = sibling(div, 2);

			{
				var consequent = ($$anchor) => {
					var fragment_2 = root_4$2();
					var node_3 = first_child(fragment_2);

					Block(node_3, {
						children: ($$anchor, $$slotProps) => {
							var div_2 = root_5$1();
							var div_3 = sibling(child(div_2), 2);
							var input = child(div_3);

							remove_input_defaults(input);

							var node_4 = sibling(input, 2);

							CopyButton(node_4, {
								get code() {
									return get(display_url);
								}
							});

							reset(div_3);
							reset(div_2);
							template_effect(() => set_value(input, get(display_url)));
							append($$anchor, div_2);
						},
						$$slots: { default: true }
					});

					next(2);
					append($$anchor, fragment_2);
				};

				if_block(node_2, ($$render) => {
					if (get(current_transport) !== "stdio") $$render(consequent);
				});
			}

			var div_4 = sibling(node_2, 2);
			var strong = child(div_4);
			var text_1 = child(strong);
			var span = sibling(text_1);
			var node_5 = child(span);

			Tool(node_5, {});
			reset(span);

			var span_1 = sibling(span, 2);
			var node_6 = child(span_1);

			Resource(node_6, {});
			reset(span_1);

			var span_2 = sibling(span_1, 2);
			var node_7 = child(span_2);

			Prompt(node_7, {});
			reset(span_2);
			next();
			reset(strong);

			var node_8 = sibling(strong, 2);

			{
				var consequent_1 = ($$anchor) => {
					var div_5 = root_6$1();
					var button_1 = child(div_5);
					var button_2 = sibling(button_1, 2);

					reset(div_5);

					event('click', button_1, () => {
						selected_tools(new Set(all_tools().map((t) => t.name)));
					});

					event('click', button_2, () => {
						selected_tools(new Set());
					});

					append($$anchor, div_5);
				};

				if_block(node_8, ($$render) => {
					if ((
						deep_read_state(all_tools()),
						untrack(() => all_tools().length > 0)
					)) $$render(consequent_1);
				});
			}

			reset(div_4);

			var div_6 = sibling(div_4, 2);

			each(
				div_6,
				5,
				() => (
					deep_read_state(all_tools()),
					deep_read_state(tools()),
					untrack(() => all_tools().length > 0 ? all_tools() : tools())
				),
				index,
				($$anchor, tool, $$index_2) => {
					const success_rate = derived_safe_equal(() => (
						deep_read_state(analytics()),
						get(tool),
						untrack(() => analytics()[get(tool).meta.endpoint_name]?.success_rate || 0)
					));

					var div_7 = root_7$1();
					var div_8 = child(div_7);
					var node_9 = child(div_8);

					{
						var consequent_2 = ($$anchor) => {
							var input_1 = root_8$1();

							remove_input_defaults(input_1);

							template_effect(
								($0) => {
									set_checked(input_1, $0);
									input_1.disabled = get(current_transport) !== "streamable_http";
									set_style(input_1, get(current_transport) !== "streamable_http" ? "opacity: 0.5; cursor: not-allowed;" : "");
								},
								[
									() => (
										deep_read_state(selected_tools()),
										get(tool),
										get(current_transport),
										untrack(() => selected_tools().has(get(tool).name) || get(current_transport) !== "streamable_http")
									)
								]
							);

							event('change', input_1, (e) => {
								if (e.currentTarget.checked) {
									selected_tools().add(get(tool).name);
								} else {
									selected_tools().delete(get(tool).name);
								}

								selected_tools(selected_tools());
							});

							append($$anchor, input_1);
						};

						if_block(node_9, ($$render) => {
							if ((
								deep_read_state(all_tools()),
								untrack(() => all_tools().length > 0)
							)) $$render(consequent_2);
						});
					}

					var button_3 = sibling(node_9, 2);
					var span_3 = child(button_3);
					var span_4 = child(span_3);
					var node_10 = child(span_4);

					{
						var consequent_3 = ($$anchor) => {
							const Icon = derived_safe_equal(() => (
								get(tool),
								untrack(() => tool_type_icons[get(tool).meta.mcp_type])
							));

							get(Icon)($$anchor, {});
						};

						if_block(node_10, ($$render) => {
							if ((
								get(tool),
								untrack(() => tool_type_icons[get(tool).meta.mcp_type])
							)) $$render(consequent_3);
						});
					}

					reset(span_4);

					var span_5 = sibling(span_4, 2);
					var text_2 = child(span_5, true);

					reset(span_5);

					var span_6 = sibling(span_5, 2);
					var text_3 = child(span_6, true);

					reset(span_6);

					var node_11 = sibling(span_6, 2);

					{
						var consequent_4 = ($$anchor) => {
							const endpoint_analytics = derived_safe_equal(() => (
								deep_read_state(analytics()),
								get(tool),
								untrack(() => analytics()[get(tool).meta.endpoint_name])
							));

							const p50 = derived_safe_equal(() => (
								deep_read_state(get(endpoint_analytics)),
								untrack(() => get(endpoint_analytics).process_time_percentiles["50th"])
							));

							var div_9 = root_10();
							var span_7 = child(div_9);
							var text_4 = child(span_7);

							reset(span_7);
							reset(div_9);

							template_effect(
								($0, $1) => set_text(text_4, `${(
									deep_read_state(get(endpoint_analytics)),
									untrack(() => get(endpoint_analytics).total_requests)
								) ?? ''} requests (${$0 ?? ''}% successful, p50: ${$1 ?? ''})`),
								[
									() => (
										deep_read_state(get(success_rate)),
										untrack(() => Math.round(get(success_rate) * 100))
									),

									() => (
										deep_read_state(format_latency),
										deep_read_state(get(p50)),
										untrack(() => format_latency(get(p50)))
									)
								]
							);

							append($$anchor, div_9);
						};

						if_block(node_11, ($$render) => {
							if ((
								deep_read_state(analytics()),
								get(tool),
								untrack(() => analytics()[get(tool).meta.endpoint_name])
							)) $$render(consequent_4);
						});
					}

					reset(span_3);

					var span_8 = sibling(span_3, 2);
					var text_5 = child(span_8, true);

					reset(span_8);
					reset(button_3);
					reset(div_8);

					var node_12 = sibling(div_8, 2);

					{
						var consequent_6 = ($$anchor) => {
							var div_10 = root_11$1();
							var node_13 = child(div_10);

							{
								var consequent_5 = ($$anchor) => {
									var div_11 = root_12$1();

									each(
										div_11,
										5,
										() => (
											get(tool),
											untrack(() => Object.entries(get(tool).parameters))
										),
										index,
										($$anchor, $$item) => {
											var $$array_1 = user_derived(() => to_array(get($$item), 2));
											let name = () => get($$array_1)[0];
											let param = () => get($$array_1)[1];
											var div_12 = root_13$1();
											var code = child(div_12);
											var text_6 = child(code, true);

											reset(code);

											var span_9 = sibling(code, 2);
											var text_7 = child(span_9);

											reset(span_9);

											var p = sibling(span_9, 2);
											var text_8 = child(p, true);

											reset(p);
											reset(div_12);

											template_effect(
												($0) => {
													set_text(text_6, name());
													set_text(text_7, `(${(param(), untrack(() => param().type)) ?? ''}${$0 ?? ''})`);

													set_text(text_8, (
														param(),
														untrack(() => param().description
															? param().description
															: "⚠︎ No description for this parameter in function docstring")
													));
												},
												[
													() => (
														param(),
														untrack(() => param().default !== undefined ? `, default: ${JSON.stringify(param().default)}` : "")
													)
												]
											);

											append($$anchor, div_12);
										}
									);

									reset(div_11);
									append($$anchor, div_11);
								};

								var alternate = ($$anchor) => {
									var p_1 = root_14$1();

									append($$anchor, p_1);
								};

								if_block(node_13, ($$render) => {
									if ((
										get(tool),
										untrack(() => Object.keys(get(tool).parameters).length > 0)
									)) $$render(consequent_5); else $$render(alternate, false);
								});
							}

							reset(div_10);
							append($$anchor, div_10);
						};

						if_block(node_12, ($$render) => {
							if ((get(tool), untrack(() => get(tool).expanded))) $$render(consequent_6);
						});
					}

					reset(div_7);

					template_effect(() => {
						set_text(text_2, (get(tool), untrack(() => get(tool).name)));

						set_text(text_3, (
							get(tool),
							untrack(() => get(tool).description
								? get(tool).description
								: "⚠︎ No description provided in function docstring")
						));

						set_text(text_5, (
							get(tool),
							untrack(() => get(tool).expanded ? "▼" : "▶")
						));
					});

					event('click', button_3, () => (
						get(tool).expanded = !get(tool).expanded,
						invalidate_inner_signals(() => (all_tools(), tools()))
					));

					append($$anchor, div_7);
				}
			);

			reset(div_6);

			var div_13 = sibling(div_6, 4);
			let classes;
			var node_14 = sibling(child(div_13), 4);

			Block(node_14, {
				children: ($$anchor, $$slotProps) => {
					var code_1 = root_15();
					var div_14 = child(code_1);
					var node_15 = child(div_14);

					CopyButton(node_15, {
						get code() {
							return (
								deep_read_state(config_snippets()),
								untrack(() => config_snippets().streamable_http)
							);
						}
					});

					reset(div_14);

					var div_15 = sibling(div_14, 2);
					var pre = child(div_15);
					var text_9 = child(pre, true);

					reset(pre);
					reset(div_15);
					reset(code_1);

					template_effect(() => set_text(text_9, (
						deep_read_state(config_snippets()),
						untrack(() => config_snippets().streamable_http)
					)));

					append($$anchor, code_1);
				},
				$$slots: { default: true }
			});

			reset(div_13);

			var div_16 = sibling(div_13, 2);
			let classes_1;
			var node_16 = sibling(child(div_16), 6);

			Block(node_16, {
				children: ($$anchor, $$slotProps) => {
					var code_2 = root_16();
					var div_17 = child(code_2);
					var node_17 = child(div_17);

					CopyButton(node_17, {
						get code() {
							return (
								deep_read_state(config_snippets()),
								untrack(() => config_snippets().stdio)
							);
						}
					});

					reset(div_17);

					var div_18 = sibling(div_17, 2);
					var pre_1 = child(div_18);
					var text_10 = child(pre_1, true);

					reset(pre_1);
					reset(div_18);
					reset(code_2);

					template_effect(() => set_text(text_10, (
						deep_read_state(config_snippets()),
						untrack(() => config_snippets().stdio)
					)));

					append($$anchor, code_2);
				},
				$$slots: { default: true }
			});

			reset(div_16);

			var node_18 = sibling(div_16, 2);

			{
				var consequent_7 = ($$anchor) => {
					var div_19 = root_17$1();
					var label = child(div_19);
					var input_2 = child(label);

					remove_input_defaults(input_2);
					next();
					reset(label);
					next(2);
					reset(div_19);
					bind_checked(input_2, () => get(include_file_upload), ($$value) => set(include_file_upload, $$value));
					append($$anchor, div_19);
				};

				if_block(node_18, ($$render) => {
					if (file_data_present()) $$render(consequent_7);
				});
			}

			var p_2 = sibling(node_18, 4);
			var a = child(p_2);

			reset(p_2);

			template_effect(() => {
				set_text(text_1, `${(
					deep_read_state(all_tools()),
					deep_read_state(tools()),
					untrack(() => all_tools().length > 0 ? all_tools().length : tools().length)
				) ?? ''} Available MCP Tools
			(`);

				classes = set_class(div_13, 1, 'svelte-1uo4thf', null, classes, { hidden: get(current_transport) !== "streamable_http" });
				classes_1 = set_class(div_16, 1, 'svelte-1uo4thf', null, classes_1, { hidden: get(current_transport) !== "stdio" });
				set_attribute(a, 'href', mcp_docs());
			});

			append($$anchor, fragment_1);
		};

		var alternate_1 = ($$anchor) => {
			var fragment_4 = root_18();

			next(6);
			append($$anchor, fragment_4);
		};

		if_block(node, ($$render) => {
			if (mcp_server_active()) $$render(consequent_8); else $$render(alternate_1, false);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

var root$2 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7h10v10"></path><path d="M7 17 17 7"></path></svg>`);

function IconArrowUpRight($$anchor, $$props) {
	push($$props, false);

	let classNames = prop($$props, 'classNames', 12, "");

	var $$exports = {
		get classNames() {
			return classNames();
		},

		set classNames($$value) {
			classNames($$value);
			flushSync();
		}
	};

	var svg = root$2();

	template_effect(() => set_class(svg, 0, clsx(classNames() || "menu-icon-arrow"), 'svelte-1s9m4x9'));
	append($$anchor, svg);

	return pop($$exports);
}

var root$1 = from_svg(`<svg width="1em" height="1em" viewBox="0 0 12 7" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 1L6 6L11 1" stroke="currentColor"></path></svg>`);

function IconCaret($$anchor, $$props) {
	push($$props, false);

	let classNames = prop($$props, 'classNames', 12, "");

	var $$exports = {
		get classNames() {
			return classNames();
		},

		set classNames($$value) {
			classNames($$value);
			flushSync();
		}
	};

	var svg = root$1();

	template_effect(() => set_class(svg, 0, clsx(classNames()), 'svelte-1u0oj9c'));
	append($$anchor, svg);

	return pop($$exports);
}

var root = from_svg(`<svg width="1em" height="1em" viewBox="5 5 22 22" fill="none" xmlns="http://www.w3.org/2000/svg"><title>HuggingChat</title><path d="M16.0006 25.9992C13.8266 25.999 11.7118 25.2901 9.97686 23.9799C8.2419 22.6698 6.98127 20.8298 6.38599 18.7388C5.79071 16.6478 5.89323 14.4198 6.678 12.3923C7.46278 10.3648 8.88705 8.64837 10.735 7.50308C12.5829 6.35779 14.7538 5.84606 16.9187 6.04544C19.0837 6.24481 21.1246 7.14442 22.7323 8.60795C24.34 10.0715 25.4268 12.0192 25.8281 14.1559C26.2293 16.2926 25.9232 18.5019 24.9561 20.449C24.7703 20.8042 24.7223 21.2155 24.8211 21.604L25.4211 23.8316C25.4803 24.0518 25.4805 24.2837 25.4216 24.5039C25.3627 24.7242 25.2468 24.925 25.0856 25.0862C24.9244 25.2474 24.7235 25.3633 24.5033 25.4222C24.283 25.4811 24.0512 25.4809 23.831 25.4217L21.6034 24.8217C21.2172 24.7248 20.809 24.7729 20.4558 24.9567C19.0683 25.6467 17.5457 26.0068 16.0006 26.0068V25.9992Z" fill="black" class="block dark:hidden svelte-pw1cre"></path><path d="M9.62598 16.0013C9.62598 15.3799 10.1294 14.8765 10.7508 14.8765C11.3721 14.8765 11.8756 15.3799 11.8756 16.0013C11.8756 17.0953 12.3102 18.1448 13.0838 18.9184C13.8574 19.692 14.9069 20.1266 16.001 20.1267C17.095 20.1267 18.1445 19.692 18.9181 18.9184C19.6918 18.1448 20.1264 17.0953 20.1264 16.0013C20.1264 15.3799 20.6299 14.8765 21.2512 14.8765C21.8725 14.8765 22.3759 15.3799 22.3759 16.0013C22.3759 17.6921 21.7046 19.3137 20.509 20.5093C19.3134 21.7049 17.6918 22.3762 16.001 22.3762C14.3102 22.3762 12.6885 21.7049 11.4929 20.5093C10.2974 19.3137 9.62598 17.6921 9.62598 16.0013Z" fill="white" class="block dark:hidden svelte-pw1cre"></path><path d="M16.0006 25.9992C13.8266 25.999 11.7118 25.2901 9.97686 23.9799C8.2419 22.6698 6.98127 20.8298 6.38599 18.7388C5.79071 16.6478 5.89323 14.4198 6.678 12.3923C7.46278 10.3648 8.88705 8.64837 10.735 7.50308C12.5829 6.35779 14.7538 5.84606 16.9187 6.04544C19.0837 6.24481 21.1246 7.14442 22.7323 8.60795C24.34 10.0715 25.4268 12.0192 25.8281 14.1559C26.2293 16.2926 25.9232 18.5019 24.9561 20.449C24.7703 20.8042 24.7223 21.2155 24.8211 21.604L25.4211 23.8316C25.4803 24.0518 25.4805 24.2837 25.4216 24.5039C25.3627 24.7242 25.2468 24.925 25.0856 25.0862C24.9244 25.2474 24.7235 25.3633 24.5033 25.4222C24.283 25.4811 24.0512 25.4809 23.831 25.4217L21.6034 24.8217C21.2172 24.7248 20.809 24.7729 20.4558 24.9567C19.0683 25.6467 17.5457 26.0068 16.0006 26.0068V25.9992Z" fill="white" class="hidden dark:block svelte-pw1cre"></path><path d="M9.62598 16.0013C9.62598 15.3799 10.1294 14.8765 10.7508 14.8765C11.3721 14.8765 11.8756 15.3799 11.8756 16.0013C11.8756 17.0953 12.3102 18.1448 13.0838 18.9184C13.8574 19.692 14.9069 20.1266 16.001 20.1267C17.095 20.1267 18.1445 19.692 18.9181 18.9184C19.6918 18.1448 20.1264 17.0953 20.1264 16.0013C20.1264 15.3799 20.6299 14.8765 21.2512 14.8765C21.8725 14.8765 22.3759 15.3799 22.3759 16.0013C22.3759 17.6921 21.7046 19.3137 20.509 20.5093C19.3134 21.7049 17.6918 22.3762 16.001 22.3762C14.3102 22.3762 12.6885 21.7049 11.4929 20.5093C10.2974 19.3137 9.62598 17.6921 9.62598 16.0013Z" fill="black" class="hidden dark:block svelte-pw1cre"></path></svg>`);

function IconHuggingChat($$anchor, $$props) {
	push($$props, false);

	let classNames = prop($$props, 'classNames', 12, "");

	var $$exports = {
		get classNames() {
			return classNames();
		},

		set classNames($$value) {
			classNames($$value);
			flushSync();
		}
	};

	var svg = root();

	template_effect(() => set_class(svg, 0, clsx(classNames() || "icon-size"), 'svelte-pw1cre'));
	append($$anchor, svg);

	return pop($$exports);
}

var root_4$1 = from_html(`<div class="backdrop-overlay svelte-de9ybk" aria-hidden="true" style="background: transparent;"></div> <div role="menu" class="menu-dropdown svelte-de9ybk" aria-label="Copy menu"><button role="menuitem" class="base-menu-item svelte-de9ybk"><div class="menu-icon-container svelte-de9ybk"><!></div> <div class="menu-text-container svelte-de9ybk"><div class="menu-text-primary svelte-de9ybk">Copy Page</div> <div class="menu-text-secondary svelte-de9ybk"> </div></div></button> <button role="menuitem" class="base-menu-item svelte-de9ybk"><div class="menu-icon-container svelte-de9ybk"><!></div> <div class="menu-text-container svelte-de9ybk"><div class="menu-text-primary svelte-de9ybk">Open in HuggingChat <!></div> <div class="menu-text-secondary svelte-de9ybk"> </div></div></button></div>`, 1);
var root_1 = from_html(`<div class="container-wrapper svelte-de9ybk"><div class="trigger-wrapper svelte-de9ybk"><button class="copy-button svelte-de9ybk" aria-live="polite"><span class="icon-wrapper svelte-de9ybk"><!></span> <span> </span></button> <button class="menu-toggle-button svelte-de9ybk" aria-haspopup="menu"><!></button></div> <!></div>`);

function CopyMarkdown($$anchor, $$props) {
	push($$props, false);

	let current_language = prop($$props, 'current_language', 12);
	let space_id = prop($$props, 'space_id', 12);
	let root = prop($$props, 'root', 12);
	let api_count = prop($$props, 'api_count', 12);
	let tools = prop($$props, 'tools', 12);
	let py_docs = prop($$props, 'py_docs', 12);
	let js_docs = prop($$props, 'js_docs', 12);
	let bash_docs = prop($$props, 'bash_docs', 12);
	let mcp_docs = prop($$props, 'mcp_docs', 12);
	let spaces_docs_suffix = prop($$props, 'spaces_docs_suffix', 12);
	let mcp_server_active = prop($$props, 'mcp_server_active', 12);
	let mcp_server_url_streamable = prop($$props, 'mcp_server_url_streamable', 12);
	let config_snippets = prop($$props, 'config_snippets', 12);
	let markdown_code_snippets = prop($$props, 'markdown_code_snippets', 12);
	let dependencies = prop($$props, 'dependencies', 12);
	let info = prop($$props, 'info', 12);
	let js_info = prop($$props, 'js_info', 12);
	let markdown_content = mutable_source({ python: "", javascript: "", bash: "", mcp: "" });

	/* eslint-disable complexity */
	/* eslint-enable complexity */
	let current_language_label = mutable_source(current_language() === "python"
		? "Python"
		: current_language() === "javascript"
			? "JavaScript"
			: current_language() === "bash" ? "Bash" : "MCP");

	let label = mutable_source(`Copy ${get(current_language_label)} Docs as Markdown for LLMs`);
	let copied = mutable_source(false);
	let open = mutable_source(false);
	let triggerEl = mutable_source(null);
	let menuEl = mutable_source(null);
	let menuStyle = mutable_source("");
	const isClient = typeof window !== "undefined";

	function openMenu() {
		set(open, true);

		if (isClient && get(triggerEl)) {
			void tick().then(() => {
				if (!get(triggerEl)) return;

				const rect = get(triggerEl).getBoundingClientRect();
				const gutter = 6;
				const minWidth = Math.max(rect.width + 80, 220);
				const right = Math.max(window.innerWidth - rect.right, gutter);

				set(menuStyle, `top:${rect.bottom + gutter}px;right:${right}px;min-width:${minWidth}px;`);
			});
		}
	}

	function closeMenu() {
		set(open, false);
	}

	function toggleMenu() {
		get(open) ? closeMenu() : openMenu();
	}

	function buildUrl() {
		const encodedPromptText = encodeURIComponent(`--------------------------------
${get(markdown_content)[current_language()]}
--------------------------------

Read the documentation above so I can ask questions about it.`);

		return `https://huggingface.co/chat/?prompt=${encodedPromptText}`;
	}

	function openHuggingChat() {
		if (isClient) {
			window.open(buildUrl(), "_blank", "noopener,noreferrer");
		}

		closeMenu();
	}

	function handleWindowPointer(event) {
		if (!get(open) || !isClient) return;

		const targetNode = event.target;

		if (get(menuEl)?.contains(targetNode) || get(triggerEl)?.contains(targetNode)) {
			return;
		}

		closeMenu();
	}

	function handleWindowKeydown(event) {
		if (event.key === "Escape" && get(open)) {
			closeMenu();
		}
	}

	function handleWindowResize() {
		if (get(open)) closeMenu();
	}

	function handleWindowScroll() {
		if (get(open)) closeMenu();
	}

	async function copyMarkdown(current_language) {
		try {
			if (!get(markdown_content)[current_language]) {
				console.warn("Nothing to copy");

				return;
			}

			const hasNavigatorClipboard = typeof navigator !== "undefined" && !!navigator.clipboard && typeof navigator.clipboard.writeText === "function";

			if (hasNavigatorClipboard) {
				await navigator.clipboard.writeText(get(markdown_content)[current_language]);
			} else {
				console.warn("Clipboard API unavailable");

				return;
			}

			set(copied, true);

			setTimeout(
				() => {
					set(copied, false);
				},
				1500
			);
		} catch(error) {
			console.error("Failed to write to clipboard", error);
		}
	}

	legacy_pre_effect(
		() => (
			deep_read_state(space_id()),
			deep_read_state(root()),
			deep_read_state(api_count()),
			deep_read_state(py_docs()),
			deep_read_state(spaces_docs_suffix()),
			deep_read_state(dependencies()),
			deep_read_state(info()),
			deep_read_state(markdown_code_snippets()),
			deep_read_state(js_info())
		),
		() => {
			mutate(markdown_content, get(markdown_content).python = `
# Python API documentation for ${space_id() || root()}
API Endpoints: ${api_count()}

1. Install the Python client [docs](${py_docs()}) if you don't already have it installed. 

\`\`\`bash
pip install gradio_client
\`\`\`

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. ${space_id()
				? "If this is a private Space, you may need to pass your Hugging Face token as well. [Read more](" + py_docs() + spaces_docs_suffix() + ")."
				: ""}

${dependencies().filter((d) => d.api_visibility === "public" && info().named_endpoints["/" + d.api_name]).map((d) => `### API Name: /${d.api_name}
${info()?.named_endpoints["/" + d.api_name]?.description
				? "Description: " + info()?.named_endpoints["/" + d.api_name]?.description
				: ""}

\`\`\`python
${markdown_code_snippets()[d.api_name]?.python}
\`\`\`

Accepts ${info()?.named_endpoints["/" + d.api_name]?.parameters?.length} parameter${info()?.named_endpoints["/" + d.api_name]?.parameters?.length != 1 ? "s" : ""}:

${info()?.named_endpoints["/" + d.api_name]?.parameters?.map((p) => {
				const required = !p.parameter_has_default;

				const defaultValue = !required
					? `Default: ${represent_value(p.parameter_default, p.python_type.type, "py")}`
					: "Required";

				const type = `${p.python_type.type}${p.parameter_has_default && p.parameter_default === null ? " | None" : ""}`;

				return `${p.parameter_name || `[${js_info().named_endpoints["/" + d.api_name]?.parameters.findIndex((p) => p.parameter_name === p.parameter_name)}]`}:\n- Type: ${type}\n- ${defaultValue}\n- The input value that is provided in the ${p.label} ${p.component} component. ${p.python_type.description}`;
			}).join("\n\n")}

Returns ${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1
				? `tuple of ${info()?.named_endpoints["/" + d.api_name]?.returns?.length} elements`
				: "1 element"}:

${info()?.named_endpoints["/" + d.api_name]?.returns?.map((r, i) => {
				const type = r.python_type.type;

				return `${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1 ? `[${i}]: ` : ""}- Type: ${type}\n- The output value that appears in the "${r.label}" ${r.component} component.`;
			}).join("\n\n")}
`).join("\n\n\n")}
`);
		}
	);

	legacy_pre_effect(
		() => (
			deep_read_state(space_id()),
			deep_read_state(root()),
			deep_read_state(api_count()),
			deep_read_state(js_docs()),
			deep_read_state(spaces_docs_suffix()),
			deep_read_state(dependencies()),
			deep_read_state(info()),
			deep_read_state(markdown_code_snippets()),
			deep_read_state(js_info())
		),
		() => {
			mutate(markdown_content, get(markdown_content).javascript = `
# JavaScript API documentation for ${space_id() || root()}
API Endpoints: ${api_count()}

1. Install the JavaScript client [docs](${js_docs()}) if you don't already have it installed. 

\`\`\`bash
npm i -D @gradio/client
\`\`\`

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. ${space_id()
				? "If this is a private Space, you may need to pass your Hugging Face token as well. [Read more](" + js_docs() + spaces_docs_suffix() + ")."
				: ""}

${dependencies().filter((d) => d.api_visibility === "public" && info().named_endpoints["/" + d.api_name]).map((d) => `### API Name: /${d.api_name}
${info()?.named_endpoints["/" + d.api_name]?.description
				? "Description: " + info()?.named_endpoints["/" + d.api_name]?.description
				: ""}

\`\`\`javascript
${markdown_code_snippets()[d.api_name]?.javascript}
\`\`\`

Accepts ${info()?.named_endpoints["/" + d.api_name]?.parameters?.length} parameter${info()?.named_endpoints["/" + d.api_name]?.parameters?.length != 1 ? "s" : ""}:

${info()?.named_endpoints["/" + d.api_name]?.parameters?.map((p) => {
				const required = !p.parameter_has_default;

				const defaultValue = !required
					? `Default: ${represent_value(p.parameter_default, p.python_type.type, "py")}`
					: "Required";

				const type = `${js_info().named_endpoints["/" + d.api_name]?.parameters.find((_p) => _p.parameter_name === p.parameter_name)?.type || "any"}`;

				return `${p.parameter_name || `[${js_info().named_endpoints["/" + d.api_name]?.parameters.findIndex((_p) => _p.parameter_name === p.parameter_name)}]`}:\n- Type: ${type}\n- ${defaultValue}\n- The input value that is provided in the ${p.label} ${p.component} component. ${p.python_type.description}`;
			}).join("\n\n")}

Returns ${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1
				? `list of ${info()?.named_endpoints["/" + d.api_name]?.returns?.length} elements`
				: "1 element"}:

${info()?.named_endpoints["/" + d.api_name]?.returns?.map((r, i) => {
				const type = js_info().named_endpoints["/" + d.api_name]?.returns[i]?.type;

				return `${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1 ? `[${i}]: ` : ""}- Type: ${type}\n- The output value that appears in the "${r.label}" ${r.component} component.`;
			}).join("\n\n")}`).join("\n\n\n")}
`);
		}
	);

	legacy_pre_effect(
		() => (
			deep_read_state(space_id()),
			deep_read_state(root()),
			deep_read_state(api_count()),
			deep_read_state(bash_docs()),
			deep_read_state(dependencies()),
			deep_read_state(info()),
			deep_read_state(markdown_code_snippets()),
			deep_read_state(js_info())
		),
		() => {
			mutate(markdown_content, get(markdown_content).bash = `
# Bash API documentation for ${space_id() || root()}
API Endpoints: ${api_count()}

1. Confirm that you have cURL installed on your system.

\`\`\`bash
curl --version
\`\`\`

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data.

Making a prediction and getting a result requires 2 requests: a POST and a GET request. The POST request returns an EVENT_ID, which is used in the second GET request to fetch the results. In these snippets, we've used awk and read to parse the results, combining these two requests into one command for ease of use. See [curl docs](${bash_docs()}).

${dependencies().filter((d) => d.api_visibility === "public" && info().named_endpoints["/" + d.api_name]).map((d) => `### API Name: /${d.api_name}
${info()?.named_endpoints["/" + d.api_name]?.description
				? "Description: " + info()?.named_endpoints["/" + d.api_name]?.description
				: ""}

\`\`\`bash
${markdown_code_snippets()[d.api_name]?.bash}
\`\`\`

Accepts ${info()?.named_endpoints["/" + d.api_name]?.parameters?.length} parameter${info()?.named_endpoints["/" + d.api_name]?.parameters?.length != 1 ? "s" : ""}:

${info()?.named_endpoints["/" + d.api_name]?.parameters?.map((p) => {
				const defaultValue = "Required";
				const type = `${js_info().named_endpoints["/" + d.api_name]?.parameters.find((_p) => _p.parameter_name === p.parameter_name)?.type || "any"}`;

				return `${`[${js_info().named_endpoints["/" + d.api_name]?.parameters.findIndex((_p) => _p.parameter_name === p.parameter_name)}]`}:\n- Type: ${type}\n- ${defaultValue}\n- The input value that is provided in the ${p.label} ${p.component} component. ${p.python_type.description}`;
			}).join("\n\n")}

Returns ${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1
				? `list of ${info()?.named_endpoints["/" + d.api_name]?.returns?.length} elements`
				: "1 element"}:

${info()?.named_endpoints["/" + d.api_name]?.returns?.map((r, i) => {
				const type = js_info().named_endpoints["/" + d.api_name]?.returns[i]?.type;

				return `${info()?.named_endpoints["/" + d.api_name]?.returns?.length > 1 ? `[${i}]: ` : ""}- Type: ${type}\n- The output value that appears in the "${r.label}" ${r.component} component.`;
			}).join("\n\n")}
`).join("\n\n\n")}
`);
		}
	);

	legacy_pre_effect(
		() => (
			deep_read_state(space_id()),
			deep_read_state(root()),
			deep_read_state(tools()),
			deep_read_state(mcp_server_active()),
			deep_read_state(mcp_server_url_streamable()),
			deep_read_state(config_snippets()),
			deep_read_state(mcp_docs())
		),
		() => {
			mutate(markdown_content, get(markdown_content).mcp = `
# MCP documentation for ${space_id() || root()}
MCP Tools: ${tools().length}

${!mcp_server_active()
				? `This Gradio app can also serve as an MCP server, with an MCP tool corresponding to each API endpoint. 
To enable this, launch this Gradio app with \`.launch(mcp_server=True)\` or set the \`.launch(mcp_server=True)\`or set the \`GRADIO_MCP_SERVER\` env variable to \`"True"\`.`
				: `

This page documents two transports: Streamable HTTP and STDIO.

### Streamable HTTP

MCP Server URL (Streamable HTTP): ${mcp_server_url_streamable()}

${tools().length} available MCP tools, resources, and prompts: 

${tools().map((tool) => `### ${tool.name}
Type: ${tool.meta.mcp_type}
Description: ${tool.description
					? tool.description
					: "No description provided in function docstring"}
Parameters: ${Object.keys(tool.parameters).length}
${Object.keys(tool.parameters).map((parameter) => {
					return `- ${parameter} (${tool.parameters[parameter].type}): ${tool.parameters[parameter].description
						? tool.parameters[parameter].description
						: "No description provided in function docstring"}`;
				}).join("\n")}
`).join("\n\n")}

Streamable HTTP Transport: To add this MCP to clients that support Streamable HTTP, simply add the following configuration to your MCP config.

\`\`\`json
${config_snippets().streamable_http}
\`\`\`

The \`upload_files_to_gradio\` tool uploads files from your local \`UPLOAD_DIRECTORY\` (or any of its subdirectories) to the Gradio app. 
This is needed because MCP servers require files to be provided as URLs. You can omit this tool if you prefer to upload files manually. This tool requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

### STDIO Transport


${tools().length} available MCP tools, resources, and prompts: 

${tools().map((tool) => `### ${tool.name}
Type: ${tool.meta.mcp_type}
Description: ${tool.description
					? tool.description
					: "No description provided in function docstring"}
Parameters: ${Object.keys(tool.parameters).length}
${Object.keys(tool.parameters).map((parameter) => {
					return `- ${parameter} (${tool.parameters[parameter].type}): ${tool.parameters[parameter].description
						? tool.parameters[parameter].description
						: "No description provided in function docstring"}`;
				}).join("\n")}
`).join("\n\n")}

STDIO Transport: For clients that only support stdio (e.g. Claude Desktop), first [install Node.js](https://nodejs.org/en/download/). Then, you can use the following command:

\`\`\`json
${config_snippets().stdio}
\`\`\`

The \`upload_files_to_gradio\` tool uploads files from your local \`UPLOAD_DIRECTORY\` (or any of its subdirectories) to the Gradio app. 
This is needed because MCP servers require files to be provided as URLs. You can omit this tool if you prefer to upload files manually. This tool requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

Read more about the MCP in the [Gradio docs](${mcp_docs()}).
`}

`);
		}
	);

	legacy_pre_effect(() => (deep_read_state(current_language())), () => {
		current_language();
	});

	legacy_pre_effect(() => (deep_read_state(current_language())), () => {
		set(current_language_label, current_language() === "python"
			? "Python"
			: current_language() === "javascript"
				? "JavaScript"
				: current_language() === "bash" ? "Bash" : "MCP");
	});

	legacy_pre_effect(() => (get(current_language_label)), () => {
		set(label, `Copy ${get(current_language_label)} Docs as Markdown for LLMs`);
	});

	legacy_pre_effect(() => (get(copied)), () => {
		get(copied);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get current_language() {
			return current_language();
		},

		set current_language($$value) {
			current_language($$value);
			flushSync();
		},

		get space_id() {
			return space_id();
		},

		set space_id($$value) {
			space_id($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get api_count() {
			return api_count();
		},

		set api_count($$value) {
			api_count($$value);
			flushSync();
		},

		get tools() {
			return tools();
		},

		set tools($$value) {
			tools($$value);
			flushSync();
		},

		get py_docs() {
			return py_docs();
		},

		set py_docs($$value) {
			py_docs($$value);
			flushSync();
		},

		get js_docs() {
			return js_docs();
		},

		set js_docs($$value) {
			js_docs($$value);
			flushSync();
		},

		get bash_docs() {
			return bash_docs();
		},

		set bash_docs($$value) {
			bash_docs($$value);
			flushSync();
		},

		get mcp_docs() {
			return mcp_docs();
		},

		set mcp_docs($$value) {
			mcp_docs($$value);
			flushSync();
		},

		get spaces_docs_suffix() {
			return spaces_docs_suffix();
		},

		set spaces_docs_suffix($$value) {
			spaces_docs_suffix($$value);
			flushSync();
		},

		get mcp_server_active() {
			return mcp_server_active();
		},

		set mcp_server_active($$value) {
			mcp_server_active($$value);
			flushSync();
		},

		get mcp_server_url_streamable() {
			return mcp_server_url_streamable();
		},

		set mcp_server_url_streamable($$value) {
			mcp_server_url_streamable($$value);
			flushSync();
		},

		get config_snippets() {
			return config_snippets();
		},

		set config_snippets($$value) {
			config_snippets($$value);
			flushSync();
		},

		get markdown_code_snippets() {
			return markdown_code_snippets();
		},

		set markdown_code_snippets($$value) {
			markdown_code_snippets($$value);
			flushSync();
		},

		get dependencies() {
			return dependencies();
		},

		set dependencies($$value) {
			dependencies($$value);
			flushSync();
		},

		get info() {
			return info();
		},

		set info($$value) {
			info($$value);
			flushSync();
		},

		get js_info() {
			return js_info();
		},

		set js_info($$value) {
			js_info($$value);
			flushSync();
		}
	};

	init();

	var div = root_1();

	event('mousedown', $window, handleWindowPointer);
	event('keydown', $window, handleWindowKeydown);
	event('resize', $window, handleWindowResize);
	event('scroll', $window, handleWindowScroll);

	var div_1 = child(div);
	var button = child(div_1);
	var span = child(button);
	var node = child(span);

	{
		var consequent = ($$anchor) => {
			IconCheck($$anchor, {});
		};

		var alternate = ($$anchor) => {
			IconCopy($$anchor, {});
		};

		if_block(node, ($$render) => {
			if (get(copied)) $$render(consequent); else $$render(alternate, false);
		});
	}

	reset(span);

	var span_1 = sibling(span, 2);
	var text = child(span_1, true);

	reset(span_1);
	reset(button);

	var button_1 = sibling(button, 2);
	var node_1 = child(button_1);

	{
		let $0 = derived_safe_equal(() => `caret-icon ${get(open) ? "rotate-180" : "rotate-0"}`);

		IconCaret(node_1, {
			get classNames() {
				return get($0);
			}
		});
	}

	reset(button_1);
	reset(div_1);
	bind_this(div_1, ($$value) => set(triggerEl, $$value), () => get(triggerEl));

	var node_2 = sibling(div_1, 2);

	{
		var consequent_1 = ($$anchor) => {
			var fragment_2 = root_4$1();
			var div_2 = first_child(fragment_2);
			var div_3 = sibling(div_2, 2);
			var button_2 = child(div_3);
			var div_4 = child(button_2);
			var node_3 = child(div_4);

			IconCopy(node_3, { classNames: 'menu-icon' });
			reset(div_4);

			var div_5 = sibling(div_4, 2);
			var div_6 = sibling(child(div_5), 2);
			var text_1 = child(div_6, true);

			reset(div_6);
			reset(div_5);
			reset(button_2);

			var button_3 = sibling(button_2, 2);
			var div_7 = child(button_3);
			var node_4 = child(div_7);

			IconHuggingChat(node_4, { classNames: 'menu-icon' });
			reset(div_7);

			var div_8 = sibling(div_7, 2);
			var div_9 = child(div_8);
			var node_5 = sibling(child(div_9));

			IconArrowUpRight(node_5, { classNames: 'menu-icon-arrow' });
			reset(div_9);

			var div_10 = sibling(div_9, 2);
			var text_2 = child(div_10);

			reset(div_10);
			reset(div_8);
			reset(button_3);
			reset(div_3);
			bind_this(div_3, ($$value) => set(menuEl, $$value), () => get(menuEl));

			template_effect(() => {
				set_style(div_3, get(menuStyle));
				set_text(text_1, get(label));
				set_text(text_2, `Ask Questions About The ${get(current_language_label) ?? ''} Docs`);
			});

			event('click', div_2, closeMenu);

			event('click', button_2, () => {
				copyMarkdown(current_language());
				closeMenu();
			});

			event('click', button_3, () => {
				openHuggingChat();
				closeMenu();
			});

			append($$anchor, fragment_2);
		};

		if_block(node_2, ($$render) => {
			if (get(open)) $$render(consequent_1);
		});
	}

	reset(div);

	template_effect(() => {
		set_text(text, get(copied)
			? `Copied ${get(current_language_label)} Docs!`
			: "Copy Page");

		set_attribute(button_1, 'aria-expanded', get(open));
		set_attribute(button_1, 'aria-label', get(open) ? "Close copy menu" : "Open copy menu");
	});

	event('click', button, () => copyMarkdown(current_language()));
	event('click', button_1, toggleMenu);
	append($$anchor, div);

	return pop($$exports);
}

var root_4 = from_html(`<li><img alt="" class="svelte-1ujc8oz"/> </li>`);

var root_5 = from_html(
	`<div><p id="num-recorded-api-calls" style="font-size: var(--text-lg); font-weight:bold; margin: 10px 0px;">🪄 Recorded API Calls <span class="api-count svelte-1ujc8oz"> </span></p> <p> </p> <!> <p>Note: Some API calls only affect the UI, so when using the
							clients, the desired result may be achieved with only a subset of
							the recorded calls.</p></div> <p style="font-size: var(--text-lg); font-weight:bold; margin: 30px 0px 10px;">API Documentation</p>`,
	1
);

var root_8 = from_html(`1. Install the <span style="text-transform:capitalize"> </span> client (<a target="_blank" class="svelte-1ujc8oz">docs</a>) if you don't already have it installed.`, 1);
var root_7 = from_html(`<p class="padded svelte-1ujc8oz"><!></p>`);

var root_12 = from_html(
	`If this is a private Space, you may need to pass
								your Hugging Face token as well (<a class="underline svelte-1ujc8oz" target="_blank">read more</a>).`,
	1
);

var root_13 = from_html(`<div class="loading-dot svelte-1ujc8oz"></div> <p class="self-baseline svelte-1ujc8oz">API Recorder</p>`, 1);

var root_14 = from_html(
	`<br/>&nbsp;<br/>Making a
								prediction and getting a result requires <strong>2 requests</strong>: a <code class="svelte-1ujc8oz">POST</code> and a <code class="svelte-1ujc8oz">GET</code> request. The <code class="svelte-1ujc8oz">POST</code> request
								returns an <code class="svelte-1ujc8oz">EVENT_ID</code>, which is used in the second <code class="svelte-1ujc8oz">GET</code> request to fetch the results. In these
								snippets, we've used <code class="svelte-1ujc8oz">awk</code> and <code class="svelte-1ujc8oz">read</code> to
								parse the results, combining these two requests into one command
								for ease of use. <!> See <a target="_blank" class="svelte-1ujc8oz">curl docs</a>.`,
	1
);

var root_11 = from_html(
	`<!> <p class="padded svelte-1ujc8oz">2. Find the API endpoint below corresponding to your desired
							function in the app. Copy the code snippet, replacing the
							placeholder values with your own input data. <!> Or use the <!> to automatically generate your API requests. <!></p>`,
	1
);

var root_6 = from_html(`<!> <div><!></div> <div><!></div> <!>`, 1);
var root_17 = from_html(`<div class="endpoint-container svelte-1ujc8oz"><!> <!> <!></div>`);
var root_3 = from_html(`<div class="banner-wrap svelte-1ujc8oz"><!></div> <div class="docs-wrap svelte-1ujc8oz"><div class="client-doc svelte-1ujc8oz" style="display: flex; align-items: center; justify-content: space-between;"><p style="font-size: var(--text-lg);">Choose one of the following ways to interact with the API.</p> <!></div> <div class="endpoint svelte-1ujc8oz"><div class="snippets svelte-1ujc8oz"></div> <!> <div></div></div></div>`, 1);

function ApiDocs($$anchor, $$props) {
	push($$props, false);

	const sorted_dependencies = mutable_source();
	const langs = mutable_source();
	const selected_tools_array = mutable_source();
	const selected_tools_without_prefix = mutable_source();
	const mcp_server_url_streamable = mutable_source();

	/* eslint-disable */
	let dependencies = prop($$props, 'dependencies', 12);

	let root = prop($$props, 'root', 12);
	let app = prop($$props, 'app', 12);
	let space_id = prop($$props, 'space_id', 12);
	let root_node = prop($$props, 'root_node', 12);
	let username = prop($$props, 'username', 12);
	let last_api_call = prop($$props, 'last_api_call', 12, null);
	const js_docs = "https://www.gradio.app/guides/getting-started-with-the-js-client";
	const py_docs = "https://www.gradio.app/guides/getting-started-with-the-python-client";
	const bash_docs = "https://www.gradio.app/guides/querying-gradio-apps-with-curl";
	const spaces_docs_suffix = "#connecting-to-a-hugging-face-space";
	const mcp_docs = "https://www.gradio.app/guides/building-mcp-server-with-gradio";
	let api_count = dependencies().filter((dependency) => dependency.api_visibility === "public").length;

	if (root() === "") {
		root(location.protocol + "//" + location.host + location.pathname);
	}

	if (!root().endsWith("/")) {
		root(root() + "/");
	}

	let api_calls = prop($$props, 'api_calls', 28, () => []);
	let current_language = mutable_source("python");

	function set_query_param(key, value) {
		const url = new URL(window.location.href);

		url.searchParams.set(key, value);
		history.replaceState(null, "", url.toString());
	}

	function get_query_param(key) {
		const url = new URL(window.location.href);

		return url.searchParams.get(key);
	}

	function is_valid_language(lang) {
		return ["python", "javascript", "bash", "skill", "mcp"].includes(lang ?? "");
	}

	let is_running = false;
	let mcp_server_active = mutable_source(false);

	async function get_info() {
		let response = await fetch(root().replace(/\/$/, "") + app().api_prefix + "/info");
		let data = await response.json();

		return data;
	}

	async function get_js_info() {
		let js_api_info = await app().view_api();

		return js_api_info;
	}

	let info = mutable_source();
	let js_info = mutable_source();
	let analytics = mutable_source();

	get_info().then((data) => {
		set(info, data);
	});

	get_js_info().then((js_api_info) => {
		set(js_info, js_api_info);
	});

	async function get_summary() {
		let response = await fetch(root().replace(/\/$/, "") + "/monitoring/summary");
		let data = await response.json();

		return data;
	}

	get_summary().then((summary) => {
		set(analytics, summary.functions);
	});

	const dispatch = createEventDispatcher();
	let tools = mutable_source([]);
	let headers = [];
	let mcp_json_stdio = mutable_source();
	let file_data_present = mutable_source(false);
	let selected_tools = mutable_source(new Set());
	let tool_prefix = space_id() ? space_id().split("/").pop() + "_" : "";

	function remove_tool_prefix(toolName) {
		if (tool_prefix && toolName.startsWith(tool_prefix)) {
			return toolName.slice(tool_prefix.length);
		}

		return toolName;
	}

	const upload_file_mcp_server = {
		command: "uvx",
		args: [
			"--from",
			"gradio[mcp]",
			"gradio",
			"upload-mcp",
			root(),
			"<UPLOAD_DIRECTORY>"
		]
	};

	async function fetch_mcp_tools() {
		try {
			let schema_url = `${root()}gradio_api/mcp/schema`;
			const response = await fetch(schema_url);
			const schema = await response.json();

			set(file_data_present, schema.map((tool) => tool.meta?.file_data_present).some((present) => present));

			set(tools, schema.map((tool) => ({
				name: tool.name,
				description: tool.description || "",
				parameters: tool.inputSchema?.properties || {},
				meta: tool.meta,
				expanded: false,
				endpoint_name: tool.endpoint_name
			})));

			set(selected_tools, new Set(get(tools).map((tool) => tool.name)));
			headers = schema.map((tool) => tool.meta?.headers || []).flat();

			if (headers.length > 0) {
				set(mcp_json_stdio, {
					mcpServers: {
						gradio: {
							command: "npx",
							args: [
								"mcp-remote",
								get(mcp_server_url_streamable),
								"--transport",
								"streamable-http",
								...headers.map((header) => ["--header", `${header}: <YOUR_HEADER_VALUE>`]).flat()
							]
						}
					}
				});
			} else {
				set(mcp_json_stdio, {
					mcpServers: {
						gradio: {
							command: "npx",
							args: [
								"mcp-remote",
								get(mcp_server_url_streamable),
								"--transport",
								"streamable-http"
							]
						}
					}
				});

				if (get(file_data_present)) {
					mutate(mcp_json_stdio, get(mcp_json_stdio).mcpServers.upload_files_to_gradio = upload_file_mcp_server);
				}
			}
		} catch(error) {
			console.error("Failed to fetch MCP tools:", error);
			set(tools, []);
		}
	}

	let markdown_code_snippets = mutable_source({});
	let config_snippets = mutable_source({});

	onMount(() => {
		const controller = new AbortController();
		const signal = controller.signal;

		document.body.style.overflow = "hidden";

		if ("parentIFrame" in window) {
			window.parentIFrame?.scrollTo(0, 0);
		}

		const lang_param = get_query_param("lang");

		if (is_valid_language(lang_param)) {
			set(current_language, lang_param);
		}

		const mcp_schema_url = `${root()}gradio_api/mcp/schema`;

		fetch(mcp_schema_url, { signal }).then((response) => {
			set(mcp_server_active, response.ok);

			if (get(mcp_server_active)) {
				fetch_mcp_tools();

				if (!is_valid_language(lang_param)) {
					set(current_language, "mcp");
				}
			} else {
				if (!is_valid_language(lang_param)) {
					set(current_language, "python");
				}
			}

			controller.abort();
		}).catch(() => {
			set(mcp_server_active, false);
		});

		return () => {
			document.body.style.overflow = "auto";
		};
	});

	legacy_pre_effect(
		() => (
			deep_read_state(dependencies()),
			get(info),
			deep_read_state(last_api_call())
		),
		() => {
			set(sorted_dependencies, (() => {
				const valid = dependencies().filter((dep) => dep.api_visibility === "public" && get(info)?.named_endpoints?.["/" + dep.api_name]);

				if (get(info) && last_api_call()) {
					const mostRecent = valid.find((dep) => dep.id === last_api_call().fn_index);
					const others = valid.filter((dep) => dep.id !== last_api_call().fn_index);

					return mostRecent ? [mostRecent, ...others] : valid;
				}

				return valid;
			})());
		}
	);

	legacy_pre_effect(
		() => (
			deep_read_state(space_id()),
			mcp
		),
		() => {
			set(langs, [
				["python", "Python", python],
				["javascript", "JavaScript", javascript],
				["bash", "cURL", bash],
				...space_id() ? [["skill", "Skill", skill]] : [],
				["mcp", "MCP", mcp]
			]);
		}
	);

	legacy_pre_effect(() => (get(selected_tools)), () => {
		set(selected_tools_array, Array.from(get(selected_tools)));
	});

	legacy_pre_effect(() => (get(selected_tools_array)), () => {
		set(selected_tools_without_prefix, get(selected_tools_array).map(remove_tool_prefix));
	});

	legacy_pre_effect(
		() => (
			get(selected_tools_array),
			get(tools),
			deep_read_state(root()),
			get(selected_tools_without_prefix)
		),
		() => {
			set(mcp_server_url_streamable, get(selected_tools_array).length > 0 && get(selected_tools_array).length < get(tools).length
				? `${root()}gradio_api/mcp/?tools=${get(selected_tools_without_prefix).join(",")}`
				: `${root()}gradio_api/mcp/`);
		}
	);

	legacy_pre_effect(() => (get(markdown_code_snippets)), () => {
		get(markdown_code_snippets);
	});

	legacy_pre_effect(() => (get(config_snippets)), () => {
		get(config_snippets);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get dependencies() {
			return dependencies();
		},

		set dependencies($$value) {
			dependencies($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get app() {
			return app();
		},

		set app($$value) {
			app($$value);
			flushSync();
		},

		get space_id() {
			return space_id();
		},

		set space_id($$value) {
			space_id($$value);
			flushSync();
		},

		get root_node() {
			return root_node();
		},

		set root_node($$value) {
			root_node($$value);
			flushSync();
		},

		get username() {
			return username();
		},

		set username($$value) {
			username($$value);
			flushSync();
		},

		get last_api_call() {
			return last_api_call();
		},

		set last_api_call($$value) {
			last_api_call($$value);
			flushSync();
		},

		get api_calls() {
			return api_calls();
		},

		set api_calls($$value) {
			api_calls($$value);
			flushSync();
		}
	};

	init();

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_10 = ($$anchor) => {
			var fragment_1 = comment();
			var node_1 = first_child(fragment_1);

			{
				var consequent_9 = ($$anchor) => {
					var fragment_2 = root_3();
					var div = first_child(fragment_2);
					var node_2 = child(div);

					{
						let $0 = derived_safe_equal(() => space_id() || root());

						ApiBanner(node_2, {
							get root() {
								return get($0);
							},

							get api_count() {
								return api_count;
							},

							get current_language() {
								return get(current_language);
							},

							$$events: {
								close($$arg) {
									bubble_event.call(this, $$props, $$arg);
								}
							}
						});
					}

					reset(div);

					var div_1 = sibling(div, 2);
					var div_2 = child(div_1);
					var node_3 = sibling(child(div_2), 2);

					CopyMarkdown(node_3, {
						get current_language() {
							return get(current_language);
						},

						get space_id() {
							return space_id();
						},

						get root() {
							return root();
						},

						get api_count() {
							return api_count;
						},

						get tools() {
							return get(tools);
						},
						py_docs,
						js_docs,
						bash_docs,
						mcp_docs,
						spaces_docs_suffix,
						get mcp_server_active() {
							return get(mcp_server_active);
						},

						get mcp_server_url_streamable() {
							return get(mcp_server_url_streamable);
						},

						get config_snippets() {
							return get(config_snippets);
						},

						get markdown_code_snippets() {
							return get(markdown_code_snippets);
						},

						get dependencies() {
							return dependencies();
						},

						get info() {
							return get(info);
						},

						get js_info() {
							return get(js_info);
						}
					});

					reset(div_2);

					var div_3 = sibling(div_2, 2);
					var div_4 = child(div_3);

					each(div_4, 5, () => get(langs), index, ($$anchor, $$item) => {
						var $$array = user_derived(() => to_array(get($$item), 3));
						let language = () => get($$array)[0];
						let display_name = () => get($$array)[1];
						let img = () => get($$array)[2];
						var li = root_4();
						var img_1 = child(li);
						var text = sibling(img_1);

						reset(li);

						template_effect(() => {
							set_class(
								li,
								1,
								`snippet
						${get(current_language) === language() ? 'current-lang' : 'inactive-lang'}`,
								'svelte-1ujc8oz'
							);

							set_attribute(img_1, 'src', img());
							set_text(text, ` ${display_name() ?? ''}`);
						});

						event('click', li, () => {
							set(current_language, language());
							set_query_param("lang", language());
						});

						append($$anchor, li);
					});

					reset(div_4);

					var node_4 = sibling(div_4, 2);

					{
						var consequent = ($$anchor) => {
							var fragment_3 = root_5();
							var div_5 = first_child(fragment_3);
							var p = child(div_5);
							var span = sibling(child(p));
							var text_1 = child(span);

							reset(span);
							reset(p);

							var p_1 = sibling(p, 2);
							var text_2 = child(p_1);

							reset(p_1);

							var node_5 = sibling(p_1, 2);

							{
								let $0 = derived_safe_equal(() => space_id() || root());

								RecordingSnippet(node_5, {
									get current_language() {
										return get(current_language);
									},

									get api_calls() {
										return api_calls();
									},

									get dependencies() {
										return dependencies();
									},

									get root() {
										return root();
									},

									get api_prefix() {
										return (deep_read_state(app()), untrack(() => app().api_prefix));
									},

									get short_root() {
										return get($0);
									},

									get username() {
										return username();
									}
								});
							}

							next(2);
							reset(div_5);
							next(2);

							template_effect(() => {
								set_text(text_1, `[${(
									deep_read_state(api_calls()),
									untrack(() => api_calls().length)
								) ?? ''}]`);

								set_text(text_2, `Here is the code snippet to replay the most recently recorded API
							calls using the ${get(current_language) ?? ''}
							client.`);
							});

							append($$anchor, fragment_3);
						};

						var alternate_1 = ($$anchor) => {
							var fragment_4 = root_6();
							var node_6 = first_child(fragment_4);

							{
								var consequent_3 = ($$anchor) => {
									var p_2 = root_7();
									var node_7 = child(p_2);

									{
										var consequent_1 = ($$anchor) => {
											var fragment_5 = root_8();
											var span_1 = sibling(first_child(fragment_5));
											var text_3 = child(span_1, true);

											reset(span_1);

											var a = sibling(span_1, 2);

											next();

											template_effect(() => {
												set_text(text_3, get(current_language));
												set_attribute(a, 'href', get(current_language) == "python" ? py_docs : js_docs);
											});

											append($$anchor, fragment_5);
										};

										var alternate = ($$anchor) => {
											var fragment_6 = comment();
											var node_8 = first_child(fragment_6);

											{
												var consequent_2 = ($$anchor) => {
													var text_4 = text('1. Confirm that you have cURL installed on your system.');

													append($$anchor, text_4);
												};

												if_block(
													node_8,
													($$render) => {
														if (get(current_language) == "bash") $$render(consequent_2);
													},
													true
												);
											}

											append($$anchor, fragment_6);
										};

										if_block(node_7, ($$render) => {
											if (get(current_language) == "python" || get(current_language) == "javascript") $$render(consequent_1); else $$render(alternate, false);
										});
									}

									reset(p_2);
									append($$anchor, p_2);
								};

								if_block(node_6, ($$render) => {
									if (get(current_language) !== "skill") $$render(consequent_3);
								});
							}

							var div_6 = sibling(node_6, 2);
							let classes;
							var node_9 = child(div_6);

							SkillSnippet(node_9, {
								get space_id() {
									return space_id();
								}
							});

							reset(div_6);

							var div_7 = sibling(div_6, 2);
							let classes_1;
							var node_10 = child(div_7);

							{
								let $0 = derived_safe_equal(() => (
									get(tools),
									get(selected_tools),
									untrack(() => get(tools).filter((tool) => get(selected_tools).has(tool.name)))
								));

								MCPSnippet(node_10, {
									get mcp_server_active() {
										return get(mcp_server_active);
									},

									get mcp_server_url_streamable() {
										return get(mcp_server_url_streamable);
									},

									get tools() {
										return get($0);
									},

									get all_tools() {
										return get(tools);
									},

									get mcp_json_stdio() {
										return get(mcp_json_stdio);
									},

									get file_data_present() {
										return get(file_data_present);
									},
									mcp_docs,
									get analytics() {
										return get(analytics);
									},

									get root() {
										return root();
									},

									get selected_tools() {
										return get(selected_tools);
									},

									set selected_tools($$value) {
										set(selected_tools, $$value);
									},

									get config_snippets() {
										return get(config_snippets);
									},

									set config_snippets($$value) {
										set(config_snippets, $$value);
									},
									$$legacy: true
								});
							}

							reset(div_7);

							var node_11 = sibling(div_7, 2);

							{
								var consequent_7 = ($$anchor) => {
									var fragment_7 = root_11();
									var node_12 = first_child(fragment_7);

									InstallSnippet(node_12, {
										get current_language() {
											return get(current_language);
										}
									});

									var p_3 = sibling(node_12, 2);
									var node_13 = sibling(child(p_3));

									{
										var consequent_4 = ($$anchor) => {
											var fragment_8 = root_12();
											var a_1 = sibling(first_child(fragment_8));

											next();

											template_effect(() => set_attribute(a_1, 'href', get(current_language) == "python"
												? py_docs + spaces_docs_suffix
												: get(current_language) == "javascript" ? js_docs + spaces_docs_suffix : bash_docs));

											append($$anchor, fragment_8);
										};

										if_block(node_13, ($$render) => {
											if (space_id()) $$render(consequent_4);
										});
									}

									var node_14 = sibling(node_13, 2);

									Button(node_14, {
										size: 'sm',
										variant: 'secondary',
										$$events: {
											click: () => dispatch("close", { api_recorder_visible: true })
										},

										children: ($$anchor, $$slotProps) => {
											var fragment_9 = root_13();

											next(2);
											append($$anchor, fragment_9);
										},
										$$slots: { default: true }
									});

									var node_15 = sibling(node_14, 2);

									{
										var consequent_6 = ($$anchor) => {
											var fragment_10 = root_14();
											var node_16 = sibling(first_child(fragment_10), 20);

											{
												var consequent_5 = ($$anchor) => {
													var text_5 = text('Note: connecting to an authenticated app requires an\n									additional request.');

													append($$anchor, text_5);
												};

												if_block(node_16, ($$render) => {
													if (username() !== null) $$render(consequent_5);
												});
											}

											var a_2 = sibling(node_16, 2);

											set_attribute(a_2, 'href', bash_docs);
											next();
											append($$anchor, fragment_10);
										};

										if_block(node_15, ($$render) => {
											if (get(current_language) == "bash") $$render(consequent_6);
										});
									}

									reset(p_3);
									append($$anchor, fragment_7);
								};

								if_block(node_11, ($$render) => {
									if (get(current_language) !== "mcp" && get(current_language) !== "skill") $$render(consequent_7);
								});
							}

							template_effect(() => {
								classes = set_class(div_6, 1, 'svelte-1ujc8oz', null, classes, { hidden: get(current_language) !== "skill" });
								classes_1 = set_class(div_7, 1, 'svelte-1ujc8oz', null, classes_1, { hidden: get(current_language) !== "mcp" });
							});

							append($$anchor, fragment_4);
						};

						if_block(node_4, ($$render) => {
							if ((
								deep_read_state(api_calls()),
								untrack(() => api_calls().length)
							)) $$render(consequent); else $$render(alternate_1, false);
						});
					}

					var div_8 = sibling(node_4, 2);
					let classes_2;

					each(div_8, 5, () => get(sorted_dependencies), index, ($$anchor, dependency) => {
						var fragment_11 = comment();
						var node_17 = first_child(fragment_11);

						{
							var consequent_8 = ($$anchor) => {
								var div_9 = root_17();
								var node_18 = child(div_9);

								CodeSnippet(node_18, {
									get dependency() {
										return get(dependency);
									},

									get current_language() {
										return get(current_language);
									},

									get api_description() {
										return (
											get(info),
											get(dependency),
											untrack(() => get(info).named_endpoints["/" + get(dependency).api_name].description)
										);
									},

									get analytics() {
										return get(analytics);
									},

									get last_api_call() {
										return last_api_call();
									},

									get code_snippets() {
										return (
											get(info),
											get(dependency),
											untrack(() => get(info).named_endpoints["/" + get(dependency).api_name].code_snippets)
										);
									},

									get markdown_code_snippets() {
										return get(markdown_code_snippets);
									},

									set markdown_code_snippets($$value) {
										set(markdown_code_snippets, $$value);
									},
									$$legacy: true
								});

								var node_19 = sibling(node_18, 2);

								ParametersSnippet(node_19, {
									get endpoint_returns() {
										return (
											get(info),
											get(dependency),
											untrack(() => get(info).named_endpoints["/" + get(dependency).api_name].parameters)
										);
									},

									get js_returns() {
										return (
											get(js_info),
											get(dependency),
											untrack(() => get(js_info).named_endpoints["/" + get(dependency).api_name].parameters)
										);
									},
									is_running,
									get current_language() {
										return get(current_language);
									}
								});

								var node_20 = sibling(node_19, 2);

								ResponseSnippet(node_20, {
									get endpoint_returns() {
										return (
											get(info),
											get(dependency),
											untrack(() => get(info).named_endpoints["/" + get(dependency).api_name].returns)
										);
									},

									get js_returns() {
										return (
											get(js_info),
											get(dependency),
											untrack(() => get(js_info).named_endpoints["/" + get(dependency).api_name].returns)
										);
									},
									is_running,
									get current_language() {
										return get(current_language);
									}
								});

								reset(div_9);
								append($$anchor, div_9);
							};

							if_block(node_17, ($$render) => {
								if ((
									get(info),
									get(dependency),
									untrack(() => get(info).named_endpoints["/" + get(dependency).api_name])
								)) $$render(consequent_8);
							});
						}

						append($$anchor, fragment_11);
					});

					reset(div_8);
					reset(div_3);
					reset(div_1);

					template_effect(() => classes_2 = set_class(div_8, 1, 'svelte-1ujc8oz', null, classes_2, {
						hidden: get(current_language) === "mcp" || get(current_language) === "skill"
					}));

					append($$anchor, fragment_2);
				};

				var alternate_2 = ($$anchor) => {
					NoApi($$anchor, {
						get root() {
							return root();
						},

						$$events: {
							close($$arg) {
								bubble_event.call(this, $$props, $$arg);
							}
						}
					});
				};

				if_block(node_1, ($$render) => {
					if (api_count) $$render(consequent_9); else $$render(alternate_2, false);
				});
			}

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if (get(info) && get(analytics)) $$render(consequent_10);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

export { ApiDocs as default };
//# sourceMappingURL=ApiDocs-9hYmXNno.js.map
