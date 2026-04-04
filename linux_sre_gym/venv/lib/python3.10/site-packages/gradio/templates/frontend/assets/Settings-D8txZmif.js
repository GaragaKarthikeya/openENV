import { a as set_attribute, q as store_get, z as $format, m as setup_stores, p as prop, j as get, $ as $locale, i as if_block, H as language_choices, I as changeLocale, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, V as child, X as sibling, Y as reset, t as template_effect, a0 as set_text, Z as event, a as append, T as pop, W as from_html, aa as onMount, x as set, u as state, v as proxy, w as get$1, a8 as next } from './index-BvBk1Iap.js';
import { s as settings_logo } from './Blocks-Wq-vWI14.js';
import { C as Clear } from './clear-D8gK36Dy.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { b as Dropdown } from './Dropdown-DUCTuRQm.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
/* empty css                                               */
import { C as Checkbox } from './Checkbox-BwK0lQJ8.js';
import './html-CdUEtR5E.js';
import './size-Cu4PabCL.js';
import './state.svelte-CCPBiyik.js';
import './index-Lz7fsHFt.js';
import './Clear-Dgs5UhTO.js';
import './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';
import './index-BJk8VOjP.js';
import './index-DnoGeqVF.js';
import './snippet-Bl5YVoCg.js';
import './input-C317uaXR.js';
import './window-BGXjFJtG.js';
import './BlockTitle-7XQ-8ENg.js';
import './Info-BrkJhX6_.js';
import './DropdownArrow-Dw1NZKCM.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './index-C7e-J7CF.js';

var root_1$1 = from_html(`<h2 class="svelte-xaoy31"><img alt="" class="svelte-xaoy31"/> <div class="title svelte-xaoy31"> <div class="url svelte-xaoy31"> </div></div></h2> <button class="svelte-xaoy31"><!></button>`, 1);

function SettingsBanner($$anchor, $$props) {
	push($$props, true);

	const $_ = () => store_get($format, '$_', $$stores);
	const [$$stores, $$cleanup] = setup_stores();
	var fragment = root_1$1();
	var h2 = first_child(fragment);
	var img = child(h2);
	var div = sibling(img, 2);
	var text = child(div);
	var div_1 = sibling(text);
	var text_1 = child(div_1, true);

	reset(div_1);
	reset(div);
	reset(h2);

	var button = sibling(h2, 2);
	var node = child(button);

	Clear(node);
	reset(button);

	template_effect(
		($0) => {
			set_attribute(img, 'src', settings_logo);
			set_text(text, `${$0 ?? ''} `);
			set_text(text_1, $$props.root);
		},
		[() => $_()("common.settings")]
	);

	event('click', button, () => $$props.onclose?.());
	append($$anchor, fragment);
	pop();
	$$cleanup();
}

const record = "data:image/svg+xml,%3csvg%20viewBox='0%200%2020%2020'%20version='1.1'%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20fill='%23000000'%3e%3cg%20id='SVGRepo_bgCarrier'%20stroke-width='0'%3e%3c/g%3e%3cg%20id='SVGRepo_tracerCarrier'%20stroke-linecap='round'%20stroke-linejoin='round'%3e%3c/g%3e%3cg%20id='SVGRepo_iconCarrier'%3e%3ctitle%3erecord%20[%23982]%3c/title%3e%3cdesc%3eCreated%20with%20Sketch.%3c/desc%3e%3cdefs%3e%3c/defs%3e%3cg%20id='Page-1'%20stroke='none'%20stroke-width='1'%20fill='none'%20fill-rule='evenodd'%3e%3cg%20id='Dribbble-Light-Preview'%20transform='translate(-380.000000,%20-3839.000000)'%20fill='%23808080'%3e%3cg%20id='icons'%20transform='translate(56.000000,%20160.000000)'%3e%3cpath%20d='M338,3689%20C338,3691.209%20336.209,3693%20334,3693%20C331.791,3693%20330,3691.209%20330,3689%20C330,3686.791%20331.791,3685%20334,3685%20C336.209,3685%20338,3686.791%20338,3689%20M334,3697%20C329.589,3697%20326,3693.411%20326,3689%20C326,3684.589%20329.589,3681%20334,3681%20C338.411,3681%20342,3684.589%20342,3689%20C342,3693.411%20338.411,3697%20334,3697%20M334,3679%20C328.477,3679%20324,3683.477%20324,3689%20C324,3694.523%20328.477,3699%20334,3699%20C339.523,3699%20344,3694.523%20344,3689%20C344,3683.477%20339.523,3679%20334,3679'%20id='record-[%23982]'%3e%3c/path%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/svg%3e";

var root_2 = from_html(`<div class="banner-wrap svelte-1t60wb3"><h2 class="svelte-1t60wb3"> </h2> <ul class="padded theme-buttons svelte-1t60wb3"><li><button class="svelte-1t60wb3">☀︎ &nbsp;Light</button></li> <li><button class="svelte-1t60wb3">⏾ &nbsp; Dark</button></li> <li><button class="svelte-1t60wb3">🖥︎ &nbsp;System</button></li></ul></div>`);
var root_3 = from_html(`You can install this app as a Progressive Web App on your device. Visit <a target="_blank" class="svelte-1t60wb3"> </a> and click the install button in the URL address bar of your browser.`, 1);

var root_4 = from_html(
	`Progressive Web App is not enabled for this app. To enable it, start your
			Gradio app with <code>launch(pwa=True)</code>.`,
	1
);

var root_1 = from_html(
	`<div class="banner-wrap svelte-1t60wb3"><!></div> <!> <div class="banner-wrap svelte-1t60wb3"><h2 class="svelte-1t60wb3"> </h2> <p class="padded svelte-1t60wb3"><!></p></div> <div class="banner-wrap svelte-1t60wb3"><h2 class="svelte-1t60wb3"> </h2> <p class="padded svelte-1t60wb3"><!></p></div> <div class="banner-wrap svelte-1t60wb3"><h2 class="svelte-1t60wb3"> <span class="beta-tag svelte-1t60wb3">beta</span></h2> <p class="padded svelte-1t60wb3">Screen Studio allows you to record your screen and generates a video of your
		app with automatically adding zoom in and zoom out effects as well as
		trimming the video to remove the prediction time. <br/><br/> Start recording by clicking the <i>Start Recording</i> button below and then
		sharing the current browser tab of your Gradio demo. Use your app as you
		would normally to generate a prediction. <br/> Stop recording by clicking the <i>Stop Recording</i> button in the footer of
		the demo. <br/><br/> <!> <!></p> <button class="record-button svelte-1t60wb3"><img alt="Start Recording" class="svelte-1t60wb3"/> Start Recording</button></div>`,
	1
);

function Settings($$anchor, $$props) {
	push($$props, true);

	/* eslint-disable */
	let root = prop($$props, 'root', 7),
		allow_zoom = prop($$props, 'allow_zoom', 15),
		allow_video_trim = prop($$props, 'allow_video_trim', 15);

	if (root() === "") {
		root(location.protocol + "//" + location.host + location.pathname);
	}

	if (!root().endsWith("/")) {
		root(root() + "/");
	}

	function setTheme(theme) {
		const url = new URL(window.location.href);

		if (theme === "system") {
			url.searchParams.delete("__theme");
			set(current_theme, "system");
		} else {
			url.searchParams.set("__theme", theme);
			set(current_theme, theme, true);
		}

		url.searchParams.delete("view");
		window.location.href = url.toString();
	}

	onMount(() => {
		// document.body.style.overflow = "hidden";
		if ("parentIFrame" in window) {
			window.parentIFrame?.scrollTo(0, 0);
		}

		const url = new URL(window.location.href);
		const theme = url.searchParams.get("__theme");

		set(current_theme, theme || "system", true);

		return () => {
			document.body.style.overflow = "auto";
		};
	});

	let current_locale = state(proxy(get($locale) ?? "en"));
	let current_theme = state("system");

	function handleLanguageChange(value) {
		const new_locale = value;

		changeLocale(new_locale);
	}

	function handleZoomChange(e) {
		allow_zoom(e.detail);
	}

	function handleVideoTrimChange(e) {
		allow_video_trim(e.detail);
	}

	var fragment = root_1();
	var div = first_child(fragment);
	var node = child(div);

	SettingsBanner(node, {
		onclose: () => $$props.onclose(),
		get root() {
			return root();
		}
	});

	reset(div);

	var node_1 = sibling(div, 2);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_2();
			var h2 = child(div_1);
			var text = child(h2, true);

			reset(h2);

			var ul = sibling(h2, 2);
			var li = child(ul);
			var button = child(li);

			reset(li);

			var li_1 = sibling(li, 2);
			var button_1 = child(li_1);

			reset(li_1);

			var li_2 = sibling(li_1, 2);
			var button_2 = child(li_2);

			reset(li_2);
			reset(ul);
			reset(div_1);

			template_effect(
				($0) => {
					set_text(text, $0);
					set_class(li, 1, `theme-button ${get$1(current_theme) === 'light' ? 'current-theme' : 'inactive-theme'}`, 'svelte-1t60wb3');
					set_class(li_1, 1, `theme-button ${get$1(current_theme) === 'dark' ? 'current-theme' : 'inactive-theme'}`, 'svelte-1t60wb3');
					set_class(li_2, 1, `theme-button ${get$1(current_theme) === 'system' ? 'current-theme' : 'inactive-theme'}`, 'svelte-1t60wb3');
				},
				[() => $$props.i18n("common.display_theme")]
			);

			event('click', button, () => setTheme("light"));
			event('click', button_1, () => setTheme("dark"));
			event('click', button_2, () => setTheme("system"));
			append($$anchor, div_1);
		};

		if_block(node_1, ($$render) => {
			if ($$props.space_id === null) $$render(consequent);
		});
	}

	var div_2 = sibling(node_1, 2);
	var h2_1 = child(div_2);
	var text_1 = child(h2_1, true);

	reset(h2_1);

	var p = sibling(h2_1, 2);
	var node_2 = child(p);

	Dropdown(node_2, {
		label: 'Language',
		get choices() {
			return language_choices;
		},
		show_label: false,
		on_change: () => handleLanguageChange(get$1(current_locale)),
		get value() {
			return get$1(current_locale);
		},

		set value($$value) {
			set(current_locale, $$value, true);
		}
	});

	reset(p);
	reset(div_2);

	var div_3 = sibling(div_2, 2);
	var h2_2 = child(div_3);
	var text_2 = child(h2_2, true);

	reset(h2_2);

	var p_1 = sibling(h2_2, 2);
	var node_3 = child(p_1);

	{
		var consequent_1 = ($$anchor) => {
			var fragment_1 = root_3();
			var a = sibling(first_child(fragment_1));
			var text_3 = child(a, true);

			reset(a);
			next();

			template_effect(() => {
				set_attribute(a, 'href', root());
				set_text(text_3, root());
			});

			append($$anchor, fragment_1);
		};

		var alternate = ($$anchor) => {
			var fragment_2 = root_4();

			next(2);
			append($$anchor, fragment_2);
		};

		if_block(node_3, ($$render) => {
			if ($$props.pwa_enabled) $$render(consequent_1); else $$render(alternate, false);
		});
	}

	reset(p_1);
	reset(div_3);

	var div_4 = sibling(div_3, 2);
	var h2_3 = child(div_4);
	var text_4 = child(h2_3);

	next();
	reset(h2_3);

	var p_2 = sibling(h2_3, 2);
	var node_4 = sibling(child(p_2), 13);

	Checkbox(node_4, {
		label: 'Include automatic zoom in/out',
		interactive: true,
		get value() {
			return allow_zoom();
		},
		$$events: { change: handleZoomChange }
	});

	var node_5 = sibling(node_4, 2);

	Checkbox(node_5, {
		label: 'Include automatic video trimming',
		interactive: true,
		get value() {
			return allow_video_trim();
		},
		$$events: { change: handleVideoTrimChange }
	});

	reset(p_2);

	var button_3 = sibling(p_2, 2);
	var img = child(button_3);

	next();
	reset(button_3);
	reset(div_4);

	template_effect(
		($0, $1, $2) => {
			set_text(text_1, $0);
			set_text(text_2, $1);
			set_text(text_4, `${$2 ?? ''} `);
			set_attribute(img, 'src', record);
		},
		[
			() => $$props.i18n("common.language"),
			() => $$props.i18n("common.pwa"),
			() => $$props.i18n("common.screen_studio")
		]
	);

	event('click', button_3, () => {
		$$props.onclose?.();
		$$props.start_recording?.();
	});

	append($$anchor, fragment);
	pop();
}

export { Settings as default };
//# sourceMappingURL=Settings-D8txZmif.js.map
