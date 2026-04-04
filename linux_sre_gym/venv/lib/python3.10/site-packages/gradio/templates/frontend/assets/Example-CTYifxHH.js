import { p as prop, i as if_block, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, a6 as comment, S as first_child, a as append, T as pop, V as child, x as set, w as get, Y as reset, t as template_effect, W as from_html, u as state, a5 as user_derived } from './index-BvBk1Iap.js';
import { V as Video } from './Video-B3xDM6GN.js';
import './snippet-Bl5YVoCg.js';
import './actions-Cit_UA5S.js';

var root_2 = from_html(`<div><!></div>`);

function Example($$anchor, $$props) {
	push($$props, true);

	let selected = prop($$props, 'selected', 3, false),
		value = prop($$props, 'value', 3, null);

	let video = state(void 0);

	async function init() {
		if (!get(video)) return;

		get(video).muted = true;
		get(video).playsInline = true;
		get(video).controls = false;
		get(video).setAttribute("muted", "");
		await get(video).play();
		get(video).pause();
	}

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_1 = ($$anchor) => {
			var fragment_1 = comment();
			var node_1 = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					var div = root_2();
					let classes;
					var node_2 = child(div);

					{
						let $0 = user_derived(() => value()?.url);

						Video(node_2, {
							muted: true,
							playsinline: true,
							onloadeddata: init,
							onmouseover: () => get(video)?.play(),
							onmouseout: () => get(video)?.pause(),
							get src() {
								return get($0);
							},
							is_stream: false,
							get loop() {
								return $$props.loop;
							},

							get node() {
								return get(video);
							},

							set node($$value) {
								set(video, $$value, true);
							}
						});
					}

					reset(div);

					template_effect(() => classes = set_class(div, 1, 'container svelte-1nl1glk', null, classes, {
						table: $$props.type === "table",
						gallery: $$props.type === "gallery",
						selected: selected()
					}));

					append($$anchor, div);
				};

				if_block(node_1, ($$render) => {
					$$render(consequent);
				});
			}

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if (value()) $$render(consequent_1);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { Example as default };
//# sourceMappingURL=Example-CTYifxHH.js.map
