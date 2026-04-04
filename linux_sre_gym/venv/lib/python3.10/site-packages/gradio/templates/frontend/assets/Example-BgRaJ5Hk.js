import { p as prop, e as init, i as if_block, b as set_class, k as each, a as set_attribute, u as index } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, W as from_html, G as deep_read_state, z as untrack, V as child, a6 as comment, S as first_child, w as get, X as sibling, Y as reset, a0 as set_text } from './index-BvBk1Iap.js';

var root_4 = from_html(`<span class="caption svelte-xds4q5"> </span>`);
var root_3 = from_html(`<div class="image-container svelte-xds4q5"><img class="svelte-xds4q5"/> <!></div>`);
var root_7 = from_html(`<span class="caption svelte-xds4q5"> </span>`);
var root_6 = from_html(`<div class="image-container svelte-xds4q5"><video preload="metadata" class="svelte-xds4q5"></video> <!></div>`, 2);
var root_8 = from_html(`<div class="more-indicator svelte-xds4q5">…</div>`);
var root_1 = from_html(`<div class="images-wrapper svelte-xds4q5"><!> <!></div>`);
var root = from_html(`<div><!></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get type() {
			return type();
		},

		set type($$value) {
			type($$value);
			flushSync();
		},

		get selected() {
			return selected();
		},

		set selected($$value) {
			selected($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	let classes;
	var node = child(div);

	{
		var consequent_5 = ($$anchor) => {
			var div_1 = root_1();
			var node_1 = child(div_1);

			each(
				node_1,
				1,
				() => (
					deep_read_state(value()),
					untrack(() => value().slice(0, 3))
				),
				index,
				($$anchor, item) => {
					var fragment = comment();
					var node_2 = first_child(fragment);

					{
						var consequent_1 = ($$anchor) => {
							var div_2 = root_3();
							var img = child(div_2);
							var node_3 = sibling(img, 2);

							{
								var consequent = ($$anchor) => {
									var span = root_4();
									var text = child(span, true);

									reset(span);
									template_effect(() => set_text(text, (get(item), untrack(() => get(item).caption))));
									append($$anchor, span);
								};

								if_block(node_3, ($$render) => {
									if ((get(item), untrack(() => get(item).caption))) $$render(consequent);
								});
							}

							reset(div_2);

							template_effect(() => {
								set_attribute(img, 'src', (get(item), untrack(() => get(item).image.url)));
								set_attribute(img, 'alt', (get(item), untrack(() => get(item).caption || "")));
							});

							append($$anchor, div_2);
						};

						var alternate = ($$anchor) => {
							var fragment_1 = comment();
							var node_4 = first_child(fragment_1);

							{
								var consequent_3 = ($$anchor) => {
									var div_3 = root_6();
									var video = child(div_3);

									video.controls = false;
									video.muted = true;

									var node_5 = sibling(video, 2);

									{
										var consequent_2 = ($$anchor) => {
											var span_1 = root_7();
											var text_1 = child(span_1, true);

											reset(span_1);
											template_effect(() => set_text(text_1, (get(item), untrack(() => get(item).caption))));
											append($$anchor, span_1);
										};

										if_block(node_5, ($$render) => {
											if ((get(item), untrack(() => get(item).caption))) $$render(consequent_2);
										});
									}

									reset(div_3);
									template_effect(() => set_attribute(video, 'src', (get(item), untrack(() => get(item).video.url))));
									append($$anchor, div_3);
								};

								if_block(
									node_4,
									($$render) => {
										if ((
											get(item),
											untrack(() => "video" in get(item) && get(item).video)
										)) $$render(consequent_3);
									},
									true
								);
							}

							append($$anchor, fragment_1);
						};

						if_block(node_2, ($$render) => {
							if ((
								get(item),
								untrack(() => "image" in get(item) && get(item).image)
							)) $$render(consequent_1); else $$render(alternate, false);
						});
					}

					append($$anchor, fragment);
				}
			);

			var node_6 = sibling(node_1, 2);

			{
				var consequent_4 = ($$anchor) => {
					var div_4 = root_8();

					append($$anchor, div_4);
				};

				if_block(node_6, ($$render) => {
					if ((
						deep_read_state(value()),
						untrack(() => value().length > 3)
					)) $$render(consequent_4);
				});
			}

			reset(div_1);
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if ((
				deep_read_state(value()),
				untrack(() => value() && value().length > 0)
			)) $$render(consequent_5);
		});
	}

	reset(div);

	template_effect(() => classes = set_class(div, 1, 'container svelte-xds4q5', null, classes, {
		table: type() === "table",
		gallery: type() === "gallery",
		selected: selected()
	}));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-BgRaJ5Hk.js.map
