import { c as attribute_effect } from './i18n-CMo5lpzy.js';
import { R as push, Z as event, a as append, T as pop, W as from_html, _ as replay_events } from './index-BvBk1Iap.js';
import { b as bubble_event } from './misc-ByVveAau.js';
/* empty css                                             */

var root = from_html(`<img/>`);

function Image($$anchor, $$props) {
	push($$props, true);

	var img = root();

	attribute_effect(
		img,
		($0) => ({
			src: $$props.src,
			class: $0,
			'data-testid': $$props.data_testid,
			...$$props.restProps
		}),
		[() => ($$props.class_names || []).join(" ")],
		void 0,
		void 0,
		'svelte-1jk6tax'
	);

	replay_events(img);

	event('load', img, function ($$arg) {
		bubble_event.call(this, $$props, $$arg);
	});

	append($$anchor, img);
	pop();
}

export { Image as I };
//# sourceMappingURL=Image-CDw5qb5O.js.map
