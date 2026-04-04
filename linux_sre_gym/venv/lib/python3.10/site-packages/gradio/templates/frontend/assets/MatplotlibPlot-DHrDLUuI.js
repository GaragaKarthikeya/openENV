import { a as set_attribute } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, W as from_html, w as get, a5 as user_derived, V as child, Y as reset } from './index-BvBk1Iap.js';

var root = from_html(`<div class="matplotlib layout svelte-n8pych"><img class="svelte-n8pych"/></div>`);

function MatplotlibPlot($$anchor, $$props) {
	push($$props, true);

	let plot = user_derived(() => $$props.value?.plot);
	var div = root();

	set_attribute(div, 'data-testid', "matplotlib");

	var img = child(div);

	reset(div);

	template_effect(() => {
		set_attribute(img, 'src', get(plot));
		set_attribute(img, 'alt', `${$$props.value.chart} plot visualising provided data`);
	});

	append($$anchor, div);
	pop();
}

export { MatplotlibPlot as default };
//# sourceMappingURL=MatplotlibPlot-DHrDLUuI.js.map
