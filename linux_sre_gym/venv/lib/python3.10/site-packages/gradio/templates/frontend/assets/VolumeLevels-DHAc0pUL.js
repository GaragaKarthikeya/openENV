import { i as if_block } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, a6 as comment, S as first_child } from './index-BvBk1Iap.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root$4 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`);

function Pause($$anchor) {
	var svg = root$4();

	append($$anchor, svg);
}

var root$3 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-scissors"><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><line x1="20" y1="4" x2="8.12" y2="15.88"></line><line x1="14.47" y1="14.48" x2="20" y2="20"></line><line x1="8.12" y1="8.12" x2="12" y2="12"></line></svg>`);

function Trim($$anchor) {
	var svg = root$3();

	append($$anchor, svg);
}

var root$2 = from_svg(`<svg width="100%" height="100%" viewBox="0 0 24 24" stroke-width="1.5" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" color="currentColor"><title>Low volume</title><path d="M19.5 7.5C19.5 7.5 21 9 21 11.5C21 14 19.5 15.5 19.5 15.5" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M2 13.8571V10.1429C2 9.03829 2.89543 8.14286 4 8.14286H6.9C7.09569 8.14286 7.28708 8.08544 7.45046 7.97772L13.4495 4.02228C14.1144 3.5839 15 4.06075 15 4.85714V19.1429C15 19.9392 14.1144 20.4161 13.4495 19.9777L7.45046 16.0223C7.28708 15.9146 7.09569 15.8571 6.9 15.8571H4C2.89543 15.8571 2 14.9617 2 13.8571Z" stroke-width="1.5"></path></svg>`);

function VolumeLow($$anchor) {
	var svg = root$2();

	append($$anchor, svg);
}

var root$1 = from_svg(`<svg width="100%" height="100%" viewBox="0 0 24 24" stroke-width="1.5" fill="none" stroke="currentColor" xmlns="http://www.w3.org/2000/svg" color="currentColor"><title>High volume</title><path d="M1 13.8571V10.1429C1 9.03829 1.89543 8.14286 3 8.14286H5.9C6.09569 8.14286 6.28708 8.08544 6.45046 7.97772L12.4495 4.02228C13.1144 3.5839 14 4.06075 14 4.85714V19.1429C14 19.9392 13.1144 20.4161 12.4495 19.9777L6.45046 16.0223C6.28708 15.9146 6.09569 15.8571 5.9 15.8571H3C1.89543 15.8571 1 14.9617 1 13.8571Z" stroke-width="1.5"></path><path d="M17.5 7.5C17.5 7.5 19 9 19 11.5C19 14 17.5 15.5 17.5 15.5" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M20.5 4.5C20.5 4.5 23 7 23 11.5C23 16 20.5 18.5 20.5 18.5" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>`);

function VolumeHigh($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

var root = from_svg(`<svg width="100%" height="100%" viewBox="0 0 24 24" stroke-width="1.5" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" color="currentColor"><title>Muted volume</title><g clip-path="url(#clip0_3173_16686)"><path d="M18 14L20.0005 12M22 10L20.0005 12M20.0005 12L18 10M20.0005 12L22 14" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M2 13.8571V10.1429C2 9.03829 2.89543 8.14286 4 8.14286H6.9C7.09569 8.14286 7.28708 8.08544 7.45046 7.97772L13.4495 4.02228C14.1144 3.5839 15 4.06075 15 4.85714V19.1429C15 19.9392 14.1144 20.4161 13.4495 19.9777L7.45046 16.0223C7.28708 15.9146 7.09569 15.8571 6.9 15.8571H4C2.89543 15.8571 2 14.9617 2 13.8571Z" stroke-width="1.5"></path></g><defs><clipPath id="clip0_3173_16686"><rect width="24" height="24" fill="white"></rect></clipPath></defs></svg>`);

function VolumeMuted($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

function VolumeLevels($$anchor, $$props) {
	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			VolumeMuted($$anchor);
		};

		var alternate_1 = ($$anchor) => {
			var fragment_2 = comment();
			var node_1 = first_child(fragment_2);

			{
				var consequent_1 = ($$anchor) => {
					VolumeLow($$anchor);
				};

				var alternate = ($$anchor) => {
					var fragment_4 = comment();
					var node_2 = first_child(fragment_4);

					{
						var consequent_2 = ($$anchor) => {
							VolumeHigh($$anchor);
						};

						if_block(
							node_2,
							($$render) => {
								if ($$props.currentVolume >= 0.5) $$render(consequent_2);
							},
							true
						);
					}

					append($$anchor, fragment_4);
				};

				if_block(
					node_1,
					($$render) => {
						if ($$props.currentVolume < 0.5) $$render(consequent_1); else $$render(alternate, false);
					},
					true
				);
			}

			append($$anchor, fragment_2);
		};

		if_block(node, ($$render) => {
			if ($$props.currentVolume == 0) $$render(consequent); else $$render(alternate_1, false);
		});
	}

	append($$anchor, fragment);
}

export { Pause as P, Trim as T, VolumeLevels as V };
//# sourceMappingURL=VolumeLevels-DHAc0pUL.js.map
