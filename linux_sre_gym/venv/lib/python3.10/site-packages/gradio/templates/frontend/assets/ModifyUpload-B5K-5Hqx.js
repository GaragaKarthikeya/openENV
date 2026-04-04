import { p as prop, i as if_block } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, X as sibling, w as get, a as append, a5 as user_derived, a6 as comment, T as pop, W as from_html } from './index-BvBk1Iap.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { D as DownloadLink } from './DownloadLink-KvrFKyMn.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { E as Edit } from './Edit-Dr58YXBX.js';
import { U as Undo } from './Undo-BxcJDM8E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';

var root_1 = from_html(`<!> <!> <!> <!> <!>`, 1);

function ModifyUpload($$anchor, $$props) {
	push($$props, true);

	let editable = prop($$props, 'editable', 3, false),
		undoable = prop($$props, 'undoable', 3, false),
		download = prop($$props, 'download', 3, null);

	IconButtonWrapper($$anchor, {
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					{
						let $0 = user_derived(() => $$props.i18n("common.edit"));

						IconButton($$anchor, {
							get Icon() {
								return Edit;
							},

							get label() {
								return get($0);
							},
							onclick: () => $$props.onedit?.()
						});
					}
				};

				if_block(node, ($$render) => {
					if (editable()) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				var consequent_1 = ($$anchor) => {
					{
						let $0 = user_derived(() => $$props.i18n("common.undo"));

						IconButton($$anchor, {
							get Icon() {
								return Undo;
							},

							get label() {
								return get($0);
							},
							onclick: () => $$props.onundo?.()
						});
					}
				};

				if_block(node_1, ($$render) => {
					if (undoable()) $$render(consequent_1);
				});
			}

			var node_2 = sibling(node_1, 2);

			{
				var consequent_2 = ($$anchor) => {
					DownloadLink($$anchor, {
						get href() {
							return download();
						},
						download: true,
						children: ($$anchor, $$slotProps) => {
							{
								let $0 = user_derived(() => $$props.i18n("common.download"));

								IconButton($$anchor, {
									get Icon() {
										return Download;
									},

									get label() {
										return get($0);
									}
								});
							}
						},
						$$slots: { default: true }
					});
				};

				if_block(node_2, ($$render) => {
					if (download()) $$render(consequent_2);
				});
			}

			var node_3 = sibling(node_2, 2);

			{
				var consequent_3 = ($$anchor) => {
					var fragment_6 = comment();
					var node_4 = first_child(fragment_6);

					snippet(node_4, () => $$props.children);
					append($$anchor, fragment_6);
				};

				if_block(node_3, ($$render) => {
					if ($$props.children) $$render(consequent_3);
				});
			}

			var node_5 = sibling(node_3, 2);

			{
				let $0 = user_derived(() => $$props.i18n("common.clear"));

				IconButton(node_5, {
					get Icon() {
						return Clear;
					},

					get label() {
						return get($0);
					},

					onclick: (event) => {
						$$props.onclear?.();
						event.stopPropagation();
					}
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { ModifyUpload as M };
//# sourceMappingURL=ModifyUpload-B5K-5Hqx.js.map
