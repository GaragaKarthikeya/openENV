import { p as prop, e as init, i as if_block, d as bind_this, f as set_style, b as set_class, k as each, u as index } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, a1 as legacy_pre_effect, G as deep_read_state, M as tick, w as get, a2 as legacy_pre_effect_reset, t as template_effect, z as untrack, Z as event, a as append, T as pop, a3 as mutable_source, x as set, U as flushSync, V as child, X as sibling, W as from_html, Y as reset, a6 as comment, S as first_child, ak as remove_textarea_child, a0 as set_text, a7 as text } from './index-BvBk1Iap.js';
import { b as bind_value } from './input-C317uaXR.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { C as Code } from './Code-C9UfsaaR.js';
/* empty css                                               */
import { M as Markdown } from './Index.svelte_svelte_type_style_lang-BiKot9ba.js';
import './MarkdownCode-Cg7CjFvQ.js';
import './index-C7e-J7CF.js';
/* empty css                                               */
import './html-CdUEtR5E.js';
import './snippet-Bl5YVoCg.js';
import './actions-Cit_UA5S.js';
import './utils.svelte-CYLB_jIA.js';
import './clone-DaB-S2nH.js';
import './Check-BlnBgjLT.js';
import './Copy-DFFray7x.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './prism-python-CbXMXbdR.js';

var root_3 = from_html(`<span class="added svelte-1s2fnws"> </span>`);
var root_4 = from_html(`<span class="removed svelte-1s2fnws"> </span>`);
var root_2 = from_html(`<span class="diff-stats svelte-1s2fnws"><!> <!></span>`);
var root_7 = from_html(`<button class="undo-button svelte-1s2fnws" title="Undo this change">Undo</button>`);
var root_6 = from_html(`<div><div class="message-content svelte-1s2fnws"><span class="message-text svelte-1s2fnws"><!></span> <!></div></div>`);
var root_8 = from_html(`<div class="no-messages svelte-1s2fnws">No messages yet</div>`);
var root_10 = from_html(`<button class="starter-query-button svelte-1s2fnws"> </button>`);
var root_9 = from_html(`<div class="starter-queries-container svelte-1s2fnws"><div class="starter-queries svelte-1s2fnws"></div></div>`);
var root_5 = from_html(`<div class="message-history svelte-1s2fnws"><!> <!> <!></div>`);
var root_12 = from_html(`<div class="code-content svelte-1s2fnws"><div class="code-editor-container svelte-1s2fnws"><!></div> <button><!></button> <button class="deploy-to-spaces-button svelte-1s2fnws"><span class="button-content svelte-1s2fnws">Deploy to <svg class="spaces-logo svelte-1s2fnws" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" preserveAspectRatio="xMidYMid meet" viewBox="0 0 39 40"><path d="M6.3712 2.04427C3.7183 2.04427 1.56771 4.19486 1.56771 6.84776V18.3546V18.6544V32.7341C1.56771 35.3868 3.71818 37.5377 6.3712 37.5377H17.878H20.7507H32.2575C34.9104 37.5377 37.0612 35.387 37.0612 32.7341V21.6204C37.0612 20.177 36.4252 18.8839 35.4189 18.004C36.4576 16.3895 37.0612 14.4666 37.0612 12.4046C37.0612 6.68274 32.4225 2.04427 26.7007 2.04427C24.6388 2.04427 22.7159 2.64776 21.1014 3.68647C20.2214 2.6802 18.9282 2.04427 17.4849 2.04427H6.3712Z" fill="black" class="stroke-white dark:stroke-white/10" stroke-width="3.07552"></path><path d="M9.56855 23.5001C8.8406 23.5001 8.25047 24.0902 8.25047 24.8182V29.5361C8.25047 30.2641 8.8406 30.8542 9.56855 30.8542H14.2864C15.0144 30.8542 15.6045 30.2641 15.6045 29.5361V24.8182C15.6045 24.0902 15.0143 23.5001 14.2864 23.5001H9.56855Z" fill="#FF3270"></path><path d="M24.3409 23.5001C23.613 23.5001 23.0228 24.0902 23.0228 24.8182V29.5361C23.0228 30.2641 23.613 30.8542 24.3409 30.8542H29.0588C29.7868 30.8542 30.3769 30.2641 30.3769 29.5361V24.8182C30.3769 24.0902 29.7868 23.5001 29.0588 23.5001H24.3409Z" fill="#861FFF"></path><path d="M9.56855 8.72815C8.8406 8.72815 8.25047 9.31827 8.25047 10.0462V14.7641C8.25047 15.4921 8.8406 16.0822 9.56855 16.0822H14.2864C15.0144 16.0822 15.6045 15.4921 15.6045 14.7641V10.0462C15.6045 9.31827 15.0143 8.72815 14.2864 8.72815H9.56855Z" fill="#097EFF"></path><path d="M26.6999 8.72815C24.6692 8.72815 23.0228 10.3744 23.0228 12.4052C23.0228 14.4359 24.6692 16.0822 26.6999 16.0822C28.7306 16.0822 30.3769 14.4359 30.3769 12.4052C30.3769 10.3744 28.7306 8.72815 26.6999 8.72815Z" fill="#FFD702"></path></svg> <span style="font-weight: 600;">Spaces</span></span></button></div>`);
var root_1 = from_html(`<div class="vibe-editor svelte-1s2fnws"><button class="resize-handle svelte-1s2fnws" aria-label="Resize sidebar"></button> <div class="tab-header svelte-1s2fnws"><button>Chat</button> <button>Code <!></button></div> <div class="tab-content svelte-1s2fnws"><!></div> <div class="input-section svelte-1s2fnws"><div class="powered-by svelte-1s2fnws">Powered by: <code>gpt-oss</code></div> <textarea placeholder="What can I add or change?" class="prompt-input svelte-1s2fnws"></textarea> <button class="submit-button svelte-1s2fnws">Send</button></div></div>`);

function Index($$anchor, $$props) {
	push($$props, false);

	let app = prop($$props, 'app', 12);
	let root = prop($$props, 'root', 12);
	let prompt = mutable_source("");
	let editorWidth = mutable_source(350);
	let isResizing = false;
	let editorElement = mutable_source();
	let activeTab = mutable_source("chat");
	let codeValue = mutable_source("");
	let diffStats = mutable_source(null);
	let message_history = mutable_source([]);
	let history_elem = mutable_source();

	const scroll_to_bottom = (behavior = "smooth") => {
		if (!get(history_elem)) return;

		get(history_elem).scrollTo({ top: get(history_elem).scrollHeight, behavior });
	};

	let starterQueries = mutable_source([]);

	const fetchStarterQueries = async () => {
		const post = app().post_data(`${root()}/gradio_api/vibe-starter-queries/`, {});

		post.then(async ([response, status_code]) => {
			if (status_code !== 200) {
				throw new Error(`Error: ${status_code}`);
			}

			const responseData = response;

			set(starterQueries, responseData.starter_queries);
		}).catch(async (error) => {
			console.error("Failed to fetch starter queries:", error);
		});
	};

	fetchStarterQueries();

	const submit = async (queryText) => {
		const textToSubmit = queryText || get(prompt);

		if (textToSubmit.trim() === "") return;

		// Clear diff stats when submitting new prompt
		set(diffStats, null);

		const userMessageIndex = get(message_history).length;

		set(message_history, [
			...get(message_history),
			{ text: textToSubmit, isBot: false }
		]);

		const botMessageIndex = get(message_history).length;

		set(message_history, [
			...get(message_history),
			{ text: "Working...", isBot: true, isPending: true }
		]);

		await tick();
		scroll_to_bottom();

		const userPrompt = textToSubmit;

		if (!queryText) {
			set(prompt, "");
		}

		const post = app().post_data(`${root()}/gradio_api/vibe-edit/`, { prompt: userPrompt });

		post.then(async ([response, status_code]) => {
			if (status_code !== 200) {
				throw new Error(`Error: ${status_code}`);
			}

			const responseData = response;

			// Update diff stats from response
			set(diffStats, responseData.diff_stats);

			set(message_history, get(message_history).map((msg, index) => {
				return index === userMessageIndex ? { ...msg, hash: responseData.hash } : msg;
			}));

			set(message_history, get(message_history).map((msg, index) => index === botMessageIndex
				? {
					text: responseData.reasoning ? responseData.reasoning : "Done.",
					isBot: true,
					isPending: false
				}
				: msg));

			await tick();
			scroll_to_bottom();
		}).catch(async (error) => {
			set(message_history, get(message_history).map((msg, index) => index === botMessageIndex
				? { text: "Error occurred.", isBot: true, isPending: false }
				: msg));

			await tick();
			scroll_to_bottom();
		});
	};

	const handleStarterQuery = async (query) => {
		await submit(query);
	};

	const undoMessage = async (hash, messageIndex) => {
		try {
			await app().post_data(`${root()}/gradio_api/undo-vibe-edit/`, { hash });

			// Clear diff stats when undoing
			set(diffStats, null);

			const messageToUndo = get(message_history)[messageIndex];

			set(prompt, messageToUndo.text);
			set(message_history, get(message_history).slice(0, messageIndex));
		} catch(error) {
			console.error("Failed to undo:", error);
		}
	};

	const handleResizeStart = (e) => {
		e.preventDefault();
		isResizing = true;
		document.addEventListener("mousemove", handleResizeMove);
		document.addEventListener("mouseup", handleResizeEnd);
		document.body.style.cursor = "col-resize";
		document.body.style.userSelect = "none";
	};

	const handleResizeMove = (e) => {
		if (!isResizing) return;

		const minWidth = 250;
		const maxWidth = Math.min(800, window.innerWidth * 0.5);
		const newWidth = window.innerWidth - e.clientX;

		set(editorWidth, Math.max(minWidth, Math.min(maxWidth, newWidth)));

		// Dispatch custom event to update the main content margin
		window.dispatchEvent(new CustomEvent("vibeEditorResize", { detail: { width: get(editorWidth) } }));
	};

	const handleResizeEnd = () => {
		isResizing = false;
		document.removeEventListener("mousemove", handleResizeMove);
		document.removeEventListener("mouseup", handleResizeEnd);
		document.body.style.cursor = "";
		document.body.style.userSelect = "";
	};

	const fetchCode = async () => {
		try {
			const response = await fetch(`${root()}/gradio_api/vibe-code/`, {
				method: "GET",
				headers: { "Content-Type": "application/json" }
			});

			if (response.ok) {
				const data = await response.json();

				set(codeValue, data.code);
			}
		} catch(error) {
			console.error("Failed to fetch code:", error);
		}
	};

	let code_updated = mutable_source(false);

	const updateCode = async () => {
		try {
			await app().post_data(`${root()}/gradio_api/vibe-code/`, { code: get(codeValue) });
			set(code_updated, true);

			setTimeout(
				() => {
					set(code_updated, false);
				},
				1000
			);
		} catch(error) {
			console.error("Failed to update code:", error);
		}
	};

	function create_spaces_url() {
		const base_URL = "https://huggingface.co/new-space";
		const params = new URLSearchParams({ name: "new-space", sdk: "gradio" });
		const encoded_content = get(codeValue).trimStart();

		params.append("files[0][path]", "app.py");
		params.append("files[0][content]", encoded_content);
		window.open(`${base_URL}?${params.toString()}`, "_blank")?.focus();
	}

	onMount(() => {
		return () => {
			document.removeEventListener("mousemove", handleResizeMove);
			document.removeEventListener("mouseup", handleResizeEnd);
		};
	});

	legacy_pre_effect(() => (deep_read_state(app())), () => {
		(app(), fetchCode());
	});

	legacy_pre_effect(() => (get(activeTab), tick), () => {
		if (get(activeTab) === "chat") {
			tick().then(() => scroll_to_bottom("auto"));
		}
	});

	legacy_pre_effect(() => (get(code_updated)), () => {
		get(code_updated);
	});

	legacy_pre_effect(() => (get(starterQueries)), () => {
		get(starterQueries);
	});

	legacy_pre_effect_reset();

	var $$exports = {
		get app() {
			return app();
		},

		set app($$value) {
			app($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		}
	};

	init();

	var div = root_1();
	var button = child(div);
	var div_1 = sibling(button, 2);
	var button_1 = child(div_1);
	let classes;
	var button_2 = sibling(button_1, 2);
	let classes_1;
	var node = sibling(child(button_2));

	{
		var consequent_2 = ($$anchor) => {
			var span = root_2();
			var node_1 = child(span);

			{
				var consequent = ($$anchor) => {
					var span_1 = root_3();
					var text = child(span_1);

					reset(span_1);

					template_effect(() => set_text(text, `+${(
						get(diffStats),
						untrack(() => get(diffStats).lines_added)
					) ?? ''}`));

					append($$anchor, span_1);
				};

				if_block(node_1, ($$render) => {
					if ((
						get(diffStats),
						untrack(() => get(diffStats).lines_added > 0)
					)) $$render(consequent);
				});
			}

			var node_2 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var span_2 = root_4();
					var text_1 = child(span_2);

					reset(span_2);

					template_effect(() => set_text(text_1, `-${(
						get(diffStats),
						untrack(() => get(diffStats).lines_removed)
					) ?? ''}`));

					append($$anchor, span_2);
				};

				if_block(node_2, ($$render) => {
					if ((
						get(diffStats),
						untrack(() => get(diffStats).lines_removed > 0)
					)) $$render(consequent_1);
				});
			}

			reset(span);
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if ((
				get(diffStats),
				untrack(() => get(diffStats) && (get(diffStats).lines_added > 0 || get(diffStats).lines_removed > 0))
			)) $$render(consequent_2);
		});
	}

	reset(button_2);
	reset(div_1);

	var div_2 = sibling(div_1, 2);
	var node_3 = child(div_2);

	{
		var consequent_6 = ($$anchor) => {
			var div_3 = root_5();
			var node_4 = child(div_3);

			each(node_4, 1, () => get(message_history), index, ($$anchor, message, index) => {
				var div_4 = root_6();
				let classes_2;
				var div_5 = child(div_4);
				var span_3 = child(div_5);
				var node_5 = child(span_3);

				Markdown(node_5, {
					get value() {
						return (get(message), untrack(() => get(message).text));
					},
					latex_delimiters: [],
					theme_mode: 'system'
				});

				reset(span_3);

				var node_6 = sibling(span_3, 2);

				{
					var consequent_3 = ($$anchor) => {
						var button_3 = root_7();

						event('click', button_3, () => undoMessage(get(message).hash || "", index));
						append($$anchor, button_3);
					};

					if_block(node_6, ($$render) => {
						if ((
							get(message),
							untrack(() => !get(message).isBot && get(message).hash && !get(message).isPending)
						)) $$render(consequent_3);
					});
				}

				reset(div_5);
				reset(div_4);

				template_effect(() => classes_2 = set_class(div_4, 1, 'message-item svelte-1s2fnws', null, classes_2, {
					'bot-message': get(message).isBot,
					'user-message': !get(message).isBot
				}));

				append($$anchor, div_4);
			});

			var node_7 = sibling(node_4, 2);

			{
				var consequent_4 = ($$anchor) => {
					var div_6 = root_8();

					append($$anchor, div_6);
				};

				if_block(node_7, ($$render) => {
					if ((
						get(message_history),
						untrack(() => get(message_history).length === 0)
					)) $$render(consequent_4);
				});
			}

			var node_8 = sibling(node_7, 2);

			{
				var consequent_5 = ($$anchor) => {
					var div_7 = root_9();
					var div_8 = child(div_7);

					each(div_8, 5, () => get(starterQueries), index, ($$anchor, query) => {
						var button_4 = root_10();
						var text_2 = child(button_4, true);

						reset(button_4);
						template_effect(() => set_text(text_2, get(query)));
						event('click', button_4, () => handleStarterQuery(get(query)));
						append($$anchor, button_4);
					});

					reset(div_8);
					reset(div_7);
					append($$anchor, div_7);
				};

				if_block(node_8, ($$render) => {
					if ((
						get(message_history),
						untrack(() => get(message_history).length === 0)
					)) $$render(consequent_5);
				});
			}

			reset(div_3);
			bind_this(div_3, ($$value) => set(history_elem, $$value), () => get(history_elem));
			append($$anchor, div_3);
		};

		var alternate_1 = ($$anchor) => {
			var fragment = comment();
			var node_9 = first_child(fragment);

			{
				var consequent_8 = ($$anchor) => {
					var div_9 = root_12();
					var div_10 = child(div_9);
					var node_10 = child(div_10);

					Code(node_10, {
						language: 'python',
						lines: 10,
						dark_mode: false,
						basic: true,
						readonly: false,
						placeholder: 'Enter your code here...',
						wrap_lines: true,
						get value() {
							return get(codeValue);
						},

						set value($$value) {
							set(codeValue, $$value);
						},
						$$legacy: true
					});

					reset(div_10);

					var button_5 = sibling(div_10, 2);
					let classes_3;
					var node_11 = child(button_5);

					{
						var consequent_7 = ($$anchor) => {
							var text_3 = text('Updated!');

							append($$anchor, text_3);
						};

						var alternate = ($$anchor) => {
							var text_4 = text('Update Code');

							append($$anchor, text_4);
						};

						if_block(node_11, ($$render) => {
							if (get(code_updated)) $$render(consequent_7); else $$render(alternate, false);
						});
					}

					reset(button_5);

					var button_6 = sibling(button_5, 2);

					reset(div_9);
					template_effect(() => classes_3 = set_class(button_5, 1, 'update-code-button svelte-1s2fnws', null, classes_3, { updating: get(code_updated) }));
					event('click', button_5, updateCode);
					event('click', button_6, () => create_spaces_url());
					append($$anchor, div_9);
				};

				if_block(
					node_9,
					($$render) => {
						if (get(activeTab) === "code") $$render(consequent_8);
					},
					true
				);
			}

			append($$anchor, fragment);
		};

		if_block(node_3, ($$render) => {
			if (get(activeTab) === "chat") $$render(consequent_6); else $$render(alternate_1, false);
		});
	}

	reset(div_2);

	var div_11 = sibling(div_2, 2);
	var textarea = sibling(child(div_11), 2);

	remove_textarea_child(textarea);

	var button_7 = sibling(textarea, 2);

	reset(div_11);
	reset(div);
	bind_this(div, ($$value) => set(editorElement, $$value), () => get(editorElement));

	template_effect(
		($0) => {
			set_style(div, `width: ${get(editorWidth) ?? ''}px;`);
			classes = set_class(button_1, 1, 'tab-button svelte-1s2fnws', null, classes, { active: get(activeTab) === "chat" });
			classes_1 = set_class(button_2, 1, 'tab-button svelte-1s2fnws', null, classes_1, { active: get(activeTab) === "code" });
			button_7.disabled = $0;
		},
		[
			() => (get(prompt), untrack(() => get(prompt).trim() === ""))
		]
	);

	event('mousedown', button, handleResizeStart);
	event('click', button_1, () => set(activeTab, "chat"));
	event('click', button_2, () => set(activeTab, "code"));

	event('keydown', textarea, (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	});

	bind_value(textarea, () => get(prompt), ($$value) => set(prompt, $$value));
	event('click', button_7, () => submit());
	append($$anchor, div);

	return pop($$exports);
}

export { Index as default };
//# sourceMappingURL=Index-DrpT8pjh.js.map
