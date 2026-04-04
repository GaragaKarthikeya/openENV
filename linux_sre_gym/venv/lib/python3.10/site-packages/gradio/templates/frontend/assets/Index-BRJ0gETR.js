const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./Blocks-Wq-vWI14.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./i18n-CMo5lpzy.js","./i18n-CGFL4PZH.css","./html-CdUEtR5E.js","./ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js","./snippet-Bl5YVoCg.js","./ScrollFade-BXTrLZiw.css","./size-Cu4PabCL.js","./StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js","./index-C7e-J7CF.js","./StreamingBar-M0OJiclA.css","./state.svelte-CCPBiyik.js","./index-Lz7fsHFt.js","./Clear-Dgs5UhTO.js","./utils.svelte-CYLB_jIA.js","./clone-DaB-S2nH.js","./index-BJk8VOjP.js","./index-DnoGeqVF.js","./Blocks-B8btW7fB.css","./Login-DbEEw7Ia.js","./BaseForm-0cuxY3Cw.js","./BaseForm-B79fGGKa.css","./Textbox-v-cNeNHQ.js","./actions-Cit_UA5S.js","./input-C317uaXR.js","./BlockTitle-7XQ-8ENg.js","./Info-BrkJhX6_.js","./Check-BlnBgjLT.js","./Copy-DFFray7x.js","./Send-Ee555fP_.js","./Square-B6INQV79.js","./IconButtonWrapper-Ba0GidgU.js","./Textbox-DbZXQzbK.css","./Block-Bx3BikCN.js","./Button-CH2w4_A4.js","./Image-CDw5qb5O.js","./misc-ByVveAau.js","./Image-DQfoGqtb.css","./Button-CoUFCYL9.css","./ImagePreview-CAbFrrhc.css","./Index.svelte_svelte_type_style_lang-C97CbKPo.js","./Index-D0q0pAfU.css","./Login-CJk1KhUZ.css","./Example-BlSzJhEn.css"])))=>i.map(i=>d[i]);
import { R as push, x as set, a4 as createEventDispatcher, aa as onMount, ae as onDestroy, a1 as legacy_pre_effect, a2 as legacy_pre_effect_reset, T as pop, U as flushSync, w as get, G as deep_read_state, a6 as comment, S as first_child, a as append, a3 as mutable_source, ad as derived_safe_equal, z as untrack, X as sibling, W as from_html, t as template_effect, a0 as set_text, V as child, af as __vitePreload, Y as reset, a8 as next } from './index-BvBk1Iap.js';
import { y as setupi18n, p as prop, e as init$1, E as Embed, m as setup_stores, i as if_block, q as store_get, z as $format, w as writable, A as prefix_css, D as mount_css, a as set_attribute, g as spread_props } from './i18n-CMo5lpzy.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

// src/inject_fonts.ts
var inject_fonts = () => {
  const source_sans_pro = document.createElement("link");
  source_sans_pro.href = "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,200;0,300;0,400;0,600;0,700;0,900;1,200;1,300;1,400;1,600;1,700;1,900&display=swap";
  source_sans_pro.rel = "stylesheet";
  const ibm_mono = document.createElement("link");
  ibm_mono.href = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap";
  ibm_mono.rel = "stylesheet";
  document.head.appendChild(source_sans_pro);
  document.head.appendChild(ibm_mono);
};

// src/header/components/box.ts
var Box = () => {
  const box = document.createElement("div");
  box.style.backgroundImage = "linear-gradient(to top, #f9fafb, white)";
  box.style.border = "1px solid #e5e7eb";
  box.style.borderRadius = "0.75rem";
  box.style.boxShadow = "0 0 10px rgba(0, 0, 0, 0.1)";
  box.style.color = "#374151";
  box.style.display = "flex";
  box.style.flexDirection = "row";
  box.style.alignItems = "center";
  box.style.height = "40px";
  box.style.justifyContent = "space-between";
  box.style.overflow = "hidden";
  box.style.position = "fixed";
  box.style.right = ".75rem";
  box.style.top = ".75rem";
  box.style.width = "auto";
  box.style.zIndex = "20";
  box.style.paddingLeft = "1rem";
  box.setAttribute("id", "huggingface-space-header");
  window.matchMedia("(max-width: 768px)").addEventListener("change", (e) => {
    if (e.matches) {
      box.style.display = "none";
    } else {
      box.style.display = "flex";
    }
  });
  return box;
};

// src/header/components/collapse/arrow.ts
var ArrowCollapse = () => {
  const arrow = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  arrow.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  arrow.setAttribute("xmlns:link", "http://www.w3.org/1999/xlink");
  arrow.setAttribute("aria-hidden", "true");
  arrow.setAttribute("focusable", "false");
  arrow.setAttribute("role", "img");
  arrow.setAttribute("width", "1em");
  arrow.setAttribute("height", "1em");
  arrow.setAttribute("preserveAspectRatio", "xMidYMid meet");
  arrow.setAttribute("viewBox", "0 0 12 12");
  arrow.setAttribute("fill", "currentColor");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute(
    "d",
    "M0.375001 10.3828L0.375 1.61719C0.375 1.104 0.816001 0.687501 1.35938 0.687501L10.6406 0.6875C10.9017 0.6875 11.1521 0.785449 11.3367 0.959797C11.5213 1.13415 11.625 1.37062 11.625 1.61719V10.3828C11.625 10.6294 11.5213 10.8659 11.3367 11.0402C11.1521 11.2145 10.9017 11.3125 10.6406 11.3125H1.35938C0.816001 11.3125 0.375001 10.896 0.375001 10.3828ZM1.35938 10.5156H10.6406C10.7183 10.5156 10.7813 10.4561 10.7813 10.3828V4.40625H1.21875V10.3828C1.21875 10.418 1.23356 10.4518 1.25994 10.4767C1.28631 10.5017 1.32208 10.5156 1.35938 10.5156ZM4.61052 6.38251L5.9999 7.69472L7.38927 6.38251C7.44083 6.33007 7.50645 6.29173 7.57913 6.27153C7.6518 6.25134 7.72898 6.25003 7.8024 6.26776C7.87583 6.28549 7.9428 6.3216 7.99628 6.37227C8.04983 6.42295 8.08785 6.48631 8.10645 6.5557C8.12528 6.62497 8.12393 6.69773 8.10263 6.76635C8.0814 6.83497 8.0409 6.8969 7.98555 6.94564L6.29802 8.53936C6.21892 8.61399 6.11169 8.65592 5.9999 8.65592C5.8881 8.65592 5.78087 8.61399 5.70177 8.53936L4.01427 6.94564C3.95874 6.89694 3.91814 6.835 3.89676 6.76633C3.87538 6.69766 3.874 6.62483 3.89277 6.55549C3.91154 6.48615 3.94977 6.42287 4.00343 6.37233C4.05708 6.32179 4.12418 6.28585 4.19765 6.2683C4.27098 6.25054 4.34803 6.25178 4.42068 6.27188C4.49334 6.29198 4.55891 6.3302 4.61052 6.38251Z"
  );
  arrow.appendChild(path);
  return arrow;
};

// src/header/components/collapse/index.ts
var Collapse = (space, callback) => {
  const box = document.createElement("div");
  box.setAttribute("id", "space-header__collapse");
  box.style.display = "flex";
  box.style.flexDirection = "row";
  box.style.alignItems = "center";
  box.style.justifyContent = "center";
  box.style.fontSize = "16px";
  box.style.paddingLeft = "10px";
  box.style.paddingRight = "10px";
  box.style.height = "40px";
  box.style.cursor = "pointer";
  box.style.color = "#40546e";
  box.style.transitionDuration = "0.1s";
  box.style.transitionProperty = "all";
  box.style.transitionTimingFunction = "ease-in-out";
  box.appendChild(ArrowCollapse());
  box.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    callback();
  });
  box.addEventListener("mouseenter", () => {
    box.style.color = "#213551";
  });
  box.addEventListener("mouseleave", () => {
    box.style.color = "#40546e";
  });
  return box;
};

// src/header/components/like/count.ts
var Count = (count) => {
  const text = document.createElement("p");
  text.style.margin = "0";
  text.style.padding = "0";
  text.style.color = "#9ca3af";
  text.style.fontSize = "14px";
  text.style.fontFamily = "Source Sans Pro, sans-serif";
  text.style.padding = "0px 6px";
  text.style.borderLeft = "1px solid #e5e7eb";
  text.style.marginLeft = "4px";
  text.textContent = (count != null ? count : 0).toString();
  return text;
};

// src/header/components/like/heart.ts
var Heart = () => {
  const heart = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  heart.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  heart.setAttribute("xmlns:link", "http://www.w3.org/1999/xlink");
  heart.setAttribute("aria-hidden", "true");
  heart.setAttribute("focusable", "false");
  heart.setAttribute("role", "img");
  heart.setAttribute("width", "1em");
  heart.setAttribute("height", "1em");
  heart.setAttribute("preserveAspectRatio", "xMidYMid meet");
  heart.setAttribute("viewBox", "0 0 32 32");
  heart.setAttribute("fill", "#6b7280");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute(
    "d",
    "M22.45,6a5.47,5.47,0,0,1,3.91,1.64,5.7,5.7,0,0,1,0,8L16,26.13,5.64,15.64a5.7,5.7,0,0,1,0-8,5.48,5.48,0,0,1,7.82,0L16,10.24l2.53-2.58A5.44,5.44,0,0,1,22.45,6m0-2a7.47,7.47,0,0,0-5.34,2.24L16,7.36,14.89,6.24a7.49,7.49,0,0,0-10.68,0,7.72,7.72,0,0,0,0,10.82L16,29,27.79,17.06a7.72,7.72,0,0,0,0-10.82A7.49,7.49,0,0,0,22.45,4Z"
  );
  heart.appendChild(path);
  return heart;
};

// src/header/components/like/index.ts
var Like = (space) => {
  const box = document.createElement("a");
  box.setAttribute("href", `https://huggingface.co/spaces/${space.id}`);
  box.setAttribute("rel", "noopener noreferrer");
  box.setAttribute("target", "_blank");
  box.style.border = "1px solid #e5e7eb";
  box.style.borderRadius = "6px";
  box.style.display = "flex";
  box.style.flexDirection = "row";
  box.style.alignItems = "center";
  box.style.margin = "0 0 0 12px";
  box.style.fontSize = "14px";
  box.style.paddingLeft = "4px";
  box.style.textDecoration = "none";
  box.appendChild(Heart());
  box.appendChild(Count(space.likes));
  return box;
};

// src/header/components/content/avatar.ts
var Avatar = (username, type = "user") => {
  const route = type === "user" ? "users" : "organizations";
  const element = document.createElement("img");
  element.src = `https://huggingface.co/api/${route}/${username}/avatar`;
  element.style.width = "0.875rem";
  element.style.height = "0.875rem";
  element.style.borderRadius = "50%";
  element.style.flex = "none";
  element.style.marginRight = "0.375rem";
  return element;
};

// src/header/components/content/namespace.ts
var Namespace = (id) => {
  const [_, spaceName] = id.split("/");
  const element = document.createElement("a");
  element.setAttribute("href", `https://huggingface.co/spaces/${id}`);
  element.setAttribute("rel", "noopener noreferrer");
  element.setAttribute("target", "_blank");
  element.style.color = "#1f2937";
  element.style.textDecoration = "none";
  element.style.fontWeight = "600";
  element.style.fontSize = "15px";
  element.style.lineHeight = "24px";
  element.style.flex = "none";
  element.style.fontFamily = "IBM Plex Mono, sans-serif";
  element.addEventListener("mouseover", () => {
    element.style.color = "#2563eb";
  });
  element.addEventListener("mouseout", () => {
    element.style.color = "#1f2937";
  });
  element.textContent = spaceName;
  return element;
};

// src/header/components/content/separation.ts
var Separation = () => {
  const separation = document.createElement("div");
  separation.style.marginLeft = ".125rem";
  separation.style.marginRight = ".125rem";
  separation.style.color = "#d1d5db";
  separation.textContent = "/";
  return separation;
};

// src/header/components/content/username.ts
var Username = (username) => {
  const element = document.createElement("a");
  element.setAttribute("href", `https://huggingface.co/${username}`);
  element.setAttribute("rel", "noopener noreferrer");
  element.setAttribute("target", "_blank");
  element.style.color = "rgb(107, 114, 128)";
  element.style.textDecoration = "none";
  element.style.fontWeight = "400";
  element.style.fontSize = "16px";
  element.style.lineHeight = "24px";
  element.style.flex = "none";
  element.style.fontFamily = "Source Sans Pro, sans-serif";
  element.addEventListener("mouseover", () => {
    element.style.color = "#2563eb";
  });
  element.addEventListener("mouseout", () => {
    element.style.color = "rgb(107, 114, 128)";
  });
  element.textContent = username;
  return element;
};

// src/header/components/content/index.ts
var Content = (space) => {
  const content = document.createElement("div");
  content.style.display = "flex";
  content.style.flexDirection = "row";
  content.style.alignItems = "center";
  content.style.justifyContent = "center";
  content.style.borderRight = "1px solid #e5e7eb";
  content.style.paddingRight = "12px";
  content.style.height = "40px";
  if (space.type !== "unknown") {
    content.appendChild(Avatar(space.author, space.type));
  }
  content.appendChild(Username(space.author));
  content.appendChild(Separation());
  content.appendChild(Namespace(space.id));
  content.appendChild(Like(space));
  return content;
};

// src/header/create.ts
var create = (space) => {
  const box = Box();
  const handleCollapse = () => box.style.display = "none";
  box.appendChild(Content(space));
  box.appendChild(Collapse(space, handleCollapse));
  return box;
};

// src/utils/check_avatar.ts
var check_avatar = async (username, type = "user") => {
  const route = type === "user" ? "users" : "organizations";
  try {
    const response = await fetch(`https://huggingface.co/api/${route}/${username}/avatar`);
    return response.ok;
  } catch (error) {
    return false;
  }
};

// src/utils/get_space.ts
var get_space = async (space_id) => {
  try {
    const response = await fetch(`https://huggingface.co/api/spaces/${space_id}`);
    const data = await response.json();
    return data;
  } catch (error) {
    return null;
  }
};

// src/inject.ts
var inject = (element, options) => {
  if (document.body === null) {
    return console.error("document.body is null");
  }
  document.body.appendChild(element);
};

// src/index.ts
async function main(initialSpace, options) {
  var _a, _b;
  if (window === void 0) return console.error("Please run this script in a browser environment");
  const has_huggingface_ancestor = Object.values(
    (_b = (_a = window.location) == null ? void 0 : _a.ancestorOrigins) != null ? _b : {
      0: window.document.referrer
    }
  ).some((origin) => {
    var _a2;
    return ((_a2 = new URL(origin)) == null ? void 0 : _a2.origin) === "https://huggingface.co";
  });
  if (has_huggingface_ancestor) return;
  inject_fonts();
  let space;
  if (typeof initialSpace === "string") {
    space = await get_space(initialSpace);
    if (space === null) return console.error("Space not found");
  } else {
    space = initialSpace;
  }
  const [user, org] = await Promise.all([check_avatar(space.author, "user"), check_avatar(space.author, "org")]);
  space.type = user ? "user" : org ? "org" : "unknown";
  const mini_header_element = create(space);
  inject(mini_header_element);
  return {
    element: mini_header_element
  };
}
var init = (space, options) => main(space);

let id = -1;
function create_intersection_store() {
  const intersecting2 = writable({});
  const els = /* @__PURE__ */ new Map();
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        let _el = els.get(entry.target);
        if (_el !== void 0) intersecting2.update((s) => ({ ...s, [_el]: true }));
      }
    });
  });
  function register(_id, el) {
    els.set(el, _id);
    observer.observe(el);
  }
  return { register, subscribe: intersecting2.subscribe };
}
const intersecting = create_intersection_store();
var root_6 = from_html(`<p>If your custom component never loads, consult the troubleshooting <a style="color: blue;" href="https://www.gradio.app/guides/frequently-asked-questions#the-development-server-didnt-work-for-me" class="svelte-15izthb">guide</a>.</p>`);
var root_5 = from_html(`<div class="load-text" slot="additional-loading-text"><!></div>`);
var root_8 = from_html(`<p class="svelte-15izthb">Please <a class="svelte-15izthb">contact the author of the space</a> to let them know.</p>`);
var root_10 = from_html(`<p class="svelte-15izthb"> </p>`);
var root_7 = from_html(`<div class="error svelte-15izthb" slot="error"><p class="svelte-15izthb"><strong> </strong></p> <!></div>`);
var root_3 = from_html(`<!> <!>`, 1);
function Index($$anchor, $$props) {
  push($$props, false);
  const $intersecting = () => store_get(intersecting, "$intersecting", $$stores);
  const $_ = () => store_get($format, "$_", $$stores);
  const [$$stores, $$cleanup] = setup_stores();
  let i18n_ready = mutable_source(false);
  setupi18n().then(() => {
    set(i18n_ready, true);
  });
  const dispatch = createEventDispatcher();
  let autoscroll = prop($$props, "autoscroll", 12);
  let version = prop($$props, "version", 12);
  let initial_height = prop($$props, "initial_height", 12);
  let app_mode = prop($$props, "app_mode", 12);
  let is_embed = prop($$props, "is_embed", 12);
  let theme_mode = prop($$props, "theme_mode", 12, "system");
  let control_page_title = prop($$props, "control_page_title", 12);
  let container = prop($$props, "container", 12);
  let info = prop($$props, "info", 12);
  let eager = prop($$props, "eager", 12);
  let stream;
  let pages = mutable_source([]);
  let current_page = mutable_source();
  let root = mutable_source();
  let Client = prop($$props, "Client", 12);
  let space = prop($$props, "space", 12);
  let src = prop($$props, "src", 12);
  let _id = id++;
  let loader_status = mutable_source("pending");
  let app_id = mutable_source(null);
  let wrapper = mutable_source();
  let ready = mutable_source(false);
  let render_complete = mutable_source(false);
  let config = mutable_source();
  let loading_text = mutable_source("Loading...");
  let active_theme_mode = mutable_source();
  let api_url;
  let css_text_stylesheet = null;
  async function mount_custom_css(css_string) {
    if (css_string) {
      if (!css_text_stylesheet) {
        css_text_stylesheet = document.createElement("style");
        document.head.appendChild(css_text_stylesheet);
      }
      css_text_stylesheet.textContent = prefix_css(css_string, version(), css_text_stylesheet);
    }
    await mount_css(get(config).root + "/theme.css?v=" + get(config).theme_hash, document.head);
    if (!get(config).stylesheets) return;
    await Promise.all(get(config).stylesheets.map((stylesheet) => {
      let absolute_link = stylesheet.startsWith("http:") || stylesheet.startsWith("https:");
      if (absolute_link) {
        return mount_css(stylesheet, document.head);
      }
      return fetch(get(config).root + "/" + stylesheet).then((response) => response.text()).then((css_string2) => {
        prefix_css(css_string2, version());
      });
    }));
  }
  async function add_custom_html_head(head_string) {
    if (head_string) {
      const parser = new DOMParser();
      const parsed_head_html = Array.from(parser.parseFromString(head_string, "text/html").head.children);
      if (parsed_head_html) {
        for (let head_element of parsed_head_html) {
          let newElement = document.createElement(head_element.tagName);
          Array.from(head_element.attributes).forEach((attr) => {
            newElement.setAttribute(attr.name, attr.value);
          });
          newElement.textContent = head_element.textContent;
          if (newElement.tagName == "META") {
            const propertyAttr = newElement.getAttribute("property");
            const nameAttr = newElement.getAttribute("name");
            if (propertyAttr || nameAttr) {
              const domMetaList = Array.from(document.head.getElementsByTagName("meta") ?? []);
              const matched = domMetaList.find((el) => {
                if (propertyAttr && el.getAttribute("property") === propertyAttr) {
                  return !el.isEqualNode(newElement);
                }
                if (nameAttr && el.getAttribute("name") === nameAttr) {
                  return !el.isEqualNode(newElement);
                }
                return false;
              });
              if (matched) {
                document.head.replaceChild(newElement, matched);
                continue;
              }
            }
          }
          document.head.appendChild(newElement);
        }
      }
    }
  }
  function handle_theme_mode(target) {
    const force_light = window.__gradio_mode__ === "website";
    let new_theme_mode;
    if (force_light) {
      new_theme_mode = "light";
    } else {
      const url = new URL(window.location.toString());
      const url_color_mode = url.searchParams.get("__theme");
      new_theme_mode = theme_mode() || url_color_mode || "system";
    }
    if (new_theme_mode === "dark" || new_theme_mode === "light") {
      apply_theme(target, new_theme_mode);
    } else {
      new_theme_mode = sync_system_theme(target);
    }
    return new_theme_mode;
  }
  function sync_system_theme(target) {
    const theme = update_scheme();
    window?.matchMedia("(prefers-color-scheme: dark)")?.addEventListener("change", update_scheme);
    function update_scheme() {
      let _theme = window?.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      apply_theme(target, _theme);
      return _theme;
    }
    return theme;
  }
  function apply_theme(target, theme) {
    const dark_class_element = is_embed() ? target.parentElement : document.body;
    const bg_element = is_embed() ? target : target.parentElement;
    bg_element.style.background = "var(--body-background-fill)";
    if (theme === "dark") {
      dark_class_element.classList.add("dark");
    } else {
      dark_class_element.classList.remove("dark");
    }
  }
  let status = mutable_source({
    message: "",
    load_status: "pending",
    status: "sleeping",
    detail: "SLEEPING"
  });
  let app = mutable_source();
  let css_ready = mutable_source(false);
  function handle_status(_status) {
    set(status, _status);
  }
  const gradio_dev_mode = window.__GRADIO_DEV__;
  let pending_deep_link_error = mutable_source(false);
  let new_message_fn = mutable_source();
  let reload_count = mutable_source(0);
  onMount(async () => {
    set(active_theme_mode, handle_theme_mode(get(wrapper)));
    const server_port = window.__GRADIO__SERVER_PORT__;
    api_url = gradio_dev_mode === "dev" ? `http://localhost:${typeof server_port === "number" ? server_port : 7860}` : space() || src() || new URL(location.pathname, location.origin).href.replace(/\/$/, "");
    const deep_link = new URLSearchParams(window.location.search).get("deep_link");
    const query_params = {};
    if (deep_link) {
      query_params.deep_link = deep_link;
    }
    set(app, await Client().connect(api_url, {
      status_callback: handle_status,
      with_null_state: true,
      events: ["data", "log", "status", "render"],
      query_params
    }));
    window.addEventListener("beforeunload", () => {
      get(app).close();
    });
    if (!get(app).config && !get(config)?.auth_required) {
      throw new Error("Could not resolve app config");
    }
    set(config, get(app).get_url_config());
    window.__gradio_space__ = get(config).space_id;
    if (get(app).config?.i18n_translations) {
      await setupi18n(get(app).config.i18n_translations);
      set(i18n_ready, true);
    }
    window.__gradio_session_hash__ = get(app).session_hash;
    set(status, {
      message: "",
      load_status: "complete",
      status: "running",
      detail: "RUNNING"
    });
    await mount_custom_css(get(config).css);
    await add_custom_html_head(get(config).head);
    set(css_ready, true);
    window.__is_colab__ = get(config).is_colab;
    const supports_zerogpu_headers = "supports-zerogpu-headers";
    window.addEventListener("message", (event) => {
      if (event.data === supports_zerogpu_headers) {
        window.supports_zerogpu_headers = true;
      }
    });
    const hostname = window.location.hostname;
    const is_hf_host = hostname.includes(".dev.") || hostname.endsWith(".hf.space");
    if (is_hf_host) {
      const origin = hostname.includes(".dev.") ? `https://moon-${hostname.split(".")[1]}.dev.spaces.huggingface.tech` : `https://huggingface.co`;
      window.parent.postMessage(supports_zerogpu_headers, origin);
    }
    dispatch("loaded");
    set(pages, get(config).pages);
    set(current_page, get(config).current_page);
    set(root, get(config).root);
    if (get(config).deep_link_state === "invalid") {
      set(pending_deep_link_error, true);
    }
    if (get(config).js) {
      try {
        const script = document.createElement("script");
        script.textContent = get(config).js;
        document.head.appendChild(script);
      } catch (e) {
        console.error("Error executing custom JS:", e);
      }
    }
    if (get(config).dev_mode) {
      setTimeout(
        () => {
          const { host } = new URL(api_url);
          let url = new URL(`${window.location.protocol}//${host}${get(app).api_prefix}/dev/reload`);
          stream = new EventSource(url);
          stream.addEventListener("error", async (e) => {
            let event_data = e.data;
            if (event_data) {
              get(new_message_fn)("Error", "Error reloading app", -1, "error", 10, true);
              console.error(JSON.parse(event_data));
            }
          });
          stream.addEventListener("reload", async (event) => {
            get(app).close();
            set(app, await Client().connect(api_url, {
              status_callback: handle_status,
              with_null_state: true,
              events: ["data", "log", "status", "render"],
              session_hash: get(app).session_hash
            }));
            if (!get(app).config) {
              throw new Error("Could not resolve app config");
            }
            set(config, get(app).get_url_config());
            window.__gradio_space__ = get(config).space_id;
            await mount_custom_css(get(config).css);
            await add_custom_html_head(get(config).head);
            set(css_ready, true);
            window.__is_colab__ = get(config).is_colab;
            set(reload_count, get(reload_count) + 1);
            dispatch("loaded");
          });
        },
        200
      );
    }
  });
  let Blocks = mutable_source();
  let Login = mutable_source();
  async function get_blocks() {
    set(Blocks, (await __vitePreload(async () => { const {default: __vite_default__} = await import('./Blocks-Wq-vWI14.js').then(n => n.B);return { default: __vite_default__ }},true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):void 0,import.meta.url)).default);
  }
  async function get_login() {
    set(Login, (await __vitePreload(async () => { const {default: __vite_default__} = await import('./Login-DbEEw7Ia.js');return { default: __vite_default__ }},true              ?__vite__mapDeps([21,3,1,2,4,5,22,23,24,25,26,6,7,8,27,28,29,30,31,32,33,34,10,11,12,35,36,37,38,39,40,41,42,14,15,43,44,45]):void 0,import.meta.url)).default);
  }
  function load_demo() {
    if (get(config).auth_required) get_login();
    else get_blocks();
  }
  let discussion_message = mutable_source();
  onMount(async () => {
    intersecting.register(_id, get(wrapper));
  });
  let spaceheader;
  async function mount_space_header(space_id, is_embed2) {
    if (space_id && !is_embed2 && window.self === window.top) {
      if (spaceheader) {
        spaceheader.remove();
        spaceheader = void 0;
      }
      const header = await init(space_id);
      if (header) spaceheader = header.element;
    }
  }
  onDestroy(() => {
    spaceheader?.remove();
  });
  legacy_pre_effect(() => get(config), () => {
    if (get(config)?.app_id) {
      set(app_id, get(config).app_id);
    }
  });
  legacy_pre_effect(() => (get(new_message_fn), get(pending_deep_link_error)), () => {
    if (get(new_message_fn) && get(pending_deep_link_error)) {
      get(new_message_fn)("Error", "Deep link was not valid", -1, "error", 10, true);
      set(pending_deep_link_error, false);
    }
  });
  legacy_pre_effect(() => (get(ready), get(status)), () => {
    set(loader_status, !get(ready) && get(status).load_status !== "error" ? "pending" : !get(ready) && get(status).load_status === "error" ? "error" : get(status).load_status);
  });
  legacy_pre_effect(() => (get(config), deep_read_state(eager()), $intersecting()), () => {
    get(config) && (eager() || $intersecting()[_id]) && load_demo();
  });
  legacy_pre_effect(() => (get(i18n_ready), $_()), () => {
    if (get(i18n_ready)) {
      set(loading_text, $_()("common.loading") + "...");
      set(discussion_message, {
        readable_error: {
          NO_APP_FILE: $_()("errors.no_app_file"),
          CONFIG_ERROR: $_()("errors.config_error"),
          BUILD_ERROR: $_()("errors.build_error"),
          RUNTIME_ERROR: $_()("errors.runtime_error"),
          PAUSED: $_()("errors.space_paused")
        },
        title(error) {
          return encodeURIComponent($_()("errors.space_not_working"));
        },
        description(error, site) {
          return encodeURIComponent(`Hello,

Firstly, thanks for creating this space!

I noticed that the space isn't working correctly because there is ${this.readable_error[error] || "an error"}.

It would be great if you could take a look at this because this space is being embedded on ${site}.

Thanks!`);
        }
      });
    }
  });
  legacy_pre_effect(() => (get(render_complete), get(wrapper)), () => {
    if (get(render_complete)) {
      get(wrapper).dispatchEvent(new CustomEvent("render", { bubbles: true, cancelable: false, composed: true }));
    }
  });
  legacy_pre_effect(() => (get(app), deep_read_state(is_embed())), () => {
    get(app)?.config && mount_space_header(get(app)?.config?.space_id, is_embed());
  });
  legacy_pre_effect_reset();
  var $$exports = {
    get autoscroll() {
      return autoscroll();
    },
    set autoscroll($$value) {
      autoscroll($$value);
      flushSync();
    },
    get version() {
      return version();
    },
    set version($$value) {
      version($$value);
      flushSync();
    },
    get initial_height() {
      return initial_height();
    },
    set initial_height($$value) {
      initial_height($$value);
      flushSync();
    },
    get app_mode() {
      return app_mode();
    },
    set app_mode($$value) {
      app_mode($$value);
      flushSync();
    },
    get is_embed() {
      return is_embed();
    },
    set is_embed($$value) {
      is_embed($$value);
      flushSync();
    },
    get theme_mode() {
      return theme_mode();
    },
    set theme_mode($$value) {
      theme_mode($$value);
      flushSync();
    },
    get control_page_title() {
      return control_page_title();
    },
    set control_page_title($$value) {
      control_page_title($$value);
      flushSync();
    },
    get container() {
      return container();
    },
    set container($$value) {
      container($$value);
      flushSync();
    },
    get info() {
      return info();
    },
    set info($$value) {
      info($$value);
      flushSync();
    },
    get eager() {
      return eager();
    },
    set eager($$value) {
      eager($$value);
      flushSync();
    },
    get Client() {
      return Client();
    },
    set Client($$value) {
      Client($$value);
      flushSync();
    },
    get space() {
      return space();
    },
    set space($$value) {
      space($$value);
      flushSync();
    },
    get src() {
      return src();
    },
    set src($$value) {
      src($$value);
      flushSync();
    }
  };
  init$1();
  {
    let $0 = derived_safe_equal(() => container() && is_embed());
    let $1 = derived_safe_equal(() => !!space() && info());
    let $2 = derived_safe_equal(() => get(loader_status) === "complete");
    let $3 = derived_safe_equal(() => (get(config), untrack(() => get(config)?.fill_width || false)));
    let $4 = derived_safe_equal(() => (get(config), untrack(() => get(config)?.components || [])));
    Embed($$anchor, {
      get display() {
        return get($0);
      },
      get is_embed() {
        return is_embed();
      },
      get info() {
        return get($1);
      },
      get version() {
        return version();
      },
      get initial_height() {
        return initial_height();
      },
      get space() {
        return space();
      },
      get loaded() {
        return get($2);
      },
      get fill_width() {
        return get($3);
      },
      get pages() {
        return get(pages);
      },
      get current_page() {
        return get(current_page);
      },
      get root() {
        return get(root);
      },
      get components() {
        return get($4);
      },
      get wrapper() {
        return get(wrapper);
      },
      set wrapper($$value) {
        set(wrapper, $$value);
      },
      children: ($$anchor2, $$slotProps) => {
        var fragment_1 = comment();
        var node = first_child(fragment_1);
        {
          var consequent_6 = ($$anchor3) => {
            var fragment_2 = root_3();
            var node_1 = first_child(fragment_2);
            {
              var consequent_3 = ($$anchor4) => {
                {
                  let $02 = derived_safe_equal(() => !is_embed());
                  Static($$anchor4, {
                    get absolute() {
                      return get($02);
                    },
                    get status() {
                      return get(loader_status);
                    },
                    timer: false,
                    queue_position: null,
                    queue_size: null,
                    translucent: true,
                    get loading_text() {
                      return get(loading_text);
                    },
                    get i18n() {
                      return $_();
                    },
                    get autoscroll() {
                      return autoscroll();
                    },
                    $$slots: {
                      "additional-loading-text": ($$anchor5, $$slotProps2) => {
                        var div = root_5();
                        var node_2 = child(div);
                        {
                          var consequent = ($$anchor6) => {
                            var p = root_6();
                            append($$anchor6, p);
                          };
                          if_block(node_2, ($$render) => {
                            if (gradio_dev_mode === "dev") $$render(consequent);
                          });
                        }
                        reset(div);
                        append($$anchor5, div);
                      },
                      error: ($$anchor5, $$slotProps2) => {
                        var div_1 = root_7();
                        var p_1 = child(div_1);
                        var strong = child(p_1);
                        var text = child(strong, true);
                        reset(strong);
                        reset(p_1);
                        var node_3 = sibling(p_1, 2);
                        {
                          var consequent_1 = ($$anchor6) => {
                            var p_2 = root_8();
                            var a = sibling(child(p_2));
                            next();
                            reset(p_2);
                            template_effect(($03, $12) => set_attribute(a, "href", `https://huggingface.co/spaces/${space() ?? ""}/discussions/new?title=${$03 ?? ""}&description=${$12 ?? ""}`), [
                              () => (get(discussion_message), get(status), untrack(() => get(discussion_message).title(get(status)?.detail))),
                              () => (get(discussion_message), get(status), untrack(() => get(discussion_message).description(get(status)?.detail, location.origin)))
                            ]);
                            append($$anchor6, p_2);
                          };
                          var alternate = ($$anchor6) => {
                            var fragment_4 = comment();
                            var node_4 = first_child(fragment_4);
                            {
                              var consequent_2 = ($$anchor7) => {
                                var p_3 = root_10();
                                var text_1 = child(p_3, true);
                                reset(p_3);
                                template_effect(($03) => set_text(text_1, $03), [
                                  () => ($_(), untrack(() => $_()("errors.contact_page_author")))
                                ]);
                                append($$anchor7, p_3);
                              };
                              if_block(
                                node_4,
                                ($$render) => {
                                  if (get(i18n_ready)) $$render(consequent_2);
                                },
                                true
                              );
                            }
                            append($$anchor6, fragment_4);
                          };
                          if_block(node_3, ($$render) => {
                            if (get(status), get(discussion_message), untrack(() => (get(status).status === "space_error" || get(status).status === "paused") && get(status).discussions_enabled && get(discussion_message))) $$render(consequent_1);
                            else $$render(alternate, false);
                          });
                        }
                        reset(div_1);
                        template_effect(() => set_text(text, (get(status), untrack(() => get(status)?.message || ""))));
                        append($$anchor5, div_1);
                      }
                    }
                  });
                }
              };
              if_block(node_1, ($$render) => {
                if (get(loader_status), get(config), untrack(() => (get(loader_status) === "pending" || get(loader_status) === "error") && !(get(config) && get(config)?.auth_required))) $$render(consequent_3);
              });
            }
            var node_5 = sibling(node_1, 2);
            {
              var consequent_4 = ($$anchor4) => {
                {
                  let $02 = derived_safe_equal(() => get(i18n_ready) ? $_() : (s) => s);
                  get(Login)($$anchor4, {
                    get auth_message() {
                      return get(config), untrack(() => get(config).auth_message);
                    },
                    get root() {
                      return get(config), untrack(() => get(config).root);
                    },
                    get space_id() {
                      return space();
                    },
                    get i18n() {
                      return get($02);
                    },
                    get app_mode() {
                      return app_mode();
                    }
                  });
                }
              };
              var alternate_1 = ($$anchor4) => {
                var fragment_6 = comment();
                var node_6 = first_child(fragment_6);
                {
                  var consequent_5 = ($$anchor5) => {
                    {
                      let $02 = derived_safe_equal(() => (deep_read_state(is_embed()), get(config), untrack(() => !is_embed() && get(config).fill_height)));
                      let $12 = derived_safe_equal(() => (deep_read_state(is_embed()), get(config), untrack(() => is_embed() ? [] : get(config).footer_links)));
                      let $22 = derived_safe_equal(() => (get(config), untrack(() => get(config).api_prefix || "")));
                      get(Blocks)($$anchor5, spread_props(
                        {
                          get app() {
                            return get(app);
                          }
                        },
                        () => get(config),
                        {
                          get fill_height() {
                            return get($02);
                          },
                          get theme_mode() {
                            return get(active_theme_mode);
                          },
                          get control_page_title() {
                            return control_page_title();
                          },
                          get target() {
                            return get(wrapper);
                          },
                          get autoscroll() {
                            return autoscroll();
                          },
                          get footer_links() {
                            return get($12);
                          },
                          get app_mode() {
                            return app_mode();
                          },
                          get version() {
                            return version();
                          },
                          get api_prefix() {
                            return get($22);
                          },
                          get max_file_size() {
                            return get(config), untrack(() => get(config).max_file_size);
                          },
                          initial_layout: void 0,
                          search_params: untrack(() => new URLSearchParams(window.location.search)),
                          get reload_count() {
                            return get(reload_count);
                          },
                          get ready() {
                            return get(ready);
                          },
                          set ready($$value) {
                            set(ready, $$value);
                          },
                          get render_complete() {
                            return get(render_complete);
                          },
                          set render_complete($$value) {
                            set(render_complete, $$value);
                          },
                          get add_new_message() {
                            return get(new_message_fn);
                          },
                          set add_new_message($$value) {
                            set(new_message_fn, $$value);
                          },
                          $$legacy: true
                        }
                      ));
                    }
                  };
                  if_block(
                    node_6,
                    ($$render) => {
                      if (get(config) && get(Blocks) && get(css_ready)) $$render(consequent_5);
                    },
                    true
                  );
                }
                append($$anchor4, fragment_6);
              };
              if_block(node_5, ($$render) => {
                if (get(config), get(Login), untrack(() => get(config)?.auth_required && get(Login))) $$render(consequent_4);
                else $$render(alternate_1, false);
              });
            }
            append($$anchor3, fragment_2);
          };
          if_block(node, ($$render) => {
            if (get(i18n_ready)) $$render(consequent_6);
          });
        }
        append($$anchor2, fragment_1);
      },
      $$slots: { default: true },
      $$legacy: true
    });
  }
  var $$pop = pop($$exports);
  $$cleanup();
  return $$pop;
}

export { Index as default };
//# sourceMappingURL=Index-BRJ0gETR.js.map
