import './async-D55cHugf.js';
import { h as head, a as attr, e as ensure_array_like, b as store_get, c as spread_props, u as unsubscribe_stores, d as bind_props, f as attr_class, g as attr_style } from './index-u8mz_F03.js';
import { r as run } from './legacy-server-CxBQ4HtA.js';
import { g as getContext } from './context-CBkBucIx.js';
import { Embed, AppTree } from './index-DuBh77GF.js';
import { $ as $format, p as prefix_css, A as AsyncFunction } from './2-BAeILJ1g.js';
import { D as DEV, B as BROWSER } from './dev-fallback-B-RpELjM.js';
import { s as stores } from './client-sDHYjkUT.js';
import { b as browser } from './index5-BZVOFaHm.js';
import { d as derived, w as writable, g as get } from './index-Cg-Pg6j3.js';
import { L as LoadingStatus, T as Toast } from './state.svelte-C5Vy1lZM.js';
import { I as I18N_MARKER, t as translate_i18n_marker } from './utils.svelte-aYomt3KL.js';
import { B as BaseForm } from './BaseForm-BRMjUqML.js';
import { T as Textbox } from './Textbox-T7HntrB1.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as Button } from './Button-C8geAOuh.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-BbzONYlY.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { h as html } from './html-CfyvkLET.js';
import './index-server-CQz6EZl_.js';
import './exports-Dz6pzFz_.js';
import './index50-nVz3eHRG.js';
import 'path';
import 'url';
import 'fs';
import './index3-BSof0MuO.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';
import './clone-Yk88IHKV.js';
import './BlockTitle-B0onSZmA.js';
import './Info-BeS-ygt-.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './Send-zUFiC8KE.js';
import './Square-CJzdI5ja.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './Image-BLyvTcTy.js';

const getStores = () => {
  const stores$1 = BROWSER ? stores : getContext("__svelte__");
  return {
    /** @type {typeof page} */
    page: {
      subscribe: stores$1.page.subscribe
    },
    /** @type {typeof navigating} */
    navigating: {
      subscribe: stores$1.navigating.subscribe
    },
    /** @type {typeof updated} */
    updated: stores$1.updated
  };
};
const page = {
  subscribe(fn) {
    const store = DEV ? get_store("page") : getStores().page;
    return store.subscribe(fn);
  }
};
function get_store(name) {
  try {
    return getStores()[name];
  } catch {
    throw new Error(
      `Cannot subscribe to '${name}' store on the server outside of a Svelte component, as it is bound to the current request via component context. This prevents state from leaking between users.For more information, see https://svelte.dev/docs/kit/state-management#avoid-shared-state-on-the-server`
    );
  }
}
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
  text.textContent = (count ?? 0).toString();
  return text;
};
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
var Namespace = (id2) => {
  const [_, spaceName] = id2.split("/");
  const element = document.createElement("a");
  element.setAttribute("href", `https://huggingface.co/spaces/${id2}`);
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
var Separation = () => {
  const separation = document.createElement("div");
  separation.style.marginLeft = ".125rem";
  separation.style.marginRight = ".125rem";
  separation.style.color = "#d1d5db";
  separation.textContent = "/";
  return separation;
};
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
var create = (space) => {
  const box = Box();
  const handleCollapse = () => box.style.display = "none";
  box.appendChild(Content(space));
  box.appendChild(Collapse(space, handleCollapse));
  return box;
};
var check_avatar = async (username, type = "user") => {
  const route = type === "user" ? "users" : "organizations";
  try {
    const response = await fetch(`https://huggingface.co/api/${route}/${username}/avatar`);
    return response.ok;
  } catch (error) {
    return false;
  }
};
var get_space = async (space_id) => {
  try {
    const response = await fetch(`https://huggingface.co/api/spaces/${space_id}`);
    const data = await response.json();
    return data;
  } catch (error) {
    return null;
  }
};
var inject = (element, options) => {
  if (document.body === null) {
    return console.error("document.body is null");
  }
  document.body.appendChild(element);
};
async function main(initialSpace, options) {
  if (window === void 0) return console.error("Please run this script in a browser environment");
  const has_huggingface_ancestor = Object.values(
    window.location?.ancestorOrigins ?? {
      0: window.document.referrer
    }
  ).some((origin) => new URL(origin)?.origin === "https://huggingface.co");
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
function MountCustomComponent($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { node, children, $$slots, $$events, ...rest } = $$props;
    $$renderer2.run([
      async () => await node.component,
      async () => await node.runtime,
      () => null,
      () => void 0
    ]);
    $$renderer2.push(`<span></span>`);
  });
}
function MountComponents($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { node, $$slots, $$events, ...rest } = $$props;
    var component;
    var $$promises = $$renderer2.run([async () => component = await node.component]);
    $$renderer2.async_block([$$promises[0]], ($$renderer3) => {
      if (node && component) {
        $$renderer3.push("<!--[-->");
        $$renderer3.async_block([$$promises[0]], ($$renderer4) => {
          if (node.props.shared_props.visible && !node.runtime) {
            $$renderer4.push("<!--[-->");
            $$renderer4.async_block([$$promises[0]], ($$renderer5) => {
              $$renderer5.push("<!---->");
              component.default?.($$renderer5, {
                shared_props: node.props.shared_props,
                props: node.props.props,
                children: ($$renderer6) => {
                  $$renderer6.async_block([$$promises[0]], ($$renderer7) => {
                    if (node.children && node.children.length) {
                      $$renderer7.push("<!--[-->");
                      $$renderer7.push(`<!--[-->`);
                      $$renderer7.async_block([$$promises[0]], ($$renderer8) => {
                        const each_array = ensure_array_like(node.children);
                        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                          let _node = each_array[$$index];
                          MountComponents($$renderer8, { node: _node });
                        }
                      });
                      $$renderer7.push(`<!--]-->`);
                    } else {
                      $$renderer7.push("<!--[!-->");
                    }
                  });
                  $$renderer6.push(`<!--]-->`);
                },
                $$slots: { default: true }
              });
            });
          } else {
            $$renderer4.push("<!--[!-->");
            $$renderer4.async_block([$$promises[0]], ($$renderer5) => {
              if (node.props.shared_props.visible && node.runtime) {
                $$renderer5.push("<!--[-->");
                $$renderer5.async_block([$$promises[0]], ($$renderer6) => {
                  MountCustomComponent($$renderer6, spread_props([
                    rest,
                    {
                      node,
                      children: ($$renderer7) => {
                        $$renderer7.async_block([$$promises[0]], ($$renderer8) => {
                          if (node.children && node.children.length) {
                            $$renderer8.push("<!--[-->");
                            $$renderer8.push(`<!--[-->`);
                            $$renderer8.async_block([$$promises[0]], ($$renderer9) => {
                              const each_array_1 = ensure_array_like(node.children);
                              for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
                                let _node = each_array_1[$$index_1];
                                MountComponents($$renderer9, { node: _node });
                              }
                            });
                            $$renderer8.push(`<!--]-->`);
                          } else {
                            $$renderer8.push("<!--[!-->");
                          }
                        });
                        $$renderer7.push(`<!--]-->`);
                      },
                      $$slots: { default: true }
                    }
                  ]));
                });
              } else {
                $$renderer5.push("<!--[!-->");
              }
            });
            $$renderer4.push(`<!--]-->`);
          }
        });
        $$renderer3.push(`<!--]-->`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
    });
    $$renderer2.push(`<!--]-->`);
  });
}
function formatter(value) {
  if (value == null) {
    return "";
  }
  const string_value = String(value);
  const translate = get($format);
  if (string_value.includes(I18N_MARKER)) {
    return translate_i18n_marker(string_value, translate);
  }
  const direct_translation = translate(string_value);
  if (typeof direct_translation === "string" && direct_translation !== string_value) {
    return direct_translation;
  }
  return string_value;
}
const reactive_formatter = derived($format, () => formatter);
const logo = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='576'%20height='576'%20viewBox='0%200%20576%20576'%20fill='none'%3e%3cpath%20d='M287.5%20229L86%20344.5L287.5%20460L489%20344.5L287.5%20229Z'%20stroke='url(%23paint0_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='round'/%3e%3cpath%20d='M287.5%20116L86%20231.5L287.5%20347L489%20231.5L287.5%20116Z'%20stroke='url(%23paint1_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='round'/%3e%3cpath%20d='M86%20344L288%20229'%20stroke='url(%23paint2_linear_102_7)'%20stroke-width='59'%20stroke-linejoin='bevel'/%3e%3cdefs%3e%3clinearGradient%20id='paint0_linear_102_7'%20x1='60'%20y1='341'%20x2='429.5'%20y2='344'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3clinearGradient%20id='paint1_linear_102_7'%20x1='513.5'%20y1='231'%20x2='143.5'%20y2='231'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3clinearGradient%20id='paint2_linear_102_7'%20x1='60'%20y1='344'%20x2='428.987'%20y2='341.811'%20gradientUnits='userSpaceOnUse'%3e%3cstop%20stop-color='%23F9D100'/%3e%3cstop%20offset='1'%20stop-color='%23F97700'/%3e%3c/linearGradient%3e%3c/defs%3e%3c/svg%3e";
const api_logo = "data:image/svg+xml,%3csvg%20width='28'%20height='28'%20viewBox='0%200%2028%2028'%20fill='none'%20xmlns='http://www.w3.org/2000/svg'%3e%3cpath%20fill-rule='evenodd'%20clip-rule='evenodd'%20d='M26.9425%202.94265C27.4632%202.42195%2027.4632%201.57773%2026.9425%201.05703C26.4218%200.536329%2025.5776%200.536329%2025.0569%201.05703L22.5713%203.54256C21.1213%202.59333%2019.5367%202.43378%2018.1753%202.64006C16.5495%202.88638%2015.1127%203.66838%2014.3905%204.39053L12.3905%206.39053C12.1405%206.64058%2012%206.97972%2012%207.33334C12%207.68697%2012.1405%208.0261%2012.3905%208.27615L19.7239%2015.6095C20.2446%2016.1302%2021.0888%2016.1302%2021.6095%2015.6095L23.6095%2013.6095C24.3316%2012.8873%2025.1136%2011.4505%2025.36%209.82475C25.5663%208.46312%2025.4066%206.87827%2024.4571%205.42807L26.9425%202.94265Z'%20fill='%233c4555'/%3e%3cpath%20fill-rule='evenodd'%20clip-rule='evenodd'%20d='M12.276%2012.9426C12.7967%2012.4219%2012.7967%2011.5777%2012.276%2011.057C11.7553%2010.5363%2010.9111%2010.5363%2010.3904%2011.057L8.66651%2012.7809L8.27615%2012.3905C8.0261%2012.1405%207.68697%2012%207.33334%2012C6.97972%2012%206.64058%2012.1405%206.39053%2012.3905L4.39053%2014.3905C3.66838%2015.1127%202.88638%2016.5495%202.64006%2018.1753C2.43377%2019.5367%202.59333%2021.1214%203.54262%2022.5714L1.05703%2025.057C0.536329%2025.5777%200.536329%2026.4219%201.05703%2026.9426C1.57773%2027.4633%202.42195%2027.4633%202.94265%2026.9426L5.42817%2024.4571C6.87835%2025.4066%208.46315%2025.5663%209.82475%2025.36C11.4505%2025.1136%2012.8873%2024.3316%2013.6095%2023.6095L15.6095%2021.6095C16.1302%2021.0888%2016.1302%2020.2446%2015.6095%2019.7239L15.2188%2019.3332L16.9426%2017.6093C17.4633%2017.0886%2017.4633%2016.2444%2016.9426%2015.7237C16.4219%2015.203%2015.5777%2015.203%2015.057%2015.7237L13.3332%2017.4475L10.5521%2014.6665L12.276%2012.9426Z'%20fill='%23FF7C00'/%3e%3c/svg%3e";
const settings_logo = "data:image/svg+xml,%3csvg%20width='24'%20height='24'%20viewBox='0%200%2024%2024'%20xmlns='http://www.w3.org/2000/svg'%3e%3c!--%20Outer%20gear%20teeth%20(gray)%20--%3e%3cpath%20d='M19.14%2012.94c.04-.3.06-.61.06-.94%200-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24%200-.43.17-.47.41l-.36%202.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47%200-.59.22L2.74%208.87c-.12.21-.08.47.12.61l2.03%201.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03%201.58c-.18.14-.23.41-.12.61l1.92%203.32c.12.22.37.29.59.22l2.39-.96c.5.38%201.03.7%201.62.94l.36%202.54c.05.24.24.41.48.41h3.84c.24%200%20.44-.17.47-.41l.36-2.54c.59-.24%201.13-.56%201.62-.94l2.39.96c.22.08.47%200%20.59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12%2015.6c-1.98%200-3.6-1.62-3.6-3.6s1.62-3.6%203.6-3.6%203.6%201.62%203.6%203.6-1.62%203.6-3.6%203.6z'%20fill='%23808080'/%3e%3c!--%20Inner%20circle%20(now%20gray)%20--%3e%3ccircle%20cx='12'%20cy='12'%20r='2.5'%20fill='%23808080'/%3e%3c/svg%3e";
const record_stop = "data:image/svg+xml,%3csvg%20viewBox='0%200%2020%2020'%20version='1.1'%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20fill='%23000000'%3e%3cg%20id='SVGRepo_bgCarrier'%20stroke-width='0'%3e%3c/g%3e%3cg%20id='SVGRepo_tracerCarrier'%20stroke-linecap='round'%20stroke-linejoin='round'%3e%3c/g%3e%3cg%20id='SVGRepo_iconCarrier'%3e%3ctitle%3erecord%20[%23982]%3c/title%3e%3cdesc%3eCreated%20with%20Sketch.%3c/desc%3e%3cdefs%3e%3c/defs%3e%3cg%20id='Page-1'%20stroke='none'%20stroke-width='1'%20fill='none'%20fill-rule='evenodd'%3e%3cg%20id='Dribbble-Light-Preview'%20transform='translate(-380.000000,%20-3839.000000)'%20fill='%23FF0000'%3e%3cg%20id='icons'%20transform='translate(56.000000,%20160.000000)'%3e%3cpath%20d='M338,3689%20C338,3691.209%20336.209,3693%20334,3693%20C331.791,3693%20330,3691.209%20330,3689%20C330,3686.791%20331.791,3685%20334,3685%20C336.209,3685%20338,3686.791%20338,3689%20M334,3697%20C329.589,3697%20326,3693.411%20326,3689%20C326,3684.589%20329.589,3681%20334,3681%20C338.411,3681%20342,3684.589%20342,3689%20C342,3693.411%20338.411,3697%20334,3697%20M334,3679%20C328.477,3679%20324,3683.477%20324,3689%20C324,3694.523%20328.477,3699%20334,3699%20C339.523,3699%20344,3694.523%20344,3689%20C344,3683.477%20339.523,3679%20334,3679'%20id='record-[%23982]'%3e%3c/path%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/g%3e%3c/svg%3e";
const MESSAGE_QUOTE_RE = /^'([^]+)'$/;
const NOVALUE = /* @__PURE__ */ Symbol("NOVALUE");
class Dependency {
  id;
  inputs;
  outputs;
  cancels;
  pending = false;
  trigger_modes;
  event_args = {};
  targets = [];
  connection_type;
  // if this dependency has any then, success or failure triggers
  triggers = [];
  // the id of the original event_id that caused this dependency to run
  // in the case of chained events, it would be the id of the initial trigger
  original_trigger_id = null;
  show_progress_on = null;
  component_prop_inputs = [];
  show_progress;
  functions;
  constructor(dep_config) {
    this.id = dep_config.id;
    this.original_trigger_id = dep_config.id;
    this.inputs = dep_config.inputs;
    this.outputs = dep_config.outputs;
    this.connection_type = dep_config.connection;
    this.show_progress = dep_config.show_progress;
    this.functions = {
      frontend: dep_config.js ? process_frontend_fn(
        dep_config.js,
        dep_config.backend_fn,
        dep_config.inputs.length,
        dep_config.outputs.length
      ) : void 0,
      backend: dep_config.backend_fn,
      backend_js: dep_config.js_implementation ? new AsyncFunction(
        `let result = await (${dep_config.js_implementation})(...arguments);
						return (!Array.isArray(result)) ? [result] : result;`
      ) : void 0
    };
    this.targets = dep_config.targets;
    this.cancels = dep_config.cancels;
    this.trigger_modes = dep_config.trigger_mode;
    this.show_progress_on = dep_config.show_progress_on || null;
    this.component_prop_inputs = dep_config.component_prop_inputs || [];
    for (let i = 0; i < dep_config.event_specific_args?.length || 0; i++) {
      const key = dep_config.event_specific_args[i];
      this.event_args[key] = dep_config[key] ?? null;
    }
  }
  async run(client, data_payload, event_data, target_id) {
    let _data_payload = data_payload;
    if (this.functions.backend_js) {
      const data = await this.functions.backend_js(..._data_payload);
      return { type: "data", data };
    }
    if (this.functions.frontend) {
      _data_payload = await this.functions.frontend(data_payload);
    }
    if (this.functions.backend) {
      return {
        type: "submit",
        data: client.submit(
          this.id,
          _data_payload,
          event_data,
          target_id,
          void 0,
          { "x-gradio-user": "app" }
        )
      };
    } else if (this.functions.frontend) {
      return { type: "data", data: _data_payload };
    }
    return { type: "void", data: null };
  }
  add_trigger(dep_id, condition) {
    this.triggers.push([dep_id, condition]);
  }
  get_triggers() {
    return {
      success: this.triggers.filter(([, condition]) => condition === "success").map(([id2]) => id2),
      failure: this.triggers.filter(([, condition]) => condition === "failure").map(([id2]) => id2),
      all: this.triggers.filter(([, condition]) => condition === "all").map(([id2]) => id2)
    };
  }
}
class DependencyManager {
  dependencies_by_fn = /* @__PURE__ */ new Map();
  dependencies_by_event = /* @__PURE__ */ new Map();
  render_id_deps = /* @__PURE__ */ new Map();
  submissions = /* @__PURE__ */ new Map();
  client;
  queue = /* @__PURE__ */ new Set();
  add_to_api_calls;
  update_state_cb;
  get_state_cb;
  rerender_cb;
  log_cb;
  on_connection_lost_cb;
  loading_stati = new LoadingStatus();
  connection_lost = false;
  constructor(dependencies, client, update_state_cb, get_state_cb, rerender_cb, log_cb, add_to_api_calls, on_connection_lost_cb) {
    this.add_to_api_calls = add_to_api_calls;
    this.log_cb = log_cb;
    this.update_state_cb = update_state_cb;
    this.get_state_cb = get_state_cb;
    this.rerender_cb = rerender_cb;
    this.on_connection_lost_cb = on_connection_lost_cb;
    this.client = client;
    this.reload(
      dependencies,
      update_state_cb,
      get_state_cb,
      rerender_cb,
      client
    );
  }
  reload(dependencies, update_state, get_state, rerender, client) {
    const { by_id, by_event } = this.create(dependencies);
    this.dependencies_by_event = by_event;
    this.dependencies_by_fn = by_id;
    this.client = client;
    this.update_state_cb = update_state;
    this.get_state_cb = get_state;
    this.rerender_cb = rerender;
    for (const [dep_id, dep] of this.dependencies_by_fn) {
      for (const [output_id] of dep.targets) {
        this.set_event_args(output_id, dep.event_args);
      }
    }
    this.register_loading_stati(by_id);
  }
  register_loading_stati(deps) {
    for (const [_, dep] of deps) {
      this.loading_stati.register(
        dep.id,
        dep.show_progress_on || dep.outputs,
        dep.inputs,
        dep.show_progress
      );
    }
  }
  clear_loading_status(component_id) {
    this.loading_stati.clear(component_id);
  }
  async update_loading_stati_state() {
    for (const [component_id, loading_status] of Object.entries(
      this.loading_stati.current
    )) {
      this.update_state_cb(
        Number(component_id),
        {
          loading_status
        },
        false
      );
    }
  }
  dispatch_state_change_events(result) {
    if (result.changed_state_ids) {
      for (const changed_id of result.changed_state_ids) {
        const change_dep = this.dependencies_by_event.get(
          "change-" + changed_id
        );
        change_dep?.forEach((dep) => {
          this.dispatch({
            type: "fn",
            fn_index: dep.id,
            target_id: changed_id,
            event_data: null
          });
        });
      }
    }
  }
  /** Dispatches an event to the appropriate dependency
   * @param event_name the name of the event
   * @param target_id the id of the component that triggered the event
   * @param event_data any additional data to pass to the dependency
   * @returns a value if there is no backend fn, a 'submission' if there is a backend fn, or null if there is no dependency
   */
  async dispatch(event_meta) {
    if (this.connection_lost) return;
    let deps;
    if (event_meta.type === "fn") {
      const dep = this.dependencies_by_fn.get(event_meta.fn_index);
      if (dep) deps = [dep];
    } else {
      deps = this.dependencies_by_event.get(
        `${event_meta.event_name}-${event_meta.target_id}`
      );
    }
    for (let i = 0; i < (deps?.length || 0); i++) {
      const dep = deps ? deps[i] : void 0;
      if (dep) {
        this.cancel(dep.cancels);
        const dispatch_status = should_dispatch(
          dep.trigger_modes,
          this.submissions.has(dep.id)
        );
        if (dispatch_status === "skip") {
          continue;
        } else if (dispatch_status === "defer") {
          this.queue.add(dep.id);
          continue;
        }
        if (dep.functions.backend) {
          this.loading_stati.update({
            status: "pending",
            fn_index: dep.id,
            stream_state: null
          });
          this.update_loading_stati_state();
        }
        const data_payload = await this.gather_state(
          dep.inputs,
          dep.component_prop_inputs
        );
        const unset_args = await Promise.all(
          dep.targets.map(
            ([output_id]) => this.set_event_args(output_id, dep.event_args)
          )
        );
        const { success, failure, all } = dep.get_triggers();
        try {
          let target_id = null;
          if (event_meta.target_id !== void 0 || event_meta.type === "event") {
            target_id = event_meta.target_id || null;
          } else {
            target_id = dep.original_trigger_id;
          }
          if (dep.connection_type === "stream" && this.submissions.has(dep.id)) {
            const submission = this.submissions.get(dep.id);
            let payload = {
              fn_index: dep.id,
              data: data_payload,
              event_data: event_meta.event_data
            };
            submission.send_chunk(payload);
            unset_args.forEach((fn) => fn());
            continue;
          }
          this.add_to_api_calls({
            fn_index: dep.id,
            data: data_payload,
            event_data: event_meta.event_data,
            trigger_id: target_id
          });
          const dep_submission = await dep.run(
            this.client,
            data_payload,
            event_meta.event_data,
            target_id
          );
          if (dep_submission.type === "void") {
            unset_args.forEach((fn) => fn());
          } else if (dep_submission.type === "data") {
            await this.handle_data(dep.outputs, dep_submission.data);
            unset_args.forEach((fn) => fn());
          } else {
            let stream_state = null;
            if (dep.connection_type === "stream" && !this.submissions.has(dep.id)) {
              stream_state = "waiting";
            }
            this.submissions.set(dep.id, dep_submission.data);
            let index = 0;
            submit_loop: for await (const result of dep_submission.data) {
              if (index === 0) {
                dep.inputs.forEach((input_id) => {
                  this.update_state_cb(
                    input_id,
                    {
                      loading_status: {
                        validation_error: null
                      }
                    },
                    false
                  );
                });
              }
              index += 1;
              if (result === null) continue;
              if (result.type === "data") {
                await this.handle_data(dep.outputs, result.data);
              }
              if (result.type === "status") {
                if (result.original_msg === "process_starts" && dep.connection_type === "stream") {
                  stream_state = "open";
                }
                const { fn_index, ...status } = result;
                if (result.stage === "complete") {
                  stream_state = "closed";
                  success.forEach((dep_id) => {
                    this.dispatch({
                      type: "fn",
                      fn_index: dep_id,
                      event_data: null,
                      target_id
                    });
                  });
                  this.dispatch_state_change_events(result);
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                  break submit_loop;
                } else if (result.stage === "generating") {
                  this.dispatch_state_change_events(result);
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                } else if (result.stage === "error") {
                  if (result.broken || result.session_not_found) {
                    if (!this.connection_lost) {
                      this.connection_lost = true;
                      this.on_connection_lost_cb();
                    }
                    this.loading_stati.update({
                      status: "complete",
                      fn_index: dep.id,
                      stream_state: null
                    });
                    this.update_loading_stati_state();
                    break submit_loop;
                  }
                  if (Array.isArray(result?.message)) {
                    result.message.forEach((m, i2) => {
                      this.update_state_cb(
                        dep.inputs[i2],
                        {
                          loading_status: {
                            validation_error: !m.is_valid ? m.message : null,
                            show_validation_error: true
                          }
                        },
                        false
                      );
                    });
                    dep.outputs.forEach((output_id) => {
                      if (dep.inputs.includes(output_id)) return;
                      this.update_state_cb(
                        output_id,
                        {
                          loading_status: {
                            status: null
                          }
                        },
                        false
                      );
                    });
                    unset_args.forEach((fn) => fn());
                    this.submissions.delete(dep.id);
                    if (this.queue.has(dep.id)) {
                      this.queue.delete(dep.id);
                      this.dispatch(event_meta);
                    }
                    return;
                  }
                  const _message = result?.message?.replace(
                    MESSAGE_QUOTE_RE,
                    (_, b) => b
                  );
                  this.log_cb(
                    //@ts-ignore
                    result?._title ?? "Error",
                    _message || "",
                    fn_index,
                    "error",
                    status.duration,
                    status.visible
                  );
                  throw new Error("Dependency function failed");
                } else {
                  this.loading_stati.update({
                    ...status,
                    status: status.stage,
                    fn_index: dep.id,
                    stream_state
                  });
                  this.update_loading_stati_state();
                }
              }
              if (result.type === "render") {
                this.loading_stati.update({
                  status: "complete",
                  fn_index: dep.id,
                  stream_state: null
                });
                this.update_loading_stati_state();
                const { layout, components, render_id, dependencies } = result.data;
                this.rerender_cb(components, layout);
                const { by_id, by_event } = this.create(
                  dependencies
                );
                this.register_loading_stati(by_id);
                by_id.forEach(
                  (dep2) => this.dependencies_by_fn.set(dep2.id, dep2)
                );
                by_event.forEach(
                  (dep2, key) => this.dependencies_by_event.set(key, dep2)
                );
                const current_deps = this.render_id_deps.get(render_id);
                if (current_deps) {
                  current_deps.forEach((old_dep_id) => {
                    if (!by_id.has(old_dep_id)) {
                      this.dependencies_by_fn.delete(old_dep_id);
                    }
                  });
                }
                this.render_id_deps.set(
                  render_id,
                  new Set(Array.from(by_id.keys()))
                );
                this.register_loading_stati(by_id);
                break submit_loop;
              }
              if (result.type === "log") {
                this.handle_log(result);
              }
            }
            all.forEach((dep_id) => {
              this.dispatch({
                type: "fn",
                fn_index: dep_id,
                event_data: null,
                target_id
              });
            });
            unset_args.forEach((fn) => fn());
            this.submissions.delete(dep.id);
            if (this.queue.has(dep.id)) {
              this.queue.delete(dep.id);
              this.dispatch(event_meta);
            }
          }
        } catch (error) {
          this.loading_stati.update({
            status: "error",
            fn_index: dep.id,
            eta: 0,
            queue: false,
            stream_state: null
          });
          this.update_loading_stati_state();
          this.submissions.delete(dep.id);
          failure.forEach((dep_id) => {
            this.dispatch({
              type: "fn",
              fn_index: dep_id,
              event_data: null
            });
          });
        }
      }
    }
    return;
  }
  /**
   *  Creates a map of dependencies for easy lookup
   *
   * @param dependencies the list of dependencies from the backend
   * @returns a map of dependencies keyed by `${event_name}-${target_id}`
   * */
  create(dependencies) {
    const _deps_by_id = /* @__PURE__ */ new Map();
    const _deps_by_event = /* @__PURE__ */ new Map();
    const then_triggers = [];
    for (const dep_config of dependencies) {
      const dependency = new Dependency(dep_config);
      for (const [target_id, event_name] of dep_config.targets) {
        if (!_deps_by_event.has(`${event_name}-${target_id}`)) {
          _deps_by_event.set(`${event_name}-${target_id}`, []);
        }
        _deps_by_event.get(`${event_name}-${target_id}`)?.push(dependency);
      }
      _deps_by_id.set(dep_config.id, dependency);
      if (dep_config.trigger_after !== void 0) {
        const then_mode = dep_config.trigger_only_on_failure ? "failure" : dep_config.trigger_only_on_success ? "success" : "all";
        then_triggers.push([
          dep_config.id,
          dep_config.trigger_after,
          then_mode
        ]);
      }
    }
    for (const [dep_id, trigger_after, condition] of then_triggers) {
      const dependency = _deps_by_id.get(trigger_after);
      if (dependency) {
        dependency.add_trigger(dep_id, condition);
        dependency.original_trigger_id = walk_after_to_original(
          dependencies,
          trigger_after
        );
      }
    }
    return { by_id: _deps_by_id, by_event: _deps_by_event };
  }
  handle_log(msg) {
    const { title, log, fn_index, level, duration, visible } = msg;
    this.log_cb(title, log, fn_index, level, duration, visible);
  }
  /**
   *  Updates the state of the outputs based on the data received from the dependency
   *
   * @param outputs the ids of the output components
   * @param data the data to update the components with
   * */
  async handle_data(outputs, data) {
    await Promise.all(
      outputs.map(async (output_id, i) => {
        const _data = data[i] === void 0 ? NOVALUE : data[i];
        if (_data === NOVALUE) return;
        if (is_prop_update(_data)) {
          let pending_visibility_update = false;
          let pending_visibility_value = null;
          for (const [update_key, update_value] of Object.entries(_data)) {
            if (update_key === "__type__") continue;
            if (update_key === "visible") {
              pending_visibility_update = true;
              pending_visibility_value = update_value;
              continue;
            }
            await this.update_state_cb(
              outputs[i],
              {
                [update_key]: update_value
              },
              false
            );
          }
          if (pending_visibility_update) {
            await this.update_state_cb(
              outputs[i],
              {
                visible: pending_visibility_value
              },
              true
            );
          }
        } else {
          await this.update_state_cb(output_id, { value: _data }, false);
        }
      })
    );
  }
  /**
   * Gathers the current state of the inputs
   *
   * @param ids the ids of the components to gather state from
   * @param prop_indices the indices (relative to ids array) that should return all component props instead of just the value
   * @returns an array of the current state of the components, in the same order as the ids
   */
  async gather_state(ids, prop_indices = []) {
    return (await Promise.all(ids.map((id2) => this.get_state_cb(id2)))).map(
      (state, index) => {
        if (prop_indices.includes(index)) {
          return state ?? null;
        }
        return state?.value ?? null;
      }
    );
  }
  /** Sets the event arguments for a specific component
   *
   * @param id the id of the component to set the event arguments for
   * @param args the event arguments to set
   * @returns a function that can be called to reset the event arguments to their previous values
   */
  async set_event_args(id2, args) {
    let current_args = {};
    const current_state = await this.get_state_cb?.(id2);
    if (!current_state) return () => {
    };
    for (const [key] of Object.entries(args)) {
      current_args[key] = current_state?.[key] ?? null;
    }
    if (Object.keys(args).length === 0) {
      return () => {
      };
    }
    await this.update_state_cb(id2, args, false);
    return () => {
      this.update_state_cb(id2, current_args, false);
    };
  }
  async cancel(ids) {
    if (!ids) return;
    for (const id2 of ids) {
      const submission = this.submissions.get(id2);
      if (submission) {
        await submission.cancel();
        this.loading_stati.update({
          status: "complete",
          fn_index: id2,
          eta: 0,
          queue: false,
          stream_state: null
        });
        this.update_loading_stati_state();
        this.submissions.delete(id2);
        const { failure, all } = this.dependencies_by_fn.get(id2)?.get_triggers() || { failure: [], all: [] };
        failure.forEach((dep_id) => {
          this.dispatch({
            type: "fn",
            fn_index: dep_id,
            event_data: null,
            target_id: id2
          });
        });
        all.forEach((dep_id) => {
          this.dispatch({
            type: "fn",
            fn_index: dep_id,
            event_data: null,
            target_id: id2
          });
        });
      }
    }
  }
  dispatch_load_events() {
    this.dependencies_by_fn.forEach((dep) => {
      dep.targets.forEach(([target_id, event_name]) => {
        if (event_name === "load") {
          this.dispatch({
            type: "fn",
            fn_index: dep.id,
            event_data: null,
            target_id
          });
        }
      });
    });
  }
  get_fns_from_targets(target_id) {
    const fn_indices = [];
    this.dependencies_by_event.forEach((deps, key) => {
      const [, dep_target_id] = key.split("-");
      if (Number(dep_target_id) === target_id) {
        deps.forEach((dep) => {
          fn_indices.push(dep.id);
        });
      }
    });
    return fn_indices;
  }
  close_stream(id2) {
    const fn_ids = this.get_fns_from_targets(id2);
    for (const fn_id of fn_ids) {
      const submission = this.submissions.get(fn_id);
      if (submission) {
        submission.close_stream();
        this.submissions.delete(fn_id);
      }
      this.loading_stati.update({
        status: "complete",
        fn_index: fn_id,
        eta: 0,
        queue: false,
        stream_state: "closed"
      });
    }
    this.update_loading_stati_state();
  }
}
function is_prop_update(payload) {
  return typeof payload === "object" && payload !== null && "__type__" in payload && payload?.__type__ === "update";
}
function should_dispatch(mode, is_running) {
  if (!is_running) return "run";
  if (mode === "always_last") {
    return "defer";
  } else if (mode === "multiple") {
    return "run";
  } else if (mode === "once") {
    return "skip";
  }
  return "run";
}
function process_frontend_fn(source, backend_fn, input_length, output_length) {
  const wrap = backend_fn ? input_length === 1 : output_length === 1;
  try {
    return new AsyncFunction(
      "__fn_args",
      `  let result = await (${source})(...__fn_args);
  if (typeof result === "undefined") return [];
  return (${wrap} && !Array.isArray(result)) ? [result] : result;`
    );
  } catch (e) {
    throw e;
  }
}
function walk_after_to_original(dependency_map, dep_id) {
  let cache = /* @__PURE__ */ new Map();
  let current_id = dep_id;
  let safety_counter = 0;
  while (safety_counter < 100) {
    const dep = cache.get(current_id) || dependency_map.find((d) => d.id === current_id);
    if (!dep) break;
    cache.set(dep.id, dep);
    if (dep.trigger_after === null || dep.trigger_after === void 0) break;
    current_id = dep.trigger_after;
    safety_counter += 1;
  }
  return current_id;
}
function Blocks($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let {
      root,
      components,
      layout,
      dependencies,
      title,
      target,
      autoscroll,
      footer_links,
      control_page_title,
      app_mode,
      theme_mode,
      app,
      space_id,
      version,
      js,
      fill_height,
      username,
      api_prefix,
      max_file_size,
      initial_layout,
      css,
      vibe_mode,
      search_params,
      render_complete = false,
      ready = false,
      reload_count = 0,
      add_new_message = void 0
    } = $$props;
    components.forEach((comp) => {
      if (!comp.props.i18n) {
        comp.props.i18n = store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter);
      }
    });
    let messages = [];
    let reconnect_interval = null;
    function gradio_event_dispatcher(id2, event, data) {
      if (event === "share") {
        const { title: title2, description } = data;
      } else if (event === "error") {
        new_message("Error", data, -1, event, 10, true);
      } else if (event === "warning") {
        new_message("Warning", data, -1, event, 10, true);
      } else if (event === "info") {
        new_message("Info", data, -1, event, 10, true);
      } else if (event === "gradio_expand" || event === "gradio_tab_select") {
        const id_ = event === "gradio_expand" ? id2 : data.component_id;
        app_tree.render_previously_invisible_children(id_);
      } else if (event == "clear_status") {
        app_tree.update_state(id2, { loading_status: {} }, false);
        dep_manager.clear_loading_status(id2);
      } else if (event == "close_stream") {
        dep_manager.close_stream(id2);
      } else if (event === "custom_button_click") {
        const button_id = data.id;
        dispatch_to_target(button_id, "click", null);
      } else {
        if (event === "select" && id2 in app_tree.initial_tabs) {
          id2 = data.id;
        }
        dep_manager.dispatch({
          type: "event",
          event_name: event,
          target_id: id2,
          event_data: data
        });
      }
    }
    let app_tree = new AppTree(
      components,
      layout,
      dependencies,
      {
        root,
        theme: theme_mode,
        version,
        api_prefix,
        max_file_size,
        autoscroll,
        fill_height
      },
      app,
      store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter),
      gradio_event_dispatcher
    );
    function dispatch_to_target(target_id, event, data) {
      dep_manager.dispatch({
        type: "event",
        event_name: event,
        target_id,
        event_data: data
      });
    }
    let api_calls = [];
    let last_api_call = null;
    let add_to_api_calls = (payload) => {
      last_api_call = payload;
      if (!api_recorder_visible) return;
      api_calls = [...api_calls, last_api_call];
    };
    function handle_connection_lost() {
      messages = messages.filter((m) => m.type !== "error");
      ++_error_id;
      messages.push({
        title: "Connection Lost",
        message: LOST_CONNECTION_MESSAGE,
        fn_index: -1,
        type: "error",
        id: _error_id,
        duration: null,
        visible: true
      });
      reconnect_interval = setInterval(
        async () => {
          try {
            const status = await app.reconnect();
            if (status === "connected" || status === "changed") {
              clearInterval(reconnect_interval);
              reconnect_interval = null;
              window.location.reload();
            }
          } catch (e) {
            console.debug(e);
          }
        },
        2e3
      );
    }
    let dep_manager = new DependencyManager(dependencies, app, app_tree.update_state.bind(app_tree), app_tree.get_state.bind(app_tree), app_tree.rerender.bind(app_tree), new_message, add_to_api_calls, handle_connection_lost);
    let vibe_editor_width = 350;
    let api_docs_visible = search_params.get("view") === "api" && footer_links.includes("api");
    let settings_visible = search_params.get("view") === "settings";
    let api_recorder_visible = search_params.get("view") === "api-recorder" && footer_links.includes("api");
    let ApiDocs = null;
    let Settings = null;
    function new_message(title2, message, fn_index, type, duration = 10, visible = false) {
      if (!visible) return;
      messages.push({
        title: title2,
        message,
        fn_index,
        type,
        id: ++_error_id,
        duration,
        visible
      });
    }
    add_new_message = new_message;
    let _error_id = -1;
    store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("blocks.long_requests_queue");
    store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("blocks.connection_can_break");
    const LOST_CONNECTION_MESSAGE = "Connection to the server was lost. Attempting reconnection...";
    store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("blocks.waiting_for_inputs");
    let is_screen_recording = writable(false);
    function handle_close(id2) {
      messages = messages.filter((m) => m.id !== id2);
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      head("zxu34v", $$renderer3, ($$renderer4) => {
        if (control_page_title) {
          $$renderer4.push("<!--[-->");
          $$renderer4.title(($$renderer5) => {
            $$renderer5.push(`<title>${escape_html(title)}</title>`);
          });
        } else {
          $$renderer4.push("<!--[!-->");
        }
        $$renderer4.push(`<!--]--> `);
        if (css) {
          $$renderer4.push("<!--[-->");
          $$renderer4.push(`${html(`<style>${prefix_css(css, version)}</style>`)}`);
        } else {
          $$renderer4.push("<!--[!-->");
        }
        $$renderer4.push(`<!--]-->`);
      });
      $$renderer3.push(`<div class="wrap svelte-zxu34v"${attr_style("", { "min-height": app_mode ? "100%" : "auto" })}><main class="contain svelte-zxu34v"${attr_style("", {
        "flex-grow": app_mode ? "1" : "auto",
        "margin-right": vibe_mode ? `${vibe_editor_width}px` : "0"
      })}>`);
      MountComponents($$renderer3, { node: app_tree.root });
      $$renderer3.push(`<!----></main> `);
      if (footer_links.length > 0) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<footer aria-label="Gradio footer navigation" class="svelte-zxu34v">`);
        if (footer_links.includes("api")) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button class="show-api svelte-zxu34v">`);
          if (app.config?.mcp_server) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`${escape_html(store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("errors.use_via_api_or_mcp"))}`);
          } else {
            $$renderer3.push("<!--[!-->");
            $$renderer3.push(`${escape_html(store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("errors.use_via_api"))}`);
          }
          $$renderer3.push(`<!--]--> <img${attr("src", api_logo)}${attr("alt", store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.logo"))} class="svelte-zxu34v"/></button>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (footer_links.includes("gradio")) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="divider show-api-divider svelte-zxu34v">·</div> <a href="https://gradio.app" class="built-with svelte-zxu34v" target="_blank" rel="noreferrer">${escape_html(store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.built_with_gradio"))} <img${attr("src", logo)}${attr("alt", store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.logo"))} class="svelte-zxu34v"/></a>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <button${attr_class("record svelte-zxu34v", void 0, {
          "hidden": !store_get($$store_subs ??= {}, "$is_screen_recording", is_screen_recording)
        })}>${escape_html(store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.stop_recording"))} <img${attr("src", record_stop)}${attr("alt", store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.stop_recording"))} class="svelte-zxu34v"/></button> <div class="divider svelte-zxu34v">·</div> `);
        if (footer_links.includes("settings")) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div${attr_class("divider svelte-zxu34v", void 0, {
            "hidden": !store_get($$store_subs ??= {}, "$is_screen_recording", is_screen_recording)
          })}>·</div> <button class="settings svelte-zxu34v">${escape_html(store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.settings"))} <img${attr("src", settings_logo)}${attr("alt", store_get($$store_subs ??= {}, "$reactive_formatter", reactive_formatter)("common.settings"))} class="svelte-zxu34v"/></button>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></footer>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (api_docs_visible && app_tree.root && ApiDocs) ;
      else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (settings_visible && app.config && app_tree.root && Settings) ;
      else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div> `);
      if (messages) {
        $$renderer3.push("<!--[-->");
        Toast($$renderer3, { messages, on_close: handle_close });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, { ready, reload_count, add_new_message });
  });
}
function Login($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let root = $$props["root"];
    let auth_message = $$props["auth_message"];
    let app_mode = $$props["app_mode"];
    let space_id = $$props["space_id"];
    let i18n = $$props["i18n"];
    let username = "";
    let password = "";
    let incorrect_credentials = false;
    const submit = async () => {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("password", password);
      const login_url = new URL("login", root.endsWith("/") ? root : root + "/").href;
      let response = await fetch(login_url, { method: "POST", body: formData });
      if (response.status === 400) {
        incorrect_credentials = true;
        username = "";
        password = "";
      } else if (response.status == 200) {
        location.reload();
      }
    };
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div${attr_class("wrap svelte-14xt79u", void 0, { "min-h-screen": app_mode })}>`);
      BaseColumn($$renderer3, {
        variant: "panel",
        min_width: 480,
        children: ($$renderer4) => {
          $$renderer4.push(`<h2 class="svelte-14xt79u">${escape_html(i18n("login.login"))}</h2> `);
          if (auth_message) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<p class="auth svelte-14xt79u">${html(auth_message)}</p>`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (space_id) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<p class="auth svelte-14xt79u">${escape_html(i18n("login.enable_cookies"))}</p>`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (incorrect_credentials) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<p class="creds svelte-14xt79u">${escape_html(i18n("login.incorrect_credentials"))}</p>`);
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          BaseForm($$renderer4, {
            children: ($$renderer5) => {
              Block($$renderer5, {
                children: ($$renderer6) => {
                  Textbox($$renderer6, {
                    label: i18n("login.username"),
                    lines: 1,
                    show_label: true,
                    max_lines: 1,
                    onsubmit: submit,
                    get value() {
                      return username;
                    },
                    set value($$value) {
                      username = $$value;
                      $$settled = false;
                    }
                  });
                },
                $$slots: { default: true }
              });
              $$renderer5.push(`<!----> `);
              Block($$renderer5, {
                children: ($$renderer6) => {
                  Textbox($$renderer6, {
                    label: i18n("login.password"),
                    lines: 1,
                    show_label: true,
                    max_lines: 1,
                    type: "password",
                    onsubmit: submit,
                    get value() {
                      return password;
                    },
                    set value($$value) {
                      password = $$value;
                      $$settled = false;
                    }
                  });
                },
                $$slots: { default: true }
              });
              $$renderer5.push(`<!---->`);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Button($$renderer4, {
            size: "lg",
            variant: "primary",
            onclick: submit,
            children: ($$renderer5) => {
              $$renderer5.push(`<!---->${escape_html(i18n("login.login"))}`);
            }
          });
          $$renderer4.push(`<!---->`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { root, auth_message, app_mode, space_id, i18n });
  });
}
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let {
      data,
      autoscroll = false,
      version = "6-11-0",
      initial_height,
      app_mode = true,
      is_embed = false,
      theme_mode = null,
      control_page_title = true,
      container,
      space
    } = $$props;
    function handle_theme_mode(target) {
      let new_theme_mode;
      const url = new URL(window.location.toString());
      const url_color_mode = url.searchParams.get("__theme");
      new_theme_mode = theme_mode || url_color_mode || "system";
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
      const dark_class_element = is_embed ? target.parentElement : document.body;
      const bg_element = is_embed ? target : target.parentElement;
      bg_element.style.background = "var(--body-background-fill)";
      dark_class_element.classList.add("theme-loaded");
      if (theme === "dark") {
        dark_class_element.classList.add("dark");
      } else {
        dark_class_element.classList.remove("dark");
      }
    }
    let active_theme_mode = void 0;
    if (browser) {
      active_theme_mode = handle_theme_mode(document.body);
    }
    let loader_status = "complete";
    let wrapper = void 0;
    let ready = false;
    let render_complete = false;
    ({ subscribe: writable({}).subscribe });
    let app = data.app;
    let pending_deep_link_error = false;
    if (browser && data.config) {
      window.gradio_config = data.config;
    }
    let new_message_fn = void 0;
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
    let config = data.config;
    let root = (() => {
      if (!browser) return config.root;
      const current_url = new URL(window.location.toString());
      const root_url = new URL(config.root);
      return new URL(root_url.pathname, current_url).toString();
    })();
    run(() => {
      if (config?.app_id) {
        config.app_id;
      }
    });
    run(() => {
      if (new_message_fn && pending_deep_link_error) {
        new_message_fn("Error", "Deep link was not valid", -1, "error", 10, true);
        pending_deep_link_error = false;
      }
    });
    run(() => {
      if (render_complete) {
        wrapper.dispatchEvent(new CustomEvent("render", { bubbles: true, cancelable: false, composed: true }));
      }
    });
    run(() => {
      app?.config && browser && mount_space_header(app?.config?.space_id, is_embed);
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      head("1gmqcbx", $$renderer3, ($$renderer4) => {
        $$renderer4.push(`<link rel="stylesheet"${attr("href", root + "/theme.css?v=" + config?.theme_hash)}/> `);
        if (config?.stylesheets) {
          $$renderer4.push("<!--[-->");
          $$renderer4.push(`<!--[-->`);
          const each_array = ensure_array_like(config.stylesheets);
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let stylesheet = each_array[$$index];
            if (stylesheet.startsWith("http:") || stylesheet.startsWith("https:")) {
              $$renderer4.push("<!--[-->");
              $$renderer4.push(`<link rel="stylesheet"${attr("href", stylesheet)}/>`);
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
          $$renderer4.push(`<!--]-->`);
        } else {
          $$renderer4.push("<!--[!-->");
        }
        $$renderer4.push(`<!--]--> <link rel="manifest" href="/manifest.json"/>`);
      });
      Embed($$renderer3, {
        display: container && is_embed,
        is_embed,
        info: false,
        version,
        initial_height,
        space,
        pages: config.pages,
        current_page: config.current_page,
        root: config.root,
        components: config.components,
        loaded: loader_status === "complete",
        fill_width: config?.fill_width || false,
        get wrapper() {
          return wrapper;
        },
        set wrapper($$value) {
          wrapper = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          if (config?.auth_required) {
            $$renderer4.push("<!--[-->");
            Login($$renderer4, {
              auth_message: config.auth_message,
              root: config.root,
              space_id: space,
              app_mode,
              i18n: store_get($$store_subs ??= {}, "$_", $format)
            });
          } else {
            $$renderer4.push("<!--[!-->");
            if (config && app) {
              $$renderer4.push("<!--[-->");
              Blocks($$renderer4, spread_props([
                { app },
                config,
                {
                  fill_height: !is_embed && config.fill_height,
                  theme_mode: active_theme_mode,
                  control_page_title,
                  target: wrapper,
                  autoscroll,
                  footer_links: is_embed ? [] : config.footer_links,
                  app_mode,
                  version,
                  search_params: store_get($$store_subs ??= {}, "$page", page).url.searchParams,
                  initial_layout: data.layout,
                  get ready() {
                    return ready;
                  },
                  set ready($$value) {
                    ready = $$value;
                    $$settled = false;
                  },
                  get render_complete() {
                    return render_complete;
                  },
                  set render_complete($$value) {
                    render_complete = $$value;
                    $$settled = false;
                  },
                  get add_new_message() {
                    return new_message_fn;
                  },
                  set add_new_message($$value) {
                    new_message_fn = $$value;
                    $$settled = false;
                  }
                }
              ]));
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}

export { _page as default };
//# sourceMappingURL=_page.svelte-pMWlccma.js.map
