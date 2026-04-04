import { D as DEV, B as BROWSER } from './dev-fallback-B-RpELjM.js';
import { n as noop } from './async-D55cHugf.js';
import { w as writable } from './index-Cg-Pg6j3.js';
import { a as assets, v as version } from './exports-Dz6pzFz_.js';

function notifiable_store(value) {
  const store = writable(value);
  let ready = true;
  function notify() {
    ready = true;
    store.update((val) => val);
  }
  function set(new_value) {
    ready = false;
    store.set(new_value);
  }
  function subscribe(run) {
    let old_value;
    return store.subscribe((new_value) => {
      if (old_value === void 0 || ready && new_value !== old_value) {
        run(old_value = new_value);
      }
    });
  }
  return { notify, set, subscribe };
}
const updated_listener = {
  v: () => {
  }
};
function create_updated_store() {
  const { set, subscribe } = writable(false);
  if (DEV || !BROWSER) {
    return {
      subscribe,
      // eslint-disable-next-line @typescript-eslint/require-await
      check: async () => false
    };
  }
  let timeout;
  async function check() {
    clearTimeout(timeout);
    try {
      const res = await fetch(`${assets}/${"_app/version.json"}`, {
        headers: {
          pragma: "no-cache",
          "cache-control": "no-cache"
        }
      });
      if (!res.ok) {
        return false;
      }
      const data = await res.json();
      const updated2 = data.version !== version;
      if (updated2) {
        set(true);
        updated_listener.v();
        clearTimeout(timeout);
      }
      return updated2;
    } catch {
      return false;
    }
  }
  return {
    subscribe,
    check
  };
}
let page;
let updated;
const is_legacy = noop.toString().includes("$$") || /function \w+\(\) \{\}/.test(noop.toString());
if (is_legacy) {
  page = {
    data: {},
    form: null,
    error: null,
    params: {},
    route: { id: null },
    state: {},
    status: -1,
    url: new URL("https://example.com")
  };
  updated = { current: false };
} else {
  page = new class Page {
    data = {};
    form = null;
    error = null;
    params = {};
    route = { id: null };
    state = {};
    status = -1;
    url = new URL("https://example.com");
  }();
  updated = new class Updated {
    current = false;
  }();
  updated_listener.v = () => updated.current = true;
}
let loading = 0;
const native_fetch = BROWSER ? window.fetch : (
  /** @type {any} */
  (() => {
  })
);
if (DEV && BROWSER) {
  let can_inspect_stack_trace = false;
  const check_stack_trace = async () => {
    const stack = (
      /** @type {string} */
      new Error().stack
    );
    can_inspect_stack_trace = stack.includes("check_stack_trace");
  };
  void check_stack_trace();
  window.fetch = (input, init) => {
    const url = input instanceof Request ? input.url : input.toString();
    const stack_array = (
      /** @type {string} */
      new Error().stack.split("\n")
    );
    const cutoff = stack_array.findIndex((a) => a.includes("load@") || a.includes("at load"));
    const stack = stack_array.slice(0, cutoff + 2).join("\n");
    const in_load_heuristic = can_inspect_stack_trace ? stack.includes("src/runtime/client/client.js") : loading;
    const used_kit_fetch = init?.__sveltekit_fetch__;
    if (in_load_heuristic && !used_kit_fetch) {
      console.warn(
        `Loading ${url} using \`window.fetch\`. For best results, use the \`fetch\` that is passed to your \`load\` function: https://svelte.dev/docs/kit/load#making-fetch-requests`
      );
    }
    const method = input instanceof Request ? input.method : init?.method || "GET";
    if (method !== "GET") {
      cache.delete(build_selector(input));
    }
    return native_fetch(input, init);
  };
} else if (BROWSER) {
  window.fetch = (input, init) => {
    const method = input instanceof Request ? input.method : init?.method || "GET";
    if (method !== "GET") {
      cache.delete(build_selector(input));
    }
    return native_fetch(input, init);
  };
}
const cache = /* @__PURE__ */ new Map();
function build_selector(resource, opts) {
  const url = JSON.stringify(resource instanceof Request ? resource.url : resource);
  let selector = `script[data-sveltekit-fetched][data-url=${url}]`;
  return selector;
}
if (DEV && BROWSER) {
  let warned = false;
  const current_module_url = import.meta.url.split("?")[0];
  const warn = () => {
    if (warned) return;
    let stack = new Error().stack?.split("\n");
    if (!stack) return;
    if (!stack[0].includes("https:") && !stack[0].includes("http:")) stack = stack.slice(1);
    stack = stack.slice(2);
    if (stack[0]?.includes(current_module_url)) return;
    warned = true;
    console.warn(
      "Avoid using `history.pushState(...)` and `history.replaceState(...)` as these will conflict with SvelteKit's router. Use the `pushState` and `replaceState` imports from `$app/navigation` instead."
    );
  };
  const push_state = history.pushState;
  history.pushState = (...args) => {
    warn();
    return push_state.apply(history, args);
  };
  const replace_state = history.replaceState;
  history.replaceState = (...args) => {
    warn();
    return replace_state.apply(history, args);
  };
}
const stores = {
  page: /* @__PURE__ */ notifiable_store({}),
  navigating: /* @__PURE__ */ writable(
    /** @type {import('@sveltejs/kit').Navigation | null} */
    null
  ),
  updated: /* @__PURE__ */ create_updated_store()
};
if (DEV) {
  const console_warn = console.warn;
  console.warn = function warn(...args) {
    if (args.length === 1 && /<(Layout|Page|Error)(_[\w$]+)?> was created (with unknown|without expected) prop '(data|form)'/.test(
      args[0]
    )) {
      return;
    }
    console_warn(...args);
  };
}

export { page as p, stores as s };
//# sourceMappingURL=client-sDHYjkUT.js.map
