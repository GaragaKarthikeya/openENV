const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./Index-BRJ0gETR.js","./i18n-CMo5lpzy.js","./i18n-CGFL4PZH.css","./index-Lz7fsHFt.js","./StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js","./html-CdUEtR5E.js","./ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js","./snippet-Bl5YVoCg.js","./ScrollFade-BXTrLZiw.css","./index-C7e-J7CF.js","./StreamingBar-M0OJiclA.css","./Clear-Dgs5UhTO.js","./Index-D1pHiZ3X.css","./index-BJk8VOjP.js","./index-DnoGeqVF.js","./utils.svelte-CYLB_jIA.js","./clone-DaB-S2nH.js"])))=>i.map(i=>d[i]);
true              &&(function polyfill() {
	const relList = document.createElement("link").relList;
	if (relList && relList.supports && relList.supports("modulepreload")) return;
	for (const link of document.querySelectorAll("link[rel=\"modulepreload\"]")) processPreload(link);
	new MutationObserver((mutations) => {
		for (const mutation of mutations) {
			if (mutation.type !== "childList") continue;
			for (const node of mutation.addedNodes) if (node.tagName === "LINK" && node.rel === "modulepreload") processPreload(node);
		}
	}).observe(document, {
		childList: true,
		subtree: true
	});
	function getFetchOpts(link) {
		const fetchOpts = {};
		if (link.integrity) fetchOpts.integrity = link.integrity;
		if (link.referrerPolicy) fetchOpts.referrerPolicy = link.referrerPolicy;
		if (link.crossOrigin === "use-credentials") fetchOpts.credentials = "include";
		else if (link.crossOrigin === "anonymous") fetchOpts.credentials = "omit";
		else fetchOpts.credentials = "same-origin";
		return fetchOpts;
	}
	function processPreload(link) {
		if (link.ep) return;
		link.ep = true;
		const fetchOpts = getFetchOpts(link);
		fetch(link.href, fetchOpts);
	}
}());

const scriptRel = 'modulepreload';const assetsURL = function(dep, importerUrl) { return new URL(dep, importerUrl).href };const seen = {};const __vitePreload = function preload(baseModule, deps, importerUrl) {
	let promise = Promise.resolve();
	if (true               && deps && deps.length > 0) {
		const links = document.getElementsByTagName("link");
		const cspNonceMeta = document.querySelector("meta[property=csp-nonce]");
		const cspNonce = cspNonceMeta?.nonce || cspNonceMeta?.getAttribute("nonce");
		function allSettled(promises$2) {
			return Promise.all(promises$2.map((p) => Promise.resolve(p).then((value$1) => ({
				status: "fulfilled",
				value: value$1
			}), (reason) => ({
				status: "rejected",
				reason
			}))));
		}
		promise = allSettled(deps.map((dep) => {
			dep = assetsURL(dep, importerUrl);
			if (dep in seen) return;
			seen[dep] = true;
			const isCss = dep.endsWith(".css");
			const cssSelector = isCss ? "[rel=\"stylesheet\"]" : "";
			if (!!importerUrl) for (let i$1 = links.length - 1; i$1 >= 0; i$1--) {
				const link$1 = links[i$1];
				if (link$1.href === dep && (!isCss || link$1.rel === "stylesheet")) return;
			}
			else if (document.querySelector(`link[href="${dep}"]${cssSelector}`)) return;
			const link = document.createElement("link");
			link.rel = isCss ? "stylesheet" : scriptRel;
			if (!isCss) link.as = "script";
			link.crossOrigin = "";
			link.href = dep;
			if (cspNonce) link.setAttribute("nonce", cspNonce);
			document.head.appendChild(link);
			if (isCss) return new Promise((res, rej) => {
				link.addEventListener("load", res);
				link.addEventListener("error", () => rej(/* @__PURE__ */ new Error(`Unable to preload CSS for ${dep}`)));
			});
		}));
	}
	function handlePreloadError(err$2) {
		const e$1 = new Event("vite:preloadError", { cancelable: true });
		e$1.payload = err$2;
		window.dispatchEvent(e$1);
		if (!e$1.defaultPrevented) throw err$2;
	}
	return promise.then((res) => {
		for (const item of res || []) {
			if (item.status !== "rejected") continue;
			handlePreloadError(item.reason);
		}
		return baseModule().catch(handlePreloadError);
	});
};

const HOST_URL = `host`;
const SSE_URL = `queue/data`;
const SSE_DATA_URL = `queue/join`;
const UPLOAD_URL = `upload`;
const LOGIN_URL = `login`;
const CONFIG_URL = `config`;
const API_INFO_URL = `info`;
const RUNTIME_URL = `runtime`;
const SLEEPTIME_URL = `sleeptime`;
const HEARTBEAT_URL = `heartbeat`;
const COMPONENT_SERVER_URL = `component_server`;
const RESET_URL = `reset`;
const CANCEL_URL = `cancel`;
const APP_ID_URL = `app_id`;
const QUEUE_FULL_MSG = "This application is currently busy. Please try again. ";
const BROKEN_CONNECTION_MSG = "Connection errored out. ";
const CONFIG_ERROR_MSG = "Could not resolve app config. ";
const SPACE_STATUS_ERROR_MSG = "Could not get space status. ";
const API_INFO_ERROR_MSG = "Could not get API info. ";
const SPACE_METADATA_ERROR_MSG = "Space metadata could not be loaded. ";
const INVALID_URL_MSG = "Invalid URL. A full URL path is required.";
const UNAUTHORIZED_MSG = "Not authorized to access this space. ";
const INVALID_CREDENTIALS_MSG = "Invalid credentials. Could not login. ";
const MISSING_CREDENTIALS_MSG = "Login credentials are required to access this space.";
const NODEJS_FS_ERROR_MSG = "File system access is only available in Node.js environments";
const ROOT_URL_ERROR_MSG = "Root URL not found in client config";
const FILE_PROCESSING_ERROR_MSG = "Error uploading file";

async function get_jwt(space, token, cookies) {
  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${space}/jwt`, {
      headers: {
        Authorization: `Bearer ${token}`,
        ...cookies ? { Cookie: cookies } : {}
      }
    });
    const jwt = (await r.json()).token;
    return jwt || false;
  } catch (e) {
    return false;
  }
}
function map_names_to_ids(fns) {
  let apis = {};
  fns.forEach(({ api_name, id }) => {
    if (api_name) apis[api_name] = id;
  });
  return apis;
}
async function resolve_config(endpoint) {
  const headers = this.options.token ? { Authorization: `Bearer ${this.options.token}` } : {};
  headers["Content-Type"] = "application/json";
  if (typeof window !== "undefined" && window.gradio_config && location.origin !== "http://localhost:9876") {
    if (window.gradio_config.current_page) {
      endpoint = endpoint.substring(0, endpoint.lastIndexOf("/"));
    }
    if (window.gradio_config.dev_mode || typeof window !== "undefined" && window?.BUILD_MODE === "dev") {
      let config_url = join_urls(
        endpoint,
        this.deep_link ? CONFIG_URL + "?deep_link=" + this.deep_link : CONFIG_URL
      );
      const response = await this.fetch(config_url, {
        headers,
        credentials: "include"
      });
      const config = await handleConfigResponse(response, !!this.options.auth);
      config.root = endpoint || config.root;
      window.gradio_config = {
        ...config,
        current_page: window.gradio_config.current_page
      };
    }
    return { ...window.gradio_config };
  } else if (endpoint) {
    let config_url = join_urls(
      endpoint,
      this.deep_link ? CONFIG_URL + "?deep_link=" + this.deep_link : CONFIG_URL
    );
    const response = await this.fetch(config_url, {
      headers,
      credentials: "include"
    });
    const config = await handleConfigResponse(response, !!this.options.auth);
    if (!config.root) {
      config.root = endpoint;
    }
    return config;
  }
  throw new Error(CONFIG_ERROR_MSG);
}
async function handleConfigResponse(response, authorized) {
  if (response?.status === 401 && !authorized) {
    const error_data = await response.json();
    const auth_message = error_data?.detail?.auth_message;
    throw new Error(auth_message || MISSING_CREDENTIALS_MSG);
  } else if (response?.status === 401 && authorized) {
    throw new Error(INVALID_CREDENTIALS_MSG);
  }
  if (response?.status === 200) {
    let config = await response.json();
    config.dependencies?.forEach((dep, i) => {
      if (dep.id === void 0) {
        dep.id = i;
      }
    });
    return config;
  } else if (response?.status === 401) {
    throw new Error(UNAUTHORIZED_MSG);
  }
  throw new Error(CONFIG_ERROR_MSG);
}
async function resolve_cookies() {
  const { http_protocol, host } = await process_endpoint(
    this.app_reference,
    this.options.token
  );
  try {
    if (this.options.auth) {
      const cookie_header = await get_cookie_header(
        http_protocol,
        host,
        this.options.auth,
        this.fetch,
        this.options.token
      );
      if (cookie_header) this.set_cookies(cookie_header);
    }
  } catch (e) {
    throw Error(e.message);
  }
}
async function get_cookie_header(http_protocol, host, auth, _fetch, token) {
  const formData = new FormData();
  formData.append("username", auth?.[0]);
  formData.append("password", auth?.[1]);
  let headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await _fetch(`${http_protocol}//${host}/${LOGIN_URL}`, {
    headers,
    method: "POST",
    body: formData,
    credentials: "include"
  });
  if (res.status === 200) {
    return res.headers.get("set-cookie");
  } else if (res.status === 401) {
    throw new Error(INVALID_CREDENTIALS_MSG);
  } else {
    throw new Error(SPACE_METADATA_ERROR_MSG);
  }
}
function determine_protocol(endpoint) {
  if (endpoint.startsWith("http")) {
    const { protocol, host, pathname } = new URL(endpoint);
    return {
      ws_protocol: protocol === "https:" ? "wss" : "ws",
      http_protocol: protocol,
      host: host + (pathname !== "/" ? pathname : "")
    };
  }
  return {
    ws_protocol: "wss",
    http_protocol: "https:",
    host: new URL(endpoint).host
  };
}
const parse_and_set_cookies = (cookie_header) => {
  let cookies = [];
  const parts = cookie_header.split(/,(?=\s*[^\s=;]+=[^\s=;]+)/);
  parts.forEach((cookie) => {
    const [cookie_name, cookie_value] = cookie.split(";")[0].split("=");
    if (cookie_name && cookie_value) {
      cookies.push(`${cookie_name.trim()}=${cookie_value.trim()}`);
    }
  });
  return cookies;
};

const RE_SPACE_NAME = /^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$/;
const RE_SPACE_DOMAIN = /.*hf\.space\/{0,1}.*$/;
async function process_endpoint(app_reference, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const _app_reference = app_reference.trim().replace(/\/$/, "");
  if (RE_SPACE_NAME.test(_app_reference)) {
    try {
      const res = await fetch(
        `https://huggingface.co/api/spaces/${_app_reference}/${HOST_URL}`,
        { headers }
      );
      const _host = (await res.json()).host;
      return {
        space_id: app_reference,
        ...determine_protocol(_host)
      };
    } catch (e) {
      throw new Error(SPACE_METADATA_ERROR_MSG);
    }
  }
  if (RE_SPACE_DOMAIN.test(_app_reference)) {
    const { ws_protocol, http_protocol, host } = determine_protocol(_app_reference);
    return {
      space_id: host.split("/")[0].replace(".hf.space", ""),
      ws_protocol,
      http_protocol,
      host
    };
  }
  return {
    space_id: false,
    ...determine_protocol(_app_reference)
  };
}
const join_urls = (...urls) => {
  try {
    return urls.reduce((base_url, part) => {
      base_url = base_url.replace(/\/+$/, "");
      part = part.replace(/^\/+/, "");
      return new URL(part, base_url + "/").toString();
    });
  } catch (e) {
    throw new Error(INVALID_URL_MSG);
  }
};
function transform_api_info(api_info, config, api_map) {
  const transformed_info = {
    named_endpoints: {},
    unnamed_endpoints: {}
  };
  Object.keys(api_info).forEach((category) => {
    if (category === "named_endpoints" || category === "unnamed_endpoints") {
      transformed_info[category] = {};
      Object.entries(api_info[category]).forEach(
        ([endpoint, { parameters, returns }]) => {
          const dependencyIndex = config.dependencies.find(
            (dep) => dep.api_name === endpoint || dep.api_name === endpoint.replace("/", "")
          )?.id || api_map[endpoint.replace("/", "")] || -1;
          const dependencyTypes = dependencyIndex !== -1 ? config.dependencies.find((dep) => dep.id == dependencyIndex)?.types : { generator: false, cancel: false };
          if (dependencyIndex !== -1 && config.dependencies.find((dep) => dep.id == dependencyIndex)?.inputs?.length !== parameters.length) {
            const components = config.dependencies.find((dep) => dep.id == dependencyIndex).inputs.map(
              (input) => config.components.find((c) => c.id === input)?.type
            );
            try {
              components.forEach((comp, idx) => {
                if (comp === "state") {
                  const new_param = {
                    component: "state",
                    example: null,
                    parameter_default: null,
                    parameter_has_default: true,
                    parameter_name: null,
                    hidden: true
                  };
                  parameters.splice(idx, 0, new_param);
                }
              });
            } catch (e) {
              console.error(e);
            }
          }
          const transform_type = (data, component, serializer, signature_type) => ({
            ...data,
            description: get_description(data?.type, serializer),
            type: get_type(data?.type, component, serializer, signature_type) || ""
          });
          transformed_info[category][endpoint] = {
            parameters: parameters.map(
              (p) => transform_type(p, p?.component, p?.serializer, "parameter")
            ),
            returns: returns.map(
              (r) => transform_type(r, r?.component, r?.serializer, "return")
            ),
            type: dependencyTypes
          };
        }
      );
    }
  });
  return transformed_info;
}
function get_type(type, component, serializer, signature_type) {
  if (component === "Api") return type.type;
  switch (type?.type) {
    case "string":
      return "string";
    case "boolean":
      return "boolean";
    case "number":
      return "number";
  }
  if (serializer === "JSONSerializable" || serializer === "StringSerializable") {
    return "any";
  } else if (serializer === "ListStringSerializable") {
    return "string[]";
  } else if (component === "Image") {
    return signature_type === "parameter" ? "Blob | File | Buffer" : "string";
  } else if (serializer === "FileSerializable") {
    if (type?.type === "array") {
      return signature_type === "parameter" ? "(Blob | File | Buffer)[]" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}[]`;
    }
    return signature_type === "parameter" ? "Blob | File | Buffer" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}`;
  } else if (serializer === "GallerySerializable") {
    return signature_type === "parameter" ? "[(Blob | File | Buffer), (string | null)][]" : `[{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}, (string | null))][]`;
  }
}
function get_description(type, serializer) {
  if (serializer === "GallerySerializable") {
    return "array of [file, label] tuples";
  } else if (serializer === "ListStringSerializable") {
    return "array of strings";
  } else if (serializer === "FileSerializable") {
    return "array of files or single file";
  }
  return type?.description;
}
function handle_message(data, last_status) {
  const queue = true;
  switch (data.msg) {
    case "send_data":
      return { type: "data" };
    case "send_hash":
      return { type: "hash" };
    case "queue_full":
      return {
        type: "update",
        status: {
          queue,
          message: QUEUE_FULL_MSG,
          stage: "error",
          code: data.code,
          success: data.success
        }
      };
    case "heartbeat":
      return {
        type: "heartbeat"
      };
    case "unexpected_error":
      return {
        type: "unexpected_error",
        status: {
          queue,
          message: data.message,
          session_not_found: data.session_not_found,
          stage: "error",
          success: false
        }
      };
    case "broken_connection":
      return {
        type: "broken_connection",
        status: {
          queue,
          message: data.message,
          stage: "error",
          success: false
        }
      };
    case "estimation":
      return {
        type: "update",
        status: {
          queue,
          stage: last_status || "pending",
          code: data.code,
          size: data.queue_size,
          position: data.rank,
          eta: data.rank_eta,
          success: data.success
        }
      };
    case "progress":
      return {
        type: "update",
        status: {
          queue,
          stage: "pending",
          code: data.code,
          progress_data: data.progress_data,
          success: data.success
        }
      };
    case "log":
      return { type: "log", data };
    case "process_generating":
      return {
        type: "generating",
        status: {
          queue,
          message: !data.success ? data.output.error : null,
          stage: data.success ? "generating" : "error",
          code: data.code,
          progress_data: data.progress_data,
          eta: data.average_duration,
          changed_state_ids: data.success ? data.output.changed_state_ids : void 0
        },
        data: data.success ? data.output : null
      };
    case "process_streaming":
      return {
        type: "streaming",
        status: {
          queue,
          message: data.output.error,
          stage: "streaming",
          time_limit: data.time_limit,
          code: data.code,
          progress_data: data.progress_data,
          eta: data.eta
        },
        data: data.output
      };
    case "process_completed":
      if ("error" in data.output) {
        return {
          type: "update",
          status: {
            queue,
            title: data.output.title ?? "Error",
            message: data.output.error ?? "An error occurred",
            visible: data.output.visible,
            duration: data.output.duration,
            stage: "error",
            code: data.code,
            success: data.success
          }
        };
      }
      return {
        type: "complete",
        status: {
          queue,
          message: !data.success ? data.output.error : void 0,
          stage: data.success ? "complete" : "error",
          code: data.code,
          progress_data: data.progress_data,
          changed_state_ids: data.success ? data.output.changed_state_ids : void 0
        },
        data: data.success ? data.output : null
      };
    case "process_starts":
      return {
        type: "update",
        status: {
          queue,
          stage: "pending",
          code: data.code,
          size: data.rank,
          position: 0,
          success: data.success,
          eta: data.eta
        },
        original_msg: "process_starts"
      };
  }
  return { type: "none", status: { stage: "error", queue } };
}
const map_data_to_params = (data = [], endpoint_info) => {
  const parameters = endpoint_info ? endpoint_info.parameters : [];
  if (Array.isArray(data)) {
    if (endpoint_info && parameters.length > 0 && data.length > parameters.length) {
      console.warn("Too many arguments provided for the endpoint.");
    }
    return data;
  }
  const resolved_data = [];
  const provided_keys = Object.keys(data);
  parameters.forEach((param, index) => {
    if (data.hasOwnProperty(param.parameter_name)) {
      resolved_data[index] = data[param.parameter_name];
    } else if (param.parameter_has_default) {
      resolved_data[index] = param.parameter_default;
    } else {
      throw new Error(
        `No value provided for required parameter: ${param.parameter_name}`
      );
    }
  });
  provided_keys.forEach((key) => {
    if (!parameters.some((param) => param.parameter_name === key)) {
      throw new Error(
        `Parameter \`${key}\` is not a valid keyword argument. Please refer to the API for usage.`
      );
    }
  });
  resolved_data.forEach((value, idx) => {
    if (value === void 0 && !parameters[idx].parameter_has_default) {
      throw new Error(
        `No value provided for required parameter: ${parameters[idx].parameter_name}`
      );
    }
  });
  return resolved_data;
};

async function view_api() {
  if (this.api_info) return this.api_info;
  const { token } = this.options;
  const { config } = this;
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  if (!config) {
    return;
  }
  try {
    let response;
    let api_info;
    if (typeof window !== "undefined" && window.gradio_api_info) {
      api_info = window.gradio_api_info;
    } else {
      const url = join_urls(config.root, this.api_prefix, API_INFO_URL);
      response = await this.fetch(url, {
        headers,
        credentials: "include"
      });
      if (!response.ok) {
        throw new Error(BROKEN_CONNECTION_MSG);
      }
      api_info = await response.json();
    }
    if ("api" in api_info) {
      api_info = api_info.api;
    }
    if (api_info.named_endpoints["/predict"] && !api_info.unnamed_endpoints["0"]) {
      api_info.unnamed_endpoints[0] = api_info.named_endpoints["/predict"];
    }
    return transform_api_info(api_info, config, this.api_map);
  } catch (e) {
    throw new Error("Could not get API info. " + e.message);
  }
}

async function upload_files(root_url, files, upload_id) {
  const headers = {};
  if (this?.options?.token) {
    headers.Authorization = `Bearer ${this.options.token}`;
  }
  const chunkSize = 1e3;
  const uploadResponses = [];
  let response;
  for (let i = 0; i < files.length; i += chunkSize) {
    const chunk = files.slice(i, i + chunkSize);
    const formData = new FormData();
    chunk.forEach((file) => {
      formData.append("files", file);
    });
    try {
      const upload_url = upload_id ? `${root_url}${this.api_prefix}/${UPLOAD_URL}?upload_id=${upload_id}` : `${root_url}${this.api_prefix}/${UPLOAD_URL}`;
      response = await this.fetch(upload_url, {
        method: "POST",
        body: formData,
        headers,
        credentials: "include"
      });
    } catch (e) {
      throw new Error(BROKEN_CONNECTION_MSG + e.message);
    }
    if (!response.ok) {
      const error_text = await response.text();
      return { error: `HTTP ${response.status}: ${error_text}` };
    }
    const output = await response.json();
    if (output) {
      uploadResponses.push(...output);
    }
  }
  return { files: uploadResponses };
}

const si = {
  radix: 1e3,
  unit: ["b", "kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]
};
const iec = {
  radix: 1024,
  unit: ["b", "Kib", "Mib", "Gib", "Tib", "Pib", "Eib", "Zib", "Yib"]
};
const jedec = {
  radix: 1024,
  unit: ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]
};
const SPECS = {
  si,
  iec,
  jedec
};
function filesize(bytes, fixed = 1, spec = "jedec") {
  bytes = Math.abs(bytes);
  const { radix, unit } = SPECS[spec] || SPECS.jedec;
  let loop = 0;
  while (bytes >= radix) {
    bytes /= radix;
    ++loop;
  }
  return `${bytes.toFixed(fixed)} ${unit[loop]}`;
}

async function upload(file_data, root_url, upload_id, max_file_size) {
  let files = (Array.isArray(file_data) ? file_data : [file_data]).map(
    (file_data2) => file_data2.blob
  );
  const oversized_files = files.filter(
    (f) => f.size > (max_file_size ?? Infinity)
  );
  if (oversized_files.length) {
    throw new Error(
      `File(s) exceed the maximum allowed size of ${filesize(max_file_size || Infinity)}: ${oversized_files.map((f) => `"${f.name}"`).join(", ")}`
    );
  }
  return await Promise.all(
    await this.upload_files(root_url, files, upload_id).then(
      async (response) => {
        if (response.error) {
          throw new Error(response.error);
        } else {
          if (response.files) {
            return response.files.map((f, i) => {
              const file = new FileData({
                ...file_data[i],
                path: f,
                url: `${root_url}${this.api_prefix}/file=${f}`
              });
              return file;
            });
          }
          return [];
        }
      }
    )
  );
}
async function prepare_files(files, is_stream) {
  return files.map(
    (f) => new FileData({
      path: f.name,
      orig_name: f.name,
      blob: f,
      size: f.size,
      mime_type: f.type,
      is_stream
    })
  );
}
class FileData {
  path;
  url;
  orig_name;
  size;
  blob;
  is_stream;
  mime_type;
  alt_text;
  b64;
  meta = { _type: "gradio.FileData" };
  constructor({
    path,
    url,
    orig_name,
    size,
    blob,
    is_stream,
    mime_type,
    alt_text,
    b64
  }) {
    this.path = path;
    this.url = url;
    this.orig_name = orig_name;
    this.size = size;
    this.blob = url ? void 0 : blob;
    this.is_stream = is_stream;
    this.mime_type = mime_type;
    this.alt_text = alt_text;
    this.b64 = b64;
  }
}

class Command {
  type;
  command;
  meta;
  fileData;
  constructor(command, meta) {
    this.type = "command";
    this.command = command;
    this.meta = meta;
  }
}

typeof process !== "undefined" && process.versions && process.versions.node;
function update_object(object, newValue, stack) {
  while (stack.length > 1) {
    const key2 = stack.shift();
    if (typeof key2 === "string" || typeof key2 === "number") {
      object = object[key2];
    } else {
      throw new Error("Invalid key type");
    }
  }
  const key = stack.shift();
  if (typeof key === "string" || typeof key === "number") {
    object[key] = newValue;
  } else {
    throw new Error("Invalid key type");
  }
}
async function walk_and_store_blobs(data, type = void 0, path = [], root = false, endpoint_info = void 0) {
  if (Array.isArray(data)) {
    let blob_refs = [];
    await Promise.all(
      data.map(async (_, index) => {
        let new_path = path.slice();
        new_path.push(String(index));
        const array_refs = await walk_and_store_blobs(
          data[index],
          root ? endpoint_info?.parameters[index]?.component || void 0 : type,
          new_path,
          false,
          endpoint_info
        );
        blob_refs = blob_refs.concat(array_refs);
      })
    );
    return blob_refs;
  } else if (globalThis.Buffer && data instanceof globalThis.Buffer || data instanceof Blob) {
    return [
      {
        path,
        blob: new Blob([data]),
        type
      }
    ];
  } else if (typeof data === "object" && data !== null) {
    let blob_refs = [];
    for (const key of Object.keys(data)) {
      const new_path = [...path, key];
      const value = data[key];
      blob_refs = blob_refs.concat(
        await walk_and_store_blobs(
          value,
          void 0,
          new_path,
          false,
          endpoint_info
        )
      );
    }
    return blob_refs;
  }
  return [];
}
function skip_queue(id, config) {
  let fn_queue = config?.dependencies?.find((dep) => dep.id == id)?.queue;
  if (fn_queue != null) {
    return !fn_queue;
  }
  return !config.enable_queue;
}
function post_message(message, origin) {
  return new Promise((res, _rej) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (({ data }) => {
      channel.port1.close();
      res(data);
    });
    window.parent.postMessage(message, origin, [channel.port2]);
  });
}
function handle_payload(resolved_payload, dependency, components, type, with_null_state = false) {
  if (type === "input" && !with_null_state) {
    throw new Error("Invalid code path. Cannot skip state inputs for input.");
  }
  if (type === "output" && with_null_state) {
    return resolved_payload;
  }
  let updated_payload = [];
  let payload_index = 0;
  const deps = type === "input" ? dependency.inputs : dependency.outputs;
  for (let i = 0; i < deps.length; i++) {
    const input_id = deps[i];
    const component = components.find((c) => c.id === input_id);
    if (component?.type === "state") {
      if (with_null_state) {
        if (resolved_payload.length === deps.length) {
          const value = resolved_payload[payload_index];
          updated_payload.push(value);
          payload_index++;
        } else {
          updated_payload.push(null);
        }
      } else {
        payload_index++;
        continue;
      }
      continue;
    } else {
      const value = resolved_payload[payload_index];
      updated_payload.push(value);
      payload_index++;
    }
  }
  return updated_payload;
}

async function handle_blob(endpoint, data, api_info) {
  const self = this;
  await process_local_file_commands(self, data);
  const blobRefs = await walk_and_store_blobs(
    data,
    void 0,
    [],
    true,
    api_info
  );
  const results = await Promise.all(
    blobRefs.map(async ({ path, blob, type }) => {
      if (!blob) return { path, type };
      const response = await self.upload_files(endpoint, [blob]);
      const file_url = response.files && response.files[0];
      return {
        path,
        file_url,
        type,
        name: typeof File !== "undefined" && blob instanceof File ? blob?.name : void 0
      };
    })
  );
  results.forEach(({ path, file_url, type, name }) => {
    if (type === "Gallery") {
      update_object(data, file_url, path);
    } else if (file_url) {
      const file = new FileData({ path: file_url, orig_name: name });
      update_object(data, file, path);
    }
  });
  return data;
}
async function process_local_file_commands(client, data) {
  const root = client.config?.root || client.config?.root_url;
  if (!root) {
    throw new Error(ROOT_URL_ERROR_MSG);
  }
  await recursively_process_commands(client, data);
}
async function recursively_process_commands(client, data, path = []) {
  for (const key in data) {
    if (data[key] instanceof Command) {
      await process_single_command(client, data, key);
    } else if (typeof data[key] === "object" && data[key] !== null) {
      await recursively_process_commands(client, data[key], [...path, key]);
    }
  }
}
async function process_single_command(client, data, key) {
  let cmd_item = data[key];
  const root = client.config?.root || client.config?.root_url;
  if (!root) {
    throw new Error(ROOT_URL_ERROR_MSG);
  }
  try {
    let fileBuffer;
    let fullPath;
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
      const fs = await __vitePreload(() => import('./__vite-browser-external-CVaeKfQx.js').then(n => n._),true              ?[]:void 0,import.meta.url);
      const path = await __vitePreload(() => import('./__vite-browser-external-CVaeKfQx.js').then(n => n._),true              ?[]:void 0,import.meta.url);
      fullPath = path.resolve(process.cwd(), cmd_item.meta.path);
      fileBuffer = await fs.readFile(fullPath);
    } else {
      throw new Error(NODEJS_FS_ERROR_MSG);
    }
    const file = new Blob([fileBuffer], {
      type: "application/octet-stream"
    });
    const response = await client.upload_files(root, [file]);
    const file_url = response.files && response.files[0];
    if (file_url) {
      const fileData = new FileData({
        path: file_url,
        orig_name: cmd_item.meta.name || ""
      });
      data[key] = fileData;
    }
  } catch (error) {
    console.error(FILE_PROCESSING_ERROR_MSG, error);
  }
}

async function post_data(url, body, additional_headers) {
  const headers = { "Content-Type": "application/json" };
  if (this.options.token) {
    headers.Authorization = `Bearer ${this.options.token}`;
  }
  try {
    var response = await this.fetch(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { ...headers, ...additional_headers },
      credentials: "include"
    });
  } catch (e) {
    return [{ error: BROKEN_CONNECTION_MSG }, 500];
  }
  let output;
  let status;
  try {
    output = await response.json();
    status = response.status;
  } catch (e) {
    output = { error: `Could not parse server response: ${e}` };
    status = 500;
  }
  return [output, status];
}

async function predict(endpoint, data = {}) {
  let data_returned = false;
  let status_complete = false;
  if (!this.config) {
    throw new Error("Could not resolve app config");
  }
  if (typeof endpoint === "number") {
    this.config.dependencies.find((dep) => dep.id == endpoint);
  } else {
    const trimmed_endpoint = endpoint.replace(/^\//, "");
    this.config.dependencies.find(
      (dep) => dep.id == this.api_map[trimmed_endpoint]
    );
  }
  return new Promise(async (resolve, reject) => {
    const app = this.submit(endpoint, data, null, null, true);
    let result;
    for await (const message of app) {
      if (message.type === "data") {
        if (status_complete) {
          resolve(result);
        }
        data_returned = true;
        result = message;
      }
      if (message.type === "status") {
        if (message.stage === "error") reject(message);
        if (message.stage === "complete") {
          status_complete = true;
          if (data_returned) {
            resolve(result);
          }
        }
      }
    }
  });
}

async function check_space_status(id, type, status_callback) {
  let endpoint = type === "subdomain" ? `https://huggingface.co/api/spaces/by-subdomain/${id}` : `https://huggingface.co/api/spaces/${id}`;
  let response;
  let _status;
  try {
    response = await fetch(endpoint);
    _status = response.status;
    if (_status !== 200) {
      throw new Error();
    }
    response = await response.json();
  } catch (e) {
    status_callback({
      status: "error",
      load_status: "error",
      message: SPACE_STATUS_ERROR_MSG,
      detail: "NOT_FOUND"
    });
    return;
  }
  if (!response || _status !== 200) return;
  const {
    runtime: { stage },
    id: space_name
  } = response;
  switch (stage) {
    case "STOPPED":
    case "SLEEPING":
      status_callback({
        status: "sleeping",
        load_status: "pending",
        message: "Space is asleep. Waking it up...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    case "PAUSED":
      status_callback({
        status: "paused",
        load_status: "error",
        message: "This space has been paused by the author. If you would like to try this demo, consider duplicating the space.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
    case "RUNNING":
    case "RUNNING_BUILDING":
      status_callback({
        status: "running",
        load_status: "complete",
        message: "Space is running.",
        detail: stage
      });
      break;
    case "BUILDING":
      status_callback({
        status: "building",
        load_status: "pending",
        message: "Space is building...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    case "APP_STARTING":
      status_callback({
        status: "starting",
        load_status: "pending",
        message: "Space is starting...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    default:
      status_callback({
        status: "space_error",
        load_status: "error",
        message: "This space is experiencing an issue.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
  }
}
const check_and_wake_space = async (space_id, status_callback) => {
  let retries = 0;
  const max_retries = 12;
  const check_interval = 5e3;
  return new Promise((resolve) => {
    check_space_status(
      space_id,
      RE_SPACE_NAME.test(space_id) ? "space_name" : "subdomain",
      (status) => {
        status_callback(status);
        if (status.status === "running") {
          resolve();
        } else if (status.status === "error" || status.status === "paused" || status.status === "space_error") {
          resolve();
        } else if (status.status === "sleeping" || status.status === "building") {
          if (retries < max_retries) {
            retries++;
            setTimeout(() => {
              check_and_wake_space(space_id, status_callback).then(resolve);
            }, check_interval);
          } else {
            resolve();
          }
        }
      }
    );
  });
};
const RE_DISABLED_DISCUSSION = /^(?=[^]*\b[dD]iscussions{0,1}\b)(?=[^]*\b[dD]isabled\b)[^]*$/;
async function discussions_enabled(space_id) {
  try {
    const r = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/discussions`,
      {
        method: "HEAD"
      }
    );
    const error = r.headers.get("x-error-message");
    if (!r.ok || error && RE_DISABLED_DISCUSSION.test(error)) return false;
    return true;
  } catch (e) {
    return false;
  }
}
async function get_space_hardware(space_id, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const res = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/${RUNTIME_URL}`,
      { headers }
    );
    if (res.status !== 200)
      throw new Error("Space hardware could not be obtained.");
    const { hardware } = await res.json();
    return hardware.current;
  } catch (e) {
    throw new Error(e.message);
  }
}
async function set_space_timeout(space_id, timeout, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const body = {
    seconds: timeout
  };
  try {
    const res = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/${SLEEPTIME_URL}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify(body)
      }
    );
    if (res.status !== 200) {
      throw new Error(
        "Could not set sleep timeout on duplicated Space. Please visit *ADD HF LINK TO SETTINGS* to set a timeout manually to reduce billing charges."
      );
    }
    const response = await res.json();
    return response;
  } catch (e) {
    throw new Error(e.message);
  }
}
const hardware_types = [
  "cpu-basic",
  "cpu-upgrade",
  "cpu-xl",
  "t4-small",
  "t4-medium",
  "a10g-small",
  "a10g-large",
  "a10g-largex2",
  "a10g-largex4",
  "a100-large",
  "zero-a10g",
  "h100",
  "h100x8"
];

async function duplicate(app_reference, options) {
  const { token, private: _private, hardware, timeout, auth } = options;
  if (hardware && !hardware_types.includes(hardware)) {
    throw new Error(
      `Invalid hardware type provided. Valid types are: ${hardware_types.map((v) => `"${v}"`).join(",")}.`
    );
  }
  const { http_protocol, host } = await process_endpoint(app_reference, token);
  let cookies = null;
  if (auth) {
    const cookie_header = await get_cookie_header(
      http_protocol,
      host,
      auth,
      fetch
    );
    if (cookie_header) cookies = parse_and_set_cookies(cookie_header);
  }
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    ...cookies ? { Cookie: cookies.join("; ") } : {}
  };
  const user = (await (await fetch(`https://huggingface.co/api/whoami-v2`, {
    headers
  })).json()).name;
  const space_name = app_reference.split("/")[1];
  const body = {
    repository: `${user}/${space_name}`
  };
  if (_private) {
    body.private = true;
  }
  let original_hardware;
  try {
    if (!hardware) {
      original_hardware = await get_space_hardware(app_reference, token);
    }
  } catch (e) {
    throw Error(SPACE_METADATA_ERROR_MSG + e.message);
  }
  const requested_hardware = hardware || original_hardware || "cpu-basic";
  body.hardware = requested_hardware;
  try {
    const response = await fetch(
      `https://huggingface.co/api/spaces/${app_reference}/duplicate`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(body)
      }
    );
    if (response.status === 409) {
      try {
        const client = await Client.connect(`${user}/${space_name}`, options);
        return client;
      } catch (error) {
        console.error("Failed to connect Client instance:", error);
        throw error;
      }
    } else if (response.status !== 200) {
      throw new Error(response.statusText);
    }
    const duplicated_space = await response.json();
    await set_space_timeout(`${user}/${space_name}`, timeout || 300, token);
    return await Client.connect(
      get_space_reference(duplicated_space.url),
      options
    );
  } catch (e) {
    throw new Error(e);
  }
}
function get_space_reference(url) {
  const regex = /https:\/\/huggingface.co\/spaces\/([^/]+\/[^/]+)/;
  const match = url.match(regex);
  if (match) {
    return match[1];
  }
}

// Copyright 2018-2024 the Deno authors. All rights reserved. MIT license.
// This module is browser compatible.
/**
 * Transform a stream into a stream where each chunk is divided by a newline,
 * be it `\n` or `\r\n`. `\r` can be enabled via the `allowCR` option.
 *
 * @example
 * ```ts
 * import { TextLineStream } from "@std/streams/text-line-stream";
 *
 * const res = await fetch("https://example.com");
 * const lines = res.body!
 *   .pipeThrough(new TextDecoderStream())
 *   .pipeThrough(new TextLineStream());
 * ```
 */
class TextLineStream extends TransformStream {
    #currentLine = "";
    /** Constructs a new instance. */
    constructor(options = { allowCR: false }) {
        super({
            transform: (chars, controller) => {
                chars = this.#currentLine + chars;
                while (true) {
                    const lfIndex = chars.indexOf("\n");
                    const crIndex = options.allowCR ? chars.indexOf("\r") : -1;
                    if (crIndex !== -1 && crIndex !== (chars.length - 1) &&
                        (lfIndex === -1 || (lfIndex - 1) > crIndex)) {
                        controller.enqueue(chars.slice(0, crIndex));
                        chars = chars.slice(crIndex + 1);
                        continue;
                    }
                    if (lfIndex === -1)
                        break;
                    const endIndex = chars[lfIndex - 1] === "\r" ? lfIndex - 1 : lfIndex;
                    controller.enqueue(chars.slice(0, endIndex));
                    chars = chars.slice(lfIndex + 1);
                }
                this.#currentLine = chars;
            },
            flush: (controller) => {
                if (this.#currentLine === "")
                    return;
                const currentLine = options.allowCR && this.#currentLine.endsWith("\r")
                    ? this.#currentLine.slice(0, -1)
                    : this.#currentLine;
                controller.enqueue(currentLine);
            },
        });
    }
}

function stream$1(input) {
    let decoder = new TextDecoderStream();
    let split = new TextLineStream({ allowCR: true });
    return input.pipeThrough(decoder).pipeThrough(split);
}
function split(input) {
    let rgx = /[:]\s*/;
    let match = rgx.exec(input);
    // ": comment" -> index=0 -> ignore
    let idx = match && match.index;
    if (idx) {
        return [
            input.substring(0, idx),
            input.substring(idx + match[0].length),
        ];
    }
}
function fallback(headers, key, value) {
    let tmp = headers.get(key);
    if (!tmp)
        headers.set(key, value);
}

/**
 * Convert a `Response` body containing Server Sent Events (SSE) into an Async Iterator that yields {@linkcode ServerSentEventMessage} objects.
 *
 * @see {@link https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events}
 *
 * @example
 * ```js
 * // Optional
 * let abort = new AbortController;
 *
 * // Manually fetch a Response
 * let res = await fetch('https://...', {
 *   method: 'POST',
 *   signal: abort.signal,
 *   headers: {
 *     'api-key': 'token <value>',
 *     'content-type': 'application/json',
 *   },
 *   body: JSON.stringify({
 *     stream: true, // <- hypothetical
 *     // ...
 *   })
 * });
 *
 * if (res.ok) {
 *   let stream = events(res, abort.signal);
 *   for await (let event of stream) {
 *     console.log('<<', event.data);
 *   }
 * }
 * ```
 */
async function* events(res, signal) {
    // TODO: throw error?
    if (!res.body)
        return;
    let iter = stream$1(res.body);
    let line, reader = iter.getReader();
    let event;
    for (;;) {
        if (signal && signal.aborted) {
            return reader.cancel();
        }
        line = await reader.read();
        if (line.done)
            return;
        if (!line.value) {
            if (event)
                yield event;
            event = undefined;
            continue;
        }
        let [field, value] = (split(line.value) || []);
        if (field === 'data') {
            event ||= {};
            event[field] = event[field] ? (event[field] + '\n' + value) : value;
        }
        else if (field === 'event') {
            event ||= {};
            event[field] = value;
        }
        else if (field === 'id') {
            event ||= {};
            event[field] = String(+value) === value ? +value : value;
        }
        else if (field === 'retry') {
            event ||= {};
            event[field] = +value || undefined;
        }
    }
}
/**
 * Convenience function that will `fetch` with the given arguments and, if ok, will return the {@linkcode events} async iterator.
 *
 * If the response is not ok (status 200-299), the `Response` is thrown.
 *
 * @example
 * ```js
 * // NOTE: throws `Response` if not 2xx status
 * let events = await stream('https://api.openai.com/...', {
 *   method: 'POST',
 *   headers: {
 *     'Authorization': 'Bearer <token>',
 *     'Content-Type': 'application/json',
 *   },
 *   body: JSON.stringify({
 *     stream: true,
 *     // ...
 *   })
 * });
 *
 * for await (let event of events) {
 *   console.log('<<', JSON.parse(event.data));
 * }
 * ```
 */
async function stream(input, init) {
    let req = new Request(input, init);
    fallback(req.headers, 'Accept', 'text/event-stream');
    fallback(req.headers, 'Content-Type', 'application/json');
    let r = await fetch(req);
    if (!r.ok)
        throw r;
    return events(r, req.signal);
}

async function open_stream() {
  let {
    event_callbacks,
    unclosed_events,
    pending_stream_messages,
    stream_status,
    config,
    jwt
  } = this;
  const that = this;
  if (!config) {
    throw new Error("Could not resolve app config");
  }
  stream_status.open = true;
  let stream2 = null;
  let params = new URLSearchParams({
    session_hash: this.session_hash
  }).toString();
  let url = new URL(`${config.root}${this.api_prefix}/${SSE_URL}?${params}`);
  if (jwt) {
    url.searchParams.set("__sign", jwt);
  }
  stream2 = this.stream(url);
  if (!stream2) {
    console.warn("Cannot connect to SSE endpoint: " + url.toString());
    return;
  }
  stream2.onmessage = async function(event) {
    let _data = JSON.parse(event.data);
    if (_data.msg === "close_stream") {
      close_stream(stream_status, that.abort_controller);
      return;
    }
    const event_id = _data.event_id;
    if (!event_id) {
      await Promise.all(
        Object.keys(event_callbacks).map(
          (event_id2) => event_callbacks[event_id2](_data)
        )
      );
    } else if (event_callbacks[event_id] && config) {
      if (_data.msg === "process_completed" && ["sse", "sse_v1", "sse_v2", "sse_v2.1", "sse_v3"].includes(
        config.protocol
      )) {
        unclosed_events.delete(event_id);
      }
      let fn = event_callbacks[event_id];
      if (typeof window !== "undefined" && typeof document !== "undefined") {
        setTimeout(fn, 0, _data);
      } else {
        fn(_data);
      }
    } else {
      if (!pending_stream_messages[event_id]) {
        pending_stream_messages[event_id] = [];
      }
      pending_stream_messages[event_id].push(_data);
    }
  };
  stream2.onerror = async function(e) {
    console.error(e);
    await Promise.all(
      Object.keys(event_callbacks).map(
        (event_id) => event_callbacks[event_id]({
          msg: "broken_connection",
          message: BROKEN_CONNECTION_MSG
        })
      )
    );
  };
}
function close_stream(stream_status, abort_controller) {
  if (stream_status) {
    stream_status.open = false;
    abort_controller?.abort();
  }
}
function apply_diff_stream(pending_diff_streams, event_id, data) {
  let is_first_generation = !pending_diff_streams[event_id];
  if (is_first_generation) {
    pending_diff_streams[event_id] = [];
    data.data.forEach((value, i) => {
      pending_diff_streams[event_id][i] = value;
    });
  } else {
    data.data.forEach((value, i) => {
      let new_data = apply_diff(pending_diff_streams[event_id][i], value);
      pending_diff_streams[event_id][i] = new_data;
      data.data[i] = new_data;
    });
  }
}
function apply_diff(obj, diff) {
  diff.forEach(([action, path, value]) => {
    obj = apply_edit(obj, path, action, value);
  });
  return obj;
}
function apply_edit(target, path, action, value) {
  if (path.length === 0) {
    if (action === "replace") {
      return value;
    } else if (action === "append") {
      return target + value;
    }
    throw new Error(`Unsupported action: ${action}`);
  }
  let current = target;
  for (let i = 0; i < path.length - 1; i++) {
    current = current[path[i]];
  }
  const last_path = path[path.length - 1];
  switch (action) {
    case "replace":
      current[last_path] = value;
      break;
    case "append":
      current[last_path] += value;
      break;
    case "add":
      if (Array.isArray(current)) {
        current.splice(Number(last_path), 0, value);
      } else {
        current[last_path] = value;
      }
      break;
    case "delete":
      if (Array.isArray(current)) {
        current.splice(Number(last_path), 1);
      } else {
        delete current[last_path];
      }
      break;
    default:
      throw new Error(`Unknown action: ${action}`);
  }
  return target;
}
function readable_stream(input, init = {}) {
  const instance = {
    close: () => {
      console.warn("Method not implemented.");
    },
    onerror: null,
    onmessage: null,
    onopen: null,
    readyState: 0,
    url: input.toString(),
    withCredentials: false,
    CONNECTING: 0,
    OPEN: 1,
    CLOSED: 2,
    addEventListener: () => {
      throw new Error("Method not implemented.");
    },
    dispatchEvent: () => {
      throw new Error("Method not implemented.");
    },
    removeEventListener: () => {
      throw new Error("Method not implemented.");
    }
  };
  stream(input, init).then(async (res) => {
    instance.readyState = instance.OPEN;
    try {
      for await (const chunk of res) {
        instance.onmessage && instance.onmessage(chunk);
      }
      instance.readyState = instance.CLOSED;
    } catch (e) {
      instance.onerror && instance.onerror(e);
      instance.readyState = instance.CLOSED;
    }
  }).catch((e) => {
    console.error(e);
    instance.onerror && instance.onerror(e);
    instance.readyState = instance.CLOSED;
  });
  return instance;
}

function submit(endpoint, data = {}, event_data, trigger_id, all_events, additional_headers) {
  try {
    let fire_event = function(event) {
      if (all_events || events_to_publish[event.type]) {
        push_event(event);
      }
    }, close = function() {
      done = true;
      while (resolvers.length > 0)
        resolvers.shift()({
          value: void 0,
          done: true
        });
    }, push = function(data2) {
      if (resolvers.length > 0) {
        resolvers.shift()(data2);
      } else {
        values.push(data2);
      }
    }, push_error = function(error) {
      push(thenable_reject(error));
      close();
    }, push_event = function(event) {
      push({ value: event, done: false });
    }, next = function() {
      if (values.length > 0) {
        return Promise.resolve(values.shift());
      }
      return new Promise((resolve) => resolvers.push(resolve));
    };
    const { token } = this.options;
    const {
      fetch,
      app_reference,
      config,
      session_hash,
      api_info,
      api_map,
      stream_status,
      pending_stream_messages,
      pending_diff_streams,
      event_callbacks,
      unclosed_events,
      post_data,
      options,
      api_prefix
    } = this;
    const addt_headers = additional_headers || { "x-gradio-user": "api" };
    const that = this;
    if (!api_info) throw new Error("No API found");
    if (!config) throw new Error("Could not resolve app config");
    let { fn_index, endpoint_info, dependency } = get_endpoint_info(
      api_info,
      endpoint,
      api_map,
      config
    );
    let resolved_data = map_data_to_params(data, endpoint_info);
    let stream;
    let protocol = config.protocol ?? "ws";
    if (protocol === "ws") {
      throw new Error("WebSocket protocol is not supported in this version");
    }
    let event_id_final = "";
    let event_id_cb = () => event_id_final;
    const _endpoint = typeof endpoint === "number" ? "/predict" : endpoint;
    let payload;
    let event_id = null;
    let complete = false;
    let last_status = {};
    let url_params = typeof window !== "undefined" && typeof document !== "undefined" ? new URLSearchParams(window.location.search).toString() : "";
    const events_to_publish = options?.events?.reduce(
      (acc, event) => {
        acc[event] = true;
        return acc;
      },
      {}
    ) || {};
    async function cancel() {
      let reset_request = {};
      let cancel_request = {};
      reset_request = { event_id };
      cancel_request = { event_id, session_hash, fn_index };
      try {
        if (!config) {
          throw new Error("Could not resolve app config");
        }
        if ("event_id" in cancel_request) {
          await fetch(`${config.root}${api_prefix}/${CANCEL_URL}`, {
            headers: { "Content-Type": "application/json" },
            method: "POST",
            body: JSON.stringify(cancel_request)
          });
        }
        await fetch(`${config.root}${api_prefix}/${RESET_URL}`, {
          headers: { "Content-Type": "application/json" },
          method: "POST",
          body: JSON.stringify(reset_request)
        });
      } catch (e) {
        console.warn(
          "The `/reset` endpoint could not be called. Subsequent endpoint results may be unreliable."
        );
      }
    }
    const resolve_heartbeat = async (config2) => {
      await this._resolve_heartbeat(config2);
    };
    async function handle_render_config(render_config) {
      if (!config) return;
      let render_id = render_config.render_id;
      config.components = [
        ...config.components.filter((c) => c.props.rendered_in !== render_id),
        ...render_config.components
      ];
      config.dependencies = [
        ...config.dependencies.filter((d) => d.rendered_in !== render_id),
        ...render_config.dependencies
      ];
      const any_state = config.components.some((c) => c.type === "state");
      const any_unload = config.dependencies.some(
        (d) => d.targets.some((t) => t[1] === "unload")
      );
      config.connect_heartbeat = any_state || any_unload;
      await resolve_heartbeat(config);
      fire_event({
        type: "render",
        data: render_config,
        endpoint: _endpoint,
        fn_index
      });
    }
    const job = this.handle_blob(
      config.root,
      resolved_data,
      endpoint_info
    ).then(async (_payload) => {
      let input_data = handle_payload(
        _payload,
        dependency,
        config.components,
        "input",
        true
      );
      payload = {
        data: input_data || [],
        event_data,
        fn_index,
        trigger_id
      };
      if (skip_queue(fn_index, config)) {
        fire_event({
          type: "status",
          endpoint: _endpoint,
          stage: "pending",
          queue: false,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        post_data(
          `${config.root}${api_prefix}/run${_endpoint.startsWith("/") ? _endpoint : `/${_endpoint}`}${url_params ? "?" + url_params : ""}`,
          {
            ...payload,
            session_hash
          },
          addt_headers
        ).then(async ([output, status_code]) => {
          const data2 = output.data;
          if (status_code == 200) {
            fire_event({
              type: "data",
              endpoint: _endpoint,
              fn_index,
              data: handle_payload(
                data2,
                dependency,
                config.components,
                "output",
                options.with_null_state
              ),
              time: /* @__PURE__ */ new Date(),
              event_data,
              trigger_id
            });
            if (output.render_config) {
              await handle_render_config(output.render_config);
            }
            fire_event({
              type: "status",
              endpoint: _endpoint,
              fn_index,
              stage: "complete",
              eta: output.average_duration,
              queue: false,
              time: /* @__PURE__ */ new Date()
            });
          } else {
            const is_connection_error = output?.error === BROKEN_CONNECTION_MSG;
            fire_event({
              type: "status",
              stage: "error",
              endpoint: _endpoint,
              fn_index,
              message: output.error,
              broken: is_connection_error,
              queue: false,
              time: /* @__PURE__ */ new Date()
            });
          }
        }).catch((e) => {
          fire_event({
            type: "status",
            stage: "error",
            message: e.message,
            endpoint: _endpoint,
            fn_index,
            queue: false,
            time: /* @__PURE__ */ new Date()
          });
        });
      } else if (protocol == "sse") {
        fire_event({
          type: "status",
          stage: "pending",
          queue: true,
          endpoint: _endpoint,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        var params = new URLSearchParams({
          fn_index: fn_index.toString(),
          session_hash
        }).toString();
        let url = new URL(
          `${config.root}${api_prefix}/${SSE_URL}?${url_params ? url_params + "&" : ""}${params}`
        );
        if (this.jwt) {
          url.searchParams.set("__sign", this.jwt);
        }
        stream = this.stream(url);
        if (!stream) {
          return Promise.reject(
            new Error("Cannot connect to SSE endpoint: " + url.toString())
          );
        }
        stream.onmessage = async function(event) {
          const _data = JSON.parse(event.data);
          const { type, status, data: data2 } = handle_message(
            _data,
            last_status[fn_index]
          );
          if (type === "update" && status && !complete) {
            fire_event({
              type: "status",
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              ...status
            });
            if (status.stage === "error") {
              stream?.close();
              close();
            }
          } else if (type === "data") {
            let [_, status2] = await post_data(
              `${config.root}${api_prefix}/queue/data`,
              {
                ...payload,
                session_hash,
                event_id
              }
            );
            if (status2 !== 200) {
              fire_event({
                type: "status",
                stage: "error",
                message: BROKEN_CONNECTION_MSG,
                queue: true,
                endpoint: _endpoint,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              stream?.close();
              close();
            }
          } else if (type === "complete") {
            complete = status;
          } else if (type === "log") {
            fire_event({
              type: "log",
              title: data2.title,
              log: data2.log,
              level: data2.level,
              endpoint: _endpoint,
              duration: data2.duration,
              visible: data2.visible,
              fn_index
            });
          } else if (type === "generating" || type === "streaming") {
            fire_event({
              type: "status",
              time: /* @__PURE__ */ new Date(),
              ...status,
              stage: status?.stage,
              queue: true,
              endpoint: _endpoint,
              fn_index
            });
          }
          if (data2) {
            fire_event({
              type: "data",
              time: /* @__PURE__ */ new Date(),
              data: handle_payload(
                data2.data,
                dependency,
                config.components,
                "output",
                options.with_null_state
              ),
              endpoint: _endpoint,
              fn_index,
              event_data,
              trigger_id
            });
            if (complete) {
              fire_event({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...complete,
                stage: status?.stage,
                queue: true,
                endpoint: _endpoint,
                fn_index
              });
              stream?.close();
              close();
            }
          }
        };
      } else if (protocol == "sse_v1" || protocol == "sse_v2" || protocol == "sse_v2.1" || protocol == "sse_v3") {
        fire_event({
          type: "status",
          stage: "pending",
          queue: true,
          endpoint: _endpoint,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        let hostname = "";
        if (typeof window !== "undefined" && typeof document !== "undefined") {
          hostname = window?.location?.hostname;
        }
        let hfhubdev = "dev.spaces.huggingface.tech";
        const origin = hostname.includes(".dev.") ? `https://moon-${hostname.split(".")[1]}.${hfhubdev}` : `https://huggingface.co`;
        const is_zerogpu_iframe = typeof window !== "undefined" && typeof document !== "undefined" && window.parent != window && window.supports_zerogpu_headers;
        const zerogpu_auth_promise = is_zerogpu_iframe ? post_message("zerogpu-headers", origin) : Promise.resolve(null);
        const post_data_promise = zerogpu_auth_promise.then((headers) => {
          const combined_headers = { ...addt_headers, ...headers || {} };
          return post_data(
            `${config.root}${api_prefix}/${SSE_DATA_URL}?${url_params}`,
            {
              ...payload,
              session_hash
            },
            combined_headers
          );
        });
        return post_data_promise.then(async ([response, status]) => {
          if (response.event_id) {
            event_id_final = response.event_id;
          }
          if (status === 503) {
            fire_event({
              type: "status",
              stage: "error",
              message: QUEUE_FULL_MSG,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
          } else if (status === 422) {
            fire_event({
              type: "status",
              stage: "error",
              message: response.detail,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              code: "validation_error",
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
            close();
          } else if (status !== 200) {
            const is_connection_error = response?.error === BROKEN_CONNECTION_MSG;
            fire_event({
              type: "status",
              stage: "error",
              broken: is_connection_error,
              message: is_connection_error ? BROKEN_CONNECTION_MSG : response.detail || response.error,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
          } else {
            event_id = response.event_id;
            event_id_final = event_id;
            let callback = async function(_data) {
              try {
                const { type, status: status2, data: data2, original_msg } = handle_message(
                  _data,
                  last_status[fn_index]
                );
                if (type == "heartbeat") {
                  return;
                }
                if (type === "update" && status2 && !complete) {
                  fire_event({
                    type: "status",
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date(),
                    original_msg,
                    ...status2
                  });
                } else if (type === "complete") {
                  complete = status2;
                } else if (type == "unexpected_error" || type == "broken_connection") {
                  console.error("Unexpected error", status2?.message);
                  const broken = type === "broken_connection";
                  fire_event({
                    type: "status",
                    stage: "error",
                    message: status2?.message || "An Unexpected Error Occurred!",
                    queue: true,
                    endpoint: _endpoint,
                    broken,
                    session_not_found: status2?.session_not_found,
                    fn_index,
                    time: /* @__PURE__ */ new Date()
                  });
                } else if (type === "log") {
                  fire_event({
                    type: "log",
                    title: data2.title,
                    log: data2.log,
                    level: data2.level,
                    endpoint: _endpoint,
                    duration: data2.duration,
                    visible: data2.visible,
                    fn_index
                  });
                  return;
                } else if (type === "generating" || type === "streaming") {
                  fire_event({
                    type: "status",
                    time: /* @__PURE__ */ new Date(),
                    ...status2,
                    stage: status2?.stage,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (data2 && dependency.connection !== "stream" && ["sse_v2", "sse_v2.1", "sse_v3"].includes(protocol)) {
                    apply_diff_stream(pending_diff_streams, event_id, data2);
                  }
                }
                if (data2) {
                  fire_event({
                    type: "data",
                    time: /* @__PURE__ */ new Date(),
                    data: handle_payload(
                      data2.data,
                      dependency,
                      config.components,
                      "output",
                      options.with_null_state
                    ),
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (data2.render_config) {
                    await handle_render_config(data2.render_config);
                  }
                  if (complete) {
                    fire_event({
                      type: "status",
                      time: /* @__PURE__ */ new Date(),
                      ...complete,
                      stage: status2?.stage,
                      queue: true,
                      endpoint: _endpoint,
                      fn_index
                    });
                    close();
                  }
                }
                if (status2?.stage === "complete" || status2?.stage === "error") {
                  if (event_callbacks[event_id]) {
                    delete event_callbacks[event_id];
                  }
                  if (event_id in pending_diff_streams) {
                    delete pending_diff_streams[event_id];
                  }
                }
              } catch (e) {
                console.error("Unexpected client exception", e);
                fire_event({
                  type: "status",
                  stage: "error",
                  message: "An Unexpected Error Occurred!",
                  queue: true,
                  endpoint: _endpoint,
                  fn_index,
                  time: /* @__PURE__ */ new Date()
                });
                if (["sse_v2", "sse_v2.1", "sse_v3"].includes(protocol)) {
                  close_stream(stream_status, that.abort_controller);
                  stream_status.open = false;
                  close();
                }
              }
            };
            if (event_id in pending_stream_messages) {
              pending_stream_messages[event_id].forEach((msg) => callback(msg));
              delete pending_stream_messages[event_id];
            }
            event_callbacks[event_id] = callback;
            unclosed_events.add(event_id);
            if (!stream_status.open) {
              await this.open_stream();
            }
          }
        });
      }
    });
    let done = false;
    const values = [];
    const resolvers = [];
    const iterator = {
      [Symbol.asyncIterator]: () => iterator,
      next,
      throw: async (value) => {
        push_error(value);
        return next();
      },
      return: async () => {
        close();
        return { value: void 0, done: true };
      },
      cancel,
      send_chunk: (payload2) => {
        this.post_data(`${config.root}${api_prefix}/stream/${event_id_final}`, {
          ...payload2,
          session_hash: this.session_hash
        });
      },
      close_stream: () => {
        this.post_data(
          `${config.root}${api_prefix}/stream/${event_id_final}/close`,
          {}
        );
        close();
      },
      event_id: () => event_id_final,
      wait_for_id: async () => {
        await job;
        return event_id;
      }
    };
    return iterator;
  } catch (error) {
    console.error("Submit function encountered an error:", error);
    throw error;
  }
}
function thenable_reject(error) {
  return {
    then: (resolve, reject) => reject(error)
  };
}
function get_endpoint_info(api_info, endpoint, api_map, config) {
  let fn_index;
  let endpoint_info;
  let dependency;
  if (typeof endpoint === "number") {
    fn_index = endpoint;
    endpoint_info = api_info.unnamed_endpoints[fn_index];
    dependency = config.dependencies.find((dep) => dep.id == endpoint);
  } else {
    const trimmed_endpoint = endpoint.replace(/^\//, "");
    fn_index = api_map[trimmed_endpoint];
    endpoint_info = api_info.named_endpoints[endpoint.trim()];
    dependency = config.dependencies.find(
      (dep) => dep.id == api_map[trimmed_endpoint]
    );
  }
  if (typeof fn_index !== "number") {
    throw new Error(
      "There is no endpoint matching that name of fn_index matching that number."
    );
  }
  return { fn_index, endpoint_info, dependency };
}

class Client {
  app_reference;
  options;
  deep_link = null;
  config;
  api_prefix = "";
  api_info;
  api_map = {};
  session_hash = Math.random().toString(36).substring(2);
  jwt = false;
  last_status = {};
  cookies = null;
  // streaming
  stream_status = { open: false };
  closed = false;
  pending_stream_messages = {};
  pending_diff_streams = {};
  event_callbacks = {};
  unclosed_events = /* @__PURE__ */ new Set();
  heartbeat_event = null;
  abort_controller = null;
  stream_instance = null;
  current_payload;
  get_url_config(url = null) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    if (url === null) {
      url = window.location.href;
    }
    const stripSlashes = (str) => str.replace(/^\/+|\/+$/g, "");
    let root_path = stripSlashes(new URL(this.config.root).pathname);
    let url_path = stripSlashes(new URL(url).pathname);
    let page;
    if (!url_path.startsWith(root_path)) {
      page = "";
    } else {
      page = stripSlashes(url_path.substring(root_path.length));
    }
    return this.get_page_config(page);
  }
  get_page_config(page) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    let config = this.config;
    if (!(page in config.page)) {
      page = "";
    }
    return {
      ...config,
      current_page: page,
      layout: config.page[page].layout,
      components: config.components.filter(
        (c) => config.page[page].components.includes(c.id)
      ),
      dependencies: this.config.dependencies.filter(
        (d) => config.page[page].dependencies.includes(d.id)
      )
    };
  }
  fetch(input, init) {
    const headers = new Headers(init?.headers || {});
    if (this && this.cookies) {
      headers.append("Cookie", this.cookies);
    }
    if (this && this.options.headers) {
      let additional_headers = new Headers(this.options.headers);
      additional_headers.forEach((value, name) => {
        headers.append(name, value);
      });
    }
    return fetch(input, { ...init, headers });
  }
  stream(url) {
    const headers = new Headers();
    if (this && this.cookies) {
      headers.append("Cookie", this.cookies);
    }
    if (this && this.options.headers) {
      let additional_headers = new Headers(this.options.headers);
      additional_headers.forEach((value, name) => {
        headers.append(name, value);
      });
    }
    if (this && this.options.token) {
      headers.append("Authorization", `Bearer ${this.options.token}`);
    }
    this.abort_controller = new AbortController();
    this.stream_instance = readable_stream(url.toString(), {
      credentials: "include",
      headers,
      signal: this.abort_controller.signal
    });
    return this.stream_instance;
  }
  view_api;
  upload_files;
  upload;
  handle_blob;
  post_data;
  submit;
  predict;
  open_stream;
  resolve_config;
  resolve_cookies;
  constructor(app_reference, options = { events: ["data"] }) {
    this.app_reference = app_reference;
    this.deep_link = options.query_params?.deep_link || null;
    if (!options.events) {
      options.events = ["data"];
    }
    this.options = options;
    this.current_payload = {};
    if (options.cookies) {
      this.cookies = options.cookies;
    }
    this.view_api = view_api.bind(this);
    this.upload_files = upload_files.bind(this);
    this.handle_blob = handle_blob.bind(this);
    this.post_data = post_data.bind(this);
    this.submit = submit.bind(this);
    this.predict = predict.bind(this);
    this.open_stream = open_stream.bind(this);
    this.resolve_config = resolve_config.bind(this);
    this.resolve_cookies = resolve_cookies.bind(this);
    this.upload = upload.bind(this);
    this.fetch = this.fetch.bind(this);
    this.handle_space_success = this.handle_space_success.bind(this);
    this.stream = this.stream.bind(this);
  }
  async init() {
    if (this.options.auth) {
      await this.resolve_cookies();
    }
    await this._resolve_config().then(
      ({ config }) => this._resolve_heartbeat(config)
    );
    this.api_info = await this.view_api();
    this.api_map = map_names_to_ids(this.config?.dependencies || []);
  }
  async _resolve_heartbeat(_config) {
    if (_config) {
      this.config = _config;
      this.api_prefix = _config.api_prefix || "";
      if (this.config && this.config.connect_heartbeat) {
        if (this.config.space_id && this.options.token) {
          this.jwt = await get_jwt(
            this.config.space_id,
            this.options.token,
            this.cookies
          );
        }
      }
    }
    if (_config.space_id && this.options.token) {
      this.jwt = await get_jwt(_config.space_id, this.options.token);
    }
    if (this.config && this.config.connect_heartbeat) {
      const heartbeat_url = new URL(
        `${this.config.root}${this.api_prefix}/${HEARTBEAT_URL}/${this.session_hash}`
      );
      if (this.jwt) {
        heartbeat_url.searchParams.set("__sign", this.jwt);
      }
      if (!this.heartbeat_event) {
        this.heartbeat_event = this.stream(heartbeat_url);
      }
    }
  }
  static async connect(app_reference, options = {
    events: ["data"]
  }) {
    const client2 = new this(app_reference, options);
    if (options.session_hash) {
      client2.session_hash = options.session_hash;
    }
    await client2.init();
    return client2;
  }
  async reconnect() {
    const app_id_url = new URL(
      `${this.config.root}${this.api_prefix}/${APP_ID_URL}`
    );
    let app_id;
    try {
      const response = await this.fetch(app_id_url);
      if (!response.ok) {
        throw new Error();
      }
      app_id = (await response.json()).app_id;
    } catch (e) {
      return "broken";
    }
    if (app_id !== this.config.app_id) {
      return "changed";
    }
    return "connected";
  }
  close() {
    this.closed = true;
    close_stream(this.stream_status, this.abort_controller);
  }
  set_current_payload(payload) {
    this.current_payload = payload;
  }
  static async duplicate(app_reference, options = {
    events: ["data"]
  }) {
    return duplicate(app_reference, options);
  }
  async _resolve_config() {
    const { http_protocol, host, space_id } = await process_endpoint(
      this.app_reference,
      this.options.token
    );
    const { status_callback } = this.options;
    if (space_id && status_callback) {
      await check_and_wake_space(space_id, status_callback);
    }
    let config;
    try {
      let configUrl = `${http_protocol}//${host}`;
      config = await this.resolve_config(configUrl);
      if (!config) {
        throw new Error(CONFIG_ERROR_MSG);
      }
      return this.config_success(config);
    } catch (e) {
      if (space_id && status_callback) {
        check_space_status(
          space_id,
          RE_SPACE_NAME.test(space_id) ? "space_name" : "subdomain",
          this.handle_space_success
        );
      } else {
        if (status_callback)
          status_callback({
            status: "error",
            message: "Could not load this space.",
            load_status: "error",
            detail: "NOT_FOUND"
          });
        throw Error(e);
      }
    }
  }
  async config_success(_config) {
    this.config = _config;
    this.api_prefix = _config.api_prefix || "";
    if (this.config.auth_required) {
      return this.prepare_return_obj();
    }
    try {
      this.api_info = await this.view_api();
    } catch (e) {
      console.error(API_INFO_ERROR_MSG + e.message);
    }
    return this.prepare_return_obj();
  }
  async handle_space_success(status) {
    if (!this) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    const { status_callback } = this.options;
    if (status_callback) status_callback(status);
    if (status.status === "running") {
      try {
        this.config = await this._resolve_config();
        this.api_prefix = this?.config?.api_prefix || "";
        if (!this.config) {
          throw new Error(CONFIG_ERROR_MSG);
        }
        const _config = await this.config_success(this.config);
        return _config;
      } catch (e) {
        if (status_callback) {
          status_callback({
            status: "error",
            message: "Could not load this space.",
            load_status: "error",
            detail: "NOT_FOUND"
          });
        }
        throw e;
      }
    }
  }
  async component_server(component_id, fn_name, data) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    const headers = {};
    const { token } = this.options;
    const { session_hash } = this;
    if (token) {
      headers.Authorization = `Bearer ${this.options.token}`;
    }
    let root_url;
    let component = this.config.components.find(
      (comp) => comp.id === component_id
    );
    if (component?.props?.root_url) {
      root_url = component.props.root_url;
    } else {
      root_url = this.config.root;
    }
    let body;
    if (typeof data === "object" && data !== null && "binary" in data) {
      const _data = data;
      body = new FormData();
      for (const key in _data.data) {
        if (key === "binary") continue;
        body.append(key, _data.data[key]);
      }
      body.set("component_id", component_id.toString());
      body.set("fn_name", fn_name);
      body.set("session_hash", session_hash);
    } else {
      body = JSON.stringify({
        data,
        component_id,
        fn_name,
        session_hash
      });
      headers["Content-Type"] = "application/json";
    }
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    try {
      const response = await this.fetch(
        `${root_url}${this.api_prefix}/${COMPONENT_SERVER_URL}/`,
        {
          method: "POST",
          body,
          headers,
          credentials: "include"
        }
      );
      if (!response.ok) {
        throw new Error(
          "Could not connect to component server: " + response.statusText
        );
      }
      const output = await response.json();
      return output;
    } catch (e) {
      console.warn(e);
    }
  }
  set_cookies(raw_cookies) {
    this.cookies = parse_and_set_cookies(raw_cookies).join("; ");
  }
  prepare_return_obj() {
    return {
      config: this.config,
      predict: this.predict,
      submit: this.submit,
      view_api: this.view_api,
      component_server: this.component_server
    };
  }
}

const node_env = "production";
const DEV = !node_env.toLowerCase().startsWith("prod");

// Store the references to globals in case someone tries to monkey patch these, causing the below
// to de-opt (this occurs often when using popular extensions).
var is_array = Array.isArray;
var index_of = Array.prototype.indexOf;
var array_from = Array.from;
var define_property = Object.defineProperty;
var get_descriptor = Object.getOwnPropertyDescriptor;
var get_descriptors = Object.getOwnPropertyDescriptors;
var object_prototype = Object.prototype;
var array_prototype = Array.prototype;
var get_prototype_of = Object.getPrototypeOf;
var is_extensible = Object.isExtensible;

/**
 * @param {any} thing
 * @returns {thing is Function}
 */
function is_function(thing) {
	return typeof thing === 'function';
}

const noop = () => {};

// Adapted from https://github.com/then/is-promise/blob/master/index.js
// Distributed under MIT License https://github.com/then/is-promise/blob/master/LICENSE

/**
 * @template [T=any]
 * @param {any} value
 * @returns {value is PromiseLike<T>}
 */
function is_promise(value) {
	return typeof value?.then === 'function';
}

/** @param {Function} fn */
function run$1(fn) {
	return fn();
}

/** @param {Array<() => void>} arr */
function run_all(arr) {
	for (var i = 0; i < arr.length; i++) {
		arr[i]();
	}
}

/**
 * TODO replace with Promise.withResolvers once supported widely enough
 * @template [T=void]
 */
function deferred() {
	/** @type {(value: T) => void} */
	var resolve;

	/** @type {(reason: any) => void} */
	var reject;

	/** @type {Promise<T>} */
	var promise = new Promise((res, rej) => {
		resolve = res;
		reject = rej;
	});

	// @ts-expect-error
	return { promise, resolve, reject };
}

/**
 * When encountering a situation like `let [a, b, c] = $derived(blah())`,
 * we need to stash an intermediate value that `a`, `b`, and `c` derive
 * from, in case it's an iterable
 * @template T
 * @param {ArrayLike<T> | Iterable<T>} value
 * @param {number} [n]
 * @returns {Array<T>}
 */
function to_array(value, n) {
	// return arrays unchanged
	if (Array.isArray(value)) {
		return value;
	}

	// if value is not iterable, or `n` is unspecified (indicates a rest
	// element, which means we're not concerned about unbounded iterables)
	// convert to an array with `Array.from`
	if (n === undefined || !(Symbol.iterator in value)) {
		return Array.from(value);
	}

	// otherwise, populate an array with `n` values

	/** @type {T[]} */
	const array = [];

	for (const element of value) {
		array.push(element);
		if (array.length === n) break;
	}

	return array;
}

// General flags
const DERIVED = 1 << 1;
const EFFECT = 1 << 2;
const RENDER_EFFECT = 1 << 3;
/**
 * An effect that does not destroy its child effects when it reruns.
 * Runs as part of render effects, i.e. not eagerly as part of tree traversal or effect flushing.
 */
const MANAGED_EFFECT = 1 << 24;
/**
 * An effect that does not destroy its child effects when it reruns (like MANAGED_EFFECT).
 * Runs eagerly as part of tree traversal or effect flushing.
 */
const BLOCK_EFFECT = 1 << 4;
const BRANCH_EFFECT = 1 << 5;
const ROOT_EFFECT = 1 << 6;
const BOUNDARY_EFFECT = 1 << 7;
/**
 * Indicates that a reaction is connected to an effect root — either it is an effect,
 * or it is a derived that is depended on by at least one effect. If a derived has
 * no dependents, we can disconnect it from the graph, allowing it to either be
 * GC'd or reconnected later if an effect comes to depend on it again
 */
const CONNECTED = 1 << 9;
const CLEAN = 1 << 10;
const DIRTY = 1 << 11;
const MAYBE_DIRTY = 1 << 12;
const INERT = 1 << 13;
const DESTROYED = 1 << 14;

// Flags exclusive to effects
/** Set once an effect that should run synchronously has run */
const EFFECT_RAN = 1 << 15;
/**
 * 'Transparent' effects do not create a transition boundary.
 * This is on a block effect 99% of the time but may also be on a branch effect if its parent block effect was pruned
 */
const EFFECT_TRANSPARENT = 1 << 16;
const EAGER_EFFECT = 1 << 17;
const HEAD_EFFECT = 1 << 18;
const EFFECT_PRESERVED = 1 << 19;
const USER_EFFECT = 1 << 20;
const EFFECT_OFFSCREEN = 1 << 25;

// Flags exclusive to deriveds
/**
 * Tells that we marked this derived and its reactions as visited during the "mark as (maybe) dirty"-phase.
 * Will be lifted during execution of the derived and during checking its dirty state (both are necessary
 * because a derived might be checked but not executed).
 */
const WAS_MARKED = 1 << 15;

// Flags used for async
const REACTION_IS_UPDATING = 1 << 21;
const ASYNC = 1 << 22;

const ERROR_VALUE = 1 << 23;

const STATE_SYMBOL = Symbol('$state');
const LEGACY_PROPS = Symbol('legacy props');
const LOADING_ATTR_SYMBOL = Symbol('');
const PROXY_PATH_SYMBOL = Symbol('proxy path');

/** allow users to ignore aborted signal errors if `reason.name === 'StaleReactionError` */
const STALE_REACTION = new (class StaleReactionError extends Error {
	name = 'StaleReactionError';
	message = 'The reaction that called `getAbortSignal()` was re-run or destroyed';
})();

const ELEMENT_NODE = 1;
const TEXT_NODE = 3;
const COMMENT_NODE = 8;

/* This file is generated by scripts/process-messages/index.js. Do not edit! */


/**
 * `%name%(...)` can only be used during component initialisation
 * @param {string} name
 * @returns {never}
 */
function lifecycle_outside_component(name) {
	if (DEV) {
		const error = new Error(`lifecycle_outside_component\n\`${name}(...)\` can only be used during component initialisation\nhttps://svelte.dev/e/lifecycle_outside_component`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/lifecycle_outside_component`);
	}
}

/* This file is generated by scripts/process-messages/index.js. Do not edit! */


/**
 * Cannot create a `$derived(...)` with an `await` expression outside of an effect tree
 * @returns {never}
 */
function async_derived_orphan() {
	if (DEV) {
		const error = new Error(`async_derived_orphan\nCannot create a \`$derived(...)\` with an \`await\` expression outside of an effect tree\nhttps://svelte.dev/e/async_derived_orphan`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/async_derived_orphan`);
	}
}

/**
 * Using `bind:value` together with a checkbox input is not allowed. Use `bind:checked` instead
 * @returns {never}
 */
function bind_invalid_checkbox_value() {
	if (DEV) {
		const error = new Error(`bind_invalid_checkbox_value\nUsing \`bind:value\` together with a checkbox input is not allowed. Use \`bind:checked\` instead\nhttps://svelte.dev/e/bind_invalid_checkbox_value`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/bind_invalid_checkbox_value`);
	}
}

/**
 * A derived value cannot reference itself recursively
 * @returns {never}
 */
function derived_references_self() {
	if (DEV) {
		const error = new Error(`derived_references_self\nA derived value cannot reference itself recursively\nhttps://svelte.dev/e/derived_references_self`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/derived_references_self`);
	}
}

/**
 * `%rune%` cannot be used inside an effect cleanup function
 * @param {string} rune
 * @returns {never}
 */
function effect_in_teardown(rune) {
	if (DEV) {
		const error = new Error(`effect_in_teardown\n\`${rune}\` cannot be used inside an effect cleanup function\nhttps://svelte.dev/e/effect_in_teardown`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/effect_in_teardown`);
	}
}

/**
 * Effect cannot be created inside a `$derived` value that was not itself created inside an effect
 * @returns {never}
 */
function effect_in_unowned_derived() {
	if (DEV) {
		const error = new Error(`effect_in_unowned_derived\nEffect cannot be created inside a \`$derived\` value that was not itself created inside an effect\nhttps://svelte.dev/e/effect_in_unowned_derived`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/effect_in_unowned_derived`);
	}
}

/**
 * `%rune%` can only be used inside an effect (e.g. during component initialisation)
 * @param {string} rune
 * @returns {never}
 */
function effect_orphan(rune) {
	if (DEV) {
		const error = new Error(`effect_orphan\n\`${rune}\` can only be used inside an effect (e.g. during component initialisation)\nhttps://svelte.dev/e/effect_orphan`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/effect_orphan`);
	}
}

/**
 * Maximum update depth exceeded. This typically indicates that an effect reads and writes the same piece of state
 * @returns {never}
 */
function effect_update_depth_exceeded() {
	if (DEV) {
		const error = new Error(`effect_update_depth_exceeded\nMaximum update depth exceeded. This typically indicates that an effect reads and writes the same piece of state\nhttps://svelte.dev/e/effect_update_depth_exceeded`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/effect_update_depth_exceeded`);
	}
}

/**
 * Could not `{@render}` snippet due to the expression being `null` or `undefined`. Consider using optional chaining `{@render snippet?.()}`
 * @returns {never}
 */
function invalid_snippet() {
	if (DEV) {
		const error = new Error(`invalid_snippet\nCould not \`{@render}\` snippet due to the expression being \`null\` or \`undefined\`. Consider using optional chaining \`{@render snippet?.()}\`\nhttps://svelte.dev/e/invalid_snippet`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/invalid_snippet`);
	}
}

/**
 * `%name%(...)` cannot be used in runes mode
 * @param {string} name
 * @returns {never}
 */
function lifecycle_legacy_only(name) {
	if (DEV) {
		const error = new Error(`lifecycle_legacy_only\n\`${name}(...)\` cannot be used in runes mode\nhttps://svelte.dev/e/lifecycle_legacy_only`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/lifecycle_legacy_only`);
	}
}

/**
 * Cannot do `bind:%key%={undefined}` when `%key%` has a fallback value
 * @param {string} key
 * @returns {never}
 */
function props_invalid_value(key) {
	if (DEV) {
		const error = new Error(`props_invalid_value\nCannot do \`bind:${key}={undefined}\` when \`${key}\` has a fallback value\nhttps://svelte.dev/e/props_invalid_value`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/props_invalid_value`);
	}
}

/**
 * Rest element properties of `$props()` such as `%property%` are readonly
 * @param {string} property
 * @returns {never}
 */
function props_rest_readonly(property) {
	if (DEV) {
		const error = new Error(`props_rest_readonly\nRest element properties of \`$props()\` such as \`${property}\` are readonly\nhttps://svelte.dev/e/props_rest_readonly`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/props_rest_readonly`);
	}
}

/**
 * The `%rune%` rune is only available inside `.svelte` and `.svelte.js/ts` files
 * @param {string} rune
 * @returns {never}
 */
function rune_outside_svelte(rune) {
	if (DEV) {
		const error = new Error(`rune_outside_svelte\nThe \`${rune}\` rune is only available inside \`.svelte\` and \`.svelte.js/ts\` files\nhttps://svelte.dev/e/rune_outside_svelte`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/rune_outside_svelte`);
	}
}

/**
 * `setContext` must be called when a component first initializes, not in a subsequent effect or after an `await` expression
 * @returns {never}
 */
function set_context_after_init() {
	if (DEV) {
		const error = new Error(`set_context_after_init\n\`setContext\` must be called when a component first initializes, not in a subsequent effect or after an \`await\` expression\nhttps://svelte.dev/e/set_context_after_init`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/set_context_after_init`);
	}
}

/**
 * Property descriptors defined on `$state` objects must contain `value` and always be `enumerable`, `configurable` and `writable`.
 * @returns {never}
 */
function state_descriptors_fixed() {
	if (DEV) {
		const error = new Error(`state_descriptors_fixed\nProperty descriptors defined on \`$state\` objects must contain \`value\` and always be \`enumerable\`, \`configurable\` and \`writable\`.\nhttps://svelte.dev/e/state_descriptors_fixed`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/state_descriptors_fixed`);
	}
}

/**
 * Cannot set prototype of `$state` object
 * @returns {never}
 */
function state_prototype_fixed() {
	if (DEV) {
		const error = new Error(`state_prototype_fixed\nCannot set prototype of \`$state\` object\nhttps://svelte.dev/e/state_prototype_fixed`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/state_prototype_fixed`);
	}
}

/**
 * Updating state inside `$derived(...)`, `$inspect(...)` or a template expression is forbidden. If the value should not be reactive, declare it without `$state`
 * @returns {never}
 */
function state_unsafe_mutation() {
	if (DEV) {
		const error = new Error(`state_unsafe_mutation\nUpdating state inside \`$derived(...)\`, \`$inspect(...)\` or a template expression is forbidden. If the value should not be reactive, declare it without \`$state\`\nhttps://svelte.dev/e/state_unsafe_mutation`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/state_unsafe_mutation`);
	}
}

/**
 * A `<svelte:boundary>` `reset` function cannot be called while an error is still being handled
 * @returns {never}
 */
function svelte_boundary_reset_onerror() {
	if (DEV) {
		const error = new Error(`svelte_boundary_reset_onerror\nA \`<svelte:boundary>\` \`reset\` function cannot be called while an error is still being handled\nhttps://svelte.dev/e/svelte_boundary_reset_onerror`);

		error.name = 'Svelte error';

		throw error;
	} else {
		throw new Error(`https://svelte.dev/e/svelte_boundary_reset_onerror`);
	}
}

const EACH_ITEM_REACTIVE = 1;
const EACH_INDEX_REACTIVE = 1 << 1;
/** See EachBlock interface metadata.is_controlled for an explanation what this is */
const EACH_IS_CONTROLLED = 1 << 2;
const EACH_IS_ANIMATED = 1 << 3;
const EACH_ITEM_IMMUTABLE = 1 << 4;

const PROPS_IS_IMMUTABLE = 1;
const PROPS_IS_RUNES = 1 << 1;
const PROPS_IS_UPDATED = 1 << 2;
const PROPS_IS_BINDABLE = 1 << 3;
const PROPS_IS_LAZY_INITIAL = 1 << 4;

const TRANSITION_IN = 1;
const TRANSITION_OUT = 1 << 1;
const TRANSITION_GLOBAL = 1 << 2;

const TEMPLATE_FRAGMENT = 1;
const TEMPLATE_USE_IMPORT_NODE = 1 << 1;

const HYDRATION_START = '[';
/** used to indicate that an `{:else}...` block was rendered */
const HYDRATION_START_ELSE = '[!';
const HYDRATION_END = ']';
const HYDRATION_ERROR = {};

const UNINITIALIZED = Symbol();

// Dev-time component properties
const FILENAME = Symbol('filename');

const NAMESPACE_HTML = 'http://www.w3.org/1999/xhtml';
const NAMESPACE_SVG = 'http://www.w3.org/2000/svg';

const ATTACHMENT_KEY = '@attach';

/* This file is generated by scripts/process-messages/index.js. Do not edit! */


var bold = 'font-weight: bold';
var normal = 'font-weight: normal';

/**
 * The `%attribute%` attribute on `%html%` changed its value between server and client renders. The client value, `%value%`, will be ignored in favour of the server value
 * @param {string} attribute
 * @param {string} html
 * @param {string} value
 */
function hydration_attribute_changed(attribute, html, value) {
	if (DEV) {
		console.warn(`%c[svelte] hydration_attribute_changed\n%cThe \`${attribute}\` attribute on \`${html}\` changed its value between server and client renders. The client value, \`${value}\`, will be ignored in favour of the server value\nhttps://svelte.dev/e/hydration_attribute_changed`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/hydration_attribute_changed`);
	}
}

/**
 * The value of an `{@html ...}` block %location% changed between server and client renders. The client value will be ignored in favour of the server value
 * @param {string | undefined | null} [location]
 */
function hydration_html_changed(location) {
	if (DEV) {
		console.warn(
			`%c[svelte] hydration_html_changed\n%c${location
				? `The value of an \`{@html ...}\` block ${location} changed between server and client renders. The client value will be ignored in favour of the server value`
				: 'The value of an `{@html ...}` block changed between server and client renders. The client value will be ignored in favour of the server value'}\nhttps://svelte.dev/e/hydration_html_changed`,
			bold,
			normal
		);
	} else {
		console.warn(`https://svelte.dev/e/hydration_html_changed`);
	}
}

/**
 * Hydration failed because the initial UI does not match what was rendered on the server. The error occurred near %location%
 * @param {string | undefined | null} [location]
 */
function hydration_mismatch(location) {
	if (DEV) {
		console.warn(
			`%c[svelte] hydration_mismatch\n%c${'Hydration failed because the initial UI does not match what was rendered on the server'}\nhttps://svelte.dev/e/hydration_mismatch`,
			bold,
			normal
		);
	} else {
		console.warn(`https://svelte.dev/e/hydration_mismatch`);
	}
}

/**
 * Tried to unmount a component that was not mounted
 */
function lifecycle_double_unmount() {
	if (DEV) {
		console.warn(`%c[svelte] lifecycle_double_unmount\n%cTried to unmount a component that was not mounted\nhttps://svelte.dev/e/lifecycle_double_unmount`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/lifecycle_double_unmount`);
	}
}

/**
 * The `value` property of a `<select multiple>` element should be an array, but it received a non-array value. The selection will be kept as is.
 */
function select_multiple_invalid_value() {
	if (DEV) {
		console.warn(`%c[svelte] select_multiple_invalid_value\n%cThe \`value\` property of a \`<select multiple>\` element should be an array, but it received a non-array value. The selection will be kept as is.\nhttps://svelte.dev/e/select_multiple_invalid_value`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/select_multiple_invalid_value`);
	}
}

/**
 * Reactive `$state(...)` proxies and the values they proxy have different identities. Because of this, comparisons with `%operator%` will produce unexpected results
 * @param {string} operator
 */
function state_proxy_equality_mismatch(operator) {
	if (DEV) {
		console.warn(`%c[svelte] state_proxy_equality_mismatch\n%cReactive \`$state(...)\` proxies and the values they proxy have different identities. Because of this, comparisons with \`${operator}\` will produce unexpected results\nhttps://svelte.dev/e/state_proxy_equality_mismatch`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/state_proxy_equality_mismatch`);
	}
}

/**
 * Tried to unmount a state proxy, rather than a component
 */
function state_proxy_unmount() {
	if (DEV) {
		console.warn(`%c[svelte] state_proxy_unmount\n%cTried to unmount a state proxy, rather than a component\nhttps://svelte.dev/e/state_proxy_unmount`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/state_proxy_unmount`);
	}
}

/**
 * A `<svelte:boundary>` `reset` function only resets the boundary the first time it is called
 */
function svelte_boundary_reset_noop() {
	if (DEV) {
		console.warn(`%c[svelte] svelte_boundary_reset_noop\n%cA \`<svelte:boundary>\` \`reset\` function only resets the boundary the first time it is called\nhttps://svelte.dev/e/svelte_boundary_reset_noop`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/svelte_boundary_reset_noop`);
	}
}

/**
 * The `slide` transition does not work correctly for elements with `display: %value%`
 * @param {string} value
 */
function transition_slide_display(value) {
	if (DEV) {
		console.warn(`%c[svelte] transition_slide_display\n%cThe \`slide\` transition does not work correctly for elements with \`display: ${value}\`\nhttps://svelte.dev/e/transition_slide_display`, bold, normal);
	} else {
		console.warn(`https://svelte.dev/e/transition_slide_display`);
	}
}

/** @import { TemplateNode } from '#client' */


/**
 * Use this variable to guard everything related to hydration code so it can be treeshaken out
 * if the user doesn't use the `hydrate` method and these code paths are therefore not needed.
 */
let hydrating = false;

/** @param {boolean} value */
function set_hydrating(value) {
	hydrating = value;
}

/**
 * The node that is currently being hydrated. This starts out as the first node inside the opening
 * <!--[--> comment, and updates each time a component calls `$.child(...)` or `$.sibling(...)`.
 * When entering a block (e.g. `{#if ...}`), `hydrate_node` is the block opening comment; by the
 * time we leave the block it is the closing comment, which serves as the block's anchor.
 * @type {TemplateNode}
 */
let hydrate_node;

/** @param {TemplateNode | null} node */
function set_hydrate_node(node) {
	if (node === null) {
		hydration_mismatch();
		throw HYDRATION_ERROR;
	}

	return (hydrate_node = node);
}

function hydrate_next() {
	return set_hydrate_node(get_next_sibling(hydrate_node));
}

/** @param {TemplateNode} node */
function reset(node) {
	if (!hydrating) return;

	// If the node has remaining siblings, something has gone wrong
	if (get_next_sibling(hydrate_node) !== null) {
		hydration_mismatch();
		throw HYDRATION_ERROR;
	}

	hydrate_node = node;
}

function next(count = 1) {
	if (hydrating) {
		var i = count;
		var node = hydrate_node;

		while (i--) {
			node = /** @type {TemplateNode} */ (get_next_sibling(node));
		}

		hydrate_node = node;
	}
}

/**
 * Skips or removes (depending on {@link remove}) all nodes starting at `hydrate_node` up until the next hydration end comment
 * @param {boolean} remove
 */
function skip_nodes(remove = true) {
	var depth = 0;
	var node = hydrate_node;

	while (true) {
		if (node.nodeType === COMMENT_NODE) {
			var data = /** @type {Comment} */ (node).data;

			if (data === HYDRATION_END) {
				if (depth === 0) return node;
				depth -= 1;
			} else if (data === HYDRATION_START || data === HYDRATION_START_ELSE) {
				depth += 1;
			}
		}

		var next = /** @type {TemplateNode} */ (get_next_sibling(node));
		if (remove) node.remove();
		node = next;
	}
}

/**
 *
 * @param {TemplateNode} node
 */
function read_hydration_instruction(node) {
	if (!node || node.nodeType !== COMMENT_NODE) {
		hydration_mismatch();
		throw HYDRATION_ERROR;
	}

	return /** @type {Comment} */ (node).data;
}

/** @import { Equals } from '#client' */

/** @type {Equals} */
function equals(value) {
	return value === this.v;
}

/**
 * @param {unknown} a
 * @param {unknown} b
 * @returns {boolean}
 */
function safe_not_equal(a, b) {
	return a != a
		? b == b
		: a !== b || (a !== null && typeof a === 'object') || typeof a === 'function';
}

/** @type {Equals} */
function safe_equals(value) {
	return !safe_not_equal(value, this.v);
}

/** True if experimental.async=true */
let async_mode_flag = false;
/** True if we're not certain that we only have Svelte 5 code in the compilation */
let legacy_mode_flag = false;
/** True if $inspect.trace is used */
let tracing_mode_flag = false;

function enable_async_mode_flag() {
	async_mode_flag = true;
}

function enable_legacy_mode_flag() {
	legacy_mode_flag = true;
}

/** @import { Derived, Reaction, Value } from '#client' */

/**
 * @param {Value} source
 * @param {string} label
 */
function tag(source, label) {
	source.label = label;
	tag_proxy(source.v, label);

	return source;
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function tag_proxy(value, label) {
	// @ts-expect-error
	value?.[PROXY_PATH_SYMBOL]?.(label);
	return value;
}

/**
 * @param {string} label
 * @returns {Error & { stack: string } | null}
 */
function get_error(label) {
	const error = new Error();
	const stack = get_stack();

	if (stack.length === 0) {
		return null;
	}

	stack.unshift('\n');

	define_property(error, 'stack', {
		value: stack.join('\n')
	});

	define_property(error, 'name', {
		value: label
	});

	return /** @type {Error & { stack: string }} */ (error);
}

/**
 * @returns {string[]}
 */
function get_stack() {
	// @ts-ignore - doesn't exist everywhere
	const limit = Error.stackTraceLimit;
	// @ts-ignore - doesn't exist everywhere
	Error.stackTraceLimit = Infinity;
	const stack = new Error().stack;
	// @ts-ignore - doesn't exist everywhere
	Error.stackTraceLimit = limit;

	if (!stack) return [];

	const lines = stack.split('\n');
	const new_lines = [];

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const posixified = line.replaceAll('\\', '/');

		if (line.trim() === 'Error') {
			continue;
		}

		if (line.includes('validate_each_keys')) {
			return [];
		}

		if (posixified.includes('svelte/src/internal') || posixified.includes('node_modules/.vite')) {
			continue;
		}

		new_lines.push(line);
	}

	return new_lines;
}

/** @import { ComponentContext, DevStackEntry, Effect } from '#client' */

/** @type {ComponentContext | null} */
let component_context = null;

/** @param {ComponentContext | null} context */
function set_component_context(context) {
	component_context = context;
}

/** @type {DevStackEntry | null} */
let dev_stack = null;

/** @param {DevStackEntry | null} stack */
function set_dev_stack(stack) {
	dev_stack = stack;
}

/**
 * The current component function. Different from current component context:
 * ```html
 * <!-- App.svelte -->
 * <Foo>
 *   <Bar /> <!-- context == Foo.svelte, function == App.svelte -->
 * </Foo>
 * ```
 * @type {ComponentContext['function']}
 */
let dev_current_component_function = null;

/** @param {ComponentContext['function']} fn */
function set_dev_current_component_function(fn) {
	dev_current_component_function = fn;
}

/**
 * Retrieves the context that belongs to the closest parent component with the specified `key`.
 * Must be called during component initialisation.
 *
 * [`createContext`](https://svelte.dev/docs/svelte/svelte#createContext) is a type-safe alternative.
 *
 * @template T
 * @param {any} key
 * @returns {T}
 */
function getContext(key) {
	const context_map = get_or_init_context_map('getContext');
	const result = /** @type {T} */ (context_map.get(key));
	return result;
}

/**
 * Associates an arbitrary `context` object with the current component and the specified `key`
 * and returns that object. The context is then available to children of the component
 * (including slotted content) with `getContext`.
 *
 * Like lifecycle functions, this must be called during component initialisation.
 *
 * [`createContext`](https://svelte.dev/docs/svelte/svelte#createContext) is a type-safe alternative.
 *
 * @template T
 * @param {any} key
 * @param {T} context
 * @returns {T}
 */
function setContext(key, context) {
	const context_map = get_or_init_context_map('setContext');

	if (async_mode_flag) {
		var flags = /** @type {Effect} */ (active_effect).f;
		var valid =
			!active_reaction &&
			(flags & BRANCH_EFFECT) !== 0 &&
			// pop() runs synchronously, so this indicates we're setting context after an await
			!(/** @type {ComponentContext} */ (component_context).i);

		if (!valid) {
			set_context_after_init();
		}
	}

	context_map.set(key, context);
	return context;
}

/**
 * @param {Record<string, unknown>} props
 * @param {any} runes
 * @param {Function} [fn]
 * @returns {void}
 */
function push(props, runes = false, fn) {
	component_context = {
		p: component_context,
		i: false,
		c: null,
		e: null,
		s: props,
		x: null,
		l: legacy_mode_flag && !runes ? { s: null, u: null, $: [] } : null
	};

	if (DEV) {
		// component function
		component_context.function = fn;
		dev_current_component_function = fn;
	}
}

/**
 * @template {Record<string, any>} T
 * @param {T} [component]
 * @returns {T}
 */
function pop(component) {
	var context = /** @type {ComponentContext} */ (component_context);
	var effects = context.e;

	if (effects !== null) {
		context.e = null;

		for (var fn of effects) {
			create_user_effect(fn);
		}
	}

	if (component !== undefined) {
		context.x = component;
	}

	context.i = true;

	component_context = context.p;

	if (DEV) {
		dev_current_component_function = component_context?.function ?? null;
	}

	return component ?? /** @type {T} */ ({});
}

/** @returns {boolean} */
function is_runes() {
	return !legacy_mode_flag || (component_context !== null && component_context.l === null);
}

/**
 * @param {string} name
 * @returns {Map<unknown, unknown>}
 */
function get_or_init_context_map(name) {
	if (component_context === null) {
		lifecycle_outside_component(name);
	}

	return (component_context.c ??= new Map(get_parent_context(component_context) || undefined));
}

/**
 * @param {ComponentContext} component_context
 * @returns {Map<unknown, unknown> | null}
 */
function get_parent_context(component_context) {
	let parent = component_context.p;
	while (parent !== null) {
		const context_map = parent.c;
		if (context_map !== null) {
			return context_map;
		}
		parent = parent.p;
	}
	return null;
}

/** @type {Array<() => void>} */
let micro_tasks = [];

function run_micro_tasks() {
	var tasks = micro_tasks;
	micro_tasks = [];
	run_all(tasks);
}

/**
 * @param {() => void} fn
 */
function queue_micro_task(fn) {
	if (micro_tasks.length === 0 && !is_flushing_sync) {
		var tasks = micro_tasks;
		queueMicrotask(() => {
			// If this is false, a flushSync happened in the meantime. Do _not_ run new scheduled microtasks in that case
			// as the ordering of microtasks would be broken at that point - consider this case:
			// - queue_micro_task schedules microtask A to flush task X
			// - synchronously after, flushSync runs, processing task X
			// - synchronously after, some other microtask B is scheduled, but not through queue_micro_task but for example a Promise.resolve() in user code
			// - synchronously after, queue_micro_task schedules microtask C to flush task Y
			// - one tick later, microtask A now resolves, flushing task Y before microtask B, which is incorrect
			// This if check prevents that race condition (that realistically will only happen in tests)
			if (tasks === micro_tasks) run_micro_tasks();
		});
	}

	micro_tasks.push(fn);
}

/**
 * Synchronously run any queued tasks.
 */
function flush_tasks() {
	while (micro_tasks.length > 0) {
		run_micro_tasks();
	}
}

/** @import { Derived, Effect } from '#client' */
/** @import { Boundary } from './dom/blocks/boundary.js' */

const adjustments = new WeakMap();

/**
 * @param {unknown} error
 */
function handle_error(error) {
	var effect = active_effect;

	// for unowned deriveds, don't throw until we read the value
	if (effect === null) {
		/** @type {Derived} */ (active_reaction).f |= ERROR_VALUE;
		return error;
	}

	if (DEV && error instanceof Error && !adjustments.has(error)) {
		adjustments.set(error, get_adjustments(error, effect));
	}

	if ((effect.f & EFFECT_RAN) === 0) {
		// if the error occurred while creating this subtree, we let it
		// bubble up until it hits a boundary that can handle it
		if ((effect.f & BOUNDARY_EFFECT) === 0) {
			if (DEV && !effect.parent && error instanceof Error) {
				apply_adjustments(error);
			}

			throw error;
		}

		/** @type {Boundary} */ (effect.b).error(error);
	} else {
		// otherwise we bubble up the effect tree ourselves
		invoke_error_boundary(error, effect);
	}
}

/**
 * @param {unknown} error
 * @param {Effect | null} effect
 */
function invoke_error_boundary(error, effect) {
	while (effect !== null) {
		if ((effect.f & BOUNDARY_EFFECT) !== 0) {
			try {
				/** @type {Boundary} */ (effect.b).error(error);
				return;
			} catch (e) {
				error = e;
			}
		}

		effect = effect.parent;
	}

	if (DEV && error instanceof Error) {
		apply_adjustments(error);
	}

	throw error;
}

/**
 * Add useful information to the error message/stack in development
 * @param {Error} error
 * @param {Effect} effect
 */
function get_adjustments(error, effect) {
	const message_descriptor = get_descriptor(error, 'message');

	// if the message was already changed and it's not configurable we can't change it
	// or it will throw a different error swallowing the original error
	if (message_descriptor && !message_descriptor.configurable) return;

	var indent = is_firefox ? '  ' : '\t';
	var component_stack = `\n${indent}in ${effect.fn?.name || '<unknown>'}`;
	var context = effect.ctx;

	while (context !== null) {
		component_stack += `\n${indent}in ${context.function?.[FILENAME].split('/').pop()}`;
		context = context.p;
	}

	return {
		message: error.message + `\n${component_stack}\n`,
		stack: error.stack
			?.split('\n')
			.filter((line) => !line.includes('svelte/src/internal'))
			.join('\n')
	};
}

/**
 * @param {Error} error
 */
function apply_adjustments(error) {
	const adjusted = adjustments.get(error);

	if (adjusted) {
		define_property(error, 'message', {
			value: adjusted.message
		});

		define_property(error, 'stack', {
			value: adjusted.stack
		});
	}
}

/** @import { Derived, Signal } from '#client' */

const STATUS_MASK = -7169;

/**
 * @param {Signal} signal
 * @param {number} status
 */
function set_signal_status(signal, status) {
	signal.f = (signal.f & STATUS_MASK) | status;
}

/**
 * Set a derived's status to CLEAN or MAYBE_DIRTY based on its connection state.
 * @param {Derived} derived
 */
function update_derived_status(derived) {
	// Only mark as MAYBE_DIRTY if disconnected and has dependencies.
	if ((derived.f & CONNECTED) !== 0 || derived.deps === null) {
		set_signal_status(derived, CLEAN);
	} else {
		set_signal_status(derived, MAYBE_DIRTY);
	}
}

/** @import { Derived, Effect, Value } from '#client' */

/**
 * @param {Value[] | null} deps
 */
function clear_marked(deps) {
	if (deps === null) return;

	for (const dep of deps) {
		if ((dep.f & DERIVED) === 0 || (dep.f & WAS_MARKED) === 0) {
			continue;
		}

		dep.f ^= WAS_MARKED;

		clear_marked(/** @type {Derived} */ (dep).deps);
	}
}

/**
 * @param {Effect} effect
 * @param {Set<Effect>} dirty_effects
 * @param {Set<Effect>} maybe_dirty_effects
 */
function defer_effect(effect, dirty_effects, maybe_dirty_effects) {
	if ((effect.f & DIRTY) !== 0) {
		dirty_effects.add(effect);
	} else if ((effect.f & MAYBE_DIRTY) !== 0) {
		maybe_dirty_effects.add(effect);
	}

	// Since we're not executing these effects now, we need to clear any WAS_MARKED flags
	// so that other batches can correctly reach these effects during their own traversal
	clear_marked(effect.deps);

	// mark as clean so they get scheduled if they depend on pending async state
	set_signal_status(effect, CLEAN);
}

/** @import { Fork } from 'svelte' */
/** @import { Derived, Effect, Reaction, Source, Value } from '#client' */
/** @import { Boundary } from '../dom/blocks/boundary' */

/** @type {Set<Batch>} */
const batches = new Set();

/** @type {Batch | null} */
let current_batch = null;

/**
 * This is needed to avoid overwriting inputs in non-async mode
 * TODO 6.0 remove this, as non-async mode will go away
 * @type {Batch | null}
 */
let previous_batch = null;

/**
 * When time travelling (i.e. working in one batch, while other batches
 * still have ongoing work), we ignore the real values of affected
 * signals in favour of their values within the batch
 * @type {Map<Value, any> | null}
 */
let batch_values = null;

// TODO this should really be a property of `batch`
/** @type {Effect[]} */
let queued_root_effects = [];

/** @type {Effect | null} */
let last_scheduled_effect = null;

let is_flushing = false;
let is_flushing_sync = false;

class Batch {
	committed = false;

	/**
	 * The current values of any sources that are updated in this batch
	 * They keys of this map are identical to `this.#previous`
	 * @type {Map<Source, any>}
	 */
	current = new Map();

	/**
	 * The values of any sources that are updated in this batch _before_ those updates took place.
	 * They keys of this map are identical to `this.#current`
	 * @type {Map<Source, any>}
	 */
	previous = new Map();

	/**
	 * When the batch is committed (and the DOM is updated), we need to remove old branches
	 * and append new ones by calling the functions added inside (if/each/key/etc) blocks
	 * @type {Set<() => void>}
	 */
	#commit_callbacks = new Set();

	/**
	 * If a fork is discarded, we need to destroy any effects that are no longer needed
	 * @type {Set<(batch: Batch) => void>}
	 */
	#discard_callbacks = new Set();

	/**
	 * The number of async effects that are currently in flight
	 */
	#pending = 0;

	/**
	 * The number of async effects that are currently in flight, _not_ inside a pending boundary
	 */
	#blocking_pending = 0;

	/**
	 * A deferred that resolves when the batch is committed, used with `settled()`
	 * TODO replace with Promise.withResolvers once supported widely enough
	 * @type {{ promise: Promise<void>, resolve: (value?: any) => void, reject: (reason: unknown) => void } | null}
	 */
	#deferred = null;

	/**
	 * Deferred effects (which run after async work has completed) that are DIRTY
	 * @type {Set<Effect>}
	 */
	#dirty_effects = new Set();

	/**
	 * Deferred effects that are MAYBE_DIRTY
	 * @type {Set<Effect>}
	 */
	#maybe_dirty_effects = new Set();

	/**
	 * A set of branches that still exist, but will be destroyed when this batch
	 * is committed — we skip over these during `process`
	 * @type {Set<Effect>}
	 */
	skipped_effects = new Set();

	is_fork = false;

	is_deferred() {
		return this.is_fork || this.#blocking_pending > 0;
	}

	/**
	 *
	 * @param {Effect[]} root_effects
	 */
	process(root_effects) {
		queued_root_effects = [];

		previous_batch = null;

		this.apply();

		/** @type {Effect[]} */
		var effects = [];

		/** @type {Effect[]} */
		var render_effects = [];

		for (const root of root_effects) {
			this.#traverse_effect_tree(root, effects, render_effects);
			// Note: #traverse_effect_tree runs block effects eagerly, which can schedule effects,
			// which means queued_root_effects now may be filled again.

			// Helpful for debugging reactivity loss that has to do with branches being skipped:
			// log_inconsistent_branches(root);
		}

		if (!this.is_fork) {
			this.#resolve();
		}

		if (this.is_deferred()) {
			this.#defer_effects(render_effects);
			this.#defer_effects(effects);
		} else {
			// If sources are written to, then work needs to happen in a separate batch, else prior sources would be mixed with
			// newly updated sources, which could lead to infinite loops when effects run over and over again.
			previous_batch = this;
			current_batch = null;

			flush_queued_effects(render_effects);
			flush_queued_effects(effects);

			previous_batch = null;

			this.#deferred?.resolve();
		}

		batch_values = null;
	}

	/**
	 * Traverse the effect tree, executing effects or stashing
	 * them for later execution as appropriate
	 * @param {Effect} root
	 * @param {Effect[]} effects
	 * @param {Effect[]} render_effects
	 */
	#traverse_effect_tree(root, effects, render_effects) {
		root.f ^= CLEAN;

		var effect = root.first;

		/** @type {Effect | null} */
		var pending_boundary = null;

		while (effect !== null) {
			var flags = effect.f;
			var is_branch = (flags & (BRANCH_EFFECT | ROOT_EFFECT)) !== 0;
			var is_skippable_branch = is_branch && (flags & CLEAN) !== 0;

			var skip = is_skippable_branch || (flags & INERT) !== 0 || this.skipped_effects.has(effect);

			// Inside a `<svelte:boundary>` with a pending snippet,
			// all effects are deferred until the boundary resolves
			// (except block/async effects, which run immediately)
			if (
				async_mode_flag &&
				pending_boundary === null &&
				(flags & BOUNDARY_EFFECT) !== 0 &&
				effect.b?.is_pending
			) {
				pending_boundary = effect;
			}

			if (!skip && effect.fn !== null) {
				if (is_branch) {
					effect.f ^= CLEAN;
				} else if (
					pending_boundary !== null &&
					(flags & (EFFECT | RENDER_EFFECT | MANAGED_EFFECT)) !== 0
				) {
					/** @type {Boundary} */ (pending_boundary.b).defer_effect(effect);
				} else if ((flags & EFFECT) !== 0) {
					effects.push(effect);
				} else if (async_mode_flag && (flags & (RENDER_EFFECT | MANAGED_EFFECT)) !== 0) {
					render_effects.push(effect);
				} else if (is_dirty(effect)) {
					if ((flags & BLOCK_EFFECT) !== 0) this.#dirty_effects.add(effect);
					update_effect(effect);
				}

				var child = effect.first;

				if (child !== null) {
					effect = child;
					continue;
				}
			}

			var parent = effect.parent;
			effect = effect.next;

			while (effect === null && parent !== null) {
				if (parent === pending_boundary) {
					pending_boundary = null;
				}

				effect = parent.next;
				parent = parent.parent;
			}
		}
	}

	/**
	 * @param {Effect[]} effects
	 */
	#defer_effects(effects) {
		for (var i = 0; i < effects.length; i += 1) {
			defer_effect(effects[i], this.#dirty_effects, this.#maybe_dirty_effects);
		}
	}

	/**
	 * Associate a change to a given source with the current
	 * batch, noting its previous and current values
	 * @param {Source} source
	 * @param {any} value
	 */
	capture(source, value) {
		if (value !== UNINITIALIZED && !this.previous.has(source)) {
			this.previous.set(source, value);
		}

		// Don't save errors in `batch_values`, or they won't be thrown in `runtime.js#get`
		if ((source.f & ERROR_VALUE) === 0) {
			this.current.set(source, source.v);
			batch_values?.set(source, source.v);
		}
	}

	activate() {
		current_batch = this;
		this.apply();
	}

	deactivate() {
		// If we're not the current batch, don't deactivate,
		// else we could create zombie batches that are never flushed
		if (current_batch !== this) return;

		current_batch = null;
		batch_values = null;
	}

	flush() {
		this.activate();

		if (queued_root_effects.length > 0) {
			flush_effects();

			if (current_batch !== null && current_batch !== this) {
				// this can happen if a new batch was created during `flush_effects()`
				return;
			}
		} else if (this.#pending === 0) {
			this.process([]); // TODO this feels awkward
		}

		this.deactivate();
	}

	discard() {
		for (const fn of this.#discard_callbacks) fn(this);
		this.#discard_callbacks.clear();
	}

	#resolve() {
		if (this.#blocking_pending === 0) {
			// append/remove branches
			for (const fn of this.#commit_callbacks) fn();
			this.#commit_callbacks.clear();
		}

		if (this.#pending === 0) {
			this.#commit();
		}
	}

	#commit() {
		// If there are other pending batches, they now need to be 'rebased' —
		// in other words, we re-run block/async effects with the newly
		// committed state, unless the batch in question has a more
		// recent value for a given source
		if (batches.size > 1) {
			this.previous.clear();

			var previous_batch_values = batch_values;
			var is_earlier = true;

			for (const batch of batches) {
				if (batch === this) {
					is_earlier = false;
					continue;
				}

				/** @type {Source[]} */
				const sources = [];

				for (const [source, value] of this.current) {
					if (batch.current.has(source)) {
						if (is_earlier && value !== batch.current.get(source)) {
							// bring the value up to date
							batch.current.set(source, value);
						} else {
							// same value or later batch has more recent value,
							// no need to re-run these effects
							continue;
						}
					}

					sources.push(source);
				}

				if (sources.length === 0) {
					continue;
				}

				// Re-run async/block effects that depend on distinct values changed in both batches
				const others = [...batch.current.keys()].filter((s) => !this.current.has(s));
				if (others.length > 0) {
					// Avoid running queued root effects on the wrong branch
					var prev_queued_root_effects = queued_root_effects;
					queued_root_effects = [];

					/** @type {Set<Value>} */
					const marked = new Set();
					/** @type {Map<Reaction, boolean>} */
					const checked = new Map();
					for (const source of sources) {
						mark_effects(source, others, marked, checked);
					}

					if (queued_root_effects.length > 0) {
						current_batch = batch;
						batch.apply();

						for (const root of queued_root_effects) {
							batch.#traverse_effect_tree(root, [], []);
						}

						// TODO do we need to do anything with the dummy effect arrays?

						batch.deactivate();
					}

					queued_root_effects = prev_queued_root_effects;
				}
			}

			current_batch = null;
			batch_values = previous_batch_values;
		}

		this.committed = true;
		batches.delete(this);
	}

	/**
	 *
	 * @param {boolean} blocking
	 */
	increment(blocking) {
		this.#pending += 1;
		if (blocking) this.#blocking_pending += 1;
	}

	/**
	 *
	 * @param {boolean} blocking
	 */
	decrement(blocking) {
		this.#pending -= 1;
		if (blocking) this.#blocking_pending -= 1;

		this.revive();
	}

	revive() {
		for (const e of this.#dirty_effects) {
			this.#maybe_dirty_effects.delete(e);
			set_signal_status(e, DIRTY);
			schedule_effect(e);
		}

		for (const e of this.#maybe_dirty_effects) {
			set_signal_status(e, MAYBE_DIRTY);
			schedule_effect(e);
		}

		this.flush();
	}

	/** @param {() => void} fn */
	oncommit(fn) {
		this.#commit_callbacks.add(fn);
	}

	/** @param {(batch: Batch) => void} fn */
	ondiscard(fn) {
		this.#discard_callbacks.add(fn);
	}

	settled() {
		return (this.#deferred ??= deferred()).promise;
	}

	static ensure() {
		if (current_batch === null) {
			const batch = (current_batch = new Batch());
			batches.add(current_batch);

			if (!is_flushing_sync) {
				Batch.enqueue(() => {
					if (current_batch !== batch) {
						// a flushSync happened in the meantime
						return;
					}

					batch.flush();
				});
			}
		}

		return current_batch;
	}

	/** @param {() => void} task */
	static enqueue(task) {
		queue_micro_task(task);
	}

	apply() {
		if (!async_mode_flag || (!this.is_fork && batches.size === 1)) return;

		// if there are multiple batches, we are 'time travelling' —
		// we need to override values with the ones in this batch...
		batch_values = new Map(this.current);

		// ...and undo changes belonging to other batches
		for (const batch of batches) {
			if (batch === this) continue;

			for (const [source, previous] of batch.previous) {
				if (!batch_values.has(source)) {
					batch_values.set(source, previous);
				}
			}
		}
	}
}

/**
 * Synchronously flush any pending updates.
 * Returns void if no callback is provided, otherwise returns the result of calling the callback.
 * @template [T=void]
 * @param {(() => T) | undefined} [fn]
 * @returns {T}
 */
function flushSync(fn) {
	var was_flushing_sync = is_flushing_sync;
	is_flushing_sync = true;

	try {
		var result;

		if (fn) ;

		while (true) {
			flush_tasks();

			if (queued_root_effects.length === 0) {
				current_batch?.flush();

				// we need to check again, in case we just updated an `$effect.pending()`
				if (queued_root_effects.length === 0) {
					// this would be reset in `flush_effects()` but since we are early returning here,
					// we need to reset it here as well in case the first time there's 0 queued root effects
					last_scheduled_effect = null;

					return /** @type {T} */ (result);
				}
			}

			flush_effects();
		}
	} finally {
		is_flushing_sync = was_flushing_sync;
	}
}

function flush_effects() {
	var was_updating_effect = is_updating_effect;
	is_flushing = true;

	var source_stacks = DEV ? new Set() : null;

	try {
		var flush_count = 0;
		set_is_updating_effect(true);

		while (queued_root_effects.length > 0) {
			var batch = Batch.ensure();

			if (flush_count++ > 1000) {
				if (DEV) {
					var updates = new Map();

					for (const source of batch.current.keys()) {
						for (const [stack, update] of source.updated ?? []) {
							var entry = updates.get(stack);

							if (!entry) {
								entry = { error: update.error, count: 0 };
								updates.set(stack, entry);
							}

							entry.count += update.count;
						}
					}

					for (const update of updates.values()) {
						if (update.error) {
							// eslint-disable-next-line no-console
							console.error(update.error);
						}
					}
				}

				infinite_loop_guard();
			}

			batch.process(queued_root_effects);
			old_values.clear();

			if (DEV) {
				for (const source of batch.current.keys()) {
					/** @type {Set<Source>} */ (source_stacks).add(source);
				}
			}
		}
	} finally {
		is_flushing = false;
		set_is_updating_effect(was_updating_effect);

		last_scheduled_effect = null;

		if (DEV) {
			for (const source of /** @type {Set<Source>} */ (source_stacks)) {
				source.updated = null;
			}
		}
	}
}

function infinite_loop_guard() {
	try {
		effect_update_depth_exceeded();
	} catch (error) {
		if (DEV) {
			// stack contains no useful information, replace it
			define_property(error, 'stack', { value: '' });
		}

		// Best effort: invoke the boundary nearest the most recent
		// effect and hope that it's relevant to the infinite loop
		invoke_error_boundary(error, last_scheduled_effect);
	}
}

/** @type {Set<Effect> | null} */
let eager_block_effects = null;

/**
 * @param {Array<Effect>} effects
 * @returns {void}
 */
function flush_queued_effects(effects) {
	var length = effects.length;
	if (length === 0) return;

	var i = 0;

	while (i < length) {
		var effect = effects[i++];

		if ((effect.f & (DESTROYED | INERT)) === 0 && is_dirty(effect)) {
			eager_block_effects = new Set();

			update_effect(effect);

			// Effects with no dependencies or teardown do not get added to the effect tree.
			// Deferred effects (e.g. `$effect(...)`) _are_ added to the tree because we
			// don't know if we need to keep them until they are executed. Doing the check
			// here (rather than in `update_effect`) allows us to skip the work for
			// immediate effects.
			if (effect.deps === null && effect.first === null && effect.nodes === null) {
				// if there's no teardown or abort controller we completely unlink
				// the effect from the graph
				if (effect.teardown === null && effect.ac === null) {
					// remove this effect from the graph
					unlink_effect(effect);
				} else {
					// keep the effect in the graph, but free up some memory
					effect.fn = null;
				}
			}

			// If update_effect() has a flushSync() in it, we may have flushed another flush_queued_effects(),
			// which already handled this logic and did set eager_block_effects to null.
			if (eager_block_effects?.size > 0) {
				old_values.clear();

				for (const e of eager_block_effects) {
					// Skip eager effects that have already been unmounted
					if ((e.f & (DESTROYED | INERT)) !== 0) continue;

					// Run effects in order from ancestor to descendant, else we could run into nullpointers
					/** @type {Effect[]} */
					const ordered_effects = [e];
					let ancestor = e.parent;
					while (ancestor !== null) {
						if (eager_block_effects.has(ancestor)) {
							eager_block_effects.delete(ancestor);
							ordered_effects.push(ancestor);
						}
						ancestor = ancestor.parent;
					}

					for (let j = ordered_effects.length - 1; j >= 0; j--) {
						const e = ordered_effects[j];
						// Skip eager effects that have already been unmounted
						if ((e.f & (DESTROYED | INERT)) !== 0) continue;
						update_effect(e);
					}
				}

				eager_block_effects.clear();
			}
		}
	}

	eager_block_effects = null;
}

/**
 * This is similar to `mark_reactions`, but it only marks async/block effects
 * depending on `value` and at least one of the other `sources`, so that
 * these effects can re-run after another batch has been committed
 * @param {Value} value
 * @param {Source[]} sources
 * @param {Set<Value>} marked
 * @param {Map<Reaction, boolean>} checked
 */
function mark_effects(value, sources, marked, checked) {
	if (marked.has(value)) return;
	marked.add(value);

	if (value.reactions !== null) {
		for (const reaction of value.reactions) {
			const flags = reaction.f;

			if ((flags & DERIVED) !== 0) {
				mark_effects(/** @type {Derived} */ (reaction), sources, marked, checked);
			} else if (
				(flags & (ASYNC | BLOCK_EFFECT)) !== 0 &&
				(flags & DIRTY) === 0 &&
				depends_on(reaction, sources, checked)
			) {
				set_signal_status(reaction, DIRTY);
				schedule_effect(/** @type {Effect} */ (reaction));
			}
		}
	}
}

/**
 * @param {Reaction} reaction
 * @param {Source[]} sources
 * @param {Map<Reaction, boolean>} checked
 */
function depends_on(reaction, sources, checked) {
	const depends = checked.get(reaction);
	if (depends !== undefined) return depends;

	if (reaction.deps !== null) {
		for (const dep of reaction.deps) {
			if (sources.includes(dep)) {
				return true;
			}

			if ((dep.f & DERIVED) !== 0 && depends_on(/** @type {Derived} */ (dep), sources, checked)) {
				checked.set(/** @type {Derived} */ (dep), true);
				return true;
			}
		}
	}

	checked.set(reaction, false);

	return false;
}

/**
 * @param {Effect} signal
 * @returns {void}
 */
function schedule_effect(signal) {
	var effect = (last_scheduled_effect = signal);

	while (effect.parent !== null) {
		effect = effect.parent;
		var flags = effect.f;

		// if the effect is being scheduled because a parent (each/await/etc) block
		// updated an internal source, bail out or we'll cause a second flush
		if (
			is_flushing &&
			effect === active_effect &&
			(flags & BLOCK_EFFECT) !== 0 &&
			(flags & HEAD_EFFECT) === 0
		) {
			return;
		}

		if ((flags & (ROOT_EFFECT | BRANCH_EFFECT)) !== 0) {
			if ((flags & CLEAN) === 0) return;
			effect.f ^= CLEAN;
		}
	}

	queued_root_effects.push(effect);
}

/**
 * Returns a `subscribe` function that integrates external event-based systems with Svelte's reactivity.
 * It's particularly useful for integrating with web APIs like `MediaQuery`, `IntersectionObserver`, or `WebSocket`.
 *
 * If `subscribe` is called inside an effect (including indirectly, for example inside a getter),
 * the `start` callback will be called with an `update` function. Whenever `update` is called, the effect re-runs.
 *
 * If `start` returns a cleanup function, it will be called when the effect is destroyed.
 *
 * If `subscribe` is called in multiple effects, `start` will only be called once as long as the effects
 * are active, and the returned teardown function will only be called when all effects are destroyed.
 *
 * It's best understood with an example. Here's an implementation of [`MediaQuery`](https://svelte.dev/docs/svelte/svelte-reactivity#MediaQuery):
 *
 * ```js
 * import { createSubscriber } from 'svelte/reactivity';
 * import { on } from 'svelte/events';
 *
 * export class MediaQuery {
 * 	#query;
 * 	#subscribe;
 *
 * 	constructor(query) {
 * 		this.#query = window.matchMedia(`(${query})`);
 *
 * 		this.#subscribe = createSubscriber((update) => {
 * 			// when the `change` event occurs, re-run any effects that read `this.current`
 * 			const off = on(this.#query, 'change', update);
 *
 * 			// stop listening when all the effects are destroyed
 * 			return () => off();
 * 		});
 * 	}
 *
 * 	get current() {
 * 		// This makes the getter reactive, if read in an effect
 * 		this.#subscribe();
 *
 * 		// Return the current state of the query, whether or not we're in an effect
 * 		return this.#query.matches;
 * 	}
 * }
 * ```
 * @param {(update: () => void) => (() => void) | void} start
 * @since 5.7.0
 */
function createSubscriber(start) {
	let subscribers = 0;
	let version = source(0);
	/** @type {(() => void) | void} */
	let stop;

	if (DEV) {
		tag(version, 'createSubscriber version');
	}

	return () => {
		if (effect_tracking()) {
			get(version);

			render_effect(() => {
				if (subscribers === 0) {
					stop = untrack(() => start(() => increment(version)));
				}

				subscribers += 1;

				return () => {
					queue_micro_task(() => {
						// Only count down after a microtask, else we would reach 0 before our own render effect reruns,
						// but reach 1 again when the tick callback of the prior teardown runs. That would mean we
						// re-subcribe unnecessarily and create a memory leak because the old subscription is never cleaned up.
						subscribers -= 1;

						if (subscribers === 0) {
							stop?.();
							stop = undefined;
							// Increment the version to ensure any dependent deriveds are marked dirty when the subscription is picked up again later.
							// If we didn't do this then the comparison of write versions would determine that the derived has a later version than
							// the subscriber, and it would not be re-run.
							increment(version);
						}
					});
				};
			});
		}
	};
}

/** @import { Effect, Source, TemplateNode, } from '#client' */

/**
 * @typedef {{
 * 	 onerror?: (error: unknown, reset: () => void) => void;
 *   failed?: (anchor: Node, error: () => unknown, reset: () => () => void) => void;
 *   pending?: (anchor: Node) => void;
 * }} BoundaryProps
 */

var flags = EFFECT_TRANSPARENT | EFFECT_PRESERVED | BOUNDARY_EFFECT;

/**
 * @param {TemplateNode} node
 * @param {BoundaryProps} props
 * @param {((anchor: Node) => void)} children
 * @returns {void}
 */
function boundary(node, props, children) {
	new Boundary(node, props, children);
}

class Boundary {
	/** @type {Boundary | null} */
	parent;

	is_pending = false;

	/** @type {TemplateNode} */
	#anchor;

	/** @type {TemplateNode | null} */
	#hydrate_open = hydrating ? hydrate_node : null;

	/** @type {BoundaryProps} */
	#props;

	/** @type {((anchor: Node) => void)} */
	#children;

	/** @type {Effect} */
	#effect;

	/** @type {Effect | null} */
	#main_effect = null;

	/** @type {Effect | null} */
	#pending_effect = null;

	/** @type {Effect | null} */
	#failed_effect = null;

	/** @type {DocumentFragment | null} */
	#offscreen_fragment = null;

	/** @type {TemplateNode | null} */
	#pending_anchor = null;

	#local_pending_count = 0;
	#pending_count = 0;

	#is_creating_fallback = false;

	/** @type {Set<Effect>} */
	#dirty_effects = new Set();

	/** @type {Set<Effect>} */
	#maybe_dirty_effects = new Set();

	/**
	 * A source containing the number of pending async deriveds/expressions.
	 * Only created if `$effect.pending()` is used inside the boundary,
	 * otherwise updating the source results in needless `Batch.ensure()`
	 * calls followed by no-op flushes
	 * @type {Source<number> | null}
	 */
	#effect_pending = null;

	#effect_pending_subscriber = createSubscriber(() => {
		this.#effect_pending = source(this.#local_pending_count);

		if (DEV) {
			tag(this.#effect_pending, '$effect.pending()');
		}

		return () => {
			this.#effect_pending = null;
		};
	});

	/**
	 * @param {TemplateNode} node
	 * @param {BoundaryProps} props
	 * @param {((anchor: Node) => void)} children
	 */
	constructor(node, props, children) {
		this.#anchor = node;
		this.#props = props;
		this.#children = children;

		this.parent = /** @type {Effect} */ (active_effect).b;

		this.is_pending = !!this.#props.pending;

		this.#effect = block(() => {
			/** @type {Effect} */ (active_effect).b = this;

			if (hydrating) {
				const comment = this.#hydrate_open;
				hydrate_next();

				const server_rendered_pending =
					/** @type {Comment} */ (comment).nodeType === COMMENT_NODE &&
					/** @type {Comment} */ (comment).data === HYDRATION_START_ELSE;

				if (server_rendered_pending) {
					this.#hydrate_pending_content();
				} else {
					this.#hydrate_resolved_content();

					if (this.#pending_count === 0) {
						this.is_pending = false;
					}
				}
			} else {
				var anchor = this.#get_anchor();

				try {
					this.#main_effect = branch(() => children(anchor));
				} catch (error) {
					this.error(error);
				}

				if (this.#pending_count > 0) {
					this.#show_pending_snippet();
				} else {
					this.is_pending = false;
				}
			}

			return () => {
				this.#pending_anchor?.remove();
			};
		}, flags);

		if (hydrating) {
			this.#anchor = hydrate_node;
		}
	}

	#hydrate_resolved_content() {
		try {
			this.#main_effect = branch(() => this.#children(this.#anchor));
		} catch (error) {
			this.error(error);
		}
	}

	#hydrate_pending_content() {
		const pending = this.#props.pending;
		if (!pending) {
			return;
		}
		this.#pending_effect = branch(() => pending(this.#anchor));

		Batch.enqueue(() => {
			var anchor = this.#get_anchor();

			this.#main_effect = this.#run(() => {
				Batch.ensure();
				return branch(() => this.#children(anchor));
			});

			if (this.#pending_count > 0) {
				this.#show_pending_snippet();
			} else {
				pause_effect(/** @type {Effect} */ (this.#pending_effect), () => {
					this.#pending_effect = null;
				});

				this.is_pending = false;
			}
		});
	}

	#get_anchor() {
		var anchor = this.#anchor;

		if (this.is_pending) {
			this.#pending_anchor = create_text();
			this.#anchor.before(this.#pending_anchor);

			anchor = this.#pending_anchor;
		}

		return anchor;
	}

	/**
	 * Defer an effect inside a pending boundary until the boundary resolves
	 * @param {Effect} effect
	 */
	defer_effect(effect) {
		defer_effect(effect, this.#dirty_effects, this.#maybe_dirty_effects);
	}

	/**
	 * Returns `false` if the effect exists inside a boundary whose pending snippet is shown
	 * @returns {boolean}
	 */
	is_rendered() {
		return !this.is_pending && (!this.parent || this.parent.is_rendered());
	}

	has_pending_snippet() {
		return !!this.#props.pending;
	}

	/**
	 * @param {() => Effect | null} fn
	 */
	#run(fn) {
		var previous_effect = active_effect;
		var previous_reaction = active_reaction;
		var previous_ctx = component_context;

		set_active_effect(this.#effect);
		set_active_reaction(this.#effect);
		set_component_context(this.#effect.ctx);

		try {
			return fn();
		} catch (e) {
			handle_error(e);
			return null;
		} finally {
			set_active_effect(previous_effect);
			set_active_reaction(previous_reaction);
			set_component_context(previous_ctx);
		}
	}

	#show_pending_snippet() {
		const pending = /** @type {(anchor: Node) => void} */ (this.#props.pending);

		if (this.#main_effect !== null) {
			this.#offscreen_fragment = document.createDocumentFragment();
			this.#offscreen_fragment.append(/** @type {TemplateNode} */ (this.#pending_anchor));
			move_effect(this.#main_effect, this.#offscreen_fragment);
		}

		if (this.#pending_effect === null) {
			this.#pending_effect = branch(() => pending(this.#anchor));
		}
	}

	/**
	 * Updates the pending count associated with the currently visible pending snippet,
	 * if any, such that we can replace the snippet with content once work is done
	 * @param {1 | -1} d
	 */
	#update_pending_count(d) {
		if (!this.has_pending_snippet()) {
			if (this.parent) {
				this.parent.#update_pending_count(d);
			}

			// if there's no parent, we're in a scope with no pending snippet
			return;
		}

		this.#pending_count += d;

		if (this.#pending_count === 0) {
			this.is_pending = false;

			// any effects that were encountered and deferred during traversal
			// should be rescheduled — after the next traversal (which will happen
			// immediately, due to the same update that brought us here)
			// the effects will be flushed
			for (const e of this.#dirty_effects) {
				set_signal_status(e, DIRTY);
				schedule_effect(e);
			}

			for (const e of this.#maybe_dirty_effects) {
				set_signal_status(e, MAYBE_DIRTY);
				schedule_effect(e);
			}

			this.#dirty_effects.clear();
			this.#maybe_dirty_effects.clear();

			if (this.#pending_effect) {
				pause_effect(this.#pending_effect, () => {
					this.#pending_effect = null;
				});
			}

			if (this.#offscreen_fragment) {
				this.#anchor.before(this.#offscreen_fragment);
				this.#offscreen_fragment = null;
			}
		}
	}

	/**
	 * Update the source that powers `$effect.pending()` inside this boundary,
	 * and controls when the current `pending` snippet (if any) is removed.
	 * Do not call from inside the class
	 * @param {1 | -1} d
	 */
	update_pending_count(d) {
		this.#update_pending_count(d);

		this.#local_pending_count += d;

		if (this.#effect_pending) {
			internal_set(this.#effect_pending, this.#local_pending_count);
		}
	}

	get_effect_pending() {
		this.#effect_pending_subscriber();
		return get(/** @type {Source<number>} */ (this.#effect_pending));
	}

	/** @param {unknown} error */
	error(error) {
		var onerror = this.#props.onerror;
		let failed = this.#props.failed;

		// If we have nothing to capture the error, or if we hit an error while
		// rendering the fallback, re-throw for another boundary to handle
		if (this.#is_creating_fallback || (!onerror && !failed)) {
			throw error;
		}

		if (this.#main_effect) {
			destroy_effect(this.#main_effect);
			this.#main_effect = null;
		}

		if (this.#pending_effect) {
			destroy_effect(this.#pending_effect);
			this.#pending_effect = null;
		}

		if (this.#failed_effect) {
			destroy_effect(this.#failed_effect);
			this.#failed_effect = null;
		}

		if (hydrating) {
			set_hydrate_node(/** @type {TemplateNode} */ (this.#hydrate_open));
			next();
			set_hydrate_node(skip_nodes());
		}

		var did_reset = false;
		var calling_on_error = false;

		const reset = () => {
			if (did_reset) {
				svelte_boundary_reset_noop();
				return;
			}

			did_reset = true;

			if (calling_on_error) {
				svelte_boundary_reset_onerror();
			}

			// If the failure happened while flushing effects, current_batch can be null
			Batch.ensure();

			this.#local_pending_count = 0;

			if (this.#failed_effect !== null) {
				pause_effect(this.#failed_effect, () => {
					this.#failed_effect = null;
				});
			}

			// we intentionally do not try to find the nearest pending boundary. If this boundary has one, we'll render it on reset
			// but it would be really weird to show the parent's boundary on a child reset.
			this.is_pending = this.has_pending_snippet();

			this.#main_effect = this.#run(() => {
				this.#is_creating_fallback = false;
				return branch(() => this.#children(this.#anchor));
			});

			if (this.#pending_count > 0) {
				this.#show_pending_snippet();
			} else {
				this.is_pending = false;
			}
		};

		var previous_reaction = active_reaction;

		try {
			set_active_reaction(null);
			calling_on_error = true;
			onerror?.(error, reset);
			calling_on_error = false;
		} catch (error) {
			invoke_error_boundary(error, this.#effect && this.#effect.parent);
		} finally {
			set_active_reaction(previous_reaction);
		}

		if (failed) {
			queue_micro_task(() => {
				this.#failed_effect = this.#run(() => {
					Batch.ensure();
					this.#is_creating_fallback = true;

					try {
						return branch(() => {
							failed(
								this.#anchor,
								() => error,
								() => reset
							);
						});
					} catch (error) {
						invoke_error_boundary(error, /** @type {Effect} */ (this.#effect.parent));
						return null;
					} finally {
						this.#is_creating_fallback = false;
					}
				});
			});
		}
	}
}

function get_boundary() {
	return /** @type {Boundary} */ (/** @type {Effect} */ (active_effect).b);
}

/** @import { Effect, TemplateNode, Value } from '#client' */

/**
 * @param {Array<Promise<void>>} blockers
 * @param {Array<() => any>} sync
 * @param {Array<() => Promise<any>>} async
 * @param {(values: Value[]) => any} fn
 */
function flatten(blockers, sync, async, fn) {
	const d = is_runes() ? derived : derived_safe_equal;

	if (async.length === 0 && blockers.length === 0) {
		fn(sync.map(d));
		return;
	}

	var batch = current_batch;
	var parent = /** @type {Effect} */ (active_effect);

	var restore = capture();

	function run() {
		Promise.all(async.map((expression) => async_derived(expression)))
			.then((result) => {
				restore();

				try {
					fn([...sync.map(d), ...result]);
				} catch (error) {
					// ignore errors in blocks that have already been destroyed
					if ((parent.f & DESTROYED) === 0) {
						invoke_error_boundary(error, parent);
					}
				}

				batch?.deactivate();
				unset_context();
			})
			.catch((error) => {
				invoke_error_boundary(error, parent);
			});
	}

	if (blockers.length > 0) {
		Promise.all(blockers).then(() => {
			restore();

			try {
				return run();
			} finally {
				batch?.deactivate();
				unset_context();
			}
		});
	} else {
		run();
	}
}

/**
 * @param {Array<Promise<void>>} blockers
 * @param {(values: Value[]) => any} fn
 */
function run_after_blockers(blockers, fn) {
	flatten(blockers, [], [], fn);
}

/**
 * Captures the current effect context so that we can restore it after
 * some asynchronous work has happened (so that e.g. `await a + b`
 * causes `b` to be registered as a dependency).
 */
function capture() {
	var previous_effect = active_effect;
	var previous_reaction = active_reaction;
	var previous_component_context = component_context;
	var previous_batch = current_batch;

	if (DEV) {
		var previous_dev_stack = dev_stack;
	}

	return function restore(activate_batch = true) {
		set_active_effect(previous_effect);
		set_active_reaction(previous_reaction);
		set_component_context(previous_component_context);
		if (activate_batch) previous_batch?.activate();

		if (DEV) {
			set_dev_stack(previous_dev_stack);
		}
	};
}

function unset_context() {
	set_active_effect(null);
	set_active_reaction(null);
	set_component_context(null);

	if (DEV) {
		set_dev_stack(null);
	}
}

/**
 * @param {Array<() => void | Promise<void>>} thunks
 */
function run(thunks) {
	const restore = capture();

	var boundary = get_boundary();
	var batch = /** @type {Batch} */ (current_batch);
	var blocking = boundary.is_rendered();

	boundary.update_pending_count(1);
	batch.increment(blocking);

	var active = /** @type {Effect} */ (active_effect);

	/** @type {null | { error: any }} */
	var errored = null;

	/** @param {any} error */
	const handle_error = (error) => {
		errored = { error }; // wrap in object in case a promise rejects with a falsy value

		if (!aborted(active)) {
			invoke_error_boundary(error, active);
		}
	};

	var promise = Promise.resolve(thunks[0]()).catch(handle_error);

	var promises = [promise];

	for (const fn of thunks.slice(1)) {
		promise = promise
			.then(() => {
				if (errored) {
					throw errored.error;
				}

				if (aborted(active)) {
					throw STALE_REACTION;
				}

				restore();
				return fn();
			})
			.catch(handle_error)
			.finally(() => {
				unset_context();
				current_batch?.deactivate();
			});

		promises.push(promise);
	}

	promise
		// wait one more tick, so that template effects are
		// guaranteed to run before `$effect(...)`
		.then(() => Promise.resolve())
		.finally(() => {
			boundary.update_pending_count(-1);
			batch.decrement(blocking);
		});

	return promises;
}

/** @import { Derived, Effect, Source } from '#client' */
/** @import { Batch } from './batch.js'; */

const recent_async_deriveds = new Set();

/**
 * @template V
 * @param {() => V} fn
 * @returns {Derived<V>}
 */
/*#__NO_SIDE_EFFECTS__*/
function derived(fn) {
	var flags = DERIVED | DIRTY;
	var parent_derived =
		active_reaction !== null && (active_reaction.f & DERIVED) !== 0
			? /** @type {Derived} */ (active_reaction)
			: null;

	if (active_effect !== null) {
		// Since deriveds are evaluated lazily, any effects created inside them are
		// created too late to ensure that the parent effect is added to the tree
		active_effect.f |= EFFECT_PRESERVED;
	}

	/** @type {Derived<V>} */
	const signal = {
		ctx: component_context,
		deps: null,
		effects: null,
		equals,
		f: flags,
		fn,
		reactions: null,
		rv: 0,
		v: /** @type {V} */ (UNINITIALIZED),
		wv: 0,
		parent: parent_derived ?? active_effect,
		ac: null
	};

	return signal;
}

/**
 * @template V
 * @param {() => V | Promise<V>} fn
 * @param {string} [label]
 * @param {string} [location] If provided, print a warning if the value is not read immediately after update
 * @returns {Promise<Source<V>>}
 */
/*#__NO_SIDE_EFFECTS__*/
function async_derived(fn, label, location) {
	let parent = /** @type {Effect | null} */ (active_effect);

	if (parent === null) {
		async_derived_orphan();
	}

	var boundary = /** @type {Boundary} */ (parent.b);

	var promise = /** @type {Promise<V>} */ (/** @type {unknown} */ (undefined));
	var signal = source(/** @type {V} */ (UNINITIALIZED));

	if (DEV) signal.label = label;

	// only suspend in async deriveds created on initialisation
	var should_suspend = !active_reaction;

	/** @type {Map<Batch, ReturnType<typeof deferred<V>>>} */
	var deferreds = new Map();

	async_effect(() => {

		/** @type {ReturnType<typeof deferred<V>>} */
		var d = deferred();
		promise = d.promise;

		try {
			// If this code is changed at some point, make sure to still access the then property
			// of fn() to read any signals it might access, so that we track them as dependencies.
			// We call `unset_context` to undo any `save` calls that happen inside `fn()`
			Promise.resolve(fn())
				.then(d.resolve, d.reject)
				.then(() => {
					if (batch === current_batch && batch.committed) {
						// if the batch was rejected as stale, we need to cleanup
						// after any `$.save(...)` calls inside `fn()`
						batch.deactivate();
					}

					unset_context();
				});
		} catch (error) {
			d.reject(error);
			unset_context();
		}

		var batch = /** @type {Batch} */ (current_batch);

		if (should_suspend) {
			var blocking = boundary.is_rendered();

			boundary.update_pending_count(1);
			batch.increment(blocking);

			deferreds.get(batch)?.reject(STALE_REACTION);
			deferreds.delete(batch); // delete to ensure correct order in Map iteration below
			deferreds.set(batch, d);
		}

		/**
		 * @param {any} value
		 * @param {unknown} error
		 */
		const handler = (value, error = undefined) => {

			batch.activate();

			if (error) {
				if (error !== STALE_REACTION) {
					signal.f |= ERROR_VALUE;

					// @ts-expect-error the error is the wrong type, but we don't care
					internal_set(signal, error);
				}
			} else {
				if ((signal.f & ERROR_VALUE) !== 0) {
					signal.f ^= ERROR_VALUE;
				}

				internal_set(signal, value);

				// All prior async derived runs are now stale
				for (const [b, d] of deferreds) {
					deferreds.delete(b);
					if (b === batch) break;
					d.reject(STALE_REACTION);
				}
			}

			if (should_suspend) {
				boundary.update_pending_count(-1);
				batch.decrement(blocking);
			}
		};

		d.promise.then(handler, (e) => handler(null, e || 'unknown'));
	});

	teardown(() => {
		for (const d of deferreds.values()) {
			d.reject(STALE_REACTION);
		}
	});

	if (DEV) {
		// add a flag that lets this be printed as a derived
		// when using `$inspect.trace()`
		signal.f |= ASYNC;
	}

	return new Promise((fulfil) => {
		/** @param {Promise<V>} p */
		function next(p) {
			function go() {
				if (p === promise) {
					fulfil(signal);
				} else {
					// if the effect re-runs before the initial promise
					// resolves, delay resolution until we have a value
					next(promise);
				}
			}

			p.then(go, go);
		}

		next(promise);
	});
}

/**
 * @template V
 * @param {() => V} fn
 * @returns {Derived<V>}
 */
/*#__NO_SIDE_EFFECTS__*/
function user_derived(fn) {
	const d = derived(fn);

	if (!async_mode_flag) push_reaction_value(d);

	return d;
}

/**
 * @template V
 * @param {() => V} fn
 * @returns {Derived<V>}
 */
/*#__NO_SIDE_EFFECTS__*/
function derived_safe_equal(fn) {
	const signal = derived(fn);
	signal.equals = safe_equals;
	return signal;
}

/**
 * @param {Derived} derived
 * @returns {void}
 */
function destroy_derived_effects(derived) {
	var effects = derived.effects;

	if (effects !== null) {
		derived.effects = null;

		for (var i = 0; i < effects.length; i += 1) {
			destroy_effect(/** @type {Effect} */ (effects[i]));
		}
	}
}

/**
 * The currently updating deriveds, used to detect infinite recursion
 * in dev mode and provide a nicer error than 'too much recursion'
 * @type {Derived[]}
 */
let stack = [];

/**
 * @param {Derived} derived
 * @returns {Effect | null}
 */
function get_derived_parent_effect(derived) {
	var parent = derived.parent;
	while (parent !== null) {
		if ((parent.f & DERIVED) === 0) {
			// The original parent effect might've been destroyed but the derived
			// is used elsewhere now - do not return the destroyed effect in that case
			return (parent.f & DESTROYED) === 0 ? /** @type {Effect} */ (parent) : null;
		}
		parent = parent.parent;
	}
	return null;
}

/**
 * @template T
 * @param {Derived} derived
 * @returns {T}
 */
function execute_derived(derived) {
	var value;
	var prev_active_effect = active_effect;

	set_active_effect(get_derived_parent_effect(derived));

	if (DEV) {
		let prev_eager_effects = eager_effects;
		set_eager_effects(new Set());
		try {
			if (stack.includes(derived)) {
				derived_references_self();
			}

			stack.push(derived);

			derived.f &= ~WAS_MARKED;
			destroy_derived_effects(derived);
			value = update_reaction(derived);
		} finally {
			set_active_effect(prev_active_effect);
			set_eager_effects(prev_eager_effects);
			stack.pop();
		}
	} else {
		try {
			derived.f &= ~WAS_MARKED;
			destroy_derived_effects(derived);
			value = update_reaction(derived);
		} finally {
			set_active_effect(prev_active_effect);
		}
	}

	return value;
}

/**
 * @param {Derived} derived
 * @returns {void}
 */
function update_derived(derived) {
	var value = execute_derived(derived);

	if (!derived.equals(value)) {
		derived.wv = increment_write_version();

		// in a fork, we don't update the underlying value, just `batch_values`.
		// the underlying value will be updated when the fork is committed.
		// otherwise, the next time we get here after a 'real world' state
		// change, `derived.equals` may incorrectly return `true`
		if (!current_batch?.is_fork || derived.deps === null) {
			derived.v = value;

			// deriveds without dependencies should never be recomputed
			if (derived.deps === null) {
				set_signal_status(derived, CLEAN);
				return;
			}
		}
	}

	// don't mark derived clean if we're reading it inside a
	// cleanup function, or it will cache a stale value
	if (is_destroying_effect) {
		return;
	}

	// During time traveling we don't want to reset the status so that
	// traversal of the graph in the other batches still happens
	if (batch_values !== null) {
		// only cache the value if we're in a tracking context, otherwise we won't
		// clear the cache in `mark_reactions` when dependencies are updated
		if (effect_tracking() || current_batch?.is_fork) {
			batch_values.set(derived, value);
		}
	} else {
		update_derived_status(derived);
	}
}

/** @import { Derived, Effect, Source, Value } from '#client' */

/** @type {Set<any>} */
let eager_effects = new Set();

/** @type {Map<Source, any>} */
const old_values = new Map();

/**
 * @param {Set<any>} v
 */
function set_eager_effects(v) {
	eager_effects = v;
}

let eager_effects_deferred = false;

function set_eager_effects_deferred() {
	eager_effects_deferred = true;
}

/**
 * @template V
 * @param {V} v
 * @param {Error | null} [stack]
 * @returns {Source<V>}
 */
// TODO rename this to `state` throughout the codebase
function source(v, stack) {
	/** @type {Value} */
	var signal = {
		f: 0, // TODO ideally we could skip this altogether, but it causes type errors
		v,
		reactions: null,
		equals,
		rv: 0,
		wv: 0
	};

	return signal;
}

/**
 * @template V
 * @param {V} v
 * @param {Error | null} [stack]
 */
/*#__NO_SIDE_EFFECTS__*/
function state(v, stack) {
	const s = source(v);

	push_reaction_value(s);

	return s;
}

/**
 * @template V
 * @param {V} initial_value
 * @param {boolean} [immutable]
 * @returns {Source<V>}
 */
/*#__NO_SIDE_EFFECTS__*/
function mutable_source(initial_value, immutable = false, trackable = true) {
	const s = source(initial_value);
	if (!immutable) {
		s.equals = safe_equals;
	}

	// bind the signal to the component context, in case we need to
	// track updates to trigger beforeUpdate/afterUpdate callbacks
	if (legacy_mode_flag && trackable && component_context !== null && component_context.l !== null) {
		(component_context.l.s ??= []).push(s);
	}

	return s;
}

/**
 * @template V
 * @param {Value<V>} source
 * @param {V} value
 */
function mutate(source, value) {
	set(
		source,
		untrack(() => get(source))
	);
	return value;
}

/**
 * @template V
 * @param {Source<V>} source
 * @param {V} value
 * @param {boolean} [should_proxy]
 * @returns {V}
 */
function set(source, value, should_proxy = false) {
	if (
		active_reaction !== null &&
		// since we are untracking the function inside `$inspect.with` we need to add this check
		// to ensure we error if state is set inside an inspect effect
		(!untracking || (active_reaction.f & EAGER_EFFECT) !== 0) &&
		is_runes() &&
		(active_reaction.f & (DERIVED | BLOCK_EFFECT | ASYNC | EAGER_EFFECT)) !== 0 &&
		!current_sources?.includes(source)
	) {
		state_unsafe_mutation();
	}

	let new_value = should_proxy ? proxy(value) : value;

	if (DEV) {
		tag_proxy(new_value, /** @type {string} */ (source.label));
	}

	return internal_set(source, new_value);
}

/**
 * @template V
 * @param {Source<V>} source
 * @param {V} value
 * @returns {V}
 */
function internal_set(source, value) {
	if (!source.equals(value)) {
		var old_value = source.v;

		if (is_destroying_effect) {
			old_values.set(source, value);
		} else {
			old_values.set(source, old_value);
		}

		source.v = value;

		var batch = Batch.ensure();
		batch.capture(source, old_value);

		if (DEV) {
			if (active_effect !== null) {
				source.updated ??= new Map();

				// For performance reasons, when not using $inspect.trace, we only start collecting stack traces
				// after the same source has been updated more than 5 times in the same flush cycle.
				const count = (source.updated.get('')?.count ?? 0) + 1;
				source.updated.set('', { error: /** @type {any} */ (null), count });

				if (count > 5) {
					const error = get_error('updated at');

					if (error !== null) {
						let entry = source.updated.get(error.stack);

						if (!entry) {
							entry = { error, count: 0 };
							source.updated.set(error.stack, entry);
						}

						entry.count++;
					}
				}
			}

			if (active_effect !== null) {
				source.set_during_effect = true;
			}
		}

		if ((source.f & DERIVED) !== 0) {
			const derived = /** @type {Derived} */ (source);

			// if we are assigning to a dirty derived we set it to clean/maybe dirty but we also eagerly execute it to track the dependencies
			if ((source.f & DIRTY) !== 0) {
				execute_derived(derived);
			}

			update_derived_status(derived);
		}

		source.wv = increment_write_version();

		// For debugging, in case you want to know which reactions are being scheduled:
		// log_reactions(source);
		mark_reactions(source, DIRTY);

		// It's possible that the current reaction might not have up-to-date dependencies
		// whilst it's actively running. So in the case of ensuring it registers the reaction
		// properly for itself, we need to ensure the current effect actually gets
		// scheduled. i.e: `$effect(() => x++)`
		if (
			is_runes() &&
			active_effect !== null &&
			(active_effect.f & CLEAN) !== 0 &&
			(active_effect.f & (BRANCH_EFFECT | ROOT_EFFECT)) === 0
		) {
			if (untracked_writes === null) {
				set_untracked_writes([source]);
			} else {
				untracked_writes.push(source);
			}
		}

		if (!batch.is_fork && eager_effects.size > 0 && !eager_effects_deferred) {
			flush_eager_effects();
		}
	}

	return value;
}

function flush_eager_effects() {
	eager_effects_deferred = false;
	var prev_is_updating_effect = is_updating_effect;
	set_is_updating_effect(true);

	const inspects = Array.from(eager_effects);

	try {
		for (const effect of inspects) {
			// Mark clean inspect-effects as maybe dirty and then check their dirtiness
			// instead of just updating the effects - this way we avoid overfiring.
			if ((effect.f & CLEAN) !== 0) {
				set_signal_status(effect, MAYBE_DIRTY);
			}

			if (is_dirty(effect)) {
				update_effect(effect);
			}
		}
	} finally {
		set_is_updating_effect(prev_is_updating_effect);
	}

	eager_effects.clear();
}

/**
 * @template {number | bigint} T
 * @param {Source<T>} source
 * @param {1 | -1} [d]
 * @returns {T}
 */
function update(source, d = 1) {
	var value = get(source);
	var result = d === 1 ? value++ : value--;

	set(source, value);

	// @ts-expect-error
	return result;
}

/**
 * Silently (without using `get`) increment a source
 * @param {Source<number>} source
 */
function increment(source) {
	set(source, source.v + 1);
}

/**
 * @param {Value} signal
 * @param {number} status should be DIRTY or MAYBE_DIRTY
 * @returns {void}
 */
function mark_reactions(signal, status) {
	var reactions = signal.reactions;
	if (reactions === null) return;

	var runes = is_runes();
	var length = reactions.length;

	for (var i = 0; i < length; i++) {
		var reaction = reactions[i];
		var flags = reaction.f;

		// In legacy mode, skip the current effect to prevent infinite loops
		if (!runes && reaction === active_effect) continue;

		// Inspect effects need to run immediately, so that the stack trace makes sense
		if (DEV && (flags & EAGER_EFFECT) !== 0) {
			eager_effects.add(reaction);
			continue;
		}

		var not_dirty = (flags & DIRTY) === 0;

		// don't set a DIRTY reaction to MAYBE_DIRTY
		if (not_dirty) {
			set_signal_status(reaction, status);
		}

		if ((flags & DERIVED) !== 0) {
			var derived = /** @type {Derived} */ (reaction);

			batch_values?.delete(derived);

			if ((flags & WAS_MARKED) === 0) {
				// Only connected deriveds can be reliably unmarked right away
				if (flags & CONNECTED) {
					reaction.f |= WAS_MARKED;
				}

				mark_reactions(derived, MAYBE_DIRTY);
			}
		} else if (not_dirty) {
			if ((flags & BLOCK_EFFECT) !== 0 && eager_block_effects !== null) {
				eager_block_effects.add(/** @type {Effect} */ (reaction));
			}

			schedule_effect(/** @type {Effect} */ (reaction));
		}
	}
}

/** @import { Source } from '#client' */

// TODO move all regexes into shared module?
const regex_is_valid_identifier = /^[a-zA-Z_$][a-zA-Z_$0-9]*$/;

/**
 * @template T
 * @param {T} value
 * @returns {T}
 */
function proxy(value) {
	// if non-proxyable, or is already a proxy, return `value`
	if (typeof value !== 'object' || value === null || STATE_SYMBOL in value) {
		return value;
	}

	const prototype = get_prototype_of(value);

	if (prototype !== object_prototype && prototype !== array_prototype) {
		return value;
	}

	/** @type {Map<any, Source<any>>} */
	var sources = new Map();
	var is_proxied_array = is_array(value);
	var version = state(0);
	var parent_version = update_version;

	/**
	 * Executes the proxy in the context of the reaction it was originally created in, if any
	 * @template T
	 * @param {() => T} fn
	 */
	var with_parent = (fn) => {
		if (update_version === parent_version) {
			return fn();
		}

		// child source is being created after the initial proxy —
		// prevent it from being associated with the current reaction
		var reaction = active_reaction;
		var version = update_version;

		set_active_reaction(null);
		set_update_version(parent_version);

		var result = fn();

		set_active_reaction(reaction);
		set_update_version(version);

		return result;
	};

	if (is_proxied_array) {
		// We need to create the length source eagerly to ensure that
		// mutations to the array are properly synced with our proxy
		sources.set('length', state(/** @type {any[]} */ (value).length));
		if (DEV) {
			value = /** @type {any} */ (inspectable_array(/** @type {any[]} */ (value)));
		}
	}

	/** Used in dev for $inspect.trace() */
	var path = '';
	let updating = false;
	/** @param {string} new_path */
	function update_path(new_path) {
		if (updating) return;
		updating = true;
		path = new_path;

		tag(version, `${path} version`);

		// rename all child sources and child proxies
		for (const [prop, source] of sources) {
			tag(source, get_label(path, prop));
		}
		updating = false;
	}

	return new Proxy(/** @type {any} */ (value), {
		defineProperty(_, prop, descriptor) {
			if (
				!('value' in descriptor) ||
				descriptor.configurable === false ||
				descriptor.enumerable === false ||
				descriptor.writable === false
			) {
				// we disallow non-basic descriptors, because unless they are applied to the
				// target object — which we avoid, so that state can be forked — we will run
				// afoul of the various invariants
				// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy/Proxy/getOwnPropertyDescriptor#invariants
				state_descriptors_fixed();
			}
			var s = sources.get(prop);
			if (s === undefined) {
				s = with_parent(() => {
					var s = state(descriptor.value);
					sources.set(prop, s);
					if (DEV && typeof prop === 'string') {
						tag(s, get_label(path, prop));
					}
					return s;
				});
			} else {
				set(s, descriptor.value, true);
			}

			return true;
		},

		deleteProperty(target, prop) {
			var s = sources.get(prop);

			if (s === undefined) {
				if (prop in target) {
					const s = with_parent(() => state(UNINITIALIZED));
					sources.set(prop, s);
					increment(version);

					if (DEV) {
						tag(s, get_label(path, prop));
					}
				}
			} else {
				set(s, UNINITIALIZED);
				increment(version);
			}

			return true;
		},

		get(target, prop, receiver) {
			if (prop === STATE_SYMBOL) {
				return value;
			}

			if (DEV && prop === PROXY_PATH_SYMBOL) {
				return update_path;
			}

			var s = sources.get(prop);
			var exists = prop in target;

			// create a source, but only if it's an own property and not a prototype property
			if (s === undefined && (!exists || get_descriptor(target, prop)?.writable)) {
				s = with_parent(() => {
					var p = proxy(exists ? target[prop] : UNINITIALIZED);
					var s = state(p);

					if (DEV) {
						tag(s, get_label(path, prop));
					}

					return s;
				});

				sources.set(prop, s);
			}

			if (s !== undefined) {
				var v = get(s);
				return v === UNINITIALIZED ? undefined : v;
			}

			return Reflect.get(target, prop, receiver);
		},

		getOwnPropertyDescriptor(target, prop) {
			var descriptor = Reflect.getOwnPropertyDescriptor(target, prop);

			if (descriptor && 'value' in descriptor) {
				var s = sources.get(prop);
				if (s) descriptor.value = get(s);
			} else if (descriptor === undefined) {
				var source = sources.get(prop);
				var value = source?.v;

				if (source !== undefined && value !== UNINITIALIZED) {
					return {
						enumerable: true,
						configurable: true,
						value,
						writable: true
					};
				}
			}

			return descriptor;
		},

		has(target, prop) {
			if (prop === STATE_SYMBOL) {
				return true;
			}

			var s = sources.get(prop);
			var has = (s !== undefined && s.v !== UNINITIALIZED) || Reflect.has(target, prop);

			if (
				s !== undefined ||
				(active_effect !== null && (!has || get_descriptor(target, prop)?.writable))
			) {
				if (s === undefined) {
					s = with_parent(() => {
						var p = has ? proxy(target[prop]) : UNINITIALIZED;
						var s = state(p);

						if (DEV) {
							tag(s, get_label(path, prop));
						}

						return s;
					});

					sources.set(prop, s);
				}

				var value = get(s);
				if (value === UNINITIALIZED) {
					return false;
				}
			}

			return has;
		},

		set(target, prop, value, receiver) {
			var s = sources.get(prop);
			var has = prop in target;

			// variable.length = value -> clear all signals with index >= value
			if (is_proxied_array && prop === 'length') {
				for (var i = value; i < /** @type {Source<number>} */ (s).v; i += 1) {
					var other_s = sources.get(i + '');
					if (other_s !== undefined) {
						set(other_s, UNINITIALIZED);
					} else if (i in target) {
						// If the item exists in the original, we need to create an uninitialized source,
						// else a later read of the property would result in a source being created with
						// the value of the original item at that index.
						other_s = with_parent(() => state(UNINITIALIZED));
						sources.set(i + '', other_s);

						if (DEV) {
							tag(other_s, get_label(path, i));
						}
					}
				}
			}

			// If we haven't yet created a source for this property, we need to ensure
			// we do so otherwise if we read it later, then the write won't be tracked and
			// the heuristics of effects will be different vs if we had read the proxied
			// object property before writing to that property.
			if (s === undefined) {
				if (!has || get_descriptor(target, prop)?.writable) {
					s = with_parent(() => state(undefined));

					if (DEV) {
						tag(s, get_label(path, prop));
					}
					set(s, proxy(value));

					sources.set(prop, s);
				}
			} else {
				has = s.v !== UNINITIALIZED;

				var p = with_parent(() => proxy(value));
				set(s, p);
			}

			var descriptor = Reflect.getOwnPropertyDescriptor(target, prop);

			// Set the new value before updating any signals so that any listeners get the new value
			if (descriptor?.set) {
				descriptor.set.call(receiver, value);
			}

			if (!has) {
				// If we have mutated an array directly, we might need to
				// signal that length has also changed. Do it before updating metadata
				// to ensure that iterating over the array as a result of a metadata update
				// will not cause the length to be out of sync.
				if (is_proxied_array && typeof prop === 'string') {
					var ls = /** @type {Source<number>} */ (sources.get('length'));
					var n = Number(prop);

					if (Number.isInteger(n) && n >= ls.v) {
						set(ls, n + 1);
					}
				}

				increment(version);
			}

			return true;
		},

		ownKeys(target) {
			get(version);

			var own_keys = Reflect.ownKeys(target).filter((key) => {
				var source = sources.get(key);
				return source === undefined || source.v !== UNINITIALIZED;
			});

			for (var [key, source] of sources) {
				if (source.v !== UNINITIALIZED && !(key in target)) {
					own_keys.push(key);
				}
			}

			return own_keys;
		},

		setPrototypeOf() {
			state_prototype_fixed();
		}
	});
}

/**
 * @param {string} path
 * @param {string | symbol} prop
 */
function get_label(path, prop) {
	if (typeof prop === 'symbol') return `${path}[Symbol(${prop.description ?? ''})]`;
	if (regex_is_valid_identifier.test(prop)) return `${path}.${prop}`;
	return /^\d+$/.test(prop) ? `${path}[${prop}]` : `${path}['${prop}']`;
}

/**
 * @param {any} value
 */
function get_proxied_value(value) {
	try {
		if (value !== null && typeof value === 'object' && STATE_SYMBOL in value) {
			return value[STATE_SYMBOL];
		}
	} catch {
		// the above if check can throw an error if the value in question
		// is the contentWindow of an iframe on another domain, in which
		// case we want to just return the value (because it's definitely
		// not a proxied value) so we don't break any JavaScript interacting
		// with that iframe (such as various payment companies client side
		// JavaScript libraries interacting with their iframes on the same
		// domain)
	}

	return value;
}

/**
 * @param {any} a
 * @param {any} b
 */
function is(a, b) {
	return Object.is(get_proxied_value(a), get_proxied_value(b));
}

const ARRAY_MUTATING_METHODS = new Set([
	'copyWithin',
	'fill',
	'pop',
	'push',
	'reverse',
	'shift',
	'sort',
	'splice',
	'unshift'
]);

/**
 * Wrap array mutating methods so $inspect is triggered only once and
 * to prevent logging an array in intermediate state (e.g. with an empty slot)
 * @param {any[]} array
 */
function inspectable_array(array) {
	return new Proxy(array, {
		get(target, prop, receiver) {
			var value = Reflect.get(target, prop, receiver);
			if (!ARRAY_MUTATING_METHODS.has(/** @type {string} */ (prop))) {
				return value;
			}

			/**
			 * @this {any[]}
			 * @param {any[]} args
			 */
			return function (...args) {
				set_eager_effects_deferred();
				var result = value.apply(this, args);
				flush_eager_effects();
				return result;
			};
		}
	});
}

function init_array_prototype_warnings() {
	const array_prototype = Array.prototype;
	// The REPL ends up here over and over, and this prevents it from adding more and more patches
	// of the same kind to the prototype, which would slow down everything over time.
	// @ts-expect-error
	const cleanup = Array.__svelte_cleanup;
	if (cleanup) {
		cleanup();
	}

	const { indexOf, lastIndexOf, includes } = array_prototype;

	array_prototype.indexOf = function (item, from_index) {
		const index = indexOf.call(this, item, from_index);

		if (index === -1) {
			for (let i = from_index ?? 0; i < this.length; i += 1) {
				if (get_proxied_value(this[i]) === item) {
					state_proxy_equality_mismatch('array.indexOf(...)');
					break;
				}
			}
		}

		return index;
	};

	array_prototype.lastIndexOf = function (item, from_index) {
		// we need to specify this.length - 1 because it's probably using something like
		// `arguments` inside so passing undefined is different from not passing anything
		const index = lastIndexOf.call(this, item, from_index ?? this.length - 1);

		if (index === -1) {
			for (let i = 0; i <= (from_index ?? this.length - 1); i += 1) {
				if (get_proxied_value(this[i]) === item) {
					state_proxy_equality_mismatch('array.lastIndexOf(...)');
					break;
				}
			}
		}

		return index;
	};

	array_prototype.includes = function (item, from_index) {
		const has = includes.call(this, item, from_index);

		if (!has) {
			for (let i = 0; i < this.length; i += 1) {
				if (get_proxied_value(this[i]) === item) {
					state_proxy_equality_mismatch('array.includes(...)');
					break;
				}
			}
		}

		return has;
	};

	// @ts-expect-error
	Array.__svelte_cleanup = () => {
		array_prototype.indexOf = indexOf;
		array_prototype.lastIndexOf = lastIndexOf;
		array_prototype.includes = includes;
	};
}

/** @import { Effect, TemplateNode } from '#client' */

// export these for reference in the compiled code, making global name deduplication unnecessary
/** @type {Window} */
var $window;

/** @type {Document} */
var $document;

/** @type {boolean} */
var is_firefox;

/** @type {() => Node | null} */
var first_child_getter;
/** @type {() => Node | null} */
var next_sibling_getter;

/**
 * Initialize these lazily to avoid issues when using the runtime in a server context
 * where these globals are not available while avoiding a separate server entry point
 */
function init_operations() {
	if ($window !== undefined) {
		return;
	}

	$window = window;
	$document = document;
	is_firefox = /Firefox/.test(navigator.userAgent);

	var element_prototype = Element.prototype;
	var node_prototype = Node.prototype;
	var text_prototype = Text.prototype;

	// @ts-ignore
	first_child_getter = get_descriptor(node_prototype, 'firstChild').get;
	// @ts-ignore
	next_sibling_getter = get_descriptor(node_prototype, 'nextSibling').get;

	if (is_extensible(element_prototype)) {
		// the following assignments improve perf of lookups on DOM nodes
		// @ts-expect-error
		element_prototype.__click = undefined;
		// @ts-expect-error
		element_prototype.__className = undefined;
		// @ts-expect-error
		element_prototype.__attributes = null;
		// @ts-expect-error
		element_prototype.__style = undefined;
		// @ts-expect-error
		element_prototype.__e = undefined;
	}

	if (is_extensible(text_prototype)) {
		// @ts-expect-error
		text_prototype.__t = undefined;
	}

	if (DEV) {
		// @ts-expect-error
		element_prototype.__svelte_meta = null;

		init_array_prototype_warnings();
	}
}

/**
 * @param {string} value
 * @returns {Text}
 */
function create_text(value = '') {
	return document.createTextNode(value);
}

/**
 * @template {Node} N
 * @param {N} node
 */
/*@__NO_SIDE_EFFECTS__*/
function get_first_child(node) {
	return /** @type {TemplateNode | null} */ (first_child_getter.call(node));
}

/**
 * @template {Node} N
 * @param {N} node
 */
/*@__NO_SIDE_EFFECTS__*/
function get_next_sibling(node) {
	return /** @type {TemplateNode | null} */ (next_sibling_getter.call(node));
}

/**
 * Don't mark this as side-effect-free, hydration needs to walk all nodes
 * @template {Node} N
 * @param {N} node
 * @param {boolean} is_text
 * @returns {TemplateNode | null}
 */
function child(node, is_text) {
	if (!hydrating) {
		return get_first_child(node);
	}

	var child = get_first_child(hydrate_node);

	// Child can be null if we have an element with a single child, like `<p>{text}</p>`, where `text` is empty
	if (child === null) {
		child = hydrate_node.appendChild(create_text());
	} else if (is_text && child.nodeType !== TEXT_NODE) {
		var text = create_text();
		child?.before(text);
		set_hydrate_node(text);
		return text;
	}

	set_hydrate_node(child);
	return child;
}

/**
 * Don't mark this as side-effect-free, hydration needs to walk all nodes
 * @param {TemplateNode} node
 * @param {boolean} [is_text]
 * @returns {TemplateNode | null}
 */
function first_child(node, is_text = false) {
	if (!hydrating) {
		var first = get_first_child(node);

		// TODO prevent user comments with the empty string when preserveComments is true
		if (first instanceof Comment && first.data === '') return get_next_sibling(first);

		return first;
	}

	// if an {expression} is empty during SSR, there might be no
	// text node to hydrate — we must therefore create one
	if (is_text && hydrate_node?.nodeType !== TEXT_NODE) {
		var text = create_text();

		hydrate_node?.before(text);
		set_hydrate_node(text);
		return text;
	}

	return hydrate_node;
}

/**
 * Don't mark this as side-effect-free, hydration needs to walk all nodes
 * @param {TemplateNode} node
 * @param {number} count
 * @param {boolean} is_text
 * @returns {TemplateNode | null}
 */
function sibling(node, count = 1, is_text = false) {
	let next_sibling = hydrating ? hydrate_node : node;
	var last_sibling;

	while (count--) {
		last_sibling = next_sibling;
		next_sibling = /** @type {TemplateNode} */ (get_next_sibling(next_sibling));
	}

	if (!hydrating) {
		return next_sibling;
	}

	// if a sibling {expression} is empty during SSR, there might be no
	// text node to hydrate — we must therefore create one
	if (is_text && next_sibling?.nodeType !== TEXT_NODE) {
		var text = create_text();
		// If the next sibling is `null` and we're handling text then it's because
		// the SSR content was empty for the text, so we need to generate a new text
		// node and insert it after the last sibling
		if (next_sibling === null) {
			last_sibling?.after(text);
		} else {
			next_sibling.before(text);
		}
		set_hydrate_node(text);
		return text;
	}

	set_hydrate_node(next_sibling);
	return next_sibling;
}

/**
 * @template {Node} N
 * @param {N} node
 * @returns {void}
 */
function clear_text_content(node) {
	node.textContent = '';
}

/**
 * Returns `true` if we're updating the current block, for example `condition` in
 * an `{#if condition}` block just changed. In this case, the branch should be
 * appended (or removed) at the same time as other updates within the
 * current `<svelte:boundary>`
 */
function should_defer_append() {
	if (!async_mode_flag) return false;
	if (eager_block_effects !== null) return false;

	var flags = /** @type {Effect} */ (active_effect).f;
	return (flags & EFFECT_RAN) !== 0;
}

/**
 * @param {HTMLElement} dom
 * @param {boolean} value
 * @returns {void}
 */
function autofocus(dom, value) {
	if (value) {
		const body = document.body;
		dom.autofocus = true;

		queue_micro_task(() => {
			if (document.activeElement === body) {
				dom.focus();
			}
		});
	}
}

/**
 * The child of a textarea actually corresponds to the defaultValue property, so we need
 * to remove it upon hydration to avoid a bug when someone resets the form value.
 * @param {HTMLTextAreaElement} dom
 * @returns {void}
 */
function remove_textarea_child(dom) {
	if (hydrating && get_first_child(dom) !== null) {
		clear_text_content(dom);
	}
}

let listening_to_form_reset = false;

function add_form_reset_listener() {
	if (!listening_to_form_reset) {
		listening_to_form_reset = true;
		document.addEventListener(
			'reset',
			(evt) => {
				// Needs to happen one tick later or else the dom properties of the form
				// elements have not updated to their reset values yet
				Promise.resolve().then(() => {
					if (!evt.defaultPrevented) {
						for (const e of /**@type {HTMLFormElement} */ (evt.target).elements) {
							// @ts-expect-error
							e.__on_r?.();
						}
					}
				});
			},
			// In the capture phase to guarantee we get noticed of it (no possibility of stopPropagation)
			{ capture: true }
		);
	}
}

/**
 * Fires the handler once immediately (unless corresponding arg is set to `false`),
 * then listens to the given events until the render effect context is destroyed
 * @param {EventTarget} target
 * @param {Array<string>} events
 * @param {(event?: Event) => void} handler
 * @param {any} call_handler_immediately
 */
function listen(target, events, handler, call_handler_immediately = true) {
	if (call_handler_immediately) {
		handler();
	}

	for (var name of events) {
		target.addEventListener(name, handler);
	}

	teardown(() => {
		for (var name of events) {
			target.removeEventListener(name, handler);
		}
	});
}

/**
 * @template T
 * @param {() => T} fn
 */
function without_reactive_context(fn) {
	var previous_reaction = active_reaction;
	var previous_effect = active_effect;
	set_active_reaction(null);
	set_active_effect(null);
	try {
		return fn();
	} finally {
		set_active_reaction(previous_reaction);
		set_active_effect(previous_effect);
	}
}

/**
 * Listen to the given event, and then instantiate a global form reset listener if not already done,
 * to notify all bindings when the form is reset
 * @param {HTMLElement} element
 * @param {string} event
 * @param {(is_reset?: true) => void} handler
 * @param {(is_reset?: true) => void} [on_reset]
 */
function listen_to_event_and_reset_event(element, event, handler, on_reset = handler) {
	element.addEventListener(event, () => without_reactive_context(handler));
	// @ts-expect-error
	const prev = element.__on_r;
	if (prev) {
		// special case for checkbox that can have multiple binds (group & checked)
		// @ts-expect-error
		element.__on_r = () => {
			prev();
			on_reset(true);
		};
	} else {
		// @ts-expect-error
		element.__on_r = () => on_reset(true);
	}

	add_form_reset_listener();
}

/** @import { ComponentContext, ComponentContextLegacy, Derived, Effect, TemplateNode, TransitionManager } from '#client' */

/**
 * @param {'$effect' | '$effect.pre' | '$inspect'} rune
 */
function validate_effect(rune) {
	if (active_effect === null) {
		if (active_reaction === null) {
			effect_orphan(rune);
		}

		effect_in_unowned_derived();
	}

	if (is_destroying_effect) {
		effect_in_teardown(rune);
	}
}

/**
 * @param {Effect} effect
 * @param {Effect} parent_effect
 */
function push_effect(effect, parent_effect) {
	var parent_last = parent_effect.last;
	if (parent_last === null) {
		parent_effect.last = parent_effect.first = effect;
	} else {
		parent_last.next = effect;
		effect.prev = parent_last;
		parent_effect.last = effect;
	}
}

/**
 * @param {number} type
 * @param {null | (() => void | (() => void))} fn
 * @param {boolean} sync
 * @returns {Effect}
 */
function create_effect(type, fn, sync) {
	var parent = active_effect;

	if (DEV) {
		// Ensure the parent is never an inspect effect
		while (parent !== null && (parent.f & EAGER_EFFECT) !== 0) {
			parent = parent.parent;
		}
	}

	if (parent !== null && (parent.f & INERT) !== 0) {
		type |= INERT;
	}

	/** @type {Effect} */
	var effect = {
		ctx: component_context,
		deps: null,
		nodes: null,
		f: type | DIRTY | CONNECTED,
		first: null,
		fn,
		last: null,
		next: null,
		parent,
		b: parent && parent.b,
		prev: null,
		teardown: null,
		wv: 0,
		ac: null
	};

	if (DEV) {
		effect.component_function = dev_current_component_function;
	}

	if (sync) {
		try {
			update_effect(effect);
			effect.f |= EFFECT_RAN;
		} catch (e) {
			destroy_effect(effect);
			throw e;
		}
	} else if (fn !== null) {
		schedule_effect(effect);
	}

	/** @type {Effect | null} */
	var e = effect;

	// if an effect has already ran and doesn't need to be kept in the tree
	// (because it won't re-run, has no DOM, and has no teardown etc)
	// then we skip it and go to its child (if any)
	if (
		sync &&
		e.deps === null &&
		e.teardown === null &&
		e.nodes === null &&
		e.first === e.last && // either `null`, or a singular child
		(e.f & EFFECT_PRESERVED) === 0
	) {
		e = e.first;
		if ((type & BLOCK_EFFECT) !== 0 && (type & EFFECT_TRANSPARENT) !== 0 && e !== null) {
			e.f |= EFFECT_TRANSPARENT;
		}
	}

	if (e !== null) {
		e.parent = parent;

		if (parent !== null) {
			push_effect(e, parent);
		}

		// if we're in a derived, add the effect there too
		if (
			active_reaction !== null &&
			(active_reaction.f & DERIVED) !== 0 &&
			(type & ROOT_EFFECT) === 0
		) {
			var derived = /** @type {Derived} */ (active_reaction);
			(derived.effects ??= []).push(e);
		}
	}

	return effect;
}

/**
 * Internal representation of `$effect.tracking()`
 * @returns {boolean}
 */
function effect_tracking() {
	return active_reaction !== null && !untracking;
}

/**
 * @param {() => void} fn
 */
function teardown(fn) {
	const effect = create_effect(RENDER_EFFECT, null, false);
	set_signal_status(effect, CLEAN);
	effect.teardown = fn;
	return effect;
}

/**
 * Internal representation of `$effect(...)`
 * @param {() => void | (() => void)} fn
 */
function user_effect(fn) {
	validate_effect('$effect');

	if (DEV) {
		define_property(fn, 'name', {
			value: '$effect'
		});
	}

	// Non-nested `$effect(...)` in a component should be deferred
	// until the component is mounted
	var flags = /** @type {Effect} */ (active_effect).f;
	var defer = !active_reaction && (flags & BRANCH_EFFECT) !== 0 && (flags & EFFECT_RAN) === 0;

	if (defer) {
		// Top-level `$effect(...)` in an unmounted component — defer until mount
		var context = /** @type {ComponentContext} */ (component_context);
		(context.e ??= []).push(fn);
	} else {
		// Everything else — create immediately
		return create_user_effect(fn);
	}
}

/**
 * @param {() => void | (() => void)} fn
 */
function create_user_effect(fn) {
	return create_effect(EFFECT | USER_EFFECT, fn, false);
}

/**
 * Internal representation of `$effect.pre(...)`
 * @param {() => void | (() => void)} fn
 * @returns {Effect}
 */
function user_pre_effect(fn) {
	validate_effect('$effect.pre');
	if (DEV) {
		define_property(fn, 'name', {
			value: '$effect.pre'
		});
	}
	return create_effect(RENDER_EFFECT | USER_EFFECT, fn, true);
}

/**
 * An effect root whose children can transition out
 * @param {() => void} fn
 * @returns {(options?: { outro?: boolean }) => Promise<void>}
 */
function component_root(fn) {
	Batch.ensure();
	const effect = create_effect(ROOT_EFFECT | EFFECT_PRESERVED, fn, true);

	return (options = {}) => {
		return new Promise((fulfil) => {
			if (options.outro) {
				pause_effect(effect, () => {
					destroy_effect(effect);
					fulfil(undefined);
				});
			} else {
				destroy_effect(effect);
				fulfil(undefined);
			}
		});
	};
}

/**
 * @param {() => void | (() => void)} fn
 * @returns {Effect}
 */
function effect(fn) {
	return create_effect(EFFECT, fn, false);
}

/**
 * Internal representation of `$: ..`
 * @param {() => any} deps
 * @param {() => void | (() => void)} fn
 */
function legacy_pre_effect(deps, fn) {
	var context = /** @type {ComponentContextLegacy} */ (component_context);

	/** @type {{ effect: null | Effect, ran: boolean, deps: () => any }} */
	var token = { effect: null, ran: false, deps };

	context.l.$.push(token);

	token.effect = render_effect(() => {
		deps();

		// If this legacy pre effect has already run before the end of the reset, then
		// bail out to emulate the same behavior.
		if (token.ran) return;

		token.ran = true;
		untrack(fn);
	});
}

function legacy_pre_effect_reset() {
	var context = /** @type {ComponentContextLegacy} */ (component_context);

	render_effect(() => {
		// Run dirty `$:` statements
		for (var token of context.l.$) {
			token.deps();

			var effect = token.effect;

			// If the effect is CLEAN, then make it MAYBE_DIRTY. This ensures we traverse through
			// the effects dependencies and correctly ensure each dependency is up-to-date.
			if ((effect.f & CLEAN) !== 0 && effect.deps !== null) {
				set_signal_status(effect, MAYBE_DIRTY);
			}

			if (is_dirty(effect)) {
				update_effect(effect);
			}

			token.ran = false;
		}
	});
}

/**
 * @param {() => void | (() => void)} fn
 * @returns {Effect}
 */
function async_effect(fn) {
	return create_effect(ASYNC | EFFECT_PRESERVED, fn, true);
}

/**
 * @param {() => void | (() => void)} fn
 * @returns {Effect}
 */
function render_effect(fn, flags = 0) {
	return create_effect(RENDER_EFFECT | flags, fn, true);
}

/**
 * @param {(...expressions: any) => void | (() => void)} fn
 * @param {Array<() => any>} sync
 * @param {Array<() => Promise<any>>} async
 * @param {Array<Promise<void>>} blockers
 */
function template_effect(fn, sync = [], async = [], blockers = []) {
	flatten(blockers, sync, async, (values) => {
		create_effect(RENDER_EFFECT, () => fn(...values.map(get)), true);
	});
}

/**
 * Like `template_effect`, but with an effect which is deferred until the batch commits
 * @param {(...expressions: any) => void | (() => void)} fn
 * @param {Array<() => any>} sync
 * @param {Array<() => Promise<any>>} async
 * @param {Array<Promise<void>>} blockers
 */
function deferred_template_effect(fn, sync = [], async = [], blockers = []) {
	var batch = /** @type {Batch} */ (current_batch);
	var is_async = async.length > 0 || blockers.length > 0;

	if (is_async) batch.increment(true);

	flatten(blockers, sync, async, (values) => {
		create_effect(EFFECT, () => fn(...values.map(get)), false);
		if (is_async) batch.decrement(true);
	});
}

/**
 * @param {(() => void)} fn
 * @param {number} flags
 */
function block(fn, flags = 0) {
	var effect = create_effect(BLOCK_EFFECT | flags, fn, true);
	if (DEV) {
		effect.dev_stack = dev_stack;
	}
	return effect;
}

/**
 * @param {(() => void)} fn
 * @param {number} flags
 */
function managed(fn, flags = 0) {
	var effect = create_effect(MANAGED_EFFECT | flags, fn, true);
	if (DEV) {
		effect.dev_stack = dev_stack;
	}
	return effect;
}

/**
 * @param {(() => void)} fn
 */
function branch(fn) {
	return create_effect(BRANCH_EFFECT | EFFECT_PRESERVED, fn, true);
}

/**
 * @param {Effect} effect
 */
function execute_effect_teardown(effect) {
	var teardown = effect.teardown;
	if (teardown !== null) {
		const previously_destroying_effect = is_destroying_effect;
		const previous_reaction = active_reaction;
		set_is_destroying_effect(true);
		set_active_reaction(null);
		try {
			teardown.call(null);
		} finally {
			set_is_destroying_effect(previously_destroying_effect);
			set_active_reaction(previous_reaction);
		}
	}
}

/**
 * @param {Effect} signal
 * @param {boolean} remove_dom
 * @returns {void}
 */
function destroy_effect_children(signal, remove_dom = false) {
	var effect = signal.first;
	signal.first = signal.last = null;

	while (effect !== null) {
		const controller = effect.ac;

		if (controller !== null) {
			without_reactive_context(() => {
				controller.abort(STALE_REACTION);
			});
		}

		var next = effect.next;

		if ((effect.f & ROOT_EFFECT) !== 0) {
			// this is now an independent root
			effect.parent = null;
		} else {
			destroy_effect(effect, remove_dom);
		}

		effect = next;
	}
}

/**
 * @param {Effect} signal
 * @returns {void}
 */
function destroy_block_effect_children(signal) {
	var effect = signal.first;

	while (effect !== null) {
		var next = effect.next;
		if ((effect.f & BRANCH_EFFECT) === 0) {
			destroy_effect(effect);
		}
		effect = next;
	}
}

/**
 * @param {Effect} effect
 * @param {boolean} [remove_dom]
 * @returns {void}
 */
function destroy_effect(effect, remove_dom = true) {
	var removed = false;

	if (
		(remove_dom || (effect.f & HEAD_EFFECT) !== 0) &&
		effect.nodes !== null &&
		effect.nodes.end !== null
	) {
		remove_effect_dom(effect.nodes.start, /** @type {TemplateNode} */ (effect.nodes.end));
		removed = true;
	}

	destroy_effect_children(effect, remove_dom && !removed);
	remove_reactions(effect, 0);
	set_signal_status(effect, DESTROYED);

	var transitions = effect.nodes && effect.nodes.t;

	if (transitions !== null) {
		for (const transition of transitions) {
			transition.stop();
		}
	}

	execute_effect_teardown(effect);

	var parent = effect.parent;

	// If the parent doesn't have any children, then skip this work altogether
	if (parent !== null && parent.first !== null) {
		unlink_effect(effect);
	}

	if (DEV) {
		effect.component_function = null;
	}

	// `first` and `child` are nulled out in destroy_effect_children
	// we don't null out `parent` so that error propagation can work correctly
	effect.next =
		effect.prev =
		effect.teardown =
		effect.ctx =
		effect.deps =
		effect.fn =
		effect.nodes =
		effect.ac =
			null;
}

/**
 *
 * @param {TemplateNode | null} node
 * @param {TemplateNode} end
 */
function remove_effect_dom(node, end) {
	while (node !== null) {
		/** @type {TemplateNode | null} */
		var next = node === end ? null : get_next_sibling(node);

		node.remove();
		node = next;
	}
}

/**
 * Detach an effect from the effect tree, freeing up memory and
 * reducing the amount of work that happens on subsequent traversals
 * @param {Effect} effect
 */
function unlink_effect(effect) {
	var parent = effect.parent;
	var prev = effect.prev;
	var next = effect.next;

	if (prev !== null) prev.next = next;
	if (next !== null) next.prev = prev;

	if (parent !== null) {
		if (parent.first === effect) parent.first = next;
		if (parent.last === effect) parent.last = prev;
	}
}

/**
 * When a block effect is removed, we don't immediately destroy it or yank it
 * out of the DOM, because it might have transitions. Instead, we 'pause' it.
 * It stays around (in memory, and in the DOM) until outro transitions have
 * completed, and if the state change is reversed then we _resume_ it.
 * A paused effect does not update, and the DOM subtree becomes inert.
 * @param {Effect} effect
 * @param {() => void} [callback]
 * @param {boolean} [destroy]
 */
function pause_effect(effect, callback, destroy = true) {
	/** @type {TransitionManager[]} */
	var transitions = [];

	pause_children(effect, transitions, true);

	var fn = () => {
		if (destroy) destroy_effect(effect);
		if (callback) callback();
	};

	var remaining = transitions.length;
	if (remaining > 0) {
		var check = () => --remaining || fn();
		for (var transition of transitions) {
			transition.out(check);
		}
	} else {
		fn();
	}
}

/**
 * @param {Effect} effect
 * @param {TransitionManager[]} transitions
 * @param {boolean} local
 */
function pause_children(effect, transitions, local) {
	if ((effect.f & INERT) !== 0) return;
	effect.f ^= INERT;

	var t = effect.nodes && effect.nodes.t;

	if (t !== null) {
		for (const transition of t) {
			if (transition.is_global || local) {
				transitions.push(transition);
			}
		}
	}

	var child = effect.first;

	while (child !== null) {
		var sibling = child.next;
		var transparent =
			(child.f & EFFECT_TRANSPARENT) !== 0 ||
			// If this is a branch effect without a block effect parent,
			// it means the parent block effect was pruned. In that case,
			// transparency information was transferred to the branch effect.
			((child.f & BRANCH_EFFECT) !== 0 && (effect.f & BLOCK_EFFECT) !== 0);
		// TODO we don't need to call pause_children recursively with a linked list in place
		// it's slightly more involved though as we have to account for `transparent` changing
		// through the tree.
		pause_children(child, transitions, transparent ? local : false);
		child = sibling;
	}
}

/**
 * The opposite of `pause_effect`. We call this if (for example)
 * `x` becomes falsy then truthy: `{#if x}...{/if}`
 * @param {Effect} effect
 */
function resume_effect(effect) {
	resume_children(effect, true);
}

/**
 * @param {Effect} effect
 * @param {boolean} local
 */
function resume_children(effect, local) {
	if ((effect.f & INERT) === 0) return;
	effect.f ^= INERT;

	// If a dependency of this effect changed while it was paused,
	// schedule the effect to update. we don't use `is_dirty`
	// here because we don't want to eagerly recompute a derived like
	// `{#if foo}{foo.bar()}{/if}` if `foo` is now `undefined
	if ((effect.f & CLEAN) === 0) {
		set_signal_status(effect, DIRTY);
		schedule_effect(effect);
	}

	var child = effect.first;

	while (child !== null) {
		var sibling = child.next;
		var transparent = (child.f & EFFECT_TRANSPARENT) !== 0 || (child.f & BRANCH_EFFECT) !== 0;
		// TODO we don't need to call resume_children recursively with a linked list in place
		// it's slightly more involved though as we have to account for `transparent` changing
		// through the tree.
		resume_children(child, transparent ? local : false);
		child = sibling;
	}

	var t = effect.nodes && effect.nodes.t;

	if (t !== null) {
		for (const transition of t) {
			if (transition.is_global || local) {
				transition.in();
			}
		}
	}
}

function aborted(effect = /** @type {Effect} */ (active_effect)) {
	return (effect.f & DESTROYED) !== 0;
}

/**
 * @param {Effect} effect
 * @param {DocumentFragment} fragment
 */
function move_effect(effect, fragment) {
	if (!effect.nodes) return;

	/** @type {TemplateNode | null} */
	var node = effect.nodes.start;
	var end = effect.nodes.end;

	while (node !== null) {
		/** @type {TemplateNode | null} */
		var next = node === end ? null : get_next_sibling(node);

		fragment.append(node);
		node = next;
	}
}

/** @import { Value } from '#client' */

/**
 * @type {Set<Value> | null}
 * @deprecated
 */
let captured_signals = null;

/**
 * Capture an array of all the signals that are read when `fn` is called
 * @template T
 * @param {() => T} fn
 */
function capture_signals(fn) {
	var previous_captured_signals = captured_signals;

	try {
		captured_signals = new Set();

		untrack(fn);

		if (previous_captured_signals !== null) {
			for (var signal of captured_signals) {
				previous_captured_signals.add(signal);
			}
		}

		return captured_signals;
	} finally {
		captured_signals = previous_captured_signals;
	}
}

/**
 * Invokes a function and captures all signals that are read during the invocation,
 * then invalidates them.
 * @param {() => any} fn
 * @deprecated
 */
function invalidate_inner_signals(fn) {
	for (var signal of capture_signals(fn)) {
		internal_set(signal, signal.v);
	}
}

/** @import { Derived, Effect, Reaction, Source, Value } from '#client' */

let is_updating_effect = false;

/** @param {boolean} value */
function set_is_updating_effect(value) {
	is_updating_effect = value;
}

let is_destroying_effect = false;

/** @param {boolean} value */
function set_is_destroying_effect(value) {
	is_destroying_effect = value;
}

/** @type {null | Reaction} */
let active_reaction = null;

let untracking = false;

/** @param {null | Reaction} reaction */
function set_active_reaction(reaction) {
	active_reaction = reaction;
}

/** @type {null | Effect} */
let active_effect = null;

/** @param {null | Effect} effect */
function set_active_effect(effect) {
	active_effect = effect;
}

/**
 * When sources are created within a reaction, reading and writing
 * them within that reaction should not cause a re-run
 * @type {null | Source[]}
 */
let current_sources = null;

/** @param {Value} value */
function push_reaction_value(value) {
	if (active_reaction !== null && (!async_mode_flag || (active_reaction.f & DERIVED) !== 0)) {
		if (current_sources === null) {
			current_sources = [value];
		} else {
			current_sources.push(value);
		}
	}
}

/**
 * The dependencies of the reaction that is currently being executed. In many cases,
 * the dependencies are unchanged between runs, and so this will be `null` unless
 * and until a new dependency is accessed — we track this via `skipped_deps`
 * @type {null | Value[]}
 */
let new_deps = null;

let skipped_deps = 0;

/**
 * Tracks writes that the effect it's executed in doesn't listen to yet,
 * so that the dependency can be added to the effect later on if it then reads it
 * @type {null | Source[]}
 */
let untracked_writes = null;

/** @param {null | Source[]} value */
function set_untracked_writes(value) {
	untracked_writes = value;
}

/**
 * @type {number} Used by sources and deriveds for handling updates.
 * Version starts from 1 so that unowned deriveds differentiate between a created effect and a run one for tracing
 **/
let write_version = 1;

/** @type {number} Used to version each read of a source of derived to avoid duplicating depedencies inside a reaction */
let read_version = 0;

let update_version = read_version;

/** @param {number} value */
function set_update_version(value) {
	update_version = value;
}

function increment_write_version() {
	return ++write_version;
}

/**
 * Determines whether a derived or effect is dirty.
 * If it is MAYBE_DIRTY, will set the status to CLEAN
 * @param {Reaction} reaction
 * @returns {boolean}
 */
function is_dirty(reaction) {
	var flags = reaction.f;

	if ((flags & DIRTY) !== 0) {
		return true;
	}

	if (flags & DERIVED) {
		reaction.f &= ~WAS_MARKED;
	}

	if ((flags & MAYBE_DIRTY) !== 0) {
		var dependencies = /** @type {Value[]} */ (reaction.deps);
		var length = dependencies.length;

		for (var i = 0; i < length; i++) {
			var dependency = dependencies[i];

			if (is_dirty(/** @type {Derived} */ (dependency))) {
				update_derived(/** @type {Derived} */ (dependency));
			}

			if (dependency.wv > reaction.wv) {
				return true;
			}
		}

		if (
			(flags & CONNECTED) !== 0 &&
			// During time traveling we don't want to reset the status so that
			// traversal of the graph in the other batches still happens
			batch_values === null
		) {
			set_signal_status(reaction, CLEAN);
		}
	}

	return false;
}

/**
 * @param {Value} signal
 * @param {Effect} effect
 * @param {boolean} [root]
 */
function schedule_possible_effect_self_invalidation(signal, effect, root = true) {
	var reactions = signal.reactions;
	if (reactions === null) return;

	if (!async_mode_flag && current_sources?.includes(signal)) {
		return;
	}

	for (var i = 0; i < reactions.length; i++) {
		var reaction = reactions[i];

		if ((reaction.f & DERIVED) !== 0) {
			schedule_possible_effect_self_invalidation(/** @type {Derived} */ (reaction), effect, false);
		} else if (effect === reaction) {
			if (root) {
				set_signal_status(reaction, DIRTY);
			} else if ((reaction.f & CLEAN) !== 0) {
				set_signal_status(reaction, MAYBE_DIRTY);
			}
			schedule_effect(/** @type {Effect} */ (reaction));
		}
	}
}

/** @param {Reaction} reaction */
function update_reaction(reaction) {
	var previous_deps = new_deps;
	var previous_skipped_deps = skipped_deps;
	var previous_untracked_writes = untracked_writes;
	var previous_reaction = active_reaction;
	var previous_sources = current_sources;
	var previous_component_context = component_context;
	var previous_untracking = untracking;
	var previous_update_version = update_version;

	var flags = reaction.f;

	new_deps = /** @type {null | Value[]} */ (null);
	skipped_deps = 0;
	untracked_writes = null;
	active_reaction = (flags & (BRANCH_EFFECT | ROOT_EFFECT)) === 0 ? reaction : null;

	current_sources = null;
	set_component_context(reaction.ctx);
	untracking = false;
	update_version = ++read_version;

	if (reaction.ac !== null) {
		without_reactive_context(() => {
			/** @type {AbortController} */ (reaction.ac).abort(STALE_REACTION);
		});

		reaction.ac = null;
	}

	try {
		reaction.f |= REACTION_IS_UPDATING;
		var fn = /** @type {Function} */ (reaction.fn);
		var result = fn();
		var deps = reaction.deps;

		if (new_deps !== null) {
			var i;

			remove_reactions(reaction, skipped_deps);

			if (deps !== null && skipped_deps > 0) {
				deps.length = skipped_deps + new_deps.length;
				for (i = 0; i < new_deps.length; i++) {
					deps[skipped_deps + i] = new_deps[i];
				}
			} else {
				reaction.deps = deps = new_deps;
			}

			if (effect_tracking() && (reaction.f & CONNECTED) !== 0) {
				for (i = skipped_deps; i < deps.length; i++) {
					(deps[i].reactions ??= []).push(reaction);
				}
			}
		} else if (deps !== null && skipped_deps < deps.length) {
			remove_reactions(reaction, skipped_deps);
			deps.length = skipped_deps;
		}

		// If we're inside an effect and we have untracked writes, then we need to
		// ensure that if any of those untracked writes result in re-invalidation
		// of the current effect, then that happens accordingly
		if (
			is_runes() &&
			untracked_writes !== null &&
			!untracking &&
			deps !== null &&
			(reaction.f & (DERIVED | MAYBE_DIRTY | DIRTY)) === 0
		) {
			for (i = 0; i < /** @type {Source[]} */ (untracked_writes).length; i++) {
				schedule_possible_effect_self_invalidation(
					untracked_writes[i],
					/** @type {Effect} */ (reaction)
				);
			}
		}

		// If we are returning to an previous reaction then
		// we need to increment the read version to ensure that
		// any dependencies in this reaction aren't marked with
		// the same version
		if (previous_reaction !== null && previous_reaction !== reaction) {
			read_version++;

			// update the `rv` of the previous reaction's deps — both existing and new —
			// so that they are not added again
			if (previous_reaction.deps !== null) {
				for (let i = 0; i < previous_skipped_deps; i += 1) {
					previous_reaction.deps[i].rv = read_version;
				}
			}

			if (previous_deps !== null) {
				for (const dep of previous_deps) {
					dep.rv = read_version;
				}
			}

			if (untracked_writes !== null) {
				if (previous_untracked_writes === null) {
					previous_untracked_writes = untracked_writes;
				} else {
					previous_untracked_writes.push(.../** @type {Source[]} */ (untracked_writes));
				}
			}
		}

		if ((reaction.f & ERROR_VALUE) !== 0) {
			reaction.f ^= ERROR_VALUE;
		}

		return result;
	} catch (error) {
		return handle_error(error);
	} finally {
		reaction.f ^= REACTION_IS_UPDATING;
		new_deps = previous_deps;
		skipped_deps = previous_skipped_deps;
		untracked_writes = previous_untracked_writes;
		active_reaction = previous_reaction;
		current_sources = previous_sources;
		set_component_context(previous_component_context);
		untracking = previous_untracking;
		update_version = previous_update_version;
	}
}

/**
 * @template V
 * @param {Reaction} signal
 * @param {Value<V>} dependency
 * @returns {void}
 */
function remove_reaction(signal, dependency) {
	let reactions = dependency.reactions;
	if (reactions !== null) {
		var index = index_of.call(reactions, signal);
		if (index !== -1) {
			var new_length = reactions.length - 1;
			if (new_length === 0) {
				reactions = dependency.reactions = null;
			} else {
				// Swap with last element and then remove.
				reactions[index] = reactions[new_length];
				reactions.pop();
			}
		}
	}

	// If the derived has no reactions, then we can disconnect it from the graph,
	// allowing it to either reconnect in the future, or be GC'd by the VM.
	if (
		reactions === null &&
		(dependency.f & DERIVED) !== 0 &&
		// Destroying a child effect while updating a parent effect can cause a dependency to appear
		// to be unused, when in fact it is used by the currently-updating parent. Checking `new_deps`
		// allows us to skip the expensive work of disconnecting and immediately reconnecting it
		(new_deps === null || !new_deps.includes(dependency))
	) {
		var derived = /** @type {Derived} */ (dependency);

		// If we are working with a derived that is owned by an effect, then mark it as being
		// disconnected and remove the mark flag, as it cannot be reliably removed otherwise
		if ((derived.f & CONNECTED) !== 0) {
			derived.f ^= CONNECTED;
			derived.f &= ~WAS_MARKED;
		}

		update_derived_status(derived);

		// Disconnect any reactions owned by this reaction
		destroy_derived_effects(derived);
		remove_reactions(derived, 0);
	}
}

/**
 * @param {Reaction} signal
 * @param {number} start_index
 * @returns {void}
 */
function remove_reactions(signal, start_index) {
	var dependencies = signal.deps;
	if (dependencies === null) return;

	for (var i = start_index; i < dependencies.length; i++) {
		remove_reaction(signal, dependencies[i]);
	}
}

/**
 * @param {Effect} effect
 * @returns {void}
 */
function update_effect(effect) {
	var flags = effect.f;

	if ((flags & DESTROYED) !== 0) {
		return;
	}

	set_signal_status(effect, CLEAN);

	var previous_effect = active_effect;
	var was_updating_effect = is_updating_effect;

	active_effect = effect;
	is_updating_effect = true;

	if (DEV) {
		var previous_component_fn = dev_current_component_function;
		set_dev_current_component_function(effect.component_function);
		var previous_stack = /** @type {any} */ (dev_stack);
		// only block effects have a dev stack, keep the current one otherwise
		set_dev_stack(effect.dev_stack ?? dev_stack);
	}

	try {
		if ((flags & (BLOCK_EFFECT | MANAGED_EFFECT)) !== 0) {
			destroy_block_effect_children(effect);
		} else {
			destroy_effect_children(effect);
		}

		execute_effect_teardown(effect);
		var teardown = update_reaction(effect);
		effect.teardown = typeof teardown === 'function' ? teardown : null;
		effect.wv = write_version;

		// In DEV, increment versions of any sources that were written to during the effect,
		// so that they are correctly marked as dirty when the effect re-runs
		var dep; if (DEV && tracing_mode_flag && (effect.f & DIRTY) !== 0 && effect.deps !== null) ;
	} finally {
		is_updating_effect = was_updating_effect;
		active_effect = previous_effect;

		if (DEV) {
			set_dev_current_component_function(previous_component_fn);
			set_dev_stack(previous_stack);
		}
	}
}

/**
 * Returns a promise that resolves once any pending state changes have been applied.
 * @returns {Promise<void>}
 */
async function tick() {
	if (async_mode_flag) {
		return new Promise((f) => {
			// Race them against each other - in almost all cases requestAnimationFrame will fire first,
			// but e.g. in case the window is not focused or a view transition happens, requestAnimationFrame
			// will be delayed and setTimeout helps us resolve fast enough in that case
			requestAnimationFrame(() => f());
			setTimeout(() => f());
		});
	}

	await Promise.resolve();

	// By calling flushSync we guarantee that any pending state changes are applied after one tick.
	// TODO look into whether we can make flushing subsequent updates synchronously in the future.
	flushSync();
}

/**
 * @template V
 * @param {Value<V>} signal
 * @returns {V}
 */
function get(signal) {
	var flags = signal.f;
	var is_derived = (flags & DERIVED) !== 0;

	captured_signals?.add(signal);

	// Register the dependency on the current reaction signal.
	if (active_reaction !== null && !untracking) {
		// if we're in a derived that is being read inside an _async_ derived,
		// it's possible that the effect was already destroyed. In this case,
		// we don't add the dependency, because that would create a memory leak
		var destroyed = active_effect !== null && (active_effect.f & DESTROYED) !== 0;

		if (!destroyed && !current_sources?.includes(signal)) {
			var deps = active_reaction.deps;

			if ((active_reaction.f & REACTION_IS_UPDATING) !== 0) {
				// we're in the effect init/update cycle
				if (signal.rv < read_version) {
					signal.rv = read_version;

					// If the signal is accessing the same dependencies in the same
					// order as it did last time, increment `skipped_deps`
					// rather than updating `new_deps`, which creates GC cost
					if (new_deps === null && deps !== null && deps[skipped_deps] === signal) {
						skipped_deps++;
					} else if (new_deps === null) {
						new_deps = [signal];
					} else {
						new_deps.push(signal);
					}
				}
			} else {
				// we're adding a dependency outside the init/update cycle
				// (i.e. after an `await`)
				(active_reaction.deps ??= []).push(signal);

				var reactions = signal.reactions;

				if (reactions === null) {
					signal.reactions = [active_reaction];
				} else if (!reactions.includes(active_reaction)) {
					reactions.push(active_reaction);
				}
			}
		}
	}

	if (DEV) {
		// TODO reinstate this, but make it actually work
		// if (current_async_effect) {
		// 	var tracking = (current_async_effect.f & REACTION_IS_UPDATING) !== 0;
		// 	var was_read = current_async_effect.deps?.includes(signal);

		// 	if (!tracking && !untracking && !was_read) {
		// 		w.await_reactivity_loss(/** @type {string} */ (signal.label));

		// 		var trace = get_error('traced at');
		// 		// eslint-disable-next-line no-console
		// 		if (trace) console.warn(trace);
		// 	}
		// }

		recent_async_deriveds.delete(signal);
	}

	if (is_destroying_effect && old_values.has(signal)) {
		return old_values.get(signal);
	}

	if (is_derived) {
		var derived = /** @type {Derived} */ (signal);

		if (is_destroying_effect) {
			var value = derived.v;

			// if the derived is dirty and has reactions, or depends on the values that just changed, re-execute
			// (a derived can be maybe_dirty due to the effect destroy removing its last reaction)
			if (
				((derived.f & CLEAN) === 0 && derived.reactions !== null) ||
				depends_on_old_values(derived)
			) {
				value = execute_derived(derived);
			}

			old_values.set(derived, value);

			return value;
		}

		// connect disconnected deriveds if we are reading them inside an effect,
		// or inside another derived that is already connected
		var should_connect =
			(derived.f & CONNECTED) === 0 &&
			!untracking &&
			active_reaction !== null &&
			(is_updating_effect || (active_reaction.f & CONNECTED) !== 0);

		var is_new = derived.deps === null;

		if (is_dirty(derived)) {
			if (should_connect) {
				// set the flag before `update_derived`, so that the derived
				// is added as a reaction to its dependencies
				derived.f |= CONNECTED;
			}

			update_derived(derived);
		}

		if (should_connect && !is_new) {
			reconnect(derived);
		}
	}

	if (batch_values?.has(signal)) {
		return batch_values.get(signal);
	}

	if ((signal.f & ERROR_VALUE) !== 0) {
		throw signal.v;
	}

	return signal.v;
}

/**
 * (Re)connect a disconnected derived, so that it is notified
 * of changes in `mark_reactions`
 * @param {Derived} derived
 */
function reconnect(derived) {
	if (derived.deps === null) return;

	derived.f |= CONNECTED;

	for (const dep of derived.deps) {
		(dep.reactions ??= []).push(derived);

		if ((dep.f & DERIVED) !== 0 && (dep.f & CONNECTED) === 0) {
			reconnect(/** @type {Derived} */ (dep));
		}
	}
}

/** @param {Derived} derived */
function depends_on_old_values(derived) {
	if (derived.v === UNINITIALIZED) return true; // we don't know, so assume the worst
	if (derived.deps === null) return false;

	for (const dep of derived.deps) {
		if (old_values.has(dep)) {
			return true;
		}

		if ((dep.f & DERIVED) !== 0 && depends_on_old_values(/** @type {Derived} */ (dep))) {
			return true;
		}
	}

	return false;
}

/**
 * When used inside a [`$derived`](https://svelte.dev/docs/svelte/$derived) or [`$effect`](https://svelte.dev/docs/svelte/$effect),
 * any state read inside `fn` will not be treated as a dependency.
 *
 * ```ts
 * $effect(() => {
 *   // this will run when `data` changes, but not when `time` changes
 *   save(data, {
 *     timestamp: untrack(() => time)
 *   });
 * });
 * ```
 * @template T
 * @param {() => T} fn
 * @returns {T}
 */
function untrack(fn) {
	var previous_untracking = untracking;
	try {
		untracking = true;
		return fn();
	} finally {
		untracking = previous_untracking;
	}
}

/**
 * Possibly traverse an object and read all its properties so that they're all reactive in case this is `$state`.
 * Does only check first level of an object for performance reasons (heuristic should be good for 99% of all cases).
 * @param {any} value
 * @returns {void}
 */
function deep_read_state(value) {
	if (typeof value !== 'object' || !value || value instanceof EventTarget) {
		return;
	}

	if (STATE_SYMBOL in value) {
		deep_read(value);
	} else if (!Array.isArray(value)) {
		for (let key in value) {
			const prop = value[key];
			if (typeof prop === 'object' && prop && STATE_SYMBOL in prop) {
				deep_read(prop);
			}
		}
	}
}

/**
 * Deeply traverse an object and read all its properties
 * so that they're all reactive in case this is `$state`
 * @param {any} value
 * @param {Set<any>} visited
 * @returns {void}
 */
function deep_read(value, visited = new Set()) {
	if (
		typeof value === 'object' &&
		value !== null &&
		// We don't want to traverse DOM elements
		!(value instanceof EventTarget) &&
		!visited.has(value)
	) {
		visited.add(value);
		// When working with a possible SvelteDate, this
		// will ensure we capture changes to it.
		if (value instanceof Date) {
			value.getTime();
		}
		for (let key in value) {
			try {
				deep_read(value[key], visited);
			} catch (e) {
				// continue
			}
		}
		const proto = get_prototype_of(value);
		if (
			proto !== Object.prototype &&
			proto !== Array.prototype &&
			proto !== Map.prototype &&
			proto !== Set.prototype &&
			proto !== Date.prototype
		) {
			const descriptors = get_descriptors(proto);
			for (let key in descriptors) {
				const get = descriptors[key].get;
				if (get) {
					try {
						get.call(value);
					} catch (e) {
						// continue
					}
				}
			}
		}
	}
}

const regex_return_characters = /\r/g;

/**
 * @param {string} str
 * @returns {string}
 */
function hash(str) {
	str = str.replace(regex_return_characters, '');
	let hash = 5381;
	let i = str.length;

	while (i--) hash = ((hash << 5) - hash) ^ str.charCodeAt(i);
	return (hash >>> 0).toString(36);
}

/**
 * @param {string} name
 */
function is_capture_event(name) {
	return name.endsWith('capture') && name !== 'gotpointercapture' && name !== 'lostpointercapture';
}

/** List of Element events that will be delegated */
const DELEGATED_EVENTS = [
	'beforeinput',
	'click',
	'change',
	'dblclick',
	'contextmenu',
	'focusin',
	'focusout',
	'input',
	'keydown',
	'keyup',
	'mousedown',
	'mousemove',
	'mouseout',
	'mouseover',
	'mouseup',
	'pointerdown',
	'pointermove',
	'pointerout',
	'pointerover',
	'pointerup',
	'touchend',
	'touchmove',
	'touchstart'
];

/**
 * Returns `true` if `event_name` is a delegated event
 * @param {string} event_name
 */
function can_delegate_event(event_name) {
	return DELEGATED_EVENTS.includes(event_name);
}

/**
 * @type {Record<string, string>}
 * List of attribute names that should be aliased to their property names
 * because they behave differently between setting them as an attribute and
 * setting them as a property.
 */
const ATTRIBUTE_ALIASES = {
	// no `class: 'className'` because we handle that separately
	formnovalidate: 'formNoValidate',
	ismap: 'isMap',
	nomodule: 'noModule',
	playsinline: 'playsInline',
	readonly: 'readOnly',
	defaultvalue: 'defaultValue',
	defaultchecked: 'defaultChecked',
	srcobject: 'srcObject',
	novalidate: 'noValidate',
	allowfullscreen: 'allowFullscreen',
	disablepictureinpicture: 'disablePictureInPicture',
	disableremoteplayback: 'disableRemotePlayback'
};

/**
 * @param {string} name
 */
function normalize_attribute(name) {
	name = name.toLowerCase();
	return ATTRIBUTE_ALIASES[name] ?? name;
}

/**
 * Subset of delegated events which should be passive by default.
 * These two are already passive via browser defaults on window, document and body.
 * But since
 * - we're delegating them
 * - they happen often
 * - they apply to mobile which is generally less performant
 * we're marking them as passive by default for other elements, too.
 */
const PASSIVE_EVENTS = ['touchstart', 'touchmove'];

/**
 * Returns `true` if `name` is a passive event
 * @param {string} name
 */
function is_passive_event(name) {
	return PASSIVE_EVENTS.includes(name);
}

/** List of elements that require raw contents and should not have SSR comments put in them */
const RAW_TEXT_ELEMENTS = /** @type {const} */ (['textarea', 'script', 'style', 'title']);

/** @param {string} name */
function is_raw_text_element(name) {
	return RAW_TEXT_ELEMENTS.includes(/** @type {typeof RAW_TEXT_ELEMENTS[number]} */ (name));
}

/**
 * Prevent devtools trying to make `location` a clickable link by inserting a zero-width space
 * @template {string | undefined} T
 * @param {T} location
 * @returns {T};
 */
function sanitize_location(location) {
	return /** @type {T} */ (location?.replace(/\//g, '/\u200b'));
}

/** @type {Set<string>} */
const all_registered_events = new Set();

/** @type {Set<(events: Array<string>) => void>} */
const root_event_handles = new Set();

/**
 * SSR adds onload and onerror attributes to catch those events before the hydration.
 * This function detects those cases, removes the attributes and replays the events.
 * @param {HTMLElement} dom
 */
function replay_events(dom) {
	if (!hydrating) return;

	dom.removeAttribute('onload');
	dom.removeAttribute('onerror');
	// @ts-expect-error
	const event = dom.__e;
	if (event !== undefined) {
		// @ts-expect-error
		dom.__e = undefined;
		queueMicrotask(() => {
			if (dom.isConnected) {
				dom.dispatchEvent(event);
			}
		});
	}
}

/**
 * @param {string} event_name
 * @param {EventTarget} dom
 * @param {EventListener} [handler]
 * @param {AddEventListenerOptions} [options]
 */
function create_event(event_name, dom, handler, options = {}) {
	/**
	 * @this {EventTarget}
	 */
	function target_handler(/** @type {Event} */ event) {
		if (!options.capture) {
			// Only call in the bubble phase, else delegated events would be called before the capturing events
			handle_event_propagation.call(dom, event);
		}
		if (!event.cancelBubble) {
			return without_reactive_context(() => {
				return handler?.call(this, event);
			});
		}
	}

	// Chrome has a bug where pointer events don't work when attached to a DOM element that has been cloned
	// with cloneNode() and the DOM element is disconnected from the document. To ensure the event works, we
	// defer the attachment till after it's been appended to the document. TODO: remove this once Chrome fixes
	// this bug. The same applies to wheel events and touch events.
	if (
		event_name.startsWith('pointer') ||
		event_name.startsWith('touch') ||
		event_name === 'wheel'
	) {
		queue_micro_task(() => {
			dom.addEventListener(event_name, target_handler, options);
		});
	} else {
		dom.addEventListener(event_name, target_handler, options);
	}

	return target_handler;
}

/**
 * @param {string} event_name
 * @param {Element} dom
 * @param {EventListener} [handler]
 * @param {boolean} [capture]
 * @param {boolean} [passive]
 * @returns {void}
 */
function event(event_name, dom, handler, capture, passive) {
	var options = { capture, passive };
	var target_handler = create_event(event_name, dom, handler, options);

	if (
		dom === document.body ||
		// @ts-ignore
		dom === window ||
		// @ts-ignore
		dom === document ||
		// Firefox has quirky behavior, it can happen that we still get "canplay" events when the element is already removed
		dom instanceof HTMLMediaElement
	) {
		teardown(() => {
			dom.removeEventListener(event_name, target_handler, options);
		});
	}
}

/**
 * @param {Array<string>} events
 * @returns {void}
 */
function delegate(events) {
	for (var i = 0; i < events.length; i++) {
		all_registered_events.add(events[i]);
	}

	for (var fn of root_event_handles) {
		fn(events);
	}
}

// used to store the reference to the currently propagated event
// to prevent garbage collection between microtasks in Firefox
// If the event object is GCed too early, the expando __root property
// set on the event object is lost, causing the event delegation
// to process the event twice
let last_propagated_event = null;

/**
 * @this {EventTarget}
 * @param {Event} event
 * @returns {void}
 */
function handle_event_propagation(event) {
	var handler_element = this;
	var owner_document = /** @type {Node} */ (handler_element).ownerDocument;
	var event_name = event.type;
	var path = event.composedPath?.() || [];
	var current_target = /** @type {null | Element} */ (path[0] || event.target);

	last_propagated_event = event;

	// composedPath contains list of nodes the event has propagated through.
	// We check __root to skip all nodes below it in case this is a
	// parent of the __root node, which indicates that there's nested
	// mounted apps. In this case we don't want to trigger events multiple times.
	var path_idx = 0;

	// the `last_propagated_event === event` check is redundant, but
	// without it the variable will be DCE'd and things will
	// fail mysteriously in Firefox
	// @ts-expect-error is added below
	var handled_at = last_propagated_event === event && event.__root;

	if (handled_at) {
		var at_idx = path.indexOf(handled_at);
		if (
			at_idx !== -1 &&
			(handler_element === document || handler_element === /** @type {any} */ (window))
		) {
			// This is the fallback document listener or a window listener, but the event was already handled
			// -> ignore, but set handle_at to document/window so that we're resetting the event
			// chain in case someone manually dispatches the same event object again.
			// @ts-expect-error
			event.__root = handler_element;
			return;
		}

		// We're deliberately not skipping if the index is higher, because
		// someone could create an event programmatically and emit it multiple times,
		// in which case we want to handle the whole propagation chain properly each time.
		// (this will only be a false negative if the event is dispatched multiple times and
		// the fallback document listener isn't reached in between, but that's super rare)
		var handler_idx = path.indexOf(handler_element);
		if (handler_idx === -1) {
			// handle_idx can theoretically be -1 (happened in some JSDOM testing scenarios with an event listener on the window object)
			// so guard against that, too, and assume that everything was handled at this point.
			return;
		}

		if (at_idx <= handler_idx) {
			path_idx = at_idx;
		}
	}

	current_target = /** @type {Element} */ (path[path_idx] || event.target);
	// there can only be one delegated event per element, and we either already handled the current target,
	// or this is the very first target in the chain which has a non-delegated listener, in which case it's safe
	// to handle a possible delegated event on it later (through the root delegation listener for example).
	if (current_target === handler_element) return;

	// Proxy currentTarget to correct target
	define_property(event, 'currentTarget', {
		configurable: true,
		get() {
			return current_target || owner_document;
		}
	});

	// This started because of Chromium issue https://chromestatus.com/feature/5128696823545856,
	// where removal or moving of of the DOM can cause sync `blur` events to fire, which can cause logic
	// to run inside the current `active_reaction`, which isn't what we want at all. However, on reflection,
	// it's probably best that all event handled by Svelte have this behaviour, as we don't really want
	// an event handler to run in the context of another reaction or effect.
	var previous_reaction = active_reaction;
	var previous_effect = active_effect;
	set_active_reaction(null);
	set_active_effect(null);

	try {
		/**
		 * @type {unknown}
		 */
		var throw_error;
		/**
		 * @type {unknown[]}
		 */
		var other_errors = [];

		while (current_target !== null) {
			/** @type {null | Element} */
			var parent_element =
				current_target.assignedSlot ||
				current_target.parentNode ||
				/** @type {any} */ (current_target).host ||
				null;

			try {
				// @ts-expect-error
				var delegated = current_target['__' + event_name];

				if (
					delegated != null &&
					(!(/** @type {any} */ (current_target).disabled) ||
						// DOM could've been updated already by the time this is reached, so we check this as well
						// -> the target could not have been disabled because it emits the event in the first place
						event.target === current_target)
				) {
					delegated.call(current_target, event);
				}
			} catch (error) {
				if (throw_error) {
					other_errors.push(error);
				} else {
					throw_error = error;
				}
			}
			if (event.cancelBubble || parent_element === handler_element || parent_element === null) {
				break;
			}
			current_target = parent_element;
		}

		if (throw_error) {
			for (let error of other_errors) {
				// Throw the rest of the errors, one-by-one on a microtask
				queueMicrotask(() => {
					throw error;
				});
			}
			throw throw_error;
		}
	} finally {
		// @ts-expect-error is used above
		event.__root = handler_element;
		// @ts-ignore remove proxy on currentTarget
		delete event.currentTarget;
		set_active_reaction(previous_reaction);
		set_active_effect(previous_effect);
	}
}

/** @param {string} html */
function create_fragment_from_html(html) {
	var elem = document.createElement('template');
	elem.innerHTML = html.replaceAll('<!>', '<!---->'); // XHTML compliance
	return elem.content;
}

/** @import { Effect, EffectNodes, TemplateNode } from '#client' */
/** @import { TemplateStructure } from './types' */

/**
 * @param {TemplateNode} start
 * @param {TemplateNode | null} end
 */
function assign_nodes(start, end) {
	var effect = /** @type {Effect} */ (active_effect);
	if (effect.nodes === null) {
		effect.nodes = { start, end, a: null, t: null };
	}
}

/**
 * @param {string} content
 * @param {number} flags
 * @returns {() => Node | Node[]}
 */
/*#__NO_SIDE_EFFECTS__*/
function from_html(content, flags) {
	var is_fragment = (flags & TEMPLATE_FRAGMENT) !== 0;
	var use_import_node = (flags & TEMPLATE_USE_IMPORT_NODE) !== 0;

	/** @type {Node} */
	var node;

	/**
	 * Whether or not the first item is a text/element node. If not, we need to
	 * create an additional comment node to act as `effect.nodes.start`
	 */
	var has_start = !content.startsWith('<!>');

	return () => {
		if (hydrating) {
			assign_nodes(hydrate_node, null);
			return hydrate_node;
		}

		if (node === undefined) {
			node = create_fragment_from_html(has_start ? content : '<!>' + content);
			if (!is_fragment) node = /** @type {TemplateNode} */ (get_first_child(node));
		}

		var clone = /** @type {TemplateNode} */ (
			use_import_node || is_firefox ? document.importNode(node, true) : node.cloneNode(true)
		);

		if (is_fragment) {
			var start = /** @type {TemplateNode} */ (get_first_child(clone));
			var end = /** @type {TemplateNode} */ (clone.lastChild);

			assign_nodes(start, end);
		} else {
			assign_nodes(clone, clone);
		}

		return clone;
	};
}

/**
 * @param {string} content
 * @param {number} flags
 * @param {'svg' | 'math'} ns
 * @returns {() => Node | Node[]}
 */
/*#__NO_SIDE_EFFECTS__*/
function from_namespace(content, flags, ns = 'svg') {
	/**
	 * Whether or not the first item is a text/element node. If not, we need to
	 * create an additional comment node to act as `effect.nodes.start`
	 */
	var has_start = !content.startsWith('<!>');
	var wrapped = `<${ns}>${has_start ? content : '<!>' + content}</${ns}>`;

	/** @type {Element | DocumentFragment} */
	var node;

	return () => {
		if (hydrating) {
			assign_nodes(hydrate_node, null);
			return hydrate_node;
		}

		if (!node) {
			var fragment = /** @type {DocumentFragment} */ (create_fragment_from_html(wrapped));
			var root = /** @type {Element} */ (get_first_child(fragment));

			{
				node = /** @type {Element} */ (get_first_child(root));
			}
		}

		var clone = /** @type {TemplateNode} */ (node.cloneNode(true));

		{
			assign_nodes(clone, clone);
		}

		return clone;
	};
}

/**
 * @param {string} content
 * @param {number} flags
 */
/*#__NO_SIDE_EFFECTS__*/
function from_svg(content, flags) {
	return from_namespace(content, flags, 'svg');
}

/**
 * Don't mark this as side-effect-free, hydration needs to walk all nodes
 * @param {any} value
 */
function text(value = '') {
	if (!hydrating) {
		var t = create_text(value + '');
		assign_nodes(t, t);
		return t;
	}

	var node = hydrate_node;

	if (node.nodeType !== TEXT_NODE) {
		// if an {expression} is empty during SSR, we need to insert an empty text node
		node.before((node = create_text()));
		set_hydrate_node(node);
	}

	assign_nodes(node, node);
	return node;
}

/**
 * @returns {TemplateNode | DocumentFragment}
 */
function comment() {
	// we're not delegating to `template` here for performance reasons
	if (hydrating) {
		assign_nodes(hydrate_node, null);
		return hydrate_node;
	}

	var frag = document.createDocumentFragment();
	var start = document.createComment('');
	var anchor = create_text();
	frag.append(start, anchor);

	assign_nodes(start, anchor);

	return frag;
}

/**
 * Assign the created (or in hydration mode, traversed) dom elements to the current block
 * and insert the elements into the dom (in client mode).
 * @param {Text | Comment | Element} anchor
 * @param {DocumentFragment | Element} dom
 */
function append(anchor, dom) {
	if (hydrating) {
		var effect = /** @type {Effect & { nodes: EffectNodes }} */ (active_effect);

		// When hydrating and outer component and an inner component is async, i.e. blocked on a promise,
		// then by the time the inner resolves we have already advanced to the end of the hydrated nodes
		// of the parent component. Check for defined for that reason to avoid rewinding the parent's end marker.
		if ((effect.f & EFFECT_RAN) === 0 || effect.nodes.end === null) {
			effect.nodes.end = hydrate_node;
		}

		hydrate_next();
		return;
	}

	if (anchor === null) {
		// edge case — void `<svelte:element>` with content
		return;
	}

	anchor.before(/** @type {Node} */ (dom));
}

/** @import { ComponentContext, Effect, EffectNodes, TemplateNode } from '#client' */
/** @import { Component, ComponentType, SvelteComponent, MountOptions } from '../../index.js' */

/**
 * This is normally true — block effects should run their intro transitions —
 * but is false during hydration (unless `options.intro` is `true`) and
 * when creating the children of a `<svelte:element>` that just changed tag
 */
let should_intro = true;

/** @param {boolean} value */
function set_should_intro(value) {
	should_intro = value;
}

/**
 * @param {Element} text
 * @param {string} value
 * @returns {void}
 */
function set_text(text, value) {
	// For objects, we apply string coercion (which might make things like $state array references in the template reactive) before diffing
	var str = value == null ? '' : typeof value === 'object' ? value + '' : value;
	// @ts-expect-error
	if (str !== (text.__t ??= text.nodeValue)) {
		// @ts-expect-error
		text.__t = str;
		text.nodeValue = str + '';
	}
}

/**
 * Mounts a component to the given target and returns the exports and potentially the props (if compiled with `accessors: true`) of the component.
 * Transitions will play during the initial render unless the `intro` option is set to `false`.
 *
 * @template {Record<string, any>} Props
 * @template {Record<string, any>} Exports
 * @param {ComponentType<SvelteComponent<Props>> | Component<Props, Exports, any>} component
 * @param {MountOptions<Props>} options
 * @returns {Exports}
 */
function mount(component, options) {
	return _mount(component, options);
}

/** @type {Map<string, number>} */
const document_listeners = new Map();

/**
 * @template {Record<string, any>} Exports
 * @param {ComponentType<SvelteComponent<any>> | Component<any>} Component
 * @param {MountOptions} options
 * @returns {Exports}
 */
function _mount(Component, { target, anchor, props = {}, events, context, intro = true }) {
	init_operations();

	/** @type {Set<string>} */
	var registered_events = new Set();

	/** @param {Array<string>} events */
	var event_handle = (events) => {
		for (var i = 0; i < events.length; i++) {
			var event_name = events[i];

			if (registered_events.has(event_name)) continue;
			registered_events.add(event_name);

			var passive = is_passive_event(event_name);

			// Add the event listener to both the container and the document.
			// The container listener ensures we catch events from within in case
			// the outer content stops propagation of the event.
			target.addEventListener(event_name, handle_event_propagation, { passive });

			var n = document_listeners.get(event_name);

			if (n === undefined) {
				// The document listener ensures we catch events that originate from elements that were
				// manually moved outside of the container (e.g. via manual portals).
				document.addEventListener(event_name, handle_event_propagation, { passive });
				document_listeners.set(event_name, 1);
			} else {
				document_listeners.set(event_name, n + 1);
			}
		}
	};

	event_handle(array_from(all_registered_events));
	root_event_handles.add(event_handle);

	/** @type {Exports} */
	// @ts-expect-error will be defined because the render effect runs synchronously
	var component = undefined;

	var unmount = component_root(() => {
		var anchor_node = anchor ?? target.appendChild(create_text());

		boundary(
			/** @type {TemplateNode} */ (anchor_node),
			{
				pending: () => {}
			},
			(anchor_node) => {
				if (context) {
					push({});
					var ctx = /** @type {ComponentContext} */ (component_context);
					ctx.c = context;
				}

				if (events) {
					// We can't spread the object or else we'd lose the state proxy stuff, if it is one
					/** @type {any} */ (props).$$events = events;
				}

				if (hydrating) {
					assign_nodes(/** @type {TemplateNode} */ (anchor_node), null);
				}

				should_intro = intro;
				// @ts-expect-error the public typings are not what the actual function looks like
				component = Component(anchor_node, props) || {};
				should_intro = true;

				if (hydrating) {
					/** @type {Effect & { nodes: EffectNodes }} */ (active_effect).nodes.end = hydrate_node;

					if (
						hydrate_node === null ||
						hydrate_node.nodeType !== COMMENT_NODE ||
						/** @type {Comment} */ (hydrate_node).data !== HYDRATION_END
					) {
						hydration_mismatch();
						throw HYDRATION_ERROR;
					}
				}

				if (context) {
					pop();
				}
			}
		);

		return () => {
			for (var event_name of registered_events) {
				target.removeEventListener(event_name, handle_event_propagation);

				var n = /** @type {number} */ (document_listeners.get(event_name));

				if (--n === 0) {
					document.removeEventListener(event_name, handle_event_propagation);
					document_listeners.delete(event_name);
				} else {
					document_listeners.set(event_name, n);
				}
			}

			root_event_handles.delete(event_handle);

			if (anchor_node !== anchor) {
				anchor_node.parentNode?.removeChild(anchor_node);
			}
		};
	});

	mounted_components.set(component, unmount);
	return component;
}

/**
 * References of the components that were mounted or hydrated.
 * Uses a `WeakMap` to avoid memory leaks.
 */
let mounted_components = new WeakMap();

/**
 * Unmounts a component that was previously mounted using `mount` or `hydrate`.
 *
 * Since 5.13.0, if `options.outro` is `true`, [transitions](https://svelte.dev/docs/svelte/transition) will play before the component is removed from the DOM.
 *
 * Returns a `Promise` that resolves after transitions have completed if `options.outro` is true, or immediately otherwise (prior to 5.13.0, returns `void`).
 *
 * ```js
 * import { mount, unmount } from 'svelte';
 * import App from './App.svelte';
 *
 * const app = mount(App, { target: document.body });
 *
 * // later...
 * unmount(app, { outro: true });
 * ```
 * @param {Record<string, any>} component
 * @param {{ outro?: boolean }} [options]
 * @returns {Promise<void>}
 */
function unmount(component, options) {
	const fn = mounted_components.get(component);

	if (fn) {
		mounted_components.delete(component);
		return fn(options);
	}

	if (DEV) {
		if (STATE_SYMBOL in component) {
			state_proxy_unmount();
		} else {
			lifecycle_double_unmount();
		}
	}

	return Promise.resolve();
}

/** @import { ComponentContext, ComponentContextLegacy } from '#client' */
/** @import { EventDispatcher } from './index.js' */
/** @import { NotFunction } from './internal/types.js' */

if (DEV) {
	/**
	 * @param {string} rune
	 */
	function throw_rune_error(rune) {
		if (!(rune in globalThis)) {
			// TODO if people start adjusting the "this can contain runes" config through v-p-s more, adjust this message
			/** @type {any} */
			let value; // let's hope noone modifies this global, but belts and braces
			Object.defineProperty(globalThis, rune, {
				configurable: true,
				// eslint-disable-next-line getter-return
				get: () => {
					if (value !== undefined) {
						return value;
					}

					rune_outside_svelte(rune);
				},
				set: (v) => {
					value = v;
				}
			});
		}
	}

	throw_rune_error('$state');
	throw_rune_error('$effect');
	throw_rune_error('$derived');
	throw_rune_error('$inspect');
	throw_rune_error('$props');
	throw_rune_error('$bindable');
}

/**
 * `onMount`, like [`$effect`](https://svelte.dev/docs/svelte/$effect), schedules a function to run as soon as the component has been mounted to the DOM.
 * Unlike `$effect`, the provided function only runs once.
 *
 * It must be called during the component's initialisation (but doesn't need to live _inside_ the component;
 * it can be called from an external module). If a function is returned _synchronously_ from `onMount`,
 * it will be called when the component is unmounted.
 *
 * `onMount` functions do not run during [server-side rendering](https://svelte.dev/docs/svelte/svelte-server#render).
 *
 * @template T
 * @param {() => NotFunction<T> | Promise<NotFunction<T>> | (() => any)} fn
 * @returns {void}
 */
function onMount(fn) {
	if (component_context === null) {
		lifecycle_outside_component('onMount');
	}

	if (legacy_mode_flag && component_context.l !== null) {
		init_update_callbacks(component_context).m.push(fn);
	} else {
		user_effect(() => {
			const cleanup = untrack(fn);
			if (typeof cleanup === 'function') return /** @type {() => void} */ (cleanup);
		});
	}
}

/**
 * Schedules a callback to run immediately before the component is unmounted.
 *
 * Out of `onMount`, `beforeUpdate`, `afterUpdate` and `onDestroy`, this is the
 * only one that runs inside a server-side component.
 *
 * @param {() => any} fn
 * @returns {void}
 */
function onDestroy(fn) {
	if (component_context === null) {
		lifecycle_outside_component('onDestroy');
	}

	onMount(() => () => untrack(fn));
}

/**
 * @template [T=any]
 * @param {string} type
 * @param {T} [detail]
 * @param {any}params_0
 * @returns {CustomEvent<T>}
 */
function create_custom_event(type, detail, { bubbles = false, cancelable = false } = {}) {
	return new CustomEvent(type, { detail, bubbles, cancelable });
}

/**
 * Creates an event dispatcher that can be used to dispatch [component events](https://svelte.dev/docs/svelte/legacy-on#Component-events).
 * Event dispatchers are functions that can take two arguments: `name` and `detail`.
 *
 * Component events created with `createEventDispatcher` create a
 * [CustomEvent](https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent).
 * These events do not [bubble](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Building_blocks/Events#Event_bubbling_and_capture).
 * The `detail` argument corresponds to the [CustomEvent.detail](https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent/detail)
 * property and can contain any type of data.
 *
 * The event dispatcher can be typed to narrow the allowed event names and the type of the `detail` argument:
 * ```ts
 * const dispatch = createEventDispatcher<{
 *  loaded: null; // does not take a detail argument
 *  change: string; // takes a detail argument of type string, which is required
 *  optional: number | null; // takes an optional detail argument of type number
 * }>();
 * ```
 *
 * @deprecated Use callback props and/or the `$host()` rune instead — see [migration guide](https://svelte.dev/docs/svelte/v5-migration-guide#Event-changes-Component-events)
 * @template {Record<string, any>} [EventMap = any]
 * @returns {EventDispatcher<EventMap>}
 */
function createEventDispatcher() {
	const active_component_context = component_context;
	if (active_component_context === null) {
		lifecycle_outside_component('createEventDispatcher');
	}

	/**
	 * @param [detail]
	 * @param [options]
	 */
	return (type, detail, options) => {
		const events = /** @type {Record<string, Function | Function[]>} */ (
			active_component_context.s.$$events
		)?.[/** @type {string} */ (type)];

		if (events) {
			const callbacks = is_array(events) ? events.slice() : [events];
			// TODO are there situations where events could be dispatched
			// in a server (non-DOM) environment?
			const event = create_custom_event(/** @type {string} */ (type), detail, options);
			for (const fn of callbacks) {
				fn.call(active_component_context.x, event);
			}
			return !event.defaultPrevented;
		}

		return true;
	};
}

/**
 * Schedules a callback to run immediately after the component has been updated.
 *
 * The first time the callback runs will be after the initial `onMount`.
 *
 * In runes mode use `$effect` instead.
 *
 * @deprecated Use [`$effect`](https://svelte.dev/docs/svelte/$effect) instead
 * @param {() => void} fn
 * @returns {void}
 */
function afterUpdate(fn) {
	if (component_context === null) {
		lifecycle_outside_component('afterUpdate');
	}

	if (component_context.l === null) {
		lifecycle_legacy_only('afterUpdate');
	}

	init_update_callbacks(component_context).a.push(fn);
}

/**
 * Legacy-mode: Init callbacks object for onMount/beforeUpdate/afterUpdate
 * @param {ComponentContext} context
 */
function init_update_callbacks(context) {
	var l = /** @type {ComponentContextLegacy} */ (context).l;
	return (l.u ??= { a: [], b: [], m: [] });
}

const ENTRY_CSS = "./assets/index-CrbFh6qt.css";
let FONTS;
FONTS = [];
let IndexComponent;
let mount_css;
let _res;
let pending = new Promise((res) => {
  _res = res;
});
globalThis.__MODE__ ??= "_NORMAL_";
const mode = globalThis.__MODE__;
async function get_index() {
  if (mode === "_CC_") {
    await __vitePreload(() => import('virtual:cc-init'),true              ?[]:void 0,import.meta.url);
  }
  const modules = await Promise.all([
    __vitePreload(() => import('./Index-BRJ0gETR.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12]):void 0,import.meta.url),
    __vitePreload(() => import('./index-BJk8VOjP.js'),true              ?__vite__mapDeps([13,1,2,14,15,16]):void 0,import.meta.url)
  ]);
  IndexComponent = modules[0].default;
  mount_css = modules[1].mount_css;
  _res();
}
function create_custom_element() {
  class GradioApp extends HTMLElement {
    control_page_title;
    initial_height;
    is_embed;
    container;
    info;
    autoscroll;
    eager;
    theme_mode;
    host;
    space;
    src;
    app;
    loading;
    updating;
    constructor() {
      super();
      this.host = this.getAttribute("host");
      this.space = this.getAttribute("space");
      this.src = this.getAttribute("src");
      this.control_page_title = this.getAttribute("control_page_title");
      this.initial_height = this.getAttribute("initial_height") ?? "300px";
      this.is_embed = this.getAttribute("embed") ?? "true";
      this.container = this.getAttribute("container") ?? "true";
      this.info = this.getAttribute("info") ?? true;
      this.autoscroll = this.getAttribute("autoscroll");
      this.eager = this.getAttribute("eager");
      this.theme_mode = this.getAttribute("theme_mode");
      this.updating = false;
      this.loading = false;
    }
    async connectedCallback() {
      await get_index();
      this.loading = true;
      if (this.app) {
        this.app.$destroy();
      }
      if (typeof FONTS !== "string") {
        FONTS.forEach((f) => mount_css(f, document.head));
      }
      await mount_css(ENTRY_CSS, document.head);
      const event = new CustomEvent("domchange", {
        bubbles: true,
        cancelable: false,
        composed: true
      });
      const observer = new MutationObserver((mutations) => {
        this.dispatchEvent(event);
      });
      observer.observe(this, { childList: true });
      const opts = {
        target: this,
        props: {
          // embed source
          space: this.space ? this.space.trim() : this.space,
          src: this.src ? this.src.trim() : this.src,
          // embed info
          info: this.info === "false" ? false : true,
          container: this.container === "false" ? false : true,
          is_embed: this.is_embed === "false" ? false : true,
          initial_height: this.initial_height,
          eager: this.eager === "true" ? true : false,
          // gradio meta info
          version: "6-11-0",
          theme_mode: this.theme_mode,
          // misc global behaviour
          autoscroll: this.autoscroll === "true" ? true : false,
          control_page_title: this.control_page_title === "true" ? true : false,
          // injectables
          Client,
          // for gradio docs
          // TODO: Remove -- i think this is just for autoscroll behavhiour, app vs embeds
          app_mode: window.__gradio_mode__ === "app"
        }
      };
      this.app = mount(IndexComponent, opts);
      if (this.updating) {
        this.setAttribute(this.updating.name, this.updating.value);
      }
      this.loading = false;
    }
    static get observedAttributes() {
      return ["src", "space", "host"];
    }
    async attributeChangedCallback(name, old_val, new_val) {
      await pending;
      if ((name === "host" || name === "space" || name === "src") && new_val !== old_val) {
        this.updating = { name, value: new_val };
        if (this.loading) return;
        if (this.app) {
          unmount(this.app);
        }
        this.space = null;
        this.host = null;
        this.src = null;
        if (name === "host") {
          this.host = new_val;
        } else if (name === "space") {
          this.space = new_val;
        } else if (name === "src") {
          this.src = new_val;
        }
        const opts = {
          target: this,
          props: {
            // embed source
            space: this.space ? this.space.trim() : this.space,
            src: this.src ? this.src.trim() : this.src,
            // embed info
            info: this.info === "false" ? false : true,
            container: this.container === "false" ? false : true,
            is_embed: this.is_embed === "false" ? false : true,
            initial_height: this.initial_height,
            eager: this.eager === "true" ? true : false,
            // gradio meta info
            version: "6-11-0",
            theme_mode: this.theme_mode,
            // misc global behaviour
            autoscroll: this.autoscroll === "true" ? true : false,
            control_page_title: this.control_page_title === "true" ? true : false,
            // injectables
            Client,
            // for gradio docs
            // TODO: Remove -- i think this is just for autoscroll behavhiour, app vs embeds
            app_mode: window.__gradio_mode__ === "app"
          }
        };
        this.app = mount(IndexComponent, opts);
        this.updating = false;
      }
    }
  }
  if (!customElements.get("gradio-app"))
    customElements.define("gradio-app", GradioApp);
}
create_custom_element();

export { is_array as $, effect as A, render_effect as B, COMMENT_NODE as C, DEV as D, EFFECT_TRANSPARENT as E, FILENAME as F, deep_read_state as G, HYDRATION_ERROR as H, safe_not_equal as I, listen_to_event_and_reset_event as J, bind_invalid_checkbox_value as K, current_batch as L, tick as M, teardown as N, queue_micro_task as O, previous_batch as P, is as Q, push as R, first_child as S, pop as T, flushSync as U, child as V, from_html as W, sibling as X, reset as Y, event as Z, replay_events as _, append as a, TRANSITION_OUT as a$, set_text as a0, legacy_pre_effect as a1, legacy_pre_effect_reset as a2, mutable_source as a3, createEventDispatcher as a4, user_derived as a5, comment as a6, text as a7, next as a8, is_runes as a9, is_promise as aA, internal_set as aB, Batch as aC, unset_context as aD, is_flushing_sync as aE, source as aF, UNINITIALIZED as aG, capture as aH, HYDRATION_START_ELSE as aI, skip_nodes as aJ, mount as aK, unmount as aL, get_descriptor as aM, update as aN, get_boundary as aO, flatten as aP, HEAD_EFFECT as aQ, run as aR, async_derived as aS, run_after_blockers as aT, deferred_template_effect as aU, $document as aV, should_intro as aW, BLOCK_EFFECT as aX, EFFECT_RAN as aY, TRANSITION_GLOBAL as aZ, TRANSITION_IN as a_, onMount as aa, listen as ab, without_reactive_context as ac, derived_safe_equal as ad, onDestroy as ae, __vitePreload as af, get_prototype_of as ag, object_prototype as ah, user_pre_effect as ai, delegate as aj, remove_textarea_child as ak, autofocus as al, set_should_intro as am, NAMESPACE_SVG as an, create_text as ao, ELEMENT_NODE as ap, is_raw_text_element as aq, set_hydrating as ar, mutate as as, to_array as at, prepare_files as au, FileData as av, $window as aw, setContext as ax, getContext as ay, afterUpdate as az, active_effect as b, is_function as b0, noop as b1, transition_slide_display as b2, invalidate_inner_signals as b3, resume_effect as b4, destroy_effect as b5, pause_effect as b6, branch as b7, move_effect as b8, should_defer_append as b9, run$1 as bA, derived as bB, define_property as bC, props_invalid_value as bD, PROPS_IS_UPDATED as bE, DESTROYED as bF, PROPS_IS_BINDABLE as bG, PROPS_IS_IMMUTABLE as bH, PROPS_IS_LAZY_INITIAL as bI, is_destroying_effect as bJ, LEGACY_PROPS as bK, set_active_effect as bL, props_rest_readonly as bM, legacy_mode_flag as bN, PROPS_IS_RUNES as bO, enable_legacy_mode_flag as bP, enable_async_mode_flag as bQ, read_hydration_instruction as ba, EACH_IS_CONTROLLED as bb, array_from as bc, EACH_ITEM_REACTIVE as bd, EACH_ITEM_IMMUTABLE as be, EACH_INDEX_REACTIVE as bf, EFFECT_OFFSCREEN as bg, INERT as bh, EACH_IS_ANIMATED as bi, clear_text_content as bj, HYDRATION_END as bk, managed as bl, select_multiple_invalid_value as bm, LOADING_ATTR_SYMBOL as bn, ATTACHMENT_KEY as bo, NAMESPACE_HTML as bp, is_capture_event as bq, create_event as br, normalize_attribute as bs, can_delegate_event as bt, get_descriptors as bu, add_form_reset_listener as bv, hydration_attribute_changed as bw, STATE_SYMBOL as bx, component_context as by, run_all as bz, create_fragment_from_html as c, assign_nodes as d, hydrate_next as e, from_svg as f, get_first_child as g, hydrating as h, hydrate_node as i, get_next_sibling as j, hydration_mismatch as k, hash as l, dev_current_component_function as m, hydration_html_changed as n, sanitize_location as o, block as p, invalid_snippet as q, remove_effect_dom as r, set_hydrate_node as s, template_effect as t, state as u, proxy as v, get as w, set as x, user_effect as y, untrack as z };
//# sourceMappingURL=index-BvBk1Iap.js.map
