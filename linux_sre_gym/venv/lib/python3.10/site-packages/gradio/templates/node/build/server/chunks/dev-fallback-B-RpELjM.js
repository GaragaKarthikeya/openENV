const BROWSER = typeof window !== "undefined";
const node_env = globalThis.process?.env?.NODE_ENV;
const DEV = node_env && !node_env.toLowerCase().startsWith("prod");

export { BROWSER as B, DEV as D };
//# sourceMappingURL=dev-fallback-B-RpELjM.js.map
