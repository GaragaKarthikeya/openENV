const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.BRx_RcKs.js",app:"_app/immutable/entry/app.4RAdwWh-.js",imports:["_app/immutable/entry/start.BRx_RcKs.js","_app/immutable/chunks/1t7wABRY.js","_app/immutable/chunks/DcfTJGrh.js","_app/immutable/chunks/43y08qZT.js","_app/immutable/entry/app.4RAdwWh-.js","_app/immutable/chunks/Db2VRVPO.js","_app/immutable/chunks/DcfTJGrh.js","_app/immutable/chunks/43y08qZT.js","_app/immutable/chunks/C903eyvM.js","_app/immutable/chunks/BEYfkQLb.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./chunks/0-Ckn-QRGb.js')),
			__memo(() => import('./chunks/1-40J_SSlg.js')),
			__memo(() => import('./chunks/2-BAeILJ1g.js').then(function (n) { return n._; }))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/([^]*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
