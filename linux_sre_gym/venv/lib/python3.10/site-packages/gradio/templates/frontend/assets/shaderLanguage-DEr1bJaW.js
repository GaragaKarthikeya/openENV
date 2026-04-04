/**
 * Language of the shader code
 */
var ShaderLanguage;
(function (ShaderLanguage) {
    /** language is GLSL (used by WebGL) */
    ShaderLanguage[ShaderLanguage["GLSL"] = 0] = "GLSL";
    /** language is WGSL (used by WebGPU) */
    ShaderLanguage[ShaderLanguage["WGSL"] = 1] = "WGSL";
})(ShaderLanguage || (ShaderLanguage = {}));

export { ShaderLanguage };
//# sourceMappingURL=shaderLanguage-DEr1bJaW.js.map
