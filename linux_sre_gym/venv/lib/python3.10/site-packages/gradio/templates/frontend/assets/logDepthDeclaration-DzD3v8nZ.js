import { S as ShaderStore } from './index-CxR0Yf_7.js';

// Do not edit.
const name = "logDepthDeclaration";
const shader = `#ifdef LOGARITHMICDEPTH
uniform float logarithmicDepthConstant;varying float vFragmentDepth;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
//# sourceMappingURL=logDepthDeclaration-DzD3v8nZ.js.map
