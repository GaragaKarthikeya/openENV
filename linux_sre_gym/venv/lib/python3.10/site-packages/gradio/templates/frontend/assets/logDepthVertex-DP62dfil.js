import { S as ShaderStore } from './index-CxR0Yf_7.js';

// Do not edit.
const name$2 = "fogVertexDeclaration";
const shader$2 = `#ifdef FOG
varying vec3 vFogDistance;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$2]) {
    ShaderStore.IncludesShadersStore[name$2] = shader$2;
}

// Do not edit.
const name$1 = "fogVertex";
const shader$1 = `#ifdef FOG
vFogDistance=(view*worldPos).xyz;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$1]) {
    ShaderStore.IncludesShadersStore[name$1] = shader$1;
}

// Do not edit.
const name = "logDepthVertex";
const shader = `#ifdef LOGARITHMICDEPTH
vFragmentDepth=1.0+gl_Position.w;gl_Position.z=log2(max(0.000001,vFragmentDepth))*logarithmicDepthConstant;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
//# sourceMappingURL=logDepthVertex-DP62dfil.js.map
