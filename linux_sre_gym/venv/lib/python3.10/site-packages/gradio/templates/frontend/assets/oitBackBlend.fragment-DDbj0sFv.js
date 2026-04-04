import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "oitBackBlendPixelShader";
const shader = `precision highp float;uniform sampler2D uBackColor;void main() {glFragColor=texelFetch(uBackColor,ivec2(gl_FragCoord.xy),0);if (glFragColor.a==0.0) { 
discard;}}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const oitBackBlendPixelShader = { name, shader };

export { oitBackBlendPixelShader };
//# sourceMappingURL=oitBackBlend.fragment-DDbj0sFv.js.map
