import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "oitFinalSimpleBlendPixelShader";
const shader = `precision highp float;uniform sampler2D uFrontColor;void main() {ivec2 fragCoord=ivec2(gl_FragCoord.xy);vec4 frontColor=texelFetch(uFrontColor,fragCoord,0);glFragColor=frontColor;}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const oitFinalSimpleBlendPixelShader = { name, shader };

export { oitFinalSimpleBlendPixelShader };
//# sourceMappingURL=oitFinalSimpleBlend.fragment-D2axY_SA.js.map
