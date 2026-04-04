import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-CiFVmLcr.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "rgbdDecodePixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=vec4(fromRGBD(texture2D(textureSampler,vUV)),1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const rgbdDecodePixelShader = { name, shader };

export { rgbdDecodePixelShader };
//# sourceMappingURL=rgbdDecode.fragment-PHs4X25h.js.map
