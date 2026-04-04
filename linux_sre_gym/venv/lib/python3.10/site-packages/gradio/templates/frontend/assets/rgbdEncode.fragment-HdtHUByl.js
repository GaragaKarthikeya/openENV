import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-CiFVmLcr.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "rgbdEncodePixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=toRGBD(texture2D(textureSampler,vUV).rgb);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const rgbdEncodePixelShader = { name, shader };

export { rgbdEncodePixelShader };
//# sourceMappingURL=rgbdEncode.fragment-HdtHUByl.js.map
