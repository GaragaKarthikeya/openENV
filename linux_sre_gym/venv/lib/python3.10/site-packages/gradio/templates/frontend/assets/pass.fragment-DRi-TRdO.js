import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "passPixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=texture2D(textureSampler,vUV);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const passPixelShader = { name, shader };

export { passPixelShader };
//# sourceMappingURL=pass.fragment-DRi-TRdO.js.map
