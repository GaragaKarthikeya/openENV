import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './imageProcessingFunctions-CBu7O15Q.js';
import './helperFunctions-CiFVmLcr.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "imageProcessingPixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<imageProcessingDeclaration>
#include<helperFunctions>
#include<imageProcessingFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void)
{vec4 result=texture2D(textureSampler,vUV);result.rgb=max(result.rgb,vec3(0.));
#ifdef IMAGEPROCESSING
#ifndef FROMLINEARSPACE
result.rgb=toLinearSpace(result.rgb);
#endif
result=applyImageProcessing(result);
#else
#ifdef FROMLINEARSPACE
result=applyImageProcessing(result);
#endif
#endif
gl_FragColor=result;}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const imageProcessingPixelShader = { name, shader };

export { imageProcessingPixelShader };
//# sourceMappingURL=imageProcessing.fragment-DCkv0aPu.js.map
