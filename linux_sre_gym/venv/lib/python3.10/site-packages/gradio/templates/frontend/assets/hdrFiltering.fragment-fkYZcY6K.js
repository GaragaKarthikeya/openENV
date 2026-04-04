import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-CiFVmLcr.js';
import './hdrFilteringFunctions-Cf_DM9A0.js';
import './pbrBRDFFunctions-Cr0QY02Q.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "hdrFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform float alphaG;uniform samplerCube inputTexture;uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=radiance(alphaG,inputTexture,direction,vFilteringInfo);gl_FragColor=vec4(color*hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const hdrFilteringPixelShader = { name, shader };

export { hdrFilteringPixelShader };
//# sourceMappingURL=hdrFiltering.fragment-fkYZcY6K.js.map
