import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-qLR1vvLh.js';
import './hdrFilteringFunctions-DQUZ_3Oj.js';
import './pbrBRDFFunctions-Ezjx-oio.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "hdrFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform alphaG: f32;var inputTextureSampler: sampler;var inputTexture: texture_cube<f32>;uniform vFilteringInfo: vec2f;uniform hdrScale: f32;varying direction: vec3f;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {var color: vec3f=radiance(uniforms.alphaG,inputTexture,inputTextureSampler,input.direction,uniforms.vFilteringInfo);fragmentOutputs.color= vec4f(color*uniforms.hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const hdrFilteringPixelShaderWGSL = { name, shader };

export { hdrFilteringPixelShaderWGSL };
//# sourceMappingURL=hdrFiltering.fragment-C8V8uM7Y.js.map
