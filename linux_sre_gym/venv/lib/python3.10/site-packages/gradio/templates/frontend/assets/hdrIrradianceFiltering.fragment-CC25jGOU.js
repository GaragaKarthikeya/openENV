import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-qLR1vvLh.js';
import './hdrFilteringFunctions-DQUZ_3Oj.js';
import './pbrBRDFFunctions-Ezjx-oio.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "hdrIrradianceFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
var inputTextureSampler: sampler;var inputTexture: texture_cube<f32>;
#ifdef IBL_CDF_FILTERING
var icdfTextureSampler: sampler;var icdfTexture: texture_2d<f32>;
#endif
uniform vFilteringInfo: vec2f;uniform hdrScale: f32;varying direction: vec3f;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {var color: vec3f=irradiance(inputTexture,inputTextureSampler,input.direction,uniforms.vFilteringInfo,0.0,vec3f(1.0),input.direction
#ifdef IBL_CDF_FILTERING
,icdfTexture,icdfTextureSampler
#endif
);fragmentOutputs.color= vec4f(color*uniforms.hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const hdrIrradianceFilteringPixelShaderWGSL = { name, shader };

export { hdrIrradianceFilteringPixelShaderWGSL };
//# sourceMappingURL=hdrIrradianceFiltering.fragment-CC25jGOU.js.map
