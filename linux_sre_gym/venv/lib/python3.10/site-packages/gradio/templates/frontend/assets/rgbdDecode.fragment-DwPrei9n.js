import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-qLR1vvLh.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "rgbdDecodePixelShader";
const shader = `varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=vec4f(fromRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV)),1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const rgbdDecodePixelShaderWGSL = { name, shader };

export { rgbdDecodePixelShaderWGSL };
//# sourceMappingURL=rgbdDecode.fragment-DwPrei9n.js.map
