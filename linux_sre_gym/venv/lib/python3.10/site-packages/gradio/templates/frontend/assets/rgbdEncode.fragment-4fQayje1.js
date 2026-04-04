import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './helperFunctions-qLR1vvLh.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "rgbdEncodePixelShader";
const shader = `varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=toRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV).rgb);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const rgbdEncodePixelShaderWGSL = { name, shader };

export { rgbdEncodePixelShaderWGSL };
//# sourceMappingURL=rgbdEncode.fragment-4fQayje1.js.map
