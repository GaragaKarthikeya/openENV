import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './kernelBlurVaryingDeclaration-CvwVDGXi.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name$1 = "kernelBlurVertex";
const shader$1 = `vertexOutputs.sampleCoord{X}=vertexOutputs.sampleCenter+uniforms.delta*KERNEL_OFFSET{X};`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name$1]) {
    ShaderStore.IncludesShadersStoreWGSL[name$1] = shader$1;
}

// Do not edit.
const name = "kernelBlurVertexShader";
const shader = `attribute position: vec2f;uniform delta: vec2f;varying sampleCenter: vec2f;
#include<kernelBlurVaryingDeclaration>[0..varyingCount]
#define CUSTOM_VERTEX_DEFINITIONS
@vertex
fn main(input : VertexInputs)->FragmentInputs {const madd: vec2f= vec2f(0.5,0.5);
#define CUSTOM_VERTEX_MAIN_BEGIN
vertexOutputs.sampleCenter=(input.position*madd+madd);
#include<kernelBlurVertex>[0..varyingCount]
vertexOutputs.position= vec4f(input.position,0.0,1.0);
#define CUSTOM_VERTEX_MAIN_END
}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const kernelBlurVertexShaderWGSL = { name, shader };

export { kernelBlurVertexShaderWGSL };
//# sourceMappingURL=kernelBlur.vertex-iIm1FHNR.js.map
