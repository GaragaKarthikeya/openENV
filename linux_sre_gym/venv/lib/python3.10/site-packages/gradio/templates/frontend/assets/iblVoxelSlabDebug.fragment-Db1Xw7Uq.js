import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "iblVoxelSlabDebugPixelShader";
const shader = `precision highp float;varying vec3 vNormalizedPosition;uniform float nearPlane;uniform float farPlane;uniform float stepSize;void main(void) {vec3 normPos=vNormalizedPosition.xyz;float chunkSize=stepSize*float(MAX_DRAW_BUFFERS);float numChunks=1.0/chunkSize;float positionInChunk=fract(normPos.z/chunkSize);float slab=floor(positionInChunk*float(MAX_DRAW_BUFFERS)) /
float(MAX_DRAW_BUFFERS);if (normPos.x<0.0 || normPos.y<0.0 || normPos.z<0.0 ||
normPos.x>1.0 || normPos.y>1.0 || normPos.z>1.0) {gl_FragColor=vec4(0.0,0.0,0.0,0.0);} else {gl_FragColor=vec4(slab,0.0,0.0,0.75);}}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const iblVoxelSlabDebugPixelShader = { name, shader };

export { iblVoxelSlabDebugPixelShader };
//# sourceMappingURL=iblVoxelSlabDebug.fragment-Db1Xw7Uq.js.map
