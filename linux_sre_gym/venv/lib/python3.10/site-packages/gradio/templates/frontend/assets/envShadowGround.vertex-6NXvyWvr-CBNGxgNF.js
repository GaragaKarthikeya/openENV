import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "envShadowGroundVertexShader";
const shader = `precision highp float;attribute vec3 position;attribute vec2 uv;uniform mat4 worldViewProjection;varying vec2 vUV;void main(void) {gl_Position=worldViewProjection*vec4(position,1.0);vUV=uv;}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const envShadowGroundVertexShader = { name, shader };

export { envShadowGroundVertexShader };
//# sourceMappingURL=envShadowGround.vertex-6NXvyWvr-CBNGxgNF.js.map
