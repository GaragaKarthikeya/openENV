import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './instancesDeclaration-Bv0KzShp.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name = "iblVoxelGridVertexShader";
const shader = `attribute vec3 position;varying vec3 vNormalizedPosition;
#include<bonesDeclaration>
#include<bakedVertexAnimationDeclaration>
#include<instancesDeclaration>
#include<morphTargetsVertexGlobalDeclaration>
#include<morphTargetsVertexDeclaration>[0..maxSimultaneousMorphTargets]
uniform mat4 invWorldScale;uniform mat4 viewMatrix;void main(void) {vec3 positionUpdated=position;
#include<morphTargetsVertexGlobal>
#include<morphTargetsVertex>[0..maxSimultaneousMorphTargets]
#include<instancesVertex>
#include<bonesVertex>
#include<bakedVertexAnimation>
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);gl_Position=viewMatrix*invWorldScale*worldPos;vNormalizedPosition.xyz=gl_Position.xyz*0.5+0.5;
#ifdef IS_NDC_HALF_ZRANGE
gl_Position.z=gl_Position.z*0.5+0.5;
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const iblVoxelGridVertexShader = { name, shader };

export { iblVoxelGridVertexShader };
//# sourceMappingURL=iblVoxelGrid.vertex-ChKJvWv-.js.map
