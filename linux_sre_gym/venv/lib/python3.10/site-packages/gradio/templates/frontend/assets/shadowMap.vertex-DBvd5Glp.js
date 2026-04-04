import { S as ShaderStore } from './index-CxR0Yf_7.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './helperFunctions-CiFVmLcr.js';
import './sceneUboDeclaration-CbezEhV0.js';
import './meshUboDeclaration-zsmLsV73.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './index-BvBk1Iap.js';

// Do not edit.
const name$7 = "sceneVertexDeclaration";
const shader$7 = `uniform mat4 viewProjection;
#ifdef MULTIVIEW
uniform mat4 viewProjectionR;
#endif
uniform mat4 view;uniform mat4 projection;uniform vec4 vEyePosition;
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$7]) {
    ShaderStore.IncludesShadersStore[name$7] = shader$7;
}

// Do not edit.
const name$6 = "meshVertexDeclaration";
const shader$6 = `uniform mat4 world;uniform float visibility;
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$6]) {
    ShaderStore.IncludesShadersStore[name$6] = shader$6;
}

// Do not edit.
const name$5 = "shadowMapVertexDeclaration";
const shader$5 = `#include<sceneVertexDeclaration>
#include<meshVertexDeclaration>
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$5]) {
    ShaderStore.IncludesShadersStore[name$5] = shader$5;
}

// Do not edit.
const name$4 = "shadowMapUboDeclaration";
const shader$4 = `layout(std140,column_major) uniform;
#include<sceneUboDeclaration>
#include<meshUboDeclaration>
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$4]) {
    ShaderStore.IncludesShadersStore[name$4] = shader$4;
}

// Do not edit.
const name$3 = "shadowMapVertexExtraDeclaration";
const shader$3 = `#if SM_NORMALBIAS==1
uniform vec3 lightDataSM;
#endif
uniform vec3 biasAndScaleSM;uniform vec2 depthValuesSM;varying float vDepthMetricSM;
#if SM_USEDISTANCE==1
varying vec3 vPositionWSM;
#endif
#if defined(SM_DEPTHCLAMP) && SM_DEPTHCLAMP==1
varying float zSM;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$3]) {
    ShaderStore.IncludesShadersStore[name$3] = shader$3;
}

// Do not edit.
const name$2 = "shadowMapVertexNormalBias";
const shader$2 = `#if SM_NORMALBIAS==1
#if SM_DIRECTIONINLIGHTDATA==1
vec3 worldLightDirSM=normalize(-lightDataSM.xyz);
#else
vec3 directionToLightSM=lightDataSM.xyz-worldPos.xyz;vec3 worldLightDirSM=normalize(directionToLightSM);
#endif
float ndlSM=dot(vNormalW,worldLightDirSM);float sinNLSM=sqrt(1.0-ndlSM*ndlSM);float normalBiasSM=biasAndScaleSM.y*sinNLSM;worldPos.xyz-=vNormalW*normalBiasSM;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$2]) {
    ShaderStore.IncludesShadersStore[name$2] = shader$2;
}

// Do not edit.
const name$1 = "shadowMapVertexMetric";
const shader$1 = `#if SM_USEDISTANCE==1
vPositionWSM=worldPos.xyz;
#endif
#if SM_DEPTHTEXTURE==1
#ifdef IS_NDC_HALF_ZRANGE
#define BIASFACTOR 0.5
#else
#define BIASFACTOR 1.0
#endif
#ifdef USE_REVERSE_DEPTHBUFFER
gl_Position.z-=biasAndScaleSM.x*gl_Position.w*BIASFACTOR;
#else
gl_Position.z+=biasAndScaleSM.x*gl_Position.w*BIASFACTOR;
#endif
#endif
#if defined(SM_DEPTHCLAMP) && SM_DEPTHCLAMP==1
zSM=gl_Position.z;gl_Position.z=0.0;
#elif SM_USEDISTANCE==0
#ifdef USE_REVERSE_DEPTHBUFFER
vDepthMetricSM=(-gl_Position.z+depthValuesSM.x)/depthValuesSM.y+biasAndScaleSM.x;
#else
vDepthMetricSM=(gl_Position.z+depthValuesSM.x)/depthValuesSM.y+biasAndScaleSM.x;
#endif
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$1]) {
    ShaderStore.IncludesShadersStore[name$1] = shader$1;
}

// Do not edit.
const name = "shadowMapVertexShader";
const shader = `attribute vec3 position;
#ifdef NORMAL
attribute vec3 normal;
#endif
#include<bonesDeclaration>
#include<bakedVertexAnimationDeclaration>
#include<morphTargetsVertexGlobalDeclaration>
#include<morphTargetsVertexDeclaration>[0..maxSimultaneousMorphTargets]
#ifdef INSTANCES
attribute vec4 world0;attribute vec4 world1;attribute vec4 world2;attribute vec4 world3;
#endif
#include<helperFunctions>
#include<__decl__shadowMapVertex>
#ifdef ALPHATEXTURE
varying vec2 vUV;uniform mat4 diffuseMatrix;
#ifdef UV1
attribute vec2 uv;
#endif
#ifdef UV2
attribute vec2 uv2;
#endif
#endif
#include<shadowMapVertexExtraDeclaration>
#include<clipPlaneVertexDeclaration>
#define CUSTOM_VERTEX_DEFINITIONS
void main(void)
{vec3 positionUpdated=position;
#ifdef UV1
vec2 uvUpdated=uv;
#endif
#ifdef UV2
vec2 uv2Updated=uv2;
#endif
#ifdef NORMAL
vec3 normalUpdated=normal;
#endif
#include<morphTargetsVertexGlobal>
#include<morphTargetsVertex>[0..maxSimultaneousMorphTargets]
#include<instancesVertex>
#include<bonesVertex>
#include<bakedVertexAnimation>
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);
#ifdef NORMAL
mat3 normWorldSM=mat3(finalWorld);
#if defined(INSTANCES) && defined(THIN_INSTANCES)
vec3 vNormalW=normalUpdated/vec3(dot(normWorldSM[0],normWorldSM[0]),dot(normWorldSM[1],normWorldSM[1]),dot(normWorldSM[2],normWorldSM[2]));vNormalW=normalize(normWorldSM*vNormalW);
#else
#ifdef NONUNIFORMSCALING
normWorldSM=transposeMat3(inverseMat3(normWorldSM));
#endif
vec3 vNormalW=normalize(normWorldSM*normalUpdated);
#endif
#endif
#include<shadowMapVertexNormalBias>
gl_Position=viewProjection*worldPos;
#include<shadowMapVertexMetric>
#ifdef ALPHATEXTURE
#ifdef UV1
vUV=vec2(diffuseMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#ifdef UV2
vUV=vec2(diffuseMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#endif
#include<clipPlaneVertex>
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const shadowMapVertexShader = { name, shader };

export { shadowMapVertexShader };
//# sourceMappingURL=shadowMap.vertex-DBvd5Glp.js.map
