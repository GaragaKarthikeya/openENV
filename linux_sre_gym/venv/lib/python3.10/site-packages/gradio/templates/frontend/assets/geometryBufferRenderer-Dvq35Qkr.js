const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./geometry.vertex-BLGjZ3ej.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./bonesDeclaration-BSA0gH1L.js","./instancesVertex-o60Do33e.js","./morphTargetsVertexDeclaration-BLOFnJa5.js","./instancesDeclaration-BUZFwkeM.js","./sceneUboDeclaration-BjEv3Rnv.js","./clipPlaneVertex-BeUdHTek.js","./morphTargetsVertex-DIZBNgIv.js","./bonesVertex-b_UrNN_W.js","./bakedVertexAnimation-CnIdycvp.js","./bumpVertex-Db-ravHy.js","./geometry.fragment-D18IfF9Y.js","./clipPlaneFragment-BegocNAa.js","./bumpFragment-CrHG21RV.js","./samplerFragmentDeclaration-DMju9ewr.js","./helperFunctions-qLR1vvLh.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { S as ShaderStore, b9 as Color4, J as VertexBuffer, a7 as MaterialFlags, b3 as PrepareDefinesAndAttributesForMorphTargets, b2 as PushAttributesForInstances, b4 as PrepareStringDefinesForClipPlanes, bz as MultiRenderTarget, T as Texture, M as Matrix, b5 as BindSceneUniformBuffer, a2 as BindClipPlane, a5 as BindMorphTargetParameters, o as Material, Y as AddClipPlaneUniforms, be as _WarnImport } from './index-CxR0Yf_7.js';
import './clipPlaneFragment-Vm6wRK67.js';
import './bumpFragment-Dxop61vv.js';
import './helperFunctions-CiFVmLcr.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './instancesDeclaration-Bv0KzShp.js';
import './sceneUboDeclaration-CbezEhV0.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './bumpVertex-fS00LARh.js';

// Do not edit.
const name$4 = "mrtFragmentDeclaration";
const shader$4 = `#if defined(WEBGL2) || defined(WEBGPU) || defined(NATIVE)
layout(location=0) out vec4 glFragData[{X}];
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$4]) {
    ShaderStore.IncludesShadersStore[name$4] = shader$4;
}

// Do not edit.
const name$3 = "geometryPixelShader";
const shader$3 = `#extension GL_EXT_draw_buffers : require
#if defined(BUMP) || !defined(NORMAL)
#extension GL_OES_standard_derivatives : enable
#endif
precision highp float;
#ifdef BUMP
varying mat4 vWorldView;varying vec3 vNormalW;
#else
varying vec3 vNormalV;
#endif
varying vec4 vViewPos;
#if defined(POSITION) || defined(BUMP)
varying vec3 vPositionW;
#endif
#if defined(VELOCITY) || defined(VELOCITY_LINEAR)
varying vec4 vCurrentPosition;varying vec4 vPreviousPosition;
#endif
#ifdef NEED_UV
varying vec2 vUV;
#endif
#ifdef BUMP
uniform vec3 vBumpInfos;uniform vec2 vTangentSpaceParams;
#endif
#if defined(REFLECTIVITY)
#if defined(ORMTEXTURE) || defined(SPECULARGLOSSINESSTEXTURE) || defined(REFLECTIVITYTEXTURE)
uniform sampler2D reflectivitySampler;varying vec2 vReflectivityUV;
#else
#ifdef METALLIC_TEXTURE
uniform sampler2D metallicSampler;varying vec2 vMetallicUV;
#endif
#ifdef ROUGHNESS_TEXTURE
uniform sampler2D roughnessSampler;varying vec2 vRoughnessUV;
#endif
#endif
#ifdef ALBEDOTEXTURE
varying vec2 vAlbedoUV;uniform sampler2D albedoSampler;
#endif
#ifdef REFLECTIVITYCOLOR
uniform vec3 reflectivityColor;
#endif
#ifdef ALBEDOCOLOR
uniform vec3 albedoColor;
#endif
#ifdef METALLIC
uniform float metallic;
#endif
#if defined(ROUGHNESS) || defined(GLOSSINESS)
uniform float glossiness;
#endif
#endif
#if defined(ALPHATEST) && defined(NEED_UV)
uniform sampler2D diffuseSampler;
#endif
#include<clipPlaneFragmentDeclaration>
#include<mrtFragmentDeclaration>[SCENE_MRT_COUNT]
#include<bumpFragmentMainFunctions>
#include<bumpFragmentFunctions>
#include<helperFunctions>
void main() {
#include<clipPlaneFragment>
#ifdef ALPHATEST
if (texture2D(diffuseSampler,vUV).a<0.4)
discard;
#endif
vec3 normalOutput;
#ifdef BUMP
vec3 normalW=normalize(vNormalW);
#include<bumpFragment>
#ifdef NORMAL_WORLDSPACE
normalOutput=normalW;
#else
normalOutput=normalize(vec3(vWorldView*vec4(normalW,0.0)));
#endif
#elif defined(HAS_NORMAL_ATTRIBUTE)
normalOutput=normalize(vNormalV);
#elif defined(POSITION)
normalOutput=normalize(-cross(dFdx(vPositionW),dFdy(vPositionW)));
#endif
#ifdef ENCODE_NORMAL
normalOutput=normalOutput*0.5+0.5;
#endif
#ifdef DEPTH
gl_FragData[DEPTH_INDEX]=vec4(vViewPos.z/vViewPos.w,0.0,0.0,1.0);
#endif
#ifdef NORMAL
gl_FragData[NORMAL_INDEX]=vec4(normalOutput,1.0);
#endif
#ifdef SCREENSPACE_DEPTH
gl_FragData[SCREENSPACE_DEPTH_INDEX]=vec4(gl_FragCoord.z,0.0,0.0,1.0);
#endif
#ifdef POSITION
gl_FragData[POSITION_INDEX]=vec4(vPositionW,1.0);
#endif
#ifdef VELOCITY
vec2 a=(vCurrentPosition.xy/vCurrentPosition.w)*0.5+0.5;vec2 b=(vPreviousPosition.xy/vPreviousPosition.w)*0.5+0.5;vec2 velocity=abs(a-b);velocity=vec2(pow(velocity.x,1.0/3.0),pow(velocity.y,1.0/3.0))*sign(a-b)*0.5+0.5;gl_FragData[VELOCITY_INDEX]=vec4(velocity,0.0,1.0);
#endif
#ifdef VELOCITY_LINEAR
vec2 velocity=vec2(0.5)*((vPreviousPosition.xy/vPreviousPosition.w) -
(vCurrentPosition.xy/vCurrentPosition.w));gl_FragData[VELOCITY_LINEAR_INDEX]=vec4(velocity,0.0,1.0);
#endif
#ifdef REFLECTIVITY
vec4 reflectivity=vec4(0.0,0.0,0.0,1.0);
#ifdef METALLICWORKFLOW
float metal=1.0;float roughness=1.0;
#ifdef ORMTEXTURE
metal*=texture2D(reflectivitySampler,vReflectivityUV).b;roughness*=texture2D(reflectivitySampler,vReflectivityUV).g;
#else
#ifdef METALLIC_TEXTURE
metal*=texture2D(metallicSampler,vMetallicUV).r;
#endif
#ifdef ROUGHNESS_TEXTURE
roughness*=texture2D(roughnessSampler,vRoughnessUV).r;
#endif
#endif
#ifdef METALLIC
metal*=metallic;
#endif
#ifdef ROUGHNESS
roughness*=(1.0-glossiness); 
#endif
reflectivity.a-=roughness;vec3 color=vec3(1.0);
#ifdef ALBEDOTEXTURE
color=texture2D(albedoSampler,vAlbedoUV).rgb;
#ifdef GAMMAALBEDO
color=toLinearSpace(color);
#endif
#endif
#ifdef ALBEDOCOLOR
color*=albedoColor.xyz;
#endif
reflectivity.rgb=mix(vec3(0.04),color,metal);
#else
#if defined(SPECULARGLOSSINESSTEXTURE) || defined(REFLECTIVITYTEXTURE)
reflectivity=texture2D(reflectivitySampler,vReflectivityUV);
#ifdef GAMMAREFLECTIVITYTEXTURE
reflectivity.rgb=toLinearSpace(reflectivity.rgb);
#endif
#else 
#ifdef REFLECTIVITYCOLOR
reflectivity.rgb=toLinearSpace(reflectivityColor.xyz);reflectivity.a=1.0;
#endif
#endif
#ifdef GLOSSINESSS
reflectivity.a*=glossiness; 
#endif
#endif
gl_FragData[REFLECTIVITY_INDEX]=reflectivity;
#endif
}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$3]) {
    ShaderStore.ShadersStore[name$3] = shader$3;
}
/** @internal */
const geometryPixelShader = { name: name$3, shader: shader$3 };

const geometry_fragment = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    geometryPixelShader
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$2 = "geometryVertexDeclaration";
const shader$2 = `uniform mat4 viewProjection;uniform mat4 view;`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$2]) {
    ShaderStore.IncludesShadersStore[name$2] = shader$2;
}

// Do not edit.
const name$1 = "geometryUboDeclaration";
const shader$1 = `#include<sceneUboDeclaration>
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$1]) {
    ShaderStore.IncludesShadersStore[name$1] = shader$1;
}

// Do not edit.
const name = "geometryVertexShader";
const shader = `precision highp float;
#include<bonesDeclaration>
#include<bakedVertexAnimationDeclaration>
#include<morphTargetsVertexGlobalDeclaration>
#include<morphTargetsVertexDeclaration>[0..maxSimultaneousMorphTargets]
#include<instancesDeclaration>
#include<__decl__geometryVertex>
#include<clipPlaneVertexDeclaration>
attribute vec3 position;
#ifdef HAS_NORMAL_ATTRIBUTE
attribute vec3 normal;
#endif
#ifdef NEED_UV
varying vec2 vUV;
#ifdef ALPHATEST
uniform mat4 diffuseMatrix;
#endif
#ifdef BUMP
uniform mat4 bumpMatrix;varying vec2 vBumpUV;
#endif
#ifdef REFLECTIVITY
uniform mat4 reflectivityMatrix;uniform mat4 albedoMatrix;varying vec2 vReflectivityUV;varying vec2 vAlbedoUV;
#endif
#ifdef METALLIC_TEXTURE
varying vec2 vMetallicUV;uniform mat4 metallicMatrix;
#endif
#ifdef ROUGHNESS_TEXTURE
varying vec2 vRoughnessUV;uniform mat4 roughnessMatrix;
#endif
#ifdef UV1
attribute vec2 uv;
#endif
#ifdef UV2
attribute vec2 uv2;
#endif
#endif
#ifdef BUMP
varying mat4 vWorldView;
#endif
#ifdef BUMP
varying vec3 vNormalW;
#else
varying vec3 vNormalV;
#endif
varying vec4 vViewPos;
#if defined(POSITION) || defined(BUMP)
varying vec3 vPositionW;
#endif
#if defined(VELOCITY) || defined(VELOCITY_LINEAR)
uniform mat4 previousViewProjection;varying vec4 vCurrentPosition;varying vec4 vPreviousPosition;
#endif
#define CUSTOM_VERTEX_DEFINITIONS
void main(void)
{vec3 positionUpdated=position;
#ifdef HAS_NORMAL_ATTRIBUTE
vec3 normalUpdated=normal;
#else
vec3 normalUpdated=vec3(0.0,0.0,0.0);
#endif
#ifdef UV1
vec2 uvUpdated=uv;
#endif
#ifdef UV2
vec2 uv2Updated=uv2;
#endif
#include<morphTargetsVertexGlobal>
#include<morphTargetsVertex>[0..maxSimultaneousMorphTargets]
#include<instancesVertex>
#if (defined(VELOCITY) || defined(VELOCITY_LINEAR)) && !defined(BONES_VELOCITY_ENABLED)
vCurrentPosition=viewProjection*finalWorld*vec4(positionUpdated,1.0);vPreviousPosition=previousViewProjection*finalPreviousWorld*vec4(positionUpdated,1.0);
#endif
#include<bonesVertex>
#include<bakedVertexAnimation>
vec4 worldPos=vec4(finalWorld*vec4(positionUpdated,1.0));
#ifdef BUMP
vWorldView=view*finalWorld;mat3 normalWorld=mat3(finalWorld);vNormalW=normalize(normalWorld*normalUpdated);
#else
#ifdef NORMAL_WORLDSPACE
vNormalV=normalize(vec3(finalWorld*vec4(normalUpdated,0.0)));
#else
vNormalV=normalize(vec3((view*finalWorld)*vec4(normalUpdated,0.0)));
#endif
#endif
vViewPos=view*worldPos;
#if (defined(VELOCITY) || defined(VELOCITY_LINEAR)) && defined(BONES_VELOCITY_ENABLED)
vCurrentPosition=viewProjection*finalWorld*vec4(positionUpdated,1.0);
#if NUM_BONE_INFLUENCERS>0
mat4 previousInfluence;previousInfluence=mPreviousBones[int(matricesIndices[0])]*matricesWeights[0];
#if NUM_BONE_INFLUENCERS>1
previousInfluence+=mPreviousBones[int(matricesIndices[1])]*matricesWeights[1];
#endif
#if NUM_BONE_INFLUENCERS>2
previousInfluence+=mPreviousBones[int(matricesIndices[2])]*matricesWeights[2];
#endif
#if NUM_BONE_INFLUENCERS>3
previousInfluence+=mPreviousBones[int(matricesIndices[3])]*matricesWeights[3];
#endif
#if NUM_BONE_INFLUENCERS>4
previousInfluence+=mPreviousBones[int(matricesIndicesExtra[0])]*matricesWeightsExtra[0];
#endif
#if NUM_BONE_INFLUENCERS>5
previousInfluence+=mPreviousBones[int(matricesIndicesExtra[1])]*matricesWeightsExtra[1];
#endif
#if NUM_BONE_INFLUENCERS>6
previousInfluence+=mPreviousBones[int(matricesIndicesExtra[2])]*matricesWeightsExtra[2];
#endif
#if NUM_BONE_INFLUENCERS>7
previousInfluence+=mPreviousBones[int(matricesIndicesExtra[3])]*matricesWeightsExtra[3];
#endif
vPreviousPosition=previousViewProjection*finalPreviousWorld*previousInfluence*vec4(positionUpdated,1.0);
#else
vPreviousPosition=previousViewProjection*finalPreviousWorld*vec4(positionUpdated,1.0);
#endif
#endif
#if defined(POSITION) || defined(BUMP)
vPositionW=worldPos.xyz/worldPos.w;
#endif
gl_Position=viewProjection*finalWorld*vec4(positionUpdated,1.0);
#include<clipPlaneVertex>
#ifdef NEED_UV
#ifdef UV1
#if defined(ALPHATEST) && defined(ALPHATEST_UV1)
vUV=vec2(diffuseMatrix*vec4(uvUpdated,1.0,0.0));
#else
vUV=uvUpdated;
#endif
#ifdef BUMP_UV1
vBumpUV=vec2(bumpMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#ifdef REFLECTIVITY_UV1
vReflectivityUV=vec2(reflectivityMatrix*vec4(uvUpdated,1.0,0.0));
#else
#ifdef METALLIC_UV1
vMetallicUV=vec2(metallicMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#ifdef ROUGHNESS_UV1
vRoughnessUV=vec2(roughnessMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#endif
#ifdef ALBEDO_UV1
vAlbedoUV=vec2(albedoMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#endif
#ifdef UV2
#if defined(ALPHATEST) && defined(ALPHATEST_UV2)
vUV=vec2(diffuseMatrix*vec4(uv2Updated,1.0,0.0));
#else
vUV=uv2Updated;
#endif
#ifdef BUMP_UV2
vBumpUV=vec2(bumpMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#ifdef REFLECTIVITY_UV2
vReflectivityUV=vec2(reflectivityMatrix*vec4(uv2Updated,1.0,0.0));
#else
#ifdef METALLIC_UV2
vMetallicUV=vec2(metallicMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#ifdef ROUGHNESS_UV2
vRoughnessUV=vec2(roughnessMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#endif
#ifdef ALBEDO_UV2
vAlbedoUV=vec2(albedoMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#endif
#endif
#include<bumpVertex>
}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const geometryVertexShader = { name, shader };

const geometry_vertex = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    geometryVertexShader
}, Symbol.toStringTag, { value: 'Module' }));

/** list the uniforms used by the geometry renderer */
const Uniforms = [
    "world",
    "mBones",
    "viewProjection",
    "diffuseMatrix",
    "view",
    "previousWorld",
    "previousViewProjection",
    "mPreviousBones",
    "bumpMatrix",
    "reflectivityMatrix",
    "albedoMatrix",
    "reflectivityColor",
    "albedoColor",
    "metallic",
    "glossiness",
    "vTangentSpaceParams",
    "vBumpInfos",
    "morphTargetInfluences",
    "morphTargetCount",
    "morphTargetTextureInfo",
    "morphTargetTextureIndices",
    "boneTextureWidth",
];
AddClipPlaneUniforms(Uniforms);
/**
 * This renderer is helpful to fill one of the render target with a geometry buffer.
 */
class GeometryBufferRenderer {
    /**
     * Gets a boolean indicating if normals are encoded in the [0,1] range in the render target. If true, you should do `normal = normal_rt * 2.0 - 1.0` to get the right normal
     */
    get normalsAreUnsigned() {
        return this._normalsAreUnsigned;
    }
    /**
     * @internal
     * Sets up internal structures to share outputs with PrePassRenderer
     * This method should only be called by the PrePassRenderer itself
     */
    _linkPrePassRenderer(prePassRenderer) {
        this._linkedWithPrePass = true;
        this._prePassRenderer = prePassRenderer;
        if (this._multiRenderTarget) {
            // prevents clearing of the RT since it's done by prepass
            this._multiRenderTarget.onClearObservable.clear();
            this._multiRenderTarget.onClearObservable.add(() => {
                // pass
            });
        }
    }
    /**
     * @internal
     * Separates internal structures from PrePassRenderer so the geometry buffer can now operate by itself.
     * This method should only be called by the PrePassRenderer itself
     */
    _unlinkPrePassRenderer() {
        this._linkedWithPrePass = false;
        this._createRenderTargets();
    }
    /**
     * @internal
     * Resets the geometry buffer layout
     */
    _resetLayout() {
        this._enableDepth = true;
        this._enableNormal = true;
        this._enablePosition = false;
        this._enableReflectivity = false;
        this._enableVelocity = false;
        this._enableVelocityLinear = false;
        this._enableScreenspaceDepth = false;
        this._attachmentsFromPrePass = [];
    }
    /**
     * @internal
     * Replaces a texture in the geometry buffer renderer
     * Useful when linking textures of the prepass renderer
     */
    _forceTextureType(geometryBufferType, index) {
        if (geometryBufferType === GeometryBufferRenderer.POSITION_TEXTURE_TYPE) {
            this._positionIndex = index;
            this._enablePosition = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.VELOCITY_TEXTURE_TYPE) {
            this._velocityIndex = index;
            this._enableVelocity = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.VELOCITY_LINEAR_TEXTURE_TYPE) {
            this._velocityLinearIndex = index;
            this._enableVelocityLinear = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE) {
            this._reflectivityIndex = index;
            this._enableReflectivity = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.DEPTH_TEXTURE_TYPE) {
            this._depthIndex = index;
            this._enableDepth = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.NORMAL_TEXTURE_TYPE) {
            this._normalIndex = index;
            this._enableNormal = true;
        }
        else if (geometryBufferType === GeometryBufferRenderer.SCREENSPACE_DEPTH_TEXTURE_TYPE) {
            this._screenspaceDepthIndex = index;
            this._enableScreenspaceDepth = true;
        }
    }
    /**
     * @internal
     * Sets texture attachments
     * Useful when linking textures of the prepass renderer
     */
    _setAttachments(attachments) {
        this._attachmentsFromPrePass = attachments;
    }
    /**
     * @internal
     * Replaces the first texture which is hard coded as a depth texture in the geometry buffer
     * Useful when linking textures of the prepass renderer
     */
    _linkInternalTexture(internalTexture) {
        this._multiRenderTarget.setInternalTexture(internalTexture, 0, false);
    }
    /**
     * Gets the render list (meshes to be rendered) used in the G buffer.
     */
    get renderList() {
        return this._multiRenderTarget.renderList;
    }
    /**
     * Set the render list (meshes to be rendered) used in the G buffer.
     */
    set renderList(meshes) {
        this._multiRenderTarget.renderList = meshes;
    }
    /**
     * Gets whether or not G buffer are supported by the running hardware.
     * This requires draw buffer supports
     */
    get isSupported() {
        return this._multiRenderTarget.isSupported;
    }
    /**
     * Returns the index of the given texture type in the G-Buffer textures array
     * @param textureType The texture type constant. For example GeometryBufferRenderer.POSITION_TEXTURE_INDEX
     * @returns the index of the given texture type in the G-Buffer textures array
     */
    getTextureIndex(textureType) {
        switch (textureType) {
            case GeometryBufferRenderer.POSITION_TEXTURE_TYPE:
                return this._positionIndex;
            case GeometryBufferRenderer.VELOCITY_TEXTURE_TYPE:
                return this._velocityIndex;
            case GeometryBufferRenderer.VELOCITY_LINEAR_TEXTURE_TYPE:
                return this._velocityLinearIndex;
            case GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE:
                return this._reflectivityIndex;
            case GeometryBufferRenderer.DEPTH_TEXTURE_TYPE:
                return this._depthIndex;
            case GeometryBufferRenderer.NORMAL_TEXTURE_TYPE:
                return this._normalIndex;
            case GeometryBufferRenderer.SCREENSPACE_DEPTH_TEXTURE_TYPE:
                return this._screenspaceDepthIndex;
            default:
                return -1;
        }
    }
    /**
     * @returns a boolean indicating if object's depths are enabled for the G buffer.
     */
    get enableDepth() {
        return this._enableDepth;
    }
    /**
     * Sets whether or not object's depths are enabled for the G buffer.
     */
    set enableDepth(enable) {
        this._enableDepth = enable;
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * @returns a boolean indicating if object's normals are enabled for the G buffer.
     */
    get enableNormal() {
        return this._enableNormal;
    }
    /**
     * Sets whether or not object's normals are enabled for the G buffer.
     */
    set enableNormal(enable) {
        this._enableNormal = enable;
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * @returns a boolean indicating if objects positions are enabled for the G buffer.
     */
    get enablePosition() {
        return this._enablePosition;
    }
    /**
     * Sets whether or not objects positions are enabled for the G buffer.
     */
    set enablePosition(enable) {
        this._enablePosition = enable;
        // PrePass handles index and texture links
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * @returns a boolean indicating if objects velocities are enabled for the G buffer.
     */
    get enableVelocity() {
        return this._enableVelocity;
    }
    /**
     * Sets whether or not objects velocities are enabled for the G buffer.
     */
    set enableVelocity(enable) {
        this._enableVelocity = enable;
        if (!enable) {
            this._previousTransformationMatrices = {};
        }
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
        this._scene.needsPreviousWorldMatrices = enable;
    }
    /**
     * @returns a boolean indicating if object's linear velocities are enabled for the G buffer.
     */
    get enableVelocityLinear() {
        return this._enableVelocityLinear;
    }
    /**
     * Sets whether or not object's linear velocities are enabled for the G buffer.
     */
    set enableVelocityLinear(enable) {
        this._enableVelocityLinear = enable;
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * Gets a boolean indicating if objects reflectivity are enabled in the G buffer.
     */
    get enableReflectivity() {
        return this._enableReflectivity;
    }
    /**
     * Sets whether or not objects reflectivity are enabled for the G buffer.
     * For Metallic-Roughness workflow with ORM texture, we assume that ORM texture is defined according to the default layout:
     * pbr.useRoughnessFromMetallicTextureAlpha = false;
     * pbr.useRoughnessFromMetallicTextureGreen = true;
     * pbr.useMetallnessFromMetallicTextureBlue = true;
     */
    set enableReflectivity(enable) {
        this._enableReflectivity = enable;
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * Sets whether or not objects screenspace depth are enabled for the G buffer.
     */
    get enableScreenspaceDepth() {
        return this._enableScreenspaceDepth;
    }
    set enableScreenspaceDepth(enable) {
        this._enableScreenspaceDepth = enable;
        if (!this._linkedWithPrePass) {
            this.dispose();
            this._createRenderTargets();
        }
    }
    /**
     * Gets the scene associated with the buffer.
     */
    get scene() {
        return this._scene;
    }
    /**
     * Gets the ratio used by the buffer during its creation.
     * How big is the buffer related to the main canvas.
     */
    get ratio() {
        return typeof this._ratioOrDimensions === "object" ? 1 : this._ratioOrDimensions;
    }
    /**
     * Gets the shader language used in this material.
     */
    get shaderLanguage() {
        return this._shaderLanguage;
    }
    /**
     * Creates a new G Buffer for the scene
     * @param scene The scene the buffer belongs to
     * @param ratioOrDimensions How big is the buffer related to the main canvas (default: 1). You can also directly pass a width and height for the generated textures
     * @param depthFormat Format of the depth texture (default: 15)
     * @param textureTypesAndFormats The types and formats of textures to create as render targets. If not provided, all textures will be RGBA and float or half float, depending on the engine capabilities.
     */
    constructor(scene, ratioOrDimensions = 1, depthFormat = 15, textureTypesAndFormats) {
        /**
         * Dictionary used to store the previous transformation matrices of each rendered mesh
         * in order to compute objects velocities when enableVelocity is set to "true"
         * @internal
         */
        this._previousTransformationMatrices = {};
        /**
         * Dictionary used to store the previous bones transformation matrices of each rendered mesh
         * in order to compute objects velocities when enableVelocity is set to "true"
         * @internal
         */
        this._previousBonesTransformationMatrices = {};
        /**
         * Array used to store the ignored skinned meshes while computing velocity map (typically used by the motion blur post-process).
         * Avoids computing bones velocities and computes only mesh's velocity itself (position, rotation, scaling).
         */
        this.excludedSkinnedMeshesFromVelocity = [];
        /** Gets or sets a boolean indicating if transparent meshes should be rendered */
        this.renderTransparentMeshes = true;
        /**
         * Gets or sets a boolean indicating if normals should be generated in world space (default: false, meaning normals are generated in view space)
         */
        this.generateNormalsInWorldSpace = false;
        this._normalsAreUnsigned = false;
        this._resizeObserver = null;
        this._enableDepth = true;
        this._enableNormal = true;
        this._enablePosition = false;
        this._enableVelocity = false;
        this._enableVelocityLinear = false;
        this._enableReflectivity = false;
        this._enableScreenspaceDepth = false;
        this._clearColor = new Color4(0, 0, 0, 0);
        this._clearDepthColor = new Color4(0, 0, 0, 1); // sets an invalid value by default - depth in the depth texture is view.z, so 0 is not possible because view.z can't be less than camera.minZ
        this._positionIndex = -1;
        this._velocityIndex = -1;
        this._velocityLinearIndex = -1;
        this._reflectivityIndex = -1;
        this._depthIndex = -1;
        this._normalIndex = -1;
        this._screenspaceDepthIndex = -1;
        this._linkedWithPrePass = false;
        /**
         * If set to true (default: false), the depth texture will be cleared with the depth value corresponding to the far plane (1 in normal mode, 0 in reverse depth buffer mode)
         * If set to false, the depth texture is always cleared with 0.
         */
        this.useSpecificClearForDepthTexture = false;
        /** Shader language used by the material */
        this._shaderLanguage = 0 /* ShaderLanguage.GLSL */;
        this._shadersLoaded = false;
        this._scene = scene;
        this._ratioOrDimensions = ratioOrDimensions;
        this._useUbo = scene.getEngine().supportsUniformBuffers;
        this._depthFormat = depthFormat;
        this._textureTypesAndFormats = textureTypesAndFormats || {};
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._initShaderSourceAsync();
        GeometryBufferRenderer._SceneComponentInitialization(this._scene);
        // Render target
        this._createRenderTargets();
    }
    async _initShaderSourceAsync() {
        const engine = this._scene.getEngine();
        if (engine.isWebGPU && !GeometryBufferRenderer.ForceGLSL) {
            this._shaderLanguage = 1 /* ShaderLanguage.WGSL */;
            await Promise.all([__vitePreload(() => import('./geometry.vertex-BLGjZ3ej.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13]):void 0,import.meta.url), __vitePreload(() => import('./geometry.fragment-D18IfF9Y.js'),true              ?__vite__mapDeps([14,1,2,3,15,16,17,18]):void 0,import.meta.url)]);
        }
        else {
            await Promise.all([__vitePreload(() => Promise.resolve().then(() => geometry_vertex),true              ?void 0:void 0,import.meta.url), __vitePreload(() => Promise.resolve().then(() => geometry_fragment),true              ?void 0:void 0,import.meta.url)]);
        }
        this._shadersLoaded = true;
    }
    /**
     * Checks whether everything is ready to render a submesh to the G buffer.
     * @param subMesh the submesh to check readiness for
     * @param useInstances is the mesh drawn using instance or not
     * @returns true if ready otherwise false
     */
    isReady(subMesh, useInstances) {
        if (!this._shadersLoaded) {
            return false;
        }
        const material = subMesh.getMaterial();
        if (material && material.disableDepthWrite) {
            return false;
        }
        const defines = [];
        const attribs = [VertexBuffer.PositionKind];
        const mesh = subMesh.getMesh();
        const hasNormals = mesh.isVerticesDataPresent(VertexBuffer.NormalKind);
        if (hasNormals) {
            defines.push("#define HAS_NORMAL_ATTRIBUTE");
            attribs.push(VertexBuffer.NormalKind);
        }
        let uv1 = false;
        let uv2 = false;
        const color = false;
        if (material) {
            let needUv = false;
            // Alpha test
            if (material.needAlphaTestingForMesh(mesh) && material.getAlphaTestTexture()) {
                defines.push("#define ALPHATEST");
                defines.push(`#define ALPHATEST_UV${material.getAlphaTestTexture().coordinatesIndex + 1}`);
                needUv = true;
            }
            // Normal map texture
            if ((material.bumpTexture || material.normalTexture || material.geometryNormalTexture) && MaterialFlags.BumpTextureEnabled) {
                const texture = material.bumpTexture || material.normalTexture || material.geometryNormalTexture;
                defines.push("#define BUMP");
                defines.push(`#define BUMP_UV${texture.coordinatesIndex + 1}`);
                needUv = true;
            }
            if (this._enableReflectivity) {
                let metallicWorkflow = false;
                // for PBR materials: cf. https://doc.babylonjs.com/features/featuresDeepDive/materials/using/masterPBR
                if (material.getClassName() === "PBRMetallicRoughnessMaterial") {
                    // if it is a PBR material in MetallicRoughness Mode:
                    if (material.metallicRoughnessTexture) {
                        defines.push("#define ORMTEXTURE");
                        defines.push(`#define REFLECTIVITY_UV${material.metallicRoughnessTexture.coordinatesIndex + 1}`);
                        defines.push("#define METALLICWORKFLOW");
                        needUv = true;
                        metallicWorkflow = true;
                    }
                    // null or undefined
                    if (material.metallic != null) {
                        defines.push("#define METALLIC");
                        defines.push("#define METALLICWORKFLOW");
                        metallicWorkflow = true;
                    }
                    // null or undefined
                    if (material.roughness != null) {
                        defines.push("#define ROUGHNESS");
                        defines.push("#define METALLICWORKFLOW");
                        metallicWorkflow = true;
                    }
                    if (metallicWorkflow) {
                        if (material.baseTexture) {
                            defines.push("#define ALBEDOTEXTURE");
                            defines.push(`#define ALBEDO_UV${material.baseTexture.coordinatesIndex + 1}`);
                            if (material.baseTexture.gammaSpace) {
                                defines.push("#define GAMMAALBEDO");
                            }
                            needUv = true;
                        }
                        if (material.baseColor) {
                            defines.push("#define ALBEDOCOLOR");
                        }
                    }
                }
                else if (material.getClassName() === "PBRSpecularGlossinessMaterial") {
                    // if it is a PBR material in Specular/Glossiness Mode:
                    if (material.specularGlossinessTexture) {
                        defines.push("#define SPECULARGLOSSINESSTEXTURE");
                        defines.push(`#define REFLECTIVITY_UV${material.specularGlossinessTexture.coordinatesIndex + 1}`);
                        needUv = true;
                        if (material.specularGlossinessTexture.gammaSpace) {
                            defines.push("#define GAMMAREFLECTIVITYTEXTURE");
                        }
                    }
                    else {
                        if (material.specularColor) {
                            defines.push("#define REFLECTIVITYCOLOR");
                        }
                    }
                    // null or undefined
                    if (material.glossiness != null) {
                        defines.push("#define GLOSSINESS");
                    }
                }
                else if (material.getClassName() === "PBRMaterial") {
                    // if it is the bigger PBRMaterial
                    if (material.metallicTexture) {
                        defines.push("#define ORMTEXTURE");
                        defines.push(`#define REFLECTIVITY_UV${material.metallicTexture.coordinatesIndex + 1}`);
                        defines.push("#define METALLICWORKFLOW");
                        needUv = true;
                        metallicWorkflow = true;
                    }
                    // null or undefined
                    if (material.metallic != null) {
                        defines.push("#define METALLIC");
                        defines.push("#define METALLICWORKFLOW");
                        metallicWorkflow = true;
                    }
                    // null or undefined
                    if (material.roughness != null) {
                        defines.push("#define ROUGHNESS");
                        defines.push("#define METALLICWORKFLOW");
                        metallicWorkflow = true;
                    }
                    if (metallicWorkflow) {
                        if (material.albedoTexture) {
                            defines.push("#define ALBEDOTEXTURE");
                            defines.push(`#define ALBEDO_UV${material.albedoTexture.coordinatesIndex + 1}`);
                            if (material.albedoTexture.gammaSpace) {
                                defines.push("#define GAMMAALBEDO");
                            }
                            needUv = true;
                        }
                        if (material.albedoColor) {
                            defines.push("#define ALBEDOCOLOR");
                        }
                    }
                    else {
                        // SpecularGlossiness Model
                        if (material.reflectivityTexture) {
                            defines.push("#define SPECULARGLOSSINESSTEXTURE");
                            defines.push(`#define REFLECTIVITY_UV${material.reflectivityTexture.coordinatesIndex + 1}`);
                            if (material.reflectivityTexture.gammaSpace) {
                                defines.push("#define GAMMAREFLECTIVITYTEXTURE");
                            }
                            needUv = true;
                        }
                        else if (material.reflectivityColor) {
                            defines.push("#define REFLECTIVITYCOLOR");
                        }
                        // null or undefined
                        if (material.microSurface != null) {
                            defines.push("#define GLOSSINESS");
                        }
                    }
                }
                else if (material.getClassName() === "StandardMaterial") {
                    // if StandardMaterial:
                    if (material.specularTexture) {
                        defines.push("#define REFLECTIVITYTEXTURE");
                        defines.push(`#define REFLECTIVITY_UV${material.specularTexture.coordinatesIndex + 1}`);
                        if (material.specularTexture.gammaSpace) {
                            defines.push("#define GAMMAREFLECTIVITYTEXTURE");
                        }
                        needUv = true;
                    }
                    if (material.specularColor) {
                        defines.push("#define REFLECTIVITYCOLOR");
                    }
                }
                else if (material.getClassName() === "OpenPBRMaterial") {
                    const pbrMaterial = material;
                    defines.push("#define METALLICWORKFLOW");
                    metallicWorkflow = true;
                    defines.push("#define METALLIC");
                    defines.push("#define ROUGHNESS");
                    if (pbrMaterial._useRoughnessFromMetallicTextureGreen && pbrMaterial.baseMetalnessTexture) {
                        defines.push("#define ORMTEXTURE");
                        defines.push(`#define REFLECTIVITY_UV${pbrMaterial.baseMetalnessTexture.coordinatesIndex + 1}`);
                        needUv = true;
                    }
                    else if (pbrMaterial.baseMetalnessTexture) {
                        defines.push("#define METALLIC_TEXTURE");
                        defines.push(`#define METALLIC_UV${pbrMaterial.baseMetalnessTexture.coordinatesIndex + 1}`);
                        needUv = true;
                    }
                    else if (pbrMaterial.specularRoughnessTexture) {
                        defines.push("#define ROUGHNESS_TEXTURE");
                        defines.push(`#define ROUGHNESS_UV${pbrMaterial.specularRoughnessTexture.coordinatesIndex + 1}`);
                        needUv = true;
                    }
                    if (pbrMaterial.baseColorTexture) {
                        defines.push("#define ALBEDOTEXTURE");
                        defines.push(`#define ALBEDO_UV${pbrMaterial.baseColorTexture.coordinatesIndex + 1}`);
                        if (pbrMaterial.baseColorTexture.gammaSpace) {
                            defines.push("#define GAMMAALBEDO");
                        }
                        needUv = true;
                    }
                    if (pbrMaterial.baseColor) {
                        defines.push("#define ALBEDOCOLOR");
                    }
                }
            }
            if (needUv) {
                defines.push("#define NEED_UV");
                if (mesh.isVerticesDataPresent(VertexBuffer.UVKind)) {
                    attribs.push(VertexBuffer.UVKind);
                    defines.push("#define UV1");
                    uv1 = true;
                }
                if (mesh.isVerticesDataPresent(VertexBuffer.UV2Kind)) {
                    attribs.push(VertexBuffer.UV2Kind);
                    defines.push("#define UV2");
                    uv2 = true;
                }
            }
        }
        // Buffers
        if (this._enableDepth) {
            defines.push("#define DEPTH");
            defines.push("#define DEPTH_INDEX " + this._depthIndex);
        }
        if (this._enableNormal) {
            defines.push("#define NORMAL");
            defines.push("#define NORMAL_INDEX " + this._normalIndex);
        }
        if (this._enablePosition) {
            defines.push("#define POSITION");
            defines.push("#define POSITION_INDEX " + this._positionIndex);
        }
        if (this._enableVelocity) {
            defines.push("#define VELOCITY");
            defines.push("#define VELOCITY_INDEX " + this._velocityIndex);
            if (this.excludedSkinnedMeshesFromVelocity.indexOf(mesh) === -1) {
                defines.push("#define BONES_VELOCITY_ENABLED");
            }
        }
        if (this._enableVelocityLinear) {
            defines.push("#define VELOCITY_LINEAR");
            defines.push("#define VELOCITY_LINEAR_INDEX " + this._velocityLinearIndex);
            if (this.excludedSkinnedMeshesFromVelocity.indexOf(mesh) === -1) {
                defines.push("#define BONES_VELOCITY_ENABLED");
            }
        }
        if (this._enableReflectivity) {
            defines.push("#define REFLECTIVITY");
            defines.push("#define REFLECTIVITY_INDEX " + this._reflectivityIndex);
        }
        if (this._enableScreenspaceDepth) {
            if (this._screenspaceDepthIndex !== -1) {
                defines.push("#define SCREENSPACE_DEPTH_INDEX " + this._screenspaceDepthIndex);
                defines.push("#define SCREENSPACE_DEPTH");
            }
        }
        if (this.generateNormalsInWorldSpace) {
            defines.push("#define NORMAL_WORLDSPACE");
        }
        if (this._normalsAreUnsigned) {
            defines.push("#define ENCODE_NORMAL");
        }
        // Bones
        if (mesh.useBones && mesh.computeBonesUsingShaders && mesh.skeleton) {
            attribs.push(VertexBuffer.MatricesIndicesKind);
            attribs.push(VertexBuffer.MatricesWeightsKind);
            if (mesh.numBoneInfluencers > 4) {
                attribs.push(VertexBuffer.MatricesIndicesExtraKind);
                attribs.push(VertexBuffer.MatricesWeightsExtraKind);
            }
            defines.push("#define NUM_BONE_INFLUENCERS " + mesh.numBoneInfluencers);
            defines.push("#define BONETEXTURE " + mesh.skeleton.isUsingTextureForMatrices);
            defines.push("#define BonesPerMesh " + (mesh.skeleton.bones.length + 1));
        }
        else {
            defines.push("#define NUM_BONE_INFLUENCERS 0");
            defines.push("#define BONETEXTURE false");
            defines.push("#define BonesPerMesh 0");
        }
        // Morph targets
        const numMorphInfluencers = mesh.morphTargetManager
            ? PrepareDefinesAndAttributesForMorphTargets(mesh.morphTargetManager, defines, attribs, mesh, true, // usePositionMorph
            true, // useNormalMorph
            false, // useTangentMorph
            uv1, // useUVMorph
            uv2, // useUV2Morph
            color // useColorMorph
            )
            : 0;
        // Instances
        if (useInstances) {
            defines.push("#define INSTANCES");
            PushAttributesForInstances(attribs, this._enableVelocity || this._enableVelocityLinear);
            if (subMesh.getRenderingMesh().hasThinInstances) {
                defines.push("#define THIN_INSTANCES");
            }
        }
        // Setup textures count
        if (this._linkedWithPrePass) {
            defines.push("#define SCENE_MRT_COUNT " + this._attachmentsFromPrePass.length);
        }
        else {
            defines.push("#define SCENE_MRT_COUNT " + this._multiRenderTarget.textures.length);
        }
        PrepareStringDefinesForClipPlanes(material, this._scene, defines);
        // Get correct effect
        const engine = this._scene.getEngine();
        const drawWrapper = subMesh._getDrawWrapper(undefined, true);
        const cachedDefines = drawWrapper.defines;
        const join = defines.join("\n");
        if (cachedDefines !== join) {
            drawWrapper.setEffect(engine.createEffect("geometry", {
                attributes: attribs,
                uniformsNames: Uniforms,
                samplers: ["diffuseSampler", "bumpSampler", "reflectivitySampler", "albedoSampler", "morphTargets", "boneSampler"],
                defines: join,
                onCompiled: null,
                fallbacks: null,
                onError: null,
                uniformBuffersNames: ["Scene"],
                indexParameters: { buffersCount: this._multiRenderTarget.textures.length - 1, maxSimultaneousMorphTargets: numMorphInfluencers },
                shaderLanguage: this.shaderLanguage,
            }, engine), join);
        }
        return drawWrapper.effect.isReady();
    }
    /**
     * Gets the current underlying G Buffer.
     * @returns the buffer
     */
    getGBuffer() {
        return this._multiRenderTarget;
    }
    /**
     * Gets the number of samples used to render the buffer (anti aliasing).
     */
    get samples() {
        return this._multiRenderTarget.samples;
    }
    /**
     * Sets the number of samples used to render the buffer (anti aliasing).
     */
    set samples(value) {
        this._multiRenderTarget.samples = value;
    }
    /**
     * Disposes the renderer and frees up associated resources.
     */
    dispose() {
        if (this._resizeObserver) {
            const engine = this._scene.getEngine();
            engine.onResizeObservable.remove(this._resizeObserver);
            this._resizeObserver = null;
        }
        this.getGBuffer().dispose();
    }
    _assignRenderTargetIndices() {
        const textureNames = [];
        const textureTypesAndFormats = [];
        let count = 0;
        if (this._enableDepth) {
            this._depthIndex = count;
            count++;
            textureNames.push("gBuffer_Depth");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.DEPTH_TEXTURE_TYPE]);
        }
        if (this._enableNormal) {
            this._normalIndex = count;
            count++;
            textureNames.push("gBuffer_Normal");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.NORMAL_TEXTURE_TYPE]);
        }
        if (this._enablePosition) {
            this._positionIndex = count;
            count++;
            textureNames.push("gBuffer_Position");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.POSITION_TEXTURE_TYPE]);
        }
        if (this._enableVelocity) {
            this._velocityIndex = count;
            count++;
            textureNames.push("gBuffer_Velocity");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.VELOCITY_TEXTURE_TYPE]);
        }
        if (this._enableVelocityLinear) {
            this._velocityLinearIndex = count;
            count++;
            textureNames.push("gBuffer_VelocityLinear");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.VELOCITY_LINEAR_TEXTURE_TYPE]);
        }
        if (this._enableReflectivity) {
            this._reflectivityIndex = count;
            count++;
            textureNames.push("gBuffer_Reflectivity");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE]);
        }
        if (this._enableScreenspaceDepth) {
            this._screenspaceDepthIndex = count;
            count++;
            textureNames.push("gBuffer_ScreenspaceDepth");
            textureTypesAndFormats.push(this._textureTypesAndFormats[GeometryBufferRenderer.SCREENSPACE_DEPTH_TEXTURE_TYPE]);
        }
        return [count, textureNames, textureTypesAndFormats];
    }
    _createRenderTargets() {
        const engine = this._scene.getEngine();
        const [count, textureNames, textureTypesAndFormat] = this._assignRenderTargetIndices();
        let type = 0;
        if (engine._caps.textureFloat && engine._caps.textureFloatLinearFiltering) {
            type = 1;
        }
        else if (engine._caps.textureHalfFloat && engine._caps.textureHalfFloatLinearFiltering) {
            type = 2;
        }
        const dimensions = this._ratioOrDimensions.width !== undefined
            ? this._ratioOrDimensions
            : { width: engine.getRenderWidth() * this._ratioOrDimensions, height: engine.getRenderHeight() * this._ratioOrDimensions };
        const textureTypes = [];
        const textureFormats = [];
        for (const typeAndFormat of textureTypesAndFormat) {
            if (typeAndFormat) {
                textureTypes.push(typeAndFormat.textureType);
                textureFormats.push(typeAndFormat.textureFormat);
            }
            else {
                textureTypes.push(type);
                textureFormats.push(5);
            }
        }
        this._normalsAreUnsigned =
            textureTypes[GeometryBufferRenderer.NORMAL_TEXTURE_TYPE] === 11 ||
                textureTypes[GeometryBufferRenderer.NORMAL_TEXTURE_TYPE] === 13;
        this._multiRenderTarget = new MultiRenderTarget("gBuffer", dimensions, count, this._scene, { generateMipMaps: false, generateDepthTexture: true, types: textureTypes, formats: textureFormats, depthTextureFormat: this._depthFormat }, textureNames.concat("gBuffer_DepthBuffer"));
        if (!this.isSupported) {
            return;
        }
        this._multiRenderTarget.wrapU = Texture.CLAMP_ADDRESSMODE;
        this._multiRenderTarget.wrapV = Texture.CLAMP_ADDRESSMODE;
        this._multiRenderTarget.refreshRate = 1;
        this._multiRenderTarget.renderParticles = false;
        this._multiRenderTarget.renderList = null;
        // Depth is always the first texture in the geometry buffer renderer!
        const layoutAttachmentsAll = [true];
        const layoutAttachmentsAllButDepth = [false];
        const layoutAttachmentsDepthOnly = [true];
        for (let i = 1; i < count; ++i) {
            layoutAttachmentsAll.push(true);
            layoutAttachmentsDepthOnly.push(false);
            layoutAttachmentsAllButDepth.push(true);
        }
        const attachmentsAll = engine.buildTextureLayout(layoutAttachmentsAll);
        const attachmentsAllButDepth = engine.buildTextureLayout(layoutAttachmentsAllButDepth);
        const attachmentsDepthOnly = engine.buildTextureLayout(layoutAttachmentsDepthOnly);
        this._multiRenderTarget.onClearObservable.add((engine) => {
            engine.bindAttachments(this.useSpecificClearForDepthTexture ? attachmentsAllButDepth : attachmentsAll);
            engine.clear(this._clearColor, true, true, true);
            if (this.useSpecificClearForDepthTexture) {
                engine.bindAttachments(attachmentsDepthOnly);
                engine.clear(this._clearDepthColor, true, true, true);
            }
            engine.bindAttachments(attachmentsAll);
        });
        this._resizeObserver = engine.onResizeObservable.add(() => {
            if (this._multiRenderTarget) {
                const dimensions = this._ratioOrDimensions.width !== undefined
                    ? this._ratioOrDimensions
                    : { width: engine.getRenderWidth() * this._ratioOrDimensions, height: engine.getRenderHeight() * this._ratioOrDimensions };
                this._multiRenderTarget.resize(dimensions);
            }
        });
        // Custom render function
        const renderSubMesh = (subMesh) => {
            const renderingMesh = subMesh.getRenderingMesh();
            const effectiveMesh = subMesh.getEffectiveMesh();
            const scene = this._scene;
            const engine = scene.getEngine();
            const material = subMesh.getMaterial();
            if (!material) {
                return;
            }
            effectiveMesh._internalAbstractMeshDataInfo._isActiveIntermediate = false;
            // Velocity
            if ((this._enableVelocity || this._enableVelocityLinear) && !this._previousTransformationMatrices[effectiveMesh.uniqueId]) {
                this._previousTransformationMatrices[effectiveMesh.uniqueId] = {
                    world: Matrix.Identity(),
                    viewProjection: scene.getTransformMatrix(),
                };
                if (renderingMesh.skeleton) {
                    const bonesTransformations = renderingMesh.skeleton.getTransformMatrices(renderingMesh);
                    this._previousBonesTransformationMatrices[renderingMesh.uniqueId] = this._copyBonesTransformationMatrices(bonesTransformations, new Float32Array(bonesTransformations.length));
                }
            }
            // Managing instances
            const batch = renderingMesh._getInstancesRenderList(subMesh._id, !!subMesh.getReplacementMesh());
            if (batch.mustReturn) {
                return;
            }
            const hardwareInstancedRendering = engine.getCaps().instancedArrays && (batch.visibleInstances[subMesh._id] !== null || renderingMesh.hasThinInstances);
            const world = effectiveMesh.getWorldMatrix();
            if (this.isReady(subMesh, hardwareInstancedRendering)) {
                const drawWrapper = subMesh._getDrawWrapper();
                if (!drawWrapper) {
                    return;
                }
                const effect = drawWrapper.effect;
                engine.enableEffect(drawWrapper);
                if (!hardwareInstancedRendering) {
                    renderingMesh._bind(subMesh, effect, material.fillMode);
                }
                if (!this._useUbo) {
                    effect.setMatrix("viewProjection", scene.getTransformMatrix());
                    effect.setMatrix("view", scene.getViewMatrix());
                }
                else {
                    BindSceneUniformBuffer(effect, this._scene.getSceneUniformBuffer());
                    this._scene.finalizeSceneUbo();
                }
                let sideOrientation;
                const instanceDataStorage = renderingMesh._instanceDataStorage;
                if (!instanceDataStorage.isFrozen && (material.backFaceCulling || material.sideOrientation !== null)) {
                    const mainDeterminant = effectiveMesh._getWorldMatrixDeterminant();
                    sideOrientation = material._getEffectiveOrientation(renderingMesh);
                    if (mainDeterminant < 0) {
                        sideOrientation = sideOrientation === Material.ClockWiseSideOrientation ? Material.CounterClockWiseSideOrientation : Material.ClockWiseSideOrientation;
                    }
                }
                else {
                    sideOrientation = renderingMesh._effectiveSideOrientation;
                }
                material._preBind(drawWrapper, sideOrientation);
                // Alpha test
                if (material.needAlphaTestingForMesh(effectiveMesh)) {
                    const alphaTexture = material.getAlphaTestTexture();
                    if (alphaTexture) {
                        effect.setTexture("diffuseSampler", alphaTexture);
                        effect.setMatrix("diffuseMatrix", alphaTexture.getTextureMatrix());
                    }
                }
                // Bump
                if ((material.bumpTexture || material.normalTexture || material.geometryNormalTexture) &&
                    scene.getEngine().getCaps().standardDerivatives &&
                    MaterialFlags.BumpTextureEnabled) {
                    const texture = material.bumpTexture || material.normalTexture || material.geometryNormalTexture;
                    effect.setFloat3("vBumpInfos", texture.coordinatesIndex, 1.0 / texture.level, material.parallaxScaleBias);
                    effect.setMatrix("bumpMatrix", texture.getTextureMatrix());
                    effect.setTexture("bumpSampler", texture);
                    effect.setFloat2("vTangentSpaceParams", material.invertNormalMapX ? -1 : 1.0, material.invertNormalMapY ? -1 : 1.0);
                }
                // Reflectivity
                if (this._enableReflectivity) {
                    // for PBR materials: cf. https://doc.babylonjs.com/features/featuresDeepDive/materials/using/masterPBR
                    if (material.getClassName() === "PBRMetallicRoughnessMaterial") {
                        // if it is a PBR material in MetallicRoughness Mode:
                        if (material.metallicRoughnessTexture !== null) {
                            effect.setTexture("reflectivitySampler", material.metallicRoughnessTexture);
                            effect.setMatrix("reflectivityMatrix", material.metallicRoughnessTexture.getTextureMatrix());
                        }
                        if (material.metallic !== null) {
                            effect.setFloat("metallic", material.metallic);
                        }
                        if (material.roughness !== null) {
                            effect.setFloat("glossiness", 1.0 - material.roughness);
                        }
                        if (material.baseTexture !== null) {
                            effect.setTexture("albedoSampler", material.baseTexture);
                            effect.setMatrix("albedoMatrix", material.baseTexture.getTextureMatrix());
                        }
                        if (material.baseColor !== null) {
                            effect.setColor3("albedoColor", material.baseColor);
                        }
                    }
                    else if (material.getClassName() === "PBRSpecularGlossinessMaterial") {
                        // if it is a PBR material in Specular/Glossiness Mode:
                        if (material.specularGlossinessTexture !== null) {
                            effect.setTexture("reflectivitySampler", material.specularGlossinessTexture);
                            effect.setMatrix("reflectivityMatrix", material.specularGlossinessTexture.getTextureMatrix());
                        }
                        else {
                            if (material.specularColor !== null) {
                                effect.setColor3("reflectivityColor", material.specularColor);
                            }
                        }
                        if (material.glossiness !== null) {
                            effect.setFloat("glossiness", material.glossiness);
                        }
                    }
                    else if (material.getClassName() === "PBRMaterial") {
                        // if it is the bigger PBRMaterial
                        if (material.metallicTexture !== null) {
                            effect.setTexture("reflectivitySampler", material.metallicTexture);
                            effect.setMatrix("reflectivityMatrix", material.metallicTexture.getTextureMatrix());
                        }
                        if (material.metallic !== null) {
                            effect.setFloat("metallic", material.metallic);
                        }
                        if (material.roughness !== null) {
                            effect.setFloat("glossiness", 1.0 - material.roughness);
                        }
                        if (material.roughness !== null || material.metallic !== null || material.metallicTexture !== null) {
                            // MetallicRoughness Model
                            if (material.albedoTexture !== null) {
                                effect.setTexture("albedoSampler", material.albedoTexture);
                                effect.setMatrix("albedoMatrix", material.albedoTexture.getTextureMatrix());
                            }
                            if (material.albedoColor !== null) {
                                effect.setColor3("albedoColor", material.albedoColor);
                            }
                        }
                        else {
                            // SpecularGlossiness Model
                            if (material.reflectivityTexture !== null) {
                                effect.setTexture("reflectivitySampler", material.reflectivityTexture);
                                effect.setMatrix("reflectivityMatrix", material.reflectivityTexture.getTextureMatrix());
                            }
                            else if (material.reflectivityColor !== null) {
                                effect.setColor3("reflectivityColor", material.reflectivityColor);
                            }
                            if (material.microSurface !== null) {
                                effect.setFloat("glossiness", material.microSurface);
                            }
                        }
                    }
                    else if (material.getClassName() === "StandardMaterial") {
                        // if StandardMaterial:
                        if (material.specularTexture !== null) {
                            effect.setTexture("reflectivitySampler", material.specularTexture);
                            effect.setMatrix("reflectivityMatrix", material.specularTexture.getTextureMatrix());
                        }
                        if (material.specularColor !== null) {
                            effect.setColor3("reflectivityColor", material.specularColor);
                        }
                    }
                    else if (material.getClassName() === "OpenPBRMaterial") {
                        // if it is a OpenPBR material:
                        const openpbrMaterial = material;
                        if (openpbrMaterial._useRoughnessFromMetallicTextureGreen && openpbrMaterial.baseMetalnessTexture) {
                            effect.setTexture("reflectivitySampler", openpbrMaterial.baseMetalnessTexture);
                            effect.setMatrix("reflectivityMatrix", openpbrMaterial.baseMetalnessTexture.getTextureMatrix());
                        }
                        else if (openpbrMaterial.baseMetalnessTexture) {
                            effect.setTexture("metallicSampler", openpbrMaterial.baseMetalnessTexture);
                            effect.setMatrix("metallicMatrix", openpbrMaterial.baseMetalnessTexture.getTextureMatrix());
                        }
                        else if (openpbrMaterial.specularRoughnessTexture) {
                            effect.setTexture("roughnessSampler", openpbrMaterial.specularRoughnessTexture);
                            effect.setMatrix("roughnessMatrix", openpbrMaterial.specularRoughnessTexture.getTextureMatrix());
                        }
                        effect.setFloat("metallic", openpbrMaterial.baseMetalness);
                        effect.setFloat("glossiness", 1.0 - openpbrMaterial.specularRoughness);
                        if (openpbrMaterial.baseColorTexture !== null) {
                            effect.setTexture("albedoSampler", openpbrMaterial.baseColorTexture);
                            effect.setMatrix("albedoMatrix", openpbrMaterial.baseColorTexture.getTextureMatrix());
                        }
                        if (openpbrMaterial.baseColor !== null) {
                            effect.setColor3("albedoColor", openpbrMaterial.baseColor);
                        }
                    }
                }
                // Clip plane
                BindClipPlane(effect, material, this._scene);
                // Bones
                if (renderingMesh.useBones && renderingMesh.computeBonesUsingShaders && renderingMesh.skeleton) {
                    const skeleton = renderingMesh.skeleton;
                    if (skeleton.isUsingTextureForMatrices && effect.getUniformIndex("boneTextureWidth") > -1) {
                        const boneTexture = skeleton.getTransformMatrixTexture(renderingMesh);
                        effect.setTexture("boneSampler", boneTexture);
                        effect.setFloat("boneTextureWidth", 4.0 * (skeleton.bones.length + 1));
                    }
                    else {
                        effect.setMatrices("mBones", renderingMesh.skeleton.getTransformMatrices(renderingMesh));
                    }
                    if (this._enableVelocity || this._enableVelocityLinear) {
                        effect.setMatrices("mPreviousBones", this._previousBonesTransformationMatrices[renderingMesh.uniqueId]);
                    }
                }
                // Morph targets
                BindMorphTargetParameters(renderingMesh, effect);
                if (renderingMesh.morphTargetManager && renderingMesh.morphTargetManager.isUsingTextureForTargets) {
                    renderingMesh.morphTargetManager._bind(effect);
                }
                // Velocity
                if (this._enableVelocity || this._enableVelocityLinear) {
                    effect.setMatrix("previousWorld", this._previousTransformationMatrices[effectiveMesh.uniqueId].world);
                    effect.setMatrix("previousViewProjection", this._previousTransformationMatrices[effectiveMesh.uniqueId].viewProjection);
                }
                if (hardwareInstancedRendering && renderingMesh.hasThinInstances) {
                    effect.setMatrix("world", world);
                }
                // Draw
                renderingMesh._processRendering(effectiveMesh, subMesh, effect, material.fillMode, batch, hardwareInstancedRendering, (isInstance, w) => {
                    if (!isInstance) {
                        effect.setMatrix("world", w);
                    }
                });
            }
            // Velocity
            if (this._enableVelocity || this._enableVelocityLinear) {
                this._previousTransformationMatrices[effectiveMesh.uniqueId].world = world.clone();
                this._previousTransformationMatrices[effectiveMesh.uniqueId].viewProjection = this._scene.getTransformMatrix().clone();
                if (renderingMesh.skeleton) {
                    this._copyBonesTransformationMatrices(renderingMesh.skeleton.getTransformMatrices(renderingMesh), this._previousBonesTransformationMatrices[effectiveMesh.uniqueId]);
                }
            }
        };
        this._multiRenderTarget.customIsReadyFunction = (mesh, refreshRate, preWarm) => {
            if ((preWarm || refreshRate === 0) && mesh.subMeshes) {
                for (let i = 0; i < mesh.subMeshes.length; ++i) {
                    const subMesh = mesh.subMeshes[i];
                    const material = subMesh.getMaterial();
                    const renderingMesh = subMesh.getRenderingMesh();
                    if (!material) {
                        continue;
                    }
                    const batch = renderingMesh._getInstancesRenderList(subMesh._id, !!subMesh.getReplacementMesh());
                    const hardwareInstancedRendering = engine.getCaps().instancedArrays && (batch.visibleInstances[subMesh._id] !== null || renderingMesh.hasThinInstances);
                    if (!this.isReady(subMesh, hardwareInstancedRendering)) {
                        return false;
                    }
                }
            }
            return true;
        };
        this._multiRenderTarget.customRenderFunction = (opaqueSubMeshes, alphaTestSubMeshes, transparentSubMeshes, depthOnlySubMeshes) => {
            let index;
            if (this._linkedWithPrePass) {
                if (!this._prePassRenderer.enabled) {
                    return;
                }
                this._scene.getEngine().bindAttachments(this._attachmentsFromPrePass);
            }
            if (depthOnlySubMeshes.length) {
                engine.setColorWrite(false);
                for (index = 0; index < depthOnlySubMeshes.length; index++) {
                    renderSubMesh(depthOnlySubMeshes.data[index]);
                }
                engine.setColorWrite(true);
            }
            for (index = 0; index < opaqueSubMeshes.length; index++) {
                renderSubMesh(opaqueSubMeshes.data[index]);
            }
            engine.setDepthWrite(false);
            for (index = 0; index < alphaTestSubMeshes.length; index++) {
                renderSubMesh(alphaTestSubMeshes.data[index]);
            }
            if (this.renderTransparentMeshes) {
                for (index = 0; index < transparentSubMeshes.length; index++) {
                    renderSubMesh(transparentSubMeshes.data[index]);
                }
            }
            engine.setDepthWrite(true);
        };
    }
    // Copies the bones transformation matrices into the target array and returns the target's reference
    _copyBonesTransformationMatrices(source, target) {
        for (let i = 0; i < source.length; i++) {
            target[i] = source[i];
        }
        return target;
    }
}
/**
 * Force all the standard materials to compile to glsl even on WebGPU engines.
 * False by default. This is mostly meant for backward compatibility.
 */
GeometryBufferRenderer.ForceGLSL = false;
/**
 * Constant used to retrieve the depth texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.DEPTH_TEXTURE_INDEX)
 */
GeometryBufferRenderer.DEPTH_TEXTURE_TYPE = 0;
/**
 * Constant used to retrieve the normal texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.NORMAL_TEXTURE_INDEX)
 */
GeometryBufferRenderer.NORMAL_TEXTURE_TYPE = 1;
/**
 * Constant used to retrieve the position texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.POSITION_TEXTURE_INDEX)
 */
GeometryBufferRenderer.POSITION_TEXTURE_TYPE = 2;
/**
 * Constant used to retrieve the velocity texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.VELOCITY_TEXTURE_INDEX)
 */
GeometryBufferRenderer.VELOCITY_TEXTURE_TYPE = 3;
/**
 * Constant used to retrieve the reflectivity texture index in the G-Buffer textures array
 * using the getIndex(GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE)
 */
GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE = 4;
/**
 * Constant used to retrieve the screen-space depth texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.SCREENSPACE_DEPTH_TEXTURE_TYPE)
 */
GeometryBufferRenderer.SCREENSPACE_DEPTH_TEXTURE_TYPE = 5;
/**
 * Constant used to retrieve the linear velocity texture index in the G-Buffer textures array
 * using getIndex(GeometryBufferRenderer.VELOCITY_LINEAR_TEXTURE_TYPE)
 */
GeometryBufferRenderer.VELOCITY_LINEAR_TEXTURE_TYPE = 6;
/**
 * @internal
 */
GeometryBufferRenderer._SceneComponentInitialization = (_) => {
    throw _WarnImport("GeometryBufferRendererSceneComponent");
};

export { GeometryBufferRenderer as G };
//# sourceMappingURL=geometryBufferRenderer-Dvq35Qkr.js.map
