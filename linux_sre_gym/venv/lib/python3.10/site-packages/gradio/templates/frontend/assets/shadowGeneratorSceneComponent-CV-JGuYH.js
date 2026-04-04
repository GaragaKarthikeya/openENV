const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./depth.vertex-k8oMaAk6.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./bonesDeclaration-BSA0gH1L.js","./instancesVertex-o60Do33e.js","./morphTargetsVertexDeclaration-BLOFnJa5.js","./clipPlaneVertex-BeUdHTek.js","./instancesDeclaration-BUZFwkeM.js","./morphTargetsVertex-DIZBNgIv.js","./bonesVertex-b_UrNN_W.js","./bakedVertexAnimation-CnIdycvp.js","./depth.fragment-PGqN_SYP.js","./clipPlaneFragment-BegocNAa.js","./packingFunctions-BnPI3m-A.js"])))=>i.map(i=>d[i]);
import { ShadowGenerator } from './shadowGenerator-Bn9ALoaN.js';
import { S as ShaderStore, T as Texture, b9 as Color4, aL as RenderTargetTexture, J as VertexBuffer, E as EffectFallbacks, b3 as PrepareDefinesAndAttributesForMorphTargets, b2 as PushAttributesForInstances, b4 as PrepareStringDefinesForClipPlanes, Y as AddClipPlaneUniforms, b1 as Camera, $ as BindBonesParameters, a2 as BindClipPlane, a5 as BindMorphTargetParameters, be as _WarnImport, aE as EffectWrapper, bn as Engine, O as Observable, bE as PostProcessManager, aA as PostProcess, V as Vector3, M as Matrix, L as Logger, aq as EngineStore, av as BoundingInfo, bd as FloatingOriginCurrentScene, bF as GetOffsetTransformMatrices, a as SceneComponentConstants } from './index-CxR0Yf_7.js';
import { af as __vitePreload } from './index-BvBk1Iap.js';
import './clipPlaneFragment-Vm6wRK67.js';
import './packingFunctions-DHPBRaLz.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './instancesDeclaration-Bv0KzShp.js';
import './pointCloudVertex-Bst85-5N.js';

// Do not edit.
const name$4 = "depthPixelShader";
const shader$4 = `#ifdef ALPHATEST
varying vec2 vUV;uniform sampler2D diffuseSampler;
#endif
#include<clipPlaneFragmentDeclaration>
varying float vDepthMetric;
#ifdef PACKED
#include<packingFunctions>
#endif
#ifdef STORE_CAMERASPACE_Z
varying vec4 vViewPos;
#endif
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void)
{
#include<clipPlaneFragment>
#ifdef ALPHATEST
if (texture2D(diffuseSampler,vUV).a<0.4)
discard;
#endif
#ifdef STORE_CAMERASPACE_Z
#ifdef PACKED
gl_FragColor=pack(vViewPos.z);
#else
gl_FragColor=vec4(vViewPos.z,0.0,0.0,1.0);
#endif
#else
#ifdef NONLINEARDEPTH
#ifdef PACKED
gl_FragColor=pack(gl_FragCoord.z);
#else
gl_FragColor=vec4(gl_FragCoord.z,0.0,0.0,0.0);
#endif
#else
#ifdef PACKED
gl_FragColor=pack(vDepthMetric);
#else
gl_FragColor=vec4(vDepthMetric,0.0,0.0,1.0);
#endif
#endif
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$4]) {
    ShaderStore.ShadersStore[name$4] = shader$4;
}
/** @internal */
const depthPixelShader = { name: name$4, shader: shader$4 };

const depth_fragment = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    depthPixelShader
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$3 = "pointCloudVertexDeclaration";
const shader$3 = `#ifdef POINTSIZE
uniform float pointSize;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$3]) {
    ShaderStore.IncludesShadersStore[name$3] = shader$3;
}

// Do not edit.
const name$2 = "depthVertexShader";
const shader$2 = `attribute vec3 position;
#include<bonesDeclaration>
#include<bakedVertexAnimationDeclaration>
#include<morphTargetsVertexGlobalDeclaration>
#include<morphTargetsVertexDeclaration>[0..maxSimultaneousMorphTargets]
#include<clipPlaneVertexDeclaration>
#include<instancesDeclaration>
uniform mat4 viewProjection;uniform vec2 depthValues;
#if defined(ALPHATEST) || defined(NEED_UV)
varying vec2 vUV;uniform mat4 diffuseMatrix;
#ifdef UV1
attribute vec2 uv;
#endif
#ifdef UV2
attribute vec2 uv2;
#endif
#endif
#ifdef STORE_CAMERASPACE_Z
uniform mat4 view;varying vec4 vViewPos;
#endif
#include<pointCloudVertexDeclaration>
varying float vDepthMetric;
#define CUSTOM_VERTEX_DEFINITIONS
void main(void)
{vec3 positionUpdated=position;
#ifdef UV1
vec2 uvUpdated=uv;
#endif
#ifdef UV2
vec2 uv2Updated=uv2;
#endif
#include<morphTargetsVertexGlobal>
#include<morphTargetsVertex>[0..maxSimultaneousMorphTargets]
#include<instancesVertex>
#include<bonesVertex>
#include<bakedVertexAnimation>
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);
#include<clipPlaneVertex>
gl_Position=viewProjection*worldPos;
#ifdef STORE_CAMERASPACE_Z
vViewPos=view*worldPos;
#else
#ifdef USE_REVERSE_DEPTHBUFFER
vDepthMetric=((-gl_Position.z+depthValues.x)/(depthValues.y));
#else
vDepthMetric=((gl_Position.z+depthValues.x)/(depthValues.y));
#endif
#endif
#if defined(ALPHATEST) || defined(BASIC_RENDER)
#ifdef UV1
vUV=vec2(diffuseMatrix*vec4(uvUpdated,1.0,0.0));
#endif
#ifdef UV2
vUV=vec2(diffuseMatrix*vec4(uv2Updated,1.0,0.0));
#endif
#endif
#include<pointCloudVertex>
}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$2]) {
    ShaderStore.ShadersStore[name$2] = shader$2;
}
/** @internal */
const depthVertexShader = { name: name$2, shader: shader$2 };

const depth_vertex = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    depthVertexShader
}, Symbol.toStringTag, { value: 'Module' }));

/**
 * This represents a depth renderer in Babylon.
 * A depth renderer will render to it's depth map every frame which can be displayed or used in post processing
 */
class DepthRenderer {
    /**
     * Gets the shader language used in this material.
     */
    get shaderLanguage() {
        return this._shaderLanguage;
    }
    /**
     * Sets a specific material to be used to render a mesh/a list of meshes by the depth renderer
     * @param mesh mesh or array of meshes
     * @param material material to use by the depth render when rendering the mesh(es). If undefined is passed, the specific material created by the depth renderer will be used.
     */
    setMaterialForRendering(mesh, material) {
        this._depthMap.setMaterialForRendering(mesh, material);
    }
    /**
     * Instantiates a depth renderer
     * @param scene The scene the renderer belongs to
     * @param type The texture type of the depth map (default: Engine.TEXTURETYPE_FLOAT)
     * @param camera The camera to be used to render the depth map (default: scene's active camera)
     * @param storeNonLinearDepth Defines whether the depth is stored linearly like in Babylon Shadows or directly like glFragCoord.z
     * @param samplingMode The sampling mode to be used with the render target (Linear, Nearest...) (default: TRILINEAR_SAMPLINGMODE)
     * @param storeCameraSpaceZ Defines whether the depth stored is the Z coordinate in camera space. If true, storeNonLinearDepth has no effect. (Default: false)
     * @param name Name of the render target (default: DepthRenderer)
     * @param existingRenderTargetTexture An existing render target texture to use (default: undefined). If not provided, a new render target texture will be created.
     */
    constructor(scene, type = 1, camera = null, storeNonLinearDepth = false, samplingMode = Texture.TRILINEAR_SAMPLINGMODE, storeCameraSpaceZ = false, name, existingRenderTargetTexture) {
        /** Shader language used by the material */
        this._shaderLanguage = 0 /* ShaderLanguage.GLSL */;
        /** Enable or disable the depth renderer. When disabled, the depth texture is not updated */
        this.enabled = true;
        /** Force writing the transparent objects into the depth map */
        this.forceDepthWriteTransparentMeshes = false;
        /**
         * Specifies that the depth renderer will only be used within
         * the camera it is created for.
         * This can help forcing its rendering during the camera processing.
         */
        this.useOnlyInActiveCamera = false;
        /** If true, reverse the culling of materials before writing to the depth texture.
         * So, basically, when "true", back facing instead of front facing faces are rasterized into the texture
         */
        this.reverseCulling = false;
        this._shadersLoaded = false;
        this._scene = scene;
        this._storeNonLinearDepth = storeNonLinearDepth;
        this._storeCameraSpaceZ = storeCameraSpaceZ;
        this.isPacked = type === 0;
        if (this.isPacked) {
            this.clearColor = new Color4(1.0, 1.0, 1.0, 1.0);
        }
        else {
            this.clearColor = new Color4(storeCameraSpaceZ ? 0.0 : 1.0, 0.0, 0.0, 1.0);
        }
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._initShaderSourceAsync();
        DepthRenderer._SceneComponentInitialization(this._scene);
        const engine = scene.getEngine();
        this._camera = camera;
        if (samplingMode !== Texture.NEAREST_SAMPLINGMODE) {
            if (type === 1 && !engine._caps.textureFloatLinearFiltering) {
                samplingMode = Texture.NEAREST_SAMPLINGMODE;
            }
            if (type === 2 && !engine._caps.textureHalfFloatLinearFiltering) {
                samplingMode = Texture.NEAREST_SAMPLINGMODE;
            }
        }
        // Render target
        const format = this.isPacked || !engine._features.supportExtendedTextureFormats ? 5 : 6;
        this._depthMap =
            existingRenderTargetTexture ??
                new RenderTargetTexture(name ?? "DepthRenderer", { width: engine.getRenderWidth(), height: engine.getRenderHeight() }, this._scene, false, true, type, false, samplingMode, undefined, undefined, undefined, format);
        this._depthMap.wrapU = Texture.CLAMP_ADDRESSMODE;
        this._depthMap.wrapV = Texture.CLAMP_ADDRESSMODE;
        this._depthMap.refreshRate = 1;
        this._depthMap.renderParticles = false;
        this._depthMap.renderList = null;
        this._depthMap.noPrePassRenderer = true;
        // Camera to get depth map from to support multiple concurrent cameras
        this._depthMap.activeCamera = this._camera;
        this._depthMap.ignoreCameraViewport = true;
        this._depthMap.useCameraPostProcesses = false;
        // set default depth value to 1.0 (far away)
        this._depthMap.onClearObservable.add((engine) => {
            engine.clear(this.clearColor, true, true, true);
        });
        this._depthMap.onBeforeBindObservable.add(() => {
            if (engine._enableGPUDebugMarkers) {
                engine.restoreDefaultFramebuffer();
                engine._debugPushGroup?.(`Depth renderer`);
            }
        });
        this._depthMap.onAfterUnbindObservable.add(() => {
            if (engine._enableGPUDebugMarkers) {
                engine._debugPopGroup?.();
            }
        });
        this._depthMap.customIsReadyFunction = (mesh, refreshRate, preWarm) => {
            if ((preWarm || refreshRate === 0) && mesh.subMeshes) {
                for (let i = 0; i < mesh.subMeshes.length; ++i) {
                    const subMesh = mesh.subMeshes[i];
                    const renderingMesh = subMesh.getRenderingMesh();
                    const batch = renderingMesh._getInstancesRenderList(subMesh._id, !!subMesh.getReplacementMesh());
                    const hardwareInstancedRendering = engine.getCaps().instancedArrays &&
                        ((batch.visibleInstances[subMesh._id] !== null && batch.visibleInstances[subMesh._id] !== undefined) || renderingMesh.hasThinInstances);
                    if (!this.isReady(subMesh, hardwareInstancedRendering)) {
                        return false;
                    }
                }
            }
            return true;
        };
        // Custom render function
        const renderSubMesh = (subMesh) => {
            const renderingMesh = subMesh.getRenderingMesh();
            const effectiveMesh = subMesh.getEffectiveMesh();
            const scene = this._scene;
            const engine = scene.getEngine();
            const material = subMesh.getMaterial();
            effectiveMesh._internalAbstractMeshDataInfo._isActiveIntermediate = false;
            if (!material || effectiveMesh.infiniteDistance || material.disableDepthWrite || subMesh.verticesCount === 0 || subMesh._renderId === scene.getRenderId()) {
                return;
            }
            // Culling
            const detNeg = effectiveMesh._getWorldMatrixDeterminant() < 0;
            let sideOrientation = material._getEffectiveOrientation(renderingMesh);
            if (detNeg) {
                sideOrientation =
                    sideOrientation === 0
                        ? 1
                        : 0;
            }
            const reverseSideOrientation = sideOrientation === 0;
            engine.setState(material.backFaceCulling, 0, false, reverseSideOrientation, this.reverseCulling ? !material.cullBackFaces : material.cullBackFaces);
            // Managing instances
            const batch = renderingMesh._getInstancesRenderList(subMesh._id, !!subMesh.getReplacementMesh());
            if (batch.mustReturn) {
                return;
            }
            const hardwareInstancedRendering = engine.getCaps().instancedArrays &&
                ((batch.visibleInstances[subMesh._id] !== null && batch.visibleInstances[subMesh._id] !== undefined) || renderingMesh.hasThinInstances);
            const camera = this._camera || scene.activeCamera;
            if (this.isReady(subMesh, hardwareInstancedRendering) && camera) {
                subMesh._renderId = scene.getRenderId();
                let renderingMaterial = effectiveMesh._internalAbstractMeshDataInfo._materialForRenderPass?.[engine.currentRenderPassId];
                if (renderingMaterial === undefined && effectiveMesh.getClassName() === "GaussianSplattingMesh") {
                    const gsMaterial = effectiveMesh.material;
                    renderingMaterial = gsMaterial.makeDepthRenderingMaterial(this._scene, this._shaderLanguage);
                    this.setMaterialForRendering(effectiveMesh, renderingMaterial);
                    if (!renderingMaterial.isReady()) {
                        return;
                    }
                }
                let drawWrapper = subMesh._getDrawWrapper();
                if (!drawWrapper && renderingMaterial) {
                    drawWrapper = renderingMaterial._getDrawWrapper();
                }
                const cameraIsOrtho = camera.mode === Camera.ORTHOGRAPHIC_CAMERA;
                if (!drawWrapper) {
                    return;
                }
                const effect = drawWrapper.effect;
                engine.enableEffect(drawWrapper);
                if (!hardwareInstancedRendering) {
                    renderingMesh._bind(subMesh, effect, material.fillMode);
                }
                if (!renderingMaterial) {
                    effect.setMatrix("viewProjection", scene.getTransformMatrix());
                    effect.setMatrix("world", effectiveMesh.getWorldMatrix());
                    if (this._storeCameraSpaceZ) {
                        effect.setMatrix("view", scene.getViewMatrix());
                    }
                }
                else {
                    renderingMaterial.bindForSubMesh(effectiveMesh.getWorldMatrix(), effectiveMesh, subMesh);
                }
                let minZ, maxZ;
                if (cameraIsOrtho) {
                    minZ = !engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : 1;
                    maxZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : 1;
                }
                else {
                    minZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? camera.minZ : engine.isNDCHalfZRange ? 0 : camera.minZ;
                    maxZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : camera.maxZ;
                }
                effect.setFloat2("depthValues", minZ, minZ + maxZ);
                if (!renderingMaterial) {
                    // Alpha test
                    if (material.needAlphaTestingForMesh(effectiveMesh)) {
                        const alphaTexture = material.getAlphaTestTexture();
                        if (alphaTexture) {
                            effect.setTexture("diffuseSampler", alphaTexture);
                            effect.setMatrix("diffuseMatrix", alphaTexture.getTextureMatrix());
                        }
                    }
                    // Bones
                    BindBonesParameters(renderingMesh, effect);
                    // Clip planes
                    BindClipPlane(effect, material, scene);
                    // Morph targets
                    BindMorphTargetParameters(renderingMesh, effect);
                    if (renderingMesh.morphTargetManager && renderingMesh.morphTargetManager.isUsingTextureForTargets) {
                        renderingMesh.morphTargetManager._bind(effect);
                    }
                    // Baked vertex animations
                    const bvaManager = subMesh.getMesh().bakedVertexAnimationManager;
                    if (bvaManager && bvaManager.isEnabled) {
                        bvaManager.bind(effect, hardwareInstancedRendering);
                    }
                    // Points cloud rendering
                    if (material.pointsCloud) {
                        effect.setFloat("pointSize", material.pointSize);
                    }
                }
                // Draw
                renderingMesh._processRendering(effectiveMesh, subMesh, effect, material.fillMode, batch, hardwareInstancedRendering, (isInstance, world) => effect.setMatrix("world", world));
            }
        };
        this._depthMap.customRenderFunction = (opaqueSubMeshes, alphaTestSubMeshes, transparentSubMeshes, depthOnlySubMeshes) => {
            let index;
            if (depthOnlySubMeshes.length) {
                for (index = 0; index < depthOnlySubMeshes.length; index++) {
                    renderSubMesh(depthOnlySubMeshes.data[index]);
                }
            }
            for (index = 0; index < opaqueSubMeshes.length; index++) {
                renderSubMesh(opaqueSubMeshes.data[index]);
            }
            for (index = 0; index < alphaTestSubMeshes.length; index++) {
                renderSubMesh(alphaTestSubMeshes.data[index]);
            }
            if (this.forceDepthWriteTransparentMeshes) {
                for (index = 0; index < transparentSubMeshes.length; index++) {
                    renderSubMesh(transparentSubMeshes.data[index]);
                }
            }
            else {
                for (index = 0; index < transparentSubMeshes.length; index++) {
                    transparentSubMeshes.data[index].getEffectiveMesh()._internalAbstractMeshDataInfo._isActiveIntermediate = false;
                }
            }
        };
    }
    async _initShaderSourceAsync(forceGLSL = false) {
        const engine = this._scene.getEngine();
        if (engine.isWebGPU && !forceGLSL && !DepthRenderer.ForceGLSL) {
            this._shaderLanguage = 1 /* ShaderLanguage.WGSL */;
            await Promise.all([__vitePreload(() => import('./depth.vertex-k8oMaAk6.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11]):void 0,import.meta.url), __vitePreload(() => import('./depth.fragment-PGqN_SYP.js'),true              ?__vite__mapDeps([12,1,2,3,13,14]):void 0,import.meta.url)]);
        }
        else {
            await Promise.all([__vitePreload(() => Promise.resolve().then(() => depth_vertex),true              ?void 0:void 0,import.meta.url), __vitePreload(() => Promise.resolve().then(() => depth_fragment),true              ?void 0:void 0,import.meta.url)]);
        }
        this._shadersLoaded = true;
    }
    /**
     * Creates the depth rendering effect and checks if the effect is ready.
     * @param subMesh The submesh to be used to render the depth map of
     * @param useInstances If multiple world instances should be used
     * @returns if the depth renderer is ready to render the depth map
     */
    isReady(subMesh, useInstances) {
        if (!this._shadersLoaded) {
            return false;
        }
        const engine = this._scene.getEngine();
        const mesh = subMesh.getMesh();
        const scene = mesh.getScene();
        const renderingMaterial = mesh._internalAbstractMeshDataInfo._materialForRenderPass?.[engine.currentRenderPassId];
        if (renderingMaterial) {
            return renderingMaterial.isReadyForSubMesh(mesh, subMesh, useInstances);
        }
        const material = subMesh.getMaterial();
        if (!material || material.disableDepthWrite) {
            return false;
        }
        const defines = [];
        const attribs = [VertexBuffer.PositionKind];
        let uv1 = false;
        let uv2 = false;
        const color = false;
        // Alpha test
        if (material.needAlphaTestingForMesh(mesh) && material.getAlphaTestTexture()) {
            defines.push("#define ALPHATEST");
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
        // Bones
        const fallbacks = new EffectFallbacks();
        if (mesh.useBones && mesh.computeBonesUsingShaders && mesh.skeleton) {
            attribs.push(VertexBuffer.MatricesIndicesKind);
            attribs.push(VertexBuffer.MatricesWeightsKind);
            if (mesh.numBoneInfluencers > 4) {
                attribs.push(VertexBuffer.MatricesIndicesExtraKind);
                attribs.push(VertexBuffer.MatricesWeightsExtraKind);
            }
            defines.push("#define NUM_BONE_INFLUENCERS " + mesh.numBoneInfluencers);
            if (mesh.numBoneInfluencers > 0) {
                fallbacks.addCPUSkinningFallback(0, mesh);
            }
            const skeleton = mesh.skeleton;
            if (skeleton.isUsingTextureForMatrices) {
                defines.push("#define BONETEXTURE");
            }
            else {
                defines.push("#define BonesPerMesh " + (skeleton.bones.length + 1));
            }
        }
        else {
            defines.push("#define NUM_BONE_INFLUENCERS 0");
        }
        // Morph targets
        const numMorphInfluencers = mesh.morphTargetManager
            ? PrepareDefinesAndAttributesForMorphTargets(mesh.morphTargetManager, defines, attribs, mesh, true, // usePositionMorph
            false, // useNormalMorph
            false, // useTangentMorph
            uv1, // useUVMorph
            uv2, // useUV2Morph
            color // useColorMorph
            )
            : 0;
        // Points cloud rendering
        if (material.pointsCloud) {
            defines.push("#define POINTSIZE");
        }
        // Instances
        if (useInstances) {
            defines.push("#define INSTANCES");
            PushAttributesForInstances(attribs);
            if (subMesh.getRenderingMesh().hasThinInstances) {
                defines.push("#define THIN_INSTANCES");
            }
        }
        // Baked vertex animations
        const bvaManager = mesh.bakedVertexAnimationManager;
        if (bvaManager && bvaManager.isEnabled) {
            defines.push("#define BAKED_VERTEX_ANIMATION_TEXTURE");
            if (useInstances) {
                attribs.push("bakedVertexAnimationSettingsInstanced");
            }
        }
        // None linear depth
        if (this._storeNonLinearDepth) {
            defines.push("#define NONLINEARDEPTH");
        }
        // Store camera space Z coordinate instead of NDC Z
        if (this._storeCameraSpaceZ) {
            defines.push("#define STORE_CAMERASPACE_Z");
        }
        // Float Mode
        if (this.isPacked) {
            defines.push("#define PACKED");
        }
        // Clip planes
        PrepareStringDefinesForClipPlanes(material, scene, defines);
        // Get correct effect
        const drawWrapper = subMesh._getDrawWrapper(undefined, true);
        const cachedDefines = drawWrapper.defines;
        const join = defines.join("\n");
        if (cachedDefines !== join) {
            const uniforms = [
                "world",
                "mBones",
                "boneTextureWidth",
                "pointSize",
                "viewProjection",
                "view",
                "diffuseMatrix",
                "depthValues",
                "morphTargetInfluences",
                "morphTargetCount",
                "morphTargetTextureInfo",
                "morphTargetTextureIndices",
                "bakedVertexAnimationSettings",
                "bakedVertexAnimationTextureSizeInverted",
                "bakedVertexAnimationTime",
                "bakedVertexAnimationTexture",
            ];
            const samplers = ["diffuseSampler", "morphTargets", "boneSampler", "bakedVertexAnimationTexture"];
            AddClipPlaneUniforms(uniforms);
            drawWrapper.setEffect(engine.createEffect("depth", {
                attributes: attribs,
                uniformsNames: uniforms,
                uniformBuffersNames: [],
                samplers: samplers,
                defines: join,
                fallbacks: fallbacks,
                onCompiled: null,
                onError: null,
                indexParameters: { maxSimultaneousMorphTargets: numMorphInfluencers },
                shaderLanguage: this._shaderLanguage,
            }, engine), join);
        }
        return drawWrapper.effect.isReady();
    }
    /**
     * Gets the texture which the depth map will be written to.
     * @returns The depth map texture
     */
    getDepthMap() {
        return this._depthMap;
    }
    /**
     * Disposes of the depth renderer.
     */
    dispose() {
        const keysToDelete = [];
        for (const key in this._scene._depthRenderer) {
            const depthRenderer = this._scene._depthRenderer[key];
            if (depthRenderer === this) {
                keysToDelete.push(key);
            }
        }
        if (keysToDelete.length > 0) {
            this._depthMap.dispose();
            for (const key of keysToDelete) {
                delete this._scene._depthRenderer[key];
            }
        }
    }
}
/**
 * Force all the depth renderer to compile to glsl even on WebGPU engines.
 * False by default. This is mostly meant for backward compatibility.
 */
DepthRenderer.ForceGLSL = false;
/**
 * @internal
 */
DepthRenderer._SceneComponentInitialization = (_) => {
    throw _WarnImport("DepthRendererSceneComponent");
};

/**
 * @internal
 */
var DepthTextureType;
(function (DepthTextureType) {
    DepthTextureType[DepthTextureType["NormalizedViewDepth"] = 0] = "NormalizedViewDepth";
    DepthTextureType[DepthTextureType["ViewDepth"] = 1] = "ViewDepth";
    DepthTextureType[DepthTextureType["ScreenDepth"] = 2] = "ScreenDepth";
})(DepthTextureType || (DepthTextureType = {}));
/**
 * @internal
 */
class ThinMinMaxReducerPostProcess extends EffectWrapper {
    _gatherImports(useWebGPU, list) {
        if (useWebGPU) {
            this._webGPUReady = true;
            list.push(__vitePreload(() => Promise.resolve().then(() => minmaxRedux_fragment),true              ?void 0:void 0,import.meta.url));
        }
        else {
            list.push(__vitePreload(() => Promise.resolve().then(() => minmaxRedux_fragment$1),true              ?void 0:void 0,import.meta.url));
        }
    }
    constructor(name, engine = null, defines = "", options) {
        super({
            ...options,
            name,
            engine: engine || Engine.LastCreatedEngine,
            useShaderStore: true,
            useAsPostProcess: true,
            fragmentShader: ThinMinMaxReducerPostProcess.FragmentUrl,
            uniforms: ThinMinMaxReducerPostProcess.Uniforms,
            defines,
        });
        this.textureWidth = 0;
        this.textureHeight = 0;
    }
    bind(noDefaultBindings = false) {
        super.bind(noDefaultBindings);
        const effect = this.drawWrapper.effect;
        if (this.textureWidth === 1 || this.textureHeight === 1) {
            effect.setInt2("texSize", this.textureWidth, this.textureHeight);
        }
        else {
            effect.setFloat2("texSize", this.textureWidth, this.textureHeight);
        }
    }
}
ThinMinMaxReducerPostProcess.FragmentUrl = "minmaxRedux";
ThinMinMaxReducerPostProcess.Uniforms = ["texSize"];
const BufferFloat = new Float32Array(4 * 1 * 1);
const BufferUint8 = new Uint8Array(4 * 1 * 1);
const MinMax = { min: 0, max: 0 };
/**
 * @internal
 */
class ThinMinMaxReducer {
    get depthRedux() {
        return this._depthRedux;
    }
    set depthRedux(value) {
        if (this._depthRedux === value) {
            return;
        }
        this._depthRedux = value;
        this._recreatePostProcesses();
    }
    get textureWidth() {
        return this._textureWidth;
    }
    get textureHeight() {
        return this._textureHeight;
    }
    constructor(scene, depthRedux = true) {
        this.onAfterReductionPerformed = new Observable();
        this._textureWidth = 0;
        this._textureHeight = 0;
        this._scene = scene;
        this._depthRedux = depthRedux;
        this.reductionSteps = [];
    }
    setTextureDimensions(width, height, depthTextureType = 0 /* DepthTextureType.NormalizedViewDepth */) {
        if (width === this._textureWidth && height === this._textureHeight && depthTextureType === this._depthTextureType) {
            return false;
        }
        this._textureWidth = width;
        this._textureHeight = height;
        this._depthTextureType = depthTextureType;
        this._recreatePostProcesses();
        return true;
    }
    readMinMax(texture) {
        // Note that we should normally await the call to _readTexturePixels!
        // But because WebGL does the read synchronously, we know the values will be updated without waiting for the promise to be resolved, which will let us get the updated values
        // in the current frame, whereas in WebGPU, the read is asynchronous and we should normally wait for the promise to be resolved to get the updated values.
        // However, it's safe to avoid waiting for the promise to be resolved in WebGPU as well, because we will simply use the current values until "buffer" is updated later on.
        // Note that it means we can suffer some rendering artifacts in WebGPU because we may use previous min/max values for the current frame.
        const isFloat = texture.type === Engine.TEXTURETYPE_FLOAT || texture.type === Engine.TEXTURETYPE_HALF_FLOAT;
        const buffer = isFloat ? BufferFloat : BufferUint8;
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._scene.getEngine()._readTexturePixels(texture, 1, 1, -1, 0, buffer, false);
        MinMax.min = buffer[0];
        MinMax.max = buffer[1];
        if (!isFloat) {
            MinMax.min = MinMax.min / 255.0;
            MinMax.max = MinMax.max / 255.0;
        }
        if (MinMax.min >= MinMax.max) {
            MinMax.min = 0;
            MinMax.max = 1;
        }
        this.onAfterReductionPerformed.notifyObservers(MinMax);
    }
    dispose(disposeAll = true) {
        if (disposeAll) {
            this.onAfterReductionPerformed.clear();
            this._textureWidth = 0;
            this._textureHeight = 0;
        }
        for (let i = 0; i < this.reductionSteps.length; ++i) {
            this.reductionSteps[i].dispose();
        }
        this.reductionSteps.length = 0;
    }
    _recreatePostProcesses() {
        this.dispose(false);
        const scene = this._scene;
        let w = this.textureWidth, h = this.textureHeight;
        const reductionInitial = new ThinMinMaxReducerPostProcess("Initial reduction phase", scene.getEngine(), "#define INITIAL" + (this._depthRedux ? "\n#define DEPTH_REDUX" : "") + (this._depthTextureType === 1 /* DepthTextureType.ViewDepth */ ? "\n#define VIEW_DEPTH" : ""));
        reductionInitial.textureWidth = w;
        reductionInitial.textureHeight = h;
        this.reductionSteps.push(reductionInitial);
        let index = 1;
        // create the additional steps
        while (w > 1 || h > 1) {
            w = Math.max(Math.round(w / 2), 1);
            h = Math.max(Math.round(h / 2), 1);
            const reduction = new ThinMinMaxReducerPostProcess("Reduction phase " + index, scene.getEngine(), "#define " + (w == 1 && h == 1 ? "LAST" : w == 1 || h == 1 ? "ONEBEFORELAST" : "MAIN"));
            reduction.textureWidth = w;
            reduction.textureHeight = h;
            this.reductionSteps.push(reduction);
            index++;
        }
    }
}

// Do not edit.
const name$1 = "minmaxReduxPixelShader";
const shader$1 = `varying vec2 vUV;uniform sampler2D textureSampler;
#if defined(INITIAL)
uniform vec2 texSize;void main(void)
{ivec2 coord=ivec2(vUV*(texSize-1.0));float f1=texelFetch(textureSampler,coord,0).r;float f2=texelFetch(textureSampler,coord+ivec2(1,0),0).r;float f3=texelFetch(textureSampler,coord+ivec2(1,1),0).r;float f4=texelFetch(textureSampler,coord+ivec2(0,1),0).r;
#ifdef DEPTH_REDUX
#ifdef VIEW_DEPTH
float minz=3.4e38;if (f1 != 0.0) { minz=f1; }
if (f2 != 0.0) { minz=min(minz,f2); }
if (f3 != 0.0) { minz=min(minz,f3); }
if (f4 != 0.0) { minz=min(minz,f4); }
float maxz=max(max(max(f1,f2),f3),f4);
#else
float minz=min(min(min(f1,f2),f3),f4);float maxz=max(max(max(sign(1.0-f1)*f1,sign(1.0-f2)*f2),sign(1.0-f3)*f3),sign(1.0-f4)*f4);
#endif
#else
float minz=min(min(min(f1,f2),f3),f4);float maxz=max(max(max(f1,f2),f3),f4);
#endif
glFragColor=vec4(minz,maxz,0.,0.);}
#elif defined(MAIN)
uniform vec2 texSize;void main(void)
{ivec2 coord=ivec2(vUV*(texSize-1.0));vec2 f1=texelFetch(textureSampler,coord,0).rg;vec2 f2=texelFetch(textureSampler,coord+ivec2(1,0),0).rg;vec2 f3=texelFetch(textureSampler,coord+ivec2(1,1),0).rg;vec2 f4=texelFetch(textureSampler,coord+ivec2(0,1),0).rg;float minz=min(min(min(f1.x,f2.x),f3.x),f4.x);float maxz=max(max(max(f1.y,f2.y),f3.y),f4.y);glFragColor=vec4(minz,maxz,0.,0.);}
#elif defined(ONEBEFORELAST)
uniform ivec2 texSize;void main(void)
{ivec2 coord=ivec2(vUV*vec2(texSize-1));vec2 f1=texelFetch(textureSampler,coord % texSize,0).rg;vec2 f2=texelFetch(textureSampler,(coord+ivec2(1,0)) % texSize,0).rg;vec2 f3=texelFetch(textureSampler,(coord+ivec2(1,1)) % texSize,0).rg;vec2 f4=texelFetch(textureSampler,(coord+ivec2(0,1)) % texSize,0).rg;float minz=min(min(min(f1.x,f2.x),f3.x),f4.x);float maxz=max(max(max(f1.y,f2.y),f3.y),f4.y);glFragColor=vec4(minz,maxz,0.,0.);}
#elif defined(LAST)
void main(void)
{glFragColor=vec4(0.);if (true) { 
discard;}}
#endif
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$1]) {
    ShaderStore.ShadersStore[name$1] = shader$1;
}
/** @internal */
const minmaxReduxPixelShader = { name: name$1, shader: shader$1 };

const minmaxRedux_fragment$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    minmaxReduxPixelShader
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name = "minmaxReduxPixelShader";
const shader = `varying vUV: vec2f;var textureSampler: texture_2d<f32>;
#if defined(INITIAL)
uniform texSize: vec2f;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {let coord=vec2i(fragmentInputs.vUV*(uniforms.texSize-1.0));let f1=textureLoad(textureSampler,coord,0).r;let f2=textureLoad(textureSampler,coord+vec2i(1,0),0).r;let f3=textureLoad(textureSampler,coord+vec2i(1,1),0).r;let f4=textureLoad(textureSampler,coord+vec2i(0,1),0).r;
#ifdef DEPTH_REDUX
#ifdef VIEW_DEPTH
var minz=3.4e38;if (f1 != 0.0) { minz=f1; }
if (f2 != 0.0) { minz=min(minz,f2); }
if (f3 != 0.0) { minz=min(minz,f3); }
if (f4 != 0.0) { minz=min(minz,f4); }
let maxz=max(max(max(f1,f2),f3),f4);
#else
let minz=min(min(min(f1,f2),f3),f4);let maxz=max(max(max(sign(1.0-f1)*f1,sign(1.0-f2)*f2),sign(1.0-f3)*f3),sign(1.0-f4)*f4);
#endif
#else
let minz=min(min(min(f1,f2),f3),f4);let maxz=max(max(max(f1,f2),f3),f4);
#endif
fragmentOutputs.color=vec4f(minz,maxz,0.,0.);}
#elif defined(MAIN)
uniform texSize: vec2f;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {let coord=vec2i(fragmentInputs.vUV*(uniforms.texSize-1.0));let f1=textureLoad(textureSampler,coord,0).rg;let f2=textureLoad(textureSampler,coord+vec2i(1,0),0).rg;let f3=textureLoad(textureSampler,coord+vec2i(1,1),0).rg;let f4=textureLoad(textureSampler,coord+vec2i(0,1),0).rg;let minz=min(min(min(f1.x,f2.x),f3.x),f4.x);let maxz=max(max(max(f1.y,f2.y),f3.y),f4.y);fragmentOutputs.color=vec4(minz,maxz,0.,0.);}
#elif defined(ONEBEFORELAST)
uniform texSize: vec2i;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {let coord=vec2i(fragmentInputs.vUV*vec2f(uniforms.texSize-1));let f1=textureLoad(textureSampler,coord % uniforms.texSize,0).rg;let f2=textureLoad(textureSampler,(coord+vec2i(1,0)) % uniforms.texSize,0).rg;let f3=textureLoad(textureSampler,(coord+vec2i(1,1)) % uniforms.texSize,0).rg;let f4=textureLoad(textureSampler,(coord+vec2i(0,1)) % uniforms.texSize,0).rg;let minz=min(min(min(f1.x,f2.x),f3.x),f4.x);let maxz=max(max(max(f1.y,f2.y),f3.y),f4.y);fragmentOutputs.color=vec4(minz,maxz,0.,0.);}
#elif defined(LAST)
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=vec4f(0.);if (true) { 
discard;}}
#endif
`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const minmaxReduxPixelShaderWGSL = { name, shader };

const minmaxRedux_fragment = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    minmaxReduxPixelShaderWGSL
}, Symbol.toStringTag, { value: 'Module' }));

/**
 * This class computes a min/max reduction from a texture: it means it computes the minimum
 * and maximum values from all values of the texture.
 * It is performed on the GPU for better performances, thanks to a succession of post processes.
 * The source values are read from the red channel of the texture.
 */
class MinMaxReducer {
    /**
     * Observable triggered when the computation has been performed
     */
    get onAfterReductionPerformed() {
        return this._thinMinMaxReducer.onAfterReductionPerformed;
    }
    /**
     * Creates a min/max reducer
     * @param camera The camera to use for the post processes
     */
    constructor(camera) {
        this._onAfterUnbindObserver = null;
        this._forceFullscreenViewport = true;
        this._activated = false;
        this._camera = camera;
        this._postProcessManager = new PostProcessManager(camera.getScene());
        this._thinMinMaxReducer = new ThinMinMaxReducer(camera.getScene());
        this._reductionSteps = [];
        this._onContextRestoredObserver = camera.getEngine().onContextRestoredObservable.add(() => {
            this._postProcessManager._rebuild();
        });
    }
    /**
     * Gets the texture used to read the values from.
     */
    get sourceTexture() {
        return this._sourceTexture;
    }
    /**
     * Sets the source texture to read the values from.
     * One must indicate if the texture is a depth texture or not through the depthRedux parameter
     * because in such textures '1' value must not be taken into account to compute the maximum
     * as this value is used to clear the texture.
     * Note that the computation is not activated by calling this function, you must call activate() for that!
     * @param sourceTexture The texture to read the values from. The values should be in the red channel.
     * @param depthRedux Indicates if the texture is a depth texture or not
     * @param type The type of the textures created for the reduction (defaults to TEXTURETYPE_HALF_FLOAT)
     * @param forceFullscreenViewport Forces the post processes used for the reduction to be applied without taking into account viewport (defaults to true)
     */
    setSourceTexture(sourceTexture, depthRedux, type = 2, forceFullscreenViewport = true) {
        if (sourceTexture === this._sourceTexture) {
            return;
        }
        this._thinMinMaxReducer.depthRedux = depthRedux;
        this.deactivate();
        this._sourceTexture = sourceTexture;
        this._forceFullscreenViewport = forceFullscreenViewport;
        if (this._thinMinMaxReducer.setTextureDimensions(sourceTexture.getRenderWidth(), sourceTexture.getRenderHeight())) {
            this._disposePostProcesses();
            const reductionSteps = this._thinMinMaxReducer.reductionSteps;
            for (let i = 0; i < reductionSteps.length; ++i) {
                const reductionStep = reductionSteps[i];
                const postProcess = new PostProcess(reductionStep.name, ThinMinMaxReducerPostProcess.FragmentUrl, {
                    effectWrapper: reductionStep,
                    samplingMode: 1,
                    engine: this._camera.getScene().getEngine(),
                    textureType: type,
                    textureFormat: 7,
                    size: { width: reductionStep.textureWidth, height: reductionStep.textureHeight },
                });
                this._reductionSteps.push(postProcess);
                postProcess.autoClear = false;
                postProcess.forceFullscreenViewport = forceFullscreenViewport;
                if (i === 0) {
                    postProcess.externalTextureSamplerBinding = true;
                    postProcess.onApplyObservable.add((effect) => {
                        effect.setTexture("textureSampler", this._sourceTexture);
                    });
                }
                if (i === reductionSteps.length - 1) {
                    this._reductionSteps[i - 1].onAfterRenderObservable.add(() => {
                        this._thinMinMaxReducer.readMinMax(postProcess.inputTexture.texture);
                    });
                }
            }
        }
    }
    /**
     * Defines the refresh rate of the computation.
     * Use 0 to compute just once, 1 to compute on every frame, 2 to compute every two frames and so on...
     */
    get refreshRate() {
        return this._sourceTexture ? this._sourceTexture.refreshRate : -1;
    }
    set refreshRate(value) {
        if (this._sourceTexture) {
            this._sourceTexture.refreshRate = value;
        }
    }
    /**
     * Gets the activation status of the reducer
     */
    get activated() {
        return this._activated;
    }
    /**
     * Activates the reduction computation.
     * When activated, the observers registered in onAfterReductionPerformed are
     * called after the computation is performed
     */
    activate() {
        if (this._onAfterUnbindObserver || !this._sourceTexture) {
            return;
        }
        this._onAfterUnbindObserver = this._sourceTexture.onAfterUnbindObservable.add(() => {
            const engine = this._camera.getScene().getEngine();
            engine._debugPushGroup?.(`min max reduction`);
            this._reductionSteps[0].activate(this._camera);
            this._postProcessManager.directRender(this._reductionSteps, this._reductionSteps[0].inputTexture, this._forceFullscreenViewport, 0, 0, true, this._reductionSteps.length - 1);
            engine.unBindFramebuffer(this._reductionSteps[this._reductionSteps.length - 1].inputTexture, false);
            engine._debugPopGroup?.();
        });
        this._activated = true;
    }
    /**
     * Deactivates the reduction computation.
     */
    deactivate() {
        if (!this._onAfterUnbindObserver || !this._sourceTexture) {
            return;
        }
        this._sourceTexture.onAfterUnbindObservable.remove(this._onAfterUnbindObserver);
        this._onAfterUnbindObserver = null;
        this._activated = false;
    }
    /**
     * Disposes the min/max reducer
     * @param disposeAll true to dispose all the resources. You should always call this function with true as the parameter (or without any parameter as it is the default one). This flag is meant to be used internally.
     */
    dispose(disposeAll = true) {
        if (!disposeAll) {
            return;
        }
        this.onAfterReductionPerformed.clear();
        this._camera.getEngine().onContextRestoredObservable.remove(this._onContextRestoredObserver);
        this._onContextRestoredObserver = undefined;
        this._disposePostProcesses();
        this._postProcessManager.dispose();
        this._postProcessManager = undefined;
        this._thinMinMaxReducer.dispose();
        this._thinMinMaxReducer = undefined;
        this._sourceTexture = null;
    }
    _disposePostProcesses() {
        for (let i = 0; i < this._reductionSteps.length; ++i) {
            this._reductionSteps[i].dispose();
        }
        this._reductionSteps.length = 0;
    }
}

/**
 * This class is a small wrapper around the MinMaxReducer class to compute the min/max values of a depth texture
 */
class DepthReducer extends MinMaxReducer {
    /**
     * Gets the depth renderer used for the computation.
     * Note that the result is null if you provide your own renderer when calling setDepthRenderer.
     */
    get depthRenderer() {
        return this._depthRenderer;
    }
    /**
     * Creates a depth reducer
     * @param camera The camera used to render the depth texture
     */
    constructor(camera) {
        super(camera);
    }
    /**
     * Sets the depth renderer to use to generate the depth map
     * @param depthRenderer The depth renderer to use. If not provided, a new one will be created automatically
     * @param type The texture type of the depth map (default: TEXTURETYPE_HALF_FLOAT)
     * @param forceFullscreenViewport Forces the post processes used for the reduction to be applied without taking into account viewport (defaults to true)
     */
    setDepthRenderer(depthRenderer = null, type = 2, forceFullscreenViewport = true) {
        const scene = this._camera.getScene();
        if (this._depthRenderer) {
            delete scene._depthRenderer[this._depthRendererId];
            this._depthRenderer.dispose();
            this._depthRenderer = null;
        }
        if (depthRenderer === null) {
            if (!scene._depthRenderer) {
                scene._depthRenderer = {};
            }
            this._depthRendererId = "minmax_" + this._camera.id;
            depthRenderer = this._depthRenderer = new DepthRenderer(scene, type, this._camera, false, 1, false, `DepthRenderer ${this._depthRendererId}`);
            depthRenderer.enabled = false;
            scene._depthRenderer[this._depthRendererId] = depthRenderer;
        }
        super.setSourceTexture(depthRenderer.getDepthMap(), true, type, forceFullscreenViewport);
    }
    /**
     * @internal
     */
    setSourceTexture(sourceTexture, depthRedux, type = 2, forceFullscreenViewport = true) {
        super.setSourceTexture(sourceTexture, depthRedux, type, forceFullscreenViewport);
    }
    /**
     * Activates the reduction computation.
     * When activated, the observers registered in onAfterReductionPerformed are
     * called after the computation is performed
     */
    activate() {
        if (this._depthRenderer) {
            this._depthRenderer.enabled = true;
        }
        super.activate();
    }
    /**
     * Deactivates the reduction computation.
     */
    deactivate() {
        super.deactivate();
        if (this._depthRenderer) {
            this._depthRenderer.enabled = false;
        }
    }
    /**
     * Disposes the depth reducer
     * @param disposeAll true to dispose all the resources. You should always call this function with true as the parameter (or without any parameter as it is the default one). This flag is meant to be used internally.
     */
    dispose(disposeAll = true) {
        super.dispose(disposeAll);
        if (this._depthRenderer && disposeAll) {
            this._depthRenderer.dispose();
            this._depthRenderer = null;
        }
    }
}

// eslint-disable-next-line @typescript-eslint/naming-convention
const UpDir = Vector3.Up();
// eslint-disable-next-line @typescript-eslint/naming-convention
const ZeroVec = Vector3.Zero();
const Tmpv1 = new Vector3();
const Tmpv2 = new Vector3();
const TmpMatrix = new Matrix();
/**
 * A CSM implementation allowing casting shadows on large scenes.
 * Documentation : https://doc.babylonjs.com/babylon101/cascadedShadows
 * Based on: https://github.com/TheRealMJP/Shadows and https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
 */
class CascadedShadowGenerator extends ShadowGenerator {
    _validateFilter(filter) {
        if (filter === ShadowGenerator.FILTER_NONE || filter === ShadowGenerator.FILTER_PCF || filter === ShadowGenerator.FILTER_PCSS) {
            return filter;
        }
        Logger.Error('Unsupported filter "' + filter + '"!');
        return ShadowGenerator.FILTER_NONE;
    }
    /**
     * Gets or set the number of cascades used by the CSM.
     */
    get numCascades() {
        return this._numCascades;
    }
    set numCascades(value) {
        value = Math.min(Math.max(value, CascadedShadowGenerator.MIN_CASCADES_COUNT), CascadedShadowGenerator.MAX_CASCADES_COUNT);
        if (value === this._numCascades) {
            return;
        }
        this._numCascades = value;
        this.recreateShadowMap();
        this._recreateSceneUBOs();
    }
    /**
     * Enables or disables the shadow casters bounding info computation.
     * If your shadow casters don't move, you can disable this feature.
     * If it is enabled, the bounding box computation is done every frame.
     */
    get freezeShadowCastersBoundingInfo() {
        return this._freezeShadowCastersBoundingInfo;
    }
    set freezeShadowCastersBoundingInfo(freeze) {
        if (this._freezeShadowCastersBoundingInfoObservable && freeze) {
            this._scene.onBeforeRenderObservable.remove(this._freezeShadowCastersBoundingInfoObservable);
            this._freezeShadowCastersBoundingInfoObservable = null;
        }
        if (!this._freezeShadowCastersBoundingInfoObservable && !freeze) {
            this._freezeShadowCastersBoundingInfoObservable = this._scene.onBeforeRenderObservable.add(() => this._computeShadowCastersBoundingInfo());
        }
        this._freezeShadowCastersBoundingInfo = freeze;
        if (freeze) {
            this._computeShadowCastersBoundingInfo();
        }
    }
    _computeShadowCastersBoundingInfo() {
        this._scbiMin.copyFromFloats(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
        this._scbiMax.copyFromFloats(-Number.MAX_VALUE, -Number.MAX_VALUE, -Number.MAX_VALUE);
        if (this._shadowMap && this._shadowMap.renderList) {
            const renderList = this._shadowMap.renderList;
            for (let meshIndex = 0; meshIndex < renderList.length; meshIndex++) {
                const mesh = renderList[meshIndex];
                if (!mesh) {
                    continue;
                }
                const boundingInfo = mesh.getBoundingInfo(), boundingBox = boundingInfo.boundingBox;
                this._scbiMin.minimizeInPlace(boundingBox.minimumWorld);
                this._scbiMax.maximizeInPlace(boundingBox.maximumWorld);
            }
        }
        this._shadowCastersBoundingInfo.reConstruct(this._scbiMin, this._scbiMax);
    }
    /**
     * Gets or sets the shadow casters bounding info.
     * If you provide your own shadow casters bounding info, first enable freezeShadowCastersBoundingInfo
     * so that the system won't overwrite the bounds you provide
     */
    get shadowCastersBoundingInfo() {
        return this._shadowCastersBoundingInfo;
    }
    set shadowCastersBoundingInfo(boundingInfo) {
        this._shadowCastersBoundingInfo = boundingInfo;
    }
    /**
     * Sets the minimal and maximal distances to use when computing the cascade breaks.
     *
     * The values of min / max are typically the depth zmin and zmax values of your scene, for a given frame.
     * If you don't know these values, simply leave them to their defaults and don't call this function.
     * @param min minimal distance for the breaks (default to 0.)
     * @param max maximal distance for the breaks (default to 1.)
     */
    setMinMaxDistance(min, max) {
        if (this._minDistance === min && this._maxDistance === max) {
            return;
        }
        if (min > max) {
            min = 0;
            max = 1;
        }
        if (min < 0) {
            min = 0;
        }
        if (max > 1) {
            max = 1;
        }
        this._minDistance = min;
        this._maxDistance = max;
        this._breaksAreDirty = true;
    }
    /** Gets the minimal distance used in the cascade break computation */
    get minDistance() {
        return this._minDistance;
    }
    /** Gets the maximal distance used in the cascade break computation */
    get maxDistance() {
        return this._maxDistance;
    }
    /**
     * Gets the class name of that object
     * @returns "CascadedShadowGenerator"
     */
    getClassName() {
        return CascadedShadowGenerator.CLASSNAME;
    }
    /**
     * Gets a cascade minimum extents
     * @param cascadeIndex index of the cascade
     * @returns the minimum cascade extents
     */
    getCascadeMinExtents(cascadeIndex) {
        return cascadeIndex >= 0 && cascadeIndex < this._numCascades ? this._cascadeMinExtents[cascadeIndex] : null;
    }
    /**
     * Gets a cascade maximum extents
     * @param cascadeIndex index of the cascade
     * @returns the maximum cascade extents
     */
    getCascadeMaxExtents(cascadeIndex) {
        return cascadeIndex >= 0 && cascadeIndex < this._numCascades ? this._cascadeMaxExtents[cascadeIndex] : null;
    }
    /**
     * Gets the shadow max z distance. It's the limit beyond which shadows are not displayed.
     * It defaults to camera.maxZ
     */
    get shadowMaxZ() {
        if (!this._getCamera()) {
            return 0;
        }
        return this._shadowMaxZ;
    }
    /**
     * Sets the shadow max z distance.
     */
    set shadowMaxZ(value) {
        const camera = this._getCamera();
        if (!camera) {
            this._shadowMaxZ = value;
            return;
        }
        if (this._shadowMaxZ === value || value < camera.minZ || (value > camera.maxZ && camera.maxZ !== 0)) {
            return;
        }
        this._shadowMaxZ = value;
        this._light._markMeshesAsLightDirty();
        this._breaksAreDirty = true;
    }
    /**
     * Gets or sets the debug flag.
     * When enabled, the cascades are materialized by different colors on the screen.
     */
    get debug() {
        return this._debug;
    }
    set debug(dbg) {
        this._debug = dbg;
        this._light._markMeshesAsLightDirty();
    }
    /**
     * Gets or sets the depth clamping value.
     *
     * When enabled, it improves the shadow quality because the near z plane of the light frustum don't need to be adjusted
     * to account for the shadow casters far away.
     *
     * Note that this property is incompatible with PCSS filtering, so it won't be used in that case.
     */
    get depthClamp() {
        return this._depthClamp;
    }
    set depthClamp(value) {
        this._depthClamp = value;
    }
    /**
     * Gets or sets the percentage of blending between two cascades (value between 0. and 1.).
     * It defaults to 0.1 (10% blending).
     */
    get cascadeBlendPercentage() {
        return this._cascadeBlendPercentage;
    }
    set cascadeBlendPercentage(value) {
        this._cascadeBlendPercentage = value;
        this._light._markMeshesAsLightDirty();
    }
    /**
     * Gets or set the lambda parameter.
     * This parameter is used to split the camera frustum and create the cascades.
     * It's a value between 0. and 1.: If 0, the split is a uniform split of the frustum, if 1 it is a logarithmic split.
     * For all values in-between, it's a linear combination of the uniform and logarithm split algorithm.
     */
    get lambda() {
        return this._lambda;
    }
    set lambda(value) {
        const lambda = Math.min(Math.max(value, 0), 1);
        if (this._lambda == lambda) {
            return;
        }
        this._lambda = lambda;
        this._breaksAreDirty = true;
    }
    /**
     * Gets the view matrix corresponding to a given cascade
     * @param cascadeNum cascade to retrieve the view matrix from
     * @returns the cascade view matrix
     */
    getCascadeViewMatrix(cascadeNum) {
        return cascadeNum >= 0 && cascadeNum < this._numCascades ? this._viewMatrices[cascadeNum] : null;
    }
    /**
     * Gets the projection matrix corresponding to a given cascade
     * @param cascadeNum cascade to retrieve the projection matrix from
     * @returns the cascade projection matrix
     */
    getCascadeProjectionMatrix(cascadeNum) {
        return cascadeNum >= 0 && cascadeNum < this._numCascades ? this._projectionMatrices[cascadeNum] : null;
    }
    /**
     * Gets the transformation matrix corresponding to a given cascade
     * @param cascadeNum cascade to retrieve the transformation matrix from
     * @returns the cascade transformation matrix
     */
    getCascadeTransformMatrix(cascadeNum) {
        return cascadeNum >= 0 && cascadeNum < this._numCascades ? this._transformMatrices[cascadeNum] : null;
    }
    /**
     * Sets the depth renderer to use when autoCalcDepthBounds is enabled.
     *
     * Note that if no depth renderer is set, a new one will be automatically created internally when necessary.
     *
     * You should call this function if you already have a depth renderer enabled in your scene, to avoid
     * doing multiple depth rendering each frame. If you provide your own depth renderer, make sure it stores linear depth!
     * @param depthRenderer The depth renderer to use when autoCalcDepthBounds is enabled. If you pass null or don't call this function at all, a depth renderer will be automatically created
     */
    setDepthRenderer(depthRenderer) {
        this._depthRenderer = depthRenderer;
        if (this._depthReducer) {
            this._depthReducer.setDepthRenderer(this._depthRenderer);
        }
    }
    /**
     * Gets or sets the autoCalcDepthBounds property.
     *
     * When enabled, a depth rendering pass is first performed (with an internally created depth renderer or with the one
     * you provide by calling setDepthRenderer). Then, a min/max reducing is applied on the depth map to compute the
     * minimal and maximal depth of the map and those values are used as inputs for the setMinMaxDistance() function.
     * It can greatly enhance the shadow quality, at the expense of more GPU works.
     * When using this option, you should increase the value of the lambda parameter, and even set it to 1 for best results.
     */
    get autoCalcDepthBounds() {
        return this._autoCalcDepthBounds;
    }
    set autoCalcDepthBounds(value) {
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        this._autoCalcDepthBounds = value;
        if (!value) {
            if (this._depthReducer) {
                this._depthReducer.deactivate();
            }
            this.setMinMaxDistance(0, 1);
            return;
        }
        if (!this._depthReducer) {
            this._depthReducer = new DepthReducer(camera);
            this._depthReducer.onAfterReductionPerformed.add((minmax) => {
                let min = minmax.min, max = minmax.max;
                if (min >= max) {
                    min = 0;
                    max = 1;
                }
                if (min != this._minDistance || max != this._maxDistance) {
                    this.setMinMaxDistance(min, max);
                }
            });
            this._depthReducer.setDepthRenderer(this._depthRenderer);
        }
        this._depthReducer.activate();
    }
    /**
     * Defines the refresh rate of the min/max computation used when autoCalcDepthBounds is set to true
     * Use 0 to compute just once, 1 to compute on every frame, 2 to compute every two frames and so on...
     * Note that if you provided your own depth renderer through a call to setDepthRenderer, you are responsible
     * for setting the refresh rate on the renderer yourself!
     */
    get autoCalcDepthBoundsRefreshRate() {
        return this._depthReducer?.depthRenderer?.getDepthMap().refreshRate ?? -1;
    }
    set autoCalcDepthBoundsRefreshRate(value) {
        if (this._depthReducer?.depthRenderer) {
            this._depthReducer.depthRenderer.getDepthMap().refreshRate = value;
        }
    }
    /**
     * Create the cascade breaks according to the lambda, shadowMaxZ and min/max distance properties, as well as the camera near and far planes.
     * This function is automatically called when updating lambda, shadowMaxZ and min/max distances, however you should call it yourself if
     * you change the camera near/far planes!
     */
    splitFrustum() {
        this._breaksAreDirty = true;
    }
    _splitFrustum() {
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        const near = camera.minZ, far = camera.maxZ || this._shadowMaxZ, // account for infinite far plane (ie. maxZ = 0)
        cameraRange = far - near, minDistance = this._minDistance, maxDistance = this._shadowMaxZ < far && this._shadowMaxZ >= near ? Math.min((this._shadowMaxZ - near) / (far - near), this._maxDistance) : this._maxDistance;
        const minZ = near + minDistance * cameraRange, maxZ = near + maxDistance * cameraRange;
        const range = maxZ - minZ, ratio = maxZ / minZ;
        for (let cascadeIndex = 0; cascadeIndex < this._cascades.length; ++cascadeIndex) {
            const p = (cascadeIndex + 1) / this._numCascades, log = minZ * ratio ** p, uniform = minZ + range * p;
            const d = this._lambda * (log - uniform) + uniform;
            this._cascades[cascadeIndex].prevBreakDistance = cascadeIndex === 0 ? minDistance : this._cascades[cascadeIndex - 1].breakDistance;
            this._cascades[cascadeIndex].breakDistance = (d - near) / cameraRange;
            this._viewSpaceFrustumsZ[cascadeIndex] = d;
            this._frustumLengths[cascadeIndex] = (this._cascades[cascadeIndex].breakDistance - this._cascades[cascadeIndex].prevBreakDistance) * cameraRange;
        }
        this._breaksAreDirty = false;
    }
    _computeMatrices() {
        const scene = this._scene;
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        Vector3.NormalizeToRef(this._light.getShadowDirection(0), this._lightDirection);
        if (Math.abs(Vector3.Dot(this._lightDirection, Vector3.Up())) === 1.0) {
            this._lightDirection.z = 0.0000000000001; // Required to avoid perfectly perpendicular light
        }
        this._cachedDirection.copyFrom(this._lightDirection);
        const useReverseDepthBuffer = scene.getEngine().useReverseDepthBuffer;
        for (let cascadeIndex = 0; cascadeIndex < this._numCascades; ++cascadeIndex) {
            this._computeFrustumInWorldSpace(cascadeIndex);
            this._computeCascadeFrustum(cascadeIndex);
            this._cascadeMaxExtents[cascadeIndex].subtractToRef(this._cascadeMinExtents[cascadeIndex], Tmpv1); // tmpv1 = cascadeExtents
            // Get position of the shadow camera
            this._frustumCenter[cascadeIndex].addToRef(this._lightDirection.scale(this._cascadeMinExtents[cascadeIndex].z), this._shadowCameraPos[cascadeIndex]);
            // Come up with a new orthographic camera for the shadow caster
            Matrix.LookAtLHToRef(this._shadowCameraPos[cascadeIndex], this._frustumCenter[cascadeIndex], UpDir, this._viewMatrices[cascadeIndex]);
            // Z extents of the current cascade, in cascade view coordinate system
            let viewMinZ = 0, viewMaxZ = Tmpv1.z;
            // Try to tighten minZ and maxZ based on the bounding box of the shadow casters
            const boundingInfo = this._shadowCastersBoundingInfo;
            boundingInfo.update(this._viewMatrices[cascadeIndex]);
            // Note that after the call to update, the boundingInfo properties that are identified as "world" coordinates are in fact view coordinates for the current cascade!
            // This is because the boundingInfo properties that are identifed as "local" are in fact world coordinates (see _computeShadowCastersBoundingInfo()), and we multiply them by the current cascade view matrix when we call update.
            const castersViewMinZ = boundingInfo.boundingBox.minimumWorld.z;
            const castersViewMaxZ = boundingInfo.boundingBox.maximumWorld.z;
            if (castersViewMinZ > viewMaxZ) ;
            else {
                if (!this._depthClamp || this.filter === ShadowGenerator.FILTER_PCSS) {
                    // If we don't use depth clamping, we must define minZ so that all shadow casters are in the cascade frustum
                    viewMinZ = Math.min(viewMinZ, castersViewMinZ);
                    if (this.filter !== ShadowGenerator.FILTER_PCSS) {
                        // We do not need the actual distance between the currently shaded pixel and the occluder when generating shadows, so we can lower the far plane to increase the accuracy of the shadow map.
                        viewMaxZ = Math.min(viewMaxZ, castersViewMaxZ);
                    }
                }
                else {
                    // If we use depth clamping (but not PCSS!), we can adjust minZ/maxZ to reduce the range [minZ, maxZ] (and obtain additional precision in the shadow map)
                    viewMaxZ = Math.min(viewMaxZ, castersViewMaxZ);
                    // Thanks to depth clamping, casters won't be Z clipped even if they fall outside the [-1,1] range, so we can move the near plane to 0 if castersViewMinZ < 0.
                    // We will generate negative Z values in the shadow map, but that's okay (they will be clamped to the 0..1 range anyway), except in PCSS case
                    // where we need the actual distance between the currently shader pixel and the occluder: that's why we don't use depth clamping in PCSS case.
                    viewMinZ = Math.max(viewMinZ, castersViewMinZ);
                    // If all the casters are behind the near plane of the cascade, minZ = 0 due to the previous line, and maxZ < 0 at this point.
                    // We need to make sure that maxZ > minZ, so in this case we set maxZ a little higher than minZ. As we are using depth clamping, the casters won't be Z clipped, so we just need to make sure that we have a valid Z range for the cascade.
                    // Having a 0 range is not ok, due to undefined behavior in the calculation in this case.
                    viewMaxZ = Math.max(viewMinZ + 1.0, viewMaxZ);
                }
            }
            Matrix.OrthoOffCenterLHToRef(this._cascadeMinExtents[cascadeIndex].x, this._cascadeMaxExtents[cascadeIndex].x, this._cascadeMinExtents[cascadeIndex].y, this._cascadeMaxExtents[cascadeIndex].y, useReverseDepthBuffer ? viewMaxZ : viewMinZ, useReverseDepthBuffer ? viewMinZ : viewMaxZ, this._projectionMatrices[cascadeIndex], scene.getEngine().isNDCHalfZRange);
            this._cascadeMinExtents[cascadeIndex].z = viewMinZ;
            this._cascadeMaxExtents[cascadeIndex].z = viewMaxZ;
            this._viewMatrices[cascadeIndex].multiplyToRef(this._projectionMatrices[cascadeIndex], this._transformMatrices[cascadeIndex]);
            // Create the rounding matrix, by projecting the world-space origin and determining
            // the fractional offset in texel space
            Vector3.TransformCoordinatesToRef(ZeroVec, this._transformMatrices[cascadeIndex], Tmpv1); // tmpv1 = shadowOrigin
            Tmpv1.scaleInPlace(this._mapSize / 2);
            Tmpv2.copyFromFloats(Math.round(Tmpv1.x), Math.round(Tmpv1.y), Math.round(Tmpv1.z)); // tmpv2 = roundedOrigin
            Tmpv2.subtractInPlace(Tmpv1).scaleInPlace(2 / this._mapSize); // tmpv2 = roundOffset
            Matrix.TranslationToRef(Tmpv2.x, Tmpv2.y, 0.0, TmpMatrix);
            this._projectionMatrices[cascadeIndex].multiplyToRef(TmpMatrix, this._projectionMatrices[cascadeIndex]);
            this._viewMatrices[cascadeIndex].multiplyToRef(this._projectionMatrices[cascadeIndex], this._transformMatrices[cascadeIndex]);
            this._transformMatrices[cascadeIndex].copyToArray(this._transformMatricesAsArray, cascadeIndex * 16);
        }
    }
    // Get the 8 points of the view frustum in world space
    _computeFrustumInWorldSpace(cascadeIndex) {
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        const prevSplitDist = this._cascades[cascadeIndex].prevBreakDistance, splitDist = this._cascades[cascadeIndex].breakDistance;
        const isNDCHalfZRange = this._scene.getEngine().isNDCHalfZRange;
        camera.getViewMatrix(); // make sure the transformation matrix we get when calling 'getTransformationMatrix()' is calculated with an up to date view matrix
        const cameraInfiniteFarPlane = camera.maxZ === 0;
        const saveCameraMaxZ = camera.maxZ;
        if (cameraInfiniteFarPlane) {
            camera.maxZ = this._shadowMaxZ;
            camera.getProjectionMatrix(true);
        }
        const invViewProj = Matrix.Invert(camera.getTransformationMatrix());
        if (cameraInfiniteFarPlane) {
            camera.maxZ = saveCameraMaxZ;
            camera.getProjectionMatrix(true);
        }
        const cornerIndexOffset = this._scene.getEngine().useReverseDepthBuffer ? 4 : 0;
        for (let cornerIndex = 0; cornerIndex < CascadedShadowGenerator._FrustumCornersNdcSpace.length; ++cornerIndex) {
            Tmpv1.copyFrom(CascadedShadowGenerator._FrustumCornersNdcSpace[(cornerIndex + cornerIndexOffset) % CascadedShadowGenerator._FrustumCornersNdcSpace.length]);
            if (isNDCHalfZRange && Tmpv1.z === -1) {
                Tmpv1.z = 0;
            }
            Vector3.TransformCoordinatesToRef(Tmpv1, invViewProj, this._frustumCornersWorldSpace[cascadeIndex][cornerIndex]);
        }
        // Get the corners of the current cascade slice of the view frustum
        for (let cornerIndex = 0; cornerIndex < CascadedShadowGenerator._FrustumCornersNdcSpace.length / 2; ++cornerIndex) {
            Tmpv1.copyFrom(this._frustumCornersWorldSpace[cascadeIndex][cornerIndex + 4]).subtractInPlace(this._frustumCornersWorldSpace[cascadeIndex][cornerIndex]);
            Tmpv2.copyFrom(Tmpv1).scaleInPlace(prevSplitDist); // near corner ray
            Tmpv1.scaleInPlace(splitDist); // far corner ray
            Tmpv1.addInPlace(this._frustumCornersWorldSpace[cascadeIndex][cornerIndex]);
            this._frustumCornersWorldSpace[cascadeIndex][cornerIndex + 4].copyFrom(Tmpv1);
            this._frustumCornersWorldSpace[cascadeIndex][cornerIndex].addInPlace(Tmpv2);
        }
    }
    _computeCascadeFrustum(cascadeIndex) {
        this._cascadeMinExtents[cascadeIndex].copyFromFloats(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
        this._cascadeMaxExtents[cascadeIndex].copyFromFloats(-Number.MAX_VALUE, -Number.MAX_VALUE, -Number.MAX_VALUE);
        this._frustumCenter[cascadeIndex].copyFromFloats(0, 0, 0);
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        // Calculate the centroid of the view frustum slice
        for (let cornerIndex = 0; cornerIndex < this._frustumCornersWorldSpace[cascadeIndex].length; ++cornerIndex) {
            this._frustumCenter[cascadeIndex].addInPlace(this._frustumCornersWorldSpace[cascadeIndex][cornerIndex]);
        }
        this._frustumCenter[cascadeIndex].scaleInPlace(1 / this._frustumCornersWorldSpace[cascadeIndex].length);
        if (this.stabilizeCascades) {
            // Calculate the radius of a bounding sphere surrounding the frustum corners
            let sphereRadius = 0;
            for (let cornerIndex = 0; cornerIndex < this._frustumCornersWorldSpace[cascadeIndex].length; ++cornerIndex) {
                const dist = this._frustumCornersWorldSpace[cascadeIndex][cornerIndex].subtractToRef(this._frustumCenter[cascadeIndex], Tmpv1).length();
                sphereRadius = Math.max(sphereRadius, dist);
            }
            sphereRadius = Math.ceil(sphereRadius * 16) / 16;
            this._cascadeMaxExtents[cascadeIndex].copyFromFloats(sphereRadius, sphereRadius, sphereRadius);
            this._cascadeMinExtents[cascadeIndex].copyFromFloats(-sphereRadius, -sphereRadius, -sphereRadius);
        }
        else {
            // Create a temporary view matrix for the light
            const lightCameraPos = this._frustumCenter[cascadeIndex];
            this._frustumCenter[cascadeIndex].addToRef(this._lightDirection, Tmpv1); // tmpv1 = look at
            Matrix.LookAtLHToRef(lightCameraPos, Tmpv1, UpDir, TmpMatrix); // matrix = lightView
            // Calculate an AABB around the frustum corners
            for (let cornerIndex = 0; cornerIndex < this._frustumCornersWorldSpace[cascadeIndex].length; ++cornerIndex) {
                Vector3.TransformCoordinatesToRef(this._frustumCornersWorldSpace[cascadeIndex][cornerIndex], TmpMatrix, Tmpv1);
                this._cascadeMinExtents[cascadeIndex].minimizeInPlace(Tmpv1);
                this._cascadeMaxExtents[cascadeIndex].maximizeInPlace(Tmpv1);
            }
        }
    }
    _recreateSceneUBOs() {
        this._disposeSceneUBOs();
        if (this._sceneUBOs) {
            for (let i = 0; i < this._numCascades; ++i) {
                this._sceneUBOs.push(this._scene.createSceneUniformBuffer(`Scene for CSM Shadow Generator (light "${this._light.name}" cascade #${i})`));
            }
        }
    }
    /**
     *  Support test.
     */
    static get IsSupported() {
        const engine = EngineStore.LastCreatedEngine;
        if (!engine) {
            return false;
        }
        return engine._features.supportCSM;
    }
    /**
     * Creates a Cascaded Shadow Generator object.
     * A ShadowGenerator is the required tool to use the shadows.
     * Each directional light casting shadows needs to use its own ShadowGenerator.
     * Documentation : https://doc.babylonjs.com/babylon101/cascadedShadows
     * @param mapSize The size of the texture what stores the shadows. Example : 1024.
     * @param light The directional light object generating the shadows.
     * @param usefulFloatFirst By default the generator will try to use half float textures but if you need precision (for self shadowing for instance), you can use this option to enforce full float texture.
     * @param camera Camera associated with this shadow generator (default: null). If null, takes the scene active camera at the time we need to access it
     * @param useRedTextureType Forces the generator to use a Red instead of a RGBA type for the shadow map texture format (default: true)
     */
    constructor(mapSize, light, usefulFloatFirst, camera, useRedTextureType = true) {
        if (!CascadedShadowGenerator.IsSupported) {
            Logger.Error("CascadedShadowMap is not supported by the current engine.");
            return;
        }
        super(mapSize, light, usefulFloatFirst, camera, useRedTextureType);
        this.usePercentageCloserFiltering = true;
    }
    _initializeGenerator() {
        this.penumbraDarkness = this.penumbraDarkness ?? 1.0;
        this._numCascades = this._numCascades ?? CascadedShadowGenerator.DEFAULT_CASCADES_COUNT;
        this.stabilizeCascades = this.stabilizeCascades ?? false;
        this._freezeShadowCastersBoundingInfoObservable = this._freezeShadowCastersBoundingInfoObservable ?? null;
        this.freezeShadowCastersBoundingInfo = this.freezeShadowCastersBoundingInfo ?? false;
        this._scbiMin = this._scbiMin ?? new Vector3(0, 0, 0);
        this._scbiMax = this._scbiMax ?? new Vector3(0, 0, 0);
        this._shadowCastersBoundingInfo = this._shadowCastersBoundingInfo ?? new BoundingInfo(new Vector3(0, 0, 0), new Vector3(0, 0, 0));
        this._breaksAreDirty = this._breaksAreDirty ?? true;
        this._minDistance = this._minDistance ?? 0;
        this._maxDistance = this._maxDistance ?? 1;
        this._currentLayer = this._currentLayer ?? 0;
        this._shadowMaxZ = this._shadowMaxZ ?? this._getCamera()?.maxZ ?? 10000;
        this._debug = this._debug ?? false;
        this._depthClamp = this._depthClamp ?? true;
        this._cascadeBlendPercentage = this._cascadeBlendPercentage ?? 0.1;
        this._lambda = this._lambda ?? 0.5;
        this._autoCalcDepthBounds = this._autoCalcDepthBounds ?? false;
        this._recreateSceneUBOs();
        super._initializeGenerator();
    }
    _createTargetRenderTexture() {
        const engine = this._scene.getEngine();
        this._shadowMap?.dispose();
        const size = { width: this._mapSize, height: this._mapSize, layers: this.numCascades };
        this._shadowMap = new RenderTargetTexture(this._light.name + "_CSMShadowMap", size, this._scene, false, true, this._textureType, false, undefined, false, false, undefined, this._useRedTextureType ? 6 : 5);
        this._shadowMap.createDepthStencilTexture(engine.useReverseDepthBuffer ? 516 : 513, true, undefined, undefined, undefined, `DepthStencilForCSMShadowGenerator-${this._light.name}`);
        this._shadowMap.noPrePassRenderer = true;
    }
    _initializeShadowMap() {
        super._initializeShadowMap();
        if (this._shadowMap === null) {
            return;
        }
        this._transformMatricesAsArray = new Float32Array(this._numCascades * 16);
        this._tempTransformMatricesAsArray = new Float32Array(this._numCascades * 16);
        this._viewSpaceFrustumsZ = new Array(this._numCascades);
        this._frustumLengths = new Array(this._numCascades);
        this._lightSizeUVCorrection = new Array(this._numCascades * 2);
        this._depthCorrection = new Array(this._numCascades);
        this._cascades = [];
        this._viewMatrices = [];
        this._projectionMatrices = [];
        this._transformMatrices = [];
        this._cascadeMinExtents = [];
        this._cascadeMaxExtents = [];
        this._frustumCenter = [];
        this._shadowCameraPos = [];
        this._frustumCornersWorldSpace = [];
        for (let cascadeIndex = 0; cascadeIndex < this._numCascades; ++cascadeIndex) {
            this._cascades[cascadeIndex] = {
                prevBreakDistance: 0,
                breakDistance: 0,
            };
            this._viewMatrices[cascadeIndex] = Matrix.Zero();
            this._projectionMatrices[cascadeIndex] = Matrix.Zero();
            this._transformMatrices[cascadeIndex] = Matrix.Zero();
            this._cascadeMinExtents[cascadeIndex] = new Vector3();
            this._cascadeMaxExtents[cascadeIndex] = new Vector3();
            this._frustumCenter[cascadeIndex] = new Vector3();
            this._shadowCameraPos[cascadeIndex] = new Vector3();
            this._frustumCornersWorldSpace[cascadeIndex] = new Array(CascadedShadowGenerator._FrustumCornersNdcSpace.length);
            for (let i = 0; i < CascadedShadowGenerator._FrustumCornersNdcSpace.length; ++i) {
                this._frustumCornersWorldSpace[cascadeIndex][i] = new Vector3();
            }
        }
        const engine = this._scene.getEngine();
        this._shadowMap.onBeforeBindObservable.clear();
        this._shadowMap.onBeforeRenderObservable.clear();
        this._shadowMap.onBeforeRenderObservable.add((layer) => {
            if (this._sceneUBOs) {
                this._scene.setSceneUniformBuffer(this._sceneUBOs[layer]);
            }
            this._currentLayer = layer;
            if (this._filter === ShadowGenerator.FILTER_PCF) {
                engine.setColorWrite(false);
            }
            FloatingOriginCurrentScene.eyeAtCamera = false;
            this._scene.setTransformMatrix(this.getCascadeViewMatrix(layer), this.getCascadeProjectionMatrix(layer));
            if (this._useUBO) {
                this._scene.getSceneUniformBuffer().unbindEffect();
                this._scene.finalizeSceneUbo();
            }
        });
        this._shadowMap.onBeforeBindObservable.add(() => {
            this._currentSceneUBO = this._scene.getSceneUniformBuffer();
            if (engine._enableGPUDebugMarkers) {
                engine.restoreDefaultFramebuffer();
                engine._debugPushGroup?.(`Cascaded shadow map generation for pass id ${engine.currentRenderPassId}`);
            }
            if (this._breaksAreDirty) {
                this._splitFrustum();
            }
            this._computeMatrices();
        });
        this._splitFrustum();
    }
    _bindCustomEffectForRenderSubMeshForShadowMap(subMesh, effect) {
        effect.setMatrix("viewProjection", this.getCascadeTransformMatrix(this._currentLayer));
    }
    _isReadyCustomDefines(defines) {
        defines.push("#define SM_DEPTHCLAMP " + (this._depthClamp && this._filter !== ShadowGenerator.FILTER_PCSS ? "1" : "0"));
    }
    /**
     * Prepare all the defines in a material relying on a shadow map at the specified light index.
     * @param defines Defines of the material we want to update
     * @param lightIndex Index of the light in the enabled light list of the material
     */
    prepareDefines(defines, lightIndex) {
        super.prepareDefines(defines, lightIndex);
        const scene = this._scene;
        const light = this._light;
        if (!scene.shadowsEnabled || !light.shadowEnabled) {
            return;
        }
        defines["SHADOWCSM" + lightIndex] = true;
        defines["SHADOWCSMDEBUG" + lightIndex] = this.debug;
        defines["SHADOWCSMNUM_CASCADES" + lightIndex] = this.numCascades;
        defines["SHADOWCSM_RIGHTHANDED" + lightIndex] = scene.useRightHandedSystem;
        const camera = this._getCamera();
        if (camera && this._shadowMaxZ <= (camera.maxZ || this._shadowMaxZ)) {
            defines["SHADOWCSMUSESHADOWMAXZ" + lightIndex] = true;
        }
        if (this.cascadeBlendPercentage === 0) {
            defines["SHADOWCSMNOBLEND" + lightIndex] = true;
        }
    }
    /**
     * Binds the shadow related information inside of an effect (information like near, far, darkness...
     * defined in the generator but impacting the effect).
     * @param lightIndex Index of the light in the enabled light list of the material owning the effect
     * @param effect The effect we are binfing the information for
     */
    bindShadowLight(lightIndex, effect) {
        const light = this._light;
        const scene = this._scene;
        if (!scene.shadowsEnabled || !light.shadowEnabled) {
            return;
        }
        const camera = this._getCamera();
        if (!camera) {
            return;
        }
        const shadowMap = this.getShadowMap();
        if (!shadowMap) {
            return;
        }
        const width = shadowMap.getSize().width;
        const transform = this._transformMatricesAsArray;
        // Doing the check for floatingOrigin here to avoid unnecessary matrix operations when offset is 0
        const lightMatrix = scene.floatingOriginMode
            ? GetOffsetTransformMatrices(this._scene.floatingOriginOffset, this._viewMatrices, this._projectionMatrices, this._numCascades, this._tempTransformMatricesAsArray)
            : transform;
        effect.setMatrices("lightMatrix" + lightIndex, lightMatrix);
        effect.setArray("viewFrustumZ" + lightIndex, this._viewSpaceFrustumsZ);
        effect.setFloat("cascadeBlendFactor" + lightIndex, this.cascadeBlendPercentage === 0 ? 10000 : 1 / this.cascadeBlendPercentage);
        effect.setArray("frustumLengths" + lightIndex, this._frustumLengths);
        // Only PCF uses depth stencil texture.
        if (this._filter === ShadowGenerator.FILTER_PCF) {
            effect.setDepthStencilTexture("shadowTexture" + lightIndex, shadowMap);
            light._uniformBuffer.updateFloat4("shadowsInfo", this.getDarkness(), width, 1 / width, this.frustumEdgeFalloff, lightIndex);
        }
        else if (this._filter === ShadowGenerator.FILTER_PCSS) {
            for (let cascadeIndex = 0; cascadeIndex < this._numCascades; ++cascadeIndex) {
                this._lightSizeUVCorrection[cascadeIndex * 2 + 0] =
                    cascadeIndex === 0
                        ? 1
                        : (this._cascadeMaxExtents[0].x - this._cascadeMinExtents[0].x) / (this._cascadeMaxExtents[cascadeIndex].x - this._cascadeMinExtents[cascadeIndex].x); // x correction
                this._lightSizeUVCorrection[cascadeIndex * 2 + 1] =
                    cascadeIndex === 0
                        ? 1
                        : (this._cascadeMaxExtents[0].y - this._cascadeMinExtents[0].y) / (this._cascadeMaxExtents[cascadeIndex].y - this._cascadeMinExtents[cascadeIndex].y); // y correction
                this._depthCorrection[cascadeIndex] =
                    cascadeIndex === 0
                        ? 1
                        : (this._cascadeMaxExtents[cascadeIndex].z - this._cascadeMinExtents[cascadeIndex].z) / (this._cascadeMaxExtents[0].z - this._cascadeMinExtents[0].z);
            }
            effect.setDepthStencilTexture("shadowTexture" + lightIndex, shadowMap);
            effect.setTexture("depthTexture" + lightIndex, shadowMap);
            effect.setArray2("lightSizeUVCorrection" + lightIndex, this._lightSizeUVCorrection);
            effect.setArray("depthCorrection" + lightIndex, this._depthCorrection);
            effect.setFloat("penumbraDarkness" + lightIndex, this.penumbraDarkness);
            light._uniformBuffer.updateFloat4("shadowsInfo", this.getDarkness(), 1 / width, this._contactHardeningLightSizeUVRatio * width, this.frustumEdgeFalloff, lightIndex);
        }
        else {
            effect.setTexture("shadowTexture" + lightIndex, shadowMap);
            light._uniformBuffer.updateFloat4("shadowsInfo", this.getDarkness(), width, 1 / width, this.frustumEdgeFalloff, lightIndex);
        }
        light._uniformBuffer.updateFloat2("depthValues", this.getLight().getDepthMinZ(camera), this.getLight().getDepthMinZ(camera) + this.getLight().getDepthMaxZ(camera), lightIndex);
    }
    /**
     * Gets the transformation matrix of the first cascade used to project the meshes into the map from the light point of view.
     * (eq to view projection * shadow projection matrices)
     * @returns The transform matrix used to create the shadow map
     */
    getTransformMatrix() {
        return this.getCascadeTransformMatrix(0);
    }
    /**
     * Disposes the ShadowGenerator.
     * Returns nothing.
     */
    dispose() {
        super.dispose();
        if (this._freezeShadowCastersBoundingInfoObservable) {
            this._scene.onBeforeRenderObservable.remove(this._freezeShadowCastersBoundingInfoObservable);
            this._freezeShadowCastersBoundingInfoObservable = null;
        }
        if (this._depthReducer) {
            this._depthReducer.dispose();
            this._depthReducer = null;
        }
    }
    /**
     * Serializes the shadow generator setup to a json object.
     * @returns The serialized JSON object
     */
    serialize() {
        const serializationObject = super.serialize();
        const shadowMap = this.getShadowMap();
        if (!shadowMap) {
            return serializationObject;
        }
        serializationObject.numCascades = this._numCascades;
        serializationObject.debug = this._debug;
        serializationObject.stabilizeCascades = this.stabilizeCascades;
        serializationObject.lambda = this._lambda;
        serializationObject.cascadeBlendPercentage = this.cascadeBlendPercentage;
        serializationObject.depthClamp = this._depthClamp;
        serializationObject.autoCalcDepthBounds = this.autoCalcDepthBounds;
        serializationObject.shadowMaxZ = this._shadowMaxZ;
        serializationObject.penumbraDarkness = this.penumbraDarkness;
        serializationObject.freezeShadowCastersBoundingInfo = this._freezeShadowCastersBoundingInfo;
        serializationObject.minDistance = this.minDistance;
        serializationObject.maxDistance = this.maxDistance;
        serializationObject.renderList = [];
        if (shadowMap.renderList) {
            for (let meshIndex = 0; meshIndex < shadowMap.renderList.length; meshIndex++) {
                const mesh = shadowMap.renderList[meshIndex];
                serializationObject.renderList.push(mesh.id);
            }
        }
        return serializationObject;
    }
    /**
     * Parses a serialized ShadowGenerator and returns a new ShadowGenerator.
     * @param parsedShadowGenerator The JSON object to parse
     * @param scene The scene to create the shadow map for
     * @returns The parsed shadow generator
     */
    static Parse(parsedShadowGenerator, scene) {
        const shadowGenerator = ShadowGenerator.Parse(parsedShadowGenerator, scene, (mapSize, light, camera) => new CascadedShadowGenerator(mapSize, light, undefined, camera));
        if (parsedShadowGenerator.numCascades !== undefined) {
            shadowGenerator.numCascades = parsedShadowGenerator.numCascades;
        }
        if (parsedShadowGenerator.debug !== undefined) {
            shadowGenerator.debug = parsedShadowGenerator.debug;
        }
        if (parsedShadowGenerator.stabilizeCascades !== undefined) {
            shadowGenerator.stabilizeCascades = parsedShadowGenerator.stabilizeCascades;
        }
        if (parsedShadowGenerator.lambda !== undefined) {
            shadowGenerator.lambda = parsedShadowGenerator.lambda;
        }
        if (parsedShadowGenerator.cascadeBlendPercentage !== undefined) {
            shadowGenerator.cascadeBlendPercentage = parsedShadowGenerator.cascadeBlendPercentage;
        }
        if (parsedShadowGenerator.depthClamp !== undefined) {
            shadowGenerator.depthClamp = parsedShadowGenerator.depthClamp;
        }
        if (parsedShadowGenerator.autoCalcDepthBounds !== undefined) {
            shadowGenerator.autoCalcDepthBounds = parsedShadowGenerator.autoCalcDepthBounds;
        }
        if (parsedShadowGenerator.shadowMaxZ !== undefined) {
            shadowGenerator.shadowMaxZ = parsedShadowGenerator.shadowMaxZ;
        }
        if (parsedShadowGenerator.penumbraDarkness !== undefined) {
            shadowGenerator.penumbraDarkness = parsedShadowGenerator.penumbraDarkness;
        }
        if (parsedShadowGenerator.freezeShadowCastersBoundingInfo !== undefined) {
            shadowGenerator.freezeShadowCastersBoundingInfo = parsedShadowGenerator.freezeShadowCastersBoundingInfo;
        }
        if (parsedShadowGenerator.minDistance !== undefined && parsedShadowGenerator.maxDistance !== undefined) {
            shadowGenerator.setMinMaxDistance(parsedShadowGenerator.minDistance, parsedShadowGenerator.maxDistance);
        }
        return shadowGenerator;
    }
}
CascadedShadowGenerator._FrustumCornersNdcSpace = [
    new Vector3(-1, 1, -1),
    new Vector3(1, 1, -1),
    new Vector3(1, -1, -1),
    new Vector3(-1, -1, -1),
    new Vector3(-1, 1, 1),
    new Vector3(1, 1, 1),
    new Vector3(1, -1, 1),
    new Vector3(-1, -1, 1),
];
/**
 * Name of the CSM class
 */
CascadedShadowGenerator.CLASSNAME = "CascadedShadowGenerator";
/**
 * Defines the default number of cascades used by the CSM.
 */
CascadedShadowGenerator.DEFAULT_CASCADES_COUNT = 4;
/**
 * Defines the minimum number of cascades used by the CSM.
 */
CascadedShadowGenerator.MIN_CASCADES_COUNT = 2;
/**
 * Defines the maximum number of cascades used by the CSM.
 */
CascadedShadowGenerator.MAX_CASCADES_COUNT = 4;
/**
 * @internal
 */
CascadedShadowGenerator._SceneComponentInitialization = (_) => {
    throw _WarnImport("ShadowGeneratorSceneComponent");
};

/**
 * Defines the shadow generator component responsible to manage any shadow generators
 * in a given scene.
 */
class ShadowGeneratorSceneComponent {
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_SHADOWGENERATOR;
        this.scene = scene;
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._gatherRenderTargetsStage.registerStep(SceneComponentConstants.STEP_GATHERRENDERTARGETS_SHADOWGENERATOR, this, this._gatherRenderTargets);
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        // Nothing To Do Here.
    }
    /**
     * Serializes the component data to the specified json object
     * @param serializationObject The object to serialize to
     */
    serialize(serializationObject) {
        // Shadows
        serializationObject.shadowGenerators = [];
        const lights = this.scene.lights;
        for (const light of lights) {
            if (light.doNotSerialize) {
                continue;
            }
            const shadowGenerators = light.getShadowGenerators();
            if (shadowGenerators) {
                const iterator = shadowGenerators.values();
                for (let key = iterator.next(); key.done !== true; key = iterator.next()) {
                    const shadowGenerator = key.value;
                    if (shadowGenerator.doNotSerialize) {
                        continue;
                    }
                    serializationObject.shadowGenerators.push(shadowGenerator.serialize());
                }
            }
        }
    }
    /**
     * Adds all the elements from the container to the scene
     * @param container the container holding the elements
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    addFromContainer(container) {
        // Nothing To Do Here. (directly attached to a light)
    }
    /**
     * Removes all the elements in the container from the scene
     * @param container contains the elements to remove
     * @param dispose if the removed element should be disposed (default: false)
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    removeFromContainer(container, dispose) {
        // Nothing To Do Here. (directly attached to a light)
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    dispose() {
        // Nothing To Do Here.
    }
    _gatherRenderTargets(renderTargets) {
        // Shadows
        const scene = this.scene;
        if (this.scene.shadowsEnabled) {
            for (let lightIndex = 0; lightIndex < scene.lights.length; lightIndex++) {
                const light = scene.lights[lightIndex];
                const shadowGenerators = light.getShadowGenerators();
                if (light.isEnabled() && light.shadowEnabled && shadowGenerators) {
                    const iterator = shadowGenerators.values();
                    for (let key = iterator.next(); key.done !== true; key = iterator.next()) {
                        const shadowGenerator = key.value;
                        const shadowMap = shadowGenerator.getShadowMap();
                        if (scene.textures.indexOf(shadowMap) !== -1) {
                            renderTargets.push(shadowMap);
                        }
                    }
                }
            }
        }
    }
}
ShadowGenerator._SceneComponentInitialization = (scene) => {
    let component = scene._getComponent(SceneComponentConstants.NAME_SHADOWGENERATOR);
    if (!component) {
        component = new ShadowGeneratorSceneComponent(scene);
        scene._addComponent(component);
    }
};

export { ShadowGeneratorSceneComponent };
//# sourceMappingURL=shadowGeneratorSceneComponent-CV-JGuYH.js.map
