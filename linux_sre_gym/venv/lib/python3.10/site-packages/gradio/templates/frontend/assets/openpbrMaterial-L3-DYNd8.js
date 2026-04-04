const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./openpbr.vertex--b02pul6.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./openpbrUboDeclaration-C1J5ZbPU.js","./sceneUboDeclaration-BjEv3Rnv.js","./meshUboDeclaration-CkiQ-QPA.js","./vertexColorMixing-DjVvSOV4.js","./mainUVVaryingDeclaration-DxK7QVw_.js","./helperFunctions-qLR1vvLh.js","./pbrBRDFFunctions-Ezjx-oio.js","./bonesDeclaration-BSA0gH1L.js","./instancesVertex-o60Do33e.js","./instancesDeclaration-BUZFwkeM.js","./harmonicsFunctions--6SqClog.js","./clipPlaneVertex-BeUdHTek.js","./logDepthVertex-CB3XkllG.js","./shadowsVertex-DKFAYpnG.js","./morphTargetsVertexDeclaration-BLOFnJa5.js","./logDepthDeclaration-BkiOZiuv.js","./morphTargetsVertex-DIZBNgIv.js","./bonesVertex-b_UrNN_W.js","./bakedVertexAnimation-CnIdycvp.js","./openpbr.fragment-CLj6ryVP.js","./oitFragment-JA55XGuc.js","./pbrDebug-CNjFg1pA.js","./shadowsFragmentFunctions-C5wXieKk.js","./samplerFragmentDeclaration-DMju9ewr.js","./imageProcessingFunctions-ClbSBfeF.js","./clipPlaneFragment-BegocNAa.js","./fogFragment-DbmcEy2X.js","./hdrFilteringFunctions-DQUZ_3Oj.js","./openpbr.vertex-DK9C96sM.js","./vertexColorMixing-paYZ10MH.js","./openpbrUboDeclaration-C8whNoom.js","./sceneUboDeclaration-CbezEhV0.js","./meshUboDeclaration-zsmLsV73.js","./mainUVVaryingDeclaration-Y0U6swm8.js","./helperFunctions-CiFVmLcr.js","./pbrBRDFFunctions-Cr0QY02Q.js","./bakedVertexAnimation-HQ8JUcrA.js","./instancesDeclaration-Bv0KzShp.js","./harmonicsFunctions-CwF9fsaj.js","./clipPlaneVertex-BKsOyoIZ.js","./logDepthVertex-DP62dfil.js","./shadowsVertex-BM51klV9.js","./morphTargetsVertex-C9IVzlEb.js","./logDepthDeclaration-DzD3v8nZ.js","./openpbr.fragment-CqiyhFmE.js","./oitFragment-mEmMQoBe.js","./pbrDebug-Bi0uJINp.js","./shadowsFragmentFunctions-C6EkE8iW.js","./bumpFragment-Dxop61vv.js","./imageProcessingFunctions-CBu7O15Q.js","./clipPlaneFragment-Vm6wRK67.js","./fogFragment-Ek2G8c0v.js","./hdrFilteringFunctions-Cf_DM9A0.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { ah as Vector2, o as Material, C as Color3, ai as Vector4, n as SmartArray, L as Logger, T as Texture, d as Tools, a7 as MaterialFlags, aj as GetEnvironmentBRDFTexture, ak as GetEnvironmentFuzzBRDFTexture, P as PrePassConfiguration, e as SerializationHelper, J as VertexBuffer, Z as PrepareUniformLayoutForIBL, u as MaterialHelperGeometryRendering, $ as BindBonesParameters, a0 as BindTextureMatrix, a1 as BindIBLParameters, c as TmpVectors, al as BindIBLSamplers, a2 as BindClipPlane, a3 as BindLights, b as Scene, a4 as BindFogParameters, a5 as BindMorphTargetParameters, a6 as BindLogDepth, E as EffectFallbacks, H as HandleFallbacksForShadows, F as PrepareAttributesForBones, K as PrepareAttributesForInstances, N as PrepareAttributesForMorphTargets, Q as PrepareAttributesForBakedVertexAnimation, U as PrepareUniformsAndSamplersForIBL, Y as AddClipPlaneUniforms, W as ImageProcessingConfiguration, X as PrepareUniformsAndSamplersList, p as PrepareDefinesForLights, q as PrepareDefinesForMultiview, r as PrepareDefinesForPrePass, t as PrepareDefinesForOIT, v as PrepareDefinesForMergedUV, w as PrepareDefinesForIBL, x as PrepareDefinesForMisc, y as PrepareDefinesForFrameBoundValues, z as PrepareDefinesForAttributes, a8 as ImageProcessingMixin, a9 as PushMaterial, aa as ImageProcessingDefinesMixin, ab as UVDefinesMixin, ac as MaterialDefines, _ as __decorate, am as addAccessorsForMaterialProperty, s as serialize, ae as expandToProperty, R as RegisterClass } from './index-CxR0Yf_7.js';

const onCreatedEffectParameters = { effect: null, subMesh: null };
class Uniform {
    populateVectorFromLinkedProperties(vector) {
        const destinationSize = vector.dimension[0];
        for (const propKey in this.linkedProperties) {
            const prop = this.linkedProperties[propKey];
            const sourceSize = prop.numComponents;
            if (destinationSize < sourceSize || prop.targetUniformComponentOffset > destinationSize - sourceSize) {
                if (sourceSize == 1) {
                    Logger.Error(`Float property ${prop.name} has an offset that is too large.`);
                }
                else {
                    Logger.Error(`Vector${sourceSize} property ${prop.name} won't fit in Vector${destinationSize} or has an offset that is too large.`);
                }
                return;
            }
            if (typeof prop.value === "number") {
                Uniform._tmpArray[prop.targetUniformComponentOffset] = prop.value;
            }
            else {
                prop.value.toArray(Uniform._tmpArray, prop.targetUniformComponentOffset);
            }
        }
        vector.fromArray(Uniform._tmpArray);
    }
    constructor(name, componentNum) {
        this.linkedProperties = {};
        this.name = name;
        this.numComponents = componentNum;
    }
}
Uniform._tmpArray = [0, 0, 0, 0];
/**
 * Defines a property for the OpenPBRMaterial.
 */
class Property {
    /**
     * Creates a new Property instance.
     * @param name The name of the property in the shader
     * @param defaultValue The default value of the property
     * @param targetUniformName The name of the property in the shader uniform block
     * @param targetUniformComponentNum The number of components in the target uniform. All properties that are
     * packed into the same uniform must agree on the size of the target uniform.
     * @param targetUniformComponentOffset The offset in the uniform where this property will be packed.
     */
    constructor(name, defaultValue, targetUniformName, targetUniformComponentNum, targetUniformComponentOffset = 0) {
        // public includeAlphaFromProp: string = "";
        /**
         * If not given a type, there will be no uniform defined for this property and
         * it will be assumed that the value will be packed into the already existing "uniformName" uniform.
         */
        this.targetUniformComponentNum = 4; // Default to vec4
        this.targetUniformComponentOffset = 0;
        this.name = name;
        this.targetUniformName = targetUniformName;
        this.defaultValue = defaultValue;
        this.value = defaultValue;
        this.targetUniformComponentNum = targetUniformComponentNum;
        this.targetUniformComponentOffset = targetUniformComponentOffset;
    }
    /**
     * Returns the number of components of the property based on its default value type.
     */
    get numComponents() {
        if (typeof this.defaultValue === "number") {
            return 1;
        }
        return this.defaultValue.dimension[0];
    }
}
class Sampler {
    /**
     * The name of the sampler used in the shader.
     * If this naming changes, we'll also need to change:
     * - samplerFragmentDeclaration.fx
     * - openpbr.fragment.fx
     */
    get samplerName() {
        return this.samplerPrefix + "Sampler";
    }
    /**
     * The name of the sampler info used in the shader.
     * If this naming changes, we'll also need to change:
     * - openpbr.vertex.fx
     * - openpbr.fragment.fx
     */
    get samplerInfoName() {
        return "v" + this.samplerPrefix.charAt(0).toUpperCase() + this.samplerPrefix.slice(1) + "Infos";
    }
    /**
     * The name of the matrix used for this sampler in the shader.
     * If this naming changes, we'll also need to change:
     * - materialHelper.functions.BindTextureMatrix
     * - samplerVertexImplementation.fx
     * - openpbr.fragment.fx
     */
    get samplerMatrixName() {
        return this.samplerPrefix + "Matrix";
    }
    /**
     * Creates a new Sampler instance.
     * @param name The name of the texture property
     * @param samplerPrefix The prefix used for the name of the sampler in the shader
     * @param textureDefine The define used in the shader for this sampler
     */
    constructor(name, samplerPrefix, textureDefine) {
        this.value = null; // Texture value, default to null
        this.samplerPrefix = ""; // The name of the sampler in the shader
        this.textureDefine = ""; // The define used in the shader for this sampler
        this.name = name;
        this.samplerPrefix = samplerPrefix;
        this.textureDefine = textureDefine;
    }
}
class OpenPBRMaterialDefinesBase extends UVDefinesMixin(MaterialDefines) {
}
/**
 * Manages the defines for the PBR Material.
 * @internal
 */
class OpenPBRMaterialDefines extends ImageProcessingDefinesMixin(OpenPBRMaterialDefinesBase) {
    /**
     * Initializes the PBR Material defines.
     * @param externalProperties The external properties
     */
    constructor(externalProperties) {
        super(externalProperties);
        this.NUM_SAMPLES = "0";
        this.REALTIME_FILTERING = false;
        this.IBL_CDF_FILTERING = false;
        this.VERTEXCOLOR = false;
        this.BAKED_VERTEX_ANIMATION_TEXTURE = false;
        this.VERTEXALPHA = false;
        this.ALPHATEST = false;
        this.DEPTHPREPASS = false;
        this.ALPHABLEND = false;
        this.ALPHA_FROM_BASE_COLOR_TEXTURE = false;
        this.ALPHATESTVALUE = "0.5";
        this.PREMULTIPLYALPHA = false;
        this.REFLECTIVITY_GAMMA = false;
        this.REFLECTIVITYDIRECTUV = 0;
        this.SPECULARTERM = false;
        this.LODBASEDMICROSFURACE = true;
        this.METALLICWORKFLOW = true;
        this.ROUGHNESSSTOREINMETALMAPALPHA = false;
        this.ROUGHNESSSTOREINMETALMAPGREEN = false;
        this.METALLNESSSTOREINMETALMAPBLUE = false;
        this.AOSTOREINMETALMAPRED = false;
        this.SPECULAR_WEIGHT_IN_ALPHA = false;
        this.SPECULAR_WEIGHT_FROM_SPECULAR_COLOR_TEXTURE = false;
        this.SPECULAR_ROUGHNESS_ANISOTROPY_FROM_TANGENT_TEXTURE = false;
        this.COAT_ROUGHNESS_FROM_GREEN_CHANNEL = false;
        this.COAT_ROUGHNESS_ANISOTROPY_FROM_TANGENT_TEXTURE = false;
        this.USE_GLTF_STYLE_ANISOTROPY = false;
        this.THIN_FILM_THICKNESS_FROM_THIN_FILM_TEXTURE = false;
        this.FUZZ_ROUGHNESS_FROM_TEXTURE_ALPHA = false;
        this.ENVIRONMENTBRDF = false;
        this.ENVIRONMENTBRDF_RGBD = false;
        this.FUZZENVIRONMENTBRDF = false;
        this.NORMAL = false;
        this.TANGENT = false;
        this.OBJECTSPACE_NORMALMAP = false;
        this.PARALLAX = false;
        this.PARALLAX_RHS = false;
        this.PARALLAXOCCLUSION = false;
        this.NORMALXYSCALE = true;
        /**
         * Enables anisotropic logic. Still needed because it's used in pbrHelperFunctions
         */
        this.ANISOTROPIC = false;
        /**
         * Tells the shader to use OpenPBR's anisotropic roughness remapping
         */
        this.ANISOTROPIC_OPENPBR = true;
        /**
         * Tells the shader to apply anisotropy to the base layer
         */
        this.ANISOTROPIC_BASE = false;
        /**
         * Tells the shader to apply anisotropy to the coat layer
         */
        this.ANISOTROPIC_COAT = false;
        /**
         * Number of samples to use for the fuzz IBL lighting calculations
         */
        this.FUZZ_IBL_SAMPLES = 6;
        /**
         * Tells the shader to enable the fuzz layer
         */
        this.FUZZ = false;
        /**
         * Tells the shader to enable the thin film layer
         */
        this.THIN_FILM = false;
        /**
         * Tells the shader to enable the legacy iridescence code
         * Iridescence is the name of thin film interference in the PBR material.
         */
        this.IRIDESCENCE = false;
        this.REFLECTION = false;
        this.REFLECTIONMAP_3D = false;
        this.REFLECTIONMAP_SPHERICAL = false;
        this.REFLECTIONMAP_PLANAR = false;
        this.REFLECTIONMAP_CUBIC = false;
        this.USE_LOCAL_REFLECTIONMAP_CUBIC = false;
        this.REFLECTIONMAP_PROJECTION = false;
        this.REFLECTIONMAP_SKYBOX = false;
        this.REFLECTIONMAP_EXPLICIT = false;
        this.REFLECTIONMAP_EQUIRECTANGULAR = false;
        this.REFLECTIONMAP_EQUIRECTANGULAR_FIXED = false;
        this.REFLECTIONMAP_MIRROREDEQUIRECTANGULAR_FIXED = false;
        this.INVERTCUBICMAP = false;
        this.USESPHERICALFROMREFLECTIONMAP = false;
        this.USEIRRADIANCEMAP = false;
        this.USE_IRRADIANCE_DOMINANT_DIRECTION = false;
        this.USESPHERICALINVERTEX = false;
        this.REFLECTIONMAP_OPPOSITEZ = false;
        this.LODINREFLECTIONALPHA = false;
        this.GAMMAREFLECTION = false;
        this.RGBDREFLECTION = false;
        this.RADIANCEOCCLUSION = false;
        this.HORIZONOCCLUSION = false;
        this.INSTANCES = false;
        this.THIN_INSTANCES = false;
        this.INSTANCESCOLOR = false;
        this.PREPASS = false;
        this.PREPASS_COLOR = false;
        this.PREPASS_COLOR_INDEX = -1;
        this.PREPASS_IRRADIANCE = false;
        this.PREPASS_IRRADIANCE_INDEX = -1;
        this.PREPASS_ALBEDO = false;
        this.PREPASS_ALBEDO_INDEX = -1;
        this.PREPASS_ALBEDO_SQRT = false;
        this.PREPASS_ALBEDO_SQRT_INDEX = -1;
        this.PREPASS_DEPTH = false;
        this.PREPASS_DEPTH_INDEX = -1;
        this.PREPASS_SCREENSPACE_DEPTH = false;
        this.PREPASS_SCREENSPACE_DEPTH_INDEX = -1;
        this.PREPASS_NORMALIZED_VIEW_DEPTH = false;
        this.PREPASS_NORMALIZED_VIEW_DEPTH_INDEX = -1;
        this.PREPASS_NORMAL = false;
        this.PREPASS_NORMAL_INDEX = -1;
        this.PREPASS_NORMAL_WORLDSPACE = false;
        this.PREPASS_WORLD_NORMAL = false;
        this.PREPASS_WORLD_NORMAL_INDEX = -1;
        this.PREPASS_POSITION = false;
        this.PREPASS_POSITION_INDEX = -1;
        this.PREPASS_LOCAL_POSITION = false;
        this.PREPASS_LOCAL_POSITION_INDEX = -1;
        this.PREPASS_VELOCITY = false;
        this.PREPASS_VELOCITY_INDEX = -1;
        this.PREPASS_VELOCITY_LINEAR = false;
        this.PREPASS_VELOCITY_LINEAR_INDEX = -1;
        this.PREPASS_REFLECTIVITY = false;
        this.PREPASS_REFLECTIVITY_INDEX = -1;
        this.SCENE_MRT_COUNT = 0;
        this.NUM_BONE_INFLUENCERS = 0;
        this.BonesPerMesh = 0;
        this.BONETEXTURE = false;
        this.BONES_VELOCITY_ENABLED = false;
        this.NONUNIFORMSCALING = false;
        this.MORPHTARGETS = false;
        this.MORPHTARGETS_POSITION = false;
        this.MORPHTARGETS_NORMAL = false;
        this.MORPHTARGETS_TANGENT = false;
        this.MORPHTARGETS_UV = false;
        this.MORPHTARGETS_UV2 = false;
        this.MORPHTARGETS_COLOR = false;
        this.MORPHTARGETTEXTURE_HASPOSITIONS = false;
        this.MORPHTARGETTEXTURE_HASNORMALS = false;
        this.MORPHTARGETTEXTURE_HASTANGENTS = false;
        this.MORPHTARGETTEXTURE_HASUVS = false;
        this.MORPHTARGETTEXTURE_HASUV2S = false;
        this.MORPHTARGETTEXTURE_HASCOLORS = false;
        this.NUM_MORPH_INFLUENCERS = 0;
        this.MORPHTARGETS_TEXTURE = false;
        this.USEPHYSICALLIGHTFALLOFF = false;
        this.USEGLTFLIGHTFALLOFF = false;
        this.TWOSIDEDLIGHTING = false;
        this.MIRRORED = false;
        this.SHADOWFLOAT = false;
        this.CLIPPLANE = false;
        this.CLIPPLANE2 = false;
        this.CLIPPLANE3 = false;
        this.CLIPPLANE4 = false;
        this.CLIPPLANE5 = false;
        this.CLIPPLANE6 = false;
        this.POINTSIZE = false;
        this.FOG = false;
        this.LOGARITHMICDEPTH = false;
        this.CAMERA_ORTHOGRAPHIC = false;
        this.CAMERA_PERSPECTIVE = false;
        this.AREALIGHTSUPPORTED = true;
        this.FORCENORMALFORWARD = false;
        this.SPECULARAA = false;
        this.UNLIT = false;
        this.DECAL_AFTER_DETAIL = false;
        this.DEBUGMODE = 0;
        this.CLUSTLIGHT_SLICES = 0;
        this.CLUSTLIGHT_BATCH = 0;
        // BRDF defines
        this.BRDF_V_HEIGHT_CORRELATED = true;
        this.MS_BRDF_ENERGY_CONSERVATION = true;
        this.SPHERICAL_HARMONICS = true;
        this.SPECULAR_GLOSSINESS_ENERGY_CONSERVATION = true;
        this.MIX_IBL_RADIANCE_WITH_IRRADIANCE = true;
        this.LEGACY_SPECULAR_ENERGY_CONSERVATION = false;
        this.BASE_DIFFUSE_MODEL = 0;
        this.DIELECTRIC_SPECULAR_MODEL = 1;
        this.CONDUCTOR_SPECULAR_MODEL = 1;
        this.rebuild();
    }
    /**
     * Resets the PBR Material defines.
     */
    reset() {
        super.reset();
        this.ALPHATESTVALUE = "0.5";
        this.NORMALXYSCALE = true;
    }
}
class OpenPBRMaterialBase extends ImageProcessingMixin(PushMaterial) {
}
/**
 * A Physically based material that follows the specification of OpenPBR.
 *
 * For more information, please refer to the documentation :
 * https://academysoftwarefoundation.github.io/OpenPBR/index.html
 */
class OpenPBRMaterial extends OpenPBRMaterialBase {
    /**
     * Defines the angle of the tangent of the material's geometry. Used only for anisotropic reflections.
     * See OpenPBR's specs for geometry_tangent
     */
    get geometryTangentAngle() {
        return Math.atan2(this.geometryTangent.y, this.geometryTangent.x);
    }
    set geometryTangentAngle(value) {
        this.geometryTangent = new Vector2(Math.cos(value), Math.sin(value));
    }
    /**
     * Defines the angle of the tangent of the material's coat layer.
     */
    get geometryCoatTangentAngle() {
        return Math.atan2(this.geometryCoatTangent.y, this.geometryCoatTangent.x);
    }
    /**
     * Defines the angle of the tangent of the material's coat layer.
     */
    set geometryCoatTangentAngle(value) {
        this.geometryCoatTangent = new Vector2(Math.cos(value), Math.sin(value));
    }
    /**
     * BJS is using an hardcoded light falloff based on a manually sets up range.
     * In PBR, one way to represents the falloff is to use the inverse squared root algorithm.
     * This parameter can help you switch back to the BJS mode in order to create scenes using both materials.
     */
    get usePhysicalLightFalloff() {
        return this._lightFalloff === Material.LIGHTFALLOFF_PHYSICAL;
    }
    /**
     * BJS is using an hardcoded light falloff based on a manually sets up range.
     * In PBR, one way to represents the falloff is to use the inverse squared root algorithm.
     * This parameter can help you switch back to the BJS mode in order to create scenes using both materials.
     */
    set usePhysicalLightFalloff(value) {
        if (value !== this.usePhysicalLightFalloff) {
            // Ensure the effect will be rebuilt.
            this._markAllSubMeshesAsTexturesDirty();
            if (value) {
                this._lightFalloff = Material.LIGHTFALLOFF_PHYSICAL;
            }
            else {
                this._lightFalloff = Material.LIGHTFALLOFF_STANDARD;
            }
        }
    }
    /**
     * In order to support the falloff compatibility with gltf, a special mode has been added
     * to reproduce the gltf light falloff.
     */
    get useGLTFLightFalloff() {
        return this._lightFalloff === Material.LIGHTFALLOFF_GLTF;
    }
    /**
     * In order to support the falloff compatibility with gltf, a special mode has been added
     * to reproduce the gltf light falloff.
     */
    set useGLTFLightFalloff(value) {
        if (value !== this.useGLTFLightFalloff) {
            // Ensure the effect will be rebuilt.
            this._markAllSubMeshesAsTexturesDirty();
            if (value) {
                this._lightFalloff = Material.LIGHTFALLOFF_GLTF;
            }
            else {
                this._lightFalloff = Material.LIGHTFALLOFF_STANDARD;
            }
        }
    }
    /**
     * Enables realtime filtering on the texture.
     */
    get realTimeFiltering() {
        return this._realTimeFiltering;
    }
    set realTimeFiltering(b) {
        this._realTimeFiltering = b;
        this.markAsDirty(1);
    }
    /**
     * Quality switch for realtime filtering
     */
    get realTimeFilteringQuality() {
        return this._realTimeFilteringQuality;
    }
    set realTimeFilteringQuality(n) {
        this._realTimeFilteringQuality = n;
        this.markAsDirty(1);
    }
    /**
     * The number of samples used to compute the fuzz IBL lighting.
     */
    get fuzzSampleNumber() {
        return this._fuzzSampleNumber;
    }
    set fuzzSampleNumber(n) {
        this._fuzzSampleNumber = n;
        this.markAsDirty(1);
    }
    /**
     * Can this material render to several textures at once
     */
    get canRenderToMRT() {
        return true;
    }
    /**
     * Instantiates a new OpenPBRMaterial instance.
     *
     * @param name The material name
     * @param scene The scene the material will be use in.
     * @param forceGLSL Use the GLSL code generation for the shader (even on WebGPU). Default is false
     */
    constructor(name, scene, forceGLSL = false) {
        super(name, scene, undefined, forceGLSL || OpenPBRMaterial.ForceGLSL);
        this._baseWeight = new Property("base_weight", 1, "vBaseWeight", 1);
        this._baseWeightTexture = new Sampler("base_weight", "baseWeight", "BASE_WEIGHT");
        this._baseColor = new Property("base_color", Color3.White(), "vBaseColor", 4);
        this._baseColorTexture = new Sampler("base_color", "baseColor", "BASE_COLOR");
        this._baseDiffuseRoughness = new Property("base_diffuse_roughness", 0, "vBaseDiffuseRoughness", 1);
        this._baseDiffuseRoughnessTexture = new Sampler("base_diffuse_roughness", "baseDiffuseRoughness", "BASE_DIFFUSE_ROUGHNESS");
        this._baseMetalness = new Property("base_metalness", 0, "vReflectanceInfo", 4, 0);
        this._baseMetalnessTexture = new Sampler("base_metalness", "baseMetalness", "BASE_METALNESS");
        this._specularWeight = new Property("specular_weight", 1, "vReflectanceInfo", 4, 3);
        this._specularWeightTexture = new Sampler("specular_weight", "specularWeight", "SPECULAR_WEIGHT");
        this._specularColor = new Property("specular_color", Color3.White(), "vSpecularColor", 4);
        this._specularColorTexture = new Sampler("specular_color", "specularColor", "SPECULAR_COLOR");
        this._specularRoughness = new Property("specular_roughness", 0.3, "vReflectanceInfo", 4, 1);
        this._specularRoughnessTexture = new Sampler("specular_roughness", "specularRoughness", "SPECULAR_ROUGHNESS");
        this._specularRoughnessAnisotropy = new Property("specular_roughness_anisotropy", 0, "vSpecularAnisotropy", 3, 2);
        this._specularRoughnessAnisotropyTexture = new Sampler("specular_roughness_anisotropy", "specularRoughnessAnisotropy", "SPECULAR_ROUGHNESS_ANISOTROPY");
        this._specularIor = new Property("specular_ior", 1.5, "vReflectanceInfo", 4, 2);
        this._coatWeight = new Property("coat_weight", 0.0, "vCoatWeight", 1, 0);
        this._coatWeightTexture = new Sampler("coat_weight", "coatWeight", "COAT_WEIGHT");
        this._coatColor = new Property("coat_color", Color3.White(), "vCoatColor", 3, 0);
        this._coatColorTexture = new Sampler("coat_color", "coatColor", "COAT_COLOR");
        this._coatRoughness = new Property("coat_roughness", 0.0, "vCoatRoughness", 1, 0);
        this._coatRoughnessTexture = new Sampler("coat_roughness", "coatRoughness", "COAT_ROUGHNESS");
        this._coatRoughnessAnisotropy = new Property("coat_roughness_anisotropy", 0, "vCoatRoughnessAnisotropy", 1);
        this._coatRoughnessAnisotropyTexture = new Sampler("coat_roughness_anisotropy", "coatRoughnessAnisotropy", "COAT_ROUGHNESS_ANISOTROPY");
        this._coatIor = new Property("coat_ior", 1.5, "vCoatIor", 1, 0);
        this._coatDarkening = new Property("coat_darkening", 1.0, "vCoatDarkening", 1, 0);
        this._coatDarkeningTexture = new Sampler("coat_darkening", "coatDarkening", "COAT_DARKENING");
        /**
         * Specifies whether the coat roughness is taken from the
         * same texture as the coat_weight.
         */
        this.useCoatRoughnessFromWeightTexture = false;
        this._fuzzWeight = new Property("fuzz_weight", 0.0, "vFuzzWeight", 1, 0);
        this._fuzzWeightTexture = new Sampler("fuzz_weight", "fuzzWeight", "FUZZ_WEIGHT");
        this._fuzzColor = new Property("fuzz_color", Color3.White(), "vFuzzColor", 3, 0);
        this._fuzzColorTexture = new Sampler("fuzz_color", "fuzzColor", "FUZZ_COLOR");
        this._fuzzRoughness = new Property("fuzz_roughness", 0.5, "vFuzzRoughness", 1, 0);
        this._fuzzRoughnessTexture = new Sampler("fuzz_roughness", "fuzzRoughness", "FUZZ_ROUGHNESS");
        this._geometryNormalTexture = new Sampler("geometry_normal", "geometryNormal", "GEOMETRY_NORMAL");
        this._geometryTangent = new Property("geometry_tangent", new Vector2(1, 0), "vSpecularAnisotropy", 3, 0);
        this._geometryTangentTexture = new Sampler("geometry_tangent", "geometryTangent", "GEOMETRY_TANGENT");
        this._geometryCoatNormalTexture = new Sampler("geometry_coat_normal", "geometryCoatNormal", "GEOMETRY_COAT_NORMAL");
        this._geometryCoatTangent = new Property("geometry_coat_tangent", new Vector2(1, 0), "vGeometryCoatTangent", 2, 0);
        this._geometryCoatTangentTexture = new Sampler("geometry_coat_tangent", "geometryCoatTangent", "GEOMETRY_COAT_TANGENT");
        this._geometryOpacity = new Property("geometry_opacity", 1.0, "vBaseColor", 4, 3);
        this._geometryOpacityTexture = new Sampler("geometry_opacity", "geometryOpacity", "GEOMETRY_OPACITY");
        this._emissionLuminance = new Property("emission_luminance", 1.0, "vLightingIntensity", 4, 1);
        this._emissionColor = new Property("emission_color", Color3.Black(), "vEmissionColor", 3);
        this._emissionColorTexture = new Sampler("emission_color", "emissionColor", "EMISSION_COLOR");
        this._thinFilmWeight = new Property("thin_film_weight", 0.0, "vThinFilmWeight", 1, 0);
        this._thinFilmWeightTexture = new Sampler("thin_film_weight", "thinFilmWeight", "THIN_FILM_WEIGHT");
        this._thinFilmThickness = new Property("thin_film_thickness", 0.5, "vThinFilmThickness", 2, 0);
        this._thinFilmThicknessMin = new Property("thin_film_thickness_min", 0.0, "vThinFilmThickness", 2, 1);
        this._thinFilmThicknessTexture = new Sampler("thin_film_thickness", "thinFilmThickness", "THIN_FILM_THICKNESS");
        this._thinFilmIor = new Property("thin_film_ior", 1.4, "vThinFilmIor", 1, 0);
        this._ambientOcclusionTexture = new Sampler("ambient_occlusion", "ambientOcclusion", "AMBIENT_OCCLUSION");
        this._uniformsList = {};
        this._samplersList = {};
        this._samplerDefines = {};
        /**
         * Intensity of the direct lights e.g. the four lights available in your scene.
         * This impacts both the direct diffuse and specular highlights.
         */
        this.directIntensity = 1.0;
        /**
         * Intensity of the environment e.g. how much the environment will light the object
         * either through harmonics for rough material or through the reflection for shiny ones.
         */
        this.environmentIntensity = 1.0;
        /**
         * Specifies that the specular weight is stored in the alpha channel of the specular weight texture.
         */
        this.useSpecularWeightFromTextureAlpha = false;
        /**
         * Enforces alpha test in opaque or blend mode in order to improve the performances of some situations.
         */
        this.forceAlphaTest = false;
        /**
         * Defines the alpha limits in alpha test mode.
         */
        this.alphaCutOff = 0.4;
        /**
         * Specifies if the metallic texture contains the ambient occlusion information in its red channel.
         */
        this.useAmbientOcclusionFromMetallicTextureRed = false;
        /**
         * Specifies if the ambient texture contains the ambient occlusion information in its red channel only.
         */
        this.useAmbientInGrayScale = false;
        /**
         * Allows using an object space normal map (instead of tangent space).
         */
        this.useObjectSpaceNormalMap = false;
        /**
         * Allows using the normal map in parallax mode.
         */
        this.useParallax = false;
        /**
         * Allows using the normal map in parallax occlusion mode.
         */
        this.useParallaxOcclusion = false;
        /**
         * Controls the scale bias of the parallax mode.
         */
        this.parallaxScaleBias = 0.05;
        /**
         * If sets to true, disables all the lights affecting the material.
         */
        this.disableLighting = false;
        /**
         * Force the shader to compute irradiance in the fragment shader in order to take normal mapping into account.
         */
        this.forceIrradianceInFragment = false;
        /**
         * Number of Simultaneous lights allowed on the material.
         */
        this.maxSimultaneousLights = 4;
        /**
         * If sets to true, x component of normal map value will invert (x = 1.0 - x).
         */
        this.invertNormalMapX = false;
        /**
         * If sets to true, y component of normal map value will invert (y = 1.0 - y).
         */
        this.invertNormalMapY = false;
        /**
         * If sets to true and backfaceCulling is false, normals will be flipped on the backside.
         */
        this.twoSidedLighting = false;
        /**
         * A fresnel is applied to the alpha of the model to ensure grazing angles edges are not alpha tested.
         * And/Or occlude the blended part. (alpha is converted to gamma to compute the fresnel)
         */
        this.useAlphaFresnel = false;
        /**
         * A fresnel is applied to the alpha of the model to ensure grazing angles edges are not alpha tested.
         * And/Or occlude the blended part. (alpha stays linear to compute the fresnel)
         */
        this.useLinearAlphaFresnel = false;
        /**
         * Let user defines the brdf lookup texture used for IBL.
         * A default 8bit version is embedded but you could point at :
         * * Default texture: https://assets.babylonjs.com/environments/correlatedMSBRDF_RGBD.png
         * * Default 16bit pixel depth texture: https://assets.babylonjs.com/environments/correlatedMSBRDF.dds
         * * LEGACY Default None correlated https://assets.babylonjs.com/environments/uncorrelatedBRDF_RGBD.png
         * * LEGACY Default None correlated 16bit pixel depth https://assets.babylonjs.com/environments/uncorrelatedBRDF.dds
         */
        this.environmentBRDFTexture = null;
        /**
         * Force normal to face away from face.
         */
        this.forceNormalForward = false;
        /**
         * Enables specular anti aliasing in the PBR shader.
         * It will both interacts on the Geometry for analytical and IBL lighting.
         * It also prefilter the roughness map based on the normalmap values.
         */
        this.enableSpecularAntiAliasing = false;
        /**
         * This parameters will enable/disable Horizon occlusion to prevent normal maps to look shiny when the normal
         * makes the reflect vector face the model (under horizon).
         */
        this.useHorizonOcclusion = true;
        /**
         * This parameters will enable/disable radiance occlusion by preventing the radiance to lit
         * too much the area relying on ambient texture to define their ambient occlusion.
         */
        this.useRadianceOcclusion = true;
        /**
         * If set to true, no lighting calculations will be applied.
         */
        this.unlit = false;
        /**
         * If sets to true, the decal map will be applied after the detail map. Else, it is applied before (default: false)
         */
        this.applyDecalMapAfterDetailMap = false;
        /**
         * This stores the direct, emissive, environment, and specular light intensities into a Vector4.
         */
        this._lightingInfos = new Vector4(this.directIntensity, 1.0, this.environmentIntensity, 1.0);
        /**
         * Stores the radiance (and, possibly, irradiance) values in a texture.
         * @internal
         */
        this._radianceTexture = null;
        /**
         * Specifies that the specular weight will be read from the alpha channel.
         * This is for compatibility with glTF's KHR_materials_specular extension.
         * @internal
         */
        this._useSpecularWeightFromAlpha = false;
        /**
         * Specifies that the specular weight will be read from the alpha channel of the specular color texture.
         * This is for compatibility with glTF's KHR_materials_specular extension.
         * @internal
         */
        this._useSpecularWeightFromSpecularColorTexture = false;
        /**
         * Specifies if the material uses anisotropy weight read from the geometry tangent texture's blue channel.
         * This is for compatibility with glTF's anisotropy extension.
         * @internal
         */
        this._useSpecularRoughnessAnisotropyFromTangentTexture = false;
        /**
         * Specifies if the material uses coat anisotropy weight read from the coat's geometry tangent texture's blue channel.
         * This is for compatibility with glTF's clearcoat_anisotropy extension.
         * @internal
         */
        this._useCoatRoughnessAnisotropyFromTangentTexture = false;
        /**
         * Specifies whether the coat roughness is taken from the green channel of the coat texture.
         * This is for compatibility with glTF's KHR_materials_clearcoat and KHR_materials_coat extensions.
         * @internal
         */
        this._useCoatRoughnessFromGreenChannel = false;
        /**
         * Assume the anisotropy data is stored in the format specified by
         * KHR_materials_anisotropy.
         * @internal
         */
        this._useGltfStyleAnisotropy = false;
        /**
         * Specifies that the fuzz roughness is stored in the alpha channel of the texture.
         * This is for compatibility with glTF where the fuzz roughness is often stored in
         * the alpha channel of the fuzz color texture.
         */
        this._useFuzzRoughnessFromTextureAlpha = false;
        /**
         * This parameters will enable/disable Horizon occlusion to prevent normal maps to look shiny when the normal
         * makes the reflect vector face the model (under horizon).
         * @internal
         */
        this._useHorizonOcclusion = true;
        /**
         * This parameters will enable/disable radiance occlusion by preventing the radiance to lit
         * too much the area relying on ambient texture to define their ambient occlusion.
         * @internal
         */
        this._useRadianceOcclusion = true;
        /**
         * Specifies that the alpha is coming from the base color texture's alpha channel.
         * This is for compatibility with glTF.
         * @internal
         */
        this._useAlphaFromBaseColorTexture = false;
        /**
         * Specifies if the metallic texture contains the ambient occlusion information in its red channel.
         * This is for compatibility with glTF.
         * @internal
         */
        this._useAmbientOcclusionFromMetallicTextureRed = false;
        /**
         * Specifies if the metallic texture contains the roughness information in its green channel.
         * This is for compatibility with glTF.
         * @internal
         */
        this._useRoughnessFromMetallicTextureGreen = false;
        /**
         * Specifies if the metallic texture contains the metallic information in its blue channel.
         * This is for compatibility with glTF.
         * @internal
         */
        this._useMetallicFromMetallicTextureBlue = false;
        /**
         * Specifies if the thin film thickness is stored in the green channel of the thin film thickness texture.
         */
        this._useThinFilmThicknessFromTextureGreen = false;
        /**
         * Defines the  falloff type used in this material.
         * It by default is Physical.
         * @internal
         */
        this._lightFalloff = Material.LIGHTFALLOFF_PHYSICAL;
        /**
         * Allows using an object space normal map (instead of tangent space).
         * @internal
         */
        this._useObjectSpaceNormalMap = false;
        /**
         * Allows using the normal map in parallax mode.
         * @internal
         */
        this._useParallax = false;
        /**
         * Allows using the normal map in parallax occlusion mode.
         * @internal
         */
        this._useParallaxOcclusion = false;
        /**
         * Controls the scale bias of the parallax mode.
         * @internal
         */
        this._parallaxScaleBias = 0.05;
        /**
         * If sets to true, disables all the lights affecting the material.
         * @internal
         */
        this._disableLighting = false;
        /**
         * Number of Simultaneous lights allowed on the material.
         * @internal
         */
        this._maxSimultaneousLights = 4;
        /**
         * If sets to true, x component of normal map value will be inverted (x = 1.0 - x).
         * @internal
         */
        this._invertNormalMapX = false;
        /**
         * If sets to true, y component of normal map value will be inverted (y = 1.0 - y).
         * @internal
         */
        this._invertNormalMapY = false;
        /**
         * If sets to true and backfaceCulling is false, normals will be flipped on the backside.
         * @internal
         */
        this._twoSidedLighting = false;
        /**
         * Defines the alpha limits in alpha test mode.
         * @internal
         */
        this._alphaCutOff = 0.4;
        /**
         * A fresnel is applied to the alpha of the model to ensure grazing angles edges are not alpha tested.
         * And/Or occlude the blended part. (alpha is converted to gamma to compute the fresnel)
         * @internal
         */
        this._useAlphaFresnel = false;
        /**
         * A fresnel is applied to the alpha of the model to ensure grazing angles edges are not alpha tested.
         * And/Or occlude the blended part. (alpha stays linear to compute the fresnel)
         * @internal
         */
        this._useLinearAlphaFresnel = false;
        /**
         * Specifies the environment BRDF texture used to compute the scale and offset roughness values
         * from cos theta and roughness:
         * http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
         * @internal
         */
        this._environmentBRDFTexture = null;
        /**
         * Specifies the environment BRDF texture used to compute the scale and offset roughness values
         * from cos theta and roughness for the fuzz layer:
         * https://github.com/tizian/ltc-sheen?tab=readme-ov-file
         * @internal
         */
        this._environmentFuzzBRDFTexture = null;
        /**
         * Force the shader to compute irradiance in the fragment shader in order to take normal mapping into account.
         * @internal
         */
        this._forceIrradianceInFragment = false;
        this._realTimeFiltering = false;
        this._realTimeFilteringQuality = 8;
        this._fuzzSampleNumber = 4;
        /**
         * Force normal to face away from face.
         * @internal
         */
        this._forceNormalForward = false;
        /**
         * Enables specular anti aliasing in the PBR shader.
         * It will both interacts on the Geometry for analytical and IBL lighting.
         * It also prefilter the roughness map based on the normalmap values.
         * @internal
         */
        this._enableSpecularAntiAliasing = false;
        /**
         * Stores the available render targets.
         */
        this._renderTargets = new SmartArray(16);
        /**
         * If set to true, no lighting calculations will be applied.
         */
        this._unlit = false;
        /**
         * If sets to true, the decal map will be applied after the detail map. Else, it is applied before (default: false)
         */
        this._applyDecalMapAfterDetailMap = false;
        this._debugMode = 0;
        this._shadersLoaded = false;
        this._breakShaderLoadedCheck = false;
        /**
         * @internal
         * This is reserved for the inspector.
         * Defines the material debug mode.
         * It helps seeing only some components of the material while troubleshooting.
         */
        this.debugMode = 0;
        /**
         * @internal
         * This is reserved for the inspector.
         * Specify from where on screen the debug mode should start.
         * The value goes from -1 (full screen) to 1 (not visible)
         * It helps with side by side comparison against the final render
         * This defaults to -1
         */
        this.debugLimit = -1;
        /**
         * @internal
         * This is reserved for the inspector.
         * As the default viewing range might not be enough (if the ambient is really small for instance)
         * You can use the factor to better multiply the final value.
         */
        this.debugFactor = 1;
        this._cacheHasRenderTargetTextures = false;
        this._transparencyMode = Material.MATERIAL_OPAQUE;
        // TODO: Check if we're running WebGL 2.0 or above
        if (this.getScene() && !this.getScene()?.getEngine().isWebGPU && this.getScene().getEngine().webGLVersion < 2) {
            Logger.Error("OpenPBRMaterial: WebGL 2.0 or above is required for this material.");
        }
        if (!OpenPBRMaterial._noiseTextures[this.getScene().uniqueId]) {
            OpenPBRMaterial._noiseTextures[this.getScene().uniqueId] = new Texture(Tools.GetAssetUrl("https://assets.babylonjs.com/core/blue_noise/blue_noise_rgb.png"), this.getScene(), false, true, 1);
            this.getScene().onDisposeObservable.addOnce(() => {
                OpenPBRMaterial._noiseTextures[this.getScene().uniqueId]?.dispose();
                delete OpenPBRMaterial._noiseTextures[this.getScene().uniqueId];
            });
        }
        // Setup the default processing configuration to the scene.
        this._attachImageProcessingConfiguration(null);
        this.getRenderTargetTextures = () => {
            this._renderTargets.reset();
            if (MaterialFlags.ReflectionTextureEnabled && this._radianceTexture && this._radianceTexture.isRenderTarget) {
                this._renderTargets.push(this._radianceTexture);
            }
            this._eventInfo.renderTargets = this._renderTargets;
            this._callbackPluginEventFillRenderTargetTextures(this._eventInfo);
            return this._renderTargets;
        };
        this._environmentBRDFTexture = GetEnvironmentBRDFTexture(this.getScene());
        this._environmentFuzzBRDFTexture = GetEnvironmentFuzzBRDFTexture(this.getScene());
        this.prePassConfiguration = new PrePassConfiguration();
        // Build the internal property list that can be used to generate and update the uniform buffer
        this._propertyList = {};
        for (const key of Object.getOwnPropertyNames(this)) {
            const value = this[key];
            if (value instanceof Property) {
                this._propertyList[key] = value;
            }
        }
        // Build the internal uniforms list that is used for combining and updating
        // property values in the uniform buffer
        const propertyKeys = Object.keys(this._propertyList);
        propertyKeys.forEach((key) => {
            const prop = this._propertyList[key];
            let uniform = this._uniformsList[prop.targetUniformName];
            if (!uniform) {
                uniform = new Uniform(prop.targetUniformName, prop.targetUniformComponentNum);
                this._uniformsList[prop.targetUniformName] = uniform;
            }
            else if (uniform.numComponents !== prop.targetUniformComponentNum) {
                Logger.Error(`Uniform ${prop.targetUniformName} already exists of size ${uniform.numComponents}, but trying to set it to ${prop.targetUniformComponentNum}.`);
            }
            uniform.linkedProperties[prop.name] = prop;
        });
        // Build the internal list of samplers
        this._samplersList = {};
        for (const key of Object.getOwnPropertyNames(this)) {
            const value = this[key];
            if (value instanceof Sampler) {
                this._samplersList[key] = value;
            }
        }
        // For each sampler in _samplersList, add defines to be added to OpenPBRMaterialDefines
        for (const samplerKey in this._samplersList) {
            const sampler = this._samplersList[samplerKey];
            const defineName = sampler.textureDefine;
            this._samplerDefines[defineName] = { type: "boolean", default: false };
            this._samplerDefines[defineName + "DIRECTUV"] = { type: "number", default: 0 };
            this._samplerDefines[defineName + "_GAMMA"] = { type: "boolean", default: false };
        }
        // Arg. Why do I have to add these references to get rid of the linting errors?
        this._baseWeight;
        this._baseWeightTexture;
        this._baseColor;
        this._baseColorTexture;
        this._baseDiffuseRoughness;
        this._baseDiffuseRoughnessTexture;
        this._baseMetalness;
        this._baseMetalnessTexture;
        this._specularWeight;
        this._specularWeightTexture;
        this._specularColor;
        this._specularColorTexture;
        this._specularRoughness;
        this._specularIor;
        this._specularRoughnessTexture;
        this._specularRoughnessAnisotropy;
        this._specularRoughnessAnisotropyTexture;
        this._coatWeight;
        this._coatWeightTexture;
        this._coatColor;
        this._coatColorTexture;
        this._coatRoughness;
        this._coatRoughnessTexture;
        this._coatRoughnessAnisotropy;
        this._coatRoughnessAnisotropyTexture;
        this._coatIor;
        this._coatDarkening;
        this._coatDarkeningTexture;
        this._fuzzWeight;
        this._fuzzWeightTexture;
        this._fuzzColor;
        this._fuzzColorTexture;
        this._fuzzRoughness;
        this._fuzzRoughnessTexture;
        this._geometryNormalTexture;
        this._geometryTangent;
        this._geometryTangentTexture;
        this._geometryCoatNormalTexture;
        this._geometryCoatTangent;
        this._geometryCoatTangentTexture;
        this._geometryOpacity;
        this._geometryOpacityTexture;
        this._thinFilmWeight;
        this._thinFilmWeightTexture;
        this._thinFilmThickness;
        this._thinFilmThicknessMin;
        this._thinFilmThicknessTexture;
        this._thinFilmIor;
        this._emissionLuminance;
        this._emissionColor;
        this._emissionColorTexture;
        this._ambientOcclusionTexture;
    }
    /**
     * Gets a boolean indicating that current material needs to register RTT
     */
    get hasRenderTargetTextures() {
        if (MaterialFlags.ReflectionTextureEnabled && this._radianceTexture && this._radianceTexture.isRenderTarget) {
            return true;
        }
        return this._cacheHasRenderTargetTextures;
    }
    /**
     * Can this material render to prepass
     */
    get isPrePassCapable() {
        return !this.disableDepthWrite;
    }
    /**
     * @returns the name of the material class.
     */
    getClassName() {
        return "OpenPBRMaterial";
    }
    get transparencyMode() {
        return this._transparencyMode;
    }
    set transparencyMode(value) {
        if (this._transparencyMode === value) {
            return;
        }
        this._transparencyMode = value;
        this._markAllSubMeshesAsTexturesAndMiscDirty();
    }
    /**
     * @returns whether or not the alpha value of the albedo texture should be used for alpha blending.
     */
    _shouldUseAlphaFromBaseColorTexture() {
        return this._hasAlphaChannel() && this._transparencyMode !== Material.MATERIAL_OPAQUE && !this.geometryOpacityTexture;
    }
    /**
     * @returns whether or not there is a usable alpha channel for transparency.
     */
    _hasAlphaChannel() {
        return (this.baseColorTexture != null && this.baseColorTexture.hasAlpha && this._useAlphaFromBaseColorTexture) || this.geometryOpacityTexture != null;
    }
    /**
     * Makes a duplicate of the current material.
     * @param name - name to use for the new material.
     * @param cloneTexturesOnlyOnce - if a texture is used in more than one channel (e.g baseColor and opacity), only clone it once and reuse it on the other channels. Default false.
     * @param rootUrl defines the root URL to use to load textures
     * @returns cloned material instance
     */
    clone(name, cloneTexturesOnlyOnce = true, rootUrl = "") {
        const clone = SerializationHelper.Clone(() => new OpenPBRMaterial(name, this.getScene()), this, { cloneTexturesOnlyOnce });
        clone.id = name;
        clone.name = name;
        this.stencil.copyTo(clone.stencil);
        this._clonePlugins(clone, rootUrl);
        return clone;
    }
    /**
     * Serializes this PBR Material.
     * @returns - An object with the serialized material.
     */
    serialize() {
        const serializationObject = super.serialize();
        serializationObject.customType = "BABYLON.OpenPBRMaterial";
        return serializationObject;
    }
    // Statics
    /**
     * Parses a PBR Material from a serialized object.
     * @param source - Serialized object.
     * @param scene - BJS scene instance.
     * @param rootUrl - url for the scene object
     * @returns - OpenPBRMaterial
     */
    static Parse(source, scene, rootUrl) {
        const material = SerializationHelper.Parse(() => new OpenPBRMaterial(source.name, scene), source, scene, rootUrl);
        if (source.stencil) {
            material.stencil.parse(source.stencil, scene, rootUrl);
        }
        Material._ParsePlugins(source, material, scene, rootUrl);
        return material;
    }
    /**
     * Force shader compilation
     * @param mesh - Define the mesh we want to force the compilation for
     * @param onCompiled - Define a callback triggered when the compilation completes
     * @param options - Define the options used to create the compilation
     */
    forceCompilation(mesh, onCompiled, options) {
        const localOptions = {
            clipPlane: false,
            useInstances: false,
            ...options,
        };
        if (!this._uniformBufferLayoutBuilt) {
            this.buildUniformLayout();
        }
        this._callbackPluginEventGeneric(4 /* MaterialPluginEvent.GetDefineNames */, this._eventInfo);
        const checkReady = () => {
            if (this._breakShaderLoadedCheck) {
                return;
            }
            const defines = new OpenPBRMaterialDefines({
                ...(this._eventInfo.defineNames || {}),
                ...(this._samplerDefines || {}),
            });
            const effect = this._prepareEffect(mesh, mesh, defines, undefined, undefined, localOptions.useInstances, localOptions.clipPlane);
            if (this._onEffectCreatedObservable) {
                onCreatedEffectParameters.effect = effect;
                onCreatedEffectParameters.subMesh = null;
                this._onEffectCreatedObservable.notifyObservers(onCreatedEffectParameters);
            }
            if (effect.isReady()) {
                if (onCompiled) {
                    onCompiled(this);
                }
            }
            else {
                effect.onCompileObservable.add(() => {
                    if (onCompiled) {
                        onCompiled(this);
                    }
                });
            }
        };
        checkReady();
    }
    /**
     * Specifies that the submesh is ready to be used.
     * @param mesh - BJS mesh.
     * @param subMesh - A submesh of the BJS mesh.  Used to check if it is ready.
     * @param useInstances - Specifies that instances should be used.
     * @returns - boolean indicating that the submesh is ready or not.
     */
    isReadyForSubMesh(mesh, subMesh, useInstances) {
        if (!this._uniformBufferLayoutBuilt) {
            this.buildUniformLayout();
        }
        const drawWrapper = subMesh._drawWrapper;
        if (drawWrapper.effect && this.isFrozen) {
            if (drawWrapper._wasPreviouslyReady && drawWrapper._wasPreviouslyUsingInstances === useInstances) {
                return true;
            }
        }
        if (!subMesh.materialDefines) {
            this._callbackPluginEventGeneric(4 /* MaterialPluginEvent.GetDefineNames */, this._eventInfo);
            subMesh.materialDefines = new OpenPBRMaterialDefines({
                ...(this._eventInfo.defineNames || {}),
                ...(this._samplerDefines || {}),
            });
        }
        const defines = subMesh.materialDefines;
        if (this._isReadyForSubMesh(subMesh)) {
            return true;
        }
        const scene = this.getScene();
        const engine = scene.getEngine();
        if (defines._areTexturesDirty) {
            this._eventInfo.hasRenderTargetTextures = false;
            this._callbackPluginEventHasRenderTargetTextures(this._eventInfo);
            this._cacheHasRenderTargetTextures = this._eventInfo.hasRenderTargetTextures;
            if (scene.texturesEnabled) {
                // Loop through samplers, check MaterialFlag and whether the texture is ready or not.
                for (const key in this._samplersList) {
                    const sampler = this._samplersList[key];
                    if (sampler.value) {
                        if (!sampler.value.isReadyOrNotBlocking()) {
                            return false;
                        }
                    }
                }
                const radianceTexture = this._getRadianceTexture();
                if (radianceTexture && MaterialFlags.ReflectionTextureEnabled) {
                    if (!radianceTexture.isReadyOrNotBlocking()) {
                        return false;
                    }
                    if (radianceTexture.irradianceTexture) {
                        if (!radianceTexture.irradianceTexture.isReadyOrNotBlocking()) {
                            return false;
                        }
                    }
                    else {
                        // Not ready until spherical are ready too.
                        if (!radianceTexture.sphericalPolynomial && radianceTexture.getInternalTexture()?._sphericalPolynomialPromise) {
                            return false;
                        }
                    }
                }
                if (this._environmentBRDFTexture && MaterialFlags.ReflectionTextureEnabled) {
                    // This is blocking.
                    if (!this._environmentBRDFTexture.isReady()) {
                        return false;
                    }
                }
                if (this._environmentFuzzBRDFTexture && MaterialFlags.ReflectionTextureEnabled) {
                    // This is blocking.
                    if (!this._environmentFuzzBRDFTexture.isReady()) {
                        return false;
                    }
                }
                if (OpenPBRMaterial._noiseTextures[scene.uniqueId]) {
                    if (!OpenPBRMaterial._noiseTextures[scene.uniqueId].isReady()) {
                        return false;
                    }
                }
            }
        }
        this._eventInfo.isReadyForSubMesh = true;
        this._eventInfo.defines = defines;
        this._eventInfo.subMesh = subMesh;
        this._callbackPluginEventIsReadyForSubMesh(this._eventInfo);
        if (!this._eventInfo.isReadyForSubMesh) {
            return false;
        }
        if (defines._areImageProcessingDirty && this._imageProcessingConfiguration) {
            if (!this._imageProcessingConfiguration.isReady()) {
                return false;
            }
        }
        // Check if Area Lights have LTC texture.
        if (defines["AREALIGHTUSED"]) {
            for (let index = 0; index < mesh.lightSources.length; index++) {
                if (!mesh.lightSources[index]._isReady()) {
                    return false;
                }
            }
        }
        if (!engine.getCaps().standardDerivatives && !mesh.isVerticesDataPresent(VertexBuffer.NormalKind)) {
            mesh.createNormals(true);
            Logger.Warn("OpenPBRMaterial: Normals have been created for the mesh: " + mesh.name);
        }
        const previousEffect = subMesh.effect;
        const lightDisposed = defines._areLightsDisposed;
        let effect = this._prepareEffect(mesh, subMesh.getRenderingMesh(), defines, this.onCompiled, this.onError, useInstances, null);
        let forceWasNotReadyPreviously = false;
        if (effect) {
            if (this._onEffectCreatedObservable) {
                onCreatedEffectParameters.effect = effect;
                onCreatedEffectParameters.subMesh = subMesh;
                this._onEffectCreatedObservable.notifyObservers(onCreatedEffectParameters);
            }
            // Use previous effect while new one is compiling
            if (this.allowShaderHotSwapping && previousEffect && !effect.isReady()) {
                effect = previousEffect;
                defines.markAsUnprocessed();
                forceWasNotReadyPreviously = this.isFrozen;
                if (lightDisposed) {
                    // re register in case it takes more than one frame.
                    defines._areLightsDisposed = true;
                    return false;
                }
            }
            else {
                scene.resetCachedMaterial();
                subMesh.setEffect(effect, defines, this._materialContext);
            }
        }
        if (!subMesh.effect || !subMesh.effect.isReady()) {
            return false;
        }
        defines._renderId = scene.getRenderId();
        drawWrapper._wasPreviouslyReady = forceWasNotReadyPreviously ? false : true;
        drawWrapper._wasPreviouslyUsingInstances = !!useInstances;
        this._checkScenePerformancePriority();
        return true;
    }
    /**
     * Initializes the uniform buffer layout for the shader.
     */
    buildUniformLayout() {
        // Order is important !
        const ubo = this._uniformBuffer;
        ubo.addUniform("vTangentSpaceParams", 2);
        ubo.addUniform("vLightingIntensity", 4);
        ubo.addUniform("pointSize", 1);
        ubo.addUniform("vDebugMode", 2);
        ubo.addUniform("cameraInfo", 4);
        PrepareUniformLayoutForIBL(ubo, true, true, true, true, true);
        Object.values(this._uniformsList).forEach((uniform) => {
            ubo.addUniform(uniform.name, uniform.numComponents);
        });
        Object.values(this._samplersList).forEach((sampler) => {
            ubo.addUniform(sampler.samplerInfoName, 2);
            ubo.addUniform(sampler.samplerMatrixName, 16);
        });
        super.buildUniformLayout();
    }
    /**
     * Binds the submesh data.
     * @param world - The world matrix.
     * @param mesh - The BJS mesh.
     * @param subMesh - A submesh of the BJS mesh.
     */
    bindForSubMesh(world, mesh, subMesh) {
        const scene = this.getScene();
        const defines = subMesh.materialDefines;
        if (!defines) {
            return;
        }
        const effect = subMesh.effect;
        if (!effect) {
            return;
        }
        this._activeEffect = effect;
        // Matrices Mesh.
        mesh.getMeshUniformBuffer().bindToEffect(effect, "Mesh");
        mesh.transferToEffect(world);
        const engine = scene.getEngine();
        // Binding unconditionally
        this._uniformBuffer.bindToEffect(effect, "Material");
        this.prePassConfiguration.bindForSubMesh(this._activeEffect, scene, mesh, world, this.isFrozen);
        MaterialHelperGeometryRendering.Bind(engine.currentRenderPassId, this._activeEffect, mesh, world, this);
        const camera = scene.activeCamera;
        if (camera) {
            this._uniformBuffer.updateFloat4("cameraInfo", camera.minZ, camera.maxZ, 0, 0);
        }
        else {
            this._uniformBuffer.updateFloat4("cameraInfo", 0, 0, 0, 0);
        }
        this._eventInfo.subMesh = subMesh;
        this._callbackPluginEventHardBindForSubMesh(this._eventInfo);
        // Normal Matrix
        if (defines.OBJECTSPACE_NORMALMAP) {
            world.toNormalMatrix(this._normalMatrix);
            this.bindOnlyNormalMatrix(this._normalMatrix);
        }
        const mustRebind = this._mustRebind(scene, effect, subMesh, mesh.visibility);
        // Bones
        BindBonesParameters(mesh, this._activeEffect, this.prePassConfiguration);
        let radianceTexture = null;
        const ubo = this._uniformBuffer;
        if (mustRebind) {
            this.bindViewProjection(effect);
            radianceTexture = this._getRadianceTexture();
            if (!ubo.useUbo || !this.isFrozen || !ubo.isSync || subMesh._drawWrapper._forceRebindOnNextCall) {
                // Texture uniforms
                if (scene.texturesEnabled) {
                    // Loop through samplers and bind info and matrix for each texture.
                    for (const key in this._samplersList) {
                        const sampler = this._samplersList[key];
                        if (sampler.value) {
                            ubo.updateFloat2(sampler.samplerInfoName, sampler.value.coordinatesIndex, sampler.value.level);
                            BindTextureMatrix(sampler.value, ubo, sampler.samplerPrefix);
                        }
                    }
                    if (this.geometryNormalTexture) {
                        if (scene._mirroredCameraPosition) {
                            ubo.updateFloat2("vTangentSpaceParams", this._invertNormalMapX ? 1.0 : -1, this._invertNormalMapY ? 1.0 : -1);
                        }
                        else {
                            ubo.updateFloat2("vTangentSpaceParams", this._invertNormalMapX ? -1 : 1.0, this._invertNormalMapY ? -1 : 1.0);
                        }
                    }
                    BindIBLParameters(scene, defines, ubo, Color3.White(), radianceTexture, this.realTimeFiltering, true, true, true, true, true);
                }
                // Point size
                if (this.pointsCloud) {
                    ubo.updateFloat("pointSize", this.pointSize);
                }
                Object.values(this._uniformsList).forEach((uniform) => {
                    // If the property actually defines a uniform, update it.
                    if (uniform.numComponents === 4) {
                        uniform.populateVectorFromLinkedProperties(TmpVectors.Vector4[0]);
                        ubo.updateVector4(uniform.name, TmpVectors.Vector4[0]);
                    }
                    else if (uniform.numComponents === 3) {
                        uniform.populateVectorFromLinkedProperties(TmpVectors.Vector3[0]);
                        ubo.updateVector3(uniform.name, TmpVectors.Vector3[0]);
                    }
                    else if (uniform.numComponents === 2) {
                        uniform.populateVectorFromLinkedProperties(TmpVectors.Vector2[0]);
                        ubo.updateFloat2(uniform.name, TmpVectors.Vector2[0].x, TmpVectors.Vector2[0].y);
                    }
                    else if (uniform.numComponents === 1) {
                        ubo.updateFloat(uniform.name, uniform.linkedProperties[Object.keys(uniform.linkedProperties)[0]].value);
                    }
                });
                // Misc
                this._lightingInfos.x = this.directIntensity;
                this._lightingInfos.y = this.emissionLuminance;
                this._lightingInfos.z = this.environmentIntensity * scene.environmentIntensity;
                this._lightingInfos.w = 1.0; // This is used to be _specularIntensity.
                ubo.updateVector4("vLightingIntensity", this._lightingInfos);
                ubo.updateFloat2("vDebugMode", this.debugLimit, this.debugFactor);
            }
            // Textures
            if (scene.texturesEnabled) {
                // Loop through samplers and set textures
                for (const key in this._samplersList) {
                    const sampler = this._samplersList[key];
                    if (sampler.value) {
                        ubo.setTexture(sampler.samplerName, sampler.value);
                    }
                }
                BindIBLSamplers(scene, defines, ubo, radianceTexture, this.realTimeFiltering);
                if (defines.ENVIRONMENTBRDF) {
                    ubo.setTexture("environmentBrdfSampler", this._environmentBRDFTexture);
                }
                if (defines.FUZZENVIRONMENTBRDF) {
                    ubo.setTexture("environmentFuzzBrdfSampler", this._environmentFuzzBRDFTexture);
                }
                if (defines.ANISOTROPIC || defines.FUZZ) {
                    ubo.setTexture("blueNoiseSampler", OpenPBRMaterial._noiseTextures[this.getScene().uniqueId]);
                }
            }
            // OIT with depth peeling
            if (this.getScene().useOrderIndependentTransparency && this.needAlphaBlendingForMesh(mesh)) {
                this.getScene().depthPeelingRenderer.bind(effect);
            }
            this._eventInfo.subMesh = subMesh;
            this._callbackPluginEventBindForSubMesh(this._eventInfo);
            // Clip plane
            BindClipPlane(this._activeEffect, this, scene);
            this.bindEyePosition(effect);
        }
        else if (scene.getEngine()._features.needToAlwaysBindUniformBuffers) {
            this._needToBindSceneUbo = true;
        }
        if (mustRebind || !this.isFrozen) {
            // Lights
            if (scene.lightsEnabled && !this._disableLighting) {
                BindLights(scene, mesh, this._activeEffect, defines, this._maxSimultaneousLights);
            }
            // View
            if ((scene.fogEnabled && mesh.applyFog && scene.fogMode !== Scene.FOGMODE_NONE) || radianceTexture || mesh.receiveShadows || defines.PREPASS) {
                this.bindView(effect);
            }
            // Fog
            BindFogParameters(scene, mesh, this._activeEffect, true);
            // Morph targets
            if (defines.NUM_MORPH_INFLUENCERS) {
                BindMorphTargetParameters(mesh, this._activeEffect);
            }
            if (defines.BAKED_VERTEX_ANIMATION_TEXTURE) {
                mesh.bakedVertexAnimationManager?.bind(effect, defines.INSTANCES);
            }
            // image processing
            this._imageProcessingConfiguration.bind(this._activeEffect);
            // Log. depth
            BindLogDepth(defines, this._activeEffect, scene);
        }
        this._afterBind(mesh, this._activeEffect, subMesh);
        ubo.update();
    }
    /**
     * Returns the animatable textures.
     * If material have animatable metallic texture, then reflectivity texture will not be returned, even if it has animations.
     * @returns - Array of animatable textures.
     */
    getAnimatables() {
        const results = super.getAnimatables();
        // Loop through samplers and push animated textures to list.
        for (const key in this._samplersList) {
            const sampler = this._samplersList[key];
            if (sampler.value && sampler.value.animations && sampler.value.animations.length > 0) {
                results.push(sampler.value);
            }
        }
        if (this._radianceTexture && this._radianceTexture.animations && this._radianceTexture.animations.length > 0) {
            results.push(this._radianceTexture);
        }
        return results;
    }
    /**
     * Returns an array of the actively used textures.
     * @returns - Array of BaseTextures
     */
    getActiveTextures() {
        const activeTextures = super.getActiveTextures();
        // Loop through samplers and push active textures
        for (const key in this._samplersList) {
            const sampler = this._samplersList[key];
            if (sampler.value) {
                activeTextures.push(sampler.value);
            }
        }
        if (this._radianceTexture) {
            activeTextures.push(this._radianceTexture);
        }
        return activeTextures;
    }
    /**
     * Checks to see if a texture is used in the material.
     * @param texture - Base texture to use.
     * @returns - Boolean specifying if a texture is used in the material.
     */
    hasTexture(texture) {
        if (super.hasTexture(texture)) {
            return true;
        }
        // Loop through samplers and check each texture for equality
        for (const key in this._samplersList) {
            const sampler = this._samplersList[key];
            if (sampler.value === texture) {
                return true;
            }
        }
        if (this._radianceTexture === texture) {
            return true;
        }
        return false;
    }
    /**
     * Sets the required values to the prepass renderer.
     * It can't be sets when subsurface scattering of this material is disabled.
     * When scene have ability to enable subsurface prepass effect, it will enable.
     * @returns - If prepass is enabled or not.
     */
    setPrePassRenderer() {
        return false;
    }
    /**
     * Disposes the resources of the material.
     * @param forceDisposeEffect - Forces the disposal of effects.
     * @param forceDisposeTextures - Forces the disposal of all textures.
     */
    dispose(forceDisposeEffect, forceDisposeTextures) {
        this._breakShaderLoadedCheck = true;
        if (forceDisposeTextures) {
            if (this._environmentBRDFTexture && this.getScene().environmentBRDFTexture !== this._environmentBRDFTexture) {
                this._environmentBRDFTexture.dispose();
            }
            if (this._environmentFuzzBRDFTexture && this.getScene().environmentFuzzBRDFTexture !== this._environmentFuzzBRDFTexture) {
                this._environmentFuzzBRDFTexture.dispose();
            }
            // Loop through samplers and dispose the textures
            for (const key in this._samplersList) {
                const sampler = this._samplersList[key];
                sampler.value?.dispose();
            }
            this._radianceTexture?.dispose();
        }
        this._renderTargets.dispose();
        if (this._imageProcessingConfiguration && this._imageProcessingObserver) {
            this._imageProcessingConfiguration.onUpdateParameters.remove(this._imageProcessingObserver);
        }
        super.dispose(forceDisposeEffect, forceDisposeTextures);
    }
    /**
     * Returns the texture used for reflections.
     * @returns - Reflection texture if present.  Otherwise, returns the environment texture.
     */
    _getRadianceTexture() {
        if (this._radianceTexture) {
            return this._radianceTexture;
        }
        return this.getScene().environmentTexture;
    }
    _prepareEffect(mesh, renderingMesh, defines, onCompiled = null, onError = null, useInstances = null, useClipPlane = null) {
        this._prepareDefines(mesh, renderingMesh, defines, useInstances, useClipPlane);
        if (!defines.isDirty) {
            return null;
        }
        defines.markAsProcessed();
        const scene = this.getScene();
        const engine = scene.getEngine();
        // Fallbacks
        const fallbacks = new EffectFallbacks();
        let fallbackRank = 0;
        if (defines.USESPHERICALINVERTEX) {
            fallbacks.addFallback(fallbackRank++, "USESPHERICALINVERTEX");
        }
        if (defines.FOG) {
            fallbacks.addFallback(fallbackRank, "FOG");
        }
        if (defines.SPECULARAA) {
            fallbacks.addFallback(fallbackRank, "SPECULARAA");
        }
        if (defines.POINTSIZE) {
            fallbacks.addFallback(fallbackRank, "POINTSIZE");
        }
        if (defines.LOGARITHMICDEPTH) {
            fallbacks.addFallback(fallbackRank, "LOGARITHMICDEPTH");
        }
        if (defines.PARALLAX) {
            fallbacks.addFallback(fallbackRank, "PARALLAX");
        }
        if (defines.PARALLAX_RHS) {
            fallbacks.addFallback(fallbackRank, "PARALLAX_RHS");
        }
        if (defines.PARALLAXOCCLUSION) {
            fallbacks.addFallback(fallbackRank++, "PARALLAXOCCLUSION");
        }
        if (defines.ENVIRONMENTBRDF) {
            fallbacks.addFallback(fallbackRank++, "ENVIRONMENTBRDF");
        }
        if (defines.TANGENT) {
            fallbacks.addFallback(fallbackRank++, "TANGENT");
        }
        fallbackRank = HandleFallbacksForShadows(defines, fallbacks, this._maxSimultaneousLights, fallbackRank++);
        if (defines.SPECULARTERM) {
            fallbacks.addFallback(fallbackRank++, "SPECULARTERM");
        }
        if (defines.USESPHERICALFROMREFLECTIONMAP) {
            fallbacks.addFallback(fallbackRank++, "USESPHERICALFROMREFLECTIONMAP");
        }
        if (defines.USEIRRADIANCEMAP) {
            fallbacks.addFallback(fallbackRank++, "USEIRRADIANCEMAP");
        }
        if (defines.NORMAL) {
            fallbacks.addFallback(fallbackRank++, "NORMAL");
        }
        if (defines.VERTEXCOLOR) {
            fallbacks.addFallback(fallbackRank++, "VERTEXCOLOR");
        }
        if (defines.MORPHTARGETS) {
            fallbacks.addFallback(fallbackRank++, "MORPHTARGETS");
        }
        if (defines.MULTIVIEW) {
            fallbacks.addFallback(0, "MULTIVIEW");
        }
        //Attributes
        const attribs = [VertexBuffer.PositionKind];
        if (defines.NORMAL) {
            attribs.push(VertexBuffer.NormalKind);
        }
        if (defines.TANGENT) {
            attribs.push(VertexBuffer.TangentKind);
        }
        for (let i = 1; i <= 6; ++i) {
            if (defines["UV" + i]) {
                attribs.push(`uv${i === 1 ? "" : i}`);
            }
        }
        if (defines.VERTEXCOLOR) {
            attribs.push(VertexBuffer.ColorKind);
        }
        PrepareAttributesForBones(attribs, mesh, defines, fallbacks);
        PrepareAttributesForInstances(attribs, defines);
        PrepareAttributesForMorphTargets(attribs, mesh, defines);
        PrepareAttributesForBakedVertexAnimation(attribs, mesh, defines);
        let shaderName = "openpbr";
        const uniforms = [
            "world",
            "view",
            "viewProjection",
            "vEyePosition",
            "vLightsType",
            "visibility",
            "vFogInfos",
            "vFogColor",
            "pointSize",
            "mBones",
            "normalMatrix",
            "vLightingIntensity",
            "logarithmicDepthConstant",
            "vTangentSpaceParams",
            "boneTextureWidth",
            "vDebugMode",
            "morphTargetTextureInfo",
            "morphTargetTextureIndices",
            "cameraInfo",
        ];
        for (const uniformName in this._uniformsList) {
            uniforms.push(uniformName);
        }
        const samplers = [
            "environmentBrdfSampler",
            "blueNoiseSampler",
            "boneSampler",
            "morphTargets",
            "oitDepthSampler",
            "oitFrontColorSampler",
            "areaLightsLTC1Sampler",
            "areaLightsLTC2Sampler",
        ];
        if (defines.FUZZENVIRONMENTBRDF) {
            samplers.push("environmentFuzzBrdfSampler");
        }
        for (const key in this._samplersList) {
            const sampler = this._samplersList[key];
            samplers.push(sampler.samplerName);
            // Push uniforms for texture infos and matrix
            uniforms.push(sampler.samplerInfoName);
            uniforms.push(sampler.samplerMatrixName);
        }
        PrepareUniformsAndSamplersForIBL(uniforms, samplers, true);
        const uniformBuffers = ["Material", "Scene", "Mesh"];
        const indexParameters = { maxSimultaneousLights: this._maxSimultaneousLights, maxSimultaneousMorphTargets: defines.NUM_MORPH_INFLUENCERS };
        this._eventInfo.fallbacks = fallbacks;
        this._eventInfo.fallbackRank = fallbackRank;
        this._eventInfo.defines = defines;
        this._eventInfo.uniforms = uniforms;
        this._eventInfo.attributes = attribs;
        this._eventInfo.samplers = samplers;
        this._eventInfo.uniformBuffersNames = uniformBuffers;
        this._eventInfo.customCode = undefined;
        this._eventInfo.mesh = mesh;
        this._eventInfo.indexParameters = indexParameters;
        this._callbackPluginEventGeneric(128 /* MaterialPluginEvent.PrepareEffect */, this._eventInfo);
        MaterialHelperGeometryRendering.AddUniformsAndSamplers(uniforms, samplers);
        PrePassConfiguration.AddUniforms(uniforms);
        AddClipPlaneUniforms(uniforms);
        if (ImageProcessingConfiguration) {
            ImageProcessingConfiguration.PrepareUniforms(uniforms, defines);
            ImageProcessingConfiguration.PrepareSamplers(samplers, defines);
        }
        PrepareUniformsAndSamplersList({
            uniformsNames: uniforms,
            uniformBuffersNames: uniformBuffers,
            samplers: samplers,
            defines: defines,
            maxSimultaneousLights: this._maxSimultaneousLights,
        });
        const csnrOptions = {};
        if (this.customShaderNameResolve) {
            shaderName = this.customShaderNameResolve(shaderName, uniforms, uniformBuffers, samplers, defines, attribs, csnrOptions);
        }
        const join = defines.toString();
        const effect = engine.createEffect(shaderName, {
            attributes: attribs,
            uniformsNames: uniforms,
            uniformBuffersNames: uniformBuffers,
            samplers: samplers,
            defines: join,
            fallbacks: fallbacks,
            onCompiled: onCompiled,
            onError: onError,
            indexParameters,
            processFinalCode: csnrOptions.processFinalCode,
            processCodeAfterIncludes: this._eventInfo.customCode,
            multiTarget: defines.PREPASS,
            shaderLanguage: this._shaderLanguage,
            extraInitializationsAsync: this._shadersLoaded
                ? undefined
                : async () => {
                    if (this.shaderLanguage === 1 /* ShaderLanguage.WGSL */) {
                        await Promise.all([__vitePreload(() => import('./openpbr.vertex--b02pul6.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]):void 0,import.meta.url), __vitePreload(() => import('./openpbr.fragment-CLj6ryVP.js'),true              ?__vite__mapDeps([23,1,2,3,24,4,5,6,25,8,26,27,28,29,19,30,9,31,14,10]):void 0,import.meta.url)]);
                    }
                    else {
                        await Promise.all([__vitePreload(() => import('./openpbr.vertex-DK9C96sM.js'),true              ?__vite__mapDeps([32,1,2,3,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]):void 0,import.meta.url), __vitePreload(() => import('./openpbr.fragment-CqiyhFmE.js'),true              ?__vite__mapDeps([48,1,2,3,49,34,35,36,50,37,51,52,53,54,47,55,38,56,42,39]):void 0,import.meta.url)]);
                    }
                    this._shadersLoaded = true;
                },
        }, engine);
        this._eventInfo.customCode = undefined;
        return effect;
    }
    _prepareDefines(mesh, renderingMesh, defines, useInstances = null, useClipPlane = null) {
        const useThinInstances = renderingMesh.hasThinInstances;
        const scene = this.getScene();
        const engine = scene.getEngine();
        // Lights
        PrepareDefinesForLights(scene, mesh, defines, true, this._maxSimultaneousLights, this._disableLighting);
        defines._needNormals = true;
        // Multiview
        PrepareDefinesForMultiview(scene, defines);
        // PrePass
        const oit = this.needAlphaBlendingForMesh(mesh) && this.getScene().useOrderIndependentTransparency;
        PrepareDefinesForPrePass(scene, defines, this.canRenderToMRT && !oit);
        // Order independant transparency
        PrepareDefinesForOIT(scene, defines, oit);
        MaterialHelperGeometryRendering.PrepareDefines(engine.currentRenderPassId, mesh, defines);
        // Textures
        defines.METALLICWORKFLOW = true;
        if (defines._areTexturesDirty) {
            defines._needUVs = false;
            for (let i = 1; i <= 6; ++i) {
                defines["MAINUV" + i] = false;
            }
            if (scene.texturesEnabled) {
                // Loop through samplers and prepare defines for each texture
                for (const key in this._samplersList) {
                    const sampler = this._samplersList[key];
                    if (sampler.value) {
                        PrepareDefinesForMergedUV(sampler.value, defines, sampler.textureDefine);
                        defines[sampler.textureDefine + "_GAMMA"] = sampler.value.gammaSpace;
                    }
                    else {
                        defines[sampler.textureDefine] = false;
                    }
                }
                const radianceTexture = this._getRadianceTexture();
                const useSHInFragment = this._forceIrradianceInFragment ||
                    this.realTimeFiltering ||
                    this._twoSidedLighting ||
                    engine.getCaps().maxVaryingVectors <= 8 ||
                    this._baseDiffuseRoughnessTexture != null;
                PrepareDefinesForIBL(scene, radianceTexture, defines, this.realTimeFiltering, this.realTimeFilteringQuality, !useSHInFragment);
                if (this._baseMetalnessTexture) {
                    defines.AOSTOREINMETALMAPRED = this._useAmbientOcclusionFromMetallicTextureRed;
                }
                defines.SPECULAR_WEIGHT_IN_ALPHA = this._useSpecularWeightFromAlpha;
                defines.SPECULAR_WEIGHT_FROM_SPECULAR_COLOR_TEXTURE = this._useSpecularWeightFromSpecularColorTexture;
                defines.SPECULAR_ROUGHNESS_ANISOTROPY_FROM_TANGENT_TEXTURE = this._useSpecularRoughnessAnisotropyFromTangentTexture;
                defines.COAT_ROUGHNESS_ANISOTROPY_FROM_TANGENT_TEXTURE = this._useCoatRoughnessAnisotropyFromTangentTexture;
                defines.COAT_ROUGHNESS_FROM_GREEN_CHANNEL = this._useCoatRoughnessFromGreenChannel;
                defines.ROUGHNESSSTOREINMETALMAPGREEN = this._useRoughnessFromMetallicTextureGreen;
                defines.FUZZ_ROUGHNESS_FROM_TEXTURE_ALPHA = this._useFuzzRoughnessFromTextureAlpha;
                defines.METALLNESSSTOREINMETALMAPBLUE = this._useMetallicFromMetallicTextureBlue;
                defines.THIN_FILM_THICKNESS_FROM_THIN_FILM_TEXTURE = this._useThinFilmThicknessFromTextureGreen;
                if (this.geometryNormalTexture) {
                    if (this._useParallax && this.baseColorTexture && MaterialFlags.DiffuseTextureEnabled) {
                        defines.PARALLAX = true;
                        defines.PARALLAX_RHS = scene.useRightHandedSystem;
                        defines.PARALLAXOCCLUSION = !!this._useParallaxOcclusion;
                    }
                    else {
                        defines.PARALLAX = false;
                    }
                    defines.OBJECTSPACE_NORMALMAP = this._useObjectSpaceNormalMap;
                }
                else {
                    defines.PARALLAX = false;
                    defines.PARALLAX_RHS = false;
                    defines.PARALLAXOCCLUSION = false;
                    defines.OBJECTSPACE_NORMALMAP = false;
                }
                if (this._environmentBRDFTexture && MaterialFlags.ReflectionTextureEnabled) {
                    defines.ENVIRONMENTBRDF = true;
                    defines.ENVIRONMENTBRDF_RGBD = this._environmentBRDFTexture.isRGBD;
                }
                else {
                    defines.ENVIRONMENTBRDF = false;
                    defines.ENVIRONMENTBRDF_RGBD = false;
                }
                if (this._environmentFuzzBRDFTexture) {
                    defines.FUZZENVIRONMENTBRDF = true;
                }
                else {
                    defines.FUZZENVIRONMENTBRDF = false;
                }
                if (this._shouldUseAlphaFromBaseColorTexture()) {
                    defines.ALPHA_FROM_BASE_COLOR_TEXTURE = true;
                }
                else {
                    defines.ALPHA_FROM_BASE_COLOR_TEXTURE = false;
                }
            }
            if (this._lightFalloff === Material.LIGHTFALLOFF_STANDARD) {
                defines.USEPHYSICALLIGHTFALLOFF = false;
                defines.USEGLTFLIGHTFALLOFF = false;
            }
            else if (this._lightFalloff === Material.LIGHTFALLOFF_GLTF) {
                defines.USEPHYSICALLIGHTFALLOFF = false;
                defines.USEGLTFLIGHTFALLOFF = true;
            }
            else {
                defines.USEPHYSICALLIGHTFALLOFF = true;
                defines.USEGLTFLIGHTFALLOFF = false;
            }
            if (!this.backFaceCulling && this._twoSidedLighting) {
                defines.TWOSIDEDLIGHTING = true;
            }
            else {
                defines.TWOSIDEDLIGHTING = false;
            }
            // We need it to not invert normals in two sided lighting mode (based on the winding of the face)
            defines.MIRRORED = !!scene._mirroredCameraPosition;
            defines.SPECULARAA = engine.getCaps().standardDerivatives && this._enableSpecularAntiAliasing;
        }
        if (defines._areTexturesDirty || defines._areMiscDirty) {
            defines.ALPHATESTVALUE = `${this._alphaCutOff}${this._alphaCutOff % 1 === 0 ? "." : ""}`;
            defines.PREMULTIPLYALPHA = this.alphaMode === 7 || this.alphaMode === 8;
            defines.ALPHABLEND = this.needAlphaBlendingForMesh(mesh);
        }
        if (defines._areImageProcessingDirty && this._imageProcessingConfiguration) {
            this._imageProcessingConfiguration.prepareDefines(defines);
        }
        defines.FORCENORMALFORWARD = this._forceNormalForward;
        defines.RADIANCEOCCLUSION = this._useRadianceOcclusion;
        defines.HORIZONOCCLUSION = this._useHorizonOcclusion;
        if ((this.specularRoughnessAnisotropy > 0.0 || this.coatRoughnessAnisotropy > 0.0) &&
            OpenPBRMaterial._noiseTextures[scene.uniqueId] &&
            MaterialFlags.ReflectionTextureEnabled) {
            // ANISOTROPIC is used to include common shader functions needed for anisotropy
            // ANISOTROPIC_BASE is used to process anisotropy for the base layer
            // ANISOTROPIC_COAT is used to process anisotropy for the coat layer
            defines.ANISOTROPIC = true;
            if (!mesh.isVerticesDataPresent(VertexBuffer.TangentKind)) {
                defines._needUVs = true;
                defines.MAINUV1 = true;
            }
            if (this._useGltfStyleAnisotropy) {
                defines.USE_GLTF_STYLE_ANISOTROPY = true;
            }
            defines.ANISOTROPIC_BASE = this.specularRoughnessAnisotropy > 0.0;
            defines.ANISOTROPIC_COAT = this.coatRoughnessAnisotropy > 0.0;
        }
        else {
            defines.ANISOTROPIC = false;
            defines.USE_GLTF_STYLE_ANISOTROPY = false;
            defines.ANISOTROPIC_BASE = false;
            defines.ANISOTROPIC_COAT = false;
        }
        defines.THIN_FILM = this.thinFilmWeight > 0.0;
        defines.IRIDESCENCE = this.thinFilmWeight > 0.0;
        defines.FUZZ = this.fuzzWeight > 0 && MaterialFlags.ReflectionTextureEnabled;
        if (defines.FUZZ) {
            if (!mesh.isVerticesDataPresent(VertexBuffer.TangentKind)) {
                defines._needUVs = true;
                defines.MAINUV1 = true;
            }
            this._environmentFuzzBRDFTexture = GetEnvironmentFuzzBRDFTexture(this.getScene());
            defines.FUZZ_IBL_SAMPLES = this.fuzzSampleNumber;
        }
        else {
            this._environmentFuzzBRDFTexture = null;
            defines.FUZZENVIRONMENTBRDF = false;
            defines.FUZZ_IBL_SAMPLES = 0;
        }
        // Misc.
        if (defines._areMiscDirty) {
            PrepareDefinesForMisc(mesh, scene, this._useLogarithmicDepth, this.pointsCloud, this.fogEnabled, this.needAlphaTestingForMesh(mesh), defines, this._applyDecalMapAfterDetailMap, this._useVertexPulling, renderingMesh, this._isVertexOutputInvariant);
            defines.UNLIT = this._unlit || ((this.pointsCloud || this.wireframe) && !mesh.isVerticesDataPresent(VertexBuffer.NormalKind));
            defines.DEBUGMODE = this._debugMode;
        }
        // Values that need to be evaluated on every frame
        PrepareDefinesForFrameBoundValues(scene, engine, this, defines, useInstances ? true : false, useClipPlane, useThinInstances);
        // External config
        this._eventInfo.defines = defines;
        this._eventInfo.mesh = mesh;
        this._callbackPluginEventPrepareDefinesBeforeAttributes(this._eventInfo);
        // Attribs
        PrepareDefinesForAttributes(mesh, defines, true, true, true, this._transparencyMode !== Material.MATERIAL_OPAQUE);
        // External config
        this._callbackPluginEventPrepareDefines(this._eventInfo);
    }
}
OpenPBRMaterial._noiseTextures = {};
/**
 * Force all the PBR materials to compile to glsl even on WebGPU engines.
 * False by default. This is mostly meant for backward compatibility.
 */
OpenPBRMaterial.ForceGLSL = false;
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseWeight")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseWeight", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseWeightTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseWeightTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseColor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseColor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseColorTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseColorTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseDiffuseRoughness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseDiffuseRoughness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseDiffuseRoughnessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseDiffuseRoughnessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseMetalness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseMetalness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "baseMetalnessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_baseMetalnessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularWeight")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularWeight", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularWeightTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularWeightTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularColor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularColor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularColorTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularColorTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularRoughness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularRoughness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularRoughnessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularRoughnessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularRoughnessAnisotropy")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularRoughnessAnisotropy", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularRoughnessAnisotropyTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularRoughnessAnisotropyTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "specularIor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_specularIor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatWeight")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatWeight", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatWeightTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatWeightTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatColor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatColor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatColorTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatColorTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatRoughness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatRoughness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatRoughnessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatRoughnessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatRoughnessAnisotropy")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatRoughnessAnisotropy", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatRoughnessAnisotropyTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatRoughnessAnisotropyTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatIor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatIor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatDarkening")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatDarkening", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "coatDarkeningTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_coatDarkeningTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzWeight")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzWeight", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzWeightTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzWeightTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzColor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzColor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzColorTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzColorTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzRoughness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzRoughness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "fuzzRoughnessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_fuzzRoughnessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryNormalTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryNormalTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryTangent")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryTangent", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryTangentTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryTangentTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryCoatNormalTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryCoatNormalTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryCoatTangent")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryCoatTangent", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryCoatTangentTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryCoatTangentTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryOpacity")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryOpacity", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "geometryOpacityTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_geometryOpacityTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "emissionLuminance")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_emissionLuminance", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "emissionColor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_emissionColor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "emissionColorTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_emissionColorTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmWeight")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmWeight", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmWeightTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmWeightTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmThickness")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmThickness", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmThicknessMin")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmThicknessMin", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmThicknessTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmThicknessTexture", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "thinFilmIor")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_thinFilmIor", void 0);
__decorate([
    addAccessorsForMaterialProperty("_markAllSubMeshesAsTexturesDirty", "ambientOcclusionTexture")
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
], OpenPBRMaterial.prototype, "_ambientOcclusionTexture", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "directIntensity", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "environmentIntensity", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useSpecularWeightFromTextureAlpha", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesAndMiscDirty")
], OpenPBRMaterial.prototype, "forceAlphaTest", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesAndMiscDirty")
], OpenPBRMaterial.prototype, "alphaCutOff", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useAmbientOcclusionFromMetallicTextureRed", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useAmbientInGrayScale", void 0);
__decorate([
    serialize()
], OpenPBRMaterial.prototype, "usePhysicalLightFalloff", null);
__decorate([
    serialize()
], OpenPBRMaterial.prototype, "useGLTFLightFalloff", null);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useObjectSpaceNormalMap", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useParallax", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useParallaxOcclusion", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "parallaxScaleBias", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsLightsDirty")
], OpenPBRMaterial.prototype, "disableLighting", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "forceIrradianceInFragment", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsLightsDirty")
], OpenPBRMaterial.prototype, "maxSimultaneousLights", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "invertNormalMapX", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "invertNormalMapY", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "twoSidedLighting", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useAlphaFresnel", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useLinearAlphaFresnel", void 0);
__decorate([
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "environmentBRDFTexture", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "forceNormalForward", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "enableSpecularAntiAliasing", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useHorizonOcclusion", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsTexturesDirty")
], OpenPBRMaterial.prototype, "useRadianceOcclusion", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsMiscDirty")
], OpenPBRMaterial.prototype, "unlit", void 0);
__decorate([
    serialize(),
    expandToProperty("_markAllSubMeshesAsMiscDirty")
], OpenPBRMaterial.prototype, "applyDecalMapAfterDetailMap", void 0);
__decorate([
    expandToProperty("_markAllSubMeshesAsMiscDirty")
], OpenPBRMaterial.prototype, "debugMode", void 0);
__decorate([
    serialize()
], OpenPBRMaterial.prototype, "transparencyMode", null);
RegisterClass("BABYLON.OpenPBRMaterial", OpenPBRMaterial);

export { OpenPBRMaterial, OpenPBRMaterialDefines };
//# sourceMappingURL=openpbrMaterial-L3-DYNd8.js.map
