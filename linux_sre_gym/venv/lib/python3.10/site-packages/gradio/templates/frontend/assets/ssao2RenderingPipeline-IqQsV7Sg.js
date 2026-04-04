const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./ssao2.fragment-3Rjrq9Rx.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./ssao2.fragment-DG9KEUgV.js","./ssaoCombine.fragment-DaSWDR4P.js","./ssaoCombine.fragment-Bvl-J2BM.js"])))=>i.map(i=>d[i]);
import { aE as EffectWrapper, b1 as Camera, bA as RandomRange, T as Texture, V as Vector3, ah as Vector2, bn as Engine, c as TmpVectors, aq as EngineStore, L as Logger, bB as PassPostProcess, aA as PostProcess, e as SerializationHelper, _ as __decorate, s as serialize, R as RegisterClass } from './index-CxR0Yf_7.js';
import { P as PostProcessRenderPipeline, a as PostProcessRenderEffect } from './postProcessRenderEffect-BIXOs3xJ.js';
import { G as GeometryBufferRenderer } from './geometryBufferRenderer-Dvq35Qkr.js';
import './postProcessRenderPipelineManagerSceneComponent-D-jwrJrt.js';
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { R as RawTexture } from './rawTexture-Bto4WZO2.js';
import './clipPlaneFragment-Vm6wRK67.js';
import './bumpFragment-Dxop61vv.js';
import './helperFunctions-CiFVmLcr.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './instancesDeclaration-Bv0KzShp.js';
import './sceneUboDeclaration-CbezEhV0.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './bumpVertex-fS00LARh.js';

/**
 * Contains all parameters needed for the prepass to perform
 * screen space subsurface scattering
 */
class SSAO2Configuration {
    constructor() {
        /**
         * Is subsurface enabled
         */
        this.enabled = false;
        /**
         * Name of the configuration
         */
        this.name = "ssao2";
        /**
         * Textures that should be present in the MRT for this effect to work
         */
        this.texturesRequired = [6, 5];
    }
}

/**
 * @internal
 */
class ThinSSAO2PostProcess extends EffectWrapper {
    _gatherImports(useWebGPU, list) {
        if (useWebGPU) {
            this._webGPUReady = true;
            list.push(__vitePreload(() => import('./ssao2.fragment-3Rjrq9Rx.js'),true              ?__vite__mapDeps([0,1,2,3]):void 0,import.meta.url));
        }
        else {
            list.push(__vitePreload(() => import('./ssao2.fragment-DG9KEUgV.js'),true              ?__vite__mapDeps([4,1,2,3]):void 0,import.meta.url));
        }
    }
    get textureWidth() {
        return this._textureWidth;
    }
    set textureWidth(width) {
        if (this._textureWidth === width) {
            return;
        }
        this._textureWidth = width;
    }
    get textureHeight() {
        return this._textureHeight;
    }
    set textureHeight(height) {
        if (this._textureHeight === height) {
            return;
        }
        this._textureHeight = height;
    }
    set samples(n) {
        this._samples = n;
        this.updateEffect();
        this._sampleSphere = this._generateHemisphere();
    }
    get samples() {
        return this._samples;
    }
    set epsilon(n) {
        this._epsilon = n;
        this.updateEffect();
    }
    get epsilon() {
        return this._epsilon;
    }
    updateEffect() {
        super.updateEffect(this._getDefinesForSSAO());
    }
    constructor(name, scene, options) {
        super({
            ...options,
            name,
            engine: scene.getEngine(),
            useShaderStore: true,
            useAsPostProcess: true,
            fragmentShader: ThinSSAO2PostProcess.FragmentUrl,
            uniforms: ThinSSAO2PostProcess.Uniforms,
            samplers: ThinSSAO2PostProcess.Samplers,
            defines: `#define SSAO\n#define SAMPLES 8\n#define EPSILON 0.0001`,
            shaderLanguage: scene.getEngine().isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
        });
        this.camera = null;
        this._textureWidth = 0;
        this._textureHeight = 0;
        this._samples = 8;
        this.totalStrength = 1.0;
        this.radius = 2.0;
        this.maxZ = 100.0;
        this.minZAspect = 0.2;
        this.base = 0;
        this._epsilon = 0.02;
        this._bits = new Uint32Array(1);
        this._scene = scene;
        this._createRandomTexture();
        this.updateEffect();
        this._sampleSphere = this._generateHemisphere();
    }
    bind(noDefaultBindings = false) {
        super.bind(noDefaultBindings);
        const effect = this._drawWrapper.effect;
        const camera = this.camera;
        if (!camera) {
            return;
        }
        const projectionMatrix = camera.getProjectionMatrix();
        effect.setArray3("sampleSphere", this._sampleSphere);
        effect.setFloat("randTextureTiles", 32.0);
        effect.setFloat("samplesFactor", 1 / this.samples);
        effect.setFloat("totalStrength", this.totalStrength);
        effect.setFloat2("texelSize", 1 / this.textureWidth, 1 / this.textureHeight);
        effect.setFloat("radius", this.radius);
        effect.setFloat("maxZ", this.maxZ);
        effect.setFloat("minZAspect", this.minZAspect);
        effect.setFloat("base", this.base);
        effect.setFloat("near", camera.minZ);
        if (camera.mode === Camera.PERSPECTIVE_CAMERA) {
            effect.setMatrix3x3("depthProjection", ThinSSAO2PostProcess.PERSPECTIVE_DEPTH_PROJECTION);
            const viewportSize = Math.tan(camera.fov / 2);
            if (camera.fovMode === Camera.FOVMODE_VERTICAL_FIXED) {
                effect.setFloat("xViewport", viewportSize * this._scene.getEngine().getAspectRatio(camera, true));
                effect.setFloat("yViewport", viewportSize);
            }
            else {
                effect.setFloat("xViewport", viewportSize);
                effect.setFloat("yViewport", viewportSize / this._scene.getEngine().getAspectRatio(camera, true));
            }
        }
        else {
            const halfWidth = this._scene.getEngine().getRenderWidth() / 2.0;
            const halfHeight = this._scene.getEngine().getRenderHeight() / 2.0;
            const orthoLeft = camera.orthoLeft ?? -halfWidth;
            const orthoRight = camera.orthoRight ?? halfWidth;
            const orthoBottom = camera.orthoBottom ?? -halfHeight;
            const orthoTop = camera.orthoTop ?? halfHeight;
            effect.setMatrix3x3("depthProjection", ThinSSAO2PostProcess.ORTHO_DEPTH_PROJECTION);
            effect.setFloat4("viewport", orthoLeft, orthoRight, orthoBottom, orthoTop);
        }
        effect.setMatrix("projection", projectionMatrix);
        effect.setTexture("randomSampler", this._randomTexture);
    }
    dispose() {
        this._randomTexture.dispose();
        super.dispose();
    }
    _createRandomTexture() {
        const size = 128;
        const data = new Uint8Array(size * size * 4);
        const randVector = Vector2.Zero();
        for (let index = 0; index < data.length;) {
            randVector.set(RandomRange(0, 1), RandomRange(0, 1)).normalize().scaleInPlace(255);
            data[index++] = Math.floor(randVector.x);
            data[index++] = Math.floor(randVector.y);
            data[index++] = 0;
            data[index++] = 255;
        }
        const texture = RawTexture.CreateRGBATexture(data, size, size, this._scene, false, false, 2);
        texture.name = "SSAORandomTexture";
        texture.wrapU = Texture.WRAP_ADDRESSMODE;
        texture.wrapV = Texture.WRAP_ADDRESSMODE;
        this._randomTexture = texture;
    }
    //Van der Corput radical inverse
    _radicalInverseVdC(i) {
        this._bits[0] = i;
        this._bits[0] = ((this._bits[0] << 16) | (this._bits[0] >> 16)) >>> 0;
        this._bits[0] = ((this._bits[0] & 0x55555555) << 1) | (((this._bits[0] & 0xaaaaaaaa) >>> 1) >>> 0);
        this._bits[0] = ((this._bits[0] & 0x33333333) << 2) | (((this._bits[0] & 0xcccccccc) >>> 2) >>> 0);
        this._bits[0] = ((this._bits[0] & 0x0f0f0f0f) << 4) | (((this._bits[0] & 0xf0f0f0f0) >>> 4) >>> 0);
        this._bits[0] = ((this._bits[0] & 0x00ff00ff) << 8) | (((this._bits[0] & 0xff00ff00) >>> 8) >>> 0);
        return this._bits[0] * 2.3283064365386963e-10; // / 0x100000000 or / 4294967296
    }
    _hammersley(i, n) {
        return [i / n, this._radicalInverseVdC(i)];
    }
    _hemisphereSampleUniform(u, v) {
        const phi = v * 2.0 * Math.PI;
        // rejecting samples that are close to tangent plane to avoid z-fighting artifacts
        const cosTheta = 1.0 - u * 0.85;
        const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);
        return new Vector3(Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta);
    }
    _generateHemisphere() {
        const numSamples = this.samples;
        const result = [];
        let vector;
        let i = 0;
        while (i < numSamples) {
            if (numSamples < 16) {
                vector = this._hemisphereSampleUniform(Math.random(), Math.random());
            }
            else {
                const rand = this._hammersley(i, numSamples);
                vector = this._hemisphereSampleUniform(rand[0], rand[1]);
            }
            result.push(vector.x, vector.y, vector.z);
            i++;
        }
        return result;
    }
    _getDefinesForSSAO() {
        const epsilon = this._epsilon ?? 0.02;
        const samples = this._samples ?? 8;
        let defines = `#define SSAO\n#define SAMPLES ${samples}\n#define EPSILON ${epsilon.toFixed(4)}`;
        if (this.camera?.mode === Camera.ORTHOGRAPHIC_CAMERA) {
            defines += `\n#define ORTHOGRAPHIC_CAMERA`;
        }
        return defines;
    }
}
ThinSSAO2PostProcess.ORTHO_DEPTH_PROJECTION = [1, 0, 0, 0, 1, 0, 0, 0, 1];
ThinSSAO2PostProcess.PERSPECTIVE_DEPTH_PROJECTION = [0, 0, 0, 0, 0, 0, 1, 1, 1];
ThinSSAO2PostProcess.FragmentUrl = "ssao2";
ThinSSAO2PostProcess.Uniforms = [
    "sampleSphere",
    "samplesFactor",
    "randTextureTiles",
    "totalStrength",
    "radius",
    "base",
    "range",
    "projection",
    "near",
    "texelSize",
    "xViewport",
    "yViewport",
    "viewport",
    "maxZ",
    "minZAspect",
    "depthProjection",
];
ThinSSAO2PostProcess.Samplers = ["randomSampler", "depthSampler", "normalSampler"];

/**
 * @internal
 */
class ThinSSAO2BlurPostProcess extends EffectWrapper {
    _gatherImports(useWebGPU, list) {
        if (useWebGPU) {
            this._webGPUReady = true;
            list.push(__vitePreload(() => import('./ssao2.fragment-3Rjrq9Rx.js'),true              ?__vite__mapDeps([0,1,2,3]):void 0,import.meta.url));
        }
        else {
            list.push(__vitePreload(() => import('./ssao2.fragment-DG9KEUgV.js'),true              ?__vite__mapDeps([4,1,2,3]):void 0,import.meta.url));
        }
    }
    constructor(name, engine = null, isHorizontal, options) {
        super({
            ...options,
            name,
            engine: engine || Engine.LastCreatedEngine,
            useShaderStore: true,
            useAsPostProcess: true,
            fragmentShader: ThinSSAO2BlurPostProcess.FragmentUrl,
            uniforms: ThinSSAO2BlurPostProcess.Uniforms,
            samplers: ThinSSAO2BlurPostProcess.Samplers,
            defines: "#define BLUR\n" + (isHorizontal ? "#define BLUR_H\n" : ""),
        });
        this._bypassBlur = false;
        this.textureSize = 0;
        this.bilateralSamples = 16;
        this.bilateralSoften = 0;
        this.bilateralTolerance = 0;
        this._expensiveBlur = true;
        this._isHorizontal = isHorizontal;
        const defines = this._getDefinesForBlur(this.expensiveBlur, this.bypassBlur);
        const samplers = this._getSamplersForBlur(this.bypassBlur);
        this.updateEffect(defines, null, samplers);
    }
    set bypassBlur(b) {
        const defines = this._getDefinesForBlur(this.expensiveBlur, b);
        const samplers = this._getSamplersForBlur(b);
        this.updateEffect(defines, null, samplers);
        this._bypassBlur = b;
    }
    get bypassBlur() {
        return this._bypassBlur;
    }
    set expensiveBlur(b) {
        const defines = this._getDefinesForBlur(b, this._bypassBlur);
        this.updateEffect(defines);
        this._expensiveBlur = b;
    }
    get expensiveBlur() {
        return this._expensiveBlur;
    }
    bind(noDefaultBindings = false) {
        super.bind(noDefaultBindings);
        const effect = this._drawWrapper.effect;
        effect.setFloat("outSize", this.textureSize);
        effect.setInt("samples", this.bilateralSamples);
        effect.setFloat("soften", this.bilateralSoften);
        effect.setFloat("tolerance", this.bilateralTolerance);
    }
    _getSamplersForBlur(disabled) {
        return disabled ? ["textureSampler"] : ["textureSampler", "depthSampler"];
    }
    _getDefinesForBlur(bilateral, disabled) {
        let define = "#define BLUR\n";
        if (disabled) {
            define += "#define BLUR_BYPASS\n";
        }
        if (!bilateral) {
            define += "#define BLUR_LEGACY\n";
        }
        return this._isHorizontal ? define + "#define BLUR_H\n" : define;
    }
}
ThinSSAO2BlurPostProcess.FragmentUrl = "ssao2";
ThinSSAO2BlurPostProcess.Uniforms = ["outSize", "samples", "soften", "tolerance"];
ThinSSAO2BlurPostProcess.Samplers = ["textureSampler", "depthSampler"];

/**
 * @internal
 */
class ThinSSAO2CombinePostProcess extends EffectWrapper {
    _gatherImports(useWebGPU, list) {
        if (useWebGPU) {
            this._webGPUReady = true;
            list.push(__vitePreload(() => import('./ssaoCombine.fragment-DaSWDR4P.js'),true              ?__vite__mapDeps([5,1,2,3]):void 0,import.meta.url));
        }
        else {
            list.push(__vitePreload(() => import('./ssaoCombine.fragment-Bvl-J2BM.js'),true              ?__vite__mapDeps([6,1,2,3]):void 0,import.meta.url));
        }
    }
    constructor(name, engine = null, options) {
        super({
            ...options,
            name,
            engine: engine || Engine.LastCreatedEngine,
            useShaderStore: true,
            useAsPostProcess: true,
            fragmentShader: ThinSSAO2CombinePostProcess.FragmentUrl,
            uniforms: ThinSSAO2CombinePostProcess.Uniforms,
            samplers: ThinSSAO2CombinePostProcess.Samplers,
        });
        this.camera = null;
        this.useViewportInCombineStage = true;
    }
    bind(noDefaultBindings = false) {
        super.bind(noDefaultBindings);
        const effect = this._drawWrapper.effect;
        if (this.camera) {
            const viewport = this.camera.viewport;
            if (this.useViewportInCombineStage) {
                effect.setVector4("viewport", TmpVectors.Vector4[0].copyFromFloats(viewport.x, viewport.y, viewport.width, viewport.height));
            }
            else {
                effect.setVector4("viewport", TmpVectors.Vector4[0].copyFromFloats(0, 0, 1, 1));
            }
        }
    }
}
ThinSSAO2CombinePostProcess.FragmentUrl = "ssaoCombine";
ThinSSAO2CombinePostProcess.Uniforms = ["viewport"];
ThinSSAO2CombinePostProcess.Samplers = ["originalColor"];

/**
 * The SSAO2 rendering pipeline is used to generate ambient occlusion effects.
 */
class ThinSSAO2RenderingPipeline {
    /**
     * The camera to which the rendering pipeline will be applied.
     */
    get camera() {
        return this._ssaoPostProcess.camera;
    }
    set camera(camera) {
        this._ssaoPostProcess.camera = camera;
        this._ssaoCombinePostProcess.camera = camera;
    }
    /**
     * Number of samples used for the SSAO calculations. Default value is 8.
     */
    set samples(n) {
        this._ssaoPostProcess.samples = n;
    }
    get samples() {
        return this._ssaoPostProcess.samples;
    }
    /**
     * The output strength of the SSAO post-process. Default value is 1.0.
     */
    get totalStrength() {
        return this._ssaoPostProcess.totalStrength;
    }
    set totalStrength(value) {
        this._ssaoPostProcess.totalStrength = value;
    }
    /**
     * The radius around the analyzed pixel used by the SSAO post-process. Default value is 2.0
     */
    get radius() {
        return this._ssaoPostProcess.radius;
    }
    set radius(value) {
        this._ssaoPostProcess.radius = value;
    }
    /**
     * Maximum depth value to still render AO. A smooth falloff makes the dimming more natural, so there will be no abrupt shading change.
     */
    get maxZ() {
        return this._ssaoPostProcess.maxZ;
    }
    set maxZ(value) {
        this._ssaoPostProcess.maxZ = value;
    }
    /**
     * In order to save performances, SSAO radius is clamped on close geometry. This ratio changes by how much.
     */
    get minZAspect() {
        return this._ssaoPostProcess.minZAspect;
    }
    set minZAspect(value) {
        this._ssaoPostProcess.minZAspect = value;
    }
    /**
     * The base color of the SSAO post-process
     * The final result is "base + ssao" between [0, 1]
     */
    get base() {
        return this._ssaoPostProcess.base;
    }
    set base(value) {
        this._ssaoPostProcess.base = value;
    }
    /**
     * Used in SSAO calculations to compensate for accuracy issues with depth values. Default 0.02.
     *
     * Normally you do not need to change this value, but you can experiment with it if you get a lot of in false self-occlusion on flat surfaces when using fewer than 16 samples. Useful range is normally [0..0.1] but higher values is allowed.
     */
    get epsilon() {
        return this._ssaoPostProcess.epsilon;
    }
    set epsilon(n) {
        this._ssaoPostProcess.epsilon = n;
    }
    /**
     * Skips the denoising (blur) stage of the SSAO calculations.
     *
     * Useful to temporarily set while experimenting with the other SSAO2 settings.
     */
    set bypassBlur(b) {
        this._ssaoBlurXPostProcess.bypassBlur = b;
        this._ssaoBlurYPostProcess.bypassBlur = b;
    }
    get bypassBlur() {
        return this._ssaoBlurXPostProcess.bypassBlur;
    }
    /**
     * Enables the configurable bilateral denoising (blurring) filter. Default is true.
     * Set to false to instead use a legacy bilateral filter that can't be configured.
     *
     * The denoising filter runs after the SSAO calculations and is a very important step. Both options results in a so called bilateral being used, but the "expensive" one can be
     * configured in several ways to fit your scene.
     */
    set expensiveBlur(b) {
        this._ssaoBlurXPostProcess.expensiveBlur = b;
        this._ssaoBlurYPostProcess.expensiveBlur = b;
    }
    get expensiveBlur() {
        return this._ssaoBlurXPostProcess.expensiveBlur;
    }
    /**
     * The number of samples the bilateral filter uses in both dimensions when denoising the SSAO calculations. Default value is 16.
     *
     * A higher value should result in smoother shadows but will use more processing time in the shaders.
     *
     * A high value can cause the shadows to get to blurry or create visible artifacts (bands) near sharp details in the geometry. The artifacts can sometimes be mitigated by increasing the bilateralSoften setting.
     */
    get bilateralSamples() {
        return this._ssaoBlurXPostProcess.bilateralSamples;
    }
    set bilateralSamples(n) {
        this._ssaoBlurXPostProcess.bilateralSamples = n;
        this._ssaoBlurYPostProcess.bilateralSamples = n;
    }
    /**
     * Controls the shape of the denoising kernel used by the bilateral filter. Default value is 0.
     *
     * By default the bilateral filter acts like a box-filter, treating all samples on the same depth with equal weights. This is effective to maximize the denoising effect given a limited set of samples. However, it also often results in visible ghosting around sharp shadow regions and can spread out lines over large areas so they are no longer visible.
     *
     * Increasing this setting will make the filter pay less attention to samples further away from the center sample, reducing many artifacts but at the same time increasing noise.
     *
     * Useful value range is [0..1].
     */
    get bilateralSoften() {
        return this._ssaoBlurXPostProcess.bilateralSoften;
    }
    set bilateralSoften(n) {
        this._ssaoBlurXPostProcess.bilateralSoften = n;
        this._ssaoBlurYPostProcess.bilateralSoften = n;
    }
    /**
     * How forgiving the bilateral denoiser should be when rejecting samples. Default value is 0.
     *
     * A higher value results in the bilateral filter being more forgiving and thus doing a better job at denoising slanted and curved surfaces, but can lead to shadows spreading out around corners or between objects that are close to each other depth wise.
     *
     * Useful value range is normally [0..1], but higher values are allowed.
     */
    get bilateralTolerance() {
        return this._ssaoBlurXPostProcess.bilateralTolerance;
    }
    set bilateralTolerance(n) {
        this._ssaoBlurXPostProcess.bilateralTolerance = n;
        this._ssaoBlurYPostProcess.bilateralTolerance = n;
    }
    /**
     * Indicates that the combine stage should use the current camera viewport to render the SSAO result on only a portion of the output texture (default: true).
     */
    get useViewportInCombineStage() {
        return this._ssaoCombinePostProcess.useViewportInCombineStage;
    }
    set useViewportInCombineStage(b) {
        this._ssaoCombinePostProcess.useViewportInCombineStage = b;
    }
    /**
     * Checks if all the post processes in the pipeline are ready.
     * @returns true if all the post processes in the pipeline are ready
     */
    isReady() {
        return this._ssaoPostProcess.isReady() && this._ssaoBlurXPostProcess.isReady() && this._ssaoBlurYPostProcess.isReady() && this._ssaoCombinePostProcess.isReady();
    }
    /**
     * Constructor of the SSR rendering pipeline
     * @param name The rendering pipeline name
     * @param scene The scene linked to this pipeline
     */
    constructor(name, scene) {
        this.name = name;
        this._scene = scene;
        this._ssaoPostProcess = new ThinSSAO2PostProcess(this.name, this._scene);
        this._ssaoBlurXPostProcess = new ThinSSAO2BlurPostProcess(this.name + " BlurX", this._scene.getEngine(), true);
        this._ssaoBlurYPostProcess = new ThinSSAO2BlurPostProcess(this.name + " BlurY", this._scene.getEngine(), false);
        this._ssaoCombinePostProcess = new ThinSSAO2CombinePostProcess(this.name + " Combiner", this._scene.getEngine());
    }
    /**
     * Disposes of the pipeline
     */
    dispose() {
        this._ssaoPostProcess?.dispose();
        this._ssaoBlurXPostProcess?.dispose();
        this._ssaoBlurYPostProcess?.dispose();
        this._ssaoCombinePostProcess?.dispose();
    }
}

/**
 * Render pipeline to produce ssao effect
 */
class SSAO2RenderingPipeline extends PostProcessRenderPipeline {
    /**
     * The output strength of the SSAO post-process. Default value is 1.0.
     */
    get totalStrength() {
        return this._thinSSAORenderingPipeline.totalStrength;
    }
    set totalStrength(value) {
        this._thinSSAORenderingPipeline.totalStrength = value;
    }
    /**
     * Maximum depth value to still render AO. A smooth falloff makes the dimming more natural, so there will be no abrupt shading change.
     */
    get maxZ() {
        return this._thinSSAORenderingPipeline.maxZ;
    }
    set maxZ(value) {
        this._thinSSAORenderingPipeline.maxZ = value;
    }
    /**
     * In order to save performances, SSAO radius is clamped on close geometry. This ratio changes by how much.
     */
    get minZAspect() {
        return this._thinSSAORenderingPipeline.minZAspect;
    }
    set minZAspect(value) {
        this._thinSSAORenderingPipeline.minZAspect = value;
    }
    /**
     * Used in SSAO calculations to compensate for accuracy issues with depth values. Default 0.02.
     *
     * Normally you do not need to change this value, but you can experiment with it if you get a lot of in false self-occlusion on flat surfaces when using fewer than 16 samples. Useful range is normally [0..0.1] but higher values is allowed.
     */
    set epsilon(n) {
        this._thinSSAORenderingPipeline.epsilon = n;
    }
    get epsilon() {
        return this._thinSSAORenderingPipeline.epsilon;
    }
    /**
     * Number of samples used for the SSAO calculations. Default value is 8.
     */
    set samples(n) {
        this._thinSSAORenderingPipeline.samples = n;
    }
    get samples() {
        return this._thinSSAORenderingPipeline.samples;
    }
    /**
     * Number of samples to use for antialiasing.
     */
    set textureSamples(n) {
        this._textureSamples = n;
        if (this._prePassRenderer) {
            this._prePassRenderer.samples = n;
        }
        else {
            this._originalColorPostProcess.samples = n;
        }
    }
    get textureSamples() {
        return this._textureSamples;
    }
    get _geometryBufferRenderer() {
        if (!this._forceGeometryBuffer) {
            return null;
        }
        return this._forcedGeometryBuffer ?? this._scene.geometryBufferRenderer;
    }
    get _prePassRenderer() {
        if (this._forceGeometryBuffer) {
            return null;
        }
        return this._scene.prePassRenderer;
    }
    /**
     * The radius around the analyzed pixel used by the SSAO post-process. Default value is 2.0
     */
    get radius() {
        return this._thinSSAORenderingPipeline.radius;
    }
    set radius(value) {
        this._thinSSAORenderingPipeline.radius = value;
    }
    /**
     * The base color of the SSAO post-process
     * The final result is "base + ssao" between [0, 1]
     */
    get base() {
        return this._thinSSAORenderingPipeline.base;
    }
    set base(value) {
        this._thinSSAORenderingPipeline.base = value;
    }
    /**
     * Skips the denoising (blur) stage of the SSAO calculations.
     *
     * Useful to temporarily set while experimenting with the other SSAO2 settings.
     */
    set bypassBlur(b) {
        this._thinSSAORenderingPipeline.bypassBlur = b;
    }
    get bypassBlur() {
        return this._thinSSAORenderingPipeline.bypassBlur;
    }
    /**
     * Enables the configurable bilateral denoising (blurring) filter. Default is true.
     * Set to false to instead use a legacy bilateral filter that can't be configured.
     *
     * The denoising filter runs after the SSAO calculations and is a very important step. Both options results in a so called bilateral being used, but the "expensive" one can be
     * configured in several ways to fit your scene.
     */
    set expensiveBlur(b) {
        this._thinSSAORenderingPipeline.expensiveBlur = b;
    }
    get expensiveBlur() {
        return this._thinSSAORenderingPipeline.expensiveBlur;
    }
    /**
     * The number of samples the bilateral filter uses in both dimensions when denoising the SSAO calculations. Default value is 16.
     *
     * A higher value should result in smoother shadows but will use more processing time in the shaders.
     *
     * A high value can cause the shadows to get to blurry or create visible artifacts (bands) near sharp details in the geometry. The artifacts can sometimes be mitigated by increasing the bilateralSoften setting.
     */
    get bilateralSamples() {
        return this._thinSSAORenderingPipeline.bilateralSamples;
    }
    set bilateralSamples(n) {
        this._thinSSAORenderingPipeline.bilateralSamples = n;
    }
    /**
     * Controls the shape of the denoising kernel used by the bilateral filter. Default value is 0.
     *
     * By default the bilateral filter acts like a box-filter, treating all samples on the same depth with equal weights. This is effective to maximize the denoising effect given a limited set of samples. However, it also often results in visible ghosting around sharp shadow regions and can spread out lines over large areas so they are no longer visible.
     *
     * Increasing this setting will make the filter pay less attention to samples further away from the center sample, reducing many artifacts but at the same time increasing noise.
     *
     * Useful value range is [0..1].
     */
    get bilateralSoften() {
        return this._thinSSAORenderingPipeline.bilateralSoften;
    }
    set bilateralSoften(n) {
        this._thinSSAORenderingPipeline.bilateralSoften = n;
    }
    /**
     * How forgiving the bilateral denoiser should be when rejecting samples. Default value is 0.
     *
     * A higher value results in the bilateral filter being more forgiving and thus doing a better job at denoising slanted and curved surfaces, but can lead to shadows spreading out around corners or between objects that are close to each other depth wise.
     *
     * Useful value range is normally [0..1], but higher values are allowed.
     */
    get bilateralTolerance() {
        return this._thinSSAORenderingPipeline.bilateralTolerance;
    }
    set bilateralTolerance(n) {
        this._thinSSAORenderingPipeline.bilateralTolerance = n;
    }
    /**
     *  Support test.
     */
    static get IsSupported() {
        const engine = EngineStore.LastCreatedEngine;
        if (!engine) {
            return false;
        }
        return engine._features.supportSSAO2;
    }
    /**
     * Indicates that the combine stage should use the current camera viewport to render the SSAO result on only a portion of the output texture (default: true).
     */
    get useViewportInCombineStage() {
        return this._thinSSAORenderingPipeline.useViewportInCombineStage;
    }
    set useViewportInCombineStage(b) {
        this._thinSSAORenderingPipeline.useViewportInCombineStage = b;
    }
    /**
     * Gets active scene
     */
    get scene() {
        return this._scene;
    }
    /**
     * Creates the SSAO2 rendering pipeline.
     * @param name The rendering pipeline name
     * @param scene The scene linked to this pipeline
     * @param ratio The size of the postprocesses. Can be a number shared between passes or an object for more precision: { ssaoRatio: 0.5, blurRatio: 1.0 }
     * @param cameras The array of cameras that the rendering pipeline will be attached to
     * @param forceGeometryBuffer Set to true if you want to use the legacy geometry buffer renderer. You can also pass an existing instance of GeometryBufferRenderer if you want to use your own geometry buffer renderer.
     * @param textureType The texture type used by the different post processes created by SSAO (default: 0)
     */
    constructor(name, scene, ratio, cameras, forceGeometryBuffer = false, textureType = 0) {
        super(scene.getEngine(), name);
        // Members
        /**
         * @ignore
         * The PassPostProcess id in the pipeline that contains the original scene color
         */
        this.SSAOOriginalSceneColorEffect = "SSAOOriginalSceneColorEffect";
        /**
         * @ignore
         * The SSAO PostProcess id in the pipeline
         */
        this.SSAORenderEffect = "SSAORenderEffect";
        /**
         * @ignore
         * The horizontal blur PostProcess id in the pipeline
         */
        this.SSAOBlurHRenderEffect = "SSAOBlurHRenderEffect";
        /**
         * @ignore
         * The vertical blur PostProcess id in the pipeline
         */
        this.SSAOBlurVRenderEffect = "SSAOBlurVRenderEffect";
        /**
         * @ignore
         * The PostProcess id in the pipeline that combines the SSAO-Blur output with the original scene color (SSAOOriginalSceneColorEffect)
         */
        this.SSAOCombineRenderEffect = "SSAOCombineRenderEffect";
        this._textureSamples = 1;
        this._forcedGeometryBuffer = null;
        /**
         * Force rendering the geometry through geometry buffer.
         */
        this._forceGeometryBuffer = false;
        this._currentCameraMode = -1;
        this._thinSSAORenderingPipeline = new ThinSSAO2RenderingPipeline(name, scene);
        this._scene = scene;
        this._ratio = ratio;
        this._textureType = textureType;
        if (forceGeometryBuffer instanceof GeometryBufferRenderer) {
            this._forceGeometryBuffer = true;
            this._forcedGeometryBuffer = forceGeometryBuffer;
        }
        else {
            this._forceGeometryBuffer = forceGeometryBuffer;
        }
        if (!this.isSupported) {
            Logger.Error("The current engine does not support SSAO 2.");
            return;
        }
        const ssaoRatio = this._ratio.ssaoRatio || ratio;
        const blurRatio = this._ratio.blurRatio || ratio;
        // Set up assets
        if (this._forceGeometryBuffer) {
            if (!this._forcedGeometryBuffer) {
                scene.enableGeometryBufferRenderer();
            }
            if (scene.geometryBufferRenderer?.generateNormalsInWorldSpace) {
                Logger.Error("SSAO2RenderingPipeline does not support generateNormalsInWorldSpace=true for the geometry buffer renderer!");
            }
        }
        else {
            scene.enablePrePassRenderer();
            if (scene.prePassRenderer?.generateNormalsInWorldSpace) {
                Logger.Error("SSAO2RenderingPipeline does not support generateNormalsInWorldSpace=true for the prepass renderer!");
            }
        }
        this._originalColorPostProcess = new PassPostProcess("SSAOOriginalSceneColor", 1.0, null, Texture.BILINEAR_SAMPLINGMODE, scene.getEngine(), undefined, this._textureType);
        this._originalColorPostProcess.onBeforeRenderObservable.add(() => {
            const camera = this._scene.activeCamera;
            this._thinSSAORenderingPipeline._ssaoPostProcess.camera = camera;
            if (camera && this._currentCameraMode !== camera.mode) {
                this._currentCameraMode = camera.mode;
                this._thinSSAORenderingPipeline._ssaoPostProcess.updateEffect();
            }
        });
        this._originalColorPostProcess.samples = this.textureSamples;
        this._createSSAOPostProcess(1.0, textureType);
        this._createBlurPostProcess(ssaoRatio, blurRatio, this._textureType);
        this._createSSAOCombinePostProcess(blurRatio, this._textureType);
        // Set up pipeline
        this.addEffect(new PostProcessRenderEffect(scene.getEngine(), this.SSAOOriginalSceneColorEffect, () => {
            return this._originalColorPostProcess;
        }, true));
        this.addEffect(new PostProcessRenderEffect(scene.getEngine(), this.SSAORenderEffect, () => {
            return this._ssaoPostProcess;
        }, true));
        this.addEffect(new PostProcessRenderEffect(scene.getEngine(), this.SSAOBlurHRenderEffect, () => {
            return this._blurHPostProcess;
        }, true));
        this.addEffect(new PostProcessRenderEffect(scene.getEngine(), this.SSAOBlurVRenderEffect, () => {
            return this._blurVPostProcess;
        }, true));
        this.addEffect(new PostProcessRenderEffect(scene.getEngine(), this.SSAOCombineRenderEffect, () => {
            return this._ssaoCombinePostProcess;
        }, true));
        // Finish
        scene.postProcessRenderPipelineManager.addPipeline(this);
        if (cameras) {
            scene.postProcessRenderPipelineManager.attachCamerasToRenderPipeline(name, cameras);
        }
    }
    // Public Methods
    /**
     * Get the class name
     * @returns "SSAO2RenderingPipeline"
     */
    getClassName() {
        return "SSAO2RenderingPipeline";
    }
    /**
     * Removes the internal pipeline assets and detaches the pipeline from the scene cameras
     * @param disableGeometryBufferRenderer Set to true if you want to disable the Geometry Buffer renderer
     */
    dispose(disableGeometryBufferRenderer = false) {
        for (let i = 0; i < this._scene.cameras.length; i++) {
            const camera = this._scene.cameras[i];
            this._originalColorPostProcess.dispose(camera);
            this._ssaoPostProcess.dispose(camera);
            this._blurHPostProcess.dispose(camera);
            this._blurVPostProcess.dispose(camera);
            this._ssaoCombinePostProcess.dispose(camera);
        }
        if (disableGeometryBufferRenderer && !this._forcedGeometryBuffer) {
            this._scene.disableGeometryBufferRenderer();
        }
        this._scene.postProcessRenderPipelineManager.detachCamerasFromRenderPipeline(this._name, this._scene.cameras);
        this._scene.postProcessRenderPipelineManager.removePipeline(this._name);
        this._thinSSAORenderingPipeline.dispose();
        super.dispose();
    }
    // Private Methods
    /** @internal */
    _rebuild() {
        super._rebuild();
    }
    _createBlurPostProcess(ssaoRatio, blurRatio, textureType) {
        this._blurHPostProcess = this._createBlurFilter("BlurH", ssaoRatio, textureType, true);
        this._blurVPostProcess = this._createBlurFilter("BlurV", blurRatio, textureType, false);
    }
    _createBlurFilter(name, ratio, textureType, horizontal) {
        const blurFilter = new PostProcess(name, ThinSSAO2BlurPostProcess.FragmentUrl, {
            size: ratio,
            samplingMode: 2,
            engine: this._scene.getEngine(),
            textureType: this._textureType,
            effectWrapper: horizontal ? this._thinSSAORenderingPipeline._ssaoBlurXPostProcess : this._thinSSAORenderingPipeline._ssaoBlurYPostProcess,
        });
        blurFilter.onApply = (effect) => {
            const ratio = this._ratio.blurRatio || this._ratio;
            const ssaoCombineSize = horizontal ? this._originalColorPostProcess.width * ratio : this._originalColorPostProcess.height * ratio;
            const originalColorSize = horizontal ? this._originalColorPostProcess.width : this._originalColorPostProcess.height;
            this._thinSSAORenderingPipeline._ssaoBlurXPostProcess.textureSize = ssaoCombineSize > 0 ? ssaoCombineSize : originalColorSize;
            this._thinSSAORenderingPipeline._ssaoBlurYPostProcess.textureSize = ssaoCombineSize > 0 ? ssaoCombineSize : originalColorSize;
            if (this._geometryBufferRenderer) {
                effect.setTexture("depthSampler", this._geometryBufferRenderer.getGBuffer().textures[0]);
            }
            else if (this._prePassRenderer) {
                effect.setTexture("depthSampler", this._prePassRenderer.getRenderTarget().textures[this._prePassRenderer.getIndex(5)]);
            }
        };
        blurFilter.samples = this.textureSamples;
        blurFilter.autoClear = false;
        return blurFilter;
    }
    _getTextureSize() {
        const engine = this._scene.getEngine();
        const prePassRenderer = this._prePassRenderer;
        let textureSize = { width: engine.getRenderWidth(), height: engine.getRenderHeight() };
        if (prePassRenderer && this._scene.activeCamera?._getFirstPostProcess() === this._ssaoPostProcess) {
            const renderTarget = prePassRenderer.getRenderTarget();
            if (renderTarget && renderTarget.textures) {
                textureSize = renderTarget.textures[prePassRenderer.getIndex(4)].getSize();
            }
        }
        else if (this._ssaoPostProcess.inputTexture) {
            textureSize.width = this._ssaoPostProcess.inputTexture.width;
            textureSize.height = this._ssaoPostProcess.inputTexture.height;
        }
        return textureSize;
    }
    _createSSAOPostProcess(ratio, textureType) {
        this._ssaoPostProcess = new PostProcess("ssao", ThinSSAO2PostProcess.FragmentUrl, {
            size: ratio,
            samplingMode: 2,
            engine: this._scene.getEngine(),
            textureType,
            effectWrapper: this._thinSSAORenderingPipeline._ssaoPostProcess,
        });
        this._ssaoPostProcess.autoClear = false;
        this._ssaoPostProcess.onApply = (effect) => {
            if (this._geometryBufferRenderer) {
                effect.setTexture("depthSampler", this._geometryBufferRenderer.getGBuffer().textures[0]);
                effect.setTexture("normalSampler", this._geometryBufferRenderer.getGBuffer().textures[1]);
            }
            else if (this._prePassRenderer) {
                effect.setTexture("depthSampler", this._prePassRenderer.getRenderTarget().textures[this._prePassRenderer.getIndex(5)]);
                effect.setTexture("normalSampler", this._prePassRenderer.getRenderTarget().textures[this._prePassRenderer.getIndex(6)]);
            }
            const textureSize = this._getTextureSize();
            this._thinSSAORenderingPipeline._ssaoPostProcess.textureWidth = textureSize.width;
            this._thinSSAORenderingPipeline._ssaoPostProcess.textureHeight = textureSize.height;
        };
        this._ssaoPostProcess.samples = this.textureSamples;
        if (!this._forceGeometryBuffer) {
            this._ssaoPostProcess._prePassEffectConfiguration = new SSAO2Configuration();
        }
    }
    _createSSAOCombinePostProcess(ratio, textureType) {
        this._ssaoCombinePostProcess = new PostProcess("ssaoCombine", ThinSSAO2CombinePostProcess.FragmentUrl, {
            size: ratio,
            samplingMode: 2,
            engine: this._scene.getEngine(),
            textureType,
            effectWrapper: this._thinSSAORenderingPipeline._ssaoCombinePostProcess,
        });
        this._ssaoCombinePostProcess.onApply = (effect) => {
            this._thinSSAORenderingPipeline._ssaoCombinePostProcess.camera = this._scene.activeCamera;
            effect.setTextureFromPostProcessOutput("originalColor", this._originalColorPostProcess);
        };
        this._ssaoCombinePostProcess.autoClear = false;
        this._ssaoCombinePostProcess.samples = this.textureSamples;
    }
    /**
     * Serialize the rendering pipeline (Used when exporting)
     * @returns the serialized object
     */
    serialize() {
        const serializationObject = SerializationHelper.Serialize(this);
        serializationObject.customType = "SSAO2RenderingPipeline";
        return serializationObject;
    }
    /**
     * Parse the serialized pipeline
     * @param source Source pipeline.
     * @param scene The scene to load the pipeline to.
     * @param rootUrl The URL of the serialized pipeline.
     * @returns An instantiated pipeline from the serialized object.
     */
    static Parse(source, scene, rootUrl) {
        return SerializationHelper.Parse(() => new SSAO2RenderingPipeline(source._name, scene, source._ratio, undefined, source._forceGeometryBuffer, source._textureType), source, scene, rootUrl);
    }
}
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "totalStrength", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "maxZ", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "minZAspect", null);
__decorate([
    serialize("epsilon")
], SSAO2RenderingPipeline.prototype, "epsilon", null);
__decorate([
    serialize("samples")
], SSAO2RenderingPipeline.prototype, "samples", null);
__decorate([
    serialize("textureSamples")
], SSAO2RenderingPipeline.prototype, "_textureSamples", void 0);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "_forceGeometryBuffer", void 0);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "_ratio", void 0);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "_textureType", void 0);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "radius", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "base", null);
__decorate([
    serialize("bypassBlur")
], SSAO2RenderingPipeline.prototype, "bypassBlur", null);
__decorate([
    serialize("expensiveBlur")
], SSAO2RenderingPipeline.prototype, "expensiveBlur", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "bilateralSamples", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "bilateralSoften", null);
__decorate([
    serialize()
], SSAO2RenderingPipeline.prototype, "bilateralTolerance", null);
RegisterClass("BABYLON.SSAO2RenderingPipeline", SSAO2RenderingPipeline);

export { SSAO2RenderingPipeline };
//# sourceMappingURL=ssao2RenderingPipeline-IqQsV7Sg.js.map
