const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./imageProcessing.fragment-B5o0MiGc.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./imageProcessingFunctions-ClbSBfeF.js","./helperFunctions-qLR1vvLh.js","./imageProcessing.fragment-DCkv0aPu.js","./imageProcessingFunctions-CBu7O15Q.js","./helperFunctions-CiFVmLcr.js"])))=>i.map(i=>d[i]);
import { aE as EffectWrapper, aq as EngineStore, W as ImageProcessingConfiguration, bn as Engine, _ as __decorate, s as serialize, aA as PostProcess, bz as MultiRenderTarget, b9 as Color4, o as Material, be as _WarnImport, a as SceneComponentConstants, b as Scene, L as Logger } from './index-CxR0Yf_7.js';
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { G as GeometryBufferRenderer } from './geometryBufferRenderer-Dvq35Qkr.js';
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
 * Post process used to apply image processing to a scene
 */
class ThinImageProcessingPostProcess extends EffectWrapper {
    _gatherImports(useWebGPU, list) {
        if (useWebGPU) {
            this._webGPUReady = true;
            list.push(__vitePreload(() => import('./imageProcessing.fragment-B5o0MiGc.js'),true              ?__vite__mapDeps([0,1,2,3,4,5]):void 0,import.meta.url));
        }
        else {
            list.push(__vitePreload(() => import('./imageProcessing.fragment-DCkv0aPu.js'),true              ?__vite__mapDeps([6,1,2,3,7,8]):void 0,import.meta.url));
        }
    }
    /**
     * Gets the image processing configuration used either in this material.
     */
    get imageProcessingConfiguration() {
        return this._imageProcessingConfiguration;
    }
    /**
     * Sets the Default image processing configuration used either in the this material.
     *
     * If sets to null, the scene one is in use.
     */
    set imageProcessingConfiguration(value) {
        // We are almost sure it is applied by post process as
        // We are in the post process :-)
        value.applyByPostProcess = true;
        this._attachImageProcessingConfiguration(value);
    }
    /**
     * Attaches a new image processing configuration to the PBR Material.
     * @param configuration
     * @param doNotBuild
     */
    _attachImageProcessingConfiguration(configuration, doNotBuild = false) {
        if (configuration === this._imageProcessingConfiguration) {
            return;
        }
        // Detaches observer.
        if (this._imageProcessingConfiguration && this._imageProcessingObserver) {
            this._imageProcessingConfiguration.onUpdateParameters.remove(this._imageProcessingObserver);
        }
        // Pick the scene configuration if needed.
        if (!configuration) {
            let scene = this.options.scene;
            if (!scene) {
                const engine = this.options.engine;
                if (engine && engine.scenes) {
                    const scenes = engine.scenes;
                    scene = scenes[scenes.length - 1];
                }
                else {
                    scene = EngineStore.LastCreatedScene;
                }
            }
            if (scene) {
                this._imageProcessingConfiguration = scene.imageProcessingConfiguration;
            }
            else {
                this._imageProcessingConfiguration = new ImageProcessingConfiguration();
            }
        }
        else {
            this._imageProcessingConfiguration = configuration;
        }
        // Attaches observer.
        if (this._imageProcessingConfiguration) {
            this._imageProcessingObserver = this._imageProcessingConfiguration.onUpdateParameters.add(() => {
                this._updateParameters();
            });
        }
        // Ensure the effect will be rebuilt.
        if (!doNotBuild) {
            this._updateParameters();
        }
    }
    /**
     * Gets Color curves setup used in the effect if colorCurvesEnabled is set to true .
     */
    get colorCurves() {
        return this.imageProcessingConfiguration.colorCurves;
    }
    /**
     * Sets Color curves setup used in the effect if colorCurvesEnabled is set to true .
     */
    set colorCurves(value) {
        this.imageProcessingConfiguration.colorCurves = value;
    }
    /**
     * Gets whether the color curves effect is enabled.
     */
    get colorCurvesEnabled() {
        return this.imageProcessingConfiguration.colorCurvesEnabled;
    }
    /**
     * Sets whether the color curves effect is enabled.
     */
    set colorCurvesEnabled(value) {
        this.imageProcessingConfiguration.colorCurvesEnabled = value;
    }
    /**
     * Gets Color grading LUT texture used in the effect if colorGradingEnabled is set to true.
     */
    get colorGradingTexture() {
        return this.imageProcessingConfiguration.colorGradingTexture;
    }
    /**
     * Sets Color grading LUT texture used in the effect if colorGradingEnabled is set to true.
     */
    set colorGradingTexture(value) {
        this.imageProcessingConfiguration.colorGradingTexture = value;
    }
    /**
     * Gets whether the color grading effect is enabled.
     */
    get colorGradingEnabled() {
        return this.imageProcessingConfiguration.colorGradingEnabled;
    }
    /**
     * Gets whether the color grading effect is enabled.
     */
    set colorGradingEnabled(value) {
        this.imageProcessingConfiguration.colorGradingEnabled = value;
    }
    /**
     * Gets exposure used in the effect.
     */
    get exposure() {
        return this.imageProcessingConfiguration.exposure;
    }
    /**
     * Sets exposure used in the effect.
     */
    set exposure(value) {
        this.imageProcessingConfiguration.exposure = value;
    }
    /**
     * Gets whether tonemapping is enabled or not.
     */
    get toneMappingEnabled() {
        return this._imageProcessingConfiguration.toneMappingEnabled;
    }
    /**
     * Sets whether tonemapping is enabled or not
     */
    set toneMappingEnabled(value) {
        this._imageProcessingConfiguration.toneMappingEnabled = value;
    }
    /**
     * Gets the type of tone mapping effect.
     */
    get toneMappingType() {
        return this._imageProcessingConfiguration.toneMappingType;
    }
    /**
     * Sets the type of tone mapping effect.
     */
    set toneMappingType(value) {
        this._imageProcessingConfiguration.toneMappingType = value;
    }
    /**
     * Gets contrast used in the effect.
     */
    get contrast() {
        return this.imageProcessingConfiguration.contrast;
    }
    /**
     * Sets contrast used in the effect.
     */
    set contrast(value) {
        this.imageProcessingConfiguration.contrast = value;
    }
    /**
     * Gets Vignette stretch size.
     */
    get vignetteStretch() {
        return this.imageProcessingConfiguration.vignetteStretch;
    }
    /**
     * Sets Vignette stretch size.
     */
    set vignetteStretch(value) {
        this.imageProcessingConfiguration.vignetteStretch = value;
    }
    /**
     * Gets Vignette center X Offset.
     * @deprecated use vignetteCenterX instead
     */
    get vignetteCentreX() {
        return this.imageProcessingConfiguration.vignetteCenterX;
    }
    /**
     * Sets Vignette center X Offset.
     * @deprecated use vignetteCenterX instead
     */
    set vignetteCentreX(value) {
        this.imageProcessingConfiguration.vignetteCenterX = value;
    }
    /**
     * Gets Vignette center Y Offset.
     * @deprecated use vignetteCenterY instead
     */
    get vignetteCentreY() {
        return this.imageProcessingConfiguration.vignetteCenterY;
    }
    /**
     * Sets Vignette center Y Offset.
     * @deprecated use vignetteCenterY instead
     */
    set vignetteCentreY(value) {
        this.imageProcessingConfiguration.vignetteCenterY = value;
    }
    /**
     * Vignette center Y Offset.
     */
    get vignetteCenterY() {
        return this.imageProcessingConfiguration.vignetteCenterY;
    }
    set vignetteCenterY(value) {
        this.imageProcessingConfiguration.vignetteCenterY = value;
    }
    /**
     * Vignette center X Offset.
     */
    get vignetteCenterX() {
        return this.imageProcessingConfiguration.vignetteCenterX;
    }
    set vignetteCenterX(value) {
        this.imageProcessingConfiguration.vignetteCenterX = value;
    }
    /**
     * Gets Vignette weight or intensity of the vignette effect.
     */
    get vignetteWeight() {
        return this.imageProcessingConfiguration.vignetteWeight;
    }
    /**
     * Sets Vignette weight or intensity of the vignette effect.
     */
    set vignetteWeight(value) {
        this.imageProcessingConfiguration.vignetteWeight = value;
    }
    /**
     * Gets Color of the vignette applied on the screen through the chosen blend mode (vignetteBlendMode)
     * if vignetteEnabled is set to true.
     */
    get vignetteColor() {
        return this.imageProcessingConfiguration.vignetteColor;
    }
    /**
     * Sets Color of the vignette applied on the screen through the chosen blend mode (vignetteBlendMode)
     * if vignetteEnabled is set to true.
     */
    set vignetteColor(value) {
        this.imageProcessingConfiguration.vignetteColor = value;
    }
    /**
     * Gets Camera field of view used by the Vignette effect.
     */
    get vignetteCameraFov() {
        return this.imageProcessingConfiguration.vignetteCameraFov;
    }
    /**
     * Sets Camera field of view used by the Vignette effect.
     */
    set vignetteCameraFov(value) {
        this.imageProcessingConfiguration.vignetteCameraFov = value;
    }
    /**
     * Gets the vignette blend mode allowing different kind of effect.
     */
    get vignetteBlendMode() {
        return this.imageProcessingConfiguration.vignetteBlendMode;
    }
    /**
     * Sets the vignette blend mode allowing different kind of effect.
     */
    set vignetteBlendMode(value) {
        this.imageProcessingConfiguration.vignetteBlendMode = value;
    }
    /**
     * Gets whether the vignette effect is enabled.
     */
    get vignetteEnabled() {
        return this.imageProcessingConfiguration.vignetteEnabled;
    }
    /**
     * Sets whether the vignette effect is enabled.
     */
    set vignetteEnabled(value) {
        this.imageProcessingConfiguration.vignetteEnabled = value;
    }
    /**
     * Gets intensity of the dithering effect.
     */
    get ditheringIntensity() {
        return this.imageProcessingConfiguration.ditheringIntensity;
    }
    /**
     * Sets intensity of the dithering effect.
     */
    set ditheringIntensity(value) {
        this.imageProcessingConfiguration.ditheringIntensity = value;
    }
    /**
     * Gets whether the dithering effect is enabled.
     */
    get ditheringEnabled() {
        return this.imageProcessingConfiguration.ditheringEnabled;
    }
    /**
     * Sets whether the dithering effect is enabled.
     */
    set ditheringEnabled(value) {
        this.imageProcessingConfiguration.ditheringEnabled = value;
    }
    /**
     * Gets whether the input of the processing is in Gamma or Linear Space.
     */
    get fromLinearSpace() {
        return this._fromLinearSpace;
    }
    /**
     * Sets whether the input of the processing is in Gamma or Linear Space.
     */
    set fromLinearSpace(value) {
        if (this._fromLinearSpace === value) {
            return;
        }
        this._fromLinearSpace = value;
        this._updateParameters();
    }
    /**
     * * Gets the width of the output texture used to store the result of the post process.
     */
    get outputTextureWidth() {
        return this.imageProcessingConfiguration.outputTextureWidth;
    }
    /**
     * * Sets the width of the output texture used to store the result of the post process.
     */
    set outputTextureWidth(value) {
        this.imageProcessingConfiguration.outputTextureWidth = value;
    }
    /**
     * * Gets the height of the output texture used to store the result of the post process.
     */
    get outputTextureHeight() {
        return this.imageProcessingConfiguration.outputTextureHeight;
    }
    /**
     * * Sets the height of the output texture used to store the result of the post process.
     */
    set outputTextureHeight(value) {
        this.imageProcessingConfiguration.outputTextureHeight = value;
    }
    /**
     * Constructs a new image processing post process
     * @param name Name of the effect
     * @param engine Engine to use to render the effect. If not provided, the last created engine will be used
     * @param options Options to configure the effect
     */
    constructor(name, engine = null, options) {
        super({
            ...options,
            name,
            engine: engine || Engine.LastCreatedEngine,
            useShaderStore: true,
            useAsPostProcess: true,
            fragmentShader: ThinImageProcessingPostProcess.FragmentUrl,
        });
        this._fromLinearSpace = true;
        /**
         * Defines cache preventing GC.
         */
        this._defines = {
            IMAGEPROCESSING: false,
            VIGNETTE: false,
            VIGNETTEBLENDMODEMULTIPLY: false,
            VIGNETTEBLENDMODEOPAQUE: false,
            TONEMAPPING: 0,
            CONTRAST: false,
            COLORCURVES: false,
            COLORGRADING: false,
            COLORGRADING3D: false,
            FROMLINEARSPACE: false,
            SAMPLER3DGREENDEPTH: false,
            SAMPLER3DBGRMAP: false,
            DITHER: false,
            IMAGEPROCESSINGPOSTPROCESS: false,
            EXPOSURE: false,
            SKIPFINALCOLORCLAMP: false,
        };
        const imageProcessingConfiguration = options?.imageProcessingConfiguration;
        // Setup the configuration as forced by the constructor. This would then not force the
        // scene materials output in linear space and let untouched the default forward pass.
        if (imageProcessingConfiguration) {
            imageProcessingConfiguration.applyByPostProcess = true;
            this._attachImageProcessingConfiguration(imageProcessingConfiguration, true);
            // This will cause the shader to be compiled
            this._updateParameters();
        }
        // Setup the default processing configuration to the scene.
        else {
            this._attachImageProcessingConfiguration(null, true);
            this.imageProcessingConfiguration.applyByPostProcess = true;
        }
    }
    /**
     * @internal
     */
    _updateParameters() {
        this._defines.FROMLINEARSPACE = this._fromLinearSpace;
        this.imageProcessingConfiguration.prepareDefines(this._defines, true);
        let defines = "";
        for (const prop in this._defines) {
            const value = this._defines[prop];
            const type = typeof value;
            switch (type) {
                case "number":
                case "string":
                    defines += `#define ${prop} ${value};\n`;
                    break;
                default:
                    if (value) {
                        defines += `#define ${prop};\n`;
                    }
                    break;
            }
        }
        const samplers = ["textureSampler"];
        const uniforms = ["scale"];
        if (ImageProcessingConfiguration) {
            ImageProcessingConfiguration.PrepareSamplers(samplers, this._defines);
            ImageProcessingConfiguration.PrepareUniforms(uniforms, this._defines);
        }
        this.updateEffect(defines, uniforms, samplers);
    }
    bind(noDefaultBindings = false) {
        super.bind(noDefaultBindings);
        this.imageProcessingConfiguration.bind(this.effect, this.overrideAspectRatio);
    }
    dispose() {
        super.dispose();
        if (this._imageProcessingConfiguration && this._imageProcessingObserver) {
            this._imageProcessingConfiguration.onUpdateParameters.remove(this._imageProcessingObserver);
        }
        if (this._imageProcessingConfiguration) {
            this.imageProcessingConfiguration.applyByPostProcess = false;
        }
    }
}
/**
 * The fragment shader url
 */
ThinImageProcessingPostProcess.FragmentUrl = "imageProcessing";

/**
 * ImageProcessingPostProcess
 * @see https://doc.babylonjs.com/features/featuresDeepDive/postProcesses/usePostProcesses#imageprocessing
 */
class ImageProcessingPostProcess extends PostProcess {
    get _imageProcessingConfiguration() {
        return this._effectWrapper.imageProcessingConfiguration;
    }
    /**
     * Gets the image processing configuration used either in this material.
     */
    get imageProcessingConfiguration() {
        return this._effectWrapper.imageProcessingConfiguration;
    }
    /**
     * Sets the Default image processing configuration used either in the this material.
     *
     * If sets to null, the scene one is in use.
     */
    set imageProcessingConfiguration(value) {
        this._effectWrapper.imageProcessingConfiguration = value;
    }
    /**
     * If the post process is supported.
     */
    get isSupported() {
        const effect = this.getEffect();
        return !effect || effect.isSupported;
    }
    /**
     * Gets Color curves setup used in the effect if colorCurvesEnabled is set to true .
     */
    get colorCurves() {
        return this.imageProcessingConfiguration.colorCurves;
    }
    /**
     * Sets Color curves setup used in the effect if colorCurvesEnabled is set to true .
     */
    set colorCurves(value) {
        this.imageProcessingConfiguration.colorCurves = value;
    }
    /**
     * Gets whether the color curves effect is enabled.
     */
    get colorCurvesEnabled() {
        return this.imageProcessingConfiguration.colorCurvesEnabled;
    }
    /**
     * Sets whether the color curves effect is enabled.
     */
    set colorCurvesEnabled(value) {
        this.imageProcessingConfiguration.colorCurvesEnabled = value;
    }
    /**
     * Gets Color grading LUT texture used in the effect if colorGradingEnabled is set to true.
     */
    get colorGradingTexture() {
        return this.imageProcessingConfiguration.colorGradingTexture;
    }
    /**
     * Sets Color grading LUT texture used in the effect if colorGradingEnabled is set to true.
     */
    set colorGradingTexture(value) {
        this.imageProcessingConfiguration.colorGradingTexture = value;
    }
    /**
     * Gets whether the color grading effect is enabled.
     */
    get colorGradingEnabled() {
        return this.imageProcessingConfiguration.colorGradingEnabled;
    }
    /**
     * Gets whether the color grading effect is enabled.
     */
    set colorGradingEnabled(value) {
        this.imageProcessingConfiguration.colorGradingEnabled = value;
    }
    /**
     * Gets exposure used in the effect.
     */
    get exposure() {
        return this.imageProcessingConfiguration.exposure;
    }
    /**
     * Sets exposure used in the effect.
     */
    set exposure(value) {
        this.imageProcessingConfiguration.exposure = value;
    }
    /**
     * Gets whether tonemapping is enabled or not.
     */
    get toneMappingEnabled() {
        return this._imageProcessingConfiguration.toneMappingEnabled;
    }
    /**
     * Sets whether tonemapping is enabled or not
     */
    set toneMappingEnabled(value) {
        this._imageProcessingConfiguration.toneMappingEnabled = value;
    }
    /**
     * Gets the type of tone mapping effect.
     */
    get toneMappingType() {
        return this._imageProcessingConfiguration.toneMappingType;
    }
    /**
     * Sets the type of tone mapping effect.
     */
    set toneMappingType(value) {
        this._imageProcessingConfiguration.toneMappingType = value;
    }
    /**
     * Gets contrast used in the effect.
     */
    get contrast() {
        return this.imageProcessingConfiguration.contrast;
    }
    /**
     * Sets contrast used in the effect.
     */
    set contrast(value) {
        this.imageProcessingConfiguration.contrast = value;
    }
    /**
     * Gets Vignette stretch size.
     */
    get vignetteStretch() {
        return this.imageProcessingConfiguration.vignetteStretch;
    }
    /**
     * Sets Vignette stretch size.
     */
    set vignetteStretch(value) {
        this.imageProcessingConfiguration.vignetteStretch = value;
    }
    /**
     * Gets Vignette center X Offset.
     * @deprecated use vignetteCenterX instead
     */
    get vignetteCentreX() {
        return this.imageProcessingConfiguration.vignetteCenterX;
    }
    /**
     * Sets Vignette center X Offset.
     * @deprecated use vignetteCenterX instead
     */
    set vignetteCentreX(value) {
        this.imageProcessingConfiguration.vignetteCenterX = value;
    }
    /**
     * Gets Vignette center Y Offset.
     * @deprecated use vignetteCenterY instead
     */
    get vignetteCentreY() {
        return this.imageProcessingConfiguration.vignetteCenterY;
    }
    /**
     * Sets Vignette center Y Offset.
     * @deprecated use vignetteCenterY instead
     */
    set vignetteCentreY(value) {
        this.imageProcessingConfiguration.vignetteCenterY = value;
    }
    /**
     * Vignette center Y Offset.
     */
    get vignetteCenterY() {
        return this.imageProcessingConfiguration.vignetteCenterY;
    }
    set vignetteCenterY(value) {
        this.imageProcessingConfiguration.vignetteCenterY = value;
    }
    /**
     * Vignette center X Offset.
     */
    get vignetteCenterX() {
        return this.imageProcessingConfiguration.vignetteCenterX;
    }
    set vignetteCenterX(value) {
        this.imageProcessingConfiguration.vignetteCenterX = value;
    }
    /**
     * Gets Vignette weight or intensity of the vignette effect.
     */
    get vignetteWeight() {
        return this.imageProcessingConfiguration.vignetteWeight;
    }
    /**
     * Sets Vignette weight or intensity of the vignette effect.
     */
    set vignetteWeight(value) {
        this.imageProcessingConfiguration.vignetteWeight = value;
    }
    /**
     * Gets Color of the vignette applied on the screen through the chosen blend mode (vignetteBlendMode)
     * if vignetteEnabled is set to true.
     */
    get vignetteColor() {
        return this.imageProcessingConfiguration.vignetteColor;
    }
    /**
     * Sets Color of the vignette applied on the screen through the chosen blend mode (vignetteBlendMode)
     * if vignetteEnabled is set to true.
     */
    set vignetteColor(value) {
        this.imageProcessingConfiguration.vignetteColor = value;
    }
    /**
     * Gets Camera field of view used by the Vignette effect.
     */
    get vignetteCameraFov() {
        return this.imageProcessingConfiguration.vignetteCameraFov;
    }
    /**
     * Sets Camera field of view used by the Vignette effect.
     */
    set vignetteCameraFov(value) {
        this.imageProcessingConfiguration.vignetteCameraFov = value;
    }
    /**
     * Gets the vignette blend mode allowing different kind of effect.
     */
    get vignetteBlendMode() {
        return this.imageProcessingConfiguration.vignetteBlendMode;
    }
    /**
     * Sets the vignette blend mode allowing different kind of effect.
     */
    set vignetteBlendMode(value) {
        this.imageProcessingConfiguration.vignetteBlendMode = value;
    }
    /**
     * Gets whether the vignette effect is enabled.
     */
    get vignetteEnabled() {
        return this.imageProcessingConfiguration.vignetteEnabled;
    }
    /**
     * Sets whether the vignette effect is enabled.
     */
    set vignetteEnabled(value) {
        this.imageProcessingConfiguration.vignetteEnabled = value;
    }
    /**
     * Gets intensity of the dithering effect.
     */
    get ditheringIntensity() {
        return this.imageProcessingConfiguration.ditheringIntensity;
    }
    /**
     * Sets intensity of the dithering effect.
     */
    set ditheringIntensity(value) {
        this.imageProcessingConfiguration.ditheringIntensity = value;
    }
    /**
     * Gets whether the dithering effect is enabled.
     */
    get ditheringEnabled() {
        return this.imageProcessingConfiguration.ditheringEnabled;
    }
    /**
     * Sets whether the dithering effect is enabled.
     */
    set ditheringEnabled(value) {
        this.imageProcessingConfiguration.ditheringEnabled = value;
    }
    /**
     * Gets whether the input of the processing is in Gamma or Linear Space.
     */
    get fromLinearSpace() {
        return this._effectWrapper.fromLinearSpace;
    }
    /**
     * Sets whether the input of the processing is in Gamma or Linear Space.
     */
    set fromLinearSpace(value) {
        this._effectWrapper.fromLinearSpace = value;
    }
    constructor(name, options, camera = null, samplingMode, engine, reusable, textureType = 0, imageProcessingConfiguration) {
        const localOptions = {
            size: typeof options === "number" ? options : undefined,
            camera,
            samplingMode,
            engine,
            reusable,
            textureType,
            imageProcessingConfiguration,
            scene: camera?.getScene(),
            ...options,
            blockCompilation: true,
        };
        super(name, ThinImageProcessingPostProcess.FragmentUrl, {
            effectWrapper: typeof options === "number" || !options.effectWrapper ? new ThinImageProcessingPostProcess(name, engine, localOptions) : undefined,
            ...localOptions,
        });
        this.onApply = () => {
            this._effectWrapper.overrideAspectRatio = this.aspectRatio;
        };
    }
    /**
     *  "ImageProcessingPostProcess"
     * @returns "ImageProcessingPostProcess"
     */
    getClassName() {
        return "ImageProcessingPostProcess";
    }
    /**
     * @internal
     */
    _updateParameters() {
        this._effectWrapper._updateParameters();
    }
    dispose(camera) {
        super.dispose(camera);
        if (this._imageProcessingConfiguration) {
            this.imageProcessingConfiguration.applyByPostProcess = false;
        }
    }
}
__decorate([
    serialize()
], ImageProcessingPostProcess.prototype, "fromLinearSpace", null);

/**
 * A multi render target designed to render the prepass.
 * Prepass is a scene component used to render information in multiple textures
 * alongside with the scene materials rendering.
 * Note : This is an internal class, and you should NOT need to instanciate this.
 * Only the `PrePassRenderer` should instanciate this class.
 * It is more likely that you need a regular `MultiRenderTarget`
 * @internal
 */
class PrePassRenderTarget extends MultiRenderTarget {
    constructor(name, renderTargetTexture, size, count, scene, options) {
        super(name, size, count, scene, options);
        /**
         * @internal
         */
        this._beforeCompositionPostProcesses = [];
        /**
         * @internal
         */
        this._internalTextureDirty = false;
        /**
         * Is this render target enabled for prepass rendering
         */
        this.enabled = false;
        /**
         * Render target associated with this prePassRenderTarget
         * If this is `null`, it means this prePassRenderTarget is associated with the scene
         */
        this.renderTargetTexture = null;
        this.renderTargetTexture = renderTargetTexture;
    }
    /**
     * Creates a composition effect for this RT
     * @internal
     */
    _createCompositionEffect() {
        this.imageProcessingPostProcess = new ImageProcessingPostProcess("prePassComposition", 1, null, undefined, this._engine);
        this.imageProcessingPostProcess._updateParameters();
    }
    /**
     * Checks that the size of this RT is still adapted to the desired render size.
     * @internal
     */
    _checkSize() {
        const requiredWidth = this._engine.getRenderWidth(true);
        const requiredHeight = this._engine.getRenderHeight(true);
        const width = this.getRenderWidth();
        const height = this.getRenderHeight();
        if (width !== requiredWidth || height !== requiredHeight) {
            this.resize({ width: requiredWidth, height: requiredHeight });
            this._internalTextureDirty = true;
        }
    }
    /**
     * Changes the number of render targets in this MRT
     * Be careful as it will recreate all the data in the new texture.
     * @param count new texture count
     * @param options Specifies texture types and sampling modes for new textures
     * @param textureNames Specifies the names of the textures (optional)
     */
    updateCount(count, options, textureNames) {
        super.updateCount(count, options, textureNames);
        this._internalTextureDirty = true;
    }
    /**
     * Resets the post processes chains applied to this RT.
     * @internal
     */
    _resetPostProcessChain() {
        this._beforeCompositionPostProcesses.length = 0;
    }
    /**
     * Diposes this render target
     */
    dispose() {
        const scene = this._scene;
        super.dispose();
        if (scene && scene.prePassRenderer) {
            const index = scene.prePassRenderer.renderTargets.indexOf(this);
            if (index !== -1) {
                scene.prePassRenderer.renderTargets.splice(index, 1);
            }
        }
        if (this.imageProcessingPostProcess) {
            this.imageProcessingPostProcess.dispose();
        }
        if (this.renderTargetTexture) {
            this.renderTargetTexture._prePassRenderTarget = null;
        }
        if (this._outputPostProcess) {
            this._outputPostProcess.autoClear = true;
            this._outputPostProcess.restoreDefaultInputTexture();
        }
    }
}

/**
 * Renders a pre pass of the scene
 * This means every mesh in the scene will be rendered to a render target texture
 * And then this texture will be composited to the rendering canvas with post processes
 * It is necessary for effects like subsurface scattering or deferred shading
 */
class PrePassRenderer {
    /**
     * Indicates if the prepass renderer is generating normals in world space or camera space (default: camera space)
     */
    get generateNormalsInWorldSpace() {
        return this._generateNormalsInWorldSpace;
    }
    set generateNormalsInWorldSpace(value) {
        if (this._generateNormalsInWorldSpace === value) {
            return;
        }
        this._generateNormalsInWorldSpace = value;
        this._markAllMaterialsAsPrePassDirty();
    }
    /**
     * Returns the index of a texture in the multi render target texture array.
     * @param type Texture type
     * @returns The index
     */
    getIndex(type) {
        return this._textureIndices[type];
    }
    /**
     * How many samples are used for MSAA of the scene render target
     */
    get samples() {
        return this.defaultRT.samples;
    }
    set samples(n) {
        this.defaultRT.samples = n;
    }
    /**
     * If set to true (default: false), the depth texture will be cleared with the depth value corresponding to the far plane (1 in normal mode, 0 in reverse depth buffer mode)
     * If set to false, the depth texture is always cleared with 0.
     */
    get useSpecificClearForDepthTexture() {
        return this._useSpecificClearForDepthTexture;
    }
    set useSpecificClearForDepthTexture(value) {
        if (this._useSpecificClearForDepthTexture === value) {
            return;
        }
        this._useSpecificClearForDepthTexture = value;
        this._isDirty = true;
    }
    /**
     * @returns the prepass render target for the rendering pass.
     * If we are currently rendering a render target, it returns the PrePassRenderTarget
     * associated with that render target. Otherwise, it returns the scene default PrePassRenderTarget
     */
    getRenderTarget() {
        return this._currentTarget;
    }
    /**
     * @internal
     * Managed by the scene component
     * @param prePassRenderTarget
     */
    _setRenderTarget(prePassRenderTarget) {
        if (prePassRenderTarget) {
            this._currentTarget = prePassRenderTarget;
        }
        else {
            this._currentTarget = this.defaultRT;
            this._engine.currentRenderPassId = this._scene.activeCamera?.renderPassId ?? this._currentTarget.renderPassId;
        }
    }
    /**
     * Returns true if the currently rendered prePassRenderTarget is the one
     * associated with the scene.
     */
    get currentRTisSceneRT() {
        return this._currentTarget === this.defaultRT;
    }
    _refreshGeometryBufferRendererLink() {
        if (!this.doNotUseGeometryRendererFallback) {
            this._geometryBuffer = this._scene.enableGeometryBufferRenderer();
            if (!this._geometryBuffer) {
                // Not supported
                this.doNotUseGeometryRendererFallback = true;
                return;
            }
            this._geometryBuffer._linkPrePassRenderer(this);
        }
        else {
            if (this._geometryBuffer) {
                this._geometryBuffer._unlinkPrePassRenderer();
            }
            this._geometryBuffer = null;
            this._scene.disableGeometryBufferRenderer();
        }
    }
    /**
     * Indicates if the prepass is enabled
     */
    get enabled() {
        return this._enabled;
    }
    /**
     * Instantiates a prepass renderer
     * @param scene The scene
     */
    constructor(scene) {
        /**
         * To save performance, we can excluded skinned meshes from the prepass
         */
        this.excludedSkinnedMesh = [];
        /**
         * Force material to be excluded from the prepass
         * Can be useful when `useGeometryBufferFallback` is set to `true`
         * and you don't want a material to show in the effect.
         */
        this.excludedMaterials = [];
        /**
         * Number of textures in the multi render target texture where the scene is directly rendered
         */
        this.mrtCount = 0;
        this._mrtTypes = [];
        this._mrtFormats = [];
        this._mrtLayout = [];
        this._mrtNames = [];
        this._textureIndices = [];
        this._generateNormalsInWorldSpace = false;
        this._useSpecificClearForDepthTexture = false;
        this._isDirty = true;
        /**
         * Configuration for prepass effects
         */
        this._effectConfigurations = [];
        /**
         * Prevents the PrePassRenderer from using the GeometryBufferRenderer as a fallback
         */
        this.doNotUseGeometryRendererFallback = true;
        /**
         * All the render targets generated by prepass
         */
        this.renderTargets = [];
        this._clearColor = new Color4(0, 0, 0, 0);
        this._clearDepthColor = new Color4(0, 0, 0, 1); //  // sets an invalid value by default - depth in the depth texture is view.z, so 0 is not possible because view.z can't be less than camera.minZ
        this._enabled = false;
        this._needsCompositionForThisPass = false;
        /**
         * Set to true to disable gamma transform in PrePass.
         * Can be useful in case you already proceed to gamma transform on a material level
         * and your post processes don't need to be in linear color space.
         */
        this.disableGammaTransform = false;
        this._scene = scene;
        this._engine = scene.getEngine();
        let type = 0;
        if (this._engine._caps.textureFloat && this._engine._caps.textureFloatLinearFiltering) {
            type = 1;
        }
        else if (this._engine._caps.textureHalfFloat && this._engine._caps.textureHalfFloatLinearFiltering) {
            type = 2;
        }
        for (let i = 0; i < PrePassRenderer.TextureFormats.length; ++i) {
            const format = PrePassRenderer.TextureFormats[i].format;
            if (PrePassRenderer.TextureFormats[i].type === 1) {
                PrePassRenderer.TextureFormats[i].type = type;
                if (type === 1 &&
                    (format === 6 || format === 7 || format === 5) &&
                    !this._engine._caps.supportFloatTexturesResolve) {
                    // We don't know in advance if the texture will be used as a resolve target, so we revert to half_float if the extension to resolve full float textures is not supported
                    PrePassRenderer.TextureFormats[i].type = 2;
                }
            }
        }
        PrePassRenderer._SceneComponentInitialization(this._scene);
        this.defaultRT = this._createRenderTarget("sceneprePassRT", null);
        this._currentTarget = this.defaultRT;
    }
    /**
     * Creates a new PrePassRenderTarget
     * This should be the only way to instantiate a `PrePassRenderTarget`
     * @param name Name of the `PrePassRenderTarget`
     * @param renderTargetTexture RenderTarget the `PrePassRenderTarget` will be attached to.
     * Can be `null` if the created `PrePassRenderTarget` is attached to the scene (default framebuffer).
     * @internal
     */
    _createRenderTarget(name, renderTargetTexture) {
        const rt = new PrePassRenderTarget(name, renderTargetTexture, { width: this._engine.getRenderWidth(), height: this._engine.getRenderHeight() }, 0, this._scene, {
            generateMipMaps: false,
            generateStencilBuffer: this._engine.isStencilEnable,
            defaultType: 0,
            types: [],
            drawOnlyOnFirstAttachmentByDefault: true,
        });
        this.renderTargets.push(rt);
        if (this._enabled) {
            // The pre-pass renderer is already enabled, so make sure we create the render target with the correct number of textures
            this._update();
        }
        return rt;
    }
    /**
     * Indicates if rendering a prepass is supported
     */
    get isSupported() {
        return this._scene.getEngine().getCaps().drawBuffersExtension;
    }
    /**
     * Sets the proper output textures to draw in the engine.
     * @param effect The effect that is drawn. It can be or not be compatible with drawing to several output textures.
     * @param subMesh Submesh on which the effect is applied
     */
    bindAttachmentsForEffect(effect, subMesh) {
        const material = subMesh.getMaterial();
        const isPrePassCapable = material && material.isPrePassCapable;
        const excluded = material && this.excludedMaterials.indexOf(material) !== -1;
        if (this.enabled && this._currentTarget.enabled) {
            if (effect._multiTarget && isPrePassCapable && !excluded) {
                this._engine.bindAttachments(this._multiRenderAttachments);
            }
            else {
                if (this._engine._currentRenderTarget) {
                    this._engine.bindAttachments(this._defaultAttachments);
                }
                else {
                    this._engine.restoreSingleAttachment();
                }
                if (this._geometryBuffer && this.currentRTisSceneRT && !excluded) {
                    this._geometryBuffer.renderList.push(subMesh.getRenderingMesh());
                }
            }
        }
    }
    _reinitializeAttachments() {
        const multiRenderLayout = [];
        const clearLayout = [false];
        const clearDepthLayout = [false];
        const defaultLayout = [true];
        for (let i = 0; i < this.mrtCount; i++) {
            multiRenderLayout.push(true);
            if (i > 0) {
                if (this._useSpecificClearForDepthTexture && this._mrtLayout[i] === 5) {
                    clearLayout.push(false);
                    clearDepthLayout.push(true);
                }
                else {
                    clearLayout.push(true);
                    clearDepthLayout.push(false);
                }
                defaultLayout.push(false);
            }
        }
        this._multiRenderAttachments = this._engine.buildTextureLayout(multiRenderLayout);
        this._clearAttachments = this._engine.buildTextureLayout(clearLayout);
        this._clearDepthAttachments = this._engine.buildTextureLayout(clearDepthLayout);
        this._defaultAttachments = this._engine.buildTextureLayout(defaultLayout);
    }
    _resetLayout() {
        for (let i = 0; i < PrePassRenderer.TextureFormats.length; i++) {
            this._textureIndices[PrePassRenderer.TextureFormats[i].purpose] = -1;
        }
        this._textureIndices[4] = 0;
        this._mrtLayout = [4];
        this._mrtTypes = [PrePassRenderer.TextureFormats[4].type];
        this._mrtFormats = [PrePassRenderer.TextureFormats[4].format];
        this._mrtNames = [PrePassRenderer.TextureFormats[4].name];
        this.mrtCount = 1;
    }
    _updateGeometryBufferLayout() {
        this._refreshGeometryBufferRendererLink();
        if (this._geometryBuffer) {
            this._geometryBuffer._resetLayout();
            const texturesActivated = [];
            for (let i = 0; i < this._mrtLayout.length; i++) {
                texturesActivated.push(false);
            }
            this._geometryBuffer._linkInternalTexture(this.defaultRT.getInternalTexture());
            const matches = [
                {
                    prePassConstant: 5,
                    geometryBufferConstant: GeometryBufferRenderer.DEPTH_TEXTURE_TYPE,
                },
                {
                    prePassConstant: 6,
                    geometryBufferConstant: GeometryBufferRenderer.NORMAL_TEXTURE_TYPE,
                },
                {
                    prePassConstant: 1,
                    geometryBufferConstant: GeometryBufferRenderer.POSITION_TEXTURE_TYPE,
                },
                {
                    prePassConstant: 3,
                    geometryBufferConstant: GeometryBufferRenderer.REFLECTIVITY_TEXTURE_TYPE,
                },
                {
                    prePassConstant: 2,
                    geometryBufferConstant: GeometryBufferRenderer.VELOCITY_TEXTURE_TYPE,
                },
            ];
            // replace textures in the geometryBuffer RT
            for (let i = 0; i < matches.length; i++) {
                const index = this._mrtLayout.indexOf(matches[i].prePassConstant);
                if (index !== -1) {
                    this._geometryBuffer._forceTextureType(matches[i].geometryBufferConstant, index);
                    texturesActivated[index] = true;
                }
            }
            this._geometryBuffer._setAttachments(this._engine.buildTextureLayout(texturesActivated));
        }
    }
    /**
     * Restores attachments for single texture draw.
     */
    restoreAttachments() {
        if (this.enabled && this._currentTarget.enabled && this._defaultAttachments) {
            if (this._engine._currentRenderTarget) {
                this._engine.bindAttachments(this._defaultAttachments);
            }
            else {
                this._engine.restoreSingleAttachment();
            }
        }
    }
    /**
     * @internal
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _beforeDraw(camera, faceIndex, layer) {
        // const previousEnabled = this._enabled && this._currentTarget.enabled;
        if (this._isDirty) {
            this._update();
        }
        if (!this._enabled || !this._currentTarget.enabled) {
            return;
        }
        if (this._geometryBuffer) {
            this._geometryBuffer.renderList = [];
        }
        this._setupOutputForThisPass(this._currentTarget, camera);
    }
    _prepareFrame(prePassRenderTarget, faceIndex, layer) {
        if (prePassRenderTarget.renderTargetTexture) {
            prePassRenderTarget.renderTargetTexture._prepareFrame(this._scene, faceIndex, layer, prePassRenderTarget.renderTargetTexture.useCameraPostProcesses);
        }
        else if (this._postProcessesSourceForThisPass.length) {
            this._scene.postProcessManager._prepareFrame();
        }
        else {
            this._engine.restoreDefaultFramebuffer();
        }
    }
    /**
     * Sets an intermediary texture between prepass and postprocesses. This texture
     * will be used as input for post processes
     * @param rt The render target texture to use
     * @returns true if there are postprocesses that will use this texture,
     * false if there is no postprocesses - and the function has no effect
     */
    setCustomOutput(rt) {
        const firstPP = this._postProcessesSourceForThisPass[0];
        if (!firstPP) {
            return false;
        }
        firstPP.inputTexture = rt.renderTarget;
        return true;
    }
    _renderPostProcesses(prePassRenderTarget, faceIndex) {
        const firstPP = this._postProcessesSourceForThisPass[0];
        const outputTexture = firstPP ? firstPP.inputTexture : prePassRenderTarget.renderTargetTexture ? prePassRenderTarget.renderTargetTexture.renderTarget : null;
        // Build post process chain for this prepass post draw
        let postProcessChain = this._currentTarget._beforeCompositionPostProcesses;
        if (this._needsCompositionForThisPass) {
            postProcessChain = postProcessChain.concat([this._currentTarget.imageProcessingPostProcess]);
        }
        // Activates and renders the chain
        if (postProcessChain.length) {
            this._scene.postProcessManager._prepareFrame(this._currentTarget.renderTarget?.texture, postProcessChain);
            this._scene.postProcessManager.directRender(postProcessChain, outputTexture, false, faceIndex);
        }
    }
    /**
     * @internal
     */
    _afterDraw(faceIndex, layer) {
        if (this._enabled && this._currentTarget.enabled) {
            this._prepareFrame(this._currentTarget, faceIndex, layer);
            this._renderPostProcesses(this._currentTarget, faceIndex);
        }
    }
    /**
     * Clears the current prepass render target (in the sense of settings pixels to the scene clear color value)
     * @internal
     */
    _clear() {
        if (this._isDirty) {
            this._update();
        }
        if (this._enabled && this._currentTarget.enabled) {
            this._bindFrameBuffer();
            // Clearing other attachment with 0 on all other attachments
            this._engine.bindAttachments(this._clearAttachments);
            this._engine.clear(this._clearColor, true, false, false);
            if (this._useSpecificClearForDepthTexture) {
                this._engine.bindAttachments(this._clearDepthAttachments);
                this._engine.clear(this._clearDepthColor, true, false, false);
            }
            // Regular clear color with the scene clear color of the 1st attachment
            this._engine.bindAttachments(this._defaultAttachments);
        }
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _bindFrameBuffer() {
        if (this._enabled && this._currentTarget.enabled) {
            this._currentTarget._checkSize();
            const internalTexture = this._currentTarget.renderTarget;
            if (internalTexture) {
                this._engine.bindFramebuffer(internalTexture);
            }
        }
    }
    _setEnabled(enabled) {
        this._enabled = enabled;
    }
    _setRenderTargetEnabled(prePassRenderTarget, enabled) {
        prePassRenderTarget.enabled = enabled;
        if (!enabled) {
            this._unlinkInternalTexture(prePassRenderTarget);
        }
    }
    /**
     * Adds an effect configuration to the prepass render target.
     * If an effect has already been added, it won't add it twice and will return the configuration
     * already present.
     * @param cfg the effect configuration
     * @returns the effect configuration now used by the prepass
     */
    addEffectConfiguration(cfg) {
        // Do not add twice
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            if (this._effectConfigurations[i].name === cfg.name) {
                return this._effectConfigurations[i];
            }
        }
        this._effectConfigurations.push(cfg);
        if (cfg.clearColor) {
            this._clearColor.copyFrom(cfg.clearColor);
        }
        return cfg;
    }
    /**
     * Retrieves an effect configuration by name
     * @param name the name of the effect configuration
     * @returns the effect configuration, or null if not present
     */
    getEffectConfiguration(name) {
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            if (this._effectConfigurations[i].name === name) {
                return this._effectConfigurations[i];
            }
        }
        return null;
    }
    _enable() {
        const previousMrtCount = this.mrtCount;
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            if (this._effectConfigurations[i].enabled) {
                this._enableTextures(this._effectConfigurations[i].texturesRequired);
            }
        }
        for (let i = 0; i < this.renderTargets.length; i++) {
            if (this.mrtCount !== previousMrtCount || this.renderTargets[i].count !== this.mrtCount) {
                this.renderTargets[i].updateCount(this.mrtCount, { types: this._mrtTypes, formats: this._mrtFormats }, this._mrtNames.concat("prePass_DepthBuffer"));
            }
            this.renderTargets[i]._resetPostProcessChain();
            for (let j = 0; j < this._effectConfigurations.length; j++) {
                if (this._effectConfigurations[j].enabled) {
                    // TODO : subsurface scattering has 1 scene-wide effect configuration
                    // solution : do not stock postProcess on effectConfiguration, but in the prepassRenderTarget (hashmap configuration => postProcess)
                    // And call createPostProcess whenever the post process does not exist in the RT
                    if (!this._effectConfigurations[j].postProcess && this._effectConfigurations[j].createPostProcess) {
                        this._effectConfigurations[j].createPostProcess();
                    }
                    if (this._effectConfigurations[j].postProcess) {
                        this.renderTargets[i]._beforeCompositionPostProcesses.push(this._effectConfigurations[j].postProcess);
                    }
                }
            }
        }
        this._reinitializeAttachments();
        this._setEnabled(true);
        this._updateGeometryBufferLayout();
    }
    _disable() {
        this._setEnabled(false);
        for (let i = 0; i < this.renderTargets.length; i++) {
            this._setRenderTargetEnabled(this.renderTargets[i], false);
        }
        this._resetLayout();
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            this._effectConfigurations[i].enabled = false;
        }
    }
    _getPostProcessesSource(prePassRenderTarget, camera) {
        if (camera) {
            return camera._postProcesses;
        }
        else if (prePassRenderTarget.renderTargetTexture) {
            if (prePassRenderTarget.renderTargetTexture.useCameraPostProcesses) {
                const camera = prePassRenderTarget.renderTargetTexture.activeCamera ? prePassRenderTarget.renderTargetTexture.activeCamera : this._scene.activeCamera;
                return camera ? camera._postProcesses : [];
            }
            else if (prePassRenderTarget.renderTargetTexture.postProcesses) {
                return prePassRenderTarget.renderTargetTexture.postProcesses;
            }
            else {
                return [];
            }
        }
        else {
            return this._scene.activeCamera ? this._scene.activeCamera._postProcesses : [];
        }
    }
    _setupOutputForThisPass(prePassRenderTarget, camera) {
        // Order is : draw ===> prePassRenderTarget._postProcesses ==> ipp ==> camera._postProcesses
        const secondaryCamera = camera && this._scene.activeCameras && !!this._scene.activeCameras.length && this._scene.activeCameras.indexOf(camera) !== 0;
        this._postProcessesSourceForThisPass = this._getPostProcessesSource(prePassRenderTarget, camera);
        this._postProcessesSourceForThisPass = this._postProcessesSourceForThisPass.filter((pp) => {
            return pp != null;
        });
        this._scene.autoClear = true;
        const cameraHasImageProcessing = this._hasImageProcessing(this._postProcessesSourceForThisPass);
        this._needsCompositionForThisPass = !cameraHasImageProcessing && !this.disableGammaTransform && this._needsImageProcessing() && !secondaryCamera;
        const firstCameraPP = this._getFirstPostProcess(this._postProcessesSourceForThisPass);
        const firstPrePassPP = prePassRenderTarget._beforeCompositionPostProcesses && prePassRenderTarget._beforeCompositionPostProcesses[0];
        let firstPP = null;
        // Setting the scene-wide post process configuration
        this._scene.imageProcessingConfiguration.applyByPostProcess = this._needsCompositionForThisPass || cameraHasImageProcessing;
        // Create composition effect if needed
        if (this._needsCompositionForThisPass && !prePassRenderTarget.imageProcessingPostProcess) {
            prePassRenderTarget._createCompositionEffect();
        }
        // Setting the prePassRenderTarget as input texture of the first PP
        if (firstPrePassPP) {
            firstPP = firstPrePassPP;
        }
        else if (this._needsCompositionForThisPass) {
            firstPP = prePassRenderTarget.imageProcessingPostProcess;
        }
        else if (firstCameraPP) {
            firstPP = firstCameraPP;
        }
        this._bindFrameBuffer();
        this._linkInternalTexture(prePassRenderTarget, firstPP);
    }
    _linkInternalTexture(prePassRenderTarget, postProcess) {
        if (postProcess) {
            postProcess.autoClear = false;
            postProcess.inputTexture = prePassRenderTarget.renderTarget;
        }
        if (prePassRenderTarget._outputPostProcess !== postProcess) {
            if (prePassRenderTarget._outputPostProcess) {
                this._unlinkInternalTexture(prePassRenderTarget);
            }
            prePassRenderTarget._outputPostProcess = postProcess;
        }
        if (prePassRenderTarget._internalTextureDirty) {
            this._updateGeometryBufferLayout();
            prePassRenderTarget._internalTextureDirty = false;
        }
    }
    /**
     * @internal
     */
    _unlinkInternalTexture(prePassRenderTarget) {
        if (prePassRenderTarget._outputPostProcess) {
            prePassRenderTarget._outputPostProcess.autoClear = true;
            prePassRenderTarget._outputPostProcess.restoreDefaultInputTexture();
            prePassRenderTarget._outputPostProcess = null;
        }
    }
    _needsImageProcessing() {
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            if (this._effectConfigurations[i].enabled && this._effectConfigurations[i].needsImageProcessing) {
                return true;
            }
        }
        return false;
    }
    _hasImageProcessing(postProcesses) {
        let isIPPAlreadyPresent = false;
        if (postProcesses) {
            for (let i = 0; i < postProcesses.length; i++) {
                if (postProcesses[i]?.getClassName() === "ImageProcessingPostProcess") {
                    isIPPAlreadyPresent = true;
                    break;
                }
            }
        }
        return isIPPAlreadyPresent;
    }
    /**
     * Internal, gets the first post proces.
     * @param postProcesses
     * @returns the first post process to be run on this camera.
     */
    _getFirstPostProcess(postProcesses) {
        for (let ppIndex = 0; ppIndex < postProcesses.length; ppIndex++) {
            if (postProcesses[ppIndex] !== null) {
                return postProcesses[ppIndex];
            }
        }
        return null;
    }
    /**
     * Marks the prepass renderer as dirty, triggering a check if the prepass is necessary for the next rendering.
     */
    markAsDirty() {
        this._isDirty = true;
    }
    /**
     * Enables a texture on the MultiRenderTarget for prepass
     * @param types
     */
    _enableTextures(types) {
        // For velocity : enable storage of previous matrices for instances
        this._scene.needsPreviousWorldMatrices = false;
        for (let i = 0; i < types.length; i++) {
            const type = types[i];
            if (this._textureIndices[type] === -1) {
                this._textureIndices[type] = this._mrtLayout.length;
                this._mrtLayout.push(type);
                this._mrtTypes.push(PrePassRenderer.TextureFormats[type].type);
                this._mrtFormats.push(PrePassRenderer.TextureFormats[type].format);
                this._mrtNames.push(PrePassRenderer.TextureFormats[type].name);
                this.mrtCount++;
            }
            if (type === 2 || type === 11) {
                this._scene.needsPreviousWorldMatrices = true;
            }
        }
    }
    /**
     * Makes sure that the prepass renderer is up to date if it has been dirtified.
     */
    update() {
        if (this._isDirty) {
            this._update();
        }
    }
    _update() {
        this._disable();
        let enablePrePass = false;
        this._scene.imageProcessingConfiguration.applyByPostProcess = false;
        if (this._scene._depthPeelingRenderer && this._scene.useOrderIndependentTransparency) {
            this._scene._depthPeelingRenderer.setPrePassRenderer(this);
            enablePrePass = true;
        }
        for (let i = 0; i < this._scene.materials.length; i++) {
            if (this._scene.materials[i].setPrePassRenderer(this)) {
                enablePrePass = true;
            }
        }
        if (enablePrePass) {
            this._setRenderTargetEnabled(this.defaultRT, true);
        }
        let postProcesses;
        for (let i = 0; i < this.renderTargets.length; i++) {
            if (this.renderTargets[i].renderTargetTexture) {
                postProcesses = this._getPostProcessesSource(this.renderTargets[i]);
            }
            else {
                // When there are multiple active cameras, we have to choose one. We assume it's the first one and not scene.activeCamera, because in a number of cases,
                // _update() will be called from an async method, meaning the active camera will be the last one in the list of active cameras,
                // which is generally not the right camera to use for the prepass setup.
                const camera = this._scene.activeCameras && this._scene.activeCameras.length > 0 ? this._scene.activeCameras[0] : this._scene.activeCamera;
                if (!camera) {
                    continue;
                }
                postProcesses = camera._postProcesses;
            }
            if (!postProcesses) {
                continue;
            }
            postProcesses = postProcesses.filter((pp) => {
                return pp != null;
            });
            if (postProcesses) {
                for (let j = 0; j < postProcesses.length; j++) {
                    if (postProcesses[j].setPrePassRenderer(this)) {
                        this._setRenderTargetEnabled(this.renderTargets[i], true);
                        enablePrePass = true;
                    }
                }
                if (this._hasImageProcessing(postProcesses)) {
                    this._scene.imageProcessingConfiguration.applyByPostProcess = true;
                }
            }
        }
        this._markAllMaterialsAsPrePassDirty();
        this._isDirty = false;
        if (enablePrePass) {
            this._enable();
        }
    }
    _markAllMaterialsAsPrePassDirty() {
        const materials = this._scene.materials;
        for (let i = 0; i < materials.length; i++) {
            materials[i].markAsDirty(Material.PrePassDirtyFlag);
        }
    }
    /**
     * Disposes the prepass renderer.
     */
    dispose() {
        for (let i = this.renderTargets.length - 1; i >= 0; i--) {
            this.renderTargets[i].dispose();
        }
        for (let i = 0; i < this._effectConfigurations.length; i++) {
            if (this._effectConfigurations[i].dispose) {
                this._effectConfigurations[i].dispose();
            }
        }
    }
}
/**
 * @internal
 */
PrePassRenderer._SceneComponentInitialization = (_) => {
    throw _WarnImport("PrePassRendererSceneComponent");
};
/**
 * Describes the types and formats of the textures used by the pre-pass renderer
 */
PrePassRenderer.TextureFormats = [
    {
        purpose: 0,
        type: 2,
        format: 5,
        name: "prePass_Irradiance",
    },
    {
        purpose: 1,
        type: 2,
        format: 5,
        name: "prePass_Position",
    },
    {
        purpose: 2,
        type: 0,
        format: 5,
        name: "prePass_Velocity",
    },
    {
        purpose: 3,
        type: 0,
        format: 5,
        name: "prePass_Reflectivity",
    },
    {
        purpose: 4,
        type: 2,
        format: 5,
        name: "prePass_Color",
    },
    {
        purpose: 5,
        type: 1,
        format: 6,
        name: "prePass_Depth",
    },
    {
        purpose: 6,
        type: 2,
        format: 5,
        name: "prePass_Normal",
    },
    {
        purpose: 7,
        type: 0,
        format: 5,
        name: "prePass_Albedo",
    },
    {
        purpose: 8,
        type: 0,
        format: 5,
        name: "prePass_WorldNormal",
    },
    {
        purpose: 9,
        type: 2,
        format: 5,
        name: "prePass_LocalPosition",
    },
    {
        purpose: 10,
        type: 1,
        format: 6,
        name: "prePass_ScreenDepth",
    },
    {
        purpose: 11,
        type: 2,
        format: 5,
        name: "prePass_VelocityLinear",
    },
];

Object.defineProperty(Scene.prototype, "prePassRenderer", {
    get: function () {
        return this._prePassRenderer;
    },
    set: function (value) {
        if (value && value.isSupported) {
            this._prePassRenderer = value;
        }
    },
    enumerable: true,
    configurable: true,
});
Scene.prototype.enablePrePassRenderer = function () {
    if (this._prePassRenderer) {
        return this._prePassRenderer;
    }
    this._prePassRenderer = new PrePassRenderer(this);
    if (!this._prePassRenderer.isSupported) {
        this._prePassRenderer = null;
        Logger.Error("PrePassRenderer needs WebGL 2 support.\n" + "Maybe you tried to use the following features that need the PrePassRenderer :\n" + " + Subsurface Scattering");
    }
    return this._prePassRenderer;
};
Scene.prototype.disablePrePassRenderer = function () {
    if (!this._prePassRenderer) {
        return;
    }
    this._prePassRenderer.dispose();
    this._prePassRenderer = null;
};
/**
 * Defines the Geometry Buffer scene component responsible to manage a G-Buffer useful
 * in several rendering techniques.
 */
class PrePassRendererSceneComponent {
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_PREPASSRENDERER;
        this.scene = scene;
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._beforeCameraDrawStage.registerStep(SceneComponentConstants.STEP_BEFORECAMERADRAW_PREPASS, this, this._beforeCameraDraw);
        this.scene._afterCameraDrawStage.registerStep(SceneComponentConstants.STEP_AFTERCAMERADRAW_PREPASS, this, this._afterCameraDraw);
        this.scene._beforeRenderTargetDrawStage.registerStep(SceneComponentConstants.STEP_BEFORERENDERTARGETDRAW_PREPASS, this, this._beforeRenderTargetDraw);
        this.scene._afterRenderTargetDrawStage.registerStep(SceneComponentConstants.STEP_AFTERCAMERADRAW_PREPASS, this, this._afterRenderTargetDraw);
        this.scene._beforeClearStage.registerStep(SceneComponentConstants.STEP_BEFORECLEAR_PREPASS, this, this._beforeClearStage);
        this.scene._beforeRenderTargetClearStage.registerStep(SceneComponentConstants.STEP_BEFORERENDERTARGETCLEAR_PREPASS, this, this._beforeRenderTargetClearStage);
        this.scene._beforeRenderingMeshStage.registerStep(SceneComponentConstants.STEP_BEFORERENDERINGMESH_PREPASS, this, this._beforeRenderingMeshStage);
        this.scene._afterRenderingMeshStage.registerStep(SceneComponentConstants.STEP_AFTERRENDERINGMESH_PREPASS, this, this._afterRenderingMeshStage);
    }
    _beforeRenderTargetDraw(renderTarget, faceIndex, layer) {
        if (this.scene.prePassRenderer && !renderTarget.noPrePassRenderer) {
            this.scene.prePassRenderer._setRenderTarget(renderTarget._prePassRenderTarget);
            this.scene.prePassRenderer._beforeDraw(undefined, faceIndex, layer);
        }
    }
    _afterRenderTargetDraw(renderTarget, faceIndex, layer) {
        if (this.scene.prePassRenderer && !renderTarget.noPrePassRenderer) {
            this.scene.prePassRenderer._afterDraw(faceIndex, layer);
        }
    }
    _beforeRenderTargetClearStage(renderTarget) {
        if (this.scene.prePassRenderer && !renderTarget.noPrePassRenderer) {
            if (!renderTarget._prePassRenderTarget) {
                renderTarget._prePassRenderTarget = this.scene.prePassRenderer._createRenderTarget(renderTarget.name + "_prePassRTT", renderTarget);
            }
            this.scene.prePassRenderer._setRenderTarget(renderTarget._prePassRenderTarget);
            this.scene.prePassRenderer._clear();
        }
    }
    _beforeCameraDraw(camera) {
        if (this.scene.prePassRenderer) {
            this.scene.prePassRenderer._setRenderTarget(null);
            this.scene.prePassRenderer._beforeDraw(camera);
        }
    }
    _afterCameraDraw() {
        if (this.scene.prePassRenderer) {
            this.scene.prePassRenderer._afterDraw();
        }
    }
    _beforeClearStage() {
        if (this.scene.prePassRenderer) {
            this.scene.prePassRenderer._setRenderTarget(null);
            this.scene.prePassRenderer._clear();
        }
    }
    _beforeRenderingMeshStage(mesh, subMesh, batch, effect) {
        if (!effect) {
            return;
        }
        // Render to MRT
        const scene = mesh.getScene();
        if (scene.prePassRenderer) {
            scene.prePassRenderer.bindAttachmentsForEffect(effect, subMesh);
        }
    }
    _afterRenderingMeshStage(mesh) {
        const scene = mesh.getScene();
        if (scene.prePassRenderer) {
            scene.prePassRenderer.restoreAttachments();
        }
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        // Nothing to do for this component
    }
    /**
     * Disposes the component and the associated resources
     */
    dispose() {
        this.scene.disablePrePassRenderer();
    }
}
PrePassRenderer._SceneComponentInitialization = (scene) => {
    // Register the G Buffer component to the scene.
    let component = scene._getComponent(SceneComponentConstants.NAME_PREPASSRENDERER);
    if (!component) {
        component = new PrePassRendererSceneComponent(scene);
        scene._addComponent(component);
    }
};

export { PrePassRendererSceneComponent };
//# sourceMappingURL=prePassRendererSceneComponent-D4VlxF_m.js.map
