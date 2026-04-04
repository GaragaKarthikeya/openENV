const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./hdrFiltering.vertex-BLHsh6Hu.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./hdrFiltering.fragment-C8V8uM7Y.js","./helperFunctions-qLR1vvLh.js","./hdrFilteringFunctions-DQUZ_3Oj.js","./pbrBRDFFunctions-Ezjx-oio.js","./hdrFiltering.vertex-Bo3CfxnN.js","./hdrFiltering.fragment-fkYZcY6K.js","./helperFunctions-CiFVmLcr.js","./hdrFilteringFunctions-Cf_DM9A0.js","./pbrBRDFFunctions-Cr0QY02Q.js","./hdrIrradianceFiltering.vertex-LT-yFOb5.js","./hdrIrradianceFiltering.fragment-CC25jGOU.js","./hdrIrradianceFiltering.vertex-DzIrR9Vi.js","./hdrIrradianceFiltering.fragment-C25Ho0q0.js"])))=>i.map(i=>d[i]);
import { az as ILog2, V as Vector3, aE as EffectWrapper, aD as EffectRenderer, B as BaseTexture, M as Matrix, O as Observable, T as Texture, d as Tools, ay as SphericalPolynomial, aP as CubeMapToSphericalPolynomialTools, br as ToGammaSpace, aO as ToHalfFloat, R as RegisterClass } from './index-CxR0Yf_7.js';
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { I as IblCdfGenerator } from './iblCdfGenerator-CtGL_NwH.js';
import { G as GetCubeMapTextureData } from './hdr-BQ3k4n6y.js';
import './rawTexture-Bto4WZO2.js';

/**
 * Filters HDR maps to get correct renderings of PBR reflections
 */
class HDRFiltering {
    /**
     * Instantiates HDR filter for reflection maps
     *
     * @param engine Thin engine
     * @param options Options
     */
    constructor(engine, options = {}) {
        this._lodGenerationOffset = 0;
        this._lodGenerationScale = 0.8;
        /**
         * Quality switch for prefiltering. Should be set to `4096` unless
         * you care about baking speed.
         */
        this.quality = 4096;
        /**
         * Scales pixel intensity for the input HDR map.
         */
        this.hdrScale = 1;
        // pass
        this._engine = engine;
        this.hdrScale = options.hdrScale || this.hdrScale;
        this.quality = options.quality || this.quality;
    }
    _createRenderTarget(size) {
        let textureType = 0;
        if (this._engine.getCaps().textureHalfFloatRender) {
            textureType = 2;
        }
        else if (this._engine.getCaps().textureFloatRender) {
            textureType = 1;
        }
        const rtWrapper = this._engine.createRenderTargetCubeTexture(size, {
            format: 5,
            type: textureType,
            createMipMaps: true,
            generateMipMaps: false,
            generateDepthBuffer: false,
            generateStencilBuffer: false,
            samplingMode: 1,
            label: "HDR_Radiance_Filtering_Target",
        });
        this._engine.updateTextureWrappingMode(rtWrapper.texture, 0, 0, 0);
        this._engine.updateTextureSamplingMode(3, rtWrapper.texture, true);
        return rtWrapper;
    }
    _prefilterInternal(texture) {
        const width = texture.getSize().width;
        const mipmapsCount = ILog2(width) + 1;
        const effect = this._effectWrapper.effect;
        const outputTexture = this._createRenderTarget(width);
        this._effectRenderer.saveStates();
        this._effectRenderer.setViewport();
        const intTexture = texture.getInternalTexture();
        if (intTexture) {
            // Just in case generate fresh clean mips.
            this._engine.updateTextureSamplingMode(3, intTexture, true);
        }
        this._effectRenderer.applyEffectWrapper(this._effectWrapper);
        const directions = [
            [new Vector3(0, 0, -1), new Vector3(0, -1, 0), new Vector3(1, 0, 0)], // PositiveX
            [new Vector3(0, 0, 1), new Vector3(0, -1, 0), new Vector3(-1, 0, 0)], // NegativeX
            [new Vector3(1, 0, 0), new Vector3(0, 0, 1), new Vector3(0, 1, 0)], // PositiveY
            [new Vector3(1, 0, 0), new Vector3(0, 0, -1), new Vector3(0, -1, 0)], // NegativeY
            [new Vector3(1, 0, 0), new Vector3(0, -1, 0), new Vector3(0, 0, 1)], // PositiveZ
            [new Vector3(-1, 0, 0), new Vector3(0, -1, 0), new Vector3(0, 0, -1)], // NegativeZ
        ];
        effect.setFloat("hdrScale", this.hdrScale);
        effect.setFloat2("vFilteringInfo", texture.getSize().width, mipmapsCount);
        effect.setTexture("inputTexture", texture);
        for (let face = 0; face < 6; face++) {
            effect.setVector3("up", directions[face][0]);
            effect.setVector3("right", directions[face][1]);
            effect.setVector3("front", directions[face][2]);
            for (let lod = 0; lod < mipmapsCount; lod++) {
                this._engine.bindFramebuffer(outputTexture, face, undefined, undefined, true, lod);
                this._effectRenderer.applyEffectWrapper(this._effectWrapper);
                let alpha = Math.pow(2, (lod - this._lodGenerationOffset) / this._lodGenerationScale) / width;
                if (lod === 0) {
                    alpha = 0;
                }
                effect.setFloat("alphaG", alpha);
                this._effectRenderer.draw();
            }
        }
        // Cleanup
        this._effectRenderer.restoreStates();
        this._engine.restoreDefaultFramebuffer();
        this._engine._releaseTexture(texture._texture);
        // Internal Swap
        const type = outputTexture.texture.type;
        const format = outputTexture.texture.format;
        outputTexture._swapAndDie(texture._texture);
        texture._texture.type = type;
        texture._texture.format = format;
        // New settings
        texture.gammaSpace = false;
        texture.lodGenerationOffset = this._lodGenerationOffset;
        texture.lodGenerationScale = this._lodGenerationScale;
        texture._prefiltered = true;
        return texture;
    }
    _createEffect(texture, onCompiled) {
        const defines = [];
        if (texture.gammaSpace) {
            defines.push("#define GAMMA_INPUT");
        }
        defines.push("#define NUM_SAMPLES " + this.quality + "u"); // unsigned int
        const isWebGPU = this._engine.isWebGPU;
        const effectWrapper = new EffectWrapper({
            engine: this._engine,
            name: "hdrFiltering",
            vertexShader: "hdrFiltering",
            fragmentShader: "hdrFiltering",
            samplerNames: ["inputTexture"],
            uniformNames: ["vSampleDirections", "vWeights", "up", "right", "front", "vFilteringInfo", "hdrScale", "alphaG"],
            useShaderStore: true,
            defines,
            onCompiled: onCompiled,
            shaderLanguage: isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
            extraInitializationsAsync: async () => {
                if (isWebGPU) {
                    await Promise.all([__vitePreload(() => import('./hdrFiltering.vertex-BLHsh6Hu.js'),true              ?__vite__mapDeps([0,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./hdrFiltering.fragment-C8V8uM7Y.js'),true              ?__vite__mapDeps([4,1,2,3,5,6,7]):void 0,import.meta.url)]);
                }
                else {
                    await Promise.all([__vitePreload(() => import('./hdrFiltering.vertex-Bo3CfxnN.js'),true              ?__vite__mapDeps([8,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./hdrFiltering.fragment-fkYZcY6K.js'),true              ?__vite__mapDeps([9,1,2,3,10,11,12]):void 0,import.meta.url)]);
                }
            },
        });
        return effectWrapper;
    }
    /**
     * Get a value indicating if the filter is ready to be used
     * @param texture Texture to filter
     * @returns true if the filter is ready
     */
    isReady(texture) {
        return texture.isReady() && this._effectWrapper.effect.isReady();
    }
    /**
     * Prefilters a cube texture to have mipmap levels representing roughness values.
     * Prefiltering will be invoked at the end of next rendering pass.
     * This has to be done once the map is loaded, and has not been prefiltered by a third party software.
     * See http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf for more information
     * @param texture Texture to filter
     * @returns Promise called when prefiltering is done
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention
    async prefilter(texture) {
        if (!this._engine._features.allowTexturePrefiltering) {
            throw new Error("HDR prefiltering is not available in WebGL 1., you can use real time filtering instead.");
        }
        this._effectRenderer = new EffectRenderer(this._engine);
        this._effectWrapper = this._createEffect(texture);
        await this._effectWrapper.effect.whenCompiledAsync();
        this._prefilterInternal(texture);
        this._effectRenderer.dispose();
        this._effectWrapper.dispose();
    }
}

/**
 * Filters HDR maps to get correct renderings of PBR reflections
 */
class HDRIrradianceFiltering {
    /**
     * Instantiates HDR filter for irradiance map
     *
     * @param engine Thin engine
     * @param options Options
     */
    constructor(engine, options = {}) {
        /**
         * Quality switch for prefiltering. Should be set to `4096` unless
         * you care about baking speed.
         */
        this.quality = 4096;
        /**
         * Scales pixel intensity for the input HDR map.
         */
        this.hdrScale = 1;
        /**
         * Use the Cumulative Distribution Function (CDF) for filtering
         */
        this.useCdf = false;
        // pass
        this._engine = engine;
        this.hdrScale = options.hdrScale || this.hdrScale;
        this.quality = options.quality || this.quality;
        this.useCdf = options.useCdf || this.useCdf;
    }
    _createRenderTarget(size) {
        let textureType = 0;
        if (this._engine.getCaps().textureHalfFloatRender) {
            textureType = 2;
        }
        else if (this._engine.getCaps().textureFloatRender) {
            textureType = 1;
        }
        const rtWrapper = this._engine.createRenderTargetCubeTexture(size, {
            format: 5,
            type: textureType,
            createMipMaps: false,
            generateMipMaps: false,
            generateDepthBuffer: false,
            generateStencilBuffer: false,
            samplingMode: 2,
            label: "HDR_Irradiance_Filtering_Target",
        });
        this._engine.updateTextureWrappingMode(rtWrapper.texture, 0, 0, 0);
        return rtWrapper;
    }
    _prefilterInternal(texture) {
        const width = texture.getSize().width;
        const mipmapsCount = ILog2(width);
        const effect = this._effectWrapper.effect;
        // Choose a power of 2 size for the irradiance map.
        // It can be much smaller than the original texture.
        const irradianceSize = Math.max(32, 1 << ILog2(width >> 3));
        const outputTexture = this._createRenderTarget(irradianceSize);
        this._effectRenderer.saveStates();
        this._effectRenderer.setViewport();
        this._effectRenderer.applyEffectWrapper(this._effectWrapper);
        const directions = [
            [new Vector3(0, 0, -1), new Vector3(0, -1, 0), new Vector3(1, 0, 0)], // PositiveX
            [new Vector3(0, 0, 1), new Vector3(0, -1, 0), new Vector3(-1, 0, 0)], // NegativeX
            [new Vector3(1, 0, 0), new Vector3(0, 0, 1), new Vector3(0, 1, 0)], // PositiveY
            [new Vector3(1, 0, 0), new Vector3(0, 0, -1), new Vector3(0, -1, 0)], // NegativeY
            [new Vector3(1, 0, 0), new Vector3(0, -1, 0), new Vector3(0, 0, 1)], // PositiveZ
            [new Vector3(-1, 0, 0), new Vector3(0, -1, 0), new Vector3(0, 0, -1)], // NegativeZ
        ];
        effect.setFloat("hdrScale", this.hdrScale);
        effect.setFloat2("vFilteringInfo", texture.getSize().width, mipmapsCount);
        effect.setTexture("inputTexture", texture);
        if (this._cdfGenerator) {
            effect.setTexture("icdfTexture", this._cdfGenerator.getIcdfTexture());
        }
        for (let face = 0; face < 6; face++) {
            effect.setVector3("up", directions[face][0]);
            effect.setVector3("right", directions[face][1]);
            effect.setVector3("front", directions[face][2]);
            this._engine.bindFramebuffer(outputTexture, face, undefined, undefined, true);
            this._effectRenderer.applyEffectWrapper(this._effectWrapper);
            this._effectRenderer.draw();
        }
        // Cleanup
        this._effectRenderer.restoreStates();
        this._engine.restoreDefaultFramebuffer();
        effect.setTexture("inputTexture", null);
        effect.setTexture("icdfTexture", null);
        const irradianceTexture = new BaseTexture(texture.getScene(), outputTexture.texture);
        irradianceTexture.name = texture.name + "_irradiance";
        irradianceTexture.displayName = texture.name + "_irradiance";
        irradianceTexture.gammaSpace = false;
        return irradianceTexture;
    }
    _createEffect(texture, onCompiled) {
        const defines = [];
        if (texture.gammaSpace) {
            defines.push("#define GAMMA_INPUT");
        }
        defines.push("#define NUM_SAMPLES " + this.quality + "u"); // unsigned int
        const isWebGPU = this._engine.isWebGPU;
        const samplers = ["inputTexture"];
        if (this._cdfGenerator) {
            samplers.push("icdfTexture");
            defines.push("#define IBL_CDF_FILTERING");
        }
        const effectWrapper = new EffectWrapper({
            engine: this._engine,
            name: "HDRIrradianceFiltering",
            vertexShader: "hdrIrradianceFiltering",
            fragmentShader: "hdrIrradianceFiltering",
            samplerNames: samplers,
            uniformNames: ["vSampleDirections", "vWeights", "up", "right", "front", "vFilteringInfo", "hdrScale"],
            useShaderStore: true,
            defines,
            onCompiled: onCompiled,
            shaderLanguage: isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
            extraInitializationsAsync: async () => {
                if (isWebGPU) {
                    await Promise.all([__vitePreload(() => import('./hdrIrradianceFiltering.vertex-LT-yFOb5.js'),true              ?__vite__mapDeps([13,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./hdrIrradianceFiltering.fragment-CC25jGOU.js'),true              ?__vite__mapDeps([14,1,2,3,5,6,7]):void 0,import.meta.url)]);
                }
                else {
                    await Promise.all([__vitePreload(() => import('./hdrIrradianceFiltering.vertex-DzIrR9Vi.js'),true              ?__vite__mapDeps([15,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./hdrIrradianceFiltering.fragment-C25Ho0q0.js'),true              ?__vite__mapDeps([16,1,2,3,10,11,12]):void 0,import.meta.url)]);
                }
            },
        });
        return effectWrapper;
    }
    /**
     * Get a value indicating if the filter is ready to be used
     * @param texture Texture to filter
     * @returns true if the filter is ready
     */
    isReady(texture) {
        return texture.isReady() && this._effectWrapper.effect.isReady();
    }
    /**
     * Prefilters a cube texture to contain IBL irradiance.
     * Prefiltering will be invoked at the end of next rendering pass.
     * This has to be done once the map is loaded, and has not been prefiltered by a third party software.
     * See http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf for more information
     * @param texture Texture to filter
     * @returns Promise called when prefiltering is done
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention
    async prefilter(texture) {
        if (!this._engine._features.allowTexturePrefiltering) {
            throw new Error("HDR prefiltering is not available in WebGL 1., you can use real time filtering instead.");
        }
        if (this.useCdf) {
            this._cdfGenerator = new IblCdfGenerator(this._engine);
            this._cdfGenerator.iblSource = texture;
            await this._cdfGenerator.renderWhenReady();
        }
        this._effectRenderer = new EffectRenderer(this._engine);
        this._effectWrapper = this._createEffect(texture);
        await this._effectWrapper.effect.whenCompiledAsync();
        const irradianceTexture = this._prefilterInternal(texture);
        if (this.useCdf) {
            // eslint-disable-next-line github/no-then
            await this._cdfGenerator.findDominantDirection().then((dir) => {
                irradianceTexture._dominantDirection = dir;
            });
        }
        this._effectRenderer.dispose();
        this._effectWrapper.dispose();
        this._cdfGenerator?.dispose();
        return irradianceTexture;
    }
}

/**
 * This represents an environment base texture which could for instance be from HDR or EXR files.
 */
class EnvCubeTexture extends BaseTexture {
    /**
     * Sets whether or not the texture is blocking during loading.
     */
    set isBlocking(value) {
        this._isBlocking = value;
    }
    /**
     * Gets whether or not the texture is blocking during loading.
     */
    get isBlocking() {
        return this._isBlocking;
    }
    /**
     * Sets texture matrix rotation angle around Y axis in radians.
     */
    set rotationY(value) {
        this._rotationY = value;
        this.setReflectionTextureMatrix(Matrix.RotationY(this._rotationY));
    }
    /**
     * Gets texture matrix rotation angle around Y axis radians.
     */
    get rotationY() {
        return this._rotationY;
    }
    /**
     * Gets or sets the size of the bounding box associated with the cube texture
     * When defined, the cubemap will switch to local mode
     * @see https://community.arm.com/graphics/b/blog/posts/reflections-based-on-local-cubemaps-in-unity
     * @example https://www.babylonjs-playground.com/#RNASML
     */
    set boundingBoxSize(value) {
        if (this._boundingBoxSize && this._boundingBoxSize.equals(value)) {
            return;
        }
        this._boundingBoxSize = value;
        const scene = this.getScene();
        if (scene) {
            scene.markAllMaterialsAsDirty(1);
        }
    }
    get boundingBoxSize() {
        return this._boundingBoxSize;
    }
    /**
     * Instantiates an EnvCubeTexture from the following parameters.
     *
     * @param url The location of the raw data (Panorama stored in RGBE format)
     * @param sceneOrEngine The scene or engine the texture will be used in
     * @param size The cubemap desired size (the more it increases the longer the generation will be)
     * @param noMipmap Forces to not generate the mipmap if true
     * @param generateHarmonics Specifies whether you want to extract the polynomial harmonics during the generation process
     * @param gammaSpace Specifies if the texture will be use in gamma or linear space (the PBR material requires those texture in linear space, but the standard material would require them in Gamma space)
     * @param prefilterOnLoad Prefilters texture to allow use of this texture as a PBR reflection texture.
     * @param onLoad on success callback function
     * @param onError on error callback function
     * @param supersample Defines if texture must be supersampled (default: false)
     * @param prefilterIrradianceOnLoad Prefilters texture to allow use of this texture for irradiance lighting.
     * @param prefilterUsingCdf Defines if the prefiltering should be done using a CDF instead of the default approach.
     */
    constructor(url, sceneOrEngine, size, noMipmap = false, generateHarmonics = true, gammaSpace = false, prefilterOnLoad = false, onLoad = null, onError = null, supersample = false, prefilterIrradianceOnLoad = false, prefilterUsingCdf = false) {
        super(sceneOrEngine);
        this._generateHarmonics = true;
        this._onError = null;
        this._isBlocking = true;
        this._rotationY = 0;
        /**
         * Gets or sets the center of the bounding box associated with the cube texture
         * It must define where the camera used to render the texture was set
         */
        this.boundingBoxPosition = Vector3.Zero();
        /**
         * Observable triggered once the texture has been loaded.
         */
        this.onLoadObservable = new Observable();
        if (!url) {
            return;
        }
        this._coordinatesMode = Texture.CUBIC_MODE;
        this.name = url;
        this.url = url;
        this.hasAlpha = false;
        this.isCube = true;
        this._textureMatrix = Matrix.Identity();
        this._prefilterOnLoad = prefilterOnLoad;
        this._prefilterIrradianceOnLoad = prefilterIrradianceOnLoad;
        this._prefilterUsingCdf = prefilterUsingCdf;
        this._onLoad = () => {
            this.onLoadObservable.notifyObservers(this);
            if (onLoad) {
                onLoad();
            }
        };
        this._onError = onError;
        this.gammaSpace = gammaSpace;
        this._noMipmap = noMipmap;
        this._size = size;
        // CDF is very sensitive to lost precision due to downsampling. This can result in
        // noticeable brightness differences with different resolutions. Enabling supersampling
        // mitigates this.
        this._supersample = supersample || prefilterUsingCdf;
        this._generateHarmonics = generateHarmonics;
        this._texture = this._getFromCache(url, this._noMipmap, undefined, undefined, undefined, this.isCube);
        if (!this._texture) {
            if (!this.getScene()?.useDelayedTextureLoading) {
                this._loadTexture();
            }
            else {
                this.delayLoadState = 4;
            }
        }
        else {
            if (this._texture.isReady) {
                Tools.SetImmediate(() => this._onLoad());
            }
            else {
                this._texture.onLoadedObservable.add(this._onLoad);
            }
        }
    }
    /**
     * Get the current class name of the texture useful for serialization or dynamic coding.
     * @returns "EnvCubeTexture"
     */
    getClassName() {
        return "EnvCubeTexture";
    }
    /**
     * Occurs when the file has been loaded.
     */
    _loadTexture() {
        const engine = this._getEngine();
        const caps = engine.getCaps();
        let textureType = 0;
        if (caps.textureFloat && caps.textureFloatLinearFiltering) {
            textureType = 1;
        }
        else if (caps.textureHalfFloat && caps.textureHalfFloatLinearFiltering) {
            textureType = 2;
        }
        // eslint-disable-next-line no-restricted-syntax
        const callback = async (buffer) => {
            this.lodGenerationOffset = 0.0;
            this.lodGenerationScale = 0.8;
            // Extract the raw linear data.
            const data = await this._getCubeMapTextureDataAsync(buffer, this._size, this._supersample);
            // Generate harmonics if needed.
            if (this._generateHarmonics) {
                const sphericalPolynomial = CubeMapToSphericalPolynomialTools.ConvertCubeMapToSphericalPolynomial(data);
                this.sphericalPolynomial = sphericalPolynomial;
            }
            const results = [];
            let byteArray = null;
            let shortArray = null;
            // Push each faces.
            for (let j = 0; j < 6; j++) {
                // Create fallback array
                if (textureType === 2) {
                    shortArray = new Uint16Array(this._size * this._size * 3);
                }
                else if (textureType === 0) {
                    // 3 channels of 1 bytes per pixel in bytes.
                    byteArray = new Uint8Array(this._size * this._size * 3);
                }
                const dataFace = data[EnvCubeTexture._FacesMapping[j]];
                // If special cases.
                if (this.gammaSpace || shortArray || byteArray) {
                    for (let i = 0; i < this._size * this._size; i++) {
                        // Put in gamma space if requested.
                        if (this.gammaSpace) {
                            dataFace[i * 3 + 0] = Math.pow(dataFace[i * 3 + 0], ToGammaSpace);
                            dataFace[i * 3 + 1] = Math.pow(dataFace[i * 3 + 1], ToGammaSpace);
                            dataFace[i * 3 + 2] = Math.pow(dataFace[i * 3 + 2], ToGammaSpace);
                        }
                        // Convert to half float texture for fallback.
                        if (shortArray) {
                            shortArray[i * 3 + 0] = ToHalfFloat(dataFace[i * 3 + 0]);
                            shortArray[i * 3 + 1] = ToHalfFloat(dataFace[i * 3 + 1]);
                            shortArray[i * 3 + 2] = ToHalfFloat(dataFace[i * 3 + 2]);
                        }
                        // Convert to int texture for fallback.
                        if (byteArray) {
                            let r = Math.max(dataFace[i * 3 + 0] * 255, 0);
                            let g = Math.max(dataFace[i * 3 + 1] * 255, 0);
                            let b = Math.max(dataFace[i * 3 + 2] * 255, 0);
                            // May use luminance instead if the result is not accurate.
                            const max = Math.max(Math.max(r, g), b);
                            if (max > 255) {
                                const scale = 255 / max;
                                r *= scale;
                                g *= scale;
                                b *= scale;
                            }
                            byteArray[i * 3 + 0] = r;
                            byteArray[i * 3 + 1] = g;
                            byteArray[i * 3 + 2] = b;
                        }
                    }
                }
                if (shortArray) {
                    results.push(shortArray);
                }
                else if (byteArray) {
                    results.push(byteArray);
                }
                else {
                    results.push(dataFace);
                }
            }
            return results;
        };
        if (engine._features.allowTexturePrefiltering && (this._prefilterOnLoad || this._prefilterIrradianceOnLoad)) {
            const previousOnLoad = this._onLoad;
            const hdrFiltering = new HDRFiltering(engine);
            this._onLoad = () => {
                let irradiancePromise = Promise.resolve(null);
                let radiancePromise = Promise.resolve();
                if (this._prefilterIrradianceOnLoad) {
                    const hdrIrradianceFiltering = new HDRIrradianceFiltering(engine, { useCdf: this._prefilterUsingCdf });
                    irradiancePromise = hdrIrradianceFiltering.prefilter(this);
                }
                if (this._prefilterOnLoad) {
                    radiancePromise = hdrFiltering.prefilter(this);
                }
                // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                Promise.all([irradiancePromise, radiancePromise]).then((results) => {
                    const irradianceTexture = results[0];
                    if (this._prefilterIrradianceOnLoad && irradianceTexture) {
                        this.irradianceTexture = irradianceTexture;
                        const scene = this.getScene();
                        if (scene) {
                            scene.markAllMaterialsAsDirty(1);
                        }
                    }
                    if (previousOnLoad) {
                        previousOnLoad();
                    }
                });
            };
        }
        this._texture = engine.createRawCubeTextureFromUrl(this.url, this.getScene(), this._size, 4, textureType, this._noMipmap, callback, null, this._onLoad, this._onError);
        if (!this._generateHarmonics && !this._texture._sphericalPolynomial) {
            this._texture._sphericalPolynomial = new SphericalPolynomial();
        }
    }
    // Methods
    delayLoad() {
        if (this.delayLoadState !== 4) {
            return;
        }
        this.delayLoadState = 1;
        this._texture = this._getFromCache(this.url, this._noMipmap);
        if (!this._texture) {
            this._loadTexture();
        }
    }
    /**
     * Get the texture reflection matrix used to rotate/transform the reflection.
     * @returns the reflection matrix
     */
    getReflectionTextureMatrix() {
        return this._textureMatrix;
    }
    /**
     * Set the texture reflection matrix used to rotate/transform the reflection.
     * @param value Define the reflection matrix to set
     */
    setReflectionTextureMatrix(value) {
        this._textureMatrix = value;
        if (value.updateFlag === this._textureMatrix.updateFlag) {
            return;
        }
        if (value.isIdentity() !== this._textureMatrix.isIdentity()) {
            this.getScene()?.markAllMaterialsAsDirty(1, (mat) => mat.getActiveTextures().indexOf(this) !== -1);
        }
    }
    /**
     * Dispose the texture and release its associated resources.
     */
    dispose() {
        this.onLoadObservable.clear();
        super.dispose();
    }
    /**
     * Serializes the texture to a JSON representation.
     * @returns the JSON representation
     */
    serialize() {
        if (!this.name) {
            return null;
        }
        const serializationObject = {};
        serializationObject.name = this.name;
        serializationObject.hasAlpha = this.hasAlpha;
        serializationObject.isCube = true;
        serializationObject.level = this.level;
        serializationObject.size = this._size;
        serializationObject.coordinatesMode = this.coordinatesMode;
        serializationObject.useInGammaSpace = this.gammaSpace;
        serializationObject.generateHarmonics = this._generateHarmonics;
        serializationObject.noMipmap = this._noMipmap;
        serializationObject.isBlocking = this._isBlocking;
        serializationObject.rotationY = this._rotationY;
        return serializationObject;
    }
    /**
     * Clones the current texture.
     * @returns the cloned texture
     */
    clone() {
        const newTexture = this._instantiateClone();
        // Base Texture
        newTexture.level = this.level;
        newTexture.wrapU = this.wrapU;
        newTexture.wrapV = this.wrapV;
        newTexture.coordinatesIndex = this.coordinatesIndex;
        newTexture.coordinatesMode = this.coordinatesMode;
        return newTexture;
    }
    static _Parse(parsedTexture, texture) {
        texture.name = parsedTexture.name;
        texture.hasAlpha = parsedTexture.hasAlpha;
        texture.level = parsedTexture.level;
        texture.coordinatesMode = parsedTexture.coordinatesMode;
        texture.isBlocking = parsedTexture.isBlocking;
        if (parsedTexture.boundingBoxPosition) {
            texture.boundingBoxPosition = Vector3.FromArray(parsedTexture.boundingBoxPosition);
        }
        if (parsedTexture.boundingBoxSize) {
            texture.boundingBoxSize = Vector3.FromArray(parsedTexture.boundingBoxSize);
        }
        if (parsedTexture.rotationY) {
            texture.rotationY = parsedTexture.rotationY;
        }
    }
}
EnvCubeTexture._FacesMapping = ["right", "left", "up", "down", "front", "back"];

/**
 * This represents a texture coming from an HDR input.
 *
 * The supported format is currently panorama picture stored in RGBE format.
 * Example of such files can be found on Poly Haven: https://polyhaven.com/hdris
 */
class HDRCubeTexture extends EnvCubeTexture {
    /**
     * Instantiates an HDRTexture from the following parameters.
     *
     * @param url The location of the HDR raw data (Panorama stored in RGBE format)
     * @param sceneOrEngine The scene or engine the texture will be used in
     * @param size The cubemap desired size (the more it increases the longer the generation will be)
     * @param noMipmap Forces to not generate the mipmap if true
     * @param generateHarmonics Specifies whether you want to extract the polynomial harmonics during the generation process
     * @param gammaSpace Specifies if the texture will be use in gamma or linear space (the PBR material requires those texture in linear space, but the standard material would require them in Gamma space)
     * @param prefilterOnLoad Prefilters HDR texture to allow use of this texture as a PBR reflection texture.
     * @param onLoad on success callback function
     * @param onError on error callback function
     * @param supersample Defines if texture must be supersampled (default: false)
     * @param prefilterIrradianceOnLoad Prefilters HDR texture to allow use of this texture for irradiance lighting.
     * @param prefilterUsingCdf Defines if the prefiltering should be done using a CDF instead of the default approach.
     */
    constructor(url, sceneOrEngine, size, noMipmap = false, generateHarmonics = true, gammaSpace = false, prefilterOnLoad = false, onLoad = null, onError = null, supersample = false, prefilterIrradianceOnLoad = false, prefilterUsingCdf = false) {
        super(url, sceneOrEngine, size, noMipmap, generateHarmonics, gammaSpace, prefilterOnLoad, onLoad, onError, supersample, prefilterIrradianceOnLoad, prefilterUsingCdf);
    }
    /**
     * Get the current class name of the texture useful for serialization or dynamic coding.
     * @returns "HDRCubeTexture"
     */
    getClassName() {
        return "HDRCubeTexture";
    }
    /**
     * Convert the raw data from the server into cubemap faces
     * @param buffer The buffer containing the texture data
     * @param size The cubemap face size
     * @param supersample Defines if texture must be supersampled
     * @returns The cube map data
     */
    async _getCubeMapTextureDataAsync(buffer, size, supersample) {
        return GetCubeMapTextureData(buffer, size, supersample);
    }
    _instantiateClone() {
        return new HDRCubeTexture(this.url, this.getScene() || this._getEngine(), this._size, this._noMipmap, this._generateHarmonics, this.gammaSpace);
    }
    /**
     * Serialize the texture to a JSON representation.
     * @returns The JSON representation of the texture
     */
    serialize() {
        const serializationObject = super.serialize();
        if (!serializationObject) {
            return null;
        }
        serializationObject.customType = "BABYLON.HDRCubeTexture";
        return serializationObject;
    }
    /**
     * Parses a JSON representation of an HDR Texture in order to create the texture
     * @param parsedTexture Define the JSON representation
     * @param scene Define the scene the texture should be created in
     * @param rootUrl Define the root url in case we need to load relative dependencies
     * @returns the newly created texture after parsing
     */
    static Parse(parsedTexture, scene, rootUrl) {
        if (!parsedTexture.name || parsedTexture.isRenderTarget) {
            return null;
        }
        const texture = new HDRCubeTexture(rootUrl + parsedTexture.name, scene, parsedTexture.size, parsedTexture.noMipmap, parsedTexture.generateHarmonics, parsedTexture.useInGammaSpace);
        this._Parse(parsedTexture, texture);
        return texture;
    }
}
RegisterClass("BABYLON.HDRCubeTexture", HDRCubeTexture);

export { HDRCubeTexture };
//# sourceMappingURL=hdrCubeTexture-yvXHZWkb.js.map
