const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./procedural.vertex-BiHXqw2j.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./procedural.vertex-BoDp_XJl.js","./iblCdfx.fragment-CMtfE_Ro.js","./iblCdfy.fragment-BpWpCgpC.js","./iblScaledLuminance.fragment-CI0s2UUc.js","./helperFunctions-qLR1vvLh.js","./iblCdfx.fragment-eSoc3zz0.js","./iblCdfy.fragment-D6HMvppy.js","./helperFunctions-CiFVmLcr.js","./iblScaledLuminance.fragment-Y8LEQjH0.js","./iblIcdf.fragment-GfR4QU_r.js","./iblDominantDirection.fragment-BAqNHMjM.js","./hdrFilteringFunctions-DQUZ_3Oj.js","./pbrBRDFFunctions-Ezjx-oio.js","./iblIcdf.fragment-DkDbLB7r.js","./iblDominantDirection.fragment-cgOKTZl4.js","./hdrFilteringFunctions-Cf_DM9A0.js","./pbrBRDFFunctions-Cr0QY02Q.js","./iblCdfDebug.fragment-D8xPZtFT.js","./iblCdfDebug.fragment-BCZOycba.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { a as SceneComponentConstants, d as Tools, T as Texture, O as Observable, aq as EngineStore, bb as DrawWrapper, J as VertexBuffer, aL as RenderTargetTexture, o as Material, _ as __decorate, s as serialize, R as RegisterClass, ai as Vector4, L as Logger, bn as Engine, aA as PostProcess, bo as _RetryWithInterval, V as Vector3, be as _WarnImport } from './index-CxR0Yf_7.js';
import { R as RawTexture } from './rawTexture-Bto4WZO2.js';

/**
 * Defines the Procedural Texture scene component responsible to manage any Procedural Texture
 * in a given scene.
 */
class ProceduralTextureSceneComponent {
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_PROCEDURALTEXTURE;
        this.scene = scene;
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._beforeClearStage.registerStep(SceneComponentConstants.STEP_BEFORECLEAR_PROCEDURALTEXTURE, this, this._beforeClear);
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        // Nothing to do here.
    }
    /**
     * Disposes the component and the associated resources.
     */
    dispose() {
        // Nothing to do here.
    }
    _beforeClear() {
        if (this.scene.proceduralTexturesEnabled) {
            Tools.StartPerformanceCounter("Procedural textures", this.scene.proceduralTextures.length > 0);
            for (let proceduralIndex = 0; proceduralIndex < this.scene.proceduralTextures.length; proceduralIndex++) {
                const proceduralTexture = this.scene.proceduralTextures[proceduralIndex];
                if (proceduralTexture._shouldRender()) {
                    proceduralTexture.render();
                }
            }
            Tools.EndPerformanceCounter("Procedural textures", this.scene.proceduralTextures.length > 0);
        }
    }
}

/**
 * Procedural texturing is a way to programmatically create a texture. There are 2 types of procedural textures: code-only, and code that references some classic 2D images, sometimes calmpler' images.
 * This is the base class of any Procedural texture and contains most of the shareable code.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/materials/using/proceduralTextures
 */
class ProceduralTexture extends Texture {
    /**
     * Gets the shader language type used to generate vertex and fragment source code.
     */
    get shaderLanguage() {
        return this._shaderLanguage;
    }
    /**
     * Instantiates a new procedural texture.
     * Procedural texturing is a way to programmatically create a texture. There are 2 types of procedural textures: code-only, and code that references some classic 2D images, sometimes called 'refMaps' or 'sampler' images.
     * This is the base class of any Procedural texture and contains most of the shareable code.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/materials/using/proceduralTextures
     * @param name  Define the name of the texture
     * @param size Define the size of the texture to create
     * @param fragment Define the fragment shader to use to generate the texture or null if it is defined later:
     *  * object: \{ fragmentElement: "fragmentShaderCode" \}, used with shader code in script tags
     *  * object: \{ fragmentSource: "fragment shader code string" \}, the string contains the shader code
     *  * string: the string contains a name "XXX" to lookup in Effect.ShadersStore["XXXFragmentShader"]
     * @param scene Define the scene the texture belongs to
     * @param fallbackTexture Define a fallback texture in case there were issues to create the custom texture
     * @param generateMipMaps Define if the texture should creates mip maps or not
     * @param isCube Define if the texture is a cube texture or not (this will render each faces of the cube)
     * @param textureType The FBO internal texture type
     */
    constructor(name, size, fragment, scene, fallbackTexture = null, generateMipMaps = true, isCube = false, textureType = 0) {
        super(null, scene, !generateMipMaps);
        /**
         * Define if the texture is enabled or not (disabled texture will not render)
         */
        this.isEnabled = true;
        /**
         * Define if the texture must be cleared before rendering (default is true)
         */
        this.autoClear = true;
        /**
         * Event raised when the texture is generated
         */
        this.onGeneratedObservable = new Observable();
        /**
         * Event raised before the texture is generated
         */
        this.onBeforeGenerationObservable = new Observable();
        /**
         * Gets or sets the node material used to create this texture (null if the texture was manually created)
         */
        this.nodeMaterialSource = null;
        /**
         * Define the list of custom preprocessor defines used in the shader
         */
        this.defines = "";
        /** @internal */
        this._textures = {};
        this._currentRefreshId = -1;
        this._frameId = -1;
        this._refreshRate = 1;
        this._vertexBuffers = {};
        this._uniforms = new Array();
        this._samplers = new Array();
        this._floats = {};
        this._ints = {};
        this._floatsArrays = {};
        this._colors3 = {};
        this._colors4 = {};
        this._vectors2 = {};
        this._vectors3 = {};
        this._vectors4 = {};
        this._matrices = {};
        this._fallbackTextureUsed = false;
        this._cachedDefines = null;
        this._contentUpdateId = -1;
        this._rtWrapper = null;
        if (fallbackTexture !== null && !(fallbackTexture instanceof Texture)) {
            this._options = fallbackTexture;
            this._fallbackTexture = fallbackTexture.fallbackTexture ?? null;
        }
        else {
            this._options = {};
            this._fallbackTexture = fallbackTexture;
        }
        this._shaderLanguage = this._options.shaderLanguage ?? 0 /* ShaderLanguage.GLSL */;
        scene = this.getScene() || EngineStore.LastCreatedScene;
        let component = scene._getComponent(SceneComponentConstants.NAME_PROCEDURALTEXTURE);
        if (!component) {
            component = new ProceduralTextureSceneComponent(scene);
            scene._addComponent(component);
        }
        scene.proceduralTextures.push(this);
        this._fullEngine = scene.getEngine();
        this.name = name;
        this.isRenderTarget = true;
        this._size = size;
        this._textureType = textureType;
        this._generateMipMaps = generateMipMaps;
        this._drawWrapper = new DrawWrapper(this._fullEngine);
        this.setFragment(fragment);
        const rtWrapper = this._createRtWrapper(isCube, size, generateMipMaps, textureType);
        this._texture = rtWrapper.texture;
        // VBO
        const vertices = [];
        vertices.push(1, 1);
        vertices.push(-1, 1);
        vertices.push(-1, -1);
        vertices.push(1, -1);
        this._vertexBuffers[VertexBuffer.PositionKind] = new VertexBuffer(this._fullEngine, vertices, VertexBuffer.PositionKind, false, false, 2);
        this._createIndexBuffer();
    }
    _createRtWrapper(isCube, size, generateMipMaps, textureType) {
        if (isCube) {
            this._rtWrapper = this._fullEngine.createRenderTargetCubeTexture(size, {
                generateMipMaps: generateMipMaps,
                generateDepthBuffer: false,
                generateStencilBuffer: false,
                type: textureType,
                ...this._options,
            });
            this.setFloat("face", 0);
        }
        else {
            this._rtWrapper = this._fullEngine.createRenderTargetTexture(size, {
                generateMipMaps: generateMipMaps,
                generateDepthBuffer: false,
                generateStencilBuffer: false,
                type: textureType,
                ...this._options,
            });
            if (this._rtWrapper.is3D) {
                this.setFloat("layer", 0);
                this.setInt("layerNum", 0);
            }
        }
        return this._rtWrapper;
    }
    /**
     * The effect that is created when initializing the post process.
     * @returns The created effect corresponding the postprocess.
     */
    getEffect() {
        return this._drawWrapper.effect;
    }
    /**
     * @internal
     */
    _setEffect(effect) {
        this._drawWrapper.effect = effect;
    }
    /**
     * Gets texture content (Use this function wisely as reading from a texture can be slow)
     * @returns an ArrayBufferView promise (Uint8Array or Float32Array)
     */
    getContent() {
        if (this._contentData && this._frameId === this._contentUpdateId) {
            return this._contentData;
        }
        if (this._contentData) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
            this._contentData.then((buffer) => {
                this._contentData = this.readPixels(0, 0, buffer);
                this._contentUpdateId = this._frameId;
            });
        }
        else {
            this._contentData = this.readPixels(0, 0);
            this._contentUpdateId = this._frameId;
        }
        return this._contentData;
    }
    _createIndexBuffer() {
        const engine = this._fullEngine;
        // Indices
        const indices = [];
        indices.push(0);
        indices.push(1);
        indices.push(2);
        indices.push(0);
        indices.push(2);
        indices.push(3);
        this._indexBuffer = engine.createIndexBuffer(indices);
    }
    /** @internal */
    _rebuild() {
        const vb = this._vertexBuffers[VertexBuffer.PositionKind];
        if (vb) {
            vb._rebuild();
        }
        this._createIndexBuffer();
        if (this.refreshRate === RenderTargetTexture.REFRESHRATE_RENDER_ONCE) {
            this.refreshRate = RenderTargetTexture.REFRESHRATE_RENDER_ONCE;
        }
    }
    /**
     * Resets the texture in order to recreate its associated resources.
     * This can be called in case of context loss or if you change the shader code and need to regenerate the texture with the new code
     */
    reset() {
        this._drawWrapper.effect?.dispose();
        this._drawWrapper.effect = null;
        this._cachedDefines = null;
    }
    _getDefines() {
        return this.defines;
    }
    /**
     * Executes a function when the texture will be ready to be drawn.
     * @param func The callback to be used.
     */
    executeWhenReady(func) {
        if (this.isReady()) {
            func(this);
            return;
        }
        const effect = this.getEffect();
        if (effect) {
            effect.executeWhenCompiled(() => {
                func(this);
            });
        }
    }
    /**
     * Is the texture ready to be used ? (rendered at least once)
     * @returns true if ready, otherwise, false.
     */
    isReady() {
        const engine = this._fullEngine;
        if (this.nodeMaterialSource) {
            return this._drawWrapper.effect.isReady();
        }
        if (!this._fragment) {
            return false;
        }
        if (this._fallbackTextureUsed) {
            return true;
        }
        if (!this._texture) {
            return false;
        }
        const defines = this._getDefines();
        if (this._drawWrapper.effect && defines === this._cachedDefines && this._drawWrapper.effect.isReady()) {
            return true;
        }
        const shaders = {
            vertex: "procedural",
            fragmentElement: this._fragment.fragmentElement,
            fragmentSource: this._fragment.fragmentSource,
            fragment: typeof this._fragment === "string" ? this._fragment : undefined,
        };
        if (this._cachedDefines !== defines) {
            this._cachedDefines = defines;
            this._drawWrapper.effect = engine.createEffect(shaders, [VertexBuffer.PositionKind], this._uniforms, this._samplers, defines, undefined, undefined, () => {
                this._rtWrapper?.dispose();
                this._rtWrapper = this._texture = null;
                if (this._fallbackTexture) {
                    this._texture = this._fallbackTexture._texture;
                    if (this._texture) {
                        this._texture.incrementReferences();
                    }
                }
                this._fallbackTextureUsed = true;
            }, undefined, this._shaderLanguage, async () => {
                if (this._options.extraInitializationsAsync) {
                    if (this.shaderLanguage === 1 /* ShaderLanguage.WGSL */) {
                        await Promise.all([__vitePreload(() => import('./procedural.vertex-BiHXqw2j.js'),true              ?__vite__mapDeps([0,1,2,3]):void 0,import.meta.url), this._options.extraInitializationsAsync()]);
                    }
                    else {
                        await Promise.all([__vitePreload(() => import('./procedural.vertex-BoDp_XJl.js'),true              ?__vite__mapDeps([4,1,2,3]):void 0,import.meta.url), this._options.extraInitializationsAsync()]);
                    }
                }
                else {
                    if (this.shaderLanguage === 1 /* ShaderLanguage.WGSL */) {
                        await __vitePreload(() => import('./procedural.vertex-BiHXqw2j.js'),true              ?__vite__mapDeps([0,1,2,3]):void 0,import.meta.url);
                    }
                    else {
                        await __vitePreload(() => import('./procedural.vertex-BoDp_XJl.js'),true              ?__vite__mapDeps([4,1,2,3]):void 0,import.meta.url);
                    }
                }
            });
        }
        return this._drawWrapper.effect.isReady();
    }
    /**
     * Resets the refresh counter of the texture and start bak from scratch.
     * Could be useful to regenerate the texture if it is setup to render only once.
     */
    resetRefreshCounter() {
        this._currentRefreshId = -1;
    }
    /**
     * Set the fragment shader to use in order to render the texture.
     * @param fragment This can be set to a path (into the shader store) or to a json object containing a fragmentElement property.
     */
    setFragment(fragment) {
        this._fragment = fragment;
    }
    /**
     * Define the refresh rate of the texture or the rendering frequency.
     * Use 0 to render just once, 1 to render on every frame, 2 to render every two frames and so on...
     */
    get refreshRate() {
        return this._refreshRate;
    }
    set refreshRate(value) {
        this._refreshRate = value;
        this.resetRefreshCounter();
    }
    /** @internal */
    _shouldRender() {
        if (!this.isEnabled || !this.isReady() || !this._texture) {
            if (this._texture) {
                this._texture.isReady = false;
            }
            return false;
        }
        if (this._fallbackTextureUsed) {
            return false;
        }
        if (this._currentRefreshId === -1) {
            // At least render once
            this._currentRefreshId = 1;
            this._frameId++;
            return true;
        }
        if (this.refreshRate === this._currentRefreshId) {
            this._currentRefreshId = 1;
            this._frameId++;
            return true;
        }
        this._currentRefreshId++;
        return false;
    }
    /**
     * Get the size the texture is rendering at.
     * @returns the size (on cube texture it is always squared)
     */
    getRenderSize() {
        return this._size;
    }
    /**
     * Resize the texture to new value.
     * @param size Define the new size the texture should have
     * @param generateMipMaps Define whether the new texture should create mip maps
     */
    resize(size, generateMipMaps) {
        if (this._fallbackTextureUsed || !this._rtWrapper || !this._texture) {
            return;
        }
        const isCube = this._texture.isCube;
        this._rtWrapper.dispose();
        const rtWrapper = this._createRtWrapper(isCube, size, generateMipMaps, this._textureType);
        this._texture = rtWrapper.texture;
        // Update properties
        this._size = size;
        this._generateMipMaps = generateMipMaps;
    }
    _checkUniform(uniformName) {
        if (this._uniforms.indexOf(uniformName) === -1) {
            this._uniforms.push(uniformName);
        }
    }
    /**
     * Set a texture in the shader program used to render.
     * @param name Define the name of the uniform samplers as defined in the shader
     * @param texture Define the texture to bind to this sampler
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setTexture(name, texture) {
        if (this._samplers.indexOf(name) === -1) {
            this._samplers.push(name);
        }
        this._textures[name] = texture;
        return this;
    }
    /**
     * Set a float in the shader.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setFloat(name, value) {
        this._checkUniform(name);
        this._floats[name] = value;
        return this;
    }
    /**
     * Set a int in the shader.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setInt(name, value) {
        this._checkUniform(name);
        this._ints[name] = value;
        return this;
    }
    /**
     * Set an array of floats in the shader.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setFloats(name, value) {
        this._checkUniform(name);
        this._floatsArrays[name] = value;
        return this;
    }
    /**
     * Set a vec3 in the shader from a Color3.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setColor3(name, value) {
        this._checkUniform(name);
        this._colors3[name] = value;
        return this;
    }
    /**
     * Set a vec4 in the shader from a Color4.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setColor4(name, value) {
        this._checkUniform(name);
        this._colors4[name] = value;
        return this;
    }
    /**
     * Set a vec2 in the shader from a Vector2.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setVector2(name, value) {
        this._checkUniform(name);
        this._vectors2[name] = value;
        return this;
    }
    /**
     * Set a vec3 in the shader from a Vector3.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setVector3(name, value) {
        this._checkUniform(name);
        this._vectors3[name] = value;
        return this;
    }
    /**
     * Set a vec4 in the shader from a Vector4.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setVector4(name, value) {
        this._checkUniform(name);
        this._vectors4[name] = value;
        return this;
    }
    /**
     * Set a mat4 in the shader from a MAtrix.
     * @param name Define the name of the uniform as defined in the shader
     * @param value Define the value to give to the uniform
     * @returns the texture itself allowing "fluent" like uniform updates
     */
    setMatrix(name, value) {
        this._checkUniform(name);
        this._matrices[name] = value;
        return this;
    }
    /**
     * Render the texture to its associated render target.
     * @param useCameraPostProcess Define if camera post process should be applied to the texture
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    render(useCameraPostProcess) {
        const scene = this.getScene();
        if (!scene) {
            return;
        }
        const engine = this._fullEngine;
        // Render
        engine.enableEffect(this._drawWrapper);
        this.onBeforeGenerationObservable.notifyObservers(this);
        engine.setState(false);
        if (!this.nodeMaterialSource) {
            // Texture
            for (const name in this._textures) {
                this._drawWrapper.effect.setTexture(name, this._textures[name]);
            }
            // Float
            for (const name in this._ints) {
                this._drawWrapper.effect.setInt(name, this._ints[name]);
            }
            // Float
            for (const name in this._floats) {
                this._drawWrapper.effect.setFloat(name, this._floats[name]);
            }
            // Floats
            for (const name in this._floatsArrays) {
                this._drawWrapper.effect.setArray(name, this._floatsArrays[name]);
            }
            // Color3
            for (const name in this._colors3) {
                this._drawWrapper.effect.setColor3(name, this._colors3[name]);
            }
            // Color4
            for (const name in this._colors4) {
                const color = this._colors4[name];
                this._drawWrapper.effect.setFloat4(name, color.r, color.g, color.b, color.a);
            }
            // Vector2
            for (const name in this._vectors2) {
                this._drawWrapper.effect.setVector2(name, this._vectors2[name]);
            }
            // Vector3
            for (const name in this._vectors3) {
                this._drawWrapper.effect.setVector3(name, this._vectors3[name]);
            }
            // Vector4
            for (const name in this._vectors4) {
                this._drawWrapper.effect.setVector4(name, this._vectors4[name]);
            }
            // Matrix
            for (const name in this._matrices) {
                this._drawWrapper.effect.setMatrix(name, this._matrices[name]);
            }
        }
        if (!this._texture || !this._rtWrapper) {
            return;
        }
        if (engine._enableGPUDebugMarkers) {
            engine.restoreDefaultFramebuffer();
            engine._debugPushGroup?.(`procedural texture generation for ${this.name}`);
        }
        const viewPort = engine.currentViewport;
        if (this.isCube) {
            for (let face = 0; face < 6; face++) {
                engine.bindFramebuffer(this._rtWrapper, face, undefined, undefined, true);
                // VBOs
                engine.bindBuffers(this._vertexBuffers, this._indexBuffer, this._drawWrapper.effect);
                this._drawWrapper.effect.setFloat("face", face);
                // Clear
                if (this.autoClear) {
                    engine.clear(scene.clearColor, true, false, false);
                }
                // Draw order
                engine.drawElementsType(Material.TriangleFillMode, 0, 6);
                // Unbind and restore viewport
                engine.unBindFramebuffer(this._rtWrapper, true);
            }
        }
        else {
            let numLayers = 1;
            if (this._rtWrapper.is3D) {
                numLayers = this._rtWrapper.depth;
            }
            else if (this._rtWrapper.is2DArray) {
                numLayers = this._rtWrapper.layers;
            }
            for (let layer = 0; layer < numLayers; layer++) {
                engine.bindFramebuffer(this._rtWrapper, 0, undefined, undefined, true, 0, layer);
                // VBOs
                engine.bindBuffers(this._vertexBuffers, this._indexBuffer, this._drawWrapper.effect);
                if (this._rtWrapper.is3D || this._rtWrapper.is2DArray) {
                    this._drawWrapper.effect?.setFloat("layer", numLayers !== 1 ? layer / (numLayers - 1) : 0);
                    this._drawWrapper.effect?.setInt("layerNum", layer);
                    for (const name in this._textures) {
                        this._drawWrapper.effect.setTexture(name, this._textures[name]);
                    }
                }
                // Clear
                if (this.autoClear) {
                    engine.clear(scene.clearColor, true, false, false);
                }
                // Draw order
                engine.drawElementsType(Material.TriangleFillMode, 0, 6);
                // Unbind and restore viewport
                engine.unBindFramebuffer(this._rtWrapper, !this._generateMipMaps);
            }
        }
        if (viewPort) {
            engine.setViewport(viewPort);
        }
        // Mipmaps
        if (this.isCube) {
            engine.generateMipMapsForCubemap(this._texture, true);
        }
        if (engine._enableGPUDebugMarkers) {
            engine._debugPopGroup?.();
        }
        if (this.onGenerated) {
            this.onGenerated();
        }
        this.onGeneratedObservable.notifyObservers(this);
    }
    /**
     * Clone the texture.
     * @returns the cloned texture
     */
    clone() {
        const textureSize = this.getSize();
        const newTexture = new ProceduralTexture(this.name, textureSize.width, this._fragment, this.getScene(), this._fallbackTexture, this._generateMipMaps);
        // Base texture
        newTexture.hasAlpha = this.hasAlpha;
        newTexture.level = this.level;
        // RenderTarget Texture
        newTexture.coordinatesMode = this.coordinatesMode;
        return newTexture;
    }
    /**
     * Dispose the texture and release its associated resources.
     */
    dispose() {
        const scene = this.getScene();
        if (!scene) {
            return;
        }
        const index = scene.proceduralTextures.indexOf(this);
        if (index >= 0) {
            scene.proceduralTextures.splice(index, 1);
        }
        const vertexBuffer = this._vertexBuffers[VertexBuffer.PositionKind];
        if (vertexBuffer) {
            vertexBuffer.dispose();
            this._vertexBuffers[VertexBuffer.PositionKind] = null;
        }
        if (this._indexBuffer && this._fullEngine._releaseBuffer(this._indexBuffer)) {
            this._indexBuffer = null;
        }
        this.onGeneratedObservable.clear();
        this.onBeforeGenerationObservable.clear();
        super.dispose();
    }
}
__decorate([
    serialize()
], ProceduralTexture.prototype, "isEnabled", void 0);
__decorate([
    serialize()
], ProceduralTexture.prototype, "autoClear", void 0);
__decorate([
    serialize()
], ProceduralTexture.prototype, "_generateMipMaps", void 0);
__decorate([
    serialize()
], ProceduralTexture.prototype, "_size", void 0);
__decorate([
    serialize()
], ProceduralTexture.prototype, "refreshRate", null);
RegisterClass("BABYLON.ProceduralTexture", ProceduralTexture);

/**
 * Build cdf maps to be used for IBL importance sampling.
 */
class IblCdfGenerator {
    /**
     * Returns whether the CDF renderer is supported by the current engine
     */
    get isSupported() {
        const engine = EngineStore.LastCreatedEngine;
        if (!engine) {
            return false;
        }
        return engine.getCaps().texelFetch;
    }
    /**
     * Gets the IBL source texture being used by the CDF renderer
     */
    get iblSource() {
        return this._iblSource;
    }
    /**
     * Sets the IBL source texture to be used by the CDF renderer.
     * This will trigger recreation of the CDF assets.
     */
    set iblSource(source) {
        if (this._iblSource === source) {
            return;
        }
        this._disposeTextures();
        this._iblSource = source;
        if (!source) {
            return;
        }
        if (source.isCube) {
            if (source.isReadyOrNotBlocking()) {
                this._recreateAssetsFromNewIbl();
            }
            else {
                source.onLoadObservable.addOnce(this._recreateAssetsFromNewIbl.bind(this, source));
            }
        }
        else {
            if (source.isReadyOrNotBlocking()) {
                this._recreateAssetsFromNewIbl();
            }
            else {
                source.onLoadObservable.addOnce(this._recreateAssetsFromNewIbl.bind(this, source));
            }
        }
    }
    _recreateAssetsFromNewIbl() {
        if (this._debugPass) {
            this._debugPass.dispose();
        }
        this._createTextures();
        if (this._debugPass) {
            // Recreate the debug pass because of the new textures
            this._createDebugPass();
        }
    }
    /**
     * Return the cumulative distribution function (CDF) texture
     * @returns Return the cumulative distribution function (CDF) texture
     */
    getIcdfTexture() {
        return this._icdfPT ? this._icdfPT : this._dummyTexture;
    }
    /**
     * Sets params that control the position and scaling of the debug display on the screen.
     * @param x Screen X offset of the debug display (0-1)
     * @param y Screen Y offset of the debug display (0-1)
     * @param widthScale X scale of the debug display (0-1)
     * @param heightScale Y scale of the debug display (0-1)
     */
    setDebugDisplayParams(x, y, widthScale, heightScale) {
        this._debugSizeParams.set(x, y, widthScale, heightScale);
    }
    /**
     * The name of the debug pass post process
     */
    get debugPassName() {
        return this._debugPassName;
    }
    /**
     * Gets the debug pass post process
     * @returns The post process
     */
    getDebugPassPP() {
        if (!this._debugPass) {
            this._createDebugPass();
        }
        return this._debugPass;
    }
    /**
     * Instanciates the CDF renderer
     * @param sceneOrEngine Scene to attach to
     * @returns The CDF renderer
     */
    constructor(sceneOrEngine) {
        this._cachedDominantDirection = null;
        /** Enable the debug view for this pass */
        this.debugEnabled = false;
        this._debugSizeParams = new Vector4(0.0, 0.0, 1.0, 1.0);
        this._debugPassName = "CDF Debug";
        /**
         * Observable that triggers when the CDF renderer is ready
         */
        this.onGeneratedObservable = new Observable();
        if (sceneOrEngine) {
            if (IblCdfGenerator._IsScene(sceneOrEngine)) {
                this._scene = sceneOrEngine;
            }
            else {
                this._engine = sceneOrEngine;
            }
        }
        else {
            this._scene = EngineStore.LastCreatedScene;
        }
        if (this._scene) {
            this._engine = this._scene.getEngine();
        }
        if (!this.isSupported) {
            Logger.Warn("CDF renderer is not supported by the current engine.");
            return;
        }
        const blackPixels = new Uint16Array([0, 0, 0, 255]);
        this._dummyTexture = new RawTexture(blackPixels, 1, 1, Engine.TEXTUREFORMAT_RGBA, sceneOrEngine, false, false, undefined, 2);
        if (this._scene) {
            IblCdfGenerator._SceneComponentInitialization(this._scene);
        }
    }
    _createTextures() {
        const size = this._iblSource ? { width: this._iblSource.getSize().width, height: this._iblSource.getSize().height } : { width: 1, height: 1 };
        if (!this._iblSource) {
            this._iblSource = RawTexture.CreateRTexture(new Uint8Array([255]), 1, 1, this._engine, false, false, 1, 0);
            this._iblSource.name = "Placeholder IBL Source";
        }
        if (this._iblSource.isCube) {
            size.width *= 4;
            size.height *= 2;
            // Force the resolution to be a power of 2 because we rely on the
            // auto-mipmap generation for the scaled luminance texture to produce
            // a 1x1 mip that represents the true average pixel intensity of the IBL.
            size.width = 1 << Math.floor(Math.log2(size.width));
            size.height = 1 << Math.floor(Math.log2(size.height));
        }
        const isWebGPU = this._engine.isWebGPU;
        // Create CDF maps (Cumulative Distribution Function) to assist in importance sampling
        const cdfOptions = {
            generateDepthBuffer: false,
            generateMipMaps: false,
            format: 6,
            type: 1,
            samplingMode: 1,
            shaderLanguage: isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
            gammaSpace: false,
            extraInitializationsAsync: async () => {
                if (isWebGPU) {
                    await Promise.all([__vitePreload(() => import('./iblCdfx.fragment-CMtfE_Ro.js'),true              ?__vite__mapDeps([5,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./iblCdfy.fragment-BpWpCgpC.js'),true              ?__vite__mapDeps([6,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./iblScaledLuminance.fragment-CI0s2UUc.js'),true              ?__vite__mapDeps([7,1,2,3,8]):void 0,import.meta.url)]);
                }
                else {
                    await Promise.all([__vitePreload(() => import('./iblCdfx.fragment-eSoc3zz0.js'),true              ?__vite__mapDeps([9,1,2,3]):void 0,import.meta.url), __vitePreload(() => import('./iblCdfy.fragment-D6HMvppy.js'),true              ?__vite__mapDeps([10,1,2,3,11]):void 0,import.meta.url), __vitePreload(() => import('./iblScaledLuminance.fragment-Y8LEQjH0.js'),true              ?__vite__mapDeps([12,1,2,3,11]):void 0,import.meta.url)]);
                }
            },
        };
        const icdfOptions = {
            generateDepthBuffer: false,
            generateMipMaps: false,
            format: 5,
            type: 2,
            samplingMode: 1,
            shaderLanguage: isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
            gammaSpace: false,
            extraInitializationsAsync: async () => {
                if (isWebGPU) {
                    await Promise.all([__vitePreload(() => import('./iblIcdf.fragment-GfR4QU_r.js'),true              ?__vite__mapDeps([13,1,2,3,8]):void 0,import.meta.url), __vitePreload(() => import('./iblDominantDirection.fragment-BAqNHMjM.js'),true              ?__vite__mapDeps([14,1,2,3,8,15,16]):void 0,import.meta.url)]);
                }
                else {
                    await Promise.all([__vitePreload(() => import('./iblIcdf.fragment-DkDbLB7r.js'),true              ?__vite__mapDeps([17,1,2,3,11]):void 0,import.meta.url), __vitePreload(() => import('./iblDominantDirection.fragment-cgOKTZl4.js'),true              ?__vite__mapDeps([18,1,2,3,11,19,20]):void 0,import.meta.url)]);
                }
            },
        };
        this._cdfyPT = new ProceduralTexture("cdfyTexture", { width: size.width, height: size.height + 1 }, "iblCdfy", this._scene, cdfOptions, false, false);
        this._cdfyPT.autoClear = false;
        this._cdfyPT.setTexture("iblSource", this._iblSource);
        this._cdfyPT.setInt("iblHeight", size.height);
        this._cdfyPT.wrapV = 0;
        this._cdfyPT.refreshRate = 0;
        if (this._iblSource.isCube) {
            this._cdfyPT.defines = "#define IBL_USE_CUBE_MAP\n";
        }
        this._cdfxPT = new ProceduralTexture("cdfxTexture", { width: size.width + 1, height: 1 }, "iblCdfx", this._scene, cdfOptions, false, false);
        this._cdfxPT.autoClear = false;
        this._cdfxPT.setTexture("cdfy", this._cdfyPT);
        this._cdfxPT.refreshRate = 0;
        this._cdfxPT.wrapU = 0;
        this._scaledLuminancePT = new ProceduralTexture("iblScaledLuminance", { width: size.width, height: size.height }, "iblScaledLuminance", this._scene, { ...cdfOptions, samplingMode: 3, generateMipMaps: true }, true, false);
        this._scaledLuminancePT.autoClear = false;
        this._scaledLuminancePT.setTexture("iblSource", this._iblSource);
        this._scaledLuminancePT.setInt("iblHeight", size.height);
        this._scaledLuminancePT.setInt("iblWidth", size.width);
        this._scaledLuminancePT.refreshRate = 0;
        if (this._iblSource.isCube) {
            this._scaledLuminancePT.defines = "#define IBL_USE_CUBE_MAP\n";
        }
        this._icdfPT = new ProceduralTexture("icdfTexture", { width: size.width, height: size.height }, "iblIcdf", this._scene, icdfOptions, false, false);
        this._icdfPT.autoClear = false;
        this._icdfPT.setTexture("cdfy", this._cdfyPT);
        this._icdfPT.setTexture("cdfx", this._cdfxPT);
        this._icdfPT.setTexture("iblSource", this._iblSource);
        this._icdfPT.setTexture("scaledLuminanceSampler", this._scaledLuminancePT);
        this._icdfPT.refreshRate = 0;
        this._icdfPT.wrapV = 0;
        this._icdfPT.wrapU = 0;
        if (this._iblSource.isCube) {
            this._icdfPT.defines = "#define IBL_USE_CUBE_MAP\n";
        }
        // Once the textures are generated, notify that they are ready to use.
        this._icdfPT.onGeneratedObservable.addOnce(() => {
            this.onGeneratedObservable.notifyObservers();
        });
        this._dominantDirectionPT = new ProceduralTexture("iblDominantDirection", { width: 1, height: 1 }, "iblDominantDirection", this._scene, icdfOptions, false, false);
        this._dominantDirectionPT.autoClear = false;
        this._dominantDirectionPT.setTexture("icdfSampler", this._icdfPT);
        this._dominantDirectionPT.refreshRate = 0;
        this._dominantDirectionPT.defines = "#define NUM_SAMPLES 32u\n";
    }
    _disposeTextures() {
        this._cdfyPT?.dispose();
        this._cdfxPT?.dispose();
        this._icdfPT?.dispose();
        this._scaledLuminancePT?.dispose();
        this._dominantDirectionPT?.dispose();
    }
    _createDebugPass() {
        if (this._debugPass) {
            this._debugPass.dispose();
        }
        const isWebGPU = this._engine.isWebGPU;
        const debugOptions = {
            width: this._engine.getRenderWidth(),
            height: this._engine.getRenderHeight(),
            samplingMode: Texture.BILINEAR_SAMPLINGMODE,
            engine: this._engine,
            textureType: 0,
            uniforms: ["sizeParams"],
            samplers: ["cdfy", "icdf", "cdfx", "iblSource"],
            defines: this._iblSource?.isCube ? "#define IBL_USE_CUBE_MAP\n" : "",
            shaderLanguage: isWebGPU ? 1 /* ShaderLanguage.WGSL */ : 0 /* ShaderLanguage.GLSL */,
            extraInitializations: (useWebGPU, list) => {
                if (useWebGPU) {
                    list.push(__vitePreload(() => import('./iblCdfDebug.fragment-D8xPZtFT.js'),true              ?__vite__mapDeps([21,1,2,3]):void 0,import.meta.url));
                }
                else {
                    list.push(__vitePreload(() => import('./iblCdfDebug.fragment-BCZOycba.js'),true              ?__vite__mapDeps([22,1,2,3]):void 0,import.meta.url));
                }
            },
        };
        this._debugPass = new PostProcess(this._debugPassName, "iblCdfDebug", debugOptions);
        const debugEffect = this._debugPass.getEffect();
        if (debugEffect) {
            debugEffect.defines = this._iblSource?.isCube ? "#define IBL_USE_CUBE_MAP\n" : "";
        }
        if (this._iblSource?.isCube) {
            this._debugPass.updateEffect("#define IBL_USE_CUBE_MAP\n");
        }
        this._debugPass.onApplyObservable.add((effect) => {
            effect.setTexture("cdfy", this._cdfyPT);
            effect.setTexture("icdf", this._icdfPT);
            effect.setTexture("cdfx", this._cdfxPT);
            effect.setTexture("iblSource", this._iblSource);
            effect.setFloat4("sizeParams", this._debugSizeParams.x, this._debugSizeParams.y, this._debugSizeParams.z, this._debugSizeParams.w);
        });
    }
    /**
     * Checks if the CDF renderer is ready
     * @returns true if the CDF renderer is ready
     */
    isReady() {
        return (this._iblSource &&
            this._iblSource.name !== "Placeholder IBL Source" &&
            this._iblSource.isReady() &&
            this._cdfyPT &&
            this._cdfyPT.isReady() &&
            this._icdfPT &&
            this._icdfPT.isReady() &&
            this._cdfxPT &&
            this._cdfxPT.isReady() &&
            this._scaledLuminancePT &&
            this._scaledLuminancePT.isReady());
    }
    /**
     * Explicitly trigger generation of CDF maps when they are ready to render.
     * @returns Promise that resolves when the CDF maps are rendered.
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention, @typescript-eslint/promise-function-async
    renderWhenReady() {
        this._cachedDominantDirection = null;
        // Even if a IBL source must be set before calling this function, _icdfPT may not yet be created because the creation may be asynchronous (see @set iblSource).
        const icdfPTPromise = new Promise((resolve, reject) => {
            _RetryWithInterval(() => !!this._icdfPT, () => resolve(void 0), () => reject(new Error("Waiting for _icdfPT creation failed")));
        });
        // eslint-disable-next-line github/no-then, @typescript-eslint/promise-function-async
        return icdfPTPromise.then(() => {
            // Once the textures are generated, notify that they are ready to use.
            this._icdfPT.onGeneratedObservable.addOnce(() => {
                this.onGeneratedObservable.notifyObservers();
            });
            const promises = [];
            const renderTargets = [this._cdfyPT, this._cdfxPT, this._scaledLuminancePT, this._icdfPT];
            for (const target of renderTargets) {
                promises.push(new Promise((resolve) => {
                    if (target.isReady()) {
                        resolve();
                    }
                    else {
                        target.getEffect().executeWhenCompiled(() => {
                            resolve();
                        });
                    }
                }));
            }
            // eslint-disable-next-line github/no-then
            return Promise.all(promises).then(() => {
                for (const target of renderTargets) {
                    target.render();
                }
            });
        });
    }
    /**
     * Finds the average direction of the highest intensity areas of the IBL source
     * @returns Async promise that resolves to the dominant direction of the IBL source
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async
    findDominantDirection() {
        if (this._cachedDominantDirection) {
            return Promise.resolve(this._cachedDominantDirection);
        }
        return new Promise((resolve) => {
            this._dominantDirectionPT.onGeneratedObservable.addOnce(() => {
                const data = new Float32Array(4);
                // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                this._dominantDirectionPT.readPixels(0, 0, data, true).then(() => {
                    const dominantDirection = new Vector3(data[0], data[1], data[2]);
                    this._cachedDominantDirection = dominantDirection;
                    resolve(dominantDirection);
                });
            });
            if (this.isReady()) {
                if (this._dominantDirectionPT.isReady()) {
                    this._dominantDirectionPT.render();
                }
                else {
                    this._dominantDirectionPT.getEffect().executeWhenCompiled(() => {
                        this._dominantDirectionPT.render();
                    });
                }
            }
            else {
                this.onGeneratedObservable.addOnce(() => {
                    if (this._dominantDirectionPT.isReady()) {
                        this._dominantDirectionPT.render();
                    }
                    else {
                        this._dominantDirectionPT.getEffect().executeWhenCompiled(() => {
                            this._dominantDirectionPT.render();
                        });
                    }
                });
            }
        });
    }
    /**
     * Disposes the CDF renderer and associated resources
     */
    dispose() {
        this._disposeTextures();
        this._dummyTexture.dispose();
        if (this._debugPass) {
            this._debugPass.dispose();
        }
        this.onGeneratedObservable.clear();
    }
    static _IsScene(sceneOrEngine) {
        return sceneOrEngine.getClassName() === "Scene";
    }
}
/**
 * @internal
 */
IblCdfGenerator._SceneComponentInitialization = (_) => {
    throw _WarnImport("IblCdfGeneratorSceneComponentSceneComponent");
};

const iblCdfGenerator = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    IblCdfGenerator
}, Symbol.toStringTag, { value: 'Module' }));

export { IblCdfGenerator as I, ProceduralTexture as P, iblCdfGenerator as i };
//# sourceMappingURL=iblCdfGenerator-CtGL_NwH.js.map
