import { aY as UniqueIdGenerator, d as Tools, _ as __decorate, s as serialize } from './index-CxR0Yf_7.js';

/**
 * PostProcessRenderPipeline
 * @see https://doc.babylonjs.com/features/featuresDeepDive/postProcesses/postProcessRenderPipeline
 */
class PostProcessRenderPipeline {
    /**
     * Gets pipeline name
     */
    get name() {
        return this._name;
    }
    /** Gets the list of attached cameras */
    get cameras() {
        return this._cameras;
    }
    /**
     * Gets the active engine
     */
    get engine() {
        return this._engine;
    }
    /**
     * Initializes a PostProcessRenderPipeline
     * @param _engine engine to add the pipeline to
     * @param name name of the pipeline
     */
    constructor(_engine, name) {
        this._engine = _engine;
        /**
         * Gets the unique id of the post process rendering pipeline
         */
        this.uniqueId = UniqueIdGenerator.UniqueId;
        this._name = name;
        this._renderEffects = {};
        this._renderEffectsForIsolatedPass = new Array();
        this._cameras = [];
    }
    /**
     * Gets the class name
     * @returns "PostProcessRenderPipeline"
     */
    getClassName() {
        return "PostProcessRenderPipeline";
    }
    /**
     * If all the render effects in the pipeline are supported
     */
    get isSupported() {
        for (const renderEffectName in this._renderEffects) {
            if (Object.prototype.hasOwnProperty.call(this._renderEffects, renderEffectName)) {
                if (!this._renderEffects[renderEffectName].isSupported) {
                    return false;
                }
            }
        }
        return true;
    }
    /**
     * Adds an effect to the pipeline
     * @param renderEffect the effect to add
     */
    addEffect(renderEffect) {
        this._renderEffects[renderEffect._name] = renderEffect;
    }
    // private
    /** @internal */
    _rebuild() { }
    /**
     * @internal
     */
    _enableEffect(renderEffectName, cameras) {
        const renderEffects = this._renderEffects[renderEffectName];
        if (!renderEffects) {
            return;
        }
        renderEffects._enable(Tools.MakeArray(cameras || this._cameras));
    }
    /**
     * @internal
     */
    _disableEffect(renderEffectName, cameras) {
        const renderEffects = this._renderEffects[renderEffectName];
        if (!renderEffects) {
            return;
        }
        renderEffects._disable(Tools.MakeArray(cameras || this._cameras));
    }
    /**
     * @internal
     */
    _attachCameras(cameras, unique) {
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        const indicesToDelete = [];
        let i;
        for (i = 0; i < cams.length; i++) {
            const camera = cams[i];
            if (!camera) {
                continue;
            }
            if (this._cameras.indexOf(camera) === -1) {
                this._cameras.push(camera);
            }
            else if (unique) {
                indicesToDelete.push(i);
            }
        }
        for (i = 0; i < indicesToDelete.length; i++) {
            cams.splice(indicesToDelete[i], 1);
        }
        for (const renderEffectName in this._renderEffects) {
            if (Object.prototype.hasOwnProperty.call(this._renderEffects, renderEffectName)) {
                this._renderEffects[renderEffectName]._attachCameras(cams);
            }
        }
    }
    /**
     * @internal
     */
    _detachCameras(cameras) {
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        for (const renderEffectName in this._renderEffects) {
            if (Object.prototype.hasOwnProperty.call(this._renderEffects, renderEffectName)) {
                this._renderEffects[renderEffectName]._detachCameras(cams);
            }
        }
        for (let i = 0; i < cams.length; i++) {
            this._cameras.splice(this._cameras.indexOf(cams[i]), 1);
        }
    }
    /** @internal */
    _update() {
        for (const renderEffectName in this._renderEffects) {
            if (Object.prototype.hasOwnProperty.call(this._renderEffects, renderEffectName)) {
                this._renderEffects[renderEffectName]._update();
            }
        }
        for (let i = 0; i < this._cameras.length; i++) {
            if (!this._cameras[i]) {
                continue;
            }
            const cameraName = this._cameras[i].name;
            if (this._renderEffectsForIsolatedPass[cameraName]) {
                this._renderEffectsForIsolatedPass[cameraName]._update();
            }
        }
    }
    /** @internal */
    _reset() {
        this._renderEffects = {};
        this._renderEffectsForIsolatedPass = new Array();
    }
    _enableMSAAOnFirstPostProcess(sampleCount) {
        if (!this._engine._features.supportMSAA) {
            return false;
        }
        // Set samples of the very first post process to 4 to enable native anti-aliasing in browsers that support webGL 2.0 (See: https://github.com/BabylonJS/Babylon.js/issues/3754)
        const effectKeys = Object.keys(this._renderEffects);
        if (effectKeys.length > 0) {
            const postProcesses = this._renderEffects[effectKeys[0]].getPostProcesses();
            if (postProcesses) {
                postProcesses[0].samples = sampleCount;
            }
        }
        return true;
    }
    /**
     * Ensures that all post processes in the pipeline are the correct size according to the
     * the viewport's required size
     */
    _adaptPostProcessesToViewPort() {
        const effectKeys = Object.keys(this._renderEffects);
        for (const effectKey of effectKeys) {
            const postProcesses = this._renderEffects[effectKey].getPostProcesses();
            if (postProcesses) {
                for (const postProcess of postProcesses) {
                    postProcess.adaptScaleToCurrentViewport = true;
                }
            }
        }
    }
    /**
     * Sets the required values to the prepass renderer.
     * @param prePassRenderer defines the prepass renderer to setup.
     * @returns true if the pre pass is needed.
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    setPrePassRenderer(prePassRenderer) {
        // Do Nothing by default
        return false;
    }
    /**
     * Disposes of the pipeline
     */
    dispose() {
        // Must be implemented by children
    }
}
__decorate([
    serialize()
], PostProcessRenderPipeline.prototype, "_name", void 0);

/**
 * This represents a set of one or more post processes in Babylon.
 * A post process can be used to apply a shader to a texture after it is rendered.
 * @example https://doc.babylonjs.com/features/featuresDeepDive/postProcesses/postProcessRenderPipeline
 */
class PostProcessRenderEffect {
    /**
     * Instantiates a post process render effect.
     * A post process can be used to apply a shader to a texture after it is rendered.
     * @param engine The engine the effect is tied to
     * @param name The name of the effect
     * @param getPostProcesses A function that returns a set of post processes which the effect will run in order to be run.
     * @param singleInstance False if this post process can be run on multiple cameras. (default: true)
     */
    constructor(engine, name, getPostProcesses, singleInstance = true) {
        this._name = name;
        this._singleInstance = singleInstance;
        this._getPostProcesses = getPostProcesses;
        this._cameras = {};
        this._indicesForCamera = {};
        this._postProcesses = {};
    }
    /**
     * Checks if all the post processes in the effect are supported.
     */
    get isSupported() {
        for (const index in this._postProcesses) {
            if (Object.prototype.hasOwnProperty.call(this._postProcesses, index)) {
                const pps = this._postProcesses[index];
                for (let ppIndex = 0; ppIndex < pps.length; ppIndex++) {
                    if (!pps[ppIndex].isSupported) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    /**
     * Updates the current state of the effect
     * @internal
     */
    _update() { }
    /**
     * Attaches the effect on cameras
     * @param cameras The camera to attach to.
     * @internal
     */
    _attachCameras(cameras) {
        let cameraKey;
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        for (let i = 0; i < cams.length; i++) {
            const camera = cams[i];
            if (!camera) {
                continue;
            }
            const cameraName = camera.name;
            if (this._singleInstance) {
                cameraKey = 0;
            }
            else {
                cameraKey = cameraName;
            }
            if (!this._postProcesses[cameraKey]) {
                const postProcess = this._getPostProcesses();
                if (postProcess) {
                    this._postProcesses[cameraKey] = Array.isArray(postProcess) ? postProcess : [postProcess];
                }
            }
            if (!this._indicesForCamera[cameraName]) {
                this._indicesForCamera[cameraName] = [];
            }
            const pps = this._postProcesses[cameraKey];
            for (const postProcess of pps) {
                const index = camera.attachPostProcess(postProcess);
                this._indicesForCamera[cameraName].push(index);
            }
            if (!this._cameras[cameraName]) {
                this._cameras[cameraName] = camera;
            }
        }
    }
    /**
     * Detaches the effect on cameras
     * @param cameras The camera to detach from.
     * @internal
     */
    _detachCameras(cameras) {
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        for (let i = 0; i < cams.length; i++) {
            const camera = cams[i];
            const cameraName = camera.name;
            const postProcesses = this._postProcesses[this._singleInstance ? 0 : cameraName];
            if (postProcesses) {
                for (const postProcess of postProcesses) {
                    camera.detachPostProcess(postProcess);
                }
            }
            if (this._cameras[cameraName]) {
                this._cameras[cameraName] = null;
            }
            delete this._indicesForCamera[cameraName];
        }
    }
    /**
     * Enables the effect on given cameras
     * @param cameras The camera to enable.
     * @internal
     */
    _enable(cameras) {
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        for (let i = 0; i < cams.length; i++) {
            const camera = cams[i];
            const cameraName = camera.name;
            const cameraKey = this._singleInstance ? 0 : cameraName;
            for (let j = 0; j < this._indicesForCamera[cameraName].length; j++) {
                const index = this._indicesForCamera[cameraName][j];
                const postProcess = camera._postProcesses[index];
                if (postProcess === undefined || postProcess === null) {
                    cams[i].attachPostProcess(this._postProcesses[cameraKey][j], index);
                }
            }
        }
    }
    /**
     * Disables the effect on the given cameras
     * @param cameras The camera to disable.
     * @internal
     */
    _disable(cameras) {
        const cams = Tools.MakeArray(cameras || this._cameras);
        if (!cams) {
            return;
        }
        for (let i = 0; i < cams.length; i++) {
            const camera = cams[i];
            const cameraName = camera.name;
            const pps = this._postProcesses[this._singleInstance ? 0 : cameraName];
            for (const postProcess of pps) {
                camera.detachPostProcess(postProcess);
            }
        }
    }
    /**
     * Gets a list of the post processes contained in the effect.
     * @param camera The camera to get the post processes on.
     * @returns The list of the post processes in the effect.
     */
    getPostProcesses(camera) {
        if (this._singleInstance) {
            return this._postProcesses[0];
        }
        else {
            if (!camera) {
                return null;
            }
            return this._postProcesses[camera.name];
        }
    }
}

export { PostProcessRenderPipeline as P, PostProcessRenderEffect as a };
//# sourceMappingURL=postProcessRenderEffect-BIXOs3xJ.js.map
