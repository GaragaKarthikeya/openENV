const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./webAudioBus-BnckzWcN.js","./abstractAudioBus-K_VBWUgv.js","./webAudioBaseSubGraph-CuchFrSM.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./spatialWebAudio-DH890j7t.js","./webAudioMainBus-Dclg8YIq.js","./webAudioStaticSound-D8ZwU_iO.js","./abstractSoundInstance-CH6fBm2E.js","./abstractSoundSource-Bo3P1qg1.js","./webAudioSoundSource-__1CMM9A.js","./webAudioStreamingSound-CvXX6uFt.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { O as Observable, V as Vector3, ao as Quaternion, L as Logger, by as PrecisionDate, M as Matrix, aq as EngineStore, A as AbstractEngine } from './index-CxR0Yf_7.js';

var AudioNodeType;
(function (AudioNodeType) {
    AudioNodeType[AudioNodeType["HAS_INPUTS"] = 1] = "HAS_INPUTS";
    AudioNodeType[AudioNodeType["HAS_OUTPUTS"] = 2] = "HAS_OUTPUTS";
    AudioNodeType[AudioNodeType["HAS_INPUTS_AND_OUTPUTS"] = 3] = "HAS_INPUTS_AND_OUTPUTS";
})(AudioNodeType || (AudioNodeType = {}));
/**
 * Abstract class for an audio node.
 *
 * An audio node is a processing unit that can receive audio data from an upstream node and/or send audio data to a
 * downstream node.
 *
 * Nodes can be connected to other nodes to create an audio graph. The audio graph represents the flow of audio data.
 *
 * There are 3 types of audio nodes:
 * 1. Input: Receives audio data from upstream nodes.
 * 2. Output: Sends audio data to downstream nodes.
 * 3. Input/Output: Receives audio data from upstream nodes and sends audio data to downstream nodes.
 */
class AbstractAudioNode {
    constructor(engine, nodeType) {
        /**
         * Observable for when the audio node is disposed.
         */
        this.onDisposeObservable = new Observable();
        this.engine = engine;
        if (nodeType & 1 /* AudioNodeType.HAS_INPUTS */) {
            this._upstreamNodes = new Set();
        }
        if (nodeType & 2 /* AudioNodeType.HAS_OUTPUTS */) {
            this._downstreamNodes = new Set();
        }
    }
    /**
     * Releases associated resources.
     * - Triggers `onDisposeObservable`.
     * @see {@link onDisposeObservable}
     */
    dispose() {
        if (this._downstreamNodes) {
            for (const node of Array.from(this._downstreamNodes)) {
                if (!this._disconnect(node)) {
                    throw new Error("Disconnect failed");
                }
            }
            this._downstreamNodes.clear();
        }
        if (this._upstreamNodes) {
            for (const node of Array.from(this._upstreamNodes)) {
                if (!node._disconnect(this)) {
                    throw new Error("Disconnect failed");
                }
            }
            this._upstreamNodes.clear();
        }
        this.onDisposeObservable.notifyObservers(this);
        this.onDisposeObservable.clear();
    }
    /**
     * Connect to a downstream audio input node.
     * @param node - The downstream audio input node to connect
     * @returns `true` if the node is successfully connected; otherwise `false`
     */
    _connect(node) {
        if (!this._downstreamNodes) {
            return false;
        }
        if (this._downstreamNodes.has(node)) {
            return false;
        }
        if (!node._onConnect(this)) {
            return false;
        }
        this._downstreamNodes.add(node);
        return true;
    }
    /**
     * Disconnects a downstream audio input node.
     * @param node - The downstream audio input node to disconnect
     * @returns `true` if the node is successfully disconnected; otherwise `false`
     */
    _disconnect(node) {
        if (!this._downstreamNodes) {
            return false;
        }
        if (!this._downstreamNodes.delete(node)) {
            return false;
        }
        return node._onDisconnect(this);
    }
    /**
     * Called when an upstream audio output node is connecting.
     * @param node - The connecting upstream audio node
     * @returns `true` if the node is successfully connected; otherwise `false`
     */
    _onConnect(node) {
        if (!this._upstreamNodes) {
            return false;
        }
        if (this._upstreamNodes.has(node)) {
            return false;
        }
        this._upstreamNodes.add(node);
        return true;
    }
    /**
     * Called when an upstream audio output node disconnects.
     * @param node - The disconnecting upstream audio node
     * @returns `true` if node is sucessfully disconnected; otherwise `false`
     */
    _onDisconnect(node) {
        return this._upstreamNodes?.delete(node) ?? false;
    }
}
/**
 * Abstract class for a named audio node.
 */
class AbstractNamedAudioNode extends AbstractAudioNode {
    constructor(name, engine, nodeType) {
        super(engine, nodeType);
        /**
         * Observable for when the audio node is renamed.
         */
        this.onNameChangedObservable = new Observable();
        this._name = name;
    }
    /**
     * The name of the audio node.
     * - Triggers `onNameChangedObservable` when changed.
     * @see {@link onNameChangedObservable}
     */
    get name() {
        return this._name;
    }
    set name(newName) {
        if (this._name === newName) {
            return;
        }
        const oldName = this._name;
        this._name = newName;
        this.onNameChangedObservable.notifyObservers({ newName, oldName, node: this });
    }
    dispose() {
        super.dispose();
        this.onNameChangedObservable.clear();
    }
}

/**
 * Provides a common interface for attaching an audio listener or source to a specific entity, ensuring only one entity
 * is attached at a time.
 * @internal
 */
class _SpatialAudioAttacherComponent {
    /** @internal */
    constructor(spatialAudioNode) {
        /** @internal */
        this._attachmentType = 3 /* SpatialAudioAttachmentType.PositionAndRotation */;
        this._position = new Vector3();
        this._rotationQuaternion = new Quaternion();
        this._sceneNode = null;
        this._useBoundingBox = false;
        /**
         * Releases associated resources.
         */
        this.dispose = () => {
            this.detach();
        };
        this._spatialAudioNode = spatialAudioNode;
    }
    /**
     * Returns `true` if attached to a scene node; otherwise returns `false`.
     */
    get isAttached() {
        return this._sceneNode !== null;
    }
    /**
     * Attaches to a scene node.
     *
     * Detaches automatically before attaching to the given scene node.
     * If `sceneNode` is `null` it is the same as calling `detach()`.
     *
     * @param sceneNode The scene node to attach to, or `null` to detach.
     * @param useBoundingBox Whether to use the scene node's bounding box for positioning. Defaults to `false`.
     * @param attachmentType Whether to attach to the scene node's position and/or rotation. Defaults to `PositionAndRotation`.
     */
    attach(sceneNode, useBoundingBox, attachmentType) {
        if (this._sceneNode === sceneNode) {
            return;
        }
        this.detach();
        if (!sceneNode) {
            return;
        }
        this._attachmentType = attachmentType;
        this._sceneNode = sceneNode;
        this._sceneNode.onDisposeObservable.add(this.dispose);
        this._useBoundingBox = useBoundingBox;
    }
    /**
     * Detaches from the scene node if attached.
     */
    detach() {
        this._sceneNode?.onDisposeObservable.removeCallback(this.dispose);
        this._sceneNode = null;
    }
    /**
     * Updates the position and rotation of the associated audio engine object in the audio rendering graph.
     *
     * This is called automatically by default and only needs to be called manually if automatic updates are disabled.
     */
    update() {
        if (this._attachmentType & 1 /* SpatialAudioAttachmentType.Position */) {
            if (this._useBoundingBox && this._sceneNode.getBoundingInfo) {
                this._position.copyFrom(this._sceneNode.getBoundingInfo().boundingBox.centerWorld);
            }
            else {
                this._sceneNode?.getWorldMatrix().getTranslationToRef(this._position);
            }
            this._spatialAudioNode.position.copyFrom(this._position);
            this._spatialAudioNode._updatePosition();
        }
        if (this._attachmentType & 2 /* SpatialAudioAttachmentType.Rotation */) {
            this._sceneNode?.getWorldMatrix().decompose(undefined, this._rotationQuaternion);
            this._spatialAudioNode.rotationQuaternion.copyFrom(this._rotationQuaternion);
            this._spatialAudioNode._updateRotation();
        }
    }
}

const _FileExtensionRegex = new RegExp("\\.(\\w{3,4})($|\\?)");
const CurveLength = 100;
const TmpLineValues = new Float32Array([0, 0]);
let TmpCurveValues = null;
let ExpCurve = null;
let LogCurve = null;
/**
 * @returns A Float32Array representing an exponential ramp from (0, 0) to (1, 1).
 */
function GetExpCurve() {
    if (!ExpCurve) {
        ExpCurve = new Float32Array(CurveLength);
        const increment = 1 / (CurveLength - 1);
        let x = increment;
        for (let i = 1; i < CurveLength; i++) {
            ExpCurve[i] = Math.exp(-11.512925464970227 * (1 - x));
            x += increment;
        }
    }
    return ExpCurve;
}
/**
 * @returns A Float32Array representing a logarithmic ramp from (0, 0) to (1, 1).
 */
function GetLogCurve() {
    if (!LogCurve) {
        LogCurve = new Float32Array(CurveLength);
        const increment = 1 / CurveLength;
        let x = increment;
        for (let i = 0; i < CurveLength; i++) {
            LogCurve[i] = 1 + Math.log10(x) / Math.log10(CurveLength);
            x += increment;
        }
    }
    return LogCurve;
}
/** @internal */
function _GetAudioParamCurveValues(shape, from, to) {
    if (!TmpCurveValues) {
        TmpCurveValues = new Float32Array(CurveLength);
    }
    let normalizedCurve;
    if (shape === "linear" /* AudioParameterRampShape.Linear */) {
        TmpLineValues[0] = from;
        TmpLineValues[1] = to;
        return TmpLineValues;
    }
    else if (shape === "exponential" /* AudioParameterRampShape.Exponential */) {
        normalizedCurve = GetExpCurve();
    }
    else if (shape === "logarithmic" /* AudioParameterRampShape.Logarithmic */) {
        normalizedCurve = GetLogCurve();
    }
    else {
        throw new Error(`Unknown ramp shape: ${shape}`);
    }
    const direction = Math.sign(to - from);
    const range = Math.abs(to - from);
    if (direction === 1) {
        for (let i = 0; i < normalizedCurve.length; i++) {
            TmpCurveValues[i] = from + range * normalizedCurve[i];
        }
    }
    else {
        let j = CurveLength - 1;
        for (let i = 0; i < normalizedCurve.length; i++, j--) {
            TmpCurveValues[i] = from - range * (1 - normalizedCurve[j]);
        }
    }
    return TmpCurveValues;
}
/** @internal */
function _CleanUrl(url) {
    return url.replace(/#/gm, "%23");
}

/**
 * Minimum duration in seconds for a ramp to be considered valid.
 *
 * If the duration is less than this value, the value will be set immediately instead of being ramped smoothly since
 * there is no perceptual difference for such short durations, so a ramp is not needed.
 */
const MinRampDuration = 0.000001;
let Warn = true;
/** @internal */
class _WebAudioParameterComponent {
    /** @internal */
    constructor(engine, param) {
        this._rampEndTime = 0;
        this._engine = engine;
        this._param = param;
        this._targetValue = param.value;
    }
    /** @internal */
    get isRamping() {
        return this._engine.currentTime < this._rampEndTime;
    }
    /** @internal */
    get targetValue() {
        return this._targetValue;
    }
    set targetValue(value) {
        this.setTargetValue(value);
    }
    /** @internal */
    get value() {
        return this._param.value;
    }
    /** @internal */
    dispose() {
        this._param = null;
        this._engine = null;
    }
    /**
     * Sets the target value of the audio parameter with an optional ramping duration and shape.
     *
     * @internal
     */
    setTargetValue(value, options = null) {
        if (!Number.isFinite(value)) {
            Logger.Warn(`Attempted to set audio parameter to non-finite value: ${value}`);
            return;
        }
        this._param.cancelScheduledValues(0);
        const shape = typeof options?.shape === "string" ? options.shape : "linear" /* AudioParameterRampShape.Linear */;
        const startTime = this._engine.currentTime;
        if (shape === "none" /* AudioParameterRampShape.None */) {
            this._param.value = this._targetValue = value;
            this._rampEndTime = startTime;
            return;
        }
        let duration = typeof options?.duration === "number" ? Math.max(options.duration, this._engine.parameterRampDuration) : this._engine.parameterRampDuration;
        this._targetValue = value;
        if ((duration = Math.max(this._engine.parameterRampDuration, duration)) < MinRampDuration) {
            this._param.setValueAtTime(value, startTime);
            return;
        }
        try {
            this._param.setValueCurveAtTime(_GetAudioParamCurveValues(shape, Number.isFinite(this._param.value) ? this._param.value : 0, value), startTime, duration);
            this._rampEndTime = startTime + duration;
        }
        catch (e) {
            if (Warn) {
                Logger.Warn(`Audio parameter ramping failed: ${e.message}`);
                Warn = false;
            }
        }
    }
}

/** @internal */
class _SpatialWebAudioUpdaterComponent {
    /** @internal */
    constructor(parent, autoUpdate, minUpdateTime) {
        this._autoUpdate = true;
        this._lastUpdateTime = 0;
        /** @internal */
        this.minUpdateTime = 0;
        if (!autoUpdate) {
            return;
        }
        this.minUpdateTime = minUpdateTime;
        const update = () => {
            if (!this._autoUpdate) {
                return;
            }
            let skipUpdate = false;
            if (0 < this.minUpdateTime) {
                const now = PrecisionDate.Now;
                if (this._lastUpdateTime && now - this._lastUpdateTime < this.minUpdateTime) {
                    skipUpdate = true;
                }
                this._lastUpdateTime = now;
            }
            if (!skipUpdate) {
                parent.update();
            }
            requestAnimationFrame(update);
        };
        requestAnimationFrame(update);
    }
    /** @internal */
    dispose() {
        this._autoUpdate = false;
    }
}

/**
 * Abstract base class for v2 audio engines.
 *
 * A v2 audio engine based on the WebAudio API can be created with the {@link CreateAudioEngineAsync} function.
 */
class AudioEngineV2 {
    constructor(options) {
        /** Not owned, but all items should be in `_nodes` container, too, which is owned. */
        this._mainBuses = new Set();
        this._sounds = new Set();
        this._soundsArray = null;
        /** Owned top-level sound and bus nodes. */
        this._nodes = new Set();
        this._defaultMainBus = null;
        this._parameterRampDuration = 0.01;
        if (typeof options.parameterRampDuration === "number") {
            this.parameterRampDuration = options.parameterRampDuration;
        }
    }
    /**
     * The default main bus that will be used for audio buses and sounds if their `outBus` option is not set.
     * @see {@link IAudioBusOptions.outBus}
     * @see {@link IAbstractSoundOptions.outBus}
     */
    get defaultMainBus() {
        if (this._mainBuses.size === 0) {
            return null;
        }
        if (!this._defaultMainBus) {
            this._defaultMainBus = Array.from(this._mainBuses)[0];
        }
        return this._defaultMainBus;
    }
    /**
     * The smoothing duration to use when changing audio parameters, in seconds. Defaults to `0.01` (10 milliseconds).
     */
    get parameterRampDuration() {
        return this._parameterRampDuration;
    }
    set parameterRampDuration(value) {
        this._parameterRampDuration = Math.max(0, value);
    }
    /**
     * The list of static and streaming sounds created by the audio engine.
     */
    get sounds() {
        if (!this._soundsArray) {
            this._soundsArray = Array.from(this._sounds);
        }
        return this._soundsArray;
    }
    /**
     * Releases associated resources.
     */
    dispose() {
        const nodeIt = this._nodes.values();
        for (let next = nodeIt.next(); !next.done; next = nodeIt.next()) {
            next.value.dispose();
        }
        this._mainBuses.clear();
        this._nodes.clear();
        this._sounds.clear();
        this._disposeSoundsArray();
        this._defaultMainBus = null;
    }
    /**
     * Unlocks the audio engine if it is locked.
     * - Note that the returned promise may already be resolved if the audio engine is already unlocked.
     * @returns A promise that is resolved when the audio engine is unlocked.
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    unlockAsync() {
        return this.resumeAsync();
    }
    _addMainBus(mainBus) {
        this._mainBuses.add(mainBus);
        this._addNode(mainBus);
    }
    _removeMainBus(mainBus) {
        this._mainBuses.delete(mainBus);
        this._defaultMainBus = null;
        this._removeNode(mainBus);
    }
    _addNode(node) {
        this._nodes.add(node);
    }
    _removeNode(node) {
        this._nodes.delete(node);
    }
    _addSound(sound) {
        this._disposeSoundsArray();
        this._sounds.add(sound);
        this._addNode(sound);
    }
    _removeSound(sound) {
        this._disposeSoundsArray();
        this._sounds.delete(sound);
        this._removeNode(sound);
    }
    _disposeSoundsArray() {
        if (this._soundsArray) {
            this._soundsArray.length = 0;
            this._soundsArray = null;
        }
    }
}

const _SpatialAudioListenerDefaults = {
    position: Vector3.Zero(),
    rotation: Vector3.Zero(),
    rotationQuaternion: new Quaternion(),
};
/**
 * @param options The spatial audio listener options to check.
 * @returns `true` if spatial audio listener options are defined, otherwise `false`.
 */
function _HasSpatialAudioListenerOptions(options) {
    return (options.listenerEnabled ||
        options.listenerMinUpdateTime !== undefined ||
        options.listenerPosition !== undefined ||
        options.listenerRotation !== undefined ||
        options.listenerRotationQuaternion !== undefined);
}
/**
 * Abstract class representing the spatial audio `listener` property on an audio engine.
 *
 * @see {@link AudioEngineV2.listener}
 */
class AbstractSpatialAudioListener {
}

/** @internal */
class _SpatialAudioListener extends AbstractSpatialAudioListener {
    constructor() {
        super();
        this._attacherComponent = null;
        this._attacherComponent = new _SpatialAudioAttacherComponent(this);
    }
    /** @internal */
    get isAttached() {
        return this._attacherComponent !== null && this._attacherComponent.isAttached;
    }
    /**
     * Attaches to a scene node.
     *
     * Detaches automatically before attaching to the given scene node.
     * If `sceneNode` is `null` it is the same as calling `detach()`.
     *
     * @param sceneNode The scene node to attach to, or `null` to detach.
     * @param useBoundingBox Whether to use the bounding box of the node for positioning. Defaults to `false`.
     * @param attachmentType Whether to attach to the node's position and/or rotation. Defaults to `PositionAndRotation`.
     */
    attach(sceneNode, useBoundingBox = false, attachmentType = 3 /* SpatialAudioAttachmentType.PositionAndRotation */) {
        if (!this._attacherComponent) {
            this._attacherComponent = new _SpatialAudioAttacherComponent(this);
        }
        this._attacherComponent.attach(sceneNode, useBoundingBox, attachmentType);
    }
    /**
     * Detaches from the scene node if attached.
     */
    detach() {
        this._attacherComponent?.detach();
    }
    /** @internal */
    dispose() {
        this._attacherComponent?.dispose();
        this._attacherComponent = null;
    }
    /** @internal */
    setOptions(options) {
        if (options.listenerMinUpdateTime !== undefined) {
            this.minUpdateTime = options.listenerMinUpdateTime;
        }
        if (options.listenerPosition) {
            this.position = options.listenerPosition.clone();
        }
        if (options.listenerRotationQuaternion) {
            this.rotationQuaternion = options.listenerRotationQuaternion.clone();
        }
        else if (options.listenerRotation) {
            this.rotation = options.listenerRotation.clone();
        }
        else {
            this.rotationQuaternion = _SpatialAudioListenerDefaults.rotationQuaternion.clone();
        }
        this.update();
    }
}

const TmpMatrix = Matrix.Zero();
const TmpQuaternion = new Quaternion();
const TmpVector1 = Vector3.Zero();
const TmpVector2 = Vector3.Zero();
/** @internal */
function _CreateSpatialAudioListener(engine, autoUpdate, minUpdateTime) {
    const listener = engine._audioContext.listener;
    if (listener.forwardX &&
        listener.forwardY &&
        listener.forwardZ &&
        listener.positionX &&
        listener.positionY &&
        listener.positionZ &&
        listener.upX &&
        listener.upY &&
        listener.upZ) {
        return new _SpatialWebAudioListener(engine, autoUpdate, minUpdateTime);
    }
    else {
        return new _SpatialWebAudioListenerFallback(engine, autoUpdate, minUpdateTime);
    }
}
class _AbstractSpatialWebAudioListener extends _SpatialAudioListener {
    /** @internal */
    constructor(engine, autoUpdate, minUpdateTime) {
        super();
        this._lastPosition = Vector3.Zero();
        this._lastRotation = Vector3.Zero();
        this._lastRotationQuaternion = new Quaternion();
        /** @internal */
        this.position = Vector3.Zero();
        /** @internal */
        this.rotation = Vector3.Zero();
        /** @internal */
        this.rotationQuaternion = new Quaternion();
        this._listener = engine._audioContext.listener;
        this.engine = engine;
        this._updaterComponent = new _SpatialWebAudioUpdaterComponent(this, autoUpdate, minUpdateTime);
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._updaterComponent.dispose();
        this._updaterComponent = null;
    }
    /** @internal */
    get minUpdateTime() {
        return this._updaterComponent.minUpdateTime;
    }
    /** @internal */
    set minUpdateTime(value) {
        this._updaterComponent.minUpdateTime = value;
    }
    /** @internal */
    update() {
        if (this.isAttached) {
            this._attacherComponent?.update();
        }
        else {
            this._updatePosition();
            this._updateRotation();
        }
    }
    _updatePosition() {
        if (this._lastPosition.equalsWithEpsilon(this.position)) {
            return;
        }
        this._setWebAudioPosition(this.position);
        this._lastPosition.copyFrom(this.position);
    }
    _updateRotation() {
        if (!this._lastRotationQuaternion.equalsWithEpsilon(this.rotationQuaternion)) {
            TmpQuaternion.copyFrom(this.rotationQuaternion);
            this._lastRotationQuaternion.copyFrom(this.rotationQuaternion);
        }
        else if (!this._lastRotation.equalsWithEpsilon(this.rotation)) {
            Quaternion.FromEulerAnglesToRef(this.rotation.x, this.rotation.y, this.rotation.z, TmpQuaternion);
            this._lastRotation.copyFrom(this.rotation);
        }
        else {
            return;
        }
        Matrix.FromQuaternionToRef(TmpQuaternion, TmpMatrix);
        // NB: The WebAudio API is right-handed.
        Vector3.TransformNormalToRef(Vector3.RightHandedForwardReadOnly, TmpMatrix, TmpVector1);
        Vector3.TransformNormalToRef(Vector3.Up(), TmpMatrix, TmpVector2);
        this._setWebAudioOrientation(TmpVector1, TmpVector2);
    }
}
/**
 * Full-featured spatial audio listener for the Web Audio API.
 *
 * Used in browsers that support the `forwardX/Y/Z`, `positionX/Y/Z`, and `upX/Y/Z` properties on the AudioContext listener.
 *
 * NB: Firefox falls back to using this implementation.
 *
 * @see _SpatialWebAudioListenerFallback for the implementation used if only `setPosition` and `setOrientation` are available.
 *
 * NB: This sub property is not backed by a sub node and all properties are set directly on the audio context listener.
 *
 * @internal
 */
class _SpatialWebAudioListener extends _AbstractSpatialWebAudioListener {
    constructor(engine, autoUpdate, minUpdateTime) {
        super(engine, autoUpdate, minUpdateTime);
        const listener = engine._audioContext.listener;
        this._forwardX = new _WebAudioParameterComponent(engine, listener.forwardX);
        this._forwardY = new _WebAudioParameterComponent(engine, listener.forwardY);
        this._forwardZ = new _WebAudioParameterComponent(engine, listener.forwardZ);
        this._positionX = new _WebAudioParameterComponent(engine, listener.positionX);
        this._positionY = new _WebAudioParameterComponent(engine, listener.positionY);
        this._positionZ = new _WebAudioParameterComponent(engine, listener.positionZ);
        this._upX = new _WebAudioParameterComponent(engine, listener.upX);
        this._upY = new _WebAudioParameterComponent(engine, listener.upY);
        this._upZ = new _WebAudioParameterComponent(engine, listener.upZ);
    }
    _setWebAudioPosition(position) {
        // If attached and there is a ramp in progress, we assume another update is coming soon that we can wait for.
        // We don't do this for unattached nodes because there may not be another update coming.
        if (this.isAttached && (this._positionX.isRamping || this._positionY.isRamping || this._positionZ.isRamping)) {
            return;
        }
        this._positionX.targetValue = position.x;
        this._positionY.targetValue = position.y;
        this._positionZ.targetValue = position.z;
    }
    _setWebAudioOrientation(forward, up) {
        // If attached and there is a ramp in progress, we assume another update is coming soon that we can wait for.
        // We don't do this for unattached nodes because there may not be another update coming.
        if (this.isAttached &&
            (this._forwardX.isRamping || this._forwardY.isRamping || this._forwardZ.isRamping || this._upX.isRamping || this._upY.isRamping || this._upZ.isRamping)) {
            return;
        }
        this._forwardX.targetValue = forward.x;
        this._forwardY.targetValue = forward.y;
        this._forwardZ.targetValue = forward.z;
        this._upX.targetValue = up.x;
        this._upY.targetValue = up.y;
        this._upZ.targetValue = up.z;
    }
}
/**
 * Fallback spatial audio listener for the Web Audio API.
 *
 * Used in browsers that do not support the `forwardX/Y/Z`, `positionX/Y/Z`, and `upX/Y/Z` properties on the
 * AudioContext listener.
 *
 * @see _SpatialWebAudioListener for the implementation used if the `forwardX/Y/Z`, `positionX/Y/Z`, and `upX/Y/Z`
 * properties are available.
 *
 * NB: This sub property is not backed by a sub node and all properties are set directly on the audio context listener.
 *
 * @internal
 */
class _SpatialWebAudioListenerFallback extends _AbstractSpatialWebAudioListener {
    _setWebAudioPosition(position) {
        this._listener.setPosition(position.x, position.y, position.z);
    }
    _setWebAudioOrientation(forward, up) {
        this._listener.setOrientation(forward.x, forward.y, forward.z, up.x, up.y, up.z);
    }
}

/**
 * Abstract class for the main audio output node.
 *
 * A main audio output is the last audio node in the audio graph before the audio is sent to the speakers.
 *
 * @see {@link AudioEngineV2.mainOut}
 * @internal
 */
class _MainAudioOut extends AbstractAudioNode {
    constructor(engine) {
        super(engine, 1 /* AudioNodeType.HAS_INPUTS */);
    }
}

/** @internal */
class _WebAudioMainOut extends _MainAudioOut {
    /** @internal */
    constructor(engine) {
        super(engine);
        this._setGainNode(new GainNode(engine._audioContext));
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._volume.dispose();
        this._gainNode.disconnect();
        this._destinationNode.disconnect();
    }
    /** @internal */
    get _inNode() {
        return this._gainNode;
    }
    set _inNode(value) {
        if (this._gainNode === value) {
            return;
        }
        this._setGainNode(value);
    }
    /** @internal */
    get volume() {
        return this._volume.targetValue;
    }
    /** @internal */
    set volume(value) {
        this._volume.targetValue = value;
    }
    get _destinationNode() {
        return this.engine._audioDestination;
    }
    /** @internal */
    getClassName() {
        return "_WebAudioMainOut";
    }
    /** @internal */
    setVolume(value, options = null) {
        this._volume.setTargetValue(value, options);
    }
    _setGainNode(gainNode) {
        if (this._gainNode === gainNode) {
            return;
        }
        this._gainNode?.disconnect();
        gainNode.connect(this._destinationNode);
        this._volume = new _WebAudioParameterComponent(this.engine, gainNode.gain);
        this._gainNode = gainNode;
    }
}

/**
 * Adds a UI button that starts the audio engine's underlying audio context when the user presses it.
 * @internal
 */
class _WebAudioUnmuteUI {
    /** @internal */
    constructor(engine, parentElement) {
        this._button = null;
        this._enabled = true;
        this._style = null;
        this._onStateChanged = () => {
            if (!this._button) {
                return;
            }
            if (this._engine.state === "running") {
                this._hide();
            }
            else {
                this._show();
            }
        };
        this._engine = engine;
        const parent = parentElement || EngineStore.LastCreatedEngine?.getInputElement()?.parentElement || document.body;
        const top = (parent?.offsetTop || 0) + 20;
        this._style = document.createElement("style");
        this._style.appendChild(document.createTextNode(`.babylonUnmute{position:absolute;top:${top}px;margin-left:20px;height:40px;width:60px;background-color:rgba(51,51,51,0.7);background-image:url("data:image/svg+xml;charset=UTF-8,%3Csvg%20version%3D%221.1%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2239%22%20height%3D%2232%22%20viewBox%3D%220%200%2039%2032%22%3E%3Cpath%20fill%3D%22white%22%20d%3D%22M9.625%2018.938l-0.031%200.016h-4.953q-0.016%200-0.031-0.016v-12.453q0-0.016%200.031-0.016h4.953q0.031%200%200.031%200.016v12.453zM12.125%207.688l8.719-8.703v27.453l-8.719-8.719-0.016-0.047v-9.938zM23.359%207.875l1.406-1.406%204.219%204.203%204.203-4.203%201.422%201.406-4.219%204.219%204.219%204.203-1.484%201.359-4.141-4.156-4.219%204.219-1.406-1.422%204.219-4.203z%22%3E%3C%2Fpath%3E%3C%2Fsvg%3E");background-size:80%;background-repeat:no-repeat;background-position:center;background-position-y:4px;border:none;outline:none;transition:transform 0.125s ease-out;cursor:pointer;z-index:9999;}.babylonUnmute:hover{transform:scale(1.05)}`));
        document.head.appendChild(this._style);
        this._button = document.createElement("button");
        this._button.className = "babylonUnmute";
        this._button.id = "babylonUnmuteButton";
        this._button.addEventListener("click", () => {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this._engine.unlockAsync();
        });
        parent.appendChild(this._button);
        this._engine.stateChangedObservable.add(this._onStateChanged);
    }
    /** @internal */
    dispose() {
        this._button?.remove();
        this._button = null;
        this._style?.remove();
        this._style = null;
        this._engine.stateChangedObservable.removeCallback(this._onStateChanged);
    }
    /** @internal */
    get enabled() {
        return this._enabled;
    }
    set enabled(value) {
        this._enabled = value;
        if (value) {
            if (this._engine.state !== "running") {
                this._show();
            }
        }
        else {
            this._hide();
        }
    }
    _show() {
        if (!this._button || !this._enabled) {
            return;
        }
        this._button.style.display = "block";
    }
    _hide() {
        if (!this._button) {
            return;
        }
        this._button.style.display = "none";
    }
}

const FormatMimeTypes = {
    aac: "audio/aac",
    ac3: "audio/ac3",
    flac: "audio/flac",
    m4a: "audio/mp4",
    mp3: 'audio/mpeg; codecs="mp3"',
    mp4: "audio/mp4",
    ogg: 'audio/ogg; codecs="vorbis"',
    wav: "audio/wav",
    webm: 'audio/webm; codecs="vorbis"',
};
/** @internal */
class _WebAudioEngine extends AudioEngineV2 {
    /** @internal */
    constructor(options = {}) {
        super(options);
        this._audioContextStarted = false;
        this._destinationNode = null;
        this._invalidFormats = new Set();
        this._isUpdating = false;
        this._listener = null;
        this._listenerAutoUpdate = true;
        this._listenerMinUpdateTime = 0;
        this._pauseCalled = false;
        this._resumeOnInteraction = true;
        this._resumeOnPause = true;
        this._resumeOnPauseRetryInterval = 1000;
        this._resumeOnPauseTimerId = null;
        this._resumePromise = null;
        this._silentHtmlAudio = null;
        this._unmuteUI = null;
        this._updateObservable = null;
        this._validFormats = new Set();
        this._volume = 1;
        /** @internal */
        this._isUsingOfflineAudioContext = false;
        /** @internal */
        this.isReadyPromise = new Promise((resolve) => {
            this._resolveIsReadyPromise = resolve;
        });
        /** @internal */
        this.stateChangedObservable = new Observable();
        /** @internal */
        this.userGestureObservable = new Observable();
        this._initAudioContextAsync = async () => {
            this._audioContext.addEventListener("statechange", this._onAudioContextStateChange);
            this._mainOut = new _WebAudioMainOut(this);
            this._mainOut.volume = this._volume;
            await this.createMainBusAsync("default");
        };
        this._onAudioContextStateChange = () => {
            if (this.state === "running") {
                clearInterval(this._resumeOnPauseTimerId);
                this._audioContextStarted = true;
                this._resumePromise = null;
            }
            if (this.state === "suspended" || this.state === "interrupted") {
                if (this._audioContextStarted && this._resumeOnPause && !this._pauseCalled) {
                    clearInterval(this._resumeOnPauseTimerId);
                    this._resumeOnPauseTimerId = setInterval(() => {
                        // eslint-disable-next-line @typescript-eslint/no-floating-promises
                        this.resumeAsync();
                    }, this._resumeOnPauseRetryInterval);
                }
            }
            this.stateChangedObservable.notifyObservers(this.state);
        };
        this._onUserGestureAsync = async () => {
            if (this._resumeOnInteraction) {
                await this._audioContext.resume();
            }
            // On iOS the ringer switch must be turned on for WebAudio to play.
            // This gets WebAudio to play with the ringer switch turned off by playing an HTMLAudioElement.
            if (!this._silentHtmlAudio) {
                this._silentHtmlAudio = document.createElement("audio");
                const audio = this._silentHtmlAudio;
                audio.controls = false;
                audio.preload = "auto";
                audio.loop = true;
                // Wave data for 0.0001 seconds of silence.
                audio.src = "data:audio/wav;base64,UklGRjAAAABXQVZFZm10IBAAAAABAAEAgLsAAAB3AQACABAAZGF0YQwAAAAAAAEA/v8CAP//AQA=";
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                audio.play();
            }
            this.userGestureObservable.notifyObservers();
        };
        this._startUpdating = () => {
            if (this._isUpdating) {
                return;
            }
            this._isUpdating = true;
            if (this.state === "running") {
                this._update();
            }
            else {
                const callback = () => {
                    if (this.state === "running") {
                        this._update();
                        this.stateChangedObservable.removeCallback(callback);
                    }
                };
                this.stateChangedObservable.add(callback);
            }
        };
        this._update = () => {
            if (this._updateObservable?.hasObservers()) {
                this._updateObservable.notifyObservers();
                requestAnimationFrame(this._update);
            }
            else {
                this._isUpdating = false;
            }
        };
        if (typeof options.listenerAutoUpdate === "boolean") {
            this._listenerAutoUpdate = options.listenerAutoUpdate;
        }
        if (typeof options.listenerMinUpdateTime === "number") {
            this._listenerMinUpdateTime = options.listenerMinUpdateTime;
        }
        this._volume = options.volume ?? 1;
        if (options.audioContext) {
            this._isUsingOfflineAudioContext = options.audioContext instanceof OfflineAudioContext;
            this._audioContext = options.audioContext;
        }
        else {
            this._audioContext = new AudioContext();
        }
        if (!options.disableDefaultUI) {
            this._unmuteUI = new _WebAudioUnmuteUI(this, options.defaultUIParentElement);
        }
    }
    /** @internal */
    async _initAsync(options) {
        this._resumeOnInteraction = typeof options.resumeOnInteraction === "boolean" ? options.resumeOnInteraction : true;
        this._resumeOnPause = typeof options.resumeOnPause === "boolean" ? options.resumeOnPause : true;
        this._resumeOnPauseRetryInterval = options.resumeOnPauseRetryInterval ?? 1000;
        document.addEventListener("click", this._onUserGestureAsync);
        await this._initAudioContextAsync();
        if (_HasSpatialAudioListenerOptions(options)) {
            this._listener = _CreateSpatialAudioListener(this, this._listenerAutoUpdate, this._listenerMinUpdateTime);
            this._listener.setOptions(options);
        }
        this._resolveIsReadyPromise();
    }
    /** @internal */
    get currentTime() {
        return this._audioContext.currentTime ?? 0;
    }
    /** @internal */
    get _inNode() {
        return this._audioContext.destination;
    }
    /** @internal */
    get mainOut() {
        return this._mainOut;
    }
    /** @internal */
    get listener() {
        return this._listener ?? (this._listener = _CreateSpatialAudioListener(this, this._listenerAutoUpdate, this._listenerMinUpdateTime));
    }
    /** @internal */
    get state() {
        // Always return "running" for OfflineAudioContext so sound `play` calls work while the context is suspended.
        return this._isUsingOfflineAudioContext ? "running" : this._audioContext.state;
    }
    /** @internal */
    get volume() {
        return this._volume;
    }
    /** @internal */
    set volume(value) {
        if (this._volume === value) {
            return;
        }
        this._volume = value;
        if (this._mainOut) {
            this._mainOut.volume = value;
        }
    }
    /**
     * This property should only be used by the legacy audio engine.
     * @internal
     * */
    get _audioDestination() {
        return this._destinationNode ? this._destinationNode : (this._destinationNode = this._audioContext.destination);
    }
    set _audioDestination(value) {
        this._destinationNode = value;
    }
    /**
     * This property should only be used by the legacy audio engine.
     * @internal
     */
    get _unmuteUIEnabled() {
        return this._unmuteUI ? this._unmuteUI.enabled : false;
    }
    set _unmuteUIEnabled(value) {
        if (this._unmuteUI) {
            this._unmuteUI.enabled = value;
        }
    }
    /** @internal */
    async createBusAsync(name, options = {}) {
        const module = await __vitePreload(() => import('./webAudioBus-BnckzWcN.js'),true              ?__vite__mapDeps([0,1,2,3,4,5,6]):void 0,import.meta.url);
        const bus = new module._WebAudioBus(name, this, options);
        await bus._initAsync(options);
        return bus;
    }
    /** @internal */
    async createMainBusAsync(name, options = {}) {
        const module = await __vitePreload(() => import('./webAudioMainBus-Dclg8YIq.js'),true              ?__vite__mapDeps([7,1,2,3,4,5]):void 0,import.meta.url);
        const bus = new module._WebAudioMainBus(name, this);
        await bus._initAsync(options);
        return bus;
    }
    /** @internal */
    async createMicrophoneSoundSourceAsync(name, options) {
        let mediaStream;
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        catch (e) {
            throw new Error("Unable to access microphone: " + e);
        }
        return await this.createSoundSourceAsync(name, new MediaStreamAudioSourceNode(this._audioContext, { mediaStream }), {
            outBusAutoDefault: false,
            ...options,
        });
    }
    /** @internal */
    async createSoundAsync(name, source, options = {}) {
        const module = await __vitePreload(() => import('./webAudioStaticSound-D8ZwU_iO.js'),true              ?__vite__mapDeps([8,9,3,4,5,10,2,6]):void 0,import.meta.url);
        const sound = new module._WebAudioStaticSound(name, this, options);
        await sound._initAsync(source, options);
        return sound;
    }
    /** @internal */
    async createSoundBufferAsync(source, options = {}) {
        const module = await __vitePreload(() => import('./webAudioStaticSound-D8ZwU_iO.js'),true              ?__vite__mapDeps([8,9,3,4,5,10,2,6]):void 0,import.meta.url);
        const soundBuffer = new module._WebAudioStaticSoundBuffer(this);
        await soundBuffer._initAsync(source, options);
        return soundBuffer;
    }
    /** @internal */
    async createSoundSourceAsync(name, source, options = {}) {
        const module = await __vitePreload(() => import('./webAudioSoundSource-__1CMM9A.js'),true              ?__vite__mapDeps([11,10,2,3,4,5,6]):void 0,import.meta.url);
        const soundSource = new module._WebAudioSoundSource(name, source, this, options);
        await soundSource._initAsync(options);
        return soundSource;
    }
    /** @internal */
    async createStreamingSoundAsync(name, source, options = {}) {
        const module = await __vitePreload(() => import('./webAudioStreamingSound-CvXX6uFt.js'),true              ?__vite__mapDeps([12,3,4,5,9,10,2,6]):void 0,import.meta.url);
        const sound = new module._WebAudioStreamingSound(name, this, options);
        await sound._initAsync(source, options);
        return sound;
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._listener?.dispose();
        this._listener = null;
        // Note that OfflineAudioContext does not have a `close` method.
        if (this._audioContext.state !== "closed" && !this._isUsingOfflineAudioContext) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this._audioContext.close();
        }
        document.removeEventListener("click", this._onUserGestureAsync);
        this._audioContext.removeEventListener("statechange", this._onAudioContextStateChange);
        this._silentHtmlAudio?.remove();
        this._updateObservable?.clear();
        this._updateObservable = null;
        this._unmuteUI?.dispose();
        this._unmuteUI = null;
        this.stateChangedObservable.clear();
    }
    /** @internal */
    flagInvalidFormat(format) {
        this._invalidFormats.add(format);
    }
    /** @internal */
    isFormatValid(format) {
        if (this._validFormats.has(format)) {
            return true;
        }
        if (this._invalidFormats.has(format)) {
            return false;
        }
        const mimeType = FormatMimeTypes[format];
        if (mimeType === undefined) {
            return false;
        }
        const audio = new Audio();
        if (audio.canPlayType(mimeType) === "") {
            this._invalidFormats.add(format);
            return false;
        }
        this._validFormats.add(format);
        return true;
    }
    /** @internal */
    async pauseAsync() {
        await this._audioContext.suspend();
        this._pauseCalled = true;
    }
    /** @internal */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    resumeAsync() {
        this._pauseCalled = false;
        if (this._resumePromise) {
            return this._resumePromise;
        }
        this._resumePromise = this._audioContext.resume();
        this.stateChangedObservable.notifyObservers(this.state);
        return this._resumePromise;
    }
    /** @internal */
    setVolume(value, options = null) {
        if (this._mainOut) {
            this._mainOut.setVolume(value, options);
        }
        else {
            throw new Error("Main output not initialized yet.");
        }
    }
    /** @internal */
    _addMainBus(mainBus) {
        super._addMainBus(mainBus);
    }
    /** @internal */
    _removeMainBus(mainBus) {
        super._removeMainBus(mainBus);
    }
    /** @internal */
    _addNode(node) {
        super._addNode(node);
    }
    /** @internal */
    _removeNode(node) {
        super._removeNode(node);
    }
    /** @internal */
    _addSound(sound) {
        super._addSound(sound);
    }
    /** @internal */
    _removeSound(sound) {
        super._removeSound(sound);
    }
    /** @internal */
    _addUpdateObserver(callback) {
        if (!this._updateObservable) {
            this._updateObservable = new Observable();
        }
        this._updateObservable.add(callback);
        this._startUpdating();
    }
    _removeUpdateObserver(callback) {
        if (this._updateObservable) {
            this._updateObservable.removeCallback(callback);
        }
    }
}

// Sets the default audio engine to Babylon.js
AbstractEngine.AudioEngineFactory = (hostElement, audioContext, audioDestination) => {
    return new AudioEngine(hostElement, audioContext, audioDestination);
};
/**
 * This represents the default audio engine used in babylon.
 * It is responsible to play, synchronize and analyse sounds throughout the  application.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic
 */
class AudioEngine {
    /**
     * The master gain node defines the global audio volume of your audio engine.
     */
    get masterGain() {
        return this._masterGain;
    }
    set masterGain(value) {
        this._masterGain = this._v2.mainOut._inNode = value;
    }
    /**
     * Defines if the audio engine relies on a custom unlocked button.
     * In this case, the embedded button will not be displayed.
     */
    get useCustomUnlockedButton() {
        return this._useCustomUnlockedButton;
    }
    set useCustomUnlockedButton(value) {
        this._useCustomUnlockedButton = value;
        this._v2._unmuteUIEnabled = !value;
    }
    /**
     * Gets the current AudioContext if available.
     */
    get audioContext() {
        if (this._v2.state === "running") {
            // Do not wait for the promise to unlock.
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this._triggerRunningStateAsync();
        }
        return this._v2._audioContext;
    }
    /**
     * Instantiates a new audio engine.
     *
     * @param hostElement defines the host element where to display the mute icon if necessary
     * @param audioContext defines the audio context to be used by the audio engine
     * @param audioDestination defines the audio destination node to be used by audio engine
     */
    constructor(hostElement = null, audioContext = null, audioDestination = null) {
        this._tryToRun = false;
        this._useCustomUnlockedButton = false;
        /**
         * Gets whether the current host supports Web Audio and thus could create AudioContexts.
         */
        this.canUseWebAudio = true;
        /**
         * Defines if Babylon should emit a warning if WebAudio is not supported.
         */
        // eslint-disable-next-line @typescript-eslint/naming-convention
        this.WarnedWebAudioUnsupported = false;
        /**
         * Gets whether or not mp3 are supported by your browser.
         */
        this.isMP3supported = false;
        /**
         * Gets whether or not ogg are supported by your browser.
         */
        this.isOGGsupported = false;
        /**
         * Gets whether audio has been unlocked on the device.
         * Some Browsers have strong restrictions about Audio and won't autoplay unless
         * a user interaction has happened.
         */
        this.unlocked = false;
        /**
         * Event raised when audio has been unlocked on the browser.
         */
        this.onAudioUnlockedObservable = new Observable();
        /**
         * Event raised when audio has been locked on the browser.
         */
        this.onAudioLockedObservable = new Observable();
        const v2 = new _WebAudioEngine({
            audioContext: audioContext ? audioContext : undefined,
            defaultUIParentElement: hostElement?.parentElement ? hostElement.parentElement : undefined,
        });
        // Historically the unmute button is disabled until a sound tries to play and can't, which results in a call
        // to `AudioEngine.lock()`, which is where the unmute button is enabled if no custom UI is requested.
        v2._unmuteUIEnabled = false;
        this._masterGain = new GainNode(v2._audioContext);
        v2._audioDestination = audioDestination;
        v2.stateChangedObservable.add((state) => {
            if (state === "running") {
                this.unlocked = true;
                this.onAudioUnlockedObservable.notifyObservers(this);
            }
            else {
                this.unlocked = false;
                this.onAudioLockedObservable.notifyObservers(this);
            }
        });
        // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
        v2._initAsync({ resumeOnInteraction: false }).then(() => {
            const mainBusOutNode = v2.defaultMainBus._outNode;
            if (mainBusOutNode) {
                mainBusOutNode.disconnect(v2.mainOut._inNode);
                mainBusOutNode.connect(this._masterGain);
            }
            v2.mainOut._inNode = this._masterGain;
            v2.stateChangedObservable.notifyObservers(v2.state);
        });
        this.isMP3supported = v2.isFormatValid("mp3");
        this.isOGGsupported = v2.isFormatValid("ogg");
        this._v2 = v2;
    }
    /**
     * Flags the audio engine in Locked state.
     * This happens due to new browser policies preventing audio to autoplay.
     */
    lock() {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._v2._audioContext.suspend();
        if (!this._useCustomUnlockedButton) {
            this._v2._unmuteUIEnabled = true;
        }
    }
    /**
     * Unlocks the audio engine once a user action has been done on the dom.
     * This is helpful to resume play once browser policies have been satisfied.
     */
    unlock() {
        if (this._v2._audioContext?.state === "running") {
            if (!this.unlocked) {
                // Notify users that the audio stack is unlocked/unmuted
                this.unlocked = true;
                this.onAudioUnlockedObservable.notifyObservers(this);
            }
            return;
        }
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._triggerRunningStateAsync();
    }
    /** @internal */
    _resumeAudioContextOnStateChange() {
        this._v2._audioContext?.addEventListener("statechange", () => {
            if (this.unlocked && this._v2._audioContext?.state !== "running") {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                this._resumeAudioContextAsync();
            }
        }, {
            once: true,
            passive: true,
            signal: AbortSignal.timeout(3000),
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _resumeAudioContextAsync() {
        if (this._v2._isUsingOfflineAudioContext) {
            return Promise.resolve();
        }
        if (this._v2._audioContext.state === "suspended" && !this._useCustomUnlockedButton) {
            this._v2._unmuteUIEnabled = true;
        }
        return this._v2._audioContext.resume();
    }
    /**
     * Destroy and release the resources associated with the audio context.
     */
    dispose() {
        this._v2.dispose();
        this.onAudioUnlockedObservable.clear();
        this.onAudioLockedObservable.clear();
    }
    /**
     * Gets the global volume sets on the master gain.
     * @returns the global volume if set or -1 otherwise
     */
    getGlobalVolume() {
        return this.masterGain.gain.value;
    }
    /**
     * Sets the global volume of your experience (sets on the master gain).
     * @param newVolume Defines the new global volume of the application
     */
    setGlobalVolume(newVolume) {
        this.masterGain.gain.value = newVolume;
    }
    /**
     * Connect the audio engine to an audio analyser allowing some amazing
     * synchronization between the sounds/music and your visualization (VuMeter for instance).
     * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#using-the-analyser
     * @param analyser The analyser to connect to the engine
     */
    connectToAnalyser(analyser) {
        if (this._connectedAnalyser) {
            this._connectedAnalyser.stopDebugCanvas();
        }
        this._connectedAnalyser = analyser;
        this.masterGain.disconnect();
        this._connectedAnalyser.connectAudioNodes(this.masterGain, this._v2._audioContext.destination);
    }
    async _triggerRunningStateAsync() {
        if (this._tryToRun) {
            void this._v2._audioContext.resume();
            return;
        }
        this._tryToRun = true;
        await this._resumeAudioContextAsync();
        this._tryToRun = false;
        this.unlocked = true;
        this.onAudioUnlockedObservable.notifyObservers(this);
    }
}

export { AbstractAudioNode as A, _CleanUrl as _, _FileExtensionRegex as a, _WebAudioParameterComponent as b, _SpatialAudioAttacherComponent as c, _SpatialWebAudioUpdaterComponent as d, AbstractNamedAudioNode as e };
//# sourceMappingURL=audioEngine-CXlZUK-6.js.map
