import { e as AbstractNamedAudioNode, b as _WebAudioParameterComponent } from './audioEngine-CXlZUK-6.js';
import { L as Logger } from './index-CxR0Yf_7.js';

/** @internal */
class _AbstractAudioSubNode extends AbstractNamedAudioNode {
    /** @internal */
    constructor(name, engine) {
        super(name, engine, 3 /* AudioNodeType.HAS_INPUTS_AND_OUTPUTS */);
    }
    /** @internal */
    connect(node) {
        if (!this._connect(node)) {
            throw new Error("Connect failed");
        }
    }
    /** @internal */
    disconnect(node) {
        if (!this._disconnect(node)) {
            throw new Error("Disconnect failed");
        }
    }
    /** @internal */
    disconnectAll() {
        if (!this._downstreamNodes) {
            throw new Error("Disconnect failed");
        }
        const it = this._downstreamNodes.values();
        for (let next = it.next(); !next.done; next = it.next()) {
            if (!this._disconnect(next.value)) {
                throw new Error("Disconnect failed");
            }
        }
    }
}

/** @internal */
const _VolumeAudioDefaults = {
    volume: 1,
};
/** @internal */
class _VolumeAudioSubNode extends _AbstractAudioSubNode {
    constructor(engine) {
        super("Volume" /* AudioSubNode.VOLUME */, engine);
    }
    /** @internal */
    setOptions(options) {
        this.setVolume(options.volume ?? _VolumeAudioDefaults.volume, { shape: "none" /* AudioParameterRampShape.None */ });
    }
}
/** @internal */
function _GetVolumeAudioSubNode(subGraph) {
    return subGraph.getSubNode("Volume" /* AudioSubNode.VOLUME */);
}
/** @internal */
function _GetVolumeAudioProperty(subGraph, property) {
    return _GetVolumeAudioSubNode(subGraph)?.[property] ?? _VolumeAudioDefaults[property];
}

const _AudioAnalyzerDefaults = {
    fftSize: 2048,
    minDecibels: -100,
    maxDecibels: -30,
    smoothing: 0.8,
};
/**
 * @param options The audio analyzer options to check.
 * @returns `true` if audio analyzer options are defined, otherwise `false`.
 */
function _HasAudioAnalyzerOptions(options) {
    return (options.analyzerEnabled ||
        options.analyzerFFTSize !== undefined ||
        options.analyzerMinDecibels !== undefined ||
        options.analyzerMaxDecibels !== undefined ||
        options.analyzerSmoothing !== undefined);
}
/**
 * An AudioAnalyzer converts time-domain audio data into the frequency-domain.
 */
class AbstractAudioAnalyzer {
    /**
     * The number of data values that will be returned when calling getByteFrequencyData() or getFloatFrequencyData(). This is always half the `fftSize`.
     */
    get frequencyBinCount() {
        return this.fftSize / 2;
    }
}

/** @internal */
class _AudioAnalyzerSubNode extends _AbstractAudioSubNode {
    constructor(engine) {
        super("Analyzer" /* AudioSubNode.ANALYZER */, engine);
    }
    /** @internal */
    setOptions(options) {
        this.fftSize = options.analyzerFFTSize ?? _AudioAnalyzerDefaults.fftSize;
        this.minDecibels = options.analyzerMinDecibels ?? _AudioAnalyzerDefaults.minDecibels;
        this.maxDecibels = options.analyzerMaxDecibels ?? _AudioAnalyzerDefaults.maxDecibels;
        this.smoothing = options.analyzerSmoothing ?? _AudioAnalyzerDefaults.smoothing;
    }
}
/** @internal */
function _GetAudioAnalyzerSubNode(subGraph) {
    return subGraph.getSubNode("Analyzer" /* AudioSubNode.ANALYZER */);
}
/** @internal */
function _SetAudioAnalyzerProperty(subGraph, property, value) {
    subGraph.callOnSubNode("Analyzer" /* AudioSubNode.ANALYZER */, (node) => {
        node[property] = value;
    });
}

let EmptyByteFrequencyData = null;
let EmptyFloatFrequencyData = null;
/** @internal */
function _GetEmptyByteFrequencyData() {
    if (!EmptyByteFrequencyData) {
        EmptyByteFrequencyData = new Uint8Array();
    }
    return EmptyByteFrequencyData;
}
/** @internal */
function _GetEmptyFloatFrequencyData() {
    if (!EmptyFloatFrequencyData) {
        EmptyFloatFrequencyData = new Float32Array();
    }
    return EmptyFloatFrequencyData;
}
/** @internal */
class _AudioAnalyzer extends AbstractAudioAnalyzer {
    /** @internal */
    constructor(subGraph) {
        super();
        this._fftSize = _AudioAnalyzerDefaults.fftSize;
        this._maxDecibels = _AudioAnalyzerDefaults.maxDecibels;
        this._minDecibels = _AudioAnalyzerDefaults.minDecibels;
        this._smoothing = _AudioAnalyzerDefaults.smoothing;
        this._subGraph = subGraph;
    }
    /** @internal */
    get fftSize() {
        return this._fftSize;
    }
    set fftSize(value) {
        this._fftSize = value;
        _SetAudioAnalyzerProperty(this._subGraph, "fftSize", value);
    }
    /** @internal */
    get isEnabled() {
        return _GetAudioAnalyzerSubNode(this._subGraph) !== null;
    }
    /** @internal */
    get minDecibels() {
        return this._minDecibels;
    }
    set minDecibels(value) {
        this._minDecibels = value;
        _SetAudioAnalyzerProperty(this._subGraph, "minDecibels", value);
    }
    /** @internal */
    get maxDecibels() {
        return this._maxDecibels;
    }
    set maxDecibels(value) {
        this._maxDecibels = value;
        _SetAudioAnalyzerProperty(this._subGraph, "maxDecibels", value);
    }
    /** @internal */
    get smoothing() {
        return this._smoothing;
    }
    set smoothing(value) {
        this._smoothing = value;
        _SetAudioAnalyzerProperty(this._subGraph, "smoothing", value);
    }
    /** @internal */
    dispose() {
        const subNode = _GetAudioAnalyzerSubNode(this._subGraph);
        if (subNode) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this._subGraph.removeSubNodeAsync(subNode);
            subNode.dispose();
        }
    }
    /** @internal */
    async enableAsync() {
        const subNode = _GetAudioAnalyzerSubNode(this._subGraph);
        if (!subNode) {
            await this._subGraph.createAndAddSubNodeAsync("Analyzer" /* AudioSubNode.ANALYZER */);
        }
    }
    /** @internal */
    getByteFrequencyData() {
        const subNode = _GetAudioAnalyzerSubNode(this._subGraph);
        if (!subNode) {
            Logger.Warn("AudioAnalyzer not enabled");
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this.enableAsync();
            return _GetEmptyByteFrequencyData();
        }
        return subNode.getByteFrequencyData();
    }
    /** @internal */
    getFloatFrequencyData() {
        const subNode = _GetAudioAnalyzerSubNode(this._subGraph);
        if (!subNode) {
            Logger.Warn("AudioAnalyzer not enabled");
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            this.enableAsync();
            return _GetEmptyFloatFrequencyData();
        }
        return subNode.getFloatFrequencyData();
    }
}

/**
 * Abstract class representing and audio output node with an analyzer and volume control.
 */
class AbstractAudioOutNode extends AbstractNamedAudioNode {
    constructor(name, engine, nodeType) {
        super(name, engine, nodeType);
        this._analyzer = null;
    }
    /**
     * The audio analyzer features.
     */
    get analyzer() {
        return this._analyzer ?? (this._analyzer = new _AudioAnalyzer(this._subGraph));
    }
    /**
     * The audio output volume.
     */
    get volume() {
        return _GetVolumeAudioProperty(this._subGraph, "volume");
    }
    set volume(value) {
        // The volume subnode is created on initialization and should always exist.
        const node = _GetVolumeAudioSubNode(this._subGraph);
        if (!node) {
            throw new Error("No volume subnode");
        }
        node.volume = value;
    }
    /**
     * Releases associated resources.
     */
    dispose() {
        super.dispose();
        this._analyzer?.dispose();
        this._analyzer = null;
        this._subGraph.dispose();
    }
    /**
     * Sets the audio output volume with optional ramping.
     * If the duration is 0 then the volume is set immediately, otherwise it is ramped to the new value over the given duration using the given shape.
     * If a ramp is already in progress then the volume is not set and an error is thrown.
     * @param value The value to set the volume to.
     * @param options The options to use for ramping the volume change.
     */
    setVolume(value, options = null) {
        const node = _GetVolumeAudioSubNode(this._subGraph);
        if (!node) {
            throw new Error("No volume subnode");
        }
        node.setVolume(value, options);
    }
}

/**
 * Adds common sub graph functionality to an audio node.
 *
 * Audio nodes such as static sounds, streaming sounds, and buses can use audio sub graphs to process audio internally
 * before sending it to connected downstream audio nodes. This is useful for applying effects, spatial audio, and other
 * audio processing tasks common to multiple audio node classes.
 *
 * A key feature of audio sub graphs is their audio sub nodes are created asynchronously on demand so the minimum set
 * of sub nodes are used at all times to save memory and CPU resources. The tradeoff is a small delay when first
 * setting a property backed by a sub node. This delay is avoided by using the appropriate options to initialize the
 * sub node on creation, e.g. `spatialEnabled` and `stereoEnabled`, or by setting any creation option backed by the
 * sub node, e.g. `spatialPosition` and `stereoPan`.
 *
 * @internal
 */
class _AbstractAudioSubGraph {
    constructor() {
        this._createSubNodePromises = {};
        this._isDisposed = false;
        this._subNodes = {};
        this._onSubNodeDisposed = (node) => {
            const subNode = node;
            delete this._subNodes[subNode.name];
            this._onSubNodesChanged();
        };
    }
    /**
     * Executes the given callback with the named sub node, creating the sub node if needed.
     *
     * @param name The name of the sub node
     * @param callback The function to call with the named sub node
     *
     * @internal
     */
    callOnSubNode(name, callback) {
        const node = this.getSubNode(name);
        if (node) {
            callback(node);
            return;
        }
        // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
        this._createSubNodePromisesResolvedAsync().then(() => {
            const node = this.getSubNode(name);
            if (node) {
                callback(node);
                return;
            }
            // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
            this.createAndAddSubNodeAsync(name).then((node) => {
                callback(node);
            });
        });
    }
    /**
     * Creates the named subnode and adds it to the sub graph.
     *
     * @param name The name of the sub node.
     * @returns A promise that resolves to the created sub node.
     *
     * @internal
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    createAndAddSubNodeAsync(name) {
        var _a;
        // eslint-disable-next-line github/no-then
        (_a = this._createSubNodePromises)[name] || (_a[name] = this._createSubNode(name).then((node) => {
            this._addSubNode(node);
            return node;
        }));
        return this._createSubNodePromises[name];
    }
    /**
     * Releases associated resources.
     *
     * @internal
     */
    dispose() {
        this._isDisposed = true;
        const subNodes = Object.values(this._subNodes);
        for (const subNode of subNodes) {
            subNode.dispose();
        }
        this._subNodes = {};
        this._createSubNodePromises = {};
    }
    /**
     * Gets a previously created sub node.
     *
     * @param name - The name of the sub node
     * @returns The named sub node, or `null` if it has not been created, yet
     *
     * @internal
     * */
    getSubNode(name) {
        return this._subNodes[name] ?? null;
    }
    /**
     * Removes a sub node from the sub graph.
     *
     * @param subNode - The sub node to remove
     * @returns A promise that resolves when the sub node is removed
     *
     * @internal
     */
    async removeSubNodeAsync(subNode) {
        if (!subNode) {
            return;
        }
        await this._createSubNodePromisesResolvedAsync();
        const name = subNode.name;
        if (this._subNodes[name]) {
            delete this._subNodes[name];
        }
        delete this._createSubNodePromises[name];
        this._onSubNodesChanged();
    }
    async _createSubNodePromisesResolvedAsync() {
        return await Promise.all(Object.values(this._createSubNodePromises));
    }
    _addSubNode(node) {
        if (this._isDisposed) {
            node.dispose();
            return;
        }
        this._subNodes[node.name] = node;
        node.onDisposeObservable.addOnce(this._onSubNodeDisposed);
        this._onSubNodesChanged();
    }
}

/** @internal */
// eslint-disable-next-line @typescript-eslint/require-await
async function _CreateVolumeAudioSubNodeAsync(engine) {
    return new _VolumeWebAudioSubNode(engine);
}
/** @internal */
class _VolumeWebAudioSubNode extends _VolumeAudioSubNode {
    /** @internal */
    constructor(engine) {
        super(engine);
        const gainNode = (this.node = new GainNode(engine._audioContext));
        this._volume = new _WebAudioParameterComponent(engine, gainNode.gain);
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._volume.dispose();
    }
    /** @internal */
    get volume() {
        return this._volume.value;
    }
    /** @internal */
    set volume(value) {
        this.setVolume(value);
    }
    /** @internal */
    get _inNode() {
        return this.node;
    }
    /** @internal */
    get _outNode() {
        return this.node;
    }
    /** @internal */
    setVolume(value, options = null) {
        this._volume.setTargetValue(value, options);
    }
    _connect(node) {
        const connected = super._connect(node);
        if (!connected) {
            return false;
        }
        // If the wrapped node is not available now, it will be connected later by the subgraph.
        if (node._inNode) {
            this.node.connect(node._inNode);
        }
        return true;
    }
    _disconnect(node) {
        const disconnected = super._disconnect(node);
        if (!disconnected) {
            return false;
        }
        if (node._inNode) {
            this.node.disconnect(node._inNode);
        }
        return true;
    }
    /** @internal */
    getClassName() {
        return "_VolumeWebAudioSubNode";
    }
}

/** @internal */
// eslint-disable-next-line @typescript-eslint/require-await
async function _CreateAudioAnalyzerSubNodeAsync(engine) {
    return new _WebAudioAnalyzerSubNode(engine);
}
/** @internal */
class _WebAudioAnalyzerSubNode extends _AudioAnalyzerSubNode {
    /** @internal */
    constructor(engine) {
        super(engine);
        this._byteFrequencyData = null;
        this._floatFrequencyData = null;
        this._analyzerNode = new AnalyserNode(engine._audioContext);
    }
    /** @internal */
    get fftSize() {
        return this._analyzerNode.fftSize;
    }
    set fftSize(value) {
        if (value === this._analyzerNode.fftSize) {
            return;
        }
        this._analyzerNode.fftSize = value;
        this._clearArrays();
    }
    /** @internal */
    get _inNode() {
        return this._analyzerNode;
    }
    /** @internal */
    get minDecibels() {
        return this._analyzerNode.minDecibels;
    }
    set minDecibels(value) {
        this._analyzerNode.minDecibels = value;
    }
    /** @internal */
    get maxDecibels() {
        return this._analyzerNode.maxDecibels;
    }
    set maxDecibels(value) {
        this._analyzerNode.maxDecibels = value;
    }
    /** @internal */
    get smoothing() {
        return this._analyzerNode.smoothingTimeConstant;
    }
    set smoothing(value) {
        this._analyzerNode.smoothingTimeConstant = value;
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._clearArrays();
        this._byteFrequencyData = null;
        this._floatFrequencyData = null;
        this._analyzerNode.disconnect();
    }
    /** @internal */
    getClassName() {
        return "_WebAudioAnalyzerSubNode";
    }
    /** @internal */
    getByteFrequencyData() {
        if (!this._byteFrequencyData || this._byteFrequencyData.length === 0) {
            this._byteFrequencyData = new Uint8Array(this._analyzerNode.frequencyBinCount);
        }
        this._analyzerNode.getByteFrequencyData(this._byteFrequencyData);
        return this._byteFrequencyData;
    }
    /** @internal */
    getFloatFrequencyData() {
        if (!this._floatFrequencyData || this._floatFrequencyData.length === 0) {
            this._floatFrequencyData = new Float32Array(this._analyzerNode.frequencyBinCount);
        }
        this._analyzerNode.getFloatFrequencyData(this._floatFrequencyData);
        return this._floatFrequencyData;
    }
    _clearArrays() {
        this._byteFrequencyData?.set(_GetEmptyByteFrequencyData());
        this._floatFrequencyData?.set(_GetEmptyFloatFrequencyData());
    }
}

/** @internal */
class _WebAudioBaseSubGraph extends _AbstractAudioSubGraph {
    /** @internal */
    constructor(owner) {
        super();
        this._outputNode = null;
        this._owner = owner;
    }
    /** @internal */
    async initAsync(options) {
        const hasAnalyzerOptions = _HasAudioAnalyzerOptions(options);
        if (hasAnalyzerOptions) {
            await this.createAndAddSubNodeAsync("Analyzer" /* AudioSubNode.ANALYZER */);
        }
        await this.createAndAddSubNodeAsync("Volume" /* AudioSubNode.VOLUME */);
        await this._createSubNodePromisesResolvedAsync();
        if (hasAnalyzerOptions) {
            const analyzerNode = _GetAudioAnalyzerSubNode(this);
            if (!analyzerNode) {
                throw new Error("No analyzer subnode.");
            }
            analyzerNode.setOptions(options);
        }
        const volumeNode = _GetVolumeAudioSubNode(this);
        if (!volumeNode) {
            throw new Error("No volume subnode.");
        }
        volumeNode.setOptions(options);
        if (volumeNode.getClassName() !== "_VolumeWebAudioSubNode") {
            throw new Error("Not a WebAudio subnode.");
        }
        this._outputNode = volumeNode.node;
        // Connect the new wrapped WebAudio node to the wrapped downstream WebAudio nodes.
        // The wrapper nodes are unaware of this change.
        if (this._outputNode && this._downstreamNodes) {
            const it = this._downstreamNodes.values();
            for (let next = it.next(); !next.done; next = it.next()) {
                const inNode = next.value._inNode;
                if (inNode) {
                    this._outputNode.connect(inNode);
                }
            }
        }
    }
    /** @internal */
    get _inNode() {
        return this._outputNode;
    }
    /** @internal */
    get _outNode() {
        return this._outputNode;
    }
    // Function is async, but throws synchronously. Avoiding breaking changes.
    // eslint-disable-next-line @typescript-eslint/promise-function-async
    _createSubNode(name) {
        switch (name) {
            case "Analyzer" /* AudioSubNode.ANALYZER */:
                return _CreateAudioAnalyzerSubNodeAsync(this._owner.engine);
            case "Volume" /* AudioSubNode.VOLUME */:
                return _CreateVolumeAudioSubNodeAsync(this._owner.engine);
            default:
                throw new Error(`Unknown subnode name: ${name}`);
        }
    }
    _onSubNodesChanged() {
        const analyzerNode = _GetAudioAnalyzerSubNode(this);
        const volumeNode = _GetVolumeAudioSubNode(this);
        if (analyzerNode && volumeNode) {
            volumeNode.connect(analyzerNode);
        }
    }
}

export { AbstractAudioOutNode as A, _WebAudioBaseSubGraph as _, _AbstractAudioSubNode as a, _GetVolumeAudioSubNode as b };
//# sourceMappingURL=webAudioBaseSubGraph-CuchFrSM.js.map
