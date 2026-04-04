import { a as _AbstractAudioSubNode, _ as _WebAudioBaseSubGraph, b as _GetVolumeAudioSubNode } from './webAudioBaseSubGraph-CuchFrSM.js';
import { ao as Quaternion, V as Vector3, M as Matrix } from './index-CxR0Yf_7.js';
import { c as _SpatialAudioAttacherComponent, b as _WebAudioParameterComponent, d as _SpatialWebAudioUpdaterComponent } from './audioEngine-CXlZUK-6.js';

const _SpatialAudioDefaults = {
    coneInnerAngle: 6.28318530718,
    coneOuterAngle: 6.28318530718,
    coneOuterVolume: 0,
    distanceModel: "linear",
    maxDistance: 10000,
    minDistance: 1,
    orientation: Vector3.Right(),
    panningModel: "equalpower",
    position: Vector3.Zero(),
    rolloffFactor: 1,
    rotation: Vector3.Zero(),
    rotationQuaternion: new Quaternion(),
};
/**
 * @param options The spatial audio options to check.
 * @returns `true` if spatial audio options are defined, otherwise `false`.
 */
function _HasSpatialAudioOptions(options) {
    return (options.spatialEnabled ||
        options.spatialAutoUpdate !== undefined ||
        options.spatialConeInnerAngle !== undefined ||
        options.spatialConeOuterAngle !== undefined ||
        options.spatialConeOuterVolume !== undefined ||
        options.spatialDistanceModel !== undefined ||
        options.spatialMaxDistance !== undefined ||
        options.spatialMinDistance !== undefined ||
        options.spatialMinUpdateTime !== undefined ||
        options.spatialOrientation !== undefined ||
        options.spatialPanningModel !== undefined ||
        options.spatialPosition !== undefined ||
        options.spatialRolloffFactor !== undefined ||
        options.spatialRotation !== undefined ||
        options.spatialRotationQuaternion !== undefined);
}
/**
 * Abstract class representing the `spatial` audio property on a sound or audio bus.
 *
 * @see {@link AudioEngineV2.listener}
 */
class AbstractSpatialAudio {
}

const _StereoAudioDefaults = {
    pan: 0,
};
/**
 * @param options The stereo audio options to check.
 * @returns `true` if stereo audio options are defined, otherwise `false`.
 */
function _HasStereoAudioOptions(options) {
    return options.stereoEnabled || options.stereoPan !== undefined;
}
/**
 * Abstract class representing the `stereo` audio property on a sound or audio bus.
 *
 * @see {@link AudioEngineV2.listener}
 */
class AbstractStereoAudio {
}

/** @internal */
class _StereoAudioSubNode extends _AbstractAudioSubNode {
    constructor(engine) {
        super("Stereo" /* AudioSubNode.STEREO */, engine);
    }
    /** @internal */
    setOptions(options) {
        this.pan = options.stereoPan ?? _StereoAudioDefaults.pan;
    }
}
/** @internal */
function _GetStereoAudioSubNode(subGraph) {
    return subGraph.getSubNode("Stereo" /* AudioSubNode.STEREO */);
}
/** @internal */
function _SetStereoAudioProperty(subGraph, property, value) {
    subGraph.callOnSubNode("Stereo" /* AudioSubNode.STEREO */, (node) => {
        node[property] = value;
    });
}

/** @internal */
class _StereoAudio extends AbstractStereoAudio {
    /** @internal */
    constructor(subGraph) {
        super();
        this._pan = _StereoAudioDefaults.pan;
        this._subGraph = subGraph;
    }
    /** @internal */
    get pan() {
        return this._pan;
    }
    set pan(value) {
        this._pan = value;
        _SetStereoAudioProperty(this._subGraph, "pan", value);
    }
}

/** @internal */
class _SpatialAudioSubNode extends _AbstractAudioSubNode {
    constructor(engine) {
        super("Spatial" /* AudioSubNode.SPATIAL */, engine);
        this._attacherComponent = null;
    }
    /** @internal */
    get isAttached() {
        return this._attacherComponent !== null && this._attacherComponent.isAttached;
    }
    /** @internal */
    attach(sceneNode, useBoundingBox, attachmentType) {
        this.detach();
        if (!this._attacherComponent) {
            this._attacherComponent = new _SpatialAudioAttacherComponent(this);
        }
        this._attacherComponent.attach(sceneNode, useBoundingBox, attachmentType);
    }
    /** @internal */
    detach() {
        this._attacherComponent?.detach();
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._attacherComponent?.dispose();
        this._attacherComponent = null;
    }
    /** @internal */
    setOptions(options) {
        this.coneInnerAngle = options.spatialConeInnerAngle ?? _SpatialAudioDefaults.coneInnerAngle;
        this.coneOuterAngle = options.spatialConeOuterAngle ?? _SpatialAudioDefaults.coneOuterAngle;
        this.coneOuterVolume = options.spatialConeOuterVolume ?? _SpatialAudioDefaults.coneOuterVolume;
        this.distanceModel = options.spatialDistanceModel ?? _SpatialAudioDefaults.distanceModel;
        this.maxDistance = options.spatialMaxDistance ?? _SpatialAudioDefaults.maxDistance;
        this.minDistance = options.spatialMinDistance ?? _SpatialAudioDefaults.minDistance;
        this.orientation = options.spatialOrientation ?? _SpatialAudioDefaults.orientation;
        this.panningModel = options.spatialPanningModel ?? _SpatialAudioDefaults.panningModel;
        this.rolloffFactor = options.spatialRolloffFactor ?? _SpatialAudioDefaults.rolloffFactor;
        if (options.spatialPosition) {
            this.position = options.spatialPosition.clone();
        }
        if (options.spatialRotationQuaternion) {
            this.rotationQuaternion = options.spatialRotationQuaternion.clone();
        }
        else if (options.spatialRotation) {
            this.rotation = options.spatialRotation.clone();
        }
        else {
            this.rotationQuaternion = _SpatialAudioDefaults.rotationQuaternion.clone();
        }
        this.update();
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
}
/** @internal */
function _GetSpatialAudioSubNode(subGraph) {
    return subGraph.getSubNode("Spatial" /* AudioSubNode.SPATIAL */);
}
/** @internal */
function _SetSpatialAudioProperty(subGraph, property, value) {
    subGraph.callOnSubNode("Spatial" /* AudioSubNode.SPATIAL */, (node) => {
        node[property] = value;
    });
}

const TmpMatrix = Matrix.Zero();
const TmpQuaternion = new Quaternion();
function D2r(degrees) {
    return (degrees * Math.PI) / 180;
}
function R2d(radians) {
    return (radians * 180) / Math.PI;
}
/** @internal */
// eslint-disable-next-line @typescript-eslint/require-await
async function _CreateSpatialAudioSubNodeAsync(engine) {
    return new _SpatialWebAudioSubNode(engine);
}
/** @internal */
class _SpatialWebAudioSubNode extends _SpatialAudioSubNode {
    /** @internal */
    constructor(engine) {
        super(engine);
        this._lastOrientation = Vector3.Zero();
        this._lastPosition = Vector3.Zero();
        this._lastRotation = Vector3.Zero();
        this._lastRotationQuaternion = new Quaternion();
        /** @internal */
        this.orientation = _SpatialAudioDefaults.orientation.clone();
        /** @internal */
        this.position = _SpatialAudioDefaults.position.clone();
        /** @internal */
        this.rotation = _SpatialAudioDefaults.rotation.clone();
        /** @internal */
        this.rotationQuaternion = _SpatialAudioDefaults.rotationQuaternion.clone();
        this.node = new PannerNode(engine._audioContext);
        this._orientationX = new _WebAudioParameterComponent(engine, this.node.orientationX);
        this._orientationY = new _WebAudioParameterComponent(engine, this.node.orientationY);
        this._orientationZ = new _WebAudioParameterComponent(engine, this.node.orientationZ);
        this._positionX = new _WebAudioParameterComponent(engine, this.node.positionX);
        this._positionY = new _WebAudioParameterComponent(engine, this.node.positionY);
        this._positionZ = new _WebAudioParameterComponent(engine, this.node.positionZ);
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._orientationX.dispose();
        this._orientationY.dispose();
        this._orientationZ.dispose();
        this._positionX.dispose();
        this._positionY.dispose();
        this._positionZ.dispose();
        this.node.disconnect();
    }
    /** @internal */
    get coneInnerAngle() {
        return D2r(this.node.coneInnerAngle);
    }
    set coneInnerAngle(value) {
        this.node.coneInnerAngle = R2d(value);
    }
    /** @internal */
    get coneOuterAngle() {
        return D2r(this.node.coneOuterAngle);
    }
    set coneOuterAngle(value) {
        this.node.coneOuterAngle = R2d(value);
    }
    /** @internal */
    get coneOuterVolume() {
        return this.node.coneOuterGain;
    }
    set coneOuterVolume(value) {
        this.node.coneOuterGain = value;
    }
    /** @internal */
    get distanceModel() {
        return this.node.distanceModel;
    }
    set distanceModel(value) {
        this.node.distanceModel = value;
        // Wiggle the max distance to make the change take effect.
        const maxDistance = this.node.maxDistance;
        this.node.maxDistance = maxDistance + 0.001;
        this.node.maxDistance = maxDistance;
    }
    /** @internal */
    get minDistance() {
        return this.node.refDistance;
    }
    set minDistance(value) {
        this.node.refDistance = value;
    }
    /** @internal */
    get maxDistance() {
        return this.node.maxDistance;
    }
    set maxDistance(value) {
        this.node.maxDistance = value;
    }
    /** @internal */
    get panningModel() {
        return this.node.panningModel;
    }
    set panningModel(value) {
        this.node.panningModel = value;
    }
    /** @internal */
    get rolloffFactor() {
        return this.node.rolloffFactor;
    }
    set rolloffFactor(value) {
        this.node.rolloffFactor = value;
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
    _updatePosition() {
        if (this._lastPosition.equalsWithEpsilon(this.position)) {
            return;
        }
        this._positionX.targetValue = this.position.x;
        this._positionY.targetValue = this.position.y;
        this._positionZ.targetValue = this.position.z;
        this._lastPosition.copyFrom(this.position);
    }
    /** @internal */
    _updateRotation() {
        let rotated = false;
        if (!this._lastRotationQuaternion.equalsWithEpsilon(this.rotationQuaternion)) {
            TmpQuaternion.copyFrom(this.rotationQuaternion);
            this._lastRotationQuaternion.copyFrom(this.rotationQuaternion);
            rotated = true;
        }
        else if (!this._lastRotation.equalsWithEpsilon(this.rotation)) {
            Quaternion.FromEulerAnglesToRef(this.rotation.x, this.rotation.y, this.rotation.z, TmpQuaternion);
            this._lastRotation.copyFrom(this.rotation);
            rotated = true;
        }
        else if (this._lastOrientation.equalsWithEpsilon(this.orientation)) {
            return;
        }
        if (rotated) {
            Matrix.FromQuaternionToRef(TmpQuaternion, TmpMatrix);
            Vector3.TransformNormalToRef(Vector3.RightReadOnly, TmpMatrix, this.orientation);
        }
        this._orientationX.targetValue = this.orientation.x;
        this._orientationY.targetValue = this.orientation.y;
        this._orientationZ.targetValue = this.orientation.z;
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
        return "_SpatialWebAudioSubNode";
    }
}

/** @internal */
// eslint-disable-next-line @typescript-eslint/require-await
async function _CreateStereoAudioSubNodeAsync(engine) {
    return new _StereoWebAudioSubNode(engine);
}
/** @internal */
class _StereoWebAudioSubNode extends _StereoAudioSubNode {
    /** @internal */
    constructor(engine) {
        super(engine);
        this.node = new StereoPannerNode(engine._audioContext);
        this._pan = new _WebAudioParameterComponent(engine, this.node.pan);
    }
    /** @internal */
    dispose() {
        super.dispose();
        this._pan.dispose();
    }
    /** @internal */
    get pan() {
        return this._pan.targetValue;
    }
    /** @internal */
    set pan(value) {
        this._pan.targetValue = value;
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
    getClassName() {
        return "_StereoWebAudioSubNode";
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
}

/** @internal */
class _WebAudioBusAndSoundSubGraph extends _WebAudioBaseSubGraph {
    constructor() {
        super(...arguments);
        this._rootNode = null;
        this._inputNode = null;
    }
    /** @internal */
    async initAsync(options) {
        await super.initAsync(options);
        let hasSpatialOptions = false;
        let hasStereoOptions = false;
        if ((hasSpatialOptions = _HasSpatialAudioOptions(options))) {
            await this.createAndAddSubNodeAsync("Spatial" /* AudioSubNode.SPATIAL */);
        }
        if ((hasStereoOptions = _HasStereoAudioOptions(options))) {
            await this.createAndAddSubNodeAsync("Stereo" /* AudioSubNode.STEREO */);
        }
        await this._createSubNodePromisesResolvedAsync();
        if (hasSpatialOptions) {
            _GetSpatialAudioSubNode(this)?.setOptions(options);
        }
        if (hasStereoOptions) {
            _GetStereoAudioSubNode(this)?.setOptions(options);
        }
    }
    /** @internal */
    get _inNode() {
        return this._inputNode;
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async
    _createSubNode(name) {
        try {
            const node = super._createSubNode(name);
            return node;
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
        }
        catch (e) { }
        switch (name) {
            case "Spatial" /* AudioSubNode.SPATIAL */:
                return _CreateSpatialAudioSubNodeAsync(this._owner.engine);
            case "Stereo" /* AudioSubNode.STEREO */:
                return _CreateStereoAudioSubNodeAsync(this._owner.engine);
            default:
                throw new Error(`Unknown subnode name: ${name}`);
        }
    }
    _onSubNodesChanged() {
        super._onSubNodesChanged();
        const spatialNode = _GetSpatialAudioSubNode(this);
        const stereoNode = _GetStereoAudioSubNode(this);
        const volumeNode = _GetVolumeAudioSubNode(this);
        if (spatialNode && spatialNode.getClassName() !== "_SpatialWebAudioSubNode") {
            throw new Error("Not a WebAudio subnode.");
        }
        if (stereoNode && stereoNode.getClassName() !== "_StereoWebAudioSubNode") {
            throw new Error("Not a WebAudio subnode.");
        }
        if (volumeNode && volumeNode.getClassName() !== "_VolumeWebAudioSubNode") {
            throw new Error("Not a WebAudio subnode.");
        }
        if (spatialNode) {
            spatialNode.disconnectAll();
            if (volumeNode) {
                spatialNode.connect(volumeNode);
            }
        }
        if (stereoNode) {
            stereoNode.disconnectAll();
            if (volumeNode) {
                stereoNode.connect(volumeNode);
            }
        }
        if (spatialNode && stereoNode) {
            this._rootNode = new GainNode(this._owner.engine._audioContext);
            this._rootNode.connect(spatialNode._outNode);
            this._rootNode.connect(stereoNode._outNode);
        }
        else {
            this._rootNode?.disconnect();
            this._rootNode = null;
        }
        let inSubNode = null;
        let inNode = null;
        if (this._rootNode) {
            inNode = this._rootNode;
        }
        else {
            if (spatialNode) {
                inSubNode = spatialNode;
            }
            else if (stereoNode) {
                inSubNode = stereoNode;
            }
            else if (volumeNode) {
                inSubNode = volumeNode;
            }
            inNode = inSubNode?.node ?? null;
        }
        if (this._inputNode !== inNode) {
            // Disconnect the wrapped upstream WebAudio nodes from the old wrapped WebAudio node.
            // The wrapper nodes are unaware of this change.
            if (this._inputNode && this._upstreamNodes) {
                const it = this._upstreamNodes.values();
                for (let next = it.next(); !next.done; next = it.next()) {
                    next.value._outNode?.disconnect(this._inputNode);
                }
            }
            this._inputNode = inNode;
            // Connect the wrapped upstream WebAudio nodes to the new wrapped WebAudio node.
            // The wrapper nodes are unaware of this change.
            if (inNode && this._upstreamNodes) {
                const it = this._upstreamNodes.values();
                for (let next = it.next(); !next.done; next = it.next()) {
                    next.value._outNode?.connect(inNode);
                }
            }
        }
    }
}

/** @internal */
class _SpatialAudio extends AbstractSpatialAudio {
    /** @internal */
    constructor(subGraph) {
        super();
        this._coneInnerAngle = _SpatialAudioDefaults.coneInnerAngle;
        this._coneOuterAngle = _SpatialAudioDefaults.coneOuterAngle;
        this._coneOuterVolume = _SpatialAudioDefaults.coneOuterVolume;
        this._distanceModel = _SpatialAudioDefaults.distanceModel;
        this._maxDistance = _SpatialAudioDefaults.maxDistance;
        this._minDistance = _SpatialAudioDefaults.minDistance;
        this._panningModel = _SpatialAudioDefaults.panningModel;
        this._rolloffFactor = _SpatialAudioDefaults.rolloffFactor;
        const subNode = _GetSpatialAudioSubNode(subGraph);
        if (subNode) {
            this._orientation = subNode.orientation.clone();
            this._position = subNode.position.clone();
            this._rotation = subNode.rotation.clone();
            this._rotationQuaternion = subNode.rotationQuaternion.clone();
        }
        else {
            this._orientation = _SpatialAudioDefaults.orientation.clone();
            this._position = _SpatialAudioDefaults.position.clone();
            this._rotation = _SpatialAudioDefaults.rotation.clone();
            this._rotationQuaternion = _SpatialAudioDefaults.rotationQuaternion.clone();
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            subGraph.createAndAddSubNodeAsync("Spatial" /* AudioSubNode.SPATIAL */);
        }
        this._subGraph = subGraph;
    }
    /** @internal */
    dispose() {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        this._subGraph.removeSubNodeAsync(_GetSpatialAudioSubNode(this._subGraph));
    }
    /** @internal */
    get coneInnerAngle() {
        return this._coneInnerAngle;
    }
    set coneInnerAngle(value) {
        this._coneInnerAngle = value;
        _SetSpatialAudioProperty(this._subGraph, "coneInnerAngle", value);
    }
    /** @internal */
    get coneOuterAngle() {
        return this._coneOuterAngle;
    }
    set coneOuterAngle(value) {
        this._coneOuterAngle = value;
        _SetSpatialAudioProperty(this._subGraph, "coneOuterAngle", value);
    }
    /** @internal */
    get coneOuterVolume() {
        return this._coneOuterVolume;
    }
    set coneOuterVolume(value) {
        this._coneOuterVolume = value;
        _SetSpatialAudioProperty(this._subGraph, "coneOuterVolume", value);
    }
    /** @internal */
    get distanceModel() {
        return this._distanceModel;
    }
    set distanceModel(value) {
        this._distanceModel = value;
        _SetSpatialAudioProperty(this._subGraph, "distanceModel", value);
    }
    /** @internal */
    get isAttached() {
        return this._subGraph.getSubNode("Spatial" /* AudioSubNode.SPATIAL */)?.isAttached ?? false;
    }
    /** @internal */
    get maxDistance() {
        return this._maxDistance;
    }
    set maxDistance(value) {
        if (value <= 0) {
            value = 0.000001;
        }
        this._maxDistance = value;
        _SetSpatialAudioProperty(this._subGraph, "maxDistance", value);
    }
    /** @internal */
    get minDistance() {
        return this._minDistance;
    }
    set minDistance(value) {
        this._minDistance = value;
        _SetSpatialAudioProperty(this._subGraph, "minDistance", value);
    }
    /** @internal */
    get orientation() {
        return this._orientation;
    }
    set orientation(value) {
        this._orientation = value;
        this._updateRotation();
    }
    /** @internal */
    get panningModel() {
        return this._panningModel;
    }
    set panningModel(value) {
        this._panningModel = value;
        _SetSpatialAudioProperty(this._subGraph, "panningModel", value);
    }
    /** @internal */
    get position() {
        return this._position;
    }
    set position(value) {
        this._position = value;
        this._updatePosition();
    }
    /** @internal */
    get rolloffFactor() {
        return this._rolloffFactor;
    }
    set rolloffFactor(value) {
        this._rolloffFactor = value;
        _SetSpatialAudioProperty(this._subGraph, "rolloffFactor", value);
    }
    /** @internal */
    get rotation() {
        return this._rotation;
    }
    set rotation(value) {
        this._rotation = value;
        this._updateRotation();
    }
    /** @internal */
    get rotationQuaternion() {
        return this._rotationQuaternion;
    }
    set rotationQuaternion(value) {
        this._rotationQuaternion = value;
        this._updateRotation();
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
        _GetSpatialAudioSubNode(this._subGraph)?.attach(sceneNode, useBoundingBox, attachmentType);
    }
    /**
     * Detaches from the scene node if attached.
     */
    detach() {
        _GetSpatialAudioSubNode(this._subGraph)?.detach();
    }
    /** @internal */
    update() {
        const subNode = _GetSpatialAudioSubNode(this._subGraph);
        if (!subNode) {
            return;
        }
        if (subNode.isAttached) {
            subNode.update();
        }
        else {
            this._updatePosition(subNode);
            this._updateRotation(subNode);
        }
    }
    _updatePosition(subNode = null) {
        if (!subNode) {
            subNode = _GetSpatialAudioSubNode(this._subGraph);
            if (!subNode) {
                return;
            }
        }
        const position = subNode.position;
        if (!position.equalsWithEpsilon(this._position)) {
            subNode.position.copyFrom(this._position);
            subNode._updatePosition();
        }
    }
    _updateRotation(subNode = null) {
        if (!subNode) {
            subNode = _GetSpatialAudioSubNode(this._subGraph);
            if (!subNode) {
                return;
            }
        }
        if (!subNode.rotationQuaternion.equalsWithEpsilon(this._rotationQuaternion)) {
            subNode.rotationQuaternion.copyFrom(this._rotationQuaternion);
            subNode._updateRotation();
            this._orientation.copyFrom(subNode.orientation);
            this._rotation.copyFrom(subNode.rotation);
        }
        else if (!subNode.rotation.equalsWithEpsilon(this._rotation)) {
            subNode.rotation.copyFrom(this._rotation);
            subNode._updateRotation();
            this._orientation.copyFrom(subNode.orientation);
            this._rotationQuaternion.copyFrom(subNode.rotationQuaternion);
        }
        else if (!subNode.orientation.equalsWithEpsilon(this._orientation)) {
            subNode.orientation.copyFrom(this._orientation);
            subNode._updateRotation();
            this._rotation.copyFrom(subNode.rotation);
            this._rotationQuaternion.copyFrom(subNode.rotationQuaternion);
        }
    }
}

/** @internal */
class _SpatialWebAudio extends _SpatialAudio {
    /** @internal */
    constructor(subGraph, autoUpdate, minUpdateTime) {
        super(subGraph);
        this._updaterComponent = new _SpatialWebAudioUpdaterComponent(this, autoUpdate, minUpdateTime);
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
    dispose() {
        super.dispose();
        this._updaterComponent.dispose();
        this._updaterComponent = null;
    }
}

export { _HasSpatialAudioOptions as _, _StereoAudio as a, _SpatialWebAudio as b, _WebAudioBusAndSoundSubGraph as c, _SpatialAudioDefaults as d };
//# sourceMappingURL=spatialWebAudio-DH890j7t.js.map
